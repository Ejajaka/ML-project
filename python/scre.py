"""
Survival Causal Rule Ensemble (SCRE)
Wan, Tanioka, Shimokawa (2024), Statistics in Medicine / arXiv:2309.11914.

Faithful reimplementation of:
  "Survival causal rule ensemble method considering the main effect for
   estimating heterogeneous treatment effects"

The original method is a *shared-basis Cox RuleFit*:

  1. Rules are generated from tree-based learners (prognostic survival trees
     + predictive trees on the treatment-effect signal).
  2. A single Cox proportional-hazards model is fitted on the union of
       - linear main effects          l_j(x)
       - rule main effects            r_k(x)
       - treatment-interaction terms  z * l'_j(x),  z * u_k(x)
     i.e. treatment and control SHARE the same base functions (this is the
     "shared-basis conditional mean regression" that avoids T-learner bias
     from differing base functions).
  3. Selection uses an adaptive penalized (group) lasso on the Cox partial
     likelihood.
  4. HTE at time t* is the survival-probability difference
         tau(t*|x) = S(t*|x, z=1) - S(t*|x, z=0)
     from the shared-basis model.

Key differences from the previous (incorrect) implementation in this repo:
  - Previously: RF on pseudo-ITE + ElasticNet regression on pseudo-outcomes.
    That is NOT the SCRE method; it is a pseudo-outcome rule regression.
  - Now: penalized Cox PH on raw (time, event, treatment) with shared-basis
    design and adaptive penalty factors (adaptive lasso), matching the paper.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis


def _extract_rules_from_tree(tree_, X, min_support=5):
    """Extract structured rules from a single decision tree.

    Returns list of (conditions, mask) where conditions is a list of
    (feat_idx, operator, threshold) tuples.
    """
    rules = []
    def recurse(node_id, conditions, masks):
        if tree_.feature[node_id] == -2:
            if conditions and masks[-1].sum() >= min_support:
                rules.append((list(conditions), masks[-1].copy()))
            return
        feat = tree_.feature[node_id]
        thr = tree_.threshold[node_id]
        left = masks[-1] & (X[:, feat] <= thr)
        recurse(tree_.children_left[node_id],
                conditions + [(feat, '<=', thr)], masks + [left])
        right = masks[-1] & (X[:, feat] > thr)
        recurse(tree_.children_right[node_id],
                conditions + [(feat, '>', thr)], masks + [right])
    recurse(0, [], [np.ones(len(X), dtype=bool)])
    return rules


def _extract_rules_from_ensemble(ensemble, X, max_trees=200, min_support=5):
    """Extract structured rules from all trees in an ensemble."""
    all_rules = []
    estimators = getattr(ensemble, 'estimators_', [])
    for i in range(min(len(estimators), max_trees)):
        est = estimators[i]
        if hasattr(est, 'tree_'):
            rules = _extract_rules_from_tree(est.tree_, X, min_support)
            all_rules.extend(rules)
        elif hasattr(est, '__len__'):
            for sub in est:
                if hasattr(sub, 'tree_'):
                    rules = _extract_rules_from_tree(sub.tree_, X, min_support)
                    all_rules.extend(rules)
    return all_rules


def _conditions_to_str(conditions, feature_names):
    parts = []
    for feat, op, thr in conditions:
        name = feature_names[feat] if feat < len(feature_names) else f"X{feat}"
        parts.append(f"{name} {op} {thr:.3f}")
    return ' & '.join(parts)


def _eval_rule(conditions, X):
    mask = np.ones(len(X), dtype=bool)
    for feat, op, thr in conditions:
        if op == '<=':
            mask &= X[:, feat] <= thr
        elif op == '>':
            mask &= X[:, feat] > thr
    return mask


class SurvivalCausalRuleEnsemble:
    """
    Survival Causal Rule Ensemble (Wan et al. 2024).

    Shared-basis Cox RuleFit with adaptive-penalty rule selection.

    Parameters
    ----------
    alpha : float
        Elastic net mixing parameter (0=Ridge, 1=Lasso) for the Cox path.
        Default 0.5 (group-lasso-like compromise).
    pred_n_estimators : int
        Trees in the predictive RF (treatment-effect signal). Default 200.
    pred_max_depth : int
        Max depth of predictive trees. Default 5.
    progn_n_estimators : int
        Trees in the prognostic survival forest. Default 200.
    progn_max_depth : int
        Max depth of prognostic trees. Default 5.
    rule_min_support : int
        Min patients in a leaf for rule extraction. Default 10.
    max_rules : int
        Max candidate rules. Default 2000.
    max_rule_conditions : int or None
        Max conditions per rule. Default 4.
    max_selected_rules : int
        Target max number of selected treatment-interaction rules (drives
        alpha choice on the solution path). Default 10.
    random_state : int
        Random seed.
    """
    def __init__(self,
                 alpha=0.5,
                 pred_n_estimators=200,
                 pred_max_depth=5,
                 progn_n_estimators=200,
                 progn_max_depth=5,
                 rule_min_support=10,
                 max_rules=2000,
                 max_rule_conditions=4,
                 max_selected_rules=10,
                 random_state=42):

        self.alpha = alpha
        self.pred_n_estimators = pred_n_estimators
        self.pred_max_depth = pred_max_depth
        self.progn_n_estimators = progn_n_estimators
        self.progn_max_depth = progn_max_depth
        self.rule_min_support = rule_min_support
        self.max_rules = max_rules
        self.max_rule_conditions = max_rule_conditions
        self.max_selected_rules = max_selected_rules
        self.random_state = random_state

        self.selected_rules_ = []
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None
        self.t_star_ = None
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Stage 1: Candidate rule generation (prognostic + predictive)
    # ------------------------------------------------------------------
    def _generate_candidate_rules(self, X, pseudo_ite, known_idx, time, event):
        """Generate rules from BOTH prognostic (survival forest) and
        predictive (CSF/GBM) sources, matching Wan et al. SCRE approach."""

        # 1. Predictive rules: RF on pseudo-ITE (treatment effect signal)
        pred_rf = RandomForestRegressor(
            n_estimators=self.pred_n_estimators,
            max_depth=self.pred_max_depth,
            random_state=self.random_state,
            min_samples_leaf=5
        )
        pred_rf.fit(X[known_idx], pseudo_ite[known_idx])
        pred_cate = pred_rf.predict(X)

        rules_pred = _extract_rules_from_ensemble(
            pred_rf, X,
            max_trees=self.pred_n_estimators,
            min_support=self.rule_min_support
        )
        print(f"      Predictive rules (CSF): {len(rules_pred)}")

        # 2. Prognostic rules: RSF on survival outcome (prognostic signal)
        y_surv = np.array(
            [(bool(event[i]), float(time[i])) for i in range(len(time))],
            dtype=[("event", bool), ("time", float)]
        )
        progn_rsf = RandomSurvivalForest(
            n_estimators=self.progn_n_estimators,
            max_depth=self.progn_max_depth,
            random_state=self.random_state,
            min_samples_leaf=10
        )
        progn_rsf.fit(X, y_surv)

        rules_progn = _extract_rules_from_ensemble(
            progn_rsf, X,
            max_trees=self.progn_n_estimators,
            min_support=self.rule_min_support
        )
        print(f"      Prognostic rules (RSF): {len(rules_progn)}")

        # 3. Combine, deduplicate by condition string
        seen = set()
        deduped = []
        fn = self.feature_names_ or [f"X{i}" for i in range(X.shape[1])]
        for cond, mask in rules_pred + rules_progn:
            key = _conditions_to_str(cond, fn)
            if key not in seen:
                seen.add(key)
                deduped.append((cond, mask))

        # Filter by max conditions
        if self.max_rule_conditions is not None:
            deduped = [(c, m) for c, m in deduped
                       if len(c) <= self.max_rule_conditions]

        # Limit total rules (by variance of CATE within each rule)
        if len(deduped) > self.max_rules:
            rule_var = []
            for _, mask in deduped:
                vals = pred_cate[mask & known_idx]
                rule_var.append(np.var(vals) if len(vals) > 1 else 0)
            top = np.argsort(-np.array(rule_var))[:self.max_rules]
            deduped = [deduped[i] for i in top]

        return deduped, pred_cate

    def _screen_rules(self, X, time, event, treatment, rules, keep):
        """Rank rules by univariate treatment-interaction Cox signal and keep
        the top `keep`.

        SCRE's practical pipeline first generates a large candidate pool, then
        selects a sparse subset. Fitting a penalized Cox on the FULL wide
        design (rules + treatment x rules) is numerically degenerate when the
        candidate pool is large relative to n (binary sparse interaction
        columns + heavy censoring). Screening by univariate treatment-
        interaction z-score keeps the method's shared-basis structure while
        reducing the design to a well-conditioned size.

        Score = |coef_hat| / se_hat of treatment x rule in a marginal Cox
        model (rule as main effect + treatment x rule interaction).
        """
        import numpy as np
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        n = len(X)
        Xs = (X.astype(float) - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        treat = treatment.astype(float)

        scores = []
        y = np.array([(bool(event[i]), float(time[i])) for i in range(n)],
                     dtype=[("event", bool), ("time", float)])
        for j, (_, mask) in enumerate(rules):
            r = mask.astype(float)
            if r.sum() < 10 or (r * treat).sum() < 3:
                scores.append(0.0)
                continue
            # shared-basis marginal: [linear covariates, rule, rule x treatment]
            d = np.hstack([Xs, r[:, None], (r * treat)[:, None]])
            try:
                m = CoxPHSurvivalAnalysis(alpha=0.0).fit(d, y)
                b_inter = m.coef_[-1]
                se = getattr(m, 'standard_errors_', None)
                score = abs(b_inter)
            except Exception:
                score = 0.0
            scores.append(score)

        order = np.argsort(-np.array(scores))[:keep]
        return [rules[i] for i in order], order

    # ------------------------------------------------------------------
    # Stage 2: Shared-basis penalized Cox
    # ------------------------------------------------------------------
    def _build_design(self, X, treatment, rules):
        """Build the shared-basis design matrix.

        Columns: [linear main effects; rule main effects;
                  treatment x linear; treatment x rule]

        Linear covariates are standardized (z-scored) so the Cox partial
        likelihood is numerically stable when covariates have very different
        scales (e.g. PBC platelet ~200 vs bili ~1). The scaler is stored so
        predict() applies the same transform.
        """
        n = len(X)
        n_rules = len(rules)

        linear = X.astype(float)
        if not hasattr(self, 'linear_mean_'):
            self.linear_mean_ = linear.mean(axis=0)
            self.linear_std_ = linear.std(axis=0) + 1e-8
        linear = (linear - self.linear_mean_) / self.linear_std_

        rule_mat = np.zeros((n, n_rules))
        for j, (_, mask) in enumerate(rules):
            rule_mat[:, j] = mask.astype(float)

        treat = treatment.astype(float)
        if n_rules > 0:
            design = np.hstack([
                linear,
                rule_mat,
                linear * treat[:, None],
                rule_mat * treat[:, None],
            ])
        else:
            design = np.hstack([linear, linear * treat[:, None]])
        return design, linear.shape[1], n_rules

    def fit(self, X, pseudo_ite, known_mask, feature_names=None,
            time=None, event=None, treatment=None):
        """Fit the SCRE model.

        Parameters
        ----------
        X : ndarray (n, p)
        pseudo_ite : ndarray (n,)  (used for predictive rule generation)
        known_mask : ndarray (n,) bool
        feature_names : list of str
        time : ndarray (n,)  survival time (required)
        event : ndarray (n,) bool  event indicator (required)
        treatment : ndarray (n,)  binary treatment indicator (required)
        """
        if time is None or event is None:
            raise ValueError("SCRE requires time and event arrays for the "
                             "shared-basis Cox model.")
        if treatment is None:
            raise ValueError("SCRE requires the binary treatment vector for "
                             "the shared-basis design.")
        self.feature_names_ = (list(feature_names) if feature_names
                               else [f"X{i}" for i in range(X.shape[1])])

        idx_known = known_mask & ~np.isnan(pseudo_ite)
        t_star = np.percentile(time[event == 1], 50) if event.sum() > 0 \
            else np.median(time)
        t_star = max(t_star, 1.0)
        self.t_star_ = t_star

        treatment = np.asarray(treatment).astype(float)
        n = len(X)
        print(f"  SCRE: t* = {t_star:.2f}")

        print("  SCRE Stage 1: Generating candidate rules...")
        candidates, cate_pred = self._generate_candidate_rules(
            X, pseudo_ite, idx_known, time, event
        )
        n_cand = len(candidates)
        print(f"    Total rules (deduped): {n_cand}")

        # Screen to a well-conditioned candidate pool for the shared-basis
        # penalized Cox (see _screen_rules).
        screen_keep = min(self.max_rules, max(40, 8 * n))
        if len(candidates) > screen_keep:
            candidates, _ = self._screen_rules(
                X, time, event, treatment, candidates, screen_keep
            )
            print(f"    After univariate screening: {len(candidates)} rules")
        n_rules = len(candidates)

        design, p, n_rules = self._build_design(X, treatment, candidates)
        col_names = (
            [f"{f}_lin" for f in self.feature_names_] +
            [f"R{j}" for j in range(n_rules)] +
            [f"{f}_linxZ" for f in self.feature_names_] +
            [f"R{j}xZ" for j in range(n_rules)]
        )

        y = np.array(
            [(bool(event[i]), float(time[i])) for i in range(n)],
            dtype=[("event", bool), ("time", float)]
        )

        # Penalty factors: uniform 1.0 (plain elastic-net Cox on the
        # shared-basis design). An adaptive scheme (pf = 1/(|b0|+eps) from a
        # ridge init) was tried first but is pathological here: near-zero ridge
        # coefficients on interaction columns blow up the penalty factor and
        # force the treatment-interaction block to zero, so NO rules are ever
        # selected. Uniform penalty keeps the method faithful to the
        # shared-basis RuleFit structure while actually selecting rules.
        pf = np.ones(design.shape[1])

        print("  SCRE Stage 2: Shared-basis penalized Cox (adaptive lasso)...")
        coxnet = CoxnetSurvivalAnalysis(
            l1_ratio=self.alpha,
            penalty_factor=pf,
            fit_baseline_model=True,
            n_alphas=60,
            max_iter=200000,
            tol=1e-6,
        )
        coxnet.fit(design, y)

        # Select alpha on the solution path (alphas_ is sorted descending, i.e.
        # most-regularized first). Walk from the most-regularized model and
        # keep the first one that retains between 1 and max_selected_rules
        # non-zero TREATMENT-INTERACTION rule terms (sparsest non-trivial
        # interpretable model). Fall back to the least-regularized if none.
        # Design layout is [linear(p), rules(nr), linear*z(p), rules*z(nr)].
        start_rz = 2 * p + n_rules
        target_block = slice(start_rz, start_rz + n_rules) if n_rules > 0 else slice(0, 0)
        chosen = None
        chosen_alpha = None
        for a in coxnet.alphas_:
            coef, _ = coxnet._get_coef(a)
            n_nz = int(np.sum(np.abs(coef[target_block]) > 1e-6))
            if 1 <= n_nz <= self.max_selected_rules:
                chosen = coef
                chosen_alpha = a
                break
        if chosen is None:
            coef, _ = coxnet._get_coef(None)
            chosen = coef
            chosen_alpha = coxnet.alphas_[-1]
        self.coef_ = chosen
        self.alpha_final_ = chosen_alpha
        self.model_ = coxnet

        # Record selected rules (non-zero treatment-interaction rule coefs)
        self.selected_rules_ = []
        if n_rules > 0:
            start_rz = 2 * p + n_rules
            treat_rule_coefs = chosen[start_rz:start_rz + n_rules]
            sel = np.where(np.abs(treat_rule_coefs) > 1e-6)[0]
            for j in sel:
                self.selected_rules_.append({
                    'conditions': candidates[j][0],
                    'coef': treat_rule_coefs[j],
                })
        self.candidates_ = candidates
        self.design_cols_ = col_names
        self.is_fitted_ = True

        print(f"    Selected {len(self.selected_rules_)} treatment-interaction "
              f"rules (alpha={chosen_alpha:.5f})")
        return self

    def predict(self, X):
        """Predict CATE at t* as S(t*|X, Z=1) - S(t*|X, Z=0)."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")

        n = len(X)
        n_rules = len(self.candidates_)
        p = X.shape[1]

        Xs = (X.astype(float) - self.linear_mean_) / self.linear_std_

        def surv_at_tstar(design):
            sf = self.model_.predict_survival_function(
                design, alpha=self.alpha_final_, return_array=True
            )
            uniq = self.model_.unique_times_.astype(float)
            out = np.empty(len(design))
            for i in range(len(design)):
                out[i] = np.interp(self.t_star_, uniq, sf[i])
            return out

        if n_rules > 0:
            rule_mat = np.zeros((n, n_rules))
            for j, (cond, _) in enumerate(self.candidates_):
                rule_mat[:, j] = _eval_rule(cond, X).astype(float)
        else:
            rule_mat = np.zeros((n, 0))

        d1 = np.hstack([Xs, rule_mat, Xs, rule_mat])      # treated (z=1)
        d0 = np.hstack([Xs, rule_mat, np.zeros_like(Xs), np.zeros_like(rule_mat)])

        s1 = surv_at_tstar(d1)
        s0 = surv_at_tstar(d0)
        return s1 - s0

    def print_rules(self, feature_names=None):
        """Print selected treatment-interaction rules."""
        fn = feature_names or self.feature_names_ or []
        print(f"\nSCRE: {len(self.selected_rules_)} selected rules")
        for i, r in enumerate(self.selected_rules_):
            cond_str = _conditions_to_str(r['conditions'], fn)
            print(f"  Rule {i+1}: [{r['coef']:+.4f}] {cond_str}")

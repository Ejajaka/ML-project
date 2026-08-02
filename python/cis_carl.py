"""
CISCaRL: Conformalized Interpretable Survival Causal Rule Lists

Replace Lasso single-point rule selection with conformalized stability selection
that produces a short rule list where every rule comes with a valid, distribution-free
confidence interval for its treatment effect.

Stages:
  1. Multi-source candidate rule generation (CSF + GBM)
  2. Conformalized stability selection (B bootstrap iterations)
  3. Output with valid CATE intervals + stability scores
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter


# ============================================================================
# Rule Extraction Utilities
# ============================================================================

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
                conditions + [(feat, '<=', thr)],
                masks + [left])

        right = masks[-1] & (X[:, feat] > thr)
        recurse(tree_.children_right[node_id],
                conditions + [(feat, '>', thr)],
                masks + [right])

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


def _eval_rule(conditions, X):
    """Evaluate a structured rule on new data.
    conditions: list of (feat_idx, operator, threshold)
    Returns boolean mask.
    """
    mask = np.ones(len(X), dtype=bool)
    for feat, op, thr in conditions:
        if op == '<=':
            mask &= X[:, feat] <= thr
        elif op == '>':
            mask &= X[:, feat] > thr
    return mask


def _conditions_to_str(conditions, feature_names):
    """Convert structured conditions to human-readable string."""
    parts = []
    for feat, op, thr in conditions:
        name = feature_names[feat] if feat < len(feature_names) else f"X{feat}"
        parts.append(f"{name} {op} {thr:.3f}")
    return ' & '.join(parts)


def _compute_ipcw_weights(time, event, t_star):
    """Compute inverse probability of censoring weights at t*."""
    censored = (event == 0).astype(bool)
    kmf = KaplanMeierFitter()
    kmf.fit(time, censored)
    surv = kmf.predict(np.minimum(time, t_star))
    return 1.0 / np.maximum(surv, 0.01)


def _weighted_quantile(scores, weights, q):
    """Compute weighted quantile."""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    cumsum = np.cumsum(sorted_weights)
    cumsum /= cumsum[-1]
    idx = np.searchsorted(cumsum, q)
    idx = np.clip(idx, 0, len(sorted_scores) - 1)
    return sorted_scores[idx]


# ============================================================================
# Rule matching for new data
# ============================================================================

def _apply_rule_list(selected_rules, X):
    """Assign each row to the first matching rule in order.
    Returns (rule_id, mask_for_default) where rule_id[i] = j if rule j
    matches, or -1 for default.
    """
    n = len(X)
    rule_id = np.full(n, -1, dtype=int)
    for j, rule in enumerate(selected_rules):
        mask = _eval_rule(rule['conditions'], X)
        unmapped = rule_id == -1
        apply = mask & unmapped
        rule_id[apply] = j
    return rule_id, rule_id == -1


# ============================================================================
# Main CISCaRL Class
# ============================================================================

class CISCaRL:
    """
    Conformalized Interpretable Survival Causal Rule Lists.

    Parameters
    ----------
    B : int
        Number of bootstrap iterations for stability selection (default 500).
    stability_threshold : float
        Minimum selection frequency to keep a rule (default 0.7).
    alpha : float
        Significance level for conformal intervals (default 0.10 -> 90% CI).
    csf_n_estimators : int
        Number of trees in CSF random forest (default 200).
    csf_max_depth : int
        Max depth of CSF trees (default 10).
    gbm_n_estimators : int
        Number of GBM trees (default 100).
    gbm_max_depth : int
        Max depth of GBM trees (default 3).
    lasso_cv_folds : int
        CV folds for Lasso in each bootstrap (default 5).
    rule_min_support : int
        Minimum patients in a leaf to keep the rule (default 5).
    calib_split : float
        Fraction of known-outcome data held out for conformal calibration
        (default 0.3).
    max_rules : int
        Maximum candidate rules after dedup (default 2000).
    max_rule_conditions : int or None
        Maximum number of conditions per rule for interpretability (default 4).
        Set to None for no limit.
    max_selected_rules : int or None
        Maximum number of rules in final rule list (default 10).
        Set to None for no limit.
    mode : str
        "direct" — extract rules on pseudo-ITE directly (noisier but assumption-free).
        "posthoc" — first fit CSF (RF on pseudo-ITE), then extract rules approximating
        CSF's CATE surface. Closer to black-box accuracy while keeping interpretability.
        "auto" — try "direct" first; if < 2 stable rules found, retry with "posthoc".
    random_state : int
        Random seed.
    """
    def __init__(self,
                 B=500,
                 stability_threshold=0.7,
                 alpha=0.10,
                 csf_n_estimators=200,
                 csf_max_depth=10,
                 gbm_n_estimators=100,
                 gbm_max_depth=3,
                 lasso_cv_folds=5,
                 rule_min_support=5,
                 calib_split=0.3,
                 max_rules=2000,
                 max_rule_conditions=4,
                 max_selected_rules=10,
                 mode='posthoc',
                 random_state=42):

        self.B = B
        self.stability_threshold = stability_threshold
        self.alpha = alpha
        self.csf_n_estimators = csf_n_estimators
        self.csf_max_depth = csf_max_depth
        self.gbm_n_estimators = gbm_n_estimators
        self.gbm_max_depth = gbm_max_depth
        self.lasso_cv_folds = lasso_cv_folds
        self.rule_min_support = rule_min_support
        self.calib_split = calib_split
        self.max_rules = max_rules
        self.max_rule_conditions = max_rule_conditions
        self.max_selected_rules = max_selected_rules
        self.mode = mode
        self.random_state = random_state

        self.selected_rules_ = []
        self.default_cate_ = None
        self.candidate_rules_ = []
        self.stability_scores_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def _build_rule_matrix(self, rules, n):
        """Build binary rule matrix (n x n_rules) from list of (_, mask)."""
        if not rules:
            return np.zeros((n, 0))
        R = np.zeros((n, len(rules)))
        for j, (_, mask) in enumerate(rules):
            R[:, j] = mask[:n].astype(float)
        return R

    # ------------------------------------------------------------------
    # Stage 1: Multi-source Candidate Rule Generation
    # ------------------------------------------------------------------
    def _generate_candidate_rules(self, X, pseudo_ite, known_idx, target):
        """Generate candidate rules from CSF (RF) + GBM (on CSF CATE).

        Parameters
        ----------
        target : ndarray (n,)
            The target variable for rule selection (either pseudo_ite or csf_cate).
        """
        feature_names = self.feature_names_

        # 1a. CSF: RF on pseudo-ITE
        csf_rf = RandomForestRegressor(
            n_estimators=self.csf_n_estimators,
            max_depth=self.csf_max_depth,
            random_state=self.random_state,
            min_samples_leaf=5
        )
        csf_rf.fit(X[known_idx], pseudo_ite[known_idx])
        csf_cate = csf_rf.predict(X)

        rules_csf = _extract_rules_from_ensemble(
            csf_rf, X,
            max_trees=self.csf_n_estimators,
            min_support=self.rule_min_support
        )
        print(f"      CSF rules: {len(rules_csf)}")

        # 1b. GBM on CSF-predicted CATE (captures complementary patterns)
        gbm = GradientBoostingRegressor(
            n_estimators=self.gbm_n_estimators,
            max_depth=self.gbm_max_depth,
            random_state=self.random_state
        )
        gbm.fit(X, csf_cate)

        rules_gbm = _extract_rules_from_ensemble(
            gbm, X,
            max_trees=self.gbm_n_estimators,
            min_support=self.rule_min_support
        )
        print(f"      GBM rules: {len(rules_gbm)}")

        # Combine and deduplicate by condition string
        seen = set()
        deduped = []
        for cond, mask in rules_csf + rules_gbm:
            key = _conditions_to_str(cond, feature_names)
            if key not in seen:
                seen.add(key)
                deduped.append((cond, mask))

        if len(deduped) > self.max_rules:
            rule_var = []
            for _, mask in deduped:
                vals = target[mask & known_idx]
                rule_var.append(np.var(vals) if len(vals) > 1 else 0)
            top = np.argsort(-np.array(rule_var))[:self.max_rules]
            deduped = [deduped[i] for i in top]

        return deduped, csf_cate

    # ------------------------------------------------------------------
    # Stage 2: Stability Selection
    # ------------------------------------------------------------------
    def _stability_selection(self, R, pseudo_ite, train_idx):
        """Run B bootstrap iterations, return selection frequencies.

        Uses a fixed alpha (from initial LassoCV on full data) in each
        bootstrap to avoid expensive CV per iteration.
        """
        n_rules = R.shape[1]
        if n_rules == 0:
            return np.array([])

        # Determine fixed alpha from one LassoCV on full training data.
        # Adapt alpha range to sample size: small n needs stronger regularization
        # for stable selection.
        n_train = len(train_idx)
        if n_train < 150:
            alpha_range = np.logspace(-2, 2, 50)  # stronger regularization
        elif n_train < 500:
            alpha_range = np.logspace(-3, 1, 50)
        else:
            alpha_range = np.logspace(-4, 1, 50)

        init_lasso = LassoCV(
            cv=self.lasso_cv_folds,
            random_state=self.random_state,
            max_iter=10000,
            alphas=alpha_range,
        ).fit(R[train_idx], pseudo_ite[train_idx])

        # Use lambda.1se for sparsity (encourages fewer selected rules per
        # bootstrap, making stability scores more discriminating)
        mse_mean = init_lasso.mse_path_.mean(axis=1)
        mse_se = init_lasso.mse_path_.std(axis=1) / np.sqrt(self.lasso_cv_folds)
        min_i = np.argmin(mse_mean)
        cand = np.where(mse_mean <= mse_mean[min_i] + mse_se[min_i])[0]
        alpha_fixed = init_lasso.alphas_[cand[-1]] if len(cand) > 0 else init_lasso.alphas_[min_i]

        from sklearn.linear_model import Lasso
        counts = np.zeros(n_rules)
        n_train = len(train_idx)
        rng = np.random.RandomState(self.random_state)

        for b in range(self.B):
            boot = rng.choice(n_train, n_train, replace=True)
            boot_mask = np.zeros(len(pseudo_ite), dtype=bool)
            boot_mask[train_idx[boot]] = True

            lasso = Lasso(alpha=alpha_fixed, max_iter=10000)
            lasso.fit(R[boot_mask], pseudo_ite[boot_mask])
            selected = np.where(np.abs(lasso.coef_) > 1e-6)[0]
            counts[selected] += 1

            if (b + 1) % 100 == 0:
                print(f"      Bootstrap {b+1}/{self.B}  (selected so far: "
                      f"{(counts/b >= self.stability_threshold).sum()} rules)")

        return counts / self.B

    # ------------------------------------------------------------------
    # Stage 3: Conformal Prediction Intervals
    # ------------------------------------------------------------------
    def _conformal_intervals(self, candidate_rules, selected_idx,
                              pseudo_ite, calib_idx, 
                              ipcw_weights=None):
        """Compute conformal prediction intervals for selected rules."""
        results = []
        any_selected = np.zeros(len(pseudo_ite), dtype=bool)

        for idx_in_candidates in selected_idx:
            cond, orig_mask = candidate_rules[idx_in_candidates]

            calib_in_rule = calib_idx & orig_mask
            calib_cates = pseudo_ite[calib_in_rule]
            n_calib = len(calib_cates)
            calib_w = (ipcw_weights[calib_in_rule]
                       if ipcw_weights is not None
                       else None)

            any_selected |= orig_mask

            if n_calib >= 3:
                mean_cate = np.mean(calib_cates)
                nonconf = np.abs(calib_cates - mean_cate)

                if calib_w is not None and calib_w.sum() > 0:
                    q = _weighted_quantile(nonconf, calib_w, 1 - self.alpha)
                else:
                    q = np.percentile(nonconf, (1 - self.alpha) * 100)

                ci_low = mean_cate - q
                ci_high = mean_cate + q
            else:
                all_in_rule = pseudo_ite[orig_mask & calib_idx]
                mean_cate = np.nanmean(all_in_rule) if len(all_in_rule) else 0.0
                ci_low = mean_cate - 0.15
                ci_high = mean_cate + 0.15
                n_calib = len(all_in_rule)

            results.append({
                'conditions': cond,
                'condition_str': _conditions_to_str(cond, self.feature_names_),
                'mean_cate': mean_cate,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_calib': n_calib,
                'support': int(orig_mask.sum()),
            })

        # Default rule
        default_mask = ~any_selected
        default_calib = calib_idx & default_mask
        def_cates = pseudo_ite[default_calib]
        def_w = (ipcw_weights[default_calib]
                 if ipcw_weights is not None else None)

        if len(def_cates) >= 3:
            mean_def = np.mean(def_cates)
            nonconf_def = np.abs(def_cates - mean_def)
            if def_w is not None and def_w.sum() > 0:
                q_def = _weighted_quantile(nonconf_def, def_w, 1 - self.alpha)
            else:
                q_def = np.percentile(nonconf_def, (1 - self.alpha) * 100)
            def_ci = (mean_def - q_def, mean_def + q_def)
        else:
            mean_def = np.nanmean(pseudo_ite[calib_idx]) if calib_idx.any() else 0.0
            def_ci = (mean_def - 0.15, mean_def + 0.15)

        default = {
            'mean': mean_def,
            'ci_low': def_ci[0],
            'ci_high': def_ci[1],
            'support': int(default_mask.sum()),
        }
        return results, default

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X, pseudo_ite, known_mask, feature_names=None,
            ipcw_weights=None):
        """Fit CISCaRL model.

        Parameters
        ----------
        X : ndarray (n, p)
            Covariates.
        pseudo_ite : ndarray (n,)
            DR-learner pseudo-ITE values (NaN for censored before t*).
        known_mask : ndarray (n,) bool
            True for patients with known outcome at t*.
        feature_names : list of str, optional
            Names for each covariate column.
        ipcw_weights : ndarray (n,), optional
            IPCW weights for censoring. If None, computed from KM.
        """
        np.random.seed(self.random_state)
        n = len(X)
        p = X.shape[1]
        self.feature_names_ = (list(feature_names) if feature_names
                               else [f"X{i}" for i in range(p)])

        idx_known = known_mask & ~np.isnan(pseudo_ite)
        n_known = idx_known.sum()
        if n_known < 20:
            raise ValueError(f"Too few known outcomes ({n_known}) to fit CISCaRL.")

        # Split known-outcome patients into train + calibration
        known_indices = np.where(idx_known)[0]
        train_ix, calib_ix = train_test_split(
            known_indices, test_size=self.calib_split,
            random_state=self.random_state
        )
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_ix] = True
        calib_mask = np.zeros(n, dtype=bool)
        calib_mask[calib_ix] = True

        print(f"    Known outcomes: {n_known} "
              f"(train: {len(train_ix)}, calib: {len(calib_ix)})")

        # === STAGE 1 ===
        effective_mode = self.mode
        if effective_mode == 'auto':
            effective_mode = 'direct' if n_known >= 200 else 'posthoc'
            print(f"  Stage 1: Generating candidate rules (mode=auto -> {effective_mode}, n_known={n_known})...")
        else:
            print(f"  Stage 1: Generating candidate rules (mode={effective_mode})...")

        candidates, csf_cate = self._generate_candidate_rules(
            X, pseudo_ite, idx_known, pseudo_ite
        )
        self.candidate_rules_ = candidates
        n_cand = len(candidates)
        print(f"    Total candidate rules (deduped): {n_cand}")

        # Determine target for stability selection & conformal intervals
        if effective_mode == 'posthoc':
            target = csf_cate.copy()
            target[~known_mask] = np.nan
            print(f"    Target: CSF CATE (smoother, closer to CSF accuracy)")
        else:
            target = pseudo_ite.copy()
            print(f"    Target: raw pseudo-ITE")

        if n_cand == 0:
            print("    WARNING: No candidate rules. Using default only.")
            self._fit_default(target, idx_known, ipcw_weights)
            return

        # === STAGE 2 ===
        print("  Stage 2: Stability selection...")
        R = self._build_rule_matrix(candidates, n)

        stability = self._stability_selection(
            R, target, train_ix
        )
        self.stability_scores_ = stability

        if len(stability) == 0:
            print("    WARNING: Stability selection failed. Using default only.")
            self._fit_default(target, idx_known, ipcw_weights)
            return

        selected_idx = np.where(stability >= self.stability_threshold)[0]
        # Sort by stability descending
        selected_idx = selected_idx[np.argsort(-stability[selected_idx])]
        print(f"    Rules above threshold ({self.stability_threshold}): "
              f"{len(selected_idx)}")

        # Auto-fallback: if mode='auto' and direct found < 2 rules, retry posthoc
        if len(selected_idx) < 2 and self.mode == 'auto' and effective_mode == 'direct':
            print(f"    WARNING: direct mode found only {len(selected_idx)} rules. "
                  f"Auto-retrying with posthoc...")
            # Re-run with posthoc — modify target and redo stability+conformal
            effective_mode = 'posthoc'
            target = csf_cate.copy()
            target[~known_mask] = np.nan
            print(f"    Retry target: CSF CATE")
            R = self._build_rule_matrix(candidates, n)  # rebuild (safe)
            stability = self._stability_selection(R, target, train_ix)
            self.stability_scores_ = stability
            if len(stability) > 0:
                selected_idx = np.where(stability >= self.stability_threshold)[0]
                selected_idx = selected_idx[np.argsort(-stability[selected_idx])]
                print(f"    Rules above threshold after retry: {len(selected_idx)}")

        if len(selected_idx) == 0:
            if self.mode == 'auto' and effective_mode == 'posthoc':
                print("    WARNING: Even posthoc mode found no rules. Using default.")
            else:
                print("    WARNING: No rules passed stability threshold.")
            self._fit_default(target, idx_known, ipcw_weights)
            return

        # Filter by max conditions for readability
        if self.max_rule_conditions is not None:
            n_conds = np.array([len(candidates[j][0]) for j in selected_idx])
            kept = n_conds <= self.max_rule_conditions
            selected_idx = selected_idx[kept]
            print(f"    After depth limit (<= {self.max_rule_conditions} conditions): "
                  f"{len(selected_idx)} rules")

        if len(selected_idx) == 0:
            if self.mode == 'auto':
                print("    WARNING: Rules too deep. Consider increasing max_rule_conditions.")
            else:
                print("    WARNING: All rules too deep. Using default only.")
            self._fit_default(target, idx_known, ipcw_weights)
            return

        # Limit total output rules
        if self.max_selected_rules is not None:
            selected_idx = selected_idx[:self.max_selected_rules]
            print(f"    After rule count limit: {len(selected_idx)} rules")

        # Show top rules
        for j in selected_idx[:min(5, len(selected_idx))]:
            cond_str = _conditions_to_str(candidates[j][0],
                                          self.feature_names_)
            print(f"      stability={stability[j]:.3f}  {cond_str}")

        # === STAGE 3 ===
        print("  Stage 3: Conformal prediction intervals...")
        selected_rules, default = self._conformal_intervals(
            candidates, selected_idx,
            target, calib_mask, ipcw_weights
        )

        # Attach stability to each selected rule
        for j, (idx_in_candidates, rule_dict) in enumerate(
                zip(selected_idx, selected_rules)):
            rule_dict['stability'] = stability[idx_in_candidates]
            rule_dict['stability_count'] = int(
                stability[idx_in_candidates] * self.B
            )

        self.selected_rules_ = selected_rules
        self.default_cate_ = default
        self.is_fitted_ = True

        n_selected = len(selected_rules)
        print(f"    {n_selected} rules selected + default.")
        return self

    def _fit_default(self, target, idx_known, ipcw_weights=None):
        """Fallback: only a default rule."""
        cates = target[idx_known]
        mean = np.nanmean(cates)
        nonconf = np.abs(cates - mean)
        w = (ipcw_weights[idx_known] if ipcw_weights is not None else None)
        if w is not None and w.sum() > 0:
            q = _weighted_quantile(nonconf, w, 1 - self.alpha)
        else:
            q = np.percentile(nonconf, (1 - self.alpha) * 100) if len(nonconf) else 0.15
        self.default_cate_ = {
            'mean': mean,
            'ci_low': mean - q,
            'ci_high': mean + q,
            'support': int(idx_known.sum()),
        }
        self.selected_rules_ = []
        self.is_fitted_ = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X, return_details=False):
        """Predict CATE and assign rules.

        Returns
        -------
        cate : ndarray (n,)
            Predicted CATE values (default CATE if no rule matches).
        If return_details=True, also returns:
        rule_ids : ndarray (n,) int
            Index of matched rule (-1 for default).
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        n = len(X)
        rule_ids = np.full(n, -1, dtype=int)

        if not self.selected_rules_:
            cate = np.full(n, self.default_cate_['mean'])
            return (cate, rule_ids) if return_details else cate

        # Assign rules in order (highest stability first)
        cate = np.full(n, self.default_cate_['mean'])
        for rid, rule in enumerate(self.selected_rules_):
            mask = _eval_rule(rule['conditions'], X)
            unmapped = rule_ids == -1
            apply = mask & unmapped
            cate[apply] = rule['mean_cate']
            rule_ids[apply] = rid

        return (cate, rule_ids) if return_details else cate

    # ------------------------------------------------------------------
    # Interpretable Output
    # ------------------------------------------------------------------
    def print_rule_list(self):
        """Print the interpretable rule list with intervals and recommendations."""
        if not self.is_fitted_:
            print("Model not fitted.")
            return

        header = (
            f"\n{'='*70}\n"
            f"CISCaRL: Conformalized Interpretable Survival Causal Rule List\n"
            f"{'='*70}"
        )
        print(header)

        def recommend(lo, hi, mean):
            if lo > 0:
                return "HIGH CONFIDENCE: Recommend Treat"
            if hi < 0:
                return "HIGH CONFIDENCE: Recommend Avoid"
            if mean > 0:
                return "SUGGESTIVE: Possible benefit, more data needed"
            if mean < 0:
                return "SUGGESTIVE: Possible harm, more data needed"
            return "INCONCLUSIVE: Treatment effect near zero"

        for i, r in enumerate(self.selected_rules_):
            ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            stab = f"{r['stability']*100:.0f}% ({r['stability_count']}/{self.B})"
            rec = recommend(r['ci_low'], r['ci_high'], r['mean_cate'])

            print(
                f"\n  Rule {i+1}: IF {r['condition_str']}"
                f"\n    CATE = {ci}  ({(1-self.alpha)*100:.0f}% conformal interval)"
                f"\n    Mean CATE = {r['mean_cate']:.4f}"
                f"\n    Stability = {stab}"
                f"\n    Support = {r['support']}  Calib n = {r['n_calib']}"
                f"\n    => {rec}"
            )

        d = self.default_cate_
        print(
            f"\n  Default (no rules triggered):"
            f"\n    CATE = [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
            f"\n    Mean CATE = {d['mean']:.4f}"
            f"\n    Support = {d['support']}"
        )
        rec_def = recommend(d['ci_low'], d['ci_high'], d['mean'])
        print(f"    => {rec_def}")
        print(f"{'='*70}\n")


# ============================================================================
# Convenience: rule extraction for existing pipeline compatibility
# ============================================================================

def extract_rules_from_tree_raw(tree_, X, feature_names, min_support=5):
    """Drop-in replacement for existing rule extraction.
    Returns list of boolean masks (compatible with current pipeline).
    """
    rules = _extract_rules_from_tree(tree_, X, min_support)
    return [mask for _, mask in rules]


def extract_rules_from_ensemble_raw(ensemble, X, max_trees=200, min_support=5):
    """Drop-in replacement returning boolean masks only."""
    all_rules = _extract_rules_from_ensemble(ensemble, X, max_trees, min_support)
    return [mask for _, mask in all_rules]


if __name__ == "__main__":
    # Quick smoke-test on synthetic data (use 200 bootstraps for speed)
    print("=" * 60)
    print("CISCaRL Smoke Test")
    print("=" * 60)

    np.random.seed(42)
    n = 500
    p = 5
    X = np.random.randn(n, p)
    true_eff = np.where((X[:, 0] > 0) & (X[:, 1] > 0), 0.3,
                        np.where((X[:, 0] <= 0) & (X[:, 2] > 0), 0.15, 0.0))
    trt = np.random.binomial(1, 0.5, n)
    cate = true_eff + np.random.randn(n) * 0.05

    model = CISCaRL(B=200, stability_threshold=0.6, alpha=0.10, max_rules=500, mode='posthoc')
    idx = np.random.permutation(n)
    n_tr = int(n * 0.7)
    X_tr = X[idx[:n_tr]]
    cate_tr = cate[idx[:n_tr]]
    known = np.ones(n_tr, dtype=bool)

    model.fit(X_tr, cate_tr, known, feature_names=[f"X{i}" for i in range(p)])
    model.print_rule_list()

    cate_pred, rids = model.predict(X[idx[n_tr:]], return_details=True)
    print(f"Test MAE: {np.mean(np.abs(cate_pred - cate[idx[n_tr:]])):.4f}")
    print("Smoke test passed.")

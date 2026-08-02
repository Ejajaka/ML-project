"""
Survival Causal Rule Ensemble (SCRE)
Wan, Tanioka, Shimokawa (2024), Statistics in Medicine.

Implementation of:
  "Survival causal rule ensemble method considering the main effect for
   estimating heterogeneous treatment effects"
   
Key differences from CISCaRL:
  - Uses elastic net (not stability selection) for rule selection
  - Generates rules from survival trees (prognostic + predictive)
  - No conformal prediction intervals
  - Output is weighted ensemble, not rule list
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest
from lifelines import CoxPHFitter
import pandas as pd


def _extract_rules_from_tree(tree_, X, min_support=5):
    """Extract structured rules from a single decision tree.
    Same format as cis_carl for compatibility.
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
    """Extract structured rules from ensemble trees."""
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

    Parameters
    ----------
    alpha : float
        ElasticNet mixing parameter (0=Ridge, 1=Lasso). Default 0.5.
    pred_n_estimators : int
        Trees in predictive RF (CSF approx). Default 200.
    pred_max_depth : int
        Max depth of predictive trees. Default 5.
    progn_n_estimators : int
        Trees in prognostic survival forest. Default 200.
    progn_max_depth : int
        Max depth of prognostic trees. Default 5.
    rule_min_support : int
        Min patients in leaf for rule extraction. Default 10.
    max_rules : int
        Max candidate rules. Default 2000.
    max_rule_conditions : int or None
        Max conditions per rule. Default 4.
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
                 random_state=42):

        self.alpha = alpha
        self.pred_n_estimators = pred_n_estimators
        self.pred_max_depth = pred_max_depth
        self.progn_n_estimators = progn_n_estimators
        self.progn_max_depth = progn_max_depth
        self.rule_min_support = rule_min_support
        self.max_rules = max_rules
        self.max_rule_conditions = max_rule_conditions
        self.random_state = random_state

        self.selected_rules_ = []
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

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

        # 3. Combine, deduplicate
        seen = set()
        deduped = []
        for cond, mask in rules_pred + rules_progn:
            key = _conditions_to_str(cond,
                                     self.feature_names_ or [f"X{i}" for i in range(X.shape[1])])
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

    def fit(self, X, pseudo_ite, known_mask, feature_names=None,
            time=None, event=None):
        """Fit SCRE model.

        Parameters
        ----------
        X : ndarray (n, p)
        pseudo_ite : ndarray (n,)
        known_mask : ndarray (n,) bool
        feature_names : list of str
        time : ndarray (n,) — survival time (required for prognostic rules)
        event : ndarray (n,) bool — event indicator (required for prognostic rules)
        """
        self.feature_names_ = (list(feature_names) if feature_names
                               else [f"X{i}" for i in range(X.shape[1])])

        idx_known = known_mask & ~np.isnan(pseudo_ite)

        print("  SCRE Stage 1: Generating candidate rules...")
        candidates, cate_pred = self._generate_candidate_rules(
            X, pseudo_ite, idx_known, time, event
        )
        n_cand = len(candidates)
        print(f"    Total rules (deduped): {n_cand}")

        if n_cand == 0:
            print("    WARNING: No candidate rules.")
            return

        # Build rule matrix
        n = len(X)
        R = np.zeros((n, n_cand))
        for j, (_, mask) in enumerate(candidates):
            R[:, j] = mask.astype(float)

        print("  SCRE Stage 2: Elastic Net selection...")
        # Use pseudo_ite as target (matching SCRE approach)
        enet = ElasticNetCV(
            l1_ratio=self.alpha,
            cv=5,
            random_state=self.random_state,
            max_iter=10000,
            alphas=np.logspace(-4, 1, 50),
        )
        enet.fit(R[idx_known], pseudo_ite[idx_known])

        # Use 1se rule for sparsity
        mse_mean = enet.mse_path_.mean(axis=1)
        mse_se = enet.mse_path_.std(axis=1) / np.sqrt(5)
        min_i = np.argmin(mse_mean)
        cand = np.where(mse_mean <= mse_mean[min_i] + mse_se[min_i])[0]
        alpha_1se = enet.alphas_[cand[-1]] if len(cand) > 0 else enet.alphas_[min_i]

        enet_final = ElasticNet(alpha=alpha_1se, l1_ratio=self.alpha, max_iter=10000)
        enet_final.fit(R[idx_known], pseudo_ite[idx_known])

        selected = np.where(np.abs(enet_final.coef_) > 1e-6)[0]

        # Store results
        self.selected_rules_ = []
        for j in selected:
            self.selected_rules_.append({
                'conditions': candidates[j][0],
                'coef': enet_final.coef_[j],
            })

        self.coef_ = enet_final.coef_
        self.intercept_ = enet_final.intercept_
        self.candidates_ = candidates
        self.R_ = R
        self.is_fitted_ = True

        print(f"    Selected {len(selected)} rules (alpha={alpha_1se:.4f})")
        return self

    def predict(self, X):
        """Predict CATE as weighted ensemble of rules."""
        if not self.is_fitted_ or len(self.selected_rules_) == 0:
            return np.zeros(len(X))

        # Build rule matrix for new data
        R = np.zeros((len(X), len(self.candidates_)))
        for j, (_, mask) in enumerate(self.candidates_):
            R[:, j] = _eval_rule(self.candidates_[j][0], X).astype(float)

        return self.intercept_ + R @ self.coef_

    def print_rules(self, feature_names=None):
        """Print selected rules."""
        fn = feature_names or self.feature_names_ or []
        print(f"\nSCRE: {len(self.selected_rules_)} selected rules")
        for i, r in enumerate(self.selected_rules_):
            cond_str = _conditions_to_str(r['conditions'], fn)
            print(f"  Rule {i+1}: [{r['coef']:+.4f}] {cond_str}")

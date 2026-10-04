"""
CISCaRL Live engine.

Fits all comparison methods on the REAL ACTG175 HIV trial, extracts readable
rules for every rule-producing method, and exposes:

    fit_engine()                -> builds and returns the serving engine
    predict_all(engine, patient)-> {method: cate} for one patient
    all_rules(engine)           -> {method: [ {rule, effect, ...}, ... ]}
    feature_spec(engine)        -> input metadata (names, ranges, demo patient)

The engine is a plain dict of fitted objects / numpy arrays and is persisted
with joblib.  Serving requires the repo's `python/` modules on sys.path
(cis_carl, scre, utils, data).
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYDIR = os.path.join(ROOT, "python")
if PYDIR not in sys.path:
    sys.path.insert(0, PYDIR)

from data import load_actg175
from utils import compute_pseudo_ite_dr
from cis_carl import (CISCaRL, _extract_rules_from_ensemble, _conditions_to_str,
                      _eval_rule as _eval_rule_cc)
from scre import SurvivalCausalRuleEnsemble

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

RANDOM_STATE = 42
ALPHAS_100 = np.logspace(-3, 2, 100)
ALPHAS_50 = np.logspace(-4, 1, 50)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _eval_rule(conditions, X):
    """conditions = list of (feat_idx, op, thr); X = 1x p array."""
    mask = np.ones(len(X), dtype=bool)
    for feat, op, thr in conditions:
        if op == "<=":
            mask &= X[:, feat] <= thr
        else:
            mask &= X[:, feat] > thr
    return mask


def _rule_matrix(rules, X):
    R = np.zeros((len(X), len(rules)))
    for j, (cond, _) in enumerate(rules):
        R[:, j] = _eval_rule(cond, X).astype(float)
    return R


def _lasso_1se(alpha_path, mse_mean, mse_std, folds=5):
    mi = int(np.argmin(mse_mean))
    cand = np.where(mse_mean <= mse_mean[mi] + mse_std[mi] / np.sqrt(folds))[0]
    return alpha_path[cand[-1]] if len(cand) else alpha_path[mi]


def _fit_rule_lasso(gb_or_rf, X, y, idx_k, feats, max_trees, kind="lasso"):
    """Extract rules from a tree ensemble, select a sparse subset, return
    (selected rules with coefs, intercept, total candidate count)."""
    rules = _extract_rules_from_ensemble(gb_or_rf, X, max_trees=max_trees,
                                         min_support=10)
    if not rules:
        return [], 0.0, 0
    R = _rule_matrix(rules, X)
    if kind == "lasso":
        cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000,
                     alphas=ALPHAS_100).fit(R[idx_k], y[idx_k])
        alpha = _lasso_1se(cv.alphas_, cv.mse_path_.mean(axis=1),
                           cv.mse_path_.std(axis=1))
        final = Lasso(alpha=alpha, max_iter=10000).fit(R[idx_k], y[idx_k])
    else:
        cv = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=RANDOM_STATE,
                          max_iter=10000, alphas=ALPHAS_50).fit(R[idx_k], y[idx_k])
        alpha = _lasso_1se(cv.alphas_, cv.mse_path_.mean(axis=1),
                           cv.mse_path_.std(axis=1))
        final = ElasticNet(alpha=alpha, l1_ratio=0.5,
                           max_iter=10000).fit(R[idx_k], y[idx_k])
    coef = final.coef_
    sel = np.where(np.abs(coef) > 1e-6)[0]
    selected = []
    for j in sel:
        selected.append({"conditions": rules[j][0],
                         "condition_str": _conditions_to_str(rules[j][0], feats),
                         "coef": float(coef[j])})
    return selected, float(final.intercept_), len(rules)


def _predict_rules_linear(sel, intercept, patient_row, feats):
    x = patient_row.reshape(1, -1)
    val = intercept
    for r in sel:
        if _eval_rule(r["conditions"], x)[0]:
            val += r["coef"]
    return float(val)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def fit_engine():
    df, covs = load_actg175()
    X = df[covs].values.astype(float)
    a = df["treatment"].values.astype(int)
    t = df["time_days"].values.astype(float)
    e = df["event"].values.astype(int)

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    X_tr, X_te = X[tr], X[te]
    a_tr, a_te = a[tr], a[te]
    t_tr, t_te = t[tr], t[te]
    e_tr, e_te = e[tr], e[te]
    t_star = float(np.percentile(t_tr[e_tr == 1], 50))

    # nuisance
    eps = 1e-8
    rf_p = RandomForestClassifier(n_estimators=200, max_depth=5,
                                  random_state=RANDOM_STATE).fit(X_tr, a_tr)
    e_tr_p = rf_p.predict_proba(X_tr)[:, 1]
    e_te_p = rf_p.predict_proba(X_te)[:, 1]

    def fit_rsf(X_, t_, e_):
        y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))],
                     dtype=[("event", bool), ("time", float)])
        return RandomSurvivalForest(n_estimators=200, max_depth=5,
                                    random_state=RANDOM_STATE,
                                    min_samples_leaf=10).fit(X_, y)

    def surv_at(m, X_, ts):
        sf = m.predict_survival_function(X_, return_array=True)
        return np.array([np.interp(ts, m.unique_times_.astype(float), sf[i])
                         for i in range(len(X_))])

    rsf_t = fit_rsf(X_tr[a_tr == 1], t_tr[a_tr == 1], e_tr[a_tr == 1])
    rsf_c = fit_rsf(X_tr[a_tr == 0], t_tr[a_tr == 0], e_tr[a_tr == 0])
    s1_tr, s0_tr = surv_at(rsf_t, X_tr, t_star), surv_at(rsf_c, X_tr, t_star)
    ystar, k = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, t_star,
                                     e_tr_p, s0_tr, s1_tr)
    idx_k = k & ~np.isnan(ystar)

    # ---- Cox T-learner (standardized + small ridge; drop constant columns) ----
    mu_x, sd_x = X_tr.mean(axis=0), X_tr.std(axis=0) + eps
    Z_tr = (X_tr - mu_x) / sd_x
    cph_t = cph_c = None
    keep_t = keep_c = list(covs)
    try:
        def _cox_fit(mask):
            sub = pd.DataFrame(Z_tr[mask], columns=covs)
            sub["time"] = t_tr[mask]; sub["event"] = e_tr[mask]
            keep = [c for c in covs if sub[c].std() > 1e-8]
            m = CoxPHFitter(penalizer=0.1).fit(sub[keep + ["time", "event"]],
                                               "time", "event")
            return m, keep
        cph_t, keep_t = _cox_fit(a_tr == 1)
        cph_c, keep_c = _cox_fit(a_tr == 0)
    except Exception as ex:
        print("  [warn] Cox fit failed:", ex)

    # ---- CSF (RF on pseudo-ITE) ----
    csf = RandomForestRegressor(n_estimators=500, max_depth=5,
                                random_state=RANDOM_STATE,
                                min_samples_leaf=10).fit(X_tr[idx_k], ystar[idx_k])

    # ---- Bo & Ding (GBM + Lasso) ----
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                   random_state=RANDOM_STATE).fit(X_tr[idx_k],
                                                                  ystar[idx_k])
    bd_sel, bd_int, bd_n = _fit_rule_lasso(gb, X_tr, ystar, idx_k, covs,
                                           max_trees=100, kind="lasso")

    # ---- Hybrid (RF + Lasso) ----
    rf_h = RandomForestRegressor(n_estimators=200, max_depth=10,
                                 random_state=RANDOM_STATE,
                                 min_samples_leaf=5).fit(X_tr[idx_k], ystar[idx_k])
    hy_sel, hy_int, hy_n = _fit_rule_lasso(rf_h, X_tr, ystar, idx_k, covs,
                                           max_trees=200, kind="lasso")

    # ---- CRE (RF + ElasticNet) ----
    cre_sel, cre_int, cre_n = _fit_rule_lasso(rf_h, X_tr, ystar, idx_k, covs,
                                              max_trees=200, kind="enet")

    # ---- SCRE ----
    scre = SurvivalCausalRuleEnsemble(alpha=0.5, pred_n_estimators=200,
                                      progn_n_estimators=200, rule_min_support=10,
                                      max_rules=2000, max_rule_conditions=4,
                                      random_state=RANDOM_STATE)
    scre.fit(X_tr, ystar, idx_k, feature_names=covs, time=t_tr,
             event=e_tr.astype(bool), treatment=a_tr)
    scre_rules = []
    for r in scre.selected_rules_:
        scre_rules.append({"conditions": r["conditions"],
                           "condition_str": _conditions_to_str(r["conditions"], covs),
                           "coef": float(r["coef"])})

    # ---- CISCaRL (posthoc) ----
    # Tuned for rule COVERAGE so arbitrary patient inputs usually land in a rule
    # (more rules, lower stability bar, <=3 conditions => broader rules).
    cis = CISCaRL(B=200, stability_threshold=0.6, alpha=0.10,
                  csf_n_estimators=300, csf_max_depth=10, gbm_n_estimators=150,
                  gbm_max_depth=3, rule_min_support=10, calib_split=0.3,
                  max_rules=3000, max_rule_conditions=3, max_selected_rules=25,
                  min_calib_support=5, shrinkage=0.5, mode="posthoc",
                  random_state=RANDOM_STATE)
    cis.fit(X_tr, ystar, idx_k, feature_names=covs)

    # ---- feature metadata (ranges from full data) ----
    ranges = {}
    for j, c in enumerate(covs):
        col = X[:, j]
        ranges[c] = {"min": float(np.min(col)), "max": float(np.max(col)),
                     "median": float(np.median(col)),
                     "p05": float(np.percentile(col, 5)),
                     "p95": float(np.percentile(col, 95))}
    # deterministic demo patient: a test patient that matches a CISCaRL rule
    try:
        _, rids_all = cis.predict(X_te, return_details=True)
        match = np.where(rids_all >= 0)[0]
        demo_local = int(match[0]) if len(match) else 0
    except Exception:
        demo_local = 0
    demo_idx = int(te[demo_local])
    demo = {c: float(X[demo_idx, j]) for j, c in enumerate(covs)}

    engine = {
        "feature_names": list(covs),
        "t_star": t_star,
        "ranges": ranges,
        "demo_patient": demo,
        "cox": {"t": cph_t, "c": cph_c, "mu": mu_x, "sd": sd_x,
                "keep_t": keep_t, "keep_c": keep_c},
        "csf": csf,
        "rule_methods": {
            "Bo & Ding": {"rules": bd_sel, "intercept": bd_int, "n_total": bd_n},
            "Hybrid": {"rules": hy_sel, "intercept": hy_int, "n_total": hy_n},
            "CRE": {"rules": cre_sel, "intercept": cre_int, "n_total": cre_n},
        },
        "scre": {"model": scre, "rules": scre_rules},
        "cis": cis,
        "meta": {"n": int(len(X)), "n_events": int(e.sum()),
                 "n_treated": int(a.sum()), "train": int(ntr), "test": int(len(te))},
    }
    return engine


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
def predict_all(engine, patient: dict) -> dict:
    covs = engine["feature_names"]
    x = np.array([[float(patient.get(c, engine["ranges"][c]["median"]))
                   for c in covs]])
    out = {}

    # Cox T-learner: S1(t*) - S0(t*)
    cox = engine.get("cox", {})
    if cox.get("t") is not None:
        try:
            xs = (x - cox["mu"]) / cox["sd"]
            dfx = pd.DataFrame(xs, columns=covs)
            st = cox["t"].predict_survival_function(dfx[cox["keep_t"]], times=[engine["t_star"]]).values[0][0]
            sc = cox["c"].predict_survival_function(dfx[cox["keep_c"]], times=[engine["t_star"]]).values[0][0]
            out["Cox T-learner"] = float(st - sc)
        except Exception:
            out["Cox T-learner"] = None
    else:
        out["Cox T-learner"] = None

    # CSF
    out["CSF (black-box)"] = float(engine["csf"].predict(x)[0])

    # rule-linear methods
    for name, m in engine["rule_methods"].items():
        out[name] = _predict_rules_linear(m["rules"], m["intercept"], x[0], covs)

    # SCRE
    try:
        out["SCRE"] = float(engine["scre"]["model"].predict(x)[0])
    except Exception:
        out["SCRE"] = None

    # CISCaRL
    try:
        cate = engine["cis"].predict(x)
        _, rids = engine["cis"].predict(x, return_details=True)
        out["CISCaRL"] = float(cate[0])
    except Exception:
        out["CISCaRL"] = None
    return out


def cis_matched_rule(engine, patient: dict):
    covs = engine["feature_names"]
    x = np.array([[float(patient.get(c, engine["ranges"][c]["median"]))
                   for c in covs]])
    _, rids = engine["cis"].predict(x, return_details=True)
    rid = int(rids[0])
    if rid == -1:
        d = engine["cis"].default_cate_
        return {"matched_index": -1, "condition_str": "Default (no rule matched)",
                "mean_cate": float(d["mean"]), "ci_low": float(d["ci_low"]),
                "ci_high": float(d["ci_high"]), "stability": None,
                "support": int(d["support"]), "recommendation": _rec(d["ci_low"], d["ci_high"], d["mean"])}
    r = engine["cis"].selected_rules_[rid]
    return {"matched_index": rid, "condition_str": r["condition_str"],
            "mean_cate": float(r["mean_cate"]), "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]), "stability": float(r["stability"]),
            "support": int(r["support"]),
            "recommendation": _rec(r["ci_low"], r["ci_high"], r["mean_cate"])}


def _rec(lo, hi, mean):
    if lo > 0:
        return "HIGH CONFIDENCE: Recommend Treat"
    if hi < 0:
        return "HIGH CONFIDENCE: Recommend Avoid"
    if mean > 0:
        return "SUGGESTIVE: Possible benefit, more data needed"
    if mean < 0:
        return "SUGGESTIVE: Possible harm, more data needed"
    return "INCONCLUSIVE: Effect near zero"


def all_rules(engine) -> dict:
    """Every method's rules (the asset) or a count if it emits none."""
    out = {}
    # CISCaRL (rich)
    cis_rules = []
    for i, r in enumerate(engine["cis"].selected_rules_):
        cis_rules.append({
            "index": i + 1, "condition_str": r["condition_str"],
            "mean_cate": float(r["mean_cate"]), "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]), "stability": float(r["stability"]),
            "support": int(r["support"]),
            "recommendation": _rec(r["ci_low"], r["ci_high"], r["mean_cate"]),
        })
    out["CISCaRL"] = {"kind": "rich", "count": len(cis_rules), "rules": cis_rules}

    scre_rules = []
    for i, r in enumerate(engine["scre"]["rules"]):
        scre_rules.append({"index": i + 1, "condition_str": r["condition_str"],
                           "coef": r["coef"]})
    out["SCRE"] = {"kind": "coef", "count": len(scre_rules), "rules": scre_rules}

    for name, m in engine["rule_methods"].items():
        ordered = sorted(m["rules"], key=lambda r: -abs(r["coef"]))
        shown = ordered[:25]
        rules = [{"index": i + 1, "condition_str": r["condition_str"],
                  "coef": r["coef"]} for i, r in enumerate(shown)]
        out[name] = {"kind": "coef", "count": len(m["rules"]),
                     "shown": len(shown), "n_candidates": m["n_total"],
                     "rules": rules}

    out["CSF (black-box)"] = {"kind": "none", "count": 0, "rules": [],
                              "note": "Black-box forest: no interpretable rules."}
    out["Cox T-learner"] = {"kind": "none", "count": 0, "rules": [],
                            "note": "Proportional-hazards model: no rule list."}
    return out

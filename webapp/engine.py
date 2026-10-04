"""
CISCaRL Live engine.

Uses the SHARED pipeline (python/pipeline.py) so the live demo is the SAME code
path as the offline benchmark. Canonical setting:
    ACTG175 + AFT-Gumbel + rescaled regime + seed 1039
which is exactly the fair_benchmark_rescaled.csv row for
(dataset=ACTG175, DGP=AFT-Gumbel, Rep=0).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYDIR = os.path.join(ROOT, "python")
if PYDIR not in sys.path:
    sys.path.insert(0, PYDIR)

import pipeline
from cis_carl import _eval_rule, _conditions_to_str

CANONICAL = {"dataset": "ACTG175", "dgp": "AFT-Gumbel", "regime": "rescaled",
             "seed": 1039, "quick": True}


def _predict_linear(mdl, x):
    val = mdl["intercept"]
    for j, (cond, _) in enumerate(mdl["rules"]):
        if _eval_rule(cond, x)[0]:
            val += float(mdl["coef"][j])
    return float(val)


def fit_engine():
    t, e, a, X, covs, true_cate, info = pipeline.make_data(
        CANONICAL["dataset"], CANONICAL["dgp"],
        seed=CANONICAL["seed"], regime=CANONICAL["regime"])
    tr, te = pipeline.split_indices(len(X))
    X_tr, X_te = X[tr], X[te]
    t_tr, t_te = t[tr], t[te]
    e_tr, e_te = e[tr].astype(int), e[te].astype(int)
    a_tr, a_te = a[tr], a[te]
    true_te = true_cate[te]

    models, preds = pipeline.fit_all(X_tr, t_tr, e_tr, a_tr, covs, X_te,
                                     quick=CANONICAL["quick"])
    evals = pipeline.evaluate(preds, true_te, models)

    # feature ranges (from the real ACTG175 covariates)
    ranges = {}
    for j, c in enumerate(covs):
        col = X[:, j]
        ranges[c] = {"min": float(np.min(col)), "max": float(np.max(col)),
                     "median": float(np.median(col)),
                     "p05": float(np.percentile(col, 5)),
                     "p95": float(np.percentile(col, 95))}

    # demo patient: a test patient that matches a CISCaRL (posthoc) rule
    cis = models["cis_posthoc"]
    try:
        _, rids = cis.predict(X_te, return_details=True)
        match = np.where(rids >= 0)[0]
        demo_local = int(match[0]) if len(match) else 0
    except Exception:
        demo_local = 0
    demo_idx = int(te[demo_local])
    demo = {c: float(X[demo_idx, j]) for j, c in enumerate(covs)}

    engine = {
        "feature_names": list(covs),
        "t_star": float(models["t_star"]),
        "ranges": ranges,
        "demo_patient": demo,
        "models": models,
        "instance_metrics": evals,
        "canonical": CANONICAL,
        "meta": {"n": int(len(X)), "n_events": int(e.sum()),
                 "n_treated": int(a.sum()), "train": int(len(X_tr)),
                 "test": int(len(X_te)),
                 "note": "semi-synthetic: real ACTG175 covariates + treatment, "
                         "simulated outcome with a known effect (rescaled regime)",
                 "note_cox": "Cox T-learner does not converge on ACTG175 "
                             "(constant covariate zprior); it returns 0, "
                             "matching the offline benchmark."},
    }
    return engine


# ---------------------------------------------------------------------------
def predict_all(engine, patient: dict) -> dict:
    covs = engine["feature_names"]
    x = np.array([[float(patient.get(c, engine["ranges"][c]["median"]))
                   for c in covs]])
    M = engine["models"]
    out = {}

    # Cox T-learner. NOTE: on ACTG175 the Cox fit does not converge (the
    # covariate `zprior` is constant), so the offline benchmark falls back to
    # zeros; we mirror that here so offline == website.
    cox = M.get("cox")
    if cox is not None:
        try:
            import pandas as pd
            dfx = pd.DataFrame(x, columns=covs)
            st = cox[0].predict_survival_function(dfx, times=[engine["t_star"]]).values[0][0]
            sc = cox[1].predict_survival_function(dfx, times=[engine["t_star"]]).values[0][0]
            out["Cox T-learner"] = float(st - sc)
        except Exception:
            out["Cox T-learner"] = 0.0
    else:
        out["Cox T-learner"] = 0.0

    out["CSF (black-box)"] = float(M["csf"].predict(x)[0]) if M.get("csf") is not None else None
    for name, key in [("Bo & Ding", "bd"), ("Hybrid", "hybrid"), ("CRE", "cre")]:
        out[name] = _predict_linear(M[key], x) if M.get(key) else None
    out["SCRE"] = float(M["scre"].predict(x)[0]) if M.get("scre") is not None else None
    try:
        out["CISCaRL"] = float(M["cis_posthoc"].predict(x)[0])
    except Exception:
        out["CISCaRL"] = None
    return out


def cis_matched_rule(engine, patient: dict):
    covs = engine["feature_names"]
    x = np.array([[float(patient.get(c, engine["ranges"][c]["median"]))
                   for c in covs]])
    cis = engine["models"]["cis_posthoc"]
    _, rids = cis.predict(x, return_details=True)
    rid = int(rids[0])
    if rid == -1:
        d = cis.default_cate_
        return {"matched_index": -1, "condition_str": "Default (no rule matched)",
                "mean_cate": float(d["mean"]), "ci_low": float(d["ci_low"]),
                "ci_high": float(d["ci_high"]), "stability": None,
                "support": int(d["support"]),
                "recommendation": _rec(d["ci_low"], d["ci_high"], d["mean"])}
    r = cis.selected_rules_[rid]
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
    covs = engine["feature_names"]
    M = engine["models"]
    out = {}

    # CISCaRL (posthoc) — rich rules
    cis = M["cis_posthoc"]
    cis_rules = []
    for i, r in enumerate(cis.selected_rules_):
        cis_rules.append({
            "index": i + 1, "condition_str": r["condition_str"],
            "mean_cate": float(r["mean_cate"]), "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]), "stability": float(r["stability"]),
            "support": int(r["support"]),
            "recommendation": _rec(r["ci_low"], r["ci_high"], r["mean_cate"]),
        })
    out["CISCaRL"] = {"kind": "rich", "count": len(cis_rules), "rules": cis_rules}

    # SCRE — coefficient rules
    scre = M.get("scre")
    s_rules = []
    if scre is not None:
        for i, r in enumerate(scre.selected_rules_):
            s_rules.append({"index": i + 1,
                            "condition_str": _conditions_to_str(r["conditions"], covs),
                            "coef": float(r["coef"])})
    out["SCRE"] = {"kind": "coef", "count": len(s_rules), "rules": s_rules}

    # Bo & Ding / Hybrid / CRE — candidate rules, keep nonzero-coef ones
    for name, key in [("Bo & Ding", "bd"), ("Hybrid", "hybrid"), ("CRE", "cre")]:
        mdl = M.get(key)
        if not mdl:
            out[name] = {"kind": "coef", "count": 0, "rules": [], "n_candidates": 0}
            continue
        rules = mdl["rules"]; coef = mdl["coef"]
        sel = np.where(np.abs(coef) > 1e-6)[0]
        ordered = sorted(sel.tolist(), key=lambda j: -abs(coef[j]))
        shown = ordered[:25]
        rlist = [{"index": i + 1,
                  "condition_str": _conditions_to_str(rules[j][0], covs),
                  "coef": float(coef[j])} for i, j in enumerate(shown)]
        out[name] = {"kind": "coef", "count": int(len(sel)),
                     "shown": len(shown), "n_candidates": len(rules), "rules": rlist}

    out["CSF (black-box)"] = {"kind": "none", "count": 0, "rules": [],
                              "note": "Black-box forest: no interpretable rules."}
    out["Cox T-learner"] = {"kind": "none", "count": 0, "rules": [],
                            "note": "Proportional-hazards model: no rule list."}
    return out

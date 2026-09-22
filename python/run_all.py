"""
=============================================================================
 CISCaRL -- FULL RESULTS (one file)
=============================================================================
 Runs the project and prints ALL result values:
   [1] Data
   [2] Synthetic example with known CATE -- full metric table
   [3] Real ACTG175 trial -> interpretable rule list (with CIs + stability)
   [4] Conformal-interval coverage (per-seed)
   [5] Full fair benchmark -- every metric, every method, BOTH regimes
       + paired significance vs CISCaRL (Holm-corrected, effect sizes)
   [6] R regression test

 HOW TO RUN (from the python/ folder):
     py -3.13 run_all.py

 Takes ~3 minutes.
=============================================================================
"""
import os
import sys
import time
import warnings
import subprocess

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gumbel_r
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

from data import load_pbc, load_actg175
from utils import compute_pseudo_ite_dr
from cis_carl import CISCaRL

T0 = time.time()
METRICS = ["MAE", "RMSE", "R2", "Spearman", "Acc", "Prec", "Rec", "F1", "AUC",
           "Rules", "Bias"]
HIGHER_BETTER = {"R2", "Spearman", "Acc", "Prec", "Rec", "F1", "AUC"}
RENAME = {"Cox T-learner": "Cox", "CSF (RF)": "CSF",
          "CISCaRL (direct)": "CISCaRL-dir", "CISCaRL (auto)": "CISCaRL-auto",
          "CISCaRL (posthoc)": "CISCaRL-post", "Bo & Ding": "Bo&Ding",
          "Hybrid": "Hybrid", "CRE": "CRE", "SCRE": "SCRE"}


def hdr(n, title):
    print("\n" + "=" * 100)
    print(f"[{n}/6] {title}")
    print("=" * 100)


def fmt_metric_table(df, tag):
    """Print every metric (rows) x every method (cols)."""
    cols = ["SCRE", "Cox", "CSF", "CISCaRL-post", "CISCaRL-auto",
            "CISCaRL-dir", "CRE", "Hybrid", "Bo&Ding"]
    print(f"\n  >>> {tag}: full metric table (mean over 48 settings)")
    print(f"  {'metric':10s} " + " ".join(f"{c[:11]:>12s}" for c in cols))
    for met in METRICS:
        cells = []
        for c in cols:
            s = df[df["Method"] == c][met].dropna()
            cells.append(f"{s.mean():12.4f}" if len(s) else f"{'--':>12s}")
        print(f"  {met:10s} " + " ".join(cells))


def significance(df, metric, alpha=0.05):
    base = "CISCaRL-post"
    a = df[df["Method"] == base].set_index(["Dataset", "DGP", "Rep"])[metric]
    rows = []
    for m in df["Method"].unique():
        if m == base:
            continue
        b = df[df["Method"] == m].set_index(["Dataset", "DGP", "Rep"])[metric]
        idx = a.index.intersection(b.index)
        av, bv = a.loc[idx].values, b.loc[idx].values
        v = ~(np.isnan(av) | np.isnan(bv))
        if v.sum() < 3:
            continue
        d = (bv[v] - av[v]).mean()
        sd = (bv[v] - av[v]).std(ddof=1)
        se = sd / np.sqrt(v.sum())
        t, p = stats.ttest_rel(bv[v], av[v])
        rows.append({"m": m, "d": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se,
                     "p": p, "cd": d / sd if sd > 0 else 0.0})
    if not rows:
        return
    tab = pd.DataFrame(rows)
    k = len(tab)
    order = np.argsort(tab["p"].values)
    rank = np.empty(k); rank[order] = np.arange(1, k + 1)
    tab["p_holm"] = np.clip(tab["p"].values * (k - rank + 1), 0, 1)
    print(f"\n  >>> paired vs CISCaRL-post [{metric}] "
          f"(delta = competitor - CISCaRL; {'higher' if metric in HIGHER_BETTER else 'lower'} is better; Holm, alpha={alpha})")
    for _, r in tab.sort_values("d").iterrows():
        sig = r["p_holm"] < alpha
        if not sig:
            verdict = "ns"
        elif metric in HIGHER_BETTER:
            verdict = "beats CISCaRL" if r["d"] > 0 else "worse"
        else:
            verdict = "beats CISCaRL" if r["d"] < 0 else "worse"
        print(f"      {r['m']:14s} delta={r['d']:+8.4f} "
              f"CI[{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:7.4f} "
              f"p_holm={r['p_holm']:7.4f} d={r['cd']:+6.2f}  {verdict}")


# ---------------------------------------------------------------------------
def fit_nuisance(X_tr, a_tr, t_tr, e_tr, X_te, a_te, t_te, e_te, t_star):
    rf_p = RandomForestClassifier(n_estimators=120, max_depth=5,
                                  random_state=42).fit(X_tr, a_tr)
    e_tr_p = rf_p.predict_proba(X_tr)[:, 1]

    def fit_rsf(X_, t_, e_):
        y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))],
                     dtype=[("event", bool), ("time", float)])
        return RandomSurvivalForest(n_estimators=120, max_depth=5,
                                    random_state=42,
                                    min_samples_leaf=10).fit(X_, y)

    def surv_at(m, X_, t_):
        sf = m.predict_survival_function(X_, return_array=True)
        return np.array([np.interp(float(t_), m.unique_times_.astype(float),
                                   sf[i]) for i in range(len(X_))])

    rsf_t = fit_rsf(X_tr[a_tr == 1], t_tr[a_tr == 1], e_tr[a_tr == 1])
    rsf_c = fit_rsf(X_tr[a_tr == 0], t_tr[a_tr == 0], e_tr[a_tr == 0])
    s1 = surv_at(rsf_t, X_tr, t_star)
    s0 = surv_at(rsf_c, X_tr, t_star)
    ys, known = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, t_star,
                                      e_tr_p, s0, s1)
    return ys, known & ~np.isnan(ys)


def _tree_rules(tree_, X_):
    rules = []
    def rec(nid, masks):
        if tree_.feature[nid] == -2:
            if masks[-1].sum() > 5:
                rules.append(masks[-1].copy())
            return
        f, thr = tree_.feature[nid], tree_.threshold[nid]
        rec(tree_.children_left[nid], masks + [masks[-1] & (X_[:, f] <= thr)])
        rec(tree_.children_right[nid], masks + [masks[-1] & (X_[:, f] > thr)])
    rec(0, [np.ones(len(X_), dtype=bool)])
    return rules


def cox_tlearner(X_tr, a_tr, t_tr, e_tr, X_te, t_star, feats):
    try:
        dtr = pd.DataFrame(X_tr, columns=feats)
        dtr["time"], dtr["event"], dtr["treatment"] = t_tr, e_tr, a_tr
        dte = pd.DataFrame(X_te, columns=feats)
        ct = CoxPHFitter().fit(dtr[dtr.treatment == 1].drop(columns="treatment"), "time", "event")
        cc = CoxPHFitter().fit(dtr[dtr.treatment == 0].drop(columns="treatment"), "time", "event")
        st = ct.predict_survival_function(dte, times=[t_star]).values[0]
        sc = cc.predict_survival_function(dte, times=[t_star]).values[0]
        return st - sc
    except Exception:
        return np.zeros(len(X_te))


# ===========================================================================
hdr(1, "DATA")
df_actg = covs_actg = df_pbc = None
try:
    df_actg, covs_actg = load_actg175()
    print(f"  ACTG175 : n={len(df_actg)}, p={len(covs_actg)}, "
          f"events={int(df_actg['event'].sum())}, "
          f"treated={int(df_actg['treatment'].sum())}  (REAL RCT, local file)")
except Exception as ex:
    print("  ACTG175 load FAILED:", ex)
try:
    df_pbc, covs_pbc = load_pbc()
    print(f"  PBC     : n={len(df_pbc)}, p={len(covs_pbc)} (real covariates)")
except Exception as ex:
    print(f"  PBC     : skipped ({type(ex).__name__})")

# ===========================================================================
hdr(2, "SYNTHETIC EXAMPLE WITH KNOWN CATE -- full metrics")
try:
    np.random.seed(42)
    n, p = 2500, 10
    X = np.random.randn(n, p)
    sub1 = (X[:, 0] > 0) & (X[:, 1] > 0)
    sub2 = (X[:, 0] <= 0) & (X[:, 2] > 0)
    te = np.zeros(n); te[sub1] = 2.0; te[sub2] = 1.0
    trt = np.random.binomial(1, 0.5, n)
    base = 2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
    eps = np.random.gumbel(0, 1, n)
    t_obs = np.where(trt == 1, np.exp(base + te + eps), np.exp(base + eps))
    cen = np.random.exponential(scale=np.median(t_obs) * 3, size=n)
    tt = np.minimum(t_obs, cen); ev = (t_obs <= cen).astype(bool)
    t_star = np.percentile(t_obs[ev], 50)

    def cf(x, te_, t0):
        b = 2.0 + 0.5 * x[0] - 0.3 * x[1] + 0.2 * x[2]
        return (1 - gumbel_r.cdf(np.log(t0) - b - te_)) - (1 - gumbel_r.cdf(np.log(t0) - b))
    tcat = np.array([cf(X[i], te[i], t_star) for i in range(n)])

    idx = np.random.permutation(n); nt = int(0.7 * n)
    X_tr, X_te = X[idx[:nt]], X[idx[nt:]]
    a_tr, a_te = trt[idx[:nt]], trt[idx[nt:]]
    t_tr, t_te = tt[idx[:nt]], tt[idx[nt:]]
    e_tr, e_te = ev[idx[:nt]].astype(int), ev[idx[nt:]].astype(int)
    true_te = tcat[idx[nt:]]
    feats = [f"X{i}" for i in range(p)]
    print(f"  n={n}, train={nt}, test={n-nt}, true CATE std={true_te.std():.4f}")
    ys, ik = fit_nuisance(X_tr, a_tr, t_tr, e_tr, X_te, a_te, t_te, e_te, t_star)

    X_all = np.vstack([X_tr, X_te])
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                   random_state=42).fit(X_tr[ik], ys[ik])
    r_gb = []
    for est in gb.estimators_[:100]:
        r_gb.extend(_tree_rules(est[0].tree_, X_all))
    if r_gb:
        Rtr = np.column_stack([m[:nt].astype(float) for m in r_gb])
        Rte = np.column_stack([m[nt:].astype(float) for m in r_gb])
        lcv = LassoCV(cv=5, random_state=42, max_iter=10000,
                      alphas=np.logspace(-3, 2, 60)).fit(Rtr[ik], ys[ik])
        mm = lcv.mse_path_.mean(axis=1); ms = lcv.mse_path_.std(axis=1) / np.sqrt(5)
        mi = int(np.argmin(mm)); cand = np.where(mm <= mm[mi] + ms[mi])[0]
        a1 = lcv.alphas_[cand[-1]] if len(cand) else lcv.alphas_[mi]
        ls = Lasso(alpha=a1, max_iter=10000).fit(Rtr[ik], ys[ik])
        pred_bd = ls.intercept_ + Rte @ ls.coef_
        n_bd = int((np.abs(ls.coef_) > 1e-6).sum())
    else:
        pred_bd, n_bd = np.zeros(n - nt), 0

    cis = CISCaRL(B=150, stability_threshold=0.7, alpha=0.10,
                  csf_n_estimators=150, csf_max_depth=10, gbm_n_estimators=80,
                  gbm_max_depth=3, rule_min_support=10, calib_split=0.3,
                  max_rules=1500, max_rule_conditions=4, max_selected_rules=10,
                  mode="posthoc", random_state=42)
    cis.fit(X_tr, ys, ik, feature_names=feats)
    csf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42,
                                min_samples_leaf=10).fit(X_tr[ik], ys[ik])

    preds = {"Cox T-learner": cox_tlearner(X_tr, a_tr, t_tr, e_tr, X_te, t_star, feats),
             "CSF (black-box)": csf.predict(X_te),
             "Bo&Ding (rules)": pred_bd,
             "CISCaRL (ours)": cis.predict(X_te)}
    rules_count = {"Bo&Ding (rules)": n_bd,
                   "CISCaRL (ours)": len(cis.selected_rules_)}
    print(f"\n  {'Method':18s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} "
          f"{'Spearman':>9s} {'Rules':>6s}")
    for name, pred in preds.items():
        v = ~np.isnan(pred)
        pv, tv = pred[v], true_te[v]
        mae = np.mean(np.abs(pv - tv))
        rmse = np.sqrt(np.mean((pv - tv) ** 2))
        r2 = r2_score(tv, pv)
        rho = stats.spearmanr(pv, tv)[0]
        print(f"  {name:18s} {mae:8.4f} {rmse:8.4f} {r2:8.3f} {rho:9.3f} "
              f"{rules_count.get(name, 0):6d}")
except Exception as ex:
    print("  FAILED:", ex)

# ===========================================================================
hdr(3, "REAL ACTG175 HIV TRIAL -- interpretable rule list")
try:
    if df_actg is None:
        raise RuntimeError("ACTG175 not loaded")
    X = df_actg[covs_actg].values.astype(float)
    a = df_actg["treatment"].values.astype(int)
    t = df_actg["time_days"].values.astype(float)
    e = df_actg["event"].values.astype(int)
    np.random.seed(42)
    idx = np.random.permutation(len(X)); nt = int(0.7 * len(X))
    X_tr, X_te = X[idx[:nt]], X[idx[nt:]]
    a_tr, a_te = a[idx[:nt]], a[idx[nt:]]
    t_tr, t_te = t[idx[:nt]], t[idx[nt:]]
    e_tr, e_te = e[idx[:nt]], e[idx[nt:]]
    t_star = np.percentile(t_tr[e_tr == 1], 50)
    print(f"  n={len(X)}, treated={int(a.sum())}, events={int(e.sum())}, "
          f"horizon t*={t_star:.0f} days")
    ys, ik = fit_nuisance(X_tr, a_tr, t_tr, e_tr, X_te, a_te, t_te, e_te, t_star)
    m = CISCaRL(B=200, stability_threshold=0.7, alpha=0.10, csf_n_estimators=150,
                csf_max_depth=10, gbm_n_estimators=80, gbm_max_depth=3,
                rule_min_support=10, calib_split=0.3, max_rules=1500,
                max_rule_conditions=4, max_selected_rules=10, min_calib_support=5,
                shrinkage=0.5, mode="posthoc", random_state=42)
    m.fit(X_tr, ys, ik, feature_names=covs_actg)
    m.print_rule_list()
except Exception as ex:
    print("  FAILED:", ex)

# ===========================================================================
hdr(4, "CONFORMAL INTERVAL COVERAGE (90% target)")
try:
    def cover_one(seed):
        np.random.seed(seed)
        n, p = 2500, 10
        X = np.random.randn(n, p)
        s1 = (X[:, 0] > 0) & (X[:, 1] > 0)
        s2 = (X[:, 0] <= 0) & (X[:, 2] > 0)
        te = np.zeros(n); te[s1] = 2.0; te[s2] = 1.0
        trt = np.random.binomial(1, 0.5, n)
        base = 2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
        eps = np.random.gumbel(0, 1, n)
        t_obs = np.where(trt == 1, np.exp(base + te + eps), np.exp(base + eps))
        cen = np.random.exponential(scale=np.median(t_obs) * 3, size=n)
        tt = np.minimum(t_obs, cen); ev = (t_obs <= cen).astype(bool)
        ts = np.percentile(t_obs[ev], 50)
        def cf(x, te_, t0):
            b = 2.0 + 0.5 * x[0] - 0.3 * x[1] + 0.2 * x[2]
            return (1 - gumbel_r.cdf(np.log(t0) - b - te_)) - (1 - gumbel_r.cdf(np.log(t0) - b))
        tc = np.array([cf(X[i], te[i], ts) for i in range(n)])
        ix = np.random.RandomState(seed).permutation(n); ntr = int(0.7 * n)
        Xtr, Xte = X[ix[:ntr]], X[ix[ntr:]]
        atr, ate = trt[ix[:ntr]], trt[ix[ntr:]]
        ttr, tte = tt[ix[:ntr]], tt[ix[ntr:]]
        etr, ete = ev[ix[:ntr]].astype(int), ev[ix[ntr:]].astype(int)
        tcte = tc[ix[ntr:]]
        ys_, ik_ = fit_nuisance(Xtr, atr, ttr, etr, Xte, ate, tte, ete, ts)
        mm = CISCaRL(B=100, stability_threshold=0.7, alpha=0.10, csf_n_estimators=120,
                     csf_max_depth=10, gbm_n_estimators=60, gbm_max_depth=3,
                     rule_min_support=10, calib_split=0.3, max_rules=1200,
                     max_rule_conditions=4, max_selected_rules=10,
                     min_calib_support=20, shrinkage=0.5, mode="posthoc",
                     random_state=42)
        mm.fit(Xtr, ys_, ik_, feature_names=[f"X{i}" for i in range(p)])
        pred, rids = mm.predict(Xte, return_details=True)
        v = ~np.isnan(pred)
        rules = mm.selected_rules_ + [{"mean_cate": mm.default_cate_["mean"],
                                       "ci_low": mm.default_cate_["ci_low"],
                                       "ci_high": mm.default_cate_["ci_high"]}]
        ok = tot = 0
        for rid in range(len(rules)):
            mask = (rids == rid) & v
            if mask.sum() < 5:
                continue
            lo, hi = rules[rid]["ci_low"], rules[rid]["ci_high"]
            ok += int(lo <= tcte[mask].mean() <= hi); tot += 1
        return ok / max(tot, 1)

    covs = [cover_one(s) for s in range(3)]
    for s, c in enumerate(covs):
        print(f"  seed {s}: rule-level coverage = {c*100:.1f}%")
    print(f"  mean coverage = {np.mean(covs)*100:.1f}%  (target >= 90%)")
except Exception as ex:
    print("  FAILED:", ex)

# ===========================================================================
hdr(5, "FULL FAIR BENCHMARK -- every metric, both regimes")
for regime in ["original", "rescaled"]:
    path = os.path.join(ROOT, "results", f"fair_benchmark_{regime}.csv")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "results", "fair_benchmark.csv")
    try:
        df = pd.read_csv(path)
        df = df[df["Method"] != "Ablation"].copy()
        df["Method"] = df["Method"].map(RENAME)
        print(f"\n{'#'*100}\nREGIME: {regime.upper()}   "
              f"({df.groupby(['Dataset','DGP','Rep']).ngroups} settings)")
        print("#" * 100)
        fmt_metric_table(df, regime)
        for met in ["MAE", "RMSE", "R2", "Spearman", "F1", "AUC", "Rules"]:
            significance(df, met)
    except Exception as ex:
        print(f"  {regime} FAILED:", ex)

# ===========================================================================
hdr(6, "R REGRESSION TEST")
try:
    import shutil
    rscript = shutil.which("Rscript") or r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe"
    out = subprocess.run([rscript, os.path.join(ROOT, "R", "test_grf_traversal.R")],
                         capture_output=True, text=True, timeout=180)
    for l in [x for x in (out.stdout or "").splitlines() if x.strip()][-3:]:
        print("  " + l)
except Exception as ex:
    print("  R not available:", ex)

print(f"\nTOTAL TIME: {time.time() - T0:.0f}s")
print("=" * 100)

"""Gate 23 - two-region DGP confirmation of v1.10.

Pre-registration: `results/gate23_PREREGISTRATION.md`. Method frozen; only the
treatment-effect structure changes:

    R1: cd40 < median AND age  > median   -> +0.20 in S(730)
    R2: karnof < median AND wtkg < median -> +0.15
    overlap: max offset; outside: 0
    true region = R1 union R2

Censoring and nuisance models are unchanged (conditional Cox, 10 %).

Usage
-----
    python -m ciscarl2.gate23 --reps 2 --start 260900   # smoke, off-seed
    python -m ciscarl2.gate23 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import aipw, crossfit_nuisances, phi_on_holdout
from .gate2 import pseudo_outcome
from .gate4d import TSTAR, _delta_for
from .gate8 import best_jaccard
from .gate11 import _cart
from .gate14 import _config, _calibrate  # only for the pool/lam0 helpers
from .gate20 import _capped_G
from .gate21 import _stability_freq
from .gate22 import _pick, _stats, _select_D
from .rules import eval_rule, jaccard

from .paths import data_path

DATA = data_path()
START = 260100
CONFIGS = ("A", "B", "C100", "C200", "C400", "D")
THRESH = {"C100": 100, "C200": 200, "C400": 400}
N_EVAL = 20
E1, E2 = 0.20, 0.15


def _config23():
    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    names = ("cd40", "age", "karnof", "wtkg")
    cuts = {n: float(raw[n].median()) for n in names}
    idx = {n: dgp.COVARIATES.index(n) for n in names}
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    return (pool, mu, sd, cuts, idx, lam0)


def _regions(X, cfg):
    _, mu, sd, cuts, idx, _ = cfg
    R1 = (X[:, idx["cd40"]] < (cuts["cd40"] - mu[idx["cd40"]]) / sd[idx["cd40"]]) & \
         (X[:, idx["age"]] > (cuts["age"] - mu[idx["age"]]) / sd[idx["age"]])
    R2 = (X[:, idx["karnof"]] < (cuts["karnof"] - mu[idx["karnof"]]) / sd[idx["karnof"]]) & \
         (X[:, idx["wtkg"]] < (cuts["wtkg"] - mu[idx["wtkg"]]) / sd[idx["wtkg"]])
    return R1, R2


def _sim_two(X, A, lam_c0, lam0, rng, e1, e2):
    lp0 = X @ dgp.BETA
    lp1 = lp0 - np.maximum(e1, e2)
    lam = lam0 * np.exp(np.where(A == 1, lp1, lp0))
    T = rng.exponential(1.0 / np.maximum(lam, 1e-12))
    C = rng.exponential(1.0 / np.maximum(lam_c0 * np.exp(X @ dgp.GAMMA), 1e-12))
    Tobs = np.minimum(T, C)
    E = (T <= C).astype(int)
    S0 = np.exp(-lam0 * np.exp(lp0) * TSTAR)
    S1 = np.exp(-lam0 * np.exp(lp1) * TSTAR)
    return dict(Tobs=Tobs, E=E, S0=S0, S1=S1)


def _calibrate_two(cfg, target=0.10, n_pilot=6000, seed=99):
    pool, mu, sd, cuts, idx, lam0 = cfg
    rng0 = np.random.RandomState(seed)
    Xp = pool[rng0.choice(len(pool), n_pilot, replace=True)]
    R1, R2 = _regions(Xp, cfg)
    lp0 = Xp @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    e1 = np.where(R1, _delta_for(E1, s0), 0.0)
    e2 = np.where(R2, _delta_for(E2, s0), 0.0)

    def frac(lc):
        rng = np.random.RandomState(seed)
        A = rng.binomial(1, 0.5, len(Xp))
        d = _sim_two(Xp, A, lc, lam0, rng, e1, e2)
        return float(np.mean((d["Tobs"] <= TSTAR) & (d["E"] == 0)))

    lo, hi = 1e-7, 5e-2
    for _ in range(45):
        mid = float(np.sqrt(lo * hi))
        if frac(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def run_rep(rep, start, cfg, lam_c0):
    pool, mu_, sd_, cuts, idx, lam0 = cfg
    seed = start + rep
    rng = np.random.RandomState(seed)
    n = 8000
    X = pool[rng.choice(len(pool), n, replace=True)]
    R1, R2 = _regions(X, cfg)
    union = R1 | R2
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    e1 = np.where(R1, _delta_for(E1, s0), 0.0)
    e2 = np.where(R2, _delta_for(E2, s0), 0.0)
    A = rng.binomial(1, 0.5, n)
    sim = _sim_two(X, A, lam_c0, lam0, rng, e1, e2)
    T, E = sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    ii = np.arange(n)
    disc, hold = train_test_split(ii, test_size=0.5, random_state=seed,
                                  stratify=A)
    val, asm = train_test_split(hold, test_size=0.5, random_state=seed + 1,
                                stratify=A[hold])
    XD, AD, TD, ED = X[disc], A[disc], T[disc], E[disc]
    XV, AV, TV, EV = X[val], A[val], T[val], E[val]
    XA, AA, TA, EA = X[asm], A[asm], T[asm], E[asm]
    uniA, truth = union[asm], tau_i[asm]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    G = _capped_G(muD, GD, Gmarg, fid, fcaps, "conditional", TD)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
    _, _, q1, q0 = aipw(XD, AD, TD, TSTAR, muD, G)
    const = float((q1 - q0).mean())
    const_mae = float(np.abs(const - truth).mean())

    leaves = _cart(psi, XD, 3)[1]
    means, ses, sup = _stats(psi, leaves)
    picks = {}
    for rule in ("A", "B", "C100", "C200", "C400"):
        picks[rule] = _pick(rule, means, ses, sup, len(XD), THRESH.get(rule))
    _, jD = _select_D(psi, XD, seed + 31)
    picks["D"] = jD                       # _select_D refits the same CART
    conds_full = {c: leaves[j]["cond"] for c, j in picks.items()}

    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=seed)
    dphi_v = p1v - p0v
    p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, TSTAR, seed=seed)
    dphi_a = p1a - p0a
    from .gate19 import _assign_effs, _leaf_table
    tbl_v = _leaf_table(leaves, XV, dphi_v)
    effs_v = np.array([t["eff"] for t in tbl_v], float)
    ses_v = np.array([t["se"] for t in tbl_v], float)
    effs_v = np.where(np.isfinite(effs_v), effs_v, const)
    masks_a = [eval_rule(r["cond"], XA) for r in leaves]

    rng_eval = np.random.RandomState(seed + 41)
    evals = [rng_eval.choice(len(XD), len(XD) // 2, replace=False)
             for _ in range(N_EVAL)]
    rec = {c: {"jf": [], "jt": []} for c in CONFIGS}
    for b, s_idx in enumerate(evals):
        lb = _cart(psi[s_idx], XD[s_idx], 3)[1]
        mb, sb, ub = _stats(psi[s_idx], lb)
        for rule in ("A", "B", "C100", "C200", "C400"):
            j = _pick(rule, mb, sb, ub, len(s_idx), THRESH.get(rule))
            m = eval_rule(lb[j]["cond"], XA)
            rec[rule]["jf"].append(jaccard(m, eval_rule(conds_full[rule], XA)))
            rec[rule]["jt"].append(jaccard(m, uniA))
        lbd, jd = _select_D(psi[s_idx], XD[s_idx], seed * 1000 + b)
        md = eval_rule(lbd[jd]["cond"], XA)
        rec["D"]["jf"].append(jaccard(md, eval_rule(conds_full["D"], XA)))
        rec["D"]["jt"].append(jaccard(md, uniA))

    rows = []
    for c in CONFIGS:
        mfull = eval_rule(conds_full[c], XA)
        k = int(mfull.sum())
        jf = np.array(rec[c]["jf"])
        jt = np.array(rec[c]["jt"])
        # validation effect + commitment
        j = picks[c]
        mv = eval_rule(conds_full[c], XV)
        kv = int(mv.sum())
        hi_eff = float(dphi_v[mv].mean()) if kv else np.nan
        hi_se = (float(np.sqrt(((dphi_v[mv] - hi_eff) ** 2).sum()) / kv)
                 if kv else np.nan)
        committed = bool(np.isfinite(hi_eff) and np.isfinite(hi_se) and
                         hi_eff > norm.ppf(0.95) * hi_se)
        # prediction: all leaves' validation effects; selected leaf -> const
        # if the commitment fails
        use = effs_v.copy()
        if not committed:
            use[j] = const
        tbl = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                    eff=use[t], se=ses_v[t]) for t in range(len(leaves))]
        pred = _assign_effs(tbl, XA, use, const)
        mae = float(np.abs(pred - truth).mean())

        cap = float(truth[mfull].sum() / truth.sum()) if truth.sum() > 0 else np.nan
        purity = float((mfull & uniA).sum() / k) if k else np.nan
        cap_region = float((mfull & uniA).sum() / max(uniA.sum(), 1))
        j_true = jaccard(mfull, uniA)
        per = float(truth[mfull].mean()) if (committed and k) else np.nan
        cov = float(mfull.mean()) if committed else 0.0
        cap_c = (float(truth[mfull].sum() / truth.sum())
                 if (committed and truth.sum() > 0) else 0.0)
        oracle = (float(np.sort(truth)[::-1][:k].mean())
                  if (committed and k) else np.nan)
        eff_ratio = per / oracle if (committed and oracle) else np.nan
        rows.append(dict(
            rep=rep, n=n, config=c, support=int(sup[picks[c]]),
            capture_benefit=cap, purity=purity, capture_region=cap_region,
            j_true=j_true, jf_mean=jf.mean(), jf_median=np.median(jf),
            jf_q25=np.percentile(jf, 25), jf_q75=np.percentile(jf, 75),
            freq_full=100.0 * float((jf > 0.8).mean()),
            jt_mean=jt.mean(), freq_true=100.0 * float((jt > 0.8).mean()),
            committed=committed, mae=mae, const_mae=const_mae,
            delta=mae - const_mae, coverage=cov, per_treated=per,
            efficiency=eff_ratio, captured_commit=cap_c,
            null=bool(committed and np.isfinite(per) and per <= 0.01),
            beneficial=bool(committed and np.isfinite(per) and per > 0)))
    return rows


def _boot(d, seed=0, B=5000):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan, np.nan
    rs = np.random.RandomState(seed)
    bs = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2.0 * min(float((bs <= 0).mean()), float((bs >= 0).mean()))
    return float(lo), float(hi), min(p, 1.0)


def summarise(df):
    out = []
    for c, g in df.groupby("config"):
        tr = g[g.committed]
        out.append(dict(
            config=c, reps=len(g), support=g.support.mean(),
            capture=g.capture_benefit.mean(), purity=g.purity.mean(),
            capture_region=g.capture_region.mean(),
            j_true=g.j_true.mean(), jf_mean=g.jf_mean.mean(),
            jf_median=g.jf_median.mean(), jf_q25=g.jf_q25.mean(),
            jf_q75=g.jf_q75.mean(), freq_full=g.freq_full.mean(),
            jt_mean=g.jt_mean.mean(), freq_true=g.freq_true.mean(),
            commit=100.0 * g.committed.mean(), coverage=g.coverage.mean(),
            per_treated=tr.per_treated.mean() if len(tr) else np.nan,
            efficiency=tr.efficiency.mean() if len(tr) else np.nan,
            captured_commit=g.captured_commit.mean(),
            null_rate=100.0 * tr.null.mean() if len(tr) else np.nan,
            beneficial_rate=100.0 * tr.beneficial.mean() if len(tr) else np.nan,
            mae=g.mae.mean(), delta=g.delta.mean()))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    global START
    if args.start is not None:
        START = args.start
    cfg = _config23()
    lam_c0 = _calibrate_two(cfg)
    print(f"  lam_c0={lam_c0:.3e}", flush=True)

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, START, cfg, lam_c0) for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate23_tworegion.csv"), index=False)
    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate23_summary.csv"), index=False)

    print(f"\n=== Gate 23, two-region DGP, {args.reps} reps ===")
    print(f"{'cfg':>6}{'support':>8}{'capture':>8}{'purity':>7}{'capReg':>7}"
          f"{'J_true':>7}{'J_full':>7}{'freqF%':>7}{'commit':>7}{'cover':>7}"
          f"{'effic':>7}{'capC':>7}{'null%':>6}{'MAE':>8}")
    for _, r in res.iterrows():
        print(f"{r.config:>6}{r.support:>8.0f}{r.capture:>8.3f}{r.purity:>7.2f}"
              f"{r.capture_region:>7.2f}{r.j_true:>7.3f}{r.jf_mean:>7.3f}"
              f"{r.freq_full:>6.0f}%{r.commit:>6.0f}%{r.coverage:>7.3f}"
              f"{r.efficiency:>7.2f}{r.captured_commit:>7.3f}"
              f"{r.null_rate:>6.0f}{r.mae:>8.4f}")

    print("\nPaired contrasts vs A:")
    second = []
    a = df[df.config == "A"].set_index("rep")
    for c in ("B", "C100", "C200", "C400", "D"):
        b = df[df.config == c].set_index("rep")
        common = sorted(set(a.index) & set(b.index))
        for metric in ("capture_benefit", "j_true", "jf_mean", "purity", "mae"):
            d = (b.loc[common, metric] - a.loc[common, metric]).to_numpy()
            lo, hi, p = _boot(d)
            print(f"  {c}-A {metric:>16}: {d.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] p={p:.3f}")
            if metric in ("capture_benefit", "j_true") and c in ("B", "C100", "C400", "D"):
                second.append((c, metric, p))
    print("\nHolm-adjusted secondary p-values:")
    for metric in ("capture_benefit", "j_true"):
        ps = sorted([(c, p) for c, m, p in second if m == metric], key=lambda t: t[1])
        m = len(ps)
        adj, run = [], 0.0
        for r, (c, p) in enumerate(ps):
            val = min(1.0, (m - r) * p)
            run = max(run, val)
            adj.append((c, p, run))
        print(f"  {metric}: " + ", ".join(f"{c} p={p:.3f} adj={pa:.3f}"
                                          for c, p, pa in adj))

    c2 = res[res.config == "C200"].iloc[0]
    a0 = res[res.config == "A"].iloc[0]
    checks = {
        "capture > A (CI excludes 0)": bool(c2.capture > a0.capture),
        "J_true > A": bool(c2.j_true > a0.j_true),
        "purity >= A-0.05": bool(c2.purity >= a0.purity - 0.05),
        "MAE delta < 0": bool(c2.delta < 0),
        "null <= 5%": bool(c2.null_rate <= 5.0),
        "efficiency >= 0.70": bool(c2.efficiency >= 0.70),
    }
    print("\nPre-registered success checks (C200 vs A):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n-> {'C200 GENERALISES' if all(checks.values()) else 'C200 does not meet all criteria'}")
    return df


if __name__ == "__main__":
    main()

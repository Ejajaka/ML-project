"""Gate 22 - valid stability measurement.

Pre-registration: `results/gate22_PREREGISTRATION.md`. Measurement gate: no rule
is added or tuned.

Each configuration is refit with ITS OWN selection procedure on disjoint
discovery half-samples. Stability is reported as:
  * reference-leaf recovery: Jaccard(subsample leaf, full-data leaf)
  * selection frequency: fraction of subsamples with that Jaccard > 0.8
  * ground-truth recovery: Jaccard(subsample leaf, true region)
The full distribution of Jaccard values is stored, not only the mean.

Support thresholds 100 / 200 (primary) / 400 are scaled by the subsample
fraction for refits.

Usage
-----
    python -m ciscarl2.gate22 --reps 2 --start 250900   # smoke, off-seed
    python -m ciscarl2.gate22 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import aipw, crossfit_nuisances
from .gate2 import pseudo_outcome
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate11 import _cart
from .gate14 import _calibrate, _config, _region, _simulate
from .gate20 import _capped_G
from .gate21 import STAB_B, _stability_freq
from .rules import eval_rule, jaccard

from .paths import data_path

DATA = data_path()
START = 250100
CELLS = ((0.0, "conditional"), (3.0, "marginal"))
CONFIGS = ("A", "B", "C100", "C200", "C400", "D")
THRESH = {"C100": 100, "C200": 200, "C400": 400}
N_EVAL = 20


def _stats(psi, leaves):
    means, ses, sup = [], [], []
    for r in leaves:
        m = r["mask"]
        n = int(m.sum())
        means.append(float(psi[m].mean()) if n else np.nan)
        ses.append(float(psi[m].std(ddof=1) / np.sqrt(n)) if n > 1 else np.inf)
        sup.append(n)
    return np.array(means), np.array(ses), np.array(sup)


def _pick(rule, means, ses, sup, n_ref, thr=None):
    if rule == "A":
        return int(np.nanargmax(means))
    if rule == "B":
        return int(np.nanargmax(means - 1.645 * ses))
    if rule.startswith("C"):
        t = thr * (n_ref / 4000.0)
        t = max(int(round(t)), 5)
        ok = sup >= t
        return int(np.nanargmax(np.where(ok, means, -np.inf))) if ok.any() \
            else int(np.nanargmax(means))
    return int(np.nanargmax(means))


def _select_D(psi, XD, seed):
    leaves = _cart(psi, XD, 3)[1]
    means, _, _ = _stats(psi, leaves)
    freq = _stability_freq(leaves, psi, XD, seed)
    return leaves, int(np.nanargmax(freq * means))


def run_rep(rep, kappa, cen, start, cfg, lam_c0):
    pool, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, lam0 = cfg
    seed = start + rep
    rng = np.random.RandomState(seed)
    n = 8000
    X = pool[rng.choice(len(pool), n, replace=True)]
    R = _region(X, cfg)
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    eff = np.where(R, _delta_for(EFFECT, s0), 0.0)
    A = rng.binomial(1, 0.5, n)
    sim = _simulate(X, A, lam_c0, lam0, rng, eff, R, kappa, cfg)
    T, E = sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    idx = np.arange(n)
    disc, hold = train_test_split(idx, test_size=0.5, random_state=seed,
                                  stratify=A)
    val, asm = train_test_split(hold, test_size=0.5, random_state=seed + 1,
                                stratify=A[hold])
    XD, AD, TD, ED = X[disc], A[disc], T[disc], E[disc]
    XA, regA = X[asm], R[asm]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    G = _capped_G(muD, GD, Gmarg, fid, fcaps, cen, TD)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)

    leaves = _cart(psi, XD, 3)[1]
    means, ses, sup = _stats(psi, leaves)

    conds_full = {}
    for rule in ("A", "B", "C100", "C200", "C400"):
        j = _pick(rule, means, ses, sup, len(XD),
                  THRESH.get(rule))
        conds_full[rule] = leaves[j]["cond"]
    leaves_D, j_D = _select_D(psi, XD, seed + 31)
    conds_full["D"] = leaves_D[j_D]["cond"]

    rng_eval = np.random.RandomState(seed + 41)
    evals = [rng_eval.choice(len(XD), len(XD) // 2, replace=False)
             for _ in range(N_EVAL)]

    rec = {c: {"jf": [], "jt": []} for c in CONFIGS}
    for b, ii in enumerate(evals):
        lb = _cart(psi[ii], XD[ii], 3)[1]
        mb, sb, ub = _stats(psi[ii], lb)
        for rule in ("A", "B", "C100", "C200", "C400"):
            j = _pick(rule, mb, sb, ub, len(ii), THRESH.get(rule))
            m = eval_rule(lb[j]["cond"], XA)
            rec[rule]["jf"].append(jaccard(m, eval_rule(conds_full[rule], XA)))
            rec[rule]["jt"].append(jaccard(m, regA))
        lbd, jd = _select_D(psi[ii], XD[ii], seed * 1000 + b)
        md = eval_rule(lbd[jd]["cond"], XA)
        rec["D"]["jf"].append(jaccard(md, eval_rule(conds_full["D"], XA)))
        rec["D"]["jt"].append(jaccard(md, regA))

    rows = []
    for c in CONFIGS:
        jf = np.array(rec[c]["jf"])
        jt = np.array(rec[c]["jt"])
        mfull = eval_rule(conds_full[c], XA)
        rows.append(dict(
            rep=rep, n=n, kappa=kappa, censoring=cen, config=c,
            jf_mean=jf.mean(), jf_median=np.median(jf),
            jf_q25=np.percentile(jf, 25), jf_q75=np.percentile(jf, 75),
            jf_min=jf.min(), jf_max=jf.max(),
            freq_full=100.0 * float((jf > 0.8).mean()),
            jt_mean=jt.mean(), jt_median=np.median(jt),
            freq_true=100.0 * float((jt > 0.8).mean()),
            j_full_true=jaccard(mfull, regA),
            capture_full=float((mfull & regA).sum() / max(regA.sum(), 1)),
            purity_full=float((mfull & regA).sum() / max(mfull.sum(), 1))))
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
    for (kappa, cen, c), g in df.groupby(["kappa", "censoring", "config"]):
        out.append(dict(
            kappa=kappa, censoring=cen, config=c, reps=len(g),
            jf_mean=g.jf_mean.mean(), jf_median=g.jf_median.mean(),
            jf_q25=g.jf_q25.mean(), jf_q75=g.jf_q75.mean(),
            jf_min=g.jf_min.mean(), freq_full=g.freq_full.mean(),
            jt_mean=g.jt_mean.mean(), freq_true=g.freq_true.mean(),
            j_full_true=g.j_full_true.mean(),
            capture_full=g.capture_full.mean(),
            purity_full=g.purity_full.mean()))
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
    cfg = _config()
    lam_c = {k: _calibrate(cfg, k) for k, _ in CELLS}

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, k, c, START, cfg, lam_c[k])
        for k, c in CELLS for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate22_stability.csv"), index=False)
    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate22_summary.csv"), index=False)

    print(f"\n=== Gate 22 stability, {args.reps} reps ===")
    print(f"{'kappa':>6}{'cens':>12}{'cfg':>6}{'J_full':>8}{'med':>6}"
          f"{'q25':>6}{'q75':>6}{'min':>6}{'freqF%':>8}{'J_true':>8}"
          f"{'freqT%':>8}{'JfullT':>8}{'capture':>8}")
    for _, r in res.sort_values(["kappa", "censoring", "config"]).iterrows():
        print(f"{r.kappa:>6.0f}{r.censoring:>12}{r.config:>6}{r.jf_mean:>8.3f}"
              f"{r.jf_median:>6.2f}{r.jf_q25:>6.2f}{r.jf_q75:>6.2f}"
              f"{r.jf_min:>6.2f}{r.freq_full:>7.0f}%{r.jt_mean:>8.3f}"
              f"{r.freq_true:>7.0f}%{r.j_full_true:>8.3f}{r.capture_full:>8.2f}")

    print("\nPaired contrasts vs A (reference-leaf recovery):")
    second = []
    for (kappa, cen), g in df.groupby(["kappa", "censoring"]):
        a = g[g.config == "A"].set_index("rep")
        for c in ("B", "C100", "C200", "C400", "D"):
            b = g[g.config == c].set_index("rep")
            common = sorted(set(a.index) & set(b.index))
            for metric in ("jf_mean", "freq_full", "jt_mean"):
                d = (b.loc[common, metric] - a.loc[common, metric]).to_numpy()
                lo, hi, p = _boot(d)
                print(f"  k={kappa:.0f} {cen:>11} {c}-A {metric:>9}: "
                      f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
                if metric == "jf_mean" and c in ("B", "C100", "C400", "D"):
                    second.append((kappa, cen, c, p))
    print("\nHolm-adjusted secondary p-values (reference-leaf recovery, vs A):")
    for (kappa, cen) in sorted({(k, c) for k, c, _, _ in second}):
        ps = [(c, p) for k, ci, c, p in second if (k, ci) == (kappa, cen)]
        ps.sort(key=lambda t: t[1])
        m = len(ps)
        adj, running = [], 0.0
        for r, (c, p) in enumerate(ps):
            val = min(1.0, (m - r) * p)
            running = max(running, val)
            adj.append((c, p, running))
        print(f"  k={kappa:.0f} {cen:>11}: " +
              ", ".join(f"{c} p={p:.3f} adj={pa:.3f}" for c, p, pa in adj))

    primary = res[(res.kappa == 3.0) & (res.censoring == "marginal")]
    a = primary[primary.config == "A"].iloc[0]
    c = primary[primary.config == "C200"].iloc[0]
    print(f"\nPrimary contrast C200 vs A (joint cell):")
    print(f"  J_full {c.jf_mean:.3f} vs {a.jf_mean:.3f}; "
          f"J_true {c.jt_mean:.3f} vs {a.jt_mean:.3f}; "
          f"capture {c.capture_full:.3f} vs {a.capture_full:.3f}")
    print("Interpretation: J_full up but J_true not up -> reproducible but not "
          "more correct; both up -> capture and reproducibility improved.")
    return df


if __name__ == "__main__":
    main()

"""Gate 21 - high-leaf selection under the winner's curse.

Pre-registration: `results/gate21_PREREGISTRATION.md`.

Only the high-leaf selection rule changes. Data roles, commitment, and metrics
are frozen from Gate 20.

    A  max discovery mean psi
    B  max lower confidence bound (psi_mean - 1.645 * SE)
    C  max mean among leaves with discovery support >= 200
    D  max (selection frequency * mean), frequency from 20 discovery halves

Stability outcome uses a disjoint set of 20 discovery halves (seed rep+21), so
Arm D is not scored on the subsamples that selected it.

Usage
-----
    python -m ciscarl2.gate21 --reps 2 --start 240900   # smoke, off-seed
    python -m ciscarl2.gate21 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import pseudo_outcome
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate8 import best_jaccard
from .gate11 import _cart
from .gate14 import _calibrate, _config, _region, _simulate
from .gate19 import _assign_effs, _leaf_table, _phi_marginal
from .gate20 import _capped_G, _phi
from .rules import eval_rule

from .paths import data_path

DATA = data_path()
START = 240100
CELLS = ((0.0, "conditional"), (3.0, "marginal"))
ARMS = ("A", "B", "C", "D")
SUPPORT_MIN = 200
Z_LCB = 1.645
STAB_B = 20
STAB_FRAC = 0.5
JAC_THR = 0.8


def _leaf_stats_discovery(psi, mask):
    n = int(mask.sum())
    if n < 2:
        return np.nan, np.nan
    m = float(psi[mask].mean())
    se = float(psi[mask].std(ddof=1) / np.sqrt(n))
    return m, se


def _stability_freq(leaves, psi, XD, seed, B=STAB_B, frac=STAB_FRAC):
    rng = np.random.RandomState(seed)
    freq = np.zeros(len(leaves))
    for _ in range(B):
        idx = rng.choice(len(XD), int(len(XD) * frac), replace=False)
        sub = _cart(psi[idx], XD[idx], 3)[1]
        if not sub:
            continue
        j = max(range(len(sub)), key=lambda t: float(psi[idx][sub[t]["mask"]].mean()))
        m = sub[j]["mask"]
        for k, r in enumerate(leaves):
            mk = r["mask"][idx]
            u = (m & mk).sum()
            if u / max((m | mk).sum(), 1) > JAC_THR:
                freq[k] += 1
    return freq / B


def _eval_stability(cond, psi, XD, seed, B=STAB_B, frac=STAB_FRAC):
    rng = np.random.RandomState(seed)
    ref = eval_rule(cond, XD)
    hits = 0
    for _ in range(B):
        idx = rng.choice(len(XD), int(len(XD) * frac), replace=False)
        sub = _cart(psi[idx], XD[idx], 3)[1]
        if not sub:
            continue
        j = max(range(len(sub)), key=lambda t: float(psi[idx][sub[t]["mask"]].mean()))
        m = sub[j]["mask"]
        mr = ref[idx]
        u = (m & mr).sum()
        if u / max((m | mr).sum(), 1) > JAC_THR:
            hits += 1
    return hits / B


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
    XV, AV, TV, EV = X[val], A[val], T[val], E[val]
    XA, AA, TA, EA = X[asm], A[asm], T[asm], E[asm]
    regA, truth = R[asm], tau_i[asm]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    G = _capped_G(muD, GD, Gmarg, fid, fcaps, cen, TD)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
    _, _, q1, q0 = aipw(XD, AD, TD, TSTAR, muD, G)
    const = float((q1 - q0).mean())
    const_mae = float(np.abs(const - truth).mean())

    leaves = _cart(psi, XD, 3)[1]
    means = np.array([_leaf_stats_discovery(psi, r["mask"])[0] for r in leaves])
    ses = np.array([_leaf_stats_discovery(psi, r["mask"])[1] for r in leaves])
    supports = np.array([int(r["mask"].sum()) for r in leaves])

    jA = int(np.argmax(means))
    jB = int(np.argmax(means - Z_LCB * ses))
    okC = supports >= SUPPORT_MIN
    jC = int(np.argmax(np.where(okC, means, -np.inf))) if okC.any() else jA
    freq = _stability_freq(leaves, psi, XD, seed + 11)
    scoreD = freq * means
    jD = int(np.argmax(scoreD)) if np.isfinite(scoreD).any() else jA
    picks = {"A": jA, "B": jB, "C": jC, "D": jD}

    p1v, p0v = _phi(XD, AD, TD, ED, XV, AV, TV, cen, seed)
    dphi_v = p1v - p0v
    p1a, p0a = _phi(XD, AD, TD, ED, XA, AA, TA, cen, seed)
    dphi_a = p1a - p0a
    tbl_v = _leaf_table(leaves, XV, dphi_v)
    effs_v = np.array([t["eff"] for t in tbl_v], float)
    ses_v = np.array([t["se"] for t in tbl_v], float)
    effs_v = np.where(np.isfinite(effs_v), effs_v, const)
    masks_a = [eval_rule(r["cond"], XA) for r in leaves]

    rows = []
    for arm, j in picks.items():
        cond = leaves[j]["cond"]
        hi_mask = masks_a[j]
        k = int(hi_mask.sum())
        hi_eff, hi_se = effs_v[j], ses_v[j]
        committed = bool(np.isfinite(hi_eff) and np.isfinite(hi_se) and
                         hi_eff > float(norm.ppf(0.95)) * hi_se)
        use = effs_v.copy()
        if not committed:
            use[j] = const
        tbl_a = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                      eff=use[t], se=ses_v[t]) for t in range(len(leaves))]
        pred = _assign_effs(tbl_a, XA, use, const)
        mae = float(np.abs(pred - truth).mean())

        cap_un = float(truth[hi_mask].sum() / truth.sum()) if truth.sum() > 0 else np.nan
        purity = float((hi_mask & regA).sum() / k) if k else np.nan
        capture = float((hi_mask & regA).sum() / max(regA.sum(), 1))
        jac = best_jaccard([leaves[j]], XA, regA)
        stab = _eval_stability(cond, psi, XD, seed + 21)
        per = float(truth[hi_mask].mean()) if (committed and k) else np.nan
        cov = float(hi_mask.mean()) if committed else 0.0
        cap_c = (float(truth[hi_mask].sum() / truth.sum())
                 if (committed and truth.sum() > 0) else 0.0)
        oracle = (float(np.sort(truth)[::-1][:k].mean())
                  if (committed and k) else np.nan)
        eff_ratio = per / oracle if (committed and oracle) else np.nan
        rows.append(dict(
            rep=rep, n=n, kappa=kappa, censoring=cen, arm=arm,
            support=int(supports[j]), n_discovery=int(len(XD)),
            committed=committed, mae=mae, const_mae=const_mae,
            delta=mae - const_mae,
            captured_uncommit=cap_un, purity=purity, capture=capture,
            jaccard=jac, stability=stab,
            coverage=cov, per_treated=per, oracle=oracle,
            efficiency=eff_ratio, captured_commit=cap_c,
            null=bool(committed and np.isfinite(per) and per <= 0.01),
            beneficial=bool(committed and np.isfinite(per) and per > 0),
            hi_share=float(hi_mask.mean())))
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
    for (kappa, cen, arm), g in df.groupby(["kappa", "censoring", "arm"]):
        tr = g[g.committed]
        out.append(dict(
            kappa=kappa, censoring=cen, arm=arm, reps=len(g),
            support=g.support.mean(),
            captured_uncommit=g.captured_uncommit.mean(),
            purity=g.purity.mean(), capture=g.capture.mean(),
            jaccard=g.jaccard.mean(), stability=g.stability.mean(),
            mae=g.mae.mean(), const_mae=g.const_mae.mean(),
            delta=g.delta.mean(),
            commit_rate=100.0 * g.committed.mean(), coverage=g.coverage.mean(),
            per_treated=tr.per_treated.mean() if len(tr) else np.nan,
            efficiency=tr.efficiency.mean() if len(tr) else np.nan,
            captured_commit=g.captured_commit.mean(),
            null_rate=100.0 * tr.null.mean() if len(tr) else np.nan,
            beneficial_rate=100.0 * tr.beneficial.mean() if len(tr) else np.nan))
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
    df.to_csv(os.path.join(root, "results", "gate21_selection.csv"), index=False)
    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate21_summary.csv"), index=False)

    print(f"\n=== Gate 21, {args.reps} reps per cell ===")
    print(f"{'kappa':>6}{'cens':>12}{'arm':>4}{'support':>8}{'cap_un':>8}"
          f"{'purity':>7}{'capture':>8}{'jacc':>6}{'stab':>6}{'commit':>7}"
          f"{'cover':>7}{'effic':>7}{'cap_c':>7}{'null%':>6}{'MAE':>8}")
    for _, r in res.sort_values(["kappa", "censoring", "arm"]).iterrows():
        print(f"{r.kappa:>6.0f}{r.censoring:>12}{r.arm:>4}{r.support:>8.0f}"
              f"{r.captured_uncommit:>8.3f}{r.purity:>7.2f}{r.capture:>8.2f}"
              f"{r.jaccard:>6.2f}{r.stability:>6.2f}{r.commit_rate:>6.0f}%"
              f"{r.coverage:>7.3f}{r.efficiency:>7.2f}{r.captured_commit:>7.3f}"
              f"{r.null_rate:>6.0f}{r.mae:>8.4f}")

    print("\nPaired contrasts vs A (selection outcomes):")
    second = []
    for (kappa, cen), g in df.groupby(["kappa", "censoring"]):
        a = g[g.arm == "A"].set_index("rep")
        for arm in ("B", "C", "D"):
            b = g[g.arm == arm].set_index("rep")
            common = sorted(set(a.index) & set(b.index))
            for metric in ("captured_uncommit", "purity", "stability", "mae"):
                d = (b.loc[common, metric] - a.loc[common, metric]).to_numpy()
                lo, hi, p = _boot(d)
                print(f"  k={kappa:.0f} {cen:>11} {arm}-A {metric:>18}: "
                      f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
                if metric == "captured_uncommit" and arm in ("B", "D"):
                    second.append((kappa, cen, arm, p))
    # Holm over the two secondary captured-benefit contrasts, per cell
    print("\nHolm-adjusted secondary p-values (captured benefit, B/D vs A):")
    for (kappa, cen) in sorted({(k, c) for k, c, _, _ in second}):
        ps = [(arm, p) for k, c, arm, p in second if (k, c) == (kappa, cen)]
        ps.sort(key=lambda t: t[1])
        m = len(ps)
        adj = []
        running = 0.0
        for r, (arm, p) in enumerate(ps):
            val = min(1.0, (m - r) * p)
            running = max(running, val)
            adj.append((arm, p, running))
        print(f"  k={kappa:.0f} {cen:>11}: " +
              ", ".join(f"{a} p={p:.3f} adj={pa:.3f}" for a, p, pa in adj))

    joint = res[(res.kappa == 3.0) & (res.censoring == "marginal")]
    a_ref = joint[joint.arm == "A"].iloc[0]
    print("\nPre-registered decision (joint cell):")
    adopted = []
    for arm in ("B", "C", "D"):
        r = joint[joint.arm == arm].iloc[0]
        checks = {
            "capture >= A+0.03 or stability >= A+0.10": bool(
                r.captured_uncommit >= a_ref.captured_uncommit + 0.03 or
                r.stability >= a_ref.stability + 0.10),
            "purity >= A-0.05": bool(r.purity >= a_ref.purity - 0.05),
            "stability >= A-0.05": bool(r.stability >= a_ref.stability - 0.05),
            "MAE delta < 0": bool(r.delta < 0),
            "MAE <= A+0.005": bool(r.mae <= a_ref.mae + 0.005),
            "null rate <= 5%": bool(r.null_rate <= 5.0),
            "efficiency >= 0.70": bool(r.efficiency >= 0.70),
        }
        print(f"  arm {arm}:")
        for k, v in checks.items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if all(checks.values()):
            adopted.append(arm)
    print(f"\n-> {'CANDIDATE: ' + ', '.join(adopted) if adopted else 'no selection rule adopted'}")
    print("Primary confirmatory contrast is C vs A; B and D are secondary.")
    return df


if __name__ == "__main__":
    main()

"""Gate 20 - honest commitment validation with disjoint data roles.

Pre-registration: `results/gate20_PREREGISTRATION.md`.

Partition (n=8000): discovery 4000 / validation 2000 / assessment 2000.
Nuisances are fitted on discovery only. The validation set estimates leaf
effects and the commitment decision; the assessment set is used only to score.

Two nested validation sizes share the same discovery and assessment sets:
    half = random 1000 of the validation pool
    full = all 2000

Arms: A no commitment (always treat the high leaf); B honest commitment
(treat only if tau_L > 1.645 * SE_L, otherwise no treatment).

Usage
-----
    python -m ciscarl2.gate20 --reps 2 --start 230900   # smoke, off-seed
    python -m ciscarl2.gate20 --reps 40
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
from .gate11 import _cart, _split_stability
from .gate14 import _calibrate, _config, _region, _simulate
from .gate8 import best_jaccard
from .gate19 import _assign_effs, _leaf_table, _phi_marginal
from .rules import eval_rule

from .paths import data_path

DATA = data_path()
START = 230100
CELLS = ((0.0, "conditional"), (3.0, "marginal"))
VSIZES = (("half", 1000), ("full", 2000))


def _capped_G(muD, GD, Gmarg, fid, fcaps, cen, Tobs):
    if cen == "conditional":
        return truncate_weights(GD, fid, fcaps)[0]
    caps = {}
    for k in np.unique(fid):
        idx = np.where(fid == k)[0]
        ar = idx[Tobs[idx] > TSTAR]
        vals = 1.0 / Gmarg[ar] if ar.size else np.array([1.0])
        caps[int(k)] = {0.99: float(np.quantile(vals, 0.99))
                        if vals.size > 5 else np.inf}
    return truncate_weights(Gmarg, fid, caps)[0]


def _phi(XD, AD, TD, ED, XH, AH, TH, cen, seed):
    if cen == "conditional":
        return phi_on_holdout(XD, AD, TD, ED, XH, AH, TH, TSTAR, seed=seed)
    return _phi_marginal(XD, AD, TD, ED, XH, AH, TH)


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
    j_hi = max(range(len(leaves)),
               key=lambda j: float(psi[leaves[j]["mask"]].mean()))
    hi_cond = leaves[j_hi]["cond"]

    p1v, p0v = _phi(XD, AD, TD, ED, XV, AV, TV, cen, seed)
    dphi_v = p1v - p0v
    p1a, p0a = _phi(XD, AD, TD, ED, XA, AA, TA, cen, seed)
    dphi_a = p1a - p0a

    masks_a = [eval_rule(r["cond"], XA) for r in leaves]
    hi_mask_a = masks_a[j_hi]
    k_hi = int(hi_mask_a.sum())
    purity = (float((hi_mask_a & regA).sum() / k_hi) if k_hi else np.nan)
    capture = (float((hi_mask_a & regA).sum() / max(regA.sum(), 1)))
    jac = best_jaccard(leaves, XA, regA)
    stab = _split_stability(psi, XD, XA, hi_cond, 3, seed + 7)

    rng_half = np.random.RandomState(seed + 2)
    half_idx = rng_half.choice(len(val), len(val) // 2, replace=False)

    rows = []
    for vname, vidx in (("half", half_idx), ("full", np.arange(len(val)))):
        tbl_v = _leaf_table(leaves, XV[vidx], dphi_v[vidx])
        effs = np.array([t["eff"] for t in tbl_v], float)
        ses = np.array([t["se"] for t in tbl_v], float)
        effs = np.where(np.isfinite(effs), effs, const)
        hi_eff, hi_se = effs[j_hi], ses[j_hi]
        committed = bool(np.isfinite(hi_eff) and np.isfinite(hi_se) and
                         hi_eff > float(norm.ppf(0.95)) * hi_se)

        for arm in ("A", "B"):
            use = effs.copy()
            if arm == "B" and not committed:
                use[j_hi] = const
            tbl_a = [dict(cond=leaves[j]["cond"], mask=masks_a[j],
                          eff=use[j], se=ses[j]) for j in range(len(leaves))]
            pred = _assign_effs(tbl_a, XA, use, const)
            mae = float(np.abs(pred - truth).mean())
            treated = bool(arm == "A" or committed)
            cov = float(hi_mask_a.mean()) if treated else 0.0
            per = float(truth[hi_mask_a].mean()) if (treated and k_hi) else np.nan
            cap = (float(truth[hi_mask_a].sum() / truth.sum())
                   if (treated and truth.sum() > 0) else 0.0)
            oracle = (float(np.sort(truth)[::-1][:k_hi].mean())
                      if (treated and k_hi) else np.nan)
            eff_ratio = per / oracle if (treated and oracle) else np.nan
            rows.append(dict(
                rep=rep, n=n, kappa=kappa, censoring=cen, vsize=vname,
                arm=arm, committed=bool(committed) if arm == "B" else True,
                n_val=len(vidx), n_assess=len(asm),
                mae=mae, const_mae=const_mae, delta=mae - const_mae,
                coverage=cov, per_treated=per, oracle=oracle,
                efficiency=eff_ratio, captured=cap,
                null=bool(treated and np.isfinite(per) and per <= 0.01),
                beneficial=bool(treated and np.isfinite(per) and per > 0),
                purity=purity, capture=capture, jaccard=jac, stability=stab,
                hi_share=float(hi_mask_a.mean())))
    return rows


def _boot(d, seed=0, B=5000):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for (kappa, cen, vsize, arm), g in df.groupby(
            ["kappa", "censoring", "vsize", "arm"]):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        tr = g[g.committed]
        out.append(dict(
            kappa=kappa, censoring=cen, vsize=vsize, arm=arm, reps=len(g),
            commit_rate=(100.0 * g.committed.mean() if arm == "B" else 100.0),
            mae=g.mae.mean(), const_mae=g.const_mae.mean(), delta=d.mean(),
            lo=lo, hi=hi, frac_better=100.0 * float((d < 0).mean()),
            coverage=g.coverage.mean(),
            per_treated=tr.per_treated.mean() if len(tr) else np.nan,
            efficiency=tr.efficiency.mean() if len(tr) else np.nan,
            captured=g.captured.mean(),
            null_rate=(100.0 * tr.null.mean() if len(tr) else np.nan),
            beneficial_rate=(100.0 * tr.beneficial.mean() if len(tr) else np.nan),
            purity=g.purity.mean(), capture=g.capture.mean(),
            jaccard=g.jaccard.mean(), stability=g.stability.mean(),
            n_val=g.n_val.mean(), n_assess=g.n_assess.mean()))
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
    df.to_csv(os.path.join(root, "results", "gate20_honest.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate20_summary.csv"), index=False)

    print(f"\n=== Gate 20, {args.reps} reps, n=8000 (4000/2000/2000) ===")
    print(f"{'kappa':>6}{'cens':>12}{'val':>6}{'arm':>4}{'commit%':>9}"
          f"{'cover':>7}{'per-tr':>8}{'effic':>7}{'capt':>7}{'null%':>7}"
          f"{'benef%':>8}{'MAE':>8}{'delta':>9}{'%bet':>6}")
    for _, r in res.sort_values(["kappa", "censoring", "vsize", "arm"]).iterrows():
        print(f"{r.kappa:>6.0f}{r.censoring:>12}{r.vsize:>6}{r.arm:>4}"
              f"{r.commit_rate:>8.0f}%{r.coverage:>7.3f}"
              f"{r.per_treated:>+8.3f}{r.efficiency:>7.2f}{r.captured:>7.3f}"
              f"{r.null_rate:>7.0f}{r.beneficial_rate:>8.0f}"
              f"{r.mae:>8.4f}{r.delta:>+9.4f}{r.frac_better:>5.0f}%")

    print("\nPaired A vs B (negative captured diff = B captures less):")
    for (k, c, v), g in df.groupby(["kappa", "censoring", "vsize"]):
        a = g[g.arm == "A"].set_index("rep")
        b = g[g.arm == "B"].set_index("rep")
        common = sorted(set(a.index) & set(b.index))
        dn = (a.loc[common, "null"].astype(float) -
              b.loc[common, "null"].astype(float)).to_numpy()
        dc = (a.loc[common, "captured"] - b.loc[common, "captured"]).to_numpy()
        dm = (b.loc[common, "mae"] - a.loc[common, "mae"]).to_numpy()
        nlo, nhi = _boot(dn)
        clo, chi = _boot(dc)
        mlo, mhi = _boot(dm)
        print(f"  k={k:.0f} {c:>11} {v:>4}: nullA-nullB {dn.mean():+.3f} "
              f"[{nlo:+.3f},{nhi:+.3f}] | capA-capB {dc.mean():+.3f} "
              f"[{clo:+.3f},{chi:+.3f}] | MAE B-A {dm.mean():+.4f} "
              f"[{mlo:+.4f},{mhi:+.4f}]")

    joint = res[(res.kappa == 3.0) & (res.censoring == "marginal") &
                (res.vsize == "full") & (res.arm == "B")].iloc[0]
    a_ref = res[(res.kappa == 3.0) & (res.censoring == "marginal") &
                (res.vsize == "full") & (res.arm == "A")].iloc[0]
    checks = {
        "safety: null rate <= 5%": bool(joint.null_rate <= 5.0),
        "utility: efficiency >= 0.70": bool(joint.efficiency >= 0.70),
        "utility: captured >= 0.20": bool(joint.captured >= 0.20),
        "captured >= 0.70 x A": bool(joint.captured >= 0.70 * a_ref.captured),
        "prediction: MAE delta < 0, CI < 0": bool(joint.delta < 0 and joint.hi < 0),
    }
    print("\nPre-registered checks (joint cell, V-full, arm B):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n-> {'HONEST COMMITMENT ADOPTED' if all(checks.values()) else 'NOT adopted; see failing checks'}")
    return df


if __name__ == "__main__":
    main()

"""Gate 15 - can a standardised plugin + AIPW blend rescue a collapsed target?

Pre-registration: `results/gate15_PREREGISTRATION.md` (frozen before this ran).

    z_plugin = TRAIN-standardised plug-in target
    z_psi    = TRAIN-standardised AIPW pseudo-outcome
    blend_l  = l * z_plugin + (1 - l) * z_psi

Primary: kappa = 3 (Gate 14's severe violation), l = 0.5, 40 fresh seeds.
Controls: l = 1 (plugin) and l = 0 (psi) on the identical seeds, plus
kappa = 0 and kappa = 1 as descriptive cells.

Everything downstream is unchanged: depth-3 CART, held-out AIPW effects,
TRAIN marginal ATE, four pre-registered checks.

Usage
-----
    python -m ciscarl2.gate15 --reps 2 --start 170900   # smoke, off-seed
    python -m ciscarl2.gate15 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import pseudo_outcome
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate8 import assign, best_jaccard
from .gate11 import _cart, _hi_leaf, _leaf_stats, _split_stability
from .gate14 import _calibrate, _config, _region, _simulate

from .paths import data_path

DATA = data_path()
LAMS = (0.0, 0.5, 1.0)
CELLS = ((3.0, 170100), (1.0, 170300), (0.0, 170200))
PRIMARY = (3.0, 0.5)


def _standardise(v):
    sd = float(np.std(v))
    if sd <= 0:
        return np.zeros_like(v)
    return (v - float(np.mean(v))) / sd


def run_rep(rep, kappa, start, cfg, lam_c0, lams=LAMS):
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

    tr, te = train_test_split(np.arange(n), test_size=0.30,
                              random_state=seed, stratify=A)
    XD, XE = X[tr], X[te]
    regE, truth = R[te], tau_i[te]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=seed, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    _, _, q1, q0 = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const = float((q1 - q0).mean())
    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=seed)
    dphi = p1 - p0

    plug = muD[:, 1] - muD[:, 0]
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    zp = _standardise(plug)
    zs = _standardise(psi)
    regD = R[tr]

    rows = []
    for lam in lams:
        target = lam * zp + (1.0 - lam) * zs
        _, leaves = _cart(target, XD, 3)
        hat = assign(leaves, XE, dphi, const)
        hi = _hi_leaf(leaves, target, XD)
        st = _leaf_stats(hi["cond"], XE, regE, dphi) if hi else {}
        mae = float(np.abs(hat - truth).mean())
        cm = float(np.abs(const - truth).mean())
        rows.append(dict(
            rep=rep, n=n, kappa=kappa, lam=lam,
            arm=f"k{int(kappa)}_l{lam:.1f}", rules=len(leaves),
            mae=mae, const_mae=cm, delta=mae - cm,
            target_sd=float(np.std(target)),
            target_corr=float(np.corrcoef(target, tau_i[tr])[0, 1]),
            target_contrast=float(target[regD].mean() - target[~regD].mean()),
            true_contrast=float(tau_i[tr][regD].mean() - tau_i[tr][~regD].mean()),
            const_ate=const, true_ate=float(tau_i[tr].mean()),
            best_jaccard=best_jaccard(leaves, XE, regE),
            hi_share=st.get("share", np.nan),
            hi_purity=st.get("purity", np.nan),
            hi_capture=st.get("capture", np.nan),
            hi_effect=st.get("effect", np.nan),
            hi_se=st.get("se", np.nan),
            stability=_split_stability(target, XD, XE, hi["cond"], 3,
                                       seed + 7) if hi else np.nan,
            region_share=100.0 * float(regE.mean())))
    return rows


def _boot(d, seed=0, B=5000):
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for (kappa, lam), g in df.groupby(["kappa", "lam"]):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        # paired against the l=1 plug-in on the same replications
        base = df[(df.kappa == kappa) & (df.lam == 1.0)].set_index("rep")
        gg = g.set_index("rep")
        pair = sorted(set(base.index) & set(gg.index))
        db = (gg.loc[pair, "mae"] - base.loc[pair, "mae"]).to_numpy()
        blo, bhi = _boot(db)
        out.append(dict(
            kappa=kappa, lam=lam, arm=g.arm.iloc[0], reps=len(g),
            rules=g.rules.mean(), mae=g.mae.mean(),
            const_mae=g.const_mae.mean(), delta=d.mean(), lo=lo, hi=hi,
            frac_better=100.0 * float((d < 0).mean()),
            rel_gain=100.0 * (1.0 - g.mae.mean() / g.const_mae.mean()),
            target_sd=g.target_sd.mean(), target_corr=g.target_corr.mean(),
            target_contrast=g.target_contrast.mean(),
            true_contrast=g.true_contrast.mean(),
            const_ate=g.const_ate.mean(), true_ate=g.true_ate.mean(),
            best_jaccard=g.best_jaccard.mean(), hi_share=g.hi_share.mean(),
            hi_purity=g.hi_purity.mean(), hi_capture=g.hi_capture.mean(),
            hi_effect=g.hi_effect.mean(), hi_se=g.hi_se.mean(),
            stability=g.stability.mean(),
            delta_vs_plug=db.mean(), dvp_lo=blo, dvp_hi=bhi,
            frac_vs_plug=100.0 * float((db < 0).mean())))
    return pd.DataFrame(out)


def decide(res):
    r = res[(res.kappa == PRIMARY[0]) & (res.lam == PRIMARY[1])].iloc[0]
    return {
        "mean delta < 0": bool(r.delta < 0),
        "95% CI entirely below 0": bool(r.hi < 0),
        "Jaccard >= 0.107": bool(r.best_jaccard >= 0.107),
        ">= 70% of reps better": bool(r.frac_better >= 70),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    cells = CELLS if args.start is None else \
        tuple((k, args.start) for k, _ in CELLS)
    lam_c = {k: _calibrate(cfg, k) for k, _ in cells}
    for k, _ in cells:
        print(f"  kappa={k}: lam_c0={lam_c[k]:.3e}", flush=True)

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, k, s, cfg, lam_c[k])
        for k, s in cells for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate15_blend.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate15_summary.csv"), index=False)
    checks = decide(res)

    print(f"\n=== Gate 15, n=8000, {args.reps} reps per cell ===")
    print(f"{'kappa':>6}{'lambda':>8}{'MAE':>9}{'const':>9}{'delta':>10}"
          f"{'95% CI':>21}{'%bet':>6}{'gain':>7}{'d vs plug':>11}"
          f"{'corr':>7}{'purity':>8}{'capt':>7}{'jacc':>7}")
    for _, r in res.sort_values(["kappa", "lam"]).iterrows():
        print(f"{r.kappa:>6.0f}{r.lam:>8.1f}{r.mae:>9.4f}{r.const_mae:>9.4f}"
              f"{r.delta:>+10.4f}   [{r.lo:+.4f},{r.hi:+.4f}]"
              f"{r.frac_better:>5.0f}%{r.rel_gain:>+6.1f}%"
              f"{r.delta_vs_plug:>+11.4f}{r.target_corr:>7.3f}"
              f"{r.hi_purity:>8.2f}{r.hi_capture:>7.2f}"
              f"{r.best_jaccard:>7.2f}")

    print(f"\nPre-registered decision checks (primary: kappa=3, lambda=0.5):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n-> {'BLEND RESCUES the collapsed target' if all(checks.values()) else 'BLEND DOES NOT rescue; report the boundary'}")
    return df


if __name__ == "__main__":
    main()

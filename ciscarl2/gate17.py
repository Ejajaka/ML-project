"""Gate 17 - the lower-n boundary of the v1.6 model (psi + depth-3 CART).

Pre-registration: `results/gate17_PREREGISTRATION.md` (frozen before this ran).

n in {2000, 4000, 6000}, kappa = 0, 40 fresh seeds per cell. Primary target psi;
comparators: the plug-in on the same seeds, and an oracle (true membership,
effects from held-out AIPW) that separates discovery failure from estimation
noise.

Usage
-----
    python -m ciscarl2.gate17 --reps 2 --start 190900   # smoke, off-seed
    python -m ciscarl2.gate17 --reps 40
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
from .gate16 import _boot

from .paths import data_path

DATA = data_path()
CELLS = ((2000, 0.0, 190100), (4000, 0.0, 190200), (6000, 0.0, 190300))


def run_rep(rep, n, kappa, start, cfg, lam_c0):
    pool, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, lam0 = cfg
    seed = start + rep
    rng = np.random.RandomState(seed)
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
    const_mae = float(np.abs(const - truth).mean())

    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    plug = muD[:, 1] - muD[:, 0]
    regD = R[tr]

    rows = [dict(rep=rep, n=n, kappa=kappa, target="constant", rules=0,
                 mae=const_mae, const_mae=const_mae, delta=0.0,
                 true_contrast=float(tau_i[tr][regD].mean() -
                                     tau_i[tr][~regD].mean()))]

    for name, target in (("psi", psi), ("plugin", plug)):
        _, leaves = _cart(target, XD, 3)
        hat = assign(leaves, XE, dphi, const)
        hi = _hi_leaf(leaves, target, XD)
        st = _leaf_stats(hi["cond"], XE, regE, dphi) if hi else {}
        mae = float(np.abs(hat - truth).mean())
        rows.append(dict(
            rep=rep, n=n, kappa=kappa, target=name, rules=len(leaves),
            mae=mae, const_mae=const_mae, delta=mae - const_mae,
            target_sd=float(np.std(target)),
            target_corr=float(np.corrcoef(target, tau_i[tr])[0, 1]),
            best_jaccard=best_jaccard(leaves, XE, regE),
            hi_purity=st.get("purity", np.nan),
            hi_capture=st.get("capture", np.nan),
            hi_effect=st.get("effect", np.nan), hi_se=st.get("se", np.nan),
            stability=_split_stability(target, XD, XE, hi["cond"], 3,
                                       seed + 7) if hi else np.nan,
            region_share=100.0 * float(regE.mean())))

    # oracle: true membership, effects estimated from held-out phi
    tin = float(dphi[regE].mean()) if regE.any() else const
    tout = float(dphi[~regE].mean()) if (~regE).any() else const
    orc = float(np.abs(np.where(regE, tin, tout) - truth).mean())
    rows.append(dict(rep=rep, n=n, kappa=kappa, target="oracle", rules=2,
                     mae=orc, const_mae=const_mae, delta=orc - const_mae,
                     true_contrast=float(tau_i[tr][regD].mean() -
                                         tau_i[tr][~regD].mean())))
    return rows


def summarise(df):
    out = []
    for (n, target), g in df.groupby(["n", "target"]):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)

        def m(c):
            return float(g[c].mean()) if c in g.columns else np.nan

        out.append(dict(
            n=n, target=target, reps=len(g), rules=float(g.rules.mean()),
            mae=float(g.mae.mean()), const_mae=float(g.const_mae.mean()),
            delta=float(d.mean()), lo=lo, hi=hi,
            frac_better=100.0 * float((d < 0).mean()),
            rel_gain=100.0 * (1.0 - g.mae.mean() / g.const_mae.mean()),
            target_corr=m("target_corr"), best_jaccard=m("best_jaccard"),
            hi_purity=m("hi_purity"), hi_capture=m("hi_capture"),
            stability=m("stability")))
    return pd.DataFrame(out)


def decide(res):
    out = {}
    for n, g in res[res.target == "psi"].groupby("n"):
        r = g.iloc[0]
        out[int(n)] = {
            "mean delta < 0": bool(r.delta < 0),
            "95% CI entirely below 0": bool(r.hi < 0),
            "Jaccard >= 0.107": bool(r.best_jaccard >= 0.107),
            ">= 70% of reps better": bool(r.frac_better >= 70),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    cells = CELLS if args.start is None else \
        tuple((n, k, args.start) for n, k, _ in CELLS)
    lam_c0 = _calibrate(cfg, 0.0)
    print(f"  lam_c0={lam_c0:.3e}", flush=True)

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, n, k, s, cfg, lam_c0)
        for n, k, s in cells for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate17_nsweep.csv"), index=False)

    res = summarise(df[df.target != "constant"])
    res.to_csv(os.path.join(root, "results", "gate17_summary.csv"), index=False)

    print(f"\n=== Gate 17, {args.reps} reps per cell, kappa=0 ===")
    print(f"{'n':>7}{'target':>9}{'rules':>7}{'MAE':>9}{'const':>9}{'delta':>10}"
          f"{'95% CI':>21}{'%bet':>6}{'gain':>7}{'corr':>7}{'jacc':>7}"
          f"{'purity':>8}{'capt':>7}{'stab':>7}")
    order = {"psi": 0, "plugin": 1, "oracle": 2}
    res["order"] = res.target.map(order)
    for _, r in res.sort_values(["n", "order"]).iterrows():
        print(f"{r.n:>7.0f}{r.target:>9}{r.rules:>7.1f}{r.mae:>9.4f}"
              f"{r.const_mae:>9.4f}{r.delta:>+10.4f}"
              f"   [{r.lo:+.4f},{r.hi:+.4f}]{r.frac_better:>5.0f}%"
              f"{r.rel_gain:>+6.1f}%{r.target_corr:>7.3f}"
              f"{r.best_jaccard:>7.2f}{r.hi_purity:>8.2f}"
              f"{r.hi_capture:>7.2f}{r.stability:>7.2f}")

    print("\nPre-registered decision checks (psi):")
    checks = decide(res)
    for n, ch in checks.items():
        print(f"  n={n}:")
        for k, v in ch.items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    ok = [n for n, ch in checks.items() if all(ch.values())]
    print(f"\n-> psi passes at n = {ok if ok else 'NONE'}")
    print("Oracle row is the ceiling: if the oracle also loses, the barrier is "
          "estimation; if it wins while psi loses, the barrier is discovery.")
    return df


if __name__ == "__main__":
    main()

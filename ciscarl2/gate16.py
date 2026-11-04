"""Gate 16 - confirm the AIPW discovery target at n = 8000 / 12000.

Pre-registration: `results/gate16_PREREGISTRATION.md` (frozen before this ran).

Cells: n in {8000, 12000} x kappa in {0 (correct), 3 (Gate 14 severe violation)}.
Targets compared on identical replications: psi (primary) and the Cox plug-in
(comparator). Everything else is frozen: depth-3 CART, held-out AIPW effects,
TRAIN marginal ATE, four pre-registered checks.

Usage
-----
    python -m ciscarl2.gate16 --reps 2 --start 180900   # smoke, off-seed
    python -m ciscarl2.gate16 --reps 40
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
CELLS = ((8000, 0.0, 180100), (12000, 0.0, 180200),
         (8000, 3.0, 180300), (12000, 3.0, 180400))


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

    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    plug = muD[:, 1] - muD[:, 0]
    regD = R[tr]

    rows = []
    for name, target in (("psi", psi), ("plugin", plug)):
        _, leaves = _cart(target, XD, 3)
        hat = assign(leaves, XE, dphi, const)
        hi = _hi_leaf(leaves, target, XD)
        st = _leaf_stats(hi["cond"], XE, regE, dphi) if hi else {}
        mae = float(np.abs(hat - truth).mean())
        cm = float(np.abs(const - truth).mean())
        rows.append(dict(
            rep=rep, n=n, kappa=kappa, target=name, rules=len(leaves),
            mae=mae, const_mae=cm, delta=mae - cm,
            target_sd=float(np.std(target)),
            target_corr=float(np.corrcoef(target, tau_i[tr])[0, 1]),
            target_contrast=float(target[regD].mean() - target[~regD].mean()),
            true_contrast=float(tau_i[tr][regD].mean() - tau_i[tr][~regD].mean()),
            const_ate=const, true_ate=float(tau_i[tr].mean()),
            cens_frac=float(np.mean((T[tr] <= TSTAR) & (E[tr] == 0))),
            event_frac=float(np.mean((T[tr] <= TSTAR) & (E[tr] == 1))),
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
    for (n, kappa, target), g in df.groupby(["n", "kappa", "target"]):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        out.append(dict(
            n=n, kappa=kappa, target=target, reps=len(g),
            rules=g.rules.mean(), mae=g.mae.mean(),
            const_mae=g.const_mae.mean(), delta=d.mean(), lo=lo, hi=hi,
            frac_better=100.0 * float((d < 0).mean()),
            rel_gain=100.0 * (1.0 - g.mae.mean() / g.const_mae.mean()),
            target_sd=g.target_sd.mean(), target_corr=g.target_corr.mean(),
            target_contrast=g.target_contrast.mean(),
            true_contrast=g.true_contrast.mean(),
            const_ate=g.const_ate.mean(), true_ate=g.true_ate.mean(),
            cens_frac=g.cens_frac.mean(), event_frac=g.event_frac.mean(),
            best_jaccard=g.best_jaccard.mean(), hi_share=g.hi_share.mean(),
            hi_purity=g.hi_purity.mean(), hi_capture=g.hi_capture.mean(),
            hi_effect=g.hi_effect.mean(), hi_se=g.hi_se.mean(),
            stability=g.stability.mean()))
    return pd.DataFrame(out)


def decide(res):
    checks = {}
    for (n, kappa), g in res[res.target == "psi"].groupby(["n", "kappa"]):
        r = g.iloc[0]
        checks[(int(n), float(kappa))] = {
            "mean delta < 0": bool(r.delta < 0),
            "95% CI entirely below 0": bool(r.hi < 0),
            "Jaccard >= 0.107": bool(r.best_jaccard >= 0.107),
            ">= 70% of reps better": bool(r.frac_better >= 70),
        }
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    cells = CELLS if args.start is None else \
        tuple((n, k, args.start) for n, k, _ in CELLS)
    lam_c = {k: _calibrate(cfg, k) for _, k, _ in cells}
    for k in sorted(lam_c):
        print(f"  kappa={k}: lam_c0={lam_c[k]:.3e}", flush=True)

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, n, k, s, cfg, lam_c[k])
        for n, k, s in cells for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate16_psi.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate16_summary.csv"), index=False)
    checks = decide(res)

    print(f"\n=== Gate 16, {args.reps} reps per cell ===")
    print(f"{'n':>7}{'kappa':>6}{'target':>8}{'rules':>7}{'MAE':>9}{'const':>9}"
          f"{'delta':>10}{'95% CI':>21}{'%bet':>6}{'gain':>7}"
          f"{'corr':>7}{'purity':>8}{'capt':>7}{'jacc':>7}{'stab':>7}")
    for _, r in res.sort_values(["n", "kappa", "target"]).iterrows():
        print(f"{r.n:>7.0f}{r.kappa:>6.0f}{r.target:>8}{r.rules:>7.1f}"
              f"{r.mae:>9.4f}{r.const_mae:>9.4f}{r.delta:>+10.4f}"
              f"   [{r.lo:+.4f},{r.hi:+.4f}]{r.frac_better:>5.0f}%"
              f"{r.rel_gain:>+6.1f}%{r.target_corr:>7.3f}"
              f"{r.hi_purity:>8.2f}{r.hi_capture:>7.2f}"
              f"{r.best_jaccard:>7.2f}{r.stability:>7.2f}")

    print("\nPaired psi vs plugin (negative favours psi):")
    for (n, kappa), g in df.groupby(["n", "kappa"]):
        p = g[g.target == "psi"].set_index("rep").mae
        q = g[g.target == "plugin"].set_index("rep").mae
        d = (p - q).dropna().to_numpy()
        lo, hi = _boot(d)
        print(f"  n={n:>5}, kappa={kappa:.0f}: {d.mean():+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]  {100*(d < 0).mean():.0f}% of reps")

    print("\nEvent-process diagnostics (Gate 15 anomaly):")
    for _, r in res[res.target == "psi"].iterrows():
        print(f"  n={r.n:>5}, kappa={r.kappa:.0f}: cens={r.cens_frac:.3f} "
              f"event={r.event_frac:.3f} ATE est={r.const_ate:+.3f} "
              f"true={r.true_ate:+.3f} tgt sd={r.target_sd:.3f}")

    print("\nPre-registered decision checks (psi):")
    ok = {}
    for (n, kappa), ch in checks.items():
        print(f"  n={n}, kappa={kappa:.0f}:")
        for k, v in ch.items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        ok[(n, kappa)] = all(ch.values())
    perf = all(ok[(n, 0.0)] for n in (8000, 12000))
    robust = all(ok[(n, 3.0)] for n in (8000, 12000))
    if perf and robust:
        print("\n-> ADOPT psi as v1.6 discovery target: performance at kappa=0 "
              "and robustness at kappa=3 both confirmed.")
    elif perf:
        print("\n-> psi adopted for performance only; the kappa=3 robustness "
              "claim is NOT confirmed.")
    else:
        print("\n-> psi NOT adopted; keep the v1.5 plug-in and record Gate 15 "
              "as seed-specific.")
    return df


if __name__ == "__main__":
    main()

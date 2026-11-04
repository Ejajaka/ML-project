"""Gate 13 - MNAR delta-tilt on the adopted depth-3 rule set.

Pre-registration: `results/gate13_PREREGISTRATION.md` (frozen before this ran).

The rule list is discovered once at delta = 0 and frozen. For each delta the
censoring weights are tilted by G ** exp(delta) (Gate 5 convention, same cap),
the rule effects and the constant are recomputed under that tilt, and the paired
MAE difference is reported.

    delta > 0  -> more MNAR-like censoring (heavier weights)
    delta = 0  -> primary analysis exactly

Robustness is declared only if the mean delta and its 95 % CI are below zero at
every delta in the grid, for each sample size.

Usage
-----
    python -m ciscarl2.gate13 --reps 2 --n 8000 --start 150900   # smoke, off-seed
    python -m ciscarl2.gate13 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (_fit_predict_nuisances, crossfit_nuisances,
                         ipcw_outcome, truncate_weights)
from .gate2 import TAU
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate8 import assign
from .gate11 import _cart, _hi_leaf
from .rules import eval_rule

from .paths import data_path

DATA = data_path()
CELLS = ((8000, 150100), (12000, 150200))
TILTS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
CAP_Q, CAP_CEIL = 0.99, 20.0


def _holdout(XD, AD, TD, ED, XE, seed):
    """TRAIN-fitted nuisances on TEST, with the v1.4 cap from an inner cross-fit."""
    _, G_in, _, _, _, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    w = np.where(TD > TSTAR, 1.0, 0.0) / G_in
    w = w[w > 0]
    cap = min(float(np.quantile(w, CAP_Q)), CAP_CEIL) if w.size else CAP_CEIL
    mu, G = _fit_predict_nuisances(XD, AD, TD, ED, XE, TSTAR)
    return mu, np.maximum(G, 1.0 / cap), cap


def run_rep(rep, n, start, cfg):
    (pool, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, lam0, lam_c0) = cfg
    seed = start + rep
    rng = np.random.RandomState(seed)
    X = pool[rng.choice(len(pool), n, replace=True)]
    reg = (X[:, i_cd4] < (cd4_cut - mu_[i_cd4]) / sd_[i_cd4]) & \
          (X[:, i_age] > (age_cut - mu_[i_age]) / sd_[i_age])
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    sim = dgp.generate(X, "covariate", lam_c0, lam0, rng, TSTAR,
                       effect_vec=np.where(reg, _delta_for(EFFECT, s0), 0.0))
    A, T, E = sim["A"], sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=seed, stratify=A)
    XD, XE = X[tr], X[te]
    AD, TD, ED = A[tr], T[tr], E[tr]
    AE, TE = A[te], T[te]
    truth = tau_i[te]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    tau_hat = muD[:, 1] - muD[:, 0]
    _, leaves = _cart(tau_hat, XD, 3)
    hi = _hi_leaf(leaves, tau_hat, XD)
    hi_cond = hi["cond"] if hi else None

    mu, G, cap = _holdout(XD, AD, TD, ED, XE, seed)
    true_ate = float(truth.mean())

    rows = []
    for tilt in TILTS:
        Gt = np.maximum(G ** np.exp(tilt), 1.0 / cap)
        Y = ipcw_outcome(TE, TSTAR, Gt)
        phi1 = mu[:, 1] + (AE == 1) / 0.5 * (Y - mu[:, 1])
        phi0 = mu[:, 0] + (AE == 0) / 0.5 * (Y - mu[:, 0])
        dphi = phi1 - phi0
        const = float(dphi.mean())
        hat = assign(leaves, XE, dphi, const)
        mae = float(np.abs(hat - truth).mean())
        cm = float(np.abs(const - truth).mean())
        hi_eff = np.nan
        if hi_cond is not None:
            m = eval_rule(hi_cond, XE)
            if m.any():
                hi_eff = float(dphi[m].mean())
        rows.append(dict(rep=rep, n=n, tilt=tilt, rules=len(leaves),
                         mae=mae, const_mae=cm, delta=mae - cm,
                         const_ate=const, true_ate=true_ate,
                         hi_effect=hi_eff, cap=cap))
    return rows


def _boot(d, seed=0, B=5000):
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for (n, tilt), g in df.groupby(["n", "tilt"]):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        out.append(dict(n=n, tilt=tilt, reps=len(g), rules=g.rules.mean(),
                        mae=g.mae.mean(), const_mae=g.const_mae.mean(),
                        delta=d.mean(), lo=lo, hi=hi,
                        frac_better=100.0 * float((d < 0).mean()),
                        const_ate=g.const_ate.mean(),
                        hi_effect=g.hi_effect.mean()))
    return pd.DataFrame(out)


def _config():
    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    cd4_cut, age_cut = float(raw["cd40"].median()), float(raw["age"].median())
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})
    return (pool, mu, sd, dgp.COVARIATES.index("cd40"),
            dgp.COVARIATES.index("age"), cd4_cut, age_cut, lam0, lam_c0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n", type=int, nargs="+", default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    cells = [c for c in CELLS if args.n is None or c[0] in args.n]
    if args.start is not None:
        cells = [(n, args.start) for n, _ in cells]

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, n, start, cfg)
        for n, start in cells for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate13_tilt.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate13_summary.csv"), index=False)

    print(f"\n=== Gate 13 - MNAR delta-tilt, depth-3 rule set, {args.reps} reps ===")
    for n in sorted(res.n.unique()):
        g = res[res.n == n]
        print(f"\nn = {int(n)}   (true marginal ATE = {g.const_ate.iloc[0]:.3f}"
              f" at tilt 0; hi-leaf effect shown for reference)")
        print(f"{'delta':>7}{'MAE':>9}{'const':>9}{'delta MAE':>11}"
              f"{'95% CI':>22}{'%bet':>6}{'const ATE':>11}{'hi effect':>11}")
        for _, r in g.iterrows():
            print(f"{r.tilt:>+7.1f}{r.mae:>9.4f}{r.const_mae:>9.4f}"
                  f"{r.delta:>+11.4f}   [{r.lo:+.4f},{r.hi:+.4f}]"
                  f"{r.frac_better:>5.0f}%{r.const_ate:>+11.3f}"
                  f"{r.hi_effect:>+11.3f}")

    print("\nPre-registered robustness check (all tilts must pass):")
    ok_all = True
    for n in sorted(res.n.unique()):
        bad = res[(res.n == n) & ~((res.delta < 0) & (res.hi < 0))]
        if bad.empty:
            print(f"  n={int(n)}: PASS - delta < 0 and CI < 0 at every tilt")
        else:
            ok_all = False
            print(f"  n={int(n)}: FAIL at tilts {list(bad.tilt.round(1))}"
                  f" (delta {list(bad.delta.round(4))}, hi {list(bad.hi.round(4))})")
    print(f"\n-> {'ROBUST across delta in [-2, 2]' if ok_all else 'NOT robust across the full grid; see failing tilts'}")
    return df


if __name__ == "__main__":
    main()

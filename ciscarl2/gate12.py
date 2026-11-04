"""Gate 12 - confirm the depth-3 rule model on fresh seeds.

Pre-registration: `results/gate12_PREREGISTRATION.md` (frozen before this ran).

    n in {8000, 12000}, 40 fresh seeds per cell
    primary   CART depth 3 on the cross-fitted Cox plug-in target
    effects   held-out AIPW phi
    decide    mean delta < 0, 95 % CI < 0, Jaccard >= 0.107, >= 70 % reps better

Depth 2 is reproduced on the same seeds as a descriptive check.

Usage
-----
    python -m ciscarl2.gate12 --reps 2 --n 8000 --start 130900   # smoke, off-seed
    python -m ciscarl2.gate12 --reps 40
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
from .gate2 import TAU
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate8 import assign, best_jaccard
from .gate11 import (_cart, _hi_leaf, _leaf_stats, _split_stability,
                     summarise)

from .paths import data_path

DATA = data_path()
CELLS = ((8000, 130100), (12000, 130200))
PRIMARY = "cart_d3"
BASELINE_JACCARD = 0.107


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
    regE, truth = reg[te], tau_i[te]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=seed, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    _, _, q1, q0 = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const = float((q1 - q0).mean())
    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=seed)
    dphi = p1 - p0
    tau_hat = muD[:, 1] - muD[:, 0]

    const_mae = float(np.abs(const - truth).mean())
    rows = [dict(rep=rep, n=n, model="constant", depth=0, rules=0,
                 mae=const_mae, const_mae=const_mae, delta=0.0)]

    for label, depth in ((PRIMARY, 3), ("cart_d2", 2)):
        tree, leaves = _cart(tau_hat, XD, depth)
        hat = assign(leaves, XE, dphi, const)
        mae = float(np.abs(hat - truth).mean())
        hi = _hi_leaf(leaves, tau_hat, XD)
        st = _leaf_stats(hi["cond"], XE, regE, dphi) if hi else {}
        rows.append(dict(
            rep=rep, n=n, model=label, depth=depth, rules=len(leaves),
            mae=mae, const_mae=const_mae, delta=mae - const_mae,
            region_share=100.0 * float(regE.mean()),
            best_jaccard=best_jaccard(leaves, XE, regE),
            hi_share=st.get("share", np.nan),
            hi_purity=st.get("purity", np.nan),
            hi_capture=st.get("capture", np.nan),
            hi_effect=st.get("effect", np.nan),
            hi_se=st.get("se", np.nan),
            hi_jaccard=st.get("jaccard", np.nan),
            stability=_split_stability(tau_hat, XD, XE, hi["cond"],
                                       depth, seed + 7) if hi else np.nan))
    return rows


def decide(res):
    out = {}
    for n, g in res[res.model == PRIMARY].groupby("n"):
        r = g.iloc[0]
        out[int(n)] = {
            "mean delta < 0": bool(r.delta < 0),
            "95% CI entirely below 0": bool(r.hi < 0),
            f"Jaccard >= {BASELINE_JACCARD}": bool(
                r.best_jaccard >= BASELINE_JACCARD),
            ">= 70% of reps better": bool(r.frac_better >= 70),
        }
    return out


def _config():
    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu = Xr.mean(0)
    sd = Xr.std(0)
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
    df.to_csv(os.path.join(root, "results", "gate12_replications.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate12_summary.csv"), index=False)
    checks = decide(res)

    print(f"\n=== Gate 12, {args.reps} reps per cell (primary = depth 3) ===")
    print(f"{'n':>7}{'model':>9}{'rules':>7}{'MAE':>9}{'const':>9}{'delta':>10}"
          f"{'95% CI':>22}{'%bet':>6}{'gain':>7}{'purity':>8}{'capture':>9}"
          f"{'hi eff':>8}{'hi se':>7}{'jacc':>7}{'stab':>7}")
    for _, r in res.sort_values(["n", "model"]).iterrows():
        print(f"{r.n:>7.0f}{r.model:>9}{r.rules:>7.1f}{r.mae:>9.4f}"
              f"{r.const_mae:>9.4f}{r.delta:>+10.4f}"
              f"   [{r.lo:+.4f},{r.hi:+.4f}]{r.frac_better:>5.0f}%"
              f"{r.rel_gain:>+6.1f}%{r.hi_purity:>8.2f}{r.hi_capture:>9.2f}"
              f"{r.hi_effect:>+8.3f}{r.hi_se:>7.3f}{r.best_jaccard:>7.2f}"
              f"{r.stability:>7.2f}")

    print("\nHigh-leaf prevalence:", {
        int(n): round(float(g[g.model == PRIMARY].hi_share.iloc[0]), 3)
        for n, g in res.groupby("n")})
    print("\nPre-registered decision checks:")
    adopted = []
    for n, ch in checks.items():
        print(f"  n={n}:")
        for k, v in ch.items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if all(ch.values()):
            adopted.append(n)
    print(f"\n-> {'ADOPT depth-3 at n = ' + str(adopted) if adopted else 'depth-3 NOT adopted; depth 2 remains the v1.5 model'}")
    return df


if __name__ == "__main__":
    main()

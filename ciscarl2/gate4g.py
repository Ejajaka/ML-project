"""Gate 4g - final validation: does reproducible recovery improve with n?

Same DGP, prevalence, discovery algorithm, Jaccard threshold (0.8), 20 half-samples
of 50 %, and evaluation protocol as Gate 4e/f. Nothing is changed except n.

The oracle here is **sklearn CART** (searches all midpoints), replacing the
quantile-grid tree used in Gate 4f, which could not place a boundary at the median.
That makes the oracle a fair representability ceiling.

Three methods, same folds, same half-samples:
    rf       RandomForestRegressor on psi  (production pipeline)
    psi_cart single CART on psi            (signal limit)
    oracle   single CART on the TRUE tau   (representability ceiling)

Usage
-----
    python -m ciscarl2.gate4g --reps 25 --n 8000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from . import dgp
from .estimators import crossfit_nuisances, truncate_weights
from .gate2 import TAU, pseudo_outcome
from .gate4d import EFFECT, MIN_SUPPORT, TSTAR, _delta_for
from .rules import dedup, eval_rule, extract_leaf_rules, rules_from_ensemble

from .paths import data_path

DATA = data_path()
STAB_B = 20
DEPTH = 3


def stats(leaves, truth):
    best = (0.0, 0.0, 0.0)
    for m in leaves:
        inter = float((m & truth).sum())
        j = inter / max(float((m | truth).sum()), 1)
        if j > best[0]:
            best = (j, inter / max(float(m.sum()), 1), inter / max(float(truth.sum()), 1))
    return best


def rf_leaves(X, y, idx, n_trees=30, leaf=MIN_SUPPORT):
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=DEPTH,
                               min_samples_leaf=leaf, random_state=0,
                               n_jobs=-1).fit(X[idx], y[idx])
    out = []
    for est in rf.estimators_:
        for cond, m in extract_leaf_rules(est, X, max(5, leaf // 4)):
            out.append(m)
    return out


def cart_leaves(X, y, idx, leaf=MIN_SUPPORT):
    t = DecisionTreeRegressor(max_depth=DEPTH, min_samples_leaf=leaf,
                              random_state=0).fit(X[idx], y[idx])
    return [m for _, m in extract_leaf_rules(t, X, 5)]


def run_rep(job):
    (rep, n, lam0, lam_c0, pool, mu, sd, i_cd4, i_age, cd4_cut, age_cut) = job
    rng = np.random.RandomState(110000 + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    in_reg = (X[:, i_cd4] < (cd4_cut - mu[i_cd4]) / sd[i_cd4]) & \
             (X[:, i_age] > (age_cut - mu[i_age]) / sd[i_age])
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    sim = dgp.generate(X, "covariate", lam_c0, lam0, rng, TSTAR,
                       effect_vec=np.where(in_reg, _delta_for(EFFECT, s0), 0.0))
    A, T, E = sim["A"], sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    tr, _ = train_test_split(np.arange(len(X)), test_size=0.30,
                             random_state=rep, stratify=A)
    XD, regD, tauD = X[tr], in_reg[tr], tau_i[tr]
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    full_idx = np.arange(len(tr))

    rows = []
    methods = {
        "rf": lambda idx, y: rf_leaves(XD, psi, idx) if y is None else None,
        "psi_cart": lambda idx, y: [eval_rule(c, XD) for c, _ in
                                    extract_leaf_rules(
                                        DecisionTreeRegressor(
                                            max_depth=DEPTH,
                                            min_samples_leaf=MIN_SUPPORT,
                                            random_state=0).fit(XD[idx], psi[idx]),
                                        XD, 5)],
        "oracle": lambda idx, y: [eval_rule(c, XD) for c, _ in
                                  extract_leaf_rules(
                                      DecisionTreeRegressor(
                                          max_depth=DEPTH,
                                          min_samples_leaf=MIN_SUPPORT,
                                          random_state=0).fit(XD[idx], tauD[idx]),
                                      XD, 5)],
    }
    for name, fn in methods.items():
        leaves = fn(full_idx, None)
        j_full, prec, rec = stats(leaves, regD)
        r2 = np.random.RandomState(rep + 1)
        hits = 0
        for _ in range(STAB_B):
            ii = r2.choice(len(tr), len(tr) // 2, replace=False)
            if stats(fn(ii, None), regD)[0] >= 0.8:
                hits += 1
        rows.append(dict(n=n, rep=rep, method=name, n_leaves=len(leaves),
                         jaccard=j_full, precision=prec, recall=rec,
                         stability=100 * hits / STAB_B))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n", type=int, nargs="+", default=[8000])
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    cd4_cut, age_cut = float(raw["cd40"].median()), float(raw["age"].median())
    i_cd4, i_age = dgp.COVARIATES.index("cd40"), dgp.COVARIATES.index("age")
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})

    from joblib import Parallel, delayed
    rows = []
    for n in args.n:
        jobs = [(rep, n, lam0, lam_c0, pool, mu, sd, i_cd4, i_age,
                 cd4_cut, age_cut) for rep in range(args.reps)]
        out = Parallel(n_jobs=args.jobs)(delayed(run_rep)(j) for j in jobs)
        rows += [r for ch in out for r in ch if r]
        print(f"  done n={n} ({args.reps} reps)", flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "gate4g_final.csv")
    df.to_csv(out_path, index=False)

    g = df.groupby(["n", "method"]).agg(
        reps=("rep", "nunique"), leaves=("n_leaves", "mean"),
        jaccard=("jaccard", "mean"), precision=("precision", "mean"),
        recall=("recall", "mean"), stability=("stability", "mean"),
        stability_sd=("stability", "std")).round(2)
    g["se"] = (g.stability_sd / np.sqrt(g.reps)).round(1)
    print("\n=== reproducible recovery vs sample size (fair oracle) ===")
    print(g.to_string())
    print("\nPRIMARY = stability. Threshold = 50 %. Shown with its standard error.")
    return df


if __name__ == "__main__":
    main()

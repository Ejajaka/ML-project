"""Gate 4e - does a variance-normalised splitting criterion make the region
reproducibly discoverable?

Motivation
----------
Gate 4d-CART showed the correct rule is IN the candidate pool (in-region 1.00 at
n=2000) but its subsample stability is 0.00-0.15, so stability selection rejects
it. The question is whether the split criterion is responsible.

Note on the criteria
--------------------
For outcome psi, the SSE reduction of a split is exactly

    (n_L n_R / n) * (psi_L - psi_R)^2

so ordinary MSE splitting IS the un-normalised heterogeneity criterion. A genuine
alternative is the variance-normalised ("maximal") statistic used by causal trees:

    T = (psi_L - psi_R)^2 / (s_L^2/n_L + s_R^2/n_R)

which rewards effect differences relative to noise and automatically penalises
tiny children. This is a pre-registered comparison of the two, holding everything
else (pseudo-outcome, folds, rule format, stability measure at Jaccard 0.8)
fixed.

Primary outcome
---------------
Stability of the TRUE REGION: the fraction of 20 half-sample trees whose best leaf
has Jaccard >= 0.8 with the true region. Measured identically for every method.

Usage
-----
    python -m ciscarl2.gate4e --reps 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import crossfit_nuisances, truncate_weights
from .gate2 import TAU, pseudo_outcome
from .gate4d import EFFECT, MIN_SUPPORT, TSTAR, _delta_for
from .rules import dedup, eval_rule, rules_from_ensemble

from .paths import data_path

DATA = data_path()
NQ = 9          # candidate thresholds per feature (quantiles)
STAB_B = 20


# ---------------------------------------------------------------------------
# custom greedy tree with a selectable split criterion
# ---------------------------------------------------------------------------
def _score(X, y, idx, j, thr, crit, min_leaf):
    left = X[idx, j] <= thr
    nL = int(left.sum())
    nR = len(idx) - nL
    if nL < min_leaf or nR < min_leaf:
        return -np.inf
    yL, yR = y[idx][left], y[idx][~left]
    mL, mR = yL.mean(), yR.mean()
    d = mL - mR
    if crit == "mse":
        return (nL * nR / len(idx)) * d * d
    # variance-normalised (causal-tree "maximal" statistic)
    vL = yL.var(ddof=1) if nL > 1 else 0.0
    vR = yR.var(ddof=1) if nR > 1 else 0.0
    denom = vL / nL + vR / nR
    return d * d / denom if denom > 0 else 0.0


def grow(X, y, idx, depth, max_depth, min_leaf, crit, cond, out):
    if depth >= max_depth or len(idx) < 2 * min_leaf:
        if cond:
            mask = np.zeros(len(X), bool)
            mask[idx] = True
            out.append((list(cond), mask))
        return
    best = (-np.inf, None, None)
    for j in range(X.shape[1]):
        qs = np.unique(np.quantile(X[idx, j], np.linspace(0.1, 0.9, NQ)))
        for thr in qs:
            s = _score(X, y, idx, j, thr, crit, min_leaf)
            if s > best[0]:
                best = (s, j, thr)
    if best[1] is None or best[0] <= 0:
        mask = np.zeros(len(X), bool)
        mask[idx] = True
        out.append((list(cond), mask))
        return
    _, j, thr = best
    left = idx[X[idx, j] <= thr]
    right = idx[X[idx, j] > thr]
    grow(X, y, left, depth + 1, max_depth, min_leaf, crit,
         cond + [(j, "<=", float(thr))], out)
    grow(X, y, right, depth + 1, max_depth, min_leaf, crit,
         cond + [(j, ">", float(thr))], out)


def tree_leaves(X, y, idx, crit, depth=3, min_leaf=MIN_SUPPORT):
    out = []
    grow(X, y, idx, 0, depth, min_leaf, crit, [], out)
    return out


def best_leaf_stats(leaves, truth):
    """(jaccard, precision, recall) of the leaf that best overlaps the truth."""
    if not leaves:
        return 0.0, 0.0, 0.0
    best = (0.0, 0.0, 0.0)
    for _, m in leaves:
        inter = float((m & truth).sum())
        j = inter / max(float((m | truth).sum()), 1)
        if j > best[0]:
            best = (j, inter / max(float(m.sum()), 1),
                    inter / max(float(truth.sum()), 1))
    return best


def run_reps(n, reps, pool, lam0, lam_c0, mu, sd, i_cd4, i_age):
    rows = []
    for rep in range(reps):
        rng = np.random.RandomState(110000 + rep)
        X = pool[rng.choice(len(pool), n, replace=True)]
        in_reg = (X[:, i_cd4] < (float(340) - mu[i_cd4]) / sd[i_cd4]) & \
                 (X[:, i_age] > (float(34) - mu[i_age]) / sd[i_age])
        lp0 = X @ dgp.BETA
        s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
        sim = dgp.generate(X, "covariate", lam_c0, lam0, rng, TSTAR,
                           effect_vec=np.where(in_reg, _delta_for(EFFECT, s0), 0.0))
        A, T, E = sim["A"], sim["Tobs"], sim["E"]
        tau_i = sim["S1"] - sim["S0"]
        tr, _ = train_test_split(np.arange(len(X)), test_size=0.30,
                                 random_state=rep, stratify=A)
        XD, regD = X[tr], in_reg[tr]
        muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
            XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
        GD, _ = truncate_weights(GD, fid, fcaps)
        psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)

        full = {
            "rf": [c for c, _ in dedup(rules_from_ensemble(
                RandomForestRegressor(n_estimators=30, max_depth=3,
                                      min_samples_leaf=MIN_SUPPORT,
                                      random_state=0, n_jobs=-1).fit(XD, psi),
                XD, MIN_SUPPORT, max_trees=30))],
            "mse": tree_leaves(XD, psi, np.arange(len(tr)), "mse"),
            "tstat": tree_leaves(XD, psi, np.arange(len(tr)), "tstat"),
            "oracle": tree_leaves(XD, tau_i[tr], np.arange(len(tr)), "mse"),
        }

        subs = {k: [] for k in full}
        r2 = np.random.RandomState(rep + 1)
        for _ in range(STAB_B):
            ii = r2.choice(len(tr), len(tr) // 2, replace=False)
            subs["rf"].append([c for c, _ in dedup(rules_from_ensemble(
                RandomForestRegressor(n_estimators=15, max_depth=3,
                                      min_samples_leaf=30, random_state=0,
                                      n_jobs=-1).fit(XD[ii], psi[ii]),
                XD, 5, max_trees=15))])
            subs["mse"].append(tree_leaves(XD, psi, ii, "mse"))
            subs["tstat"].append(tree_leaves(XD, psi, ii, "tstat"))
            subs["oracle"].append(tree_leaves(XD, tau_i[tr], ii, "mse"))

        for k in full:
            def masks_of(rules):
                if k == "rf":
                    return [(c, eval_rule(c, XD)) for c in rules]
                return rules
            fullm = masks_of(full[k])
            j_full, prec, rec = best_leaf_stats(fullm, regD)
            hits = 0
            for sr in subs[k]:
                if best_leaf_stats(masks_of(sr), regD)[0] >= 0.8:
                    hits += 1
            rows.append(dict(n=n, rep=rep, method=k,
                             n_leaves_full=len(fullm),
                             best_jaccard_full=j_full,
                             precision=prec, recall=rec,
                             stability=100 * hits / STAB_B))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    i_cd4, i_age = dgp.COVARIATES.index("cd40"), dgp.COVARIATES.index("age")
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})

    rows = []
    for n in (589, 2000):
        rows += run_reps(n, args.reps, pool, lam0, lam_c0, mu, sd, i_cd4, i_age)
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "gate4e_summary.csv")
    df.to_csv(out, index=False)

    g = df.groupby(["n", "method"]).agg(
        reps=("rep", "nunique"),
        leaves=("n_leaves_full", "mean"),
        best_jaccard_full=("best_jaccard_full", "mean"),
        stability=("stability", "mean")).round(2)
    print("\n=== stability of the TRUE REGION (Jaccard >= 0.8 in half-samples) ===")
    print(g.to_string())
    print("\nPrimary outcome: stability. 5 % = appears in 1 of 20 half-samples.")
    return df


if __name__ == "__main__":
    main()

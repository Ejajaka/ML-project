"""Gate 4d-CART: oracle vs pseudo-outcome vs CISCaRL search.

Three-way recoverability test on the SAME discovery fold, pseudo-outcome and seeds.

  1. CART fitted to the TRUE treatment effect   -> is the region recoverable at all
                                                   under the tree representation?
  2. CART fitted to psi (the DR pseudo-outcome) -> is the signal usable by a tree?
  3. CISCaRL's RF-on-psi search                 -> is the search the weak link?

Region (fixed, same everywhere):  CD4 < median(340)  AND  age > median(34).

For each configuration we report, for the purest leaf rule:
    in-region fraction, sensitivity, false-positive rate, coverage, conditions
plus the number of leaves with >= 70 % in-region purity.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

sys.path.insert(0, r"C:\Users\roshn\OneDrive\Desktop\Amrita\SEM 5\ML\Project")
from ciscarl2 import dgp
from ciscarl2.estimators import crossfit_nuisances, truncate_weights
from ciscarl2.gate2 import TAU, pseudo_outcome
from ciscarl2.gate4d import EFFECT, MIN_SUPPORT, THR, TSTAR, _delta_for
from ciscarl2.rules import dedup, extract_leaf_rules, rules_from_ensemble, stability_scores

from .paths import data_path

DATA = data_path()
DEPTHS = (3, 5)
LEAVES = (20, 40, 60)
MIN_SUP = 10


def leaves_quality(tree, X, reg, min_support=MIN_SUP):
    out = []
    for cond, m in extract_leaf_rules(tree, X, min_support):
        n = int(m.sum())
        if n < 5:
            continue
        inter = int((m & reg).sum())
        out.append(dict(n=n, inreg=inter / n,
                        sens=inter / max(int(reg.sum()), 1),
                        fp=(n - inter) / max(int((~reg).sum()), 1),
                        cond=" & ".join(f"x{f}{op}{t:.1f}" for f, op, t in cond)))
    return out


def main():
    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    i_cd4, i_age = dgp.COVARIATES.index("cd40"), dgp.COVARIATES.index("age")
    cd4_cut = float(raw["cd40"].median())
    age_cut = float(raw["age"].median())
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})

    rows = []
    shown = False
    for n in (589, 2000, 4000):
        for rep in range(12):
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

            if not shown and n == 589 and rep == 0:
                shown = True
                q = np.quantile(psi[regD], [0.1, 0.5, 0.9])
                q0 = np.quantile(psi[~regD], [0.1, 0.5, 0.9])
                print("psi distribution on the discovery fold (n=589, rep=0):")
                print(f"  in-region   q10/q50/q90 = {q[0]:+.2f} / {q[1]:+.2f} / {q[2]:+.2f}")
                print(f"  out-region  q10/q50/q90 = {q0[0]:+.2f} / {q0[1]:+.2f} / {q0[2]:+.2f}")
                print(f"  mean separation = {psi[regD].mean()-psi[~regD].mean():+.3f}, "
                      f"overlap of central 80% is large if the quantile ranges overlap\n")

            for target_name, target in (("oracle", tauD), ("psi", psi)):
                for d in DEPTHS:
                    for leaf in LEAVES:
                        t = DecisionTreeRegressor(max_depth=d, min_samples_leaf=leaf,
                                                  random_state=0).fit(XD, target)
                        q_ = leaves_quality(t, XD, regD)
                        if not q_:
                            continue
                        b = max(q_, key=lambda z: z["inreg"])
                        rows.append(dict(n=n, rep=rep, target=target_name, depth=d,
                                         leaf=leaf, n_leaves=len(q_),
                                         best_inreg=b["inreg"], best_sens=b["sens"],
                                         best_fp=b["fp"], coverage=b["n"] / len(XD),
                                         pure_leaves=sum(1 for z in q_ if z["inreg"] >= 0.7),
                                         cond=b["cond"]))
                        if shown and target_name == "oracle" and n == 589 and rep == 0 \
                                and d == 3 and leaf == 40:
                            print("oracle CART (depth 3, leaf 40) structure:")
                            print(export_text(t, feature_names=[f"x{i}" for i in
                                                                range(X.shape[1])],
                                              max_depth=3))

            # CISCaRL's own search, for reference
            rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                                       min_samples_leaf=MIN_SUPPORT,
                                       random_state=0, n_jobs=-1).fit(XD, psi)
            cands = dedup(rules_from_ensemble(rf, XD, MIN_SUPPORT, max_trees=30))
            if cands:
                subs = []
                r2 = np.random.RandomState(rep + 1)
                for _ in range(20):
                    ii = r2.choice(len(tr), len(tr) // 2, replace=False)
                    t_ = RandomForestRegressor(n_estimators=15, max_depth=3,
                                               min_samples_leaf=10, random_state=0,
                                               n_jobs=-1).fit(XD[ii], psi[ii])
                    subs.append(rules_from_ensemble(t_, XD, 5, max_trees=15))
                freq = stability_scores(cands, subs, XD)
                qs = leaves_quality(type("T", (), {"tree_": None})(), XD, regD) if False else []
                for (cond, m), f in zip(cands, freq):
                    nn = int(m.sum())
                    inter = int((m & regD).sum())
                    qs.append(dict(n=nn, inreg=inter / max(nn, 1),
                                   sens=inter / max(int(regD.sum()), 1),
                                   fp=(nn - inter) / max(int((~regD).sum()), 1),
                                   stab=float(f),
                                   cond=" & ".join(f"x{a}{o}{t:.1f}" for a, o, t in cond)))
                b = max(qs, key=lambda z: z["inreg"])
                sel = [z for z in qs if z["stab"] >= THR]
                rows.append(dict(n=n, rep=rep, target="ciscarl", depth=3,
                                 leaf=MIN_SUPPORT, n_leaves=len(qs),
                                 best_inreg=b["inreg"], best_sens=b["sens"],
                                 best_fp=b["fp"], coverage=b["n"] / len(XD),
                                 pure_leaves=sum(1 for z in qs if z["inreg"] >= 0.7),
                                 cond=b["cond"],
                                 n_selected=len(sel),
                                 sel_inreg=(max((z["inreg"] for z in sel), default=np.nan))))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "gate4d_cart.csv"), index=False)

    print("\n=== recoverability of the true region ===")
    print(f"{'n':>6} {'target':<9} {'depth':>6} {'leaf':>5} {'leaves':>7} "
          f"{'best in-reg':>12} {'sens':>6} {'FP':>6} {'cov':>6} {'pure':>5}")
    for (n, tgt), g in df.groupby(["n", "target"]):
        for _, r in g.iterrows():
            if r.target == "ciscarl" and not (r.depth == 3 and r.leaf == 60):
                continue
            print(f"{n:>6} {tgt:<9} {r.depth:>6.0f} {r.leaf:>5.0f} "
                  f"{r.n_leaves:>7.0f} {r.best_inreg:>12.2f} {r.best_sens:>6.2f} "
                  f"{r.best_fp:>6.2f} {r.coverage:>6.2f} {r.pure_leaves:>5.0f}")


if __name__ == "__main__":
    main()

"""Gate 4d discovery funnel: is the true region representable, generated, or filtered?

Gate 4d showed 0.0 rules/rep at n=589 and 0.3 at n=2000, with the few rules found
covering the NULL region (in-region fraction 0.00-0.01). This separates three
explanations:

  A. the region is never a candidate          -> candidate generation / psi problem
  B. it is a candidate but fails stability    -> threshold or power problem
  C. it is selected but the estimate is flat   -> estimation problem

For each replication we record the funnel and, for the BEST candidate by in-region
fraction, its overlap, support and stability.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, r"C:\Users\roshn\OneDrive\Desktop\Amrita\SEM 5\ML\Project")
from ciscarl2 import dgp
from ciscarl2.estimators import crossfit_nuisances, truncate_weights
from ciscarl2.gate2 import TAU, pseudo_outcome
from ciscarl2.gate4d import (AGE_CUT, CD4_CUT, EFFECT, MIN_SUPPORT, THR, TSTAR,
                             _delta_for)
from ciscarl2.rules import dedup, eval_rule, rules_from_ensemble, stability_scores

from .paths import data_path

DATA = data_path()


def one(rep, n, pool, lam0, lam_c0, mu, sd, i_cd4, i_age):
    rng = np.random.RandomState(110000 + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    in_reg = (X[:, i_cd4] < (CD4_CUT - mu[i_cd4]) / sd[i_cd4]) & \
             (X[:, i_age] > (AGE_CUT - mu[i_age]) / sd[i_age])
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    sim = dgp.generate(X, "covariate", lam_c0, lam0, rng, TSTAR,
                       effect_vec=np.where(in_reg, _delta_for(EFFECT, s0), 0.0))
    A, T, E = sim["A"], sim["Tobs"], sim["E"]

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=rep, stratify=A)
    XD = X[tr]
    reg_D = in_reg[tr]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)

    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=MIN_SUPPORT, random_state=0,
                               n_jobs=-1).fit(XD, psi)
    cands = dedup(rules_from_ensemble(rf, XD, MIN_SUPPORT, max_trees=30))

    subs = []
    r2 = np.random.RandomState(rep + 1)
    for _ in range(20):
        ii = r2.choice(len(tr), len(tr) // 2, replace=False)
        t_ = RandomForestRegressor(n_estimators=15, max_depth=3,
                                   min_samples_leaf=10, random_state=0,
                                   n_jobs=-1).fit(XD[ii], psi[ii])
        subs.append(rules_from_ensemble(t_, XD, 5, max_trees=15))
    freq = stability_scores(cands, subs, XD) if cands else np.array([])

    if len(cands) == 0:
        return dict(rep=rep, n=n, n_cand=0, n_stab=0, best_inreg=0.0,
                    best_stab=0.0, best_support=0, any_overlap=0,
                    psi_sd=float(np.std(psi)),
                    psi_sep=float(np.mean(psi[reg_D]) - np.mean(psi[~reg_D])))

    inregs = np.array([float((m & reg_D).sum() / max(m.sum(), 1))
                       for _, m in cands])
    supports = np.array([int(m.sum()) for _, m in cands])
    j = int(np.argmax(inregs))
    return dict(rep=rep, n=n, n_cand=len(cands),
                n_stab=int((freq >= THR).sum()),
                best_inreg=float(inregs[j]), best_stab=float(freq[j]),
                best_support=int(supports[j]),
                any_overlap=int((inregs >= 0.7).sum()),
                psi_sd=float(np.std(psi)),
                psi_sep=float(np.mean(psi[reg_D]) - np.mean(psi[~reg_D])))


def main():
    import pandas as pd_
    pool = dgp.load_covariate_pool(DATA)
    raw = pd_.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    i_cd4, i_age = dgp.COVARIATES.index("cd40"), dgp.COVARIATES.index("age")
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})

    for n in (589, 2000, 4000):
        reps = 15
        rows = [one(r, n, pool, lam0, lam_c0, mu, sd, i_cd4, i_age)
                for r in range(reps)]
        d = pd.DataFrame(rows)
        print(f"\n=== n={n} ({reps} reps) ===")
        print(f"  candidates generated per rep      : {d.n_cand.mean():.0f}")
        print(f"  candidates passing stability {THR} : {d.n_stab.mean():.1f}")
        print(f"  candidates with >=70 % in-region  : {d.any_overlap.mean():.1f}")
        print(f"  BEST candidate's in-region frac   : {d.best_inreg.mean():.2f}")
        print(f"  BEST candidate's stability        : {d.best_stab.mean():.2f}")
        print(f"  BEST candidate's support          : {d.best_support.mean():.0f}")
        print(f"  psi sd / psi separation (region)  : "
              f"{d.psi_sd.mean():.2f} / {d.psi_sep.mean():+.3f}")


if __name__ == "__main__":
    main()

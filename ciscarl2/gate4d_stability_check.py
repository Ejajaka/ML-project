"""Is the good rule found but rejected by stability selection?"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, r"C:\Users\roshn\OneDrive\Desktop\Amrita\SEM 5\ML\Project")
from ciscarl2 import dgp
from ciscarl2.estimators import crossfit_nuisances, truncate_weights
from ciscarl2.gate2 import TAU, pseudo_outcome
from ciscarl2.gate4d import EFFECT, MIN_SUPPORT, THR, TSTAR, _delta_for
from ciscarl2.rules import dedup, rules_from_ensemble, stability_scores

from .paths import data_path

DATA = data_path()
pool = dgp.load_covariate_pool(DATA)
raw = pd.read_csv(DATA); raw = raw[raw.arms.isin([0, 1])]
Xr = raw[dgp.COVARIATES].to_numpy(float)
mu, sd = Xr.mean(0), Xr.std(0)
cd4_cut, age_cut = float(raw["cd40"].median()), float(raw["age"].median())
i_cd4, i_age = dgp.COVARIATES.index("cd40"), dgp.COVARIATES.index("age")
lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                             gen_kwargs={"tau": TAU})

print(f"{'n':>6} {'rep':>4} {'cands':>6} {'best_inreg':>10} {'stab_of_best':>13} "
      f"{'best_stab_among_good':>20} {'n_stab>=.5':>10} {'n_good_stab>=.5':>15}")
for n in (589, 2000):
    for rep in range(10):
        rng = np.random.RandomState(110000 + rep)
        X = pool[rng.choice(len(pool), n, replace=True)]
        in_reg = (X[:, i_cd4] < (cd4_cut - mu[i_cd4]) / sd[i_cd4]) & \
                 (X[:, i_age] > (age_cut - mu[i_age]) / sd[i_age])
        lp0 = X @ dgp.BETA
        s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
        sim = dgp.generate(X, "covariate", lam_c0, lam0, rng, TSTAR,
                           effect_vec=np.where(in_reg, _delta_for(EFFECT, s0), 0.0))
        A, T, E = sim["A"], sim["Tobs"], sim["E"]
        tr, _ = train_test_split(np.arange(len(X)), test_size=0.30,
                                 random_state=rep, stratify=A)
        XD, regD = X[tr], in_reg[tr]
        muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
            XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
        GD, _ = truncate_weights(GD, fid, fcaps)
        psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)

        rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                                   min_samples_leaf=MIN_SUPPORT, random_state=0,
                                   n_jobs=-1).fit(XD, psi)
        cands = dedup(rules_from_ensemble(rf, XD, MIN_SUPPORT, max_trees=30))
        if not cands:
            continue
        subs = []
        r2 = np.random.RandomState(rep + 1)
        for _ in range(20):
            ii = r2.choice(len(tr), len(tr) // 2, replace=False)
            t_ = RandomForestRegressor(n_estimators=15, max_depth=3,
                                       min_samples_leaf=10, random_state=0,
                                       n_jobs=-1).fit(XD[ii], psi[ii])
            subs.append(rules_from_ensemble(t_, XD, 5, max_trees=15))
        freq = stability_scores(cands, subs, XD)
        inregs = np.array([float((m & regD).sum() / max(m.sum(), 1)) for _, m in cands])
        j = int(np.argmax(inregs))
        good = inregs >= 0.9
        print(f"{n:>6} {rep:>4} {len(cands):>6} {inregs[j]:>10.2f} {freq[j]:>13.2f} "
              f"{(freq[good].max() if good.any() else 0):>20.2f} "
              f"{int((freq >= THR).sum()):>10} "
              f"{int(((freq >= THR) & good).sum()):>15}")

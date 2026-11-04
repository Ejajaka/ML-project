"""Gate 6 - is the failure the pseudo-outcome or the rule-extraction pipeline?

Same data, same folds, same rule extraction, same selection, same evaluation. The
ONLY thing that changes is the target the discovery forest is fitted to:

    psi   the AIPW pseudo-outcome (current pipeline)
    tau   the TRUE individual treatment effect (perfect effect information)
    cart  a single CART on the true effect (reference for the representation)

If rules discovered from the true effect recover the region and beat the constant
baseline, the bottleneck is the pseudo-outcome. If they do not, the bottleneck is
the rule representation / extraction step.

Every method is evaluated the same way: rules discovered on the training fold are
applied to the held-out test fold, each rule's effect is estimated with AIPW on the
test fold, and overall MAE is computed against the true individual tau on the same
patients. The constant baseline is the marginal ATE estimated on TRAIN.

Usage
-----
    python -m ciscarl2.gate6 --reps 20 --n 589 2000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import TAU, pseudo_outcome
from .gate4d import EFFECT, MIN_SUPPORT, THR, TSTAR, _delta_for, _verdict
from .rules import (dedup, eval_rule, extract_leaf_rules, prune,
                    rules_from_ensemble, stability_scores)

from .paths import data_path

DATA = data_path()
STAB_B = 20


def discover(target, Xtr, psi, rng, stab_b=STAB_B):
    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=MIN_SUPPORT, random_state=0,
                               n_jobs=-1).fit(Xtr, target)
    cands = dedup(rules_from_ensemble(rf, Xtr, MIN_SUPPORT, max_trees=30))
    if not cands:
        return [], np.array([])
    subs = []
    for _ in range(stab_b):
        ii = rng.choice(len(Xtr), len(Xtr) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, MIN_SUPPORT // 2),
                                  random_state=0, n_jobs=-1).fit(Xtr[ii], target[ii])
        subs.append(rules_from_ensemble(t, Xtr, 5, max_trees=15))
    freq = stability_scores(cands, subs, Xtr)
    sel = [dict(cond=c, mask=m, score=f) for (c, m), f in zip(cands, freq)
           if f >= THR]
    return (prune(sel, Xtr, max_rules=99, overlap=0.8) if sel else []), freq


def assign(sel, Xte, dphi, default):
    tau = np.full(len(Xte), default)
    verd = np.full(len(Xte), _verdict(default - 1e9, default + 1e9), dtype=object)
    assigned = np.zeros(len(Xte), bool)
    for r in sorted(sel, key=lambda d: -len(d["cond"])):
        m = eval_rule(r["cond"], Xte)
        nr = int(m.sum())
        if nr < 5:
            continue
        t_r = float(dphi[m].mean())
        s_r = float(np.sqrt(((dphi[m] - t_r) ** 2).sum()) / nr)
        q = float(tdist.ppf(0.975, max(nr - 1, 1)))
        take = m & ~assigned
        tau[take] = t_r
        verd[take] = _verdict(t_r - q * s_r, t_r + q * s_r)
        assigned |= take
    return tau, verd


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

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=rep, stratify=A)
    XD, XE = X[tr], X[te]
    regD, regE = in_reg[tr], in_reg[te]
    tt = tau_i[te]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)

    # constant baseline (marginal ATE on TRAIN)
    _, _, p1t, p0t = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const_ate = float((p1t - p0t).mean())

    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=rep)
    dphi = p1 - p0
    default = const_ate

    rows = []
    # constant
    rows.append(dict(rep=rep, n=n, method="constant", n_rules=0,
                     jaccard=np.nan, stability=np.nan,
                     overall_mae=float(np.abs(np.full(len(te), const_ate) - tt).mean())))

    for name, target in (("rf_on_psi", psi), ("rf_on_true_tau", tau_i[tr])):
        sel, freq = discover(target, XD, psi, np.random.RandomState(rep + 1))
        tau_hat, verd = assign(sel, XE, dphi, default)
        jac = 0.0
        for r in sel:
            m = eval_rule(r["cond"], XE)
            jac = max(jac, float((m & regE).sum() / max((m | regE).sum(), 1)))
        rows.append(dict(rep=rep, n=n, method=name, n_rules=len(sel),
                         jaccard=jac,
                         stability=(100.0 * float(np.max(freq)) if len(freq) else 0.0),
                         overall_mae=float(np.abs(tau_hat - tt).mean())))

    # single CART on the true effect, same extraction
    t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=MIN_SUPPORT,
                              random_state=0).fit(XD, tau_i[tr])
    leaves = [dict(cond=c, mask=m, score=1.0)
              for c, m in extract_leaf_rules(t, XD, MIN_SUPPORT)]
    tau_hat, _ = assign(leaves, XE, dphi, default)
    jac = 0.0
    for r in leaves:
        m = eval_rule(r["cond"], XE)
        jac = max(jac, float((m & regE).sum() / max((m | regE).sum(), 1)))
    rows.append(dict(rep=rep, n=n, method="cart_on_true_tau", n_rules=len(leaves),
                     jaccard=jac, stability=np.nan,
                     overall_mae=float(np.abs(tau_hat - tt).mean())))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n", type=int, nargs="+", default=[589, 2000])
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
        print(f"  done n={n}", flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "gate6_discovery_target.csv")
    df.to_csv(out_path, index=False)

    g = df.groupby(["n", "method"]).agg(
        reps=("rep", "nunique"), rules=("n_rules", "mean"),
        jaccard=("jaccard", "mean"), best_stability=("stability", "mean"),
        overall_mae=("overall_mae", "mean"),
        mae_se=("overall_mae", "sem")).round(4)
    print("\n=== does perfect effect information fix discovery? ===")
    print(g.to_string())
    print("\nMAE is against the true individual tau on the same held-out patients.")
    return df


if __name__ == "__main__":
    main()

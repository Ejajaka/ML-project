"""Gate 7 - min_support sweep at n=2000, paired against the constant baseline.

Motivation
----------
Gate 6 showed a 4-rule CART fitted to the TRUE effect reaches MAE 0.0581 against a
constant baseline of 0.0743, while the current RF (12.6 rules from psi) reaches
0.0720. The difference is partition granularity: more rules means fewer patients
per rule, and per-rule estimation noise dominates. That experiment used the exact
treatment-effect structure, so it establishes the principle, not the mechanism.

This sweep tests whether the same lever works through the production pipeline.

Primary endpoint
----------------
    delta = MAE_CISCaRL(min_support) - MAE_constant

computed **paired within replication**, on the same held-out test patients, against
the same training-derived constant.

Pre-registered adoption rule (fixed before running)
---------------------------------------------------
Adopt a higher support floor only if ALL of:
  1. mean paired delta < 0
  2. the 95 % paired bootstrap CI excludes zero
  3. recovery (Jaccard vs the true region) is not materially worse than the
     current configuration
  4. the improvement is not driven by one or two replications
     (>= 70 % of replications must have delta < 0)

If none qualifies, the current min_support stays and the result is reported.

Usage
-----
    python -m ciscarl2.gate7 --reps 30 --n 2000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import TAU, pseudo_outcome
from .gate4d import EFFECT, THR, TSTAR, _delta_for, _verdict
from .rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

from .paths import data_path

DATA = data_path()
SUPPORTS = (60, 100, 150, 200)
STAB_B = 20
BASE = 60


def discover(ms, Xtr, target, seed):
    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=ms, random_state=0,
                               n_jobs=-1).fit(Xtr, target)
    cands = dedup(rules_from_ensemble(rf, Xtr, ms, max_trees=30))
    if not cands:
        return [], np.array([])
    rng = np.random.RandomState(seed)          # identical draws across configs
    subs = []
    for _ in range(STAB_B):
        ii = rng.choice(len(Xtr), len(Xtr) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, ms // 2),
                                  random_state=0, n_jobs=-1).fit(Xtr[ii], target[ii])
        subs.append(rules_from_ensemble(t, Xtr, 5, max_trees=15))
    freq = stability_scores(cands, subs, Xtr)
    sel = [dict(cond=c, mask=m, score=f) for (c, m), f in zip(cands, freq)
           if f >= THR]
    return (prune(sel, Xtr, max_rules=99, overlap=0.8) if sel else []), freq


def assign(sel, XE, dphi, default):
    tau = np.full(len(XE), default)
    assigned = np.zeros(len(XE), bool)
    for r in sorted(sel, key=lambda d: -len(d["cond"])):
        m = eval_rule(r["cond"], XE)
        nr = int(m.sum())
        if nr < 5:
            continue
        t_r = float(dphi[m].mean())
        take = m & ~assigned
        tau[take] = t_r
        assigned |= take
    return tau


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
    regE, tt = in_reg[te], tau_i[te]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    _, _, p1t, p0t = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const_ate = float((p1t - p0t).mean())
    const_mae = float(np.abs(np.full(len(te), const_ate) - tt).mean())

    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=rep)
    dphi = p1 - p0

    rows = []
    for ms in SUPPORTS:
        sel, freq = discover(ms, XD, psi, seed=rep + 1)
        tau_hat = assign(sel, XE, dphi, const_ate)
        mae = float(np.abs(tau_hat - tt).mean())
        jac = 0.0
        for r in sel:
            m = eval_rule(r["cond"], XE)
            jac = max(jac, float((m & regE).sum() / max((m | regE).sum(), 1)))
        rows.append(dict(rep=rep, n=n, min_support=ms, n_rules=len(sel),
                         jaccard=jac, mae=mae, const_mae=const_mae,
                         delta=mae - const_mae))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n", type=int, default=2000)
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
    jobs = [(rep, args.n, lam0, lam_c0, pool, mu, sd, i_cd4, i_age,
             cd4_cut, age_cut) for rep in range(args.reps)]
    out = Parallel(n_jobs=args.jobs)(delayed(run_rep)(j) for j in jobs)
    df = pd.DataFrame([r for ch in out for r in ch if r])
    df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "gate7_support_sweep.csv"), index=False)

    print(f"\n=== min_support sweep, n={args.n}, {args.reps} reps ===")
    print(f"constant baseline MAE = {df.const_mae.mean():.4f}")
    print(f"{'min_sup':>8}{'rules':>7}{'Jaccard':>9}{'MAE':>9}{'delta':>9}"
          f"{'95% CI':>18}{'%reps better':>14}  adopt?")
    rows = []
    for ms, g in df.groupby("min_support"):
        d = g.delta.to_numpy()
        rs = np.random.RandomState(0)
        bs = [rs.choice(d, len(d), replace=True).mean() for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        base_j = df[df.min_support == BASE].jaccard.mean()
        ok = (d.mean() < 0) and (hi < 0) and (g.jaccard.mean() >= 0.8 * base_j) \
             and ((d < 0).mean() >= 0.7)
        rows.append(dict(min_support=ms, rules=g.n_rules.mean(),
                         jaccard=g.jaccard.mean(), mae=g.mae.mean(),
                         delta=d.mean(), lo=lo, hi=hi,
                         frac_better=100 * (d < 0).mean(), adopt=bool(ok)))
        print(f"{ms:>8}{g.n_rules.mean():>7.1f}{g.jaccard.mean():>9.3f}"
              f"{g.mae.mean():>9.4f}{d.mean():>+9.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]{100*(d<0).mean():>13.0f} %"
              f"  {'YES' if ok else 'no'}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "gate7_summary.csv"), index=False)
    best = res[res.adopt]
    print("\nAdoption rule: delta<0 AND CI excludes 0 AND recovery not worse than "
          f"80 % of the min_support={BASE} value AND >=70 % of reps better.")
    print(f"-> {'ADOPT min_support=' + str(int(best.iloc[0].min_support)) if len(best) else 'NONE qualifies; keep 60'}")
    return df


if __name__ == "__main__":
    main()

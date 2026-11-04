"""Gate 9 - does a smooth plug-in effect target fix subgroup discovery?

Pre-registration: `results/gate9_PREREGISTRATION.md` (written before this ran).

Only the discovery target changes. The discovery pipeline (RF depth/support,
stability bootstrap, threshold, pruning), the effect estimator (held-out AIPW
phi), the constant baseline, the oracle and the evaluation patients are all
frozen at v1.4 and shared across arms.

    aipw_psi     psi            current production target
    cox_plugin   mu1 - mu0      cross-fitted per-arm Cox
    rsf_plugin   mu1 - mu0      cross-fitted per-arm Random Survival Forest

Arm 3 tests whether any gain depends on the Cox outcome model being correctly
specified in this simulation.

Pre-registered adoption rule (per arm)
--------------------------------------
  1. mean paired delta = MAE(CISCaRL) - MAE(constant) < 0
  2. 95 % paired bootstrap CI entirely below 0
  3. mean best-rule Jaccard not worse than aiwm_psi (tolerance 0.01)
  4. >= 70 % of replications have delta < 0

Usage
-----
    python -m ciscarl2.gate9 --reps 2 --n 2000      # smoke test
    python -m ciscarl2.gate9 --reps 30 --n 2000
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
from .gate2 import TAU, pseudo_outcome
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate8 import assign, best_jaccard, discover, oracle_mae

from .paths import data_path

DATA = data_path()
BASELINE = "aipw_psi"
TOL_JACCARD = 0.01


def run_rep(job):
    (rep, n, lam0, lam_c0, pool, mu_, sd_, i_cd4, i_age,
     cd4_cut, age_cut) = job
    rng = np.random.RandomState(110000 + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    in_reg = (X[:, i_cd4] < (cd4_cut - mu_[i_cd4]) / sd_[i_cd4]) & \
             (X[:, i_age] > (age_cut - mu_[i_age]) / sd_[i_age])
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
    truth, truth_tr = tau_i[te], tau_i[tr]

    # ---- shared nuisances (identical in every arm) ----
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    _, _, q1, q0 = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const_ate = float((q1 - q0).mean())

    # held-out AIPW phi: identical for every arm, fitted on TRAIN only
    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=rep)
    dphi = p1 - p0

    const_mae = float(np.abs(const_ate - truth).mean())
    orc_mae = oracle_mae(dphi, regE, truth)

    # ---- discovery targets ----
    targets = {"aipw_psi": psi,
               "cox_plugin": muD[:, 1] - muD[:, 0]}

    muR, _, _, fidR, fcapsR, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, mu_model="rsf",
        return_folds=True)
    targets["rsf_plugin"] = muR[:, 1] - muR[:, 0]

    arms = {}
    for name, target in targets.items():
        sel, stab = discover(XD, target, rep + 1)
        hat = assign(sel, XE, dphi, const_ate)
        arms[name] = dict(
            rep=rep, n=n, arm=name, n_rules=len(sel),
            jaccard=best_jaccard(sel, XE, regE), stability=stab,
            target_sd=float(np.std(target)),
            target_corr=float(np.corrcoef(target, truth_tr)[0, 1]),
            target_contrast=float(target[regD].mean() - target[~regD].mean()),
            mae=float(np.abs(hat - truth).mean()),
            const_mae=const_mae, oracle_mae=orc_mae,
            region_share=100.0 * float(regE.mean()))

    base_mae = arms[BASELINE]["mae"]
    for d in arms.values():
        d["baseline_mae"] = base_mae
        d["delta_const"] = d["mae"] - d["const_mae"]
        d["delta_baseline"] = d["mae"] - base_mae
    return list(arms.values())


def _boot(d, n=5000, seed=0):
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for arm, g in df.groupby("arm"):
        out.append(dict(
            arm=arm, reps=len(g),
            target_sd=g.target_sd.mean(), target_corr=g.target_corr.mean(),
            target_contrast=g.target_contrast.mean(),
            rules=g.n_rules.mean(), jaccard=g.jaccard.mean(),
            stability=g.stability.mean(), mae=g.mae.mean(),
            const_mae=g.const_mae.mean(), oracle_mae=g.oracle_mae.mean(),
            delta_const=g.delta_const.mean(),
            dc_lo=_boot(g.delta_const.to_numpy())[0],
            dc_hi=_boot(g.delta_const.to_numpy())[1],
            frac_better_const=100.0 * float((g.delta_const < 0).mean()),
            delta_base=g.delta_baseline.mean(),
            db_lo=_boot(g.delta_baseline.to_numpy())[0],
            db_hi=_boot(g.delta_baseline.to_numpy())[1],
            frac_better_base=100.0 * float((g.delta_baseline < 0).mean()),
            rel_gain=100.0 * (1.0 - g.mae.mean() / g.const_mae.mean())))
    return pd.DataFrame(out)


def decide(res):
    base = res[res.arm == BASELINE].iloc[0]
    checks = {}
    for _, r in res.iterrows():
        if r.arm == BASELINE:
            continue
        checks[r.arm] = {
            "mean delta vs constant < 0": bool(r.delta_const < 0),
            "95% CI entirely below 0": bool(r.dc_hi < 0),
            "Jaccard not worse than AIPW-psi": bool(
                r.jaccard >= base.jaccard - TOL_JACCARD),
            ">= 70% of reps better than constant": bool(
                r.frac_better_const >= 70),
        }
    return checks


def secondary_support_sweep(df_jobs, n, reps, jobs, root):
    """Gate 7 support sweep repeated with the winning target (only if adopted)."""
    return None      # implemented after a primary adoption


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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate9_targets.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate9_summary.csv"), index=False)
    checks = decide(res)
    base = res[res.arm == BASELINE].iloc[0]

    print(f"\n=== discovery target, n={args.n}, {args.reps} reps ===")
    print(f"constant MAE = {base.const_mae:.4f};  oracle MAE = {base.oracle_mae:.4f}")
    print(f"{'target':>12}{'tgt sd':>9}{'corr':>7}{'contrast':>10}{'rules':>7}"
          f"{'Jacc':>7}{'stab':>7}{'MAE':>9}{'const':>9}"
          f"{'d vs const':>12}{'95% CI':>20}{'%bet':>6}")
    for _, r in res.iterrows():
        print(f"{r.arm:>12}{r.target_sd:>9.3f}{r.target_corr:>7.2f}"
              f"{r.target_contrast:>10.3f}{r.rules:>7.1f}{r.jaccard:>7.3f}"
              f"{r.stability:>7.2f}{r.mae:>9.4f}{r.const_mae:>9.4f}"
              f"{r.delta_const:>+12.4f}"
              f"   [{r.dc_lo:+.4f},{r.dc_hi:+.4f}]{r.frac_better_const:>5.0f}%")
    print("\npaired difference against the AIPW-psi baseline (negative = better):")
    for _, r in res.iterrows():
        if r.arm == BASELINE:
            continue
        print(f"  {r.arm:>12}: {r.delta_base:+.4f} "
              f"[{r.db_lo:+.4f},{r.db_hi:+.4f}]  "
              f"{r.frac_better_base:.0f}% of reps better")

    print("\nPre-registered adoption checks:")
    adopted = []
    for arm, ch in checks.items():
        print(f"  {arm}:")
        for k, v in ch.items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if all(ch.values()):
            adopted.append(arm)
    print(f"\n-> {'ADOPT ' + ', '.join(adopted) if adopted else 'NO plug-in target is adopted'}")
    print("Secondary analyses (support sweep, MNAR tilt) are only run after a "
          "primary adoption.")
    return df


if __name__ == "__main__":
    main()

"""Gate 10 - conservative rule extraction on the cross-fitted plug-in target.

Pre-registration: `results/gate10_PREREGISTRATION.md`, frozen before this ran.

Frozen design
-------------
    seeds        110100-110179  (Gate 9's 110000-110029 are excluded)
    n            2000
    target       cross-fitted Cox tau_hat = mu_hat_1 - mu_hat_0
    extractor    DecisionTreeRegressor(max_depth=1, min_samples_leaf=60)
    effects      held-out AIPW phi, nuisances fitted on TRAIN only
    default      TRAIN marginal ATE
    comparator   same-replication TRAIN marginal ATE

Nothing from the held-out fold enters the split search, the threshold, the leaf
conditions, the default, or any tuning decision (see the leakage audit in the
pre-registration).

Decision rule (per arm): mean paired delta < 0, 95 % CI entirely below 0,
best-leaf Jaccard >= 0.107, and >= 70 % of replications better.

Usage
-----
    python -m ciscarl2.gate10 --reps 2 --start 110900    # smoke test, off-seed
    python -m ciscarl2.gate10 --reps 80 --start 110100
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights, _fit_predict_nuisances)
from .gate2 import TAU, pseudo_outcome
from .gate4d import EFFECT, MIN_SUPPORT, TSTAR, _delta_for
from .gate8 import assign, best_jaccard, discover
from .rules import extract_leaf_rules

from .paths import data_path

DATA = data_path()
BASELINE_JACCARD = 0.107      # frozen AIPW-psi production baseline (Gate 9)
PRIMARY = "cart_cox_d1"
ROBUST = "cart_rsf_d1"
DESCRIPTIVE = ("cart_cox_d2", "cart_cox_d3", "rf_cox", "smooth_cox")


def _cart(target, XD, depth):
    t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=MIN_SUPPORT,
                              random_state=0).fit(XD, target)
    rules = [dict(cond=c, mask=m, score=1.0)
             for c, m in extract_leaf_rules(t, XD, MIN_SUPPORT)]
    return rules, t


def run_rep(rep, start, cfg):
    pool, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, lam0, lam_c0 = cfg
    seed = start + rep
    rng = np.random.RandomState(seed)
    n = 2000
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
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    _, _, q1, q0 = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const = float((q1 - q0).mean())

    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=seed)
    dphi = p1 - p0

    muR, _, _ = crossfit_nuisances(XD, A[tr], T[tr], E[tr], TSTAR,
                                   seed=seed, mu_model="rsf")
    muH, _ = _fit_predict_nuisances(XD, A[tr], T[tr], E[tr], XE, TSTAR)

    tau_cox = muD[:, 1] - muD[:, 0]
    tau_rsf = muR[:, 1] - muR[:, 0]
    const_mae = float(np.abs(const - truth).mean())

    rows = [dict(rep=rep, arm="constant", n_rules=0, jaccard=np.nan,
                 mae=const_mae, const_mae=const_mae, delta=0.0)]

    specs = [
        ("cart_cox_d1", tau_cox, 1, "cox"),
        ("cart_cox_d2", tau_cox, 2, "cox"),
        ("cart_cox_d3", tau_cox, 3, "cox"),
        ("cart_rsf_d1", tau_rsf, 1, "rsf"),
    ]
    for name, target, depth, which in specs:
        rules, tree = _cart(target, XD, depth)
        hat = assign(rules, XE, dphi, const)
        mae = float(np.abs(hat - truth).mean())
        row = dict(rep=rep, arm=name, n_rules=len(rules),
                   jaccard=best_jaccard(rules, XE, regE), mae=mae,
                   const_mae=const_mae, delta=mae - const_mae)
        if depth == 1:
            f = int(tree.tree_.feature[0])
            thr = float(tree.tree_.threshold[0])
            hi = XE[:, f] > thr
            row.update(split_feat=dgp.COVARIATES[f], split_thr=thr,
                       hi_share=float(hi.mean()),
                       hi_effect=float(dphi[hi].mean()) if hi.any() else np.nan,
                       lo_effect=float(dphi[~hi].mean()) if (~hi).any() else np.nan)
        rows.append(row)

    # descriptive: production RF+stability on the same target
    sel, stab = discover(XD, tau_cox, seed + 1)
    rows.append(dict(rep=rep, arm="rf_cox", n_rules=len(sel),
                     jaccard=best_jaccard(sel, XE, regE),
                     mae=float(np.abs(assign(sel, XE, dphi, const) - truth).mean()),
                     const_mae=const_mae,
                     delta=float(np.abs(assign(sel, XE, dphi, const) - truth).mean()) - const_mae,
                     stability=stab))

    # descriptive: smooth predictor, no rules
    smooth = muH[:, 1] - muH[:, 0]
    rows.append(dict(rep=rep, arm="smooth_cox", n_rules=0, jaccard=np.nan,
                     mae=float(np.abs(smooth - truth).mean()),
                     const_mae=const_mae,
                     delta=float(np.abs(smooth - truth).mean()) - const_mae))

    for r in rows:
        r["n"] = n
    return rows


def _boot(d, seed=0, B=5000):
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for arm, g in df.groupby("arm"):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        out.append(dict(
            arm=arm, reps=len(g), rules=g.n_rules.mean(),
            jaccard=g.jaccard.mean(), mae=g.mae.mean(),
            const_mae=g.const_mae.mean(), delta=d.mean(),
            lo=lo, hi=hi, frac_better=100.0 * float((d < 0).mean()),
            rel_gain=100.0 * (1.0 - g.mae.mean() / g.const_mae.mean())))
    return pd.DataFrame(out)


def decide(res):
    checks = {}
    for _, r in res.iterrows():
        if r.arm == "constant":
            continue
        checks[r.arm] = {
            "mean delta < 0": bool(r.delta < 0),
            "95% CI entirely below 0": bool(r.hi < 0),
            f"Jaccard >= {BASELINE_JACCARD}": bool(
                np.isnan(r.jaccard) or r.jaccard >= BASELINE_JACCARD),
            ">= 70% of reps better": bool(r.frac_better >= 70),
        }
    return checks


def _config():
    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    cd4_cut, age_cut = float(raw["cd40"].median()), float(raw["age"].median())
    i_cd4 = dgp.COVARIATES.index("cd40")
    i_age = dgp.COVARIATES.index("age")
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})
    return (pool, mu, sd, i_cd4, i_age, cd4_cut, age_cut, lam0, lam_c0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--start", type=int, default=110100)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, args.start, cfg) for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate10_replications.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate10_summary.csv"), index=False)
    checks = decide(res)

    print(f"\n=== Gate 10, seeds {args.start}-{args.start + args.reps - 1}, "
          f"n=2000, {args.reps} reps ===")
    print(f"constant MAE = {res[res.arm == 'constant'].const_mae.iloc[0]:.4f}")
    print(f"{'arm':>12}{'rules':>7}{'Jacc':>7}{'MAE':>9}{'const':>9}"
          f"{'delta':>10}{'95% CI':>22}{'%bet':>6}{'gain':>7}")
    order = ["constant", "smooth_cox", PRIMARY, ROBUST] + list(DESCRIPTIVE)
    for arm in order:
        r = res[res.arm == arm].iloc[0]
        print(f"{arm:>12}{r.rules:>7.1f}{r.jaccard:>7.3f}{r.mae:>9.4f}"
              f"{r.const_mae:>9.4f}{r.delta:>+10.4f}"
              f"   [{r.lo:+.4f},{r.hi:+.4f}]{r.frac_better:>5.0f}%"
              f"{r.rel_gain:>+6.1f}%")

    d1 = df[df.arm == PRIMARY]
    print(f"\n{PRIMARY}: split on", d1.split_feat.value_counts().to_dict(),
          "| high-leaf share", round(float(d1.hi_share.mean()), 3),
          "| high-leaf AIPW effect", round(float(d1.hi_effect.mean()), 3),
          "| low-leaf AIPW effect", round(float(d1.lo_effect.mean()), 3))

    print("\nPre-registered decision checks:")
    for arm in (PRIMARY, ROBUST):
        print(f"  {arm}:")
        for k, v in checks[arm].items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    adopted = [a for a in (PRIMARY, ROBUST) if all(checks[a].values())]
    if adopted:
        print(f"\n-> ADOPT {', '.join(adopted)}")
    else:
        print("\n-> No arm adopted; report the loss and do not change thresholds.")
    return df


if __name__ == "__main__":
    main()

"""Gate 4d diagnostic: is the failure in the estimate, the interval, or the decision?

For each replication we record, per discovered rule:
    conditions, test support, fraction of its members inside the TRUE region,
    estimated effect, CI, verdict

and, per test patient in the true region:
    the assigned estimate, whether the assigned interval excludes zero,
    and whether that produced a TREAT verdict

That separates three candidate failures:
  (a) estimates are wrong            -> mean tau_hat in region far from 0.195
  (b) estimates right, too uncertain -> tau_hat near 0.195 but CI includes 0
  (c) representation never found it  -> no rule with high in-region fraction
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, r"C:\Users\roshn\OneDrive\Desktop\Amrita\SEM 5\ML\Project")
from ciscarl2 import dgp
from ciscarl2.estimators import crossfit_nuisances, phi_on_holdout, truncate_weights
from ciscarl2.gate2 import TAU, pseudo_outcome
from ciscarl2.gate4d import EFFECT, MIN_SUPPORT, THR, TSTAR, _delta_for, _verdict
from ciscarl2.rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

from .paths import data_path

DATA = data_path()


def one(rep, n, pool, lam0, lam_c0, mu, sd, i_cd4, i_age, cd4_cut, age_cut):
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
    reg = in_reg[te]
    tt = tau_i[te]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        X[tr], A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=MIN_SUPPORT, random_state=0,
                               n_jobs=-1).fit(X[tr], psi)
    cands = dedup(rules_from_ensemble(rf, X[tr], MIN_SUPPORT, max_trees=30))
    subs = []
    r2 = np.random.RandomState(rep + 1)
    for _ in range(20):
        ii = r2.choice(len(tr), len(tr) // 2, replace=False)
        t_ = RandomForestRegressor(n_estimators=15, max_depth=3,
                                   min_samples_leaf=10, random_state=0,
                                   n_jobs=-1).fit(X[tr][ii], psi[ii])
        subs.append(rules_from_ensemble(t_, X[tr], 5, max_trees=15))
    freq = stability_scores(cands, subs, X[tr]) if cands else np.array([])
    sel = [dict(cond=c, mask=m, score=f) for (c, m), f in zip(cands, freq)
           if f >= THR]
    sel = prune(sel, X[tr], max_rules=99, overlap=0.8) if sel else []

    p1, p0 = phi_on_holdout(X[tr], A[tr], T[tr], E[tr],
                            X[te], A[te], T[te], TSTAR, seed=rep)
    dphi = p1 - p0
    default = float(dphi.mean())
    dse = float(np.sqrt(((dphi - default) ** 2).sum()) / len(te))
    dq = float(tdist.ppf(0.975, len(te) - 1))
    default_v = _verdict(default - dq * dse, default + dq * dse)

    tau_assigned = np.full(len(te), default)
    lo_assigned = np.full(len(te), default - dq * dse)
    hi_assigned = np.full(len(te), default + dq * dse)
    verd = np.full(len(te), default_v, dtype=object)
    assigned = np.zeros(len(te), bool)

    rules = []
    for r in sorted(sel, key=lambda d: -len(d["cond"])):
        m = eval_rule(r["cond"], X[te])
        nr = int(m.sum())
        if nr < 5:
            continue
        t_r = float(dphi[m].mean())
        s_r = float(np.sqrt(((dphi[m] - t_r) ** 2).sum()) / nr)
        q = float(tdist.ppf(0.975, max(nr - 1, 1)))
        v = _verdict(t_r - q * s_r, t_r + q * s_r)
        inreg = float((m & reg).sum() / max(nr, 1))
        rules.append(dict(rep=rep, n=n, cond=" & ".join(
            f"x{f}{op}{th:.1f}" for f, op, th in r["cond"]),
            support=nr, in_region=inreg, stab=r["score"],
            tau_hat=t_r, lo=t_r - q * s_r, hi=t_r + q * s_r, verdict=v,
            true_in_rule=float(tt[m].mean())))
        take = m & ~assigned
        tau_assigned[take] = t_r
        lo_assigned[take] = t_r - q * s_r
        hi_assigned[take] = t_r + q * s_r
        verd[take] = v
        assigned |= take

    # region-level decomposition
    excl0 = (lo_assigned > 0) | (hi_assigned < 0)
    summary = dict(rep=rep, n=n, n_rules=len(rules),
                   region_n=int(reg.sum()),
                   default_tau=default, default_verdict=default_v,
                   region_tau_hat=float(tau_assigned[reg].mean()),
                   region_tau_true=float(tt[reg].mean()),
                   region_abs_err=float(np.abs(tau_assigned[reg] -
                                               tt[reg]).mean()),
                   region_ci_excl0=100 * float(excl0[reg].mean()),
                   region_treat=100 * float((verd[reg] == "TREAT").mean()),
                   region_positive_hat=100 * float((tau_assigned[reg] > 0).mean()),
                   null_tau_hat=float(tau_assigned[~reg].mean()),
                   null_abs_err=float(np.abs(tau_assigned[~reg] - tt[~reg]).mean()),
                   null_ci_excl0=100 * float(excl0[~reg].mean()))
    return summary, rules


def main():
    import pandas as pd_
    pool = dgp.load_covariate_pool(DATA)
    raw = pd_.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    i_cd4, i_age = dgp.COVARIATES.index("cd40"), dgp.COVARIATES.index("age")
    cd4_cut = float(raw["cd40"].median())
    age_cut = float(raw["age"].median())
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})

    for n, reps in ((589, 12), (2000, 12)):
        S, R, shown = [], [], 0
        for rep in range(reps):
            s, r = one(rep, n, pool, lam0, lam_c0, mu, sd, i_cd4, i_age,
                       cd4_cut, age_cut)
            S.append(s)
            R.extend(r)
            if shown < 2 and r:
                shown += 1
                print(f"\n--- n={n} rep={rep}: learned rule list (test set) ---")
                print(f"{'conditions':<34}{'n':>4}{'in-reg':>8}{'stab':>6}"
                      f"{'tau_hat':>9}{'95% CI':>18}{'true':>7}  verdict")
                for q in r:
                    print(f"{q['cond']:<34}{q['support']:>4}{q['in_region']:>8.2f}"
                          f"{q['stab']:>6.2f}{q['tau_hat']:>+9.3f}"
                          f"   [{q['lo']:+.3f},{q['hi']:+.3f}]{q['true_in_rule']:>+7.3f}"
                          f"  {q['verdict']}")
        s = pd.DataFrame(S)
        print(f"\n=== n={n}: {reps} reps ===")
        print(f"  rules discovered per rep : {s.n_rules.mean():.1f}")
        print(f"  default effect / verdict : {s.default_tau.mean():+.3f} / "
              f"{(s.default_verdict == 'TREAT').mean()*100:.0f} % TREAT")
        print(f"  TRUE REGION (tau={s.region_tau_true.mean():.3f}):")
        print(f"     mean tau_hat         : {s.region_tau_hat.mean():+.3f}")
        print(f"     mean |error|         : {s.region_abs_err.mean():.3f}")
        print(f"     CI excludes zero     : {s.region_ci_excl0.mean():.0f} %")
        print(f"     tau_hat > 0          : {s.region_positive_hat.mean():.0f} %")
        print(f"     TREAT verdict        : {s.region_treat.mean():.0f} %")
        print(f"  NULL REGION (tau=0):")
        print(f"     mean tau_hat         : {s.null_tau_hat.mean():+.3f}")
        print(f"     mean |error|         : {s.null_abs_err.mean():.3f}")
        print(f"     CI excludes zero     : {s.null_ci_excl0.mean():.0f} %")
        if R:
            rdf = pd.DataFrame(R)
            print(f"  mean fraction of a rule's members inside the true region: "
                  f"{rdf.in_region.mean():.2f}")
            print(f"  best rule in-region fraction (max over reps): "
                  f"{rdf.groupby('rep').in_region.max().mean():.2f}")


if __name__ == "__main__":
    main()

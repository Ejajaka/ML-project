"""Gate 4d - piecewise-constant DGP vs SCRE.

Definitions and the decision rule are frozen in
`results/gate4d_PREREGISTRATION.md`, written before this ran.

  tau(X) = 0.20 * 1{CD4 < 200 and age > 40}

All metrics are computed on the held-out test patients. CISCaRL assigns each patient
the verdict of the most specific matching rule (default verdict if none matches);
SCRE has no abstention so it is treated as TREAT wherever tau_hat > 0.

Usage
-----
    python -m ciscarl2.gate4d --reps 30 --n 589
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import TAU, pseudo_outcome
from .gate4 import REPO, scre_select_alpha
from .rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

if REPO not in sys.path:
    sys.path.insert(0, REPO)
from scre import SurvivalCausalRuleEnsemble          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
from .paths import data_path

DATA = data_path()
TSTAR = 730.0
THR, MIN_SUPPORT = 0.50, 60
EFFECT = 0.20
CD4_CUT, AGE_CUT = 200.0, 40.0
NEAR_ZERO = 0.01


def _verdict(lo, hi):
    return "TREAT" if lo > 0 else "AVOID" if hi < 0 else "INSUFFICIENT"


def _delta_for(eff, s0):
    """Log-hazard offset giving S1 = S0 + eff (capped for numerical safety)."""
    s1 = np.clip(s0 + eff, 1e-4, 0.995)
    ratio = np.log(s1) / np.log(np.clip(s0, 1e-4, 0.999))
    return -np.log(np.clip(ratio, 1e-6, None))


def run_one(job):
    (rep, lam0, lam_c0, pool, n, mu, sd, i_cd4, i_age,
     cd4_cut, age_cut) = job
    rng = np.random.RandomState(110000 + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    in_region = (X[:, i_cd4] < (cd4_cut - mu[i_cd4]) / sd[i_cd4]) & \
                (X[:, i_age] > (age_cut - mu[i_age]) / sd[i_age])

    # analytic S0 under the base hazard, then the offset that gives +0.20 in-region
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    eff_vec = np.where(in_region, _delta_for(EFFECT, s0), 0.0)
    sim = dgp.generate(X, "covariate", lam_c0, lam0, rng, TSTAR, effect_vec=eff_vec)
    A, T, E = sim["A"], sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=rep, stratify=A)
    reg = in_region[te]
    tt = tau_i[te]

    # ---- CISCaRL ----
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
                                   min_samples_leaf=max(10, MIN_SUPPORT // 2),
                                   random_state=0, n_jobs=-1).fit(X[tr][ii], psi[ii])
        subs.append(rules_from_ensemble(t_, X[tr], 5, max_trees=15))
    freq = stability_scores(cands, subs, X[tr]) if cands else np.array([])
    sel = [dict(cond=c, mask=m, score=f)
           for (c, m), f in zip(cands, freq) if f >= THR]
    sel = prune(sel, X[tr], max_rules=99, overlap=0.8) if sel else []

    p1, p0 = phi_on_holdout(X[tr], A[tr], T[tr], E[tr],
                            X[te], A[te], T[te], TSTAR, seed=rep)
    dphi = p1 - p0
    default = float(dphi.mean())
    dse = float(np.sqrt(((dphi - default) ** 2).sum()) / len(te))
    dq = float(tdist.ppf(0.975, len(te) - 1))
    cis_tau = np.full(len(te), default)
    cis_v = np.full(len(te), _verdict(default - dq * dse, default + dq * dse),
                    dtype=object)
    assigned = np.zeros(len(te), bool)
    for r in sorted(sel, key=lambda d: -len(d["cond"])):
        m = eval_rule(r["cond"], X[te])
        nr = int(m.sum())
        if nr < 5:
            continue
        t_r = float(dphi[m].mean())
        s_r = float(np.sqrt(((dphi[m] - t_r) ** 2).sum()) / nr)
        q = float(tdist.ppf(0.975, max(nr - 1, 1)))
        take = m & ~assigned
        cis_tau[take] = t_r
        cis_v[take] = _verdict(t_r - q * s_r, t_r + q * s_r)
        assigned |= take

    # ---- SCRE ----
    inner, val = train_test_split(np.arange(len(tr)), test_size=0.25,
                                  random_state=rep, stratify=A[tr])
    f, v = tr[inner], tr[val]
    s = SurvivalCausalRuleEnsemble(alpha=0.5, pred_n_estimators=100,
                                   progn_n_estimators=100, rule_min_support=20,
                                   max_rules=200, max_rule_conditions=3,
                                   max_selected_rules=10, random_state=42)
    s.fit(X[f], psi[inner], np.isfinite(psi[inner]),
          feature_names=[f"x{i}" for i in range(X.shape[1])],
          time=T[f], event=E[f].astype(bool), treatment=A[f])
    pick = scre_select_alpha(s, X[v], A[v], T[v], E[v].astype(bool), TSTAR)
    if pick is not None:
        s.alpha_ = s.alpha_final_ = pick[1]
        s.coef_ = pick[2]
    s.t_star_ = TSTAR
    scre_tau = np.asarray(s.predict(X[te]), float)

    def region_metrics(pred_tau, committed, verdicts):
        P = committed & (verdicts == "TREAT") if verdicts is not None else pred_tau > 0
        out = dict(
            region_mae=float(np.abs(pred_tau[reg] - tt[reg]).mean()),
            overall_mae=float(np.abs(pred_tau - tt).mean()),
            sensitivity=float((P & reg).sum() / max(reg.sum(), 1)),
            specificity=float(((~P) & (~reg)).sum() / max((~reg).sum(), 1)),
            false_positive=float((P & (~reg)).sum() / max((~reg).sum(), 1)),
            commitment=100 * float(committed.mean()))
        nz = np.abs(tt) > NEAR_ZERO
        sel_c = committed & nz
        out["precision"] = (100 * float(np.mean(np.sign(pred_tau[sel_c]) ==
                                                np.sign(tt[sel_c])))
                            if sel_c.any() else np.nan)
        return out

    cis_m = region_metrics(cis_tau, cis_v != "INSUFFICIENT", cis_v)
    scr_m = region_metrics(scre_tau, np.ones(len(te), bool), None)

    # ---- constant baseline: marginal ATE estimated on TRAIN only ----
    muT, GT, _, fidT, fcapsT, _ = crossfit_nuisances(
        X[tr], A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GT, _ = truncate_weights(GT, fidT, fcapsT)
    _, _, p1t, p0t = aipw(X[tr], A[tr], T[tr], TSTAR, muT, GT)
    const_ate = float((p1t - p0t).mean())
    const_tau = np.full(len(te), const_ate)
    const_m = region_metrics(const_tau, np.ones(len(te), bool), None)

    # ---- oracle: TRUE region membership, effect estimated from the data ----
    # isolates estimation error from discovery error (perfect discovery)
    tau_in_hat = float(dphi[reg].mean()) if reg.any() else 0.0
    tau_out_hat = float(dphi[~reg].mean()) if (~reg).any() else 0.0
    orc_tau = np.where(reg, tau_in_hat, tau_out_hat)
    orc_v = np.full(len(te), "TREAT" if tau_in_hat > 0 else "AVOID", dtype=object)
    orc_m = region_metrics(orc_tau, np.ones(len(te), bool), None)

    base = dict(rep=rep, n=n, region_share=100 * float(reg.mean()),
                tau_in=float(tt[reg].mean()), tau_out=float(tt[~reg].mean()),
                n_rules=len(sel), const_ate=const_ate,
                tau_in_hat=tau_in_hat, tau_out_hat=tau_out_hat)
    return [dict(base, method="CISCaRL", **cis_m),
            dict(base, method="SCRE", **scr_m),
            dict(base, method="constant", **const_m),
            dict(base, method="oracle_est", **orc_m)]


def summarise(df):
    out = []
    for n, g in df.groupby("n"):
        cis = g[g.method == "CISCaRL"].set_index("rep")
        scr = g[g.method == "SCRE"].set_index("rep")
        common = sorted(set(cis.index) & set(scr.index))
        delta = (scr.loc[common, "region_mae"] - cis.loc[common, "region_mae"]).to_numpy()
        rs = np.random.RandomState(0)
        bs = [rs.choice(delta, len(delta), replace=True).mean() for _ in range(5000)]

        def ci(x):
            x = np.asarray(x)
            b = [rs.choice(x, len(x), replace=True).mean() for _ in range(2000)]
            return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

        row = dict(n=n, reps=len(common),
                   region_share=cis.loc[common, "region_share"].mean(),
                   tau_in=cis.loc[common, "tau_in"].mean(),
                   tau_out=cis.loc[common, "tau_out"].mean(),
                   delta_region_mae=float(delta.mean()),
                   delta_lo=float(np.percentile(bs, 2.5)),
                   delta_hi=float(np.percentile(bs, 97.5)),
                   const_ate=float(g[g.method == "constant"].const_ate.mean()))
        const = g[g.method == "constant"].set_index("rep")
        orc = g[g.method == "oracle_est"].set_index("rep")
        for meth, d_ in (("cis", cis), ("scr", scr), ("const", const),
                         ("orc", orc)):
            d_p = d_.loc[common]
            for k in ("region_mae", "overall_mae", "sensitivity", "specificity",
                      "false_positive", "commitment", "precision"):
                row[f"{meth}_{k}"] = float(d_p[k].mean())
            lo, hi = ci(d_p["sensitivity"].to_numpy())
            row[f"{meth}_sens_lo"], row[f"{meth}_sens_hi"] = lo, hi
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(s):
    L = ["# Gate 4d - piecewise-constant DGP\n",
         "Pre-registered rule (`results/gate4d_PREREGISTRATION.md`): advantage "
         "supported only if Delta region MAE > 0 with CI excluding zero, "
         "sensitivity >= 0.60, and false-positive TREAT <= 0.10.\n",
         "All rows are computed on the SAME held-out test patients from the same "
         "replication. Overall MAE is the mean over those patients, so the 25 % / 75 % "
         "region weighting is implicit and identical everywhere. The constant is the "
         "marginal ATE estimated on TRAIN and applied to every test patient.\n",
         "| n | reps | region | tau in/out | const ATE | **overall MAE: const / "
         "CISCaRL / SCRE / oracle** | region MAE: const / CIS / SCRE / oracle |",
         "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(
            f"| {r.n} | {r.reps} | {r.region_share:.0f} % | "
            f"{r.tau_in:.3f} / {r.tau_out:.3f} | {r.const_ate:+.3f} | "
            f"**{r.const_overall_mae:.4f} / {r.cis_overall_mae:.4f} / "
            f"{r.scr_overall_mae:.4f} / {r.orc_overall_mae:.4f}** | "
            f"{r.const_region_mae:.4f} / {r.cis_region_mae:.4f} / "
            f"{r.scr_region_mae:.4f} / {r.orc_region_mae:.4f} |")
    L.append("")
    L.append("Delta region MAE = MAE(SCRE) - MAE(CISCaRL) on the true region; "
             "positive favours CISCaRL.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n", type=int, default=589)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(DATA)
    mu = np.array([pool[:, j].mean() for j in range(pool.shape[1])])  # standardised -> 0
    sd = np.ones(pool.shape[1])
    # recover the original scale for the clinical cut-points
    import pandas as pd_
    raw = pd_.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    i_cd4 = dgp.COVARIATES.index("cd40")
    i_age = dgp.COVARIATES.index("age")
    cd4_cut = float(raw["cd40"].median())
    age_cut = float(raw["age"].median())

    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})

    from joblib import Parallel, delayed
    t0 = time.time()
    jobs = [(rep, lam0, lam_c0, pool, args.n, mu, sd, i_cd4, i_age,
             cd4_cut, age_cut) for rep in range(args.reps)]
    out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
    df = pd.DataFrame([r for ch in out for r in ch if r])
    tag = f"n{args.n}"
    df.to_csv(os.path.join(RESULTS, f"gate4d_replications_{tag}.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, f"gate4d_summary_{tag}.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, f"gate4d_summary_{tag}.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)
    print(f"\n{args.reps} reps in {time.time()-t0:.0f}s")
    print(md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

"""Gate 8 - does an RMST estimand reduce pseudo-outcome noise enough to help discovery?

Pre-registration: `results/gate8_PREREGISTRATION.md` (written before this ran).

Every step is rebuilt for RMST, not swapped in:

    mu_a(X)   = integral_0^tau  S_a(t|X) dt          (per-arm Cox, cross-fitted)
    Y_i^RMST  = integral_0^{min(T_i,tau)} dt / G(t|X_i)
    phi_a,i   = mu_a(X_i) + 1{A_i=a}/e * (Y_i^RMST - mu_a(X_i))
    psi_i     = phi_1,i - phi_0,i

The identity E[ integral_0^tau 1{t < T}/G(t|X) dt | X ] = integral_0^tau S(t|X) dt
holds under the same conditional independent-censoring assumption as the horizon
estimator, so the two estimands are compared on equal footing. The true RMST
effects are computed analytically from the exponential DGP.

Both estimands get their OWN constant baseline, estimated on TRAIN, and are
evaluated on the same held-out patients. Comparing an RMST model against a
probability-scale baseline would be meaningless.

Pre-registered decision rule (RMST arm only)
--------------------------------------------
Adopt RMST only if ALL of
  1. mean paired delta = MAE(RMST CISCaRL) - MAE(RMST constant) < 0
  2. the 95 % paired bootstrap CI excludes zero
  3. subgroup recovery (best-rule Jaccard) is not worse than 80 % of the
     day-730 value
  4. >= 70 % of replications show delta < 0

Otherwise the day-730 estimand stays and the result is reported as a loss.

Usage
-----
    python -m ciscarl2.gate8 --reps 30 --n 2000
    python -m ciscarl2.gate8 --reps 2 --n 2000          # smoke test
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split

from . import dgp
from .estimators import (W_CAP, _fit_cox, aipw, crossfit_nuisances,
                         phi_on_holdout, step_value, truncate_weights)
from .gate2 import TAU, pseudo_outcome
from .gate4d import EFFECT, MIN_SUPPORT, THR, TSTAR, _delta_for
from .rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

from .paths import data_path

DATA = data_path()
STAB_B = 20
NGRID = 120          # intervals on [0, TSTAR] for the RMST integral
CAP_Q = 0.99         # same weight cap rule as v1.4
CAP_CEIL = min(W_CAP, 20.0)


# ---------------------------------------------------------------------------
# RMST nuisances
# ---------------------------------------------------------------------------
def _cox_model(X, T, D):
    """Fit a CoxPHFitter, or return None if every penaliser fails."""
    df = pd.DataFrame(np.asarray(X, float))
    df["_T"] = np.asarray(T, float)
    df["_D"] = np.asarray(D, int)
    try:
        return _fit_cox(df, "_T", "_D")
    except Exception:                                  # pragma: no cover
        return None


def _cox_curve(cph, Xpred, grid):
    """S(t|X) = S0(t) ^ exp(X beta) evaluated with the step baseline.

    Uses the same right-continuous step convention as estimators._cox_fit.
    The grid must include 0, where S = 1 by construction.
    """
    bh = cph.baseline_survival_
    tv = np.asarray(bh.index.values, float)
    sv = np.asarray(bh.iloc[:, 0].values, float)
    S0 = np.array([step_value(tv, sv, t) for t in grid])
    lp = np.asarray(Xpred, float) @ np.asarray(cph.params_.values, float)
    return np.power(S0[None, :], np.exp(lp)[:, None])


def _trapz_rows(S, grid):
    d = np.diff(grid)
    return 0.5 * (S[:, 1:] + S[:, :-1]) @ d


def _lengths(Tmin, grid):
    """Length of [g_j, g_{j+1}] intersected with [0, Tmin_i] (exact)."""
    lo = grid[None, :-1]
    hi = np.minimum(grid[None, 1:], np.asarray(Tmin, float)[:, None])
    return np.clip(hi - lo, 0.0, None)


def _censoring_cap(Gfit_last, quantile=CAP_Q, ceiling=CAP_CEIL):
    w = 1.0 / np.clip(Gfit_last, 1e-3, 1.0)
    return float(min(np.quantile(w, quantile), ceiling))


def crossfit_rmst(X, A, T, E, tau, seed=0, n_splits=5, ngrid=NGRID,
                  cap_quantile=CAP_Q, cap_ceiling=CAP_CEIL):
    """Cross-fitted RMST nuisances. Returns mu (n,2) and Y_ipcw (n).

    mu[hold, a] comes from a Cox fit on the fitting rows of the fold; the
    censoring model and the weight cap are fold-specific and use fitting rows
    only, mirroring `estimators.crossfit_nuisances` + `truncate_weights`.
    """
    X = np.asarray(X, float)
    T = np.asarray(T, float)
    E = np.asarray(E, int)
    A = np.asarray(A, int)
    n = len(X)
    grid = np.linspace(0.0, tau, ngrid + 1)
    mid = 0.5 * (grid[:-1] + grid[1:])
    mu = np.zeros((n, 2))
    Y = np.zeros(n)
    Tmin = np.minimum(T, tau)
    caps = np.full(n, np.nan)

    for fit, hold in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        for a in (0, 1):
            m = fit[A[fit] == a]
            cph = _cox_model(X[m], T[m], E[m]) if E[m].sum() >= 2 else None
            if cph is None:                            # degenerarte fold
                mu[hold, a] = float(np.mean(Tmin[m])) if len(m) else float(tau)
            else:
                mu[hold, a] = _trapz_rows(_cox_curve(cph, X[hold], grid), grid)

        cphc = _cox_model(X[fit], T[fit], (E[fit] == 0).astype(int))
        if cphc is None:                               # pragma: no cover
            G = np.ones((len(hold), ngrid))
            cap = cap_ceiling
        else:
            Gfit_last = _cox_curve(cphc, X[fit], np.array([0.0, tau]))[:, 1]
            cap = _censoring_cap(Gfit_last, cap_quantile, cap_ceiling)
            G = np.clip(_cox_curve(cphc, X[hold], mid), 1.0 / cap, 1.0)
        Y[hold] = (_lengths(Tmin[hold], grid) / G).sum(axis=1)
        caps[hold] = cap

    return mu, Y, caps


def rmst_on_holdout(Xtr, Atr, Ttr, Etr, Xte, Tte, tau, ngrid=NGRID,
                    cap_quantile=CAP_Q, cap_ceiling=CAP_CEIL):
    """Train-fitted RMST nuisances applied to the held-out patients."""
    Xtr, Ttr, Etr = np.asarray(Xtr, float), np.asarray(Ttr, float), np.asarray(Etr, int)
    Xte = np.asarray(Xte, float)
    grid = np.linspace(0.0, tau, ngrid + 1)
    mid = 0.5 * (grid[:-1] + grid[1:])
    mu = np.zeros((len(Xte), 2))
    for a in (0, 1):
        m = Atr == a
        cph = _cox_model(Xtr[m], Ttr[m], Etr[m]) if Etr[m].sum() >= 2 else None
        if cph is None:                                # pragma: no cover
            mu[:, a] = float(np.mean(np.minimum(Ttr[m], tau)))
        else:
            mu[:, a] = _trapz_rows(_cox_curve(cph, Xte, grid), grid)

    cphc = _cox_model(Xtr, Ttr, (Etr == 0).astype(int))
    if cphc is None:                                   # pragma: no cover
        G = np.ones((len(Xte), ngrid))
    else:
        Gfit_last = _cox_curve(cphc, Xtr, np.array([0.0, tau]))[:, 1]
        cap = _censoring_cap(Gfit_last, cap_quantile, cap_ceiling)
        G = np.clip(_cox_curve(cphc, Xte, mid), 1.0 / cap, 1.0)
    Y = (_lengths(np.minimum(Tte, tau), grid) / G).sum(axis=1)
    return mu, Y


def phi_rmst(mu, Y, A, e=0.5):
    A = np.asarray(A, int)
    phi1 = mu[:, 1] + (A == 1) / e * (Y - mu[:, 1])
    phi0 = mu[:, 0] + (A == 0) / (1.0 - e) * (Y - mu[:, 0])
    return phi1, phi0


def rmst_true(S, tau):
    """RMST to tau for an exponential survival curve with S = S(tau).

    S(t) = exp(-lambda t) with lambda = -log(S)/tau, so
    RMST = (1 - S) / lambda = (1 - S) * tau / (-log S).
    """
    S = np.clip(np.asarray(S, float), 1e-6, 1.0 - 1e-9)
    return (1.0 - S) * tau / (-np.log(S))


# ---------------------------------------------------------------------------
# Discovery and assignment (identical to the production pipeline, Gate 7)
# ---------------------------------------------------------------------------
def discover(Xtr, target, seed):
    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=MIN_SUPPORT, random_state=0,
                               n_jobs=-1).fit(Xtr, target)
    cands = dedup(rules_from_ensemble(rf, Xtr, MIN_SUPPORT, max_trees=30))
    if not cands:
        return [], 0.0
    rng = np.random.RandomState(seed)
    subs = []
    for _ in range(STAB_B):
        ii = rng.choice(len(Xtr), len(Xtr) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, MIN_SUPPORT // 2),
                                  random_state=0, n_jobs=-1).fit(Xtr[ii], target[ii])
        subs.append(rules_from_ensemble(t, Xtr, 5, max_trees=15))
    freq = stability_scores(cands, subs, Xtr)
    sel = [dict(cond=c, mask=m, score=f) for (c, m), f in zip(cands, freq)
           if f >= THR]
    sel = prune(sel, Xtr, max_rules=99, overlap=0.8) if sel else []
    return sel, float(np.max(freq)) if len(freq) else 0.0


def assign(sel, XE, dphi, default):
    tau = np.full(len(XE), default)
    assigned = np.zeros(len(XE), bool)
    for r in sorted(sel, key=lambda d: -len(d["cond"])):
        m = eval_rule(r["cond"], XE)
        if int(m.sum()) < 5:
            continue
        take = m & ~assigned
        tau[take] = float(dphi[m].mean())
        assigned |= take
    return tau


def best_jaccard(sel, XE, regE):
    jac = 0.0
    for r in sel:
        m = eval_rule(r["cond"], XE)
        jac = max(jac, float((m & regE).sum() / max((m | regE).sum(), 1)))
    return jac


def oracle_mae(dphi, regE, truth):
    tin = float(dphi[regE].mean()) if regE.any() else 0.0
    tout = float(dphi[~regE].mean()) if (~regE).any() else 0.0
    return float(np.abs(np.where(regE, tin, tout) - truth).mean())


# ---------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------
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

    tau_prob = sim["S1"] - sim["S0"]                       # probability scale
    tau_rmst = rmst_true(sim["S1"], TSTAR) - rmst_true(sim["S0"], TSTAR)  # days

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=rep, stratify=A)
    XD, XE = X[tr], X[te]
    regD, regE = in_reg[tr], in_reg[te]
    truth_prob, truth_rmst = tau_prob[te], tau_rmst[te]

    rows = []

    def add(est, method, mae, n_rules=0, jac=0.0, stab=0.0,
            psi_sd=np.nan, contrast=np.nan):
        rows.append(dict(rep=rep, n=n, estimand=est, method=method,
                         n_rules=n_rules, jaccard=jac, stability=stab,
                         psi_sd=psi_sd, tau_contrast=contrast,
                         snr=abs(contrast) / psi_sd if psi_sd and psi_sd > 0
                         else np.nan, mae=float(mae)))

    # ================= day-730 survival probability (current) =================
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi_p = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)
    _, _, q1, q0 = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const_p = float((q1 - q0).mean())
    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=rep)
    dphi_p = p1 - p0

    sel_p, stab_p = discover(XD, psi_p, rep + 1)
    hat_p = assign(sel_p, XE, dphi_p, const_p)
    contrast_p = (float(tau_prob[tr][regD].mean() - tau_prob[tr][~regD].mean())
                  if regD.any() and (~regD).any() else np.nan)
    add("prob_730", "ciscarl", np.abs(hat_p - truth_prob).mean(),
        len(sel_p), best_jaccard(sel_p, XE, regE), stab_p,
        float(np.std(psi_p)), contrast_p)
    add("prob_730", "constant", np.abs(const_p - truth_prob).mean())
    add("prob_730", "oracle", oracle_mae(dphi_p, regE, truth_prob))

    # ============================ RMST to 730 days ============================
    muR, YR, caps = crossfit_rmst(XD, A[tr], T[tr], E[tr], TSTAR, seed=rep)
    r1, r0 = phi_rmst(muR, YR, A[tr])
    psi_r = r1 - r0
    const_r = float(psi_r.mean())

    m_h, Y_h = rmst_on_holdout(XD, A[tr], T[tr], E[tr], XE, T[te], TSTAR)
    e1, e0 = phi_rmst(m_h, Y_h, A[te])
    dphi_r = e1 - e0

    sel_r, stab_r = discover(XD, psi_r, rep + 1)
    hat_r = assign(sel_r, XE, dphi_r, const_r)
    contrast_r = (float(tau_rmst[tr][regD].mean() - tau_rmst[tr][~regD].mean())
                  if regD.any() and (~regD).any() else np.nan)
    add("rmst_730", "ciscarl", np.abs(hat_r - truth_rmst).mean(),
        len(sel_r), best_jaccard(sel_r, XE, regE), stab_r,
        float(np.std(psi_r)), contrast_r)
    add("rmst_730", "constant", np.abs(const_r - truth_rmst).mean())
    add("rmst_730", "oracle", oracle_mae(dphi_r, regE, truth_rmst))

    for r in rows:
        r["region_share"] = 100.0 * float(regE.mean())
        r["cap_median"] = float(np.median(caps)) if caps.size else np.nan
    return rows


# ---------------------------------------------------------------------------
# Summary + pre-registered decision
# ---------------------------------------------------------------------------
def summarise(df):
    out = []
    for est, g in df.groupby("estimand"):
        cis = g[g.method == "ciscarl"].set_index("rep")
        con = g[g.method == "constant"].set_index("rep")
        orc = g[g.method == "oracle"].set_index("rep")
        d = (cis.mae - con.mae).to_numpy()
        rs = np.random.RandomState(0)
        bs = np.array([rs.choice(d, len(d), replace=True).mean()
                       for _ in range(5000)])
        out.append(dict(
            estimand=est, reps=len(d),
            psi_sd=cis.psi_sd.mean(), tau_contrast=cis.tau_contrast.mean(),
            snr=cis.snr.mean(), rules=cis.n_rules.mean(),
            jaccard=cis.jaccard.mean(), stability=cis.stability.mean(),
            mae=cis.mae.mean(), const_mae=con.mae.mean(),
            oracle_mae=orc.mae.mean(), delta=d.mean(),
            delta_lo=np.percentile(bs, 2.5), delta_hi=np.percentile(bs, 97.5),
            frac_better=100.0 * float((d < 0).mean()),
            rel_gain=100.0 * (1.0 - cis.mae.mean() / con.mae.mean())))
    return pd.DataFrame(out)


def decide(res):
    r = res[res.estimand == "rmst_730"].iloc[0]
    p = res[res.estimand == "prob_730"].iloc[0]
    checks = {
        "delta < 0": bool(r.delta < 0),
        "95% CI excludes 0": bool(r.delta_hi < 0),
        "recovery not worse than 80% of prob": bool(
            p.jaccard <= 0 or r.jaccard >= 0.8 * p.jaccard),
        ">= 70% of reps better": bool(r.frac_better >= 70),
    }
    return checks


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
    df.to_csv(os.path.join(root, "results", "gate8_rmst.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate8_summary.csv"), index=False)
    checks = decide(res)

    print(f"\n=== RMST vs day-730 probability, n={args.n}, {args.reps} reps ===")
    print(f"{'estimand':>10}{'psi sd':>9}{'signal':>9}{'SNR':>7}{'rules':>7}"
          f"{'Jacc':>7}{'stab':>7}{'MAE':>10}{'const':>10}{'oracle':>9}"
          f"{'delta':>10}{'95% CI':>20}{'%bet':>6}{'gain':>7}")
    for _, r in res.iterrows():
        print(f"{r.estimand:>10}{r.psi_sd:>9.2f}{r.tau_contrast:>9.1f}"
              f"{r.snr:>7.2f}{r.rules:>7.1f}{r.jaccard:>7.3f}{r.stability:>7.2f}"
              f"{r.mae:>10.3f}{r.const_mae:>10.3f}{r.oracle_mae:>9.3f}"
              f"{r.delta:>+10.3f}   [{r.delta_lo:+.3f},{r.delta_hi:+.3f}]"
              f"{r.frac_better:>5.0f}%{r.rel_gain:>+6.1f}%")
    print("\ndelta = MAE(CISCaRL) - MAE(constant), per estimand; negative favours"
          " CISCaRL. MAE is in the native units of each estimand (probability /"
          " days), so columns are comparable only within a row-block.")
    print("\nPre-registered RMST adoption checks:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print(f"\n-> {'ADOPT RMST as the discovery estimand' if ok else 'Do NOT adopt; keep day-730 survival probability'}")
    return df


if __name__ == "__main__":
    main()

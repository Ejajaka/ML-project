"""Gate 5 - subgroup-level MNAR sensitivity with verdict-change tipping points.

Gate 3 validated the censoring-weight tilt for the marginal ATE only. CISCaRL's
output is per rule, so the sensitivity analysis has to be per rule, and the
quantity of interest is not where the point estimate moves but **where the
verdict changes**.

For each rule r and censoring tilt delta (G_eff = G ** exp(delta)):

    tau_r(delta), CI_r(delta), verdict_r(delta)

and the tipping points, defined as the smallest |delta| at which the verdict
differs from baseline, reported separately by transition:

    TREAT -> INSUFFICIENT
    AVOID -> INSUFFICIENT
    TREAT -> AVOID  (or AVOID -> TREAT)

delta = 0 reproduces the primary analysis exactly.

Usage
-----
    python -m ciscarl2.gate5                 # real ACTG175 analysis
    python -m ciscarl2.gate5 --simulate      # synthetic check with known truth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (_fit_predict_nuisances, crossfit_nuisances,
                         ipcw_outcome, truncate_weights)
from .gate2 import TAU, pseudo_outcome
from .gate4d import MIN_SUPPORT, TSTAR, _delta_for, _verdict
from .rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

from .paths import data_path

DATA = data_path()
FEATS = dgp.COVARIATES
THR = 0.50
GRID = np.round(np.arange(-2.0, 2.001, 0.05), 3)
CAP_Q, CAP_CEIL = 0.99, 20.0


def fit_and_discover(Xtr, A_tr, T_tr, E_tr, feats, seed=2024, stab_b=20):
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        Xtr, A_tr, T_tr, E_tr, TSTAR, seed=seed, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A_tr, T_tr, TSTAR, muD, GD)
    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=MIN_SUPPORT, random_state=0,
                               n_jobs=-1).fit(Xtr, psi)
    cands = dedup(rules_from_ensemble(rf, Xtr, MIN_SUPPORT, max_trees=30))
    if not cands:
        return []
    subs = []
    rng = np.random.RandomState(seed)
    for _ in range(stab_b):
        ii = rng.choice(len(Xtr), len(Xtr) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, MIN_SUPPORT // 2),
                                  random_state=0, n_jobs=-1).fit(Xtr[ii], psi[ii])
        subs.append(rules_from_ensemble(t, Xtr, 5, max_trees=15))
    freq = stability_scores(cands, subs, Xtr)
    sel = [dict(cond=c, mask=m, score=f) for (c, m), f in zip(cands, freq)
           if f >= THR]
    return prune(sel, Xtr, max_rules=99, overlap=0.8) if sel else []


def curves(mu, G, A, T, t_star, mask, cap):
    """tau(delta), CI(delta), verdict(delta) for one rule over the delta grid.

    The tilt is applied WITHIN the production weight cap: the same
    min(Q_0.99, 20) truncation that the primary estimator uses. Without it, delta>0
    inflates the weights without bound and the estimate explodes (tau = 4-13),
    which would make the tipping point an artefact of variance rather than a
    sensitivity result.
    """
    out = []
    nr = int(mask.sum())
    for delta in GRID:
        Gs = np.maximum(G ** np.exp(delta), 1.0 / cap)
        Y = ipcw_outcome(T, t_star, Gs)
        phi1 = mu[:, 1] + (A == 1) / 0.5 * (Y - mu[:, 1])
        phi0 = mu[:, 0] + (A == 0) / 0.5 * (Y - mu[:, 0])
        d = (phi1 - phi0)[mask]
        tau = float(d.mean())
        se = float(np.sqrt(((d - tau) ** 2).sum()) / nr)
        q = float(tdist.ppf(0.975, max(nr - 1, 1)))
        out.append((float(delta), tau, tau - q * se, tau + q * se,
                    _verdict(tau - q * se, tau + q * se)))
    return out


def tipping(curve):
    """Smallest |delta| at which the verdict differs from delta = 0."""
    base = curve[len(GRID) // 2][4]          # delta = 0.0
    for d, tau, lo, hi, v in curve:
        if v != base:
            return base, v, d
    return base, base, np.nan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.simulate:
        pool = dgp.load_covariate_pool(DATA)
        rng = np.random.RandomState(300000 + args.seed)
        X = pool[rng.choice(len(pool), args.n, replace=True)]
        lp0 = X @ dgp.BETA
        s0 = np.exp(-0.00033 * TSTAR * np.exp(lp0))
        sim = dgp.generate(X, "covariate", 0.00015, 0.000333, rng, TSTAR, tau=TAU)
        A, T, E = sim["A"], sim["Tobs"], sim["E"]
        feats = FEATS
        label = f"SIMULATED (n={args.n}, seed={args.seed}, tau={sim['tau_true']:.3f})"
    else:
        raw = pd.read_csv(DATA)
        raw = raw[raw.arms.isin([0, 1])].reset_index(drop=True)
        Xr = raw[FEATS].to_numpy(float)
        mu_, sd_ = Xr.mean(0), Xr.std(0)
        X = (Xr - mu_) / sd_
        A = (raw.arms == 1).to_numpy(int)
        T = raw.days.to_numpy(float)
        E = raw.cens.to_numpy(int)
        feats = FEATS
        label = "ACTG175 (progression-free composite, t*=730)"

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=2024, stratify=A)
    print(f"\n{label}")
    print(f"discovery n={len(tr)}  evaluation n={len(te)}")

    sel = fit_and_discover(X[tr], A[tr], T[tr], E[tr], feats)
    if not sel:
        print("no rules discovered"); return
    mu, G = _fit_predict_nuisances(X[tr], A[tr], T[tr], E[tr], X[te], TSTAR)
    w_atrisk = np.where(T[te] > TSTAR, 1.0 / G, 0.0)
    w_atrisk = w_atrisk[w_atrisk > 0]
    cap = min(float(np.quantile(w_atrisk, CAP_Q)), CAP_CEIL) if w_atrisk.size else CAP_CEIL
    print(f"weight cap applied to every tilt: {cap:.2f}")

    print(f"\n{'rule':<34}{'n':>4}{'verdict':>20}{'tau(0)':>8}"
          f"{'tau(-2)':>9}{'tau(+2)':>9}{'tip |d|':>10}  transition")
    print("-" * 100)
    rows = []
    for r in sel:
        m = eval_rule(r["cond"], X[te])
        if m.sum() < 10:
            continue
        cv = curves(mu, G, A[te], T[te], TSTAR, m, cap)
        b, v, d = tipping(cv)
        if args.simulate:
            txt = " & ".join(f"{feats[f]} {o} {thr:.1f}" for f, o, thr in r["cond"])
        else:
            txt = " & ".join(f"{feats[f]} {o} {thr*sd_[f]+mu_[f]:.1f}"
                             for f, o, thr in r["cond"])
        tau_m3 = cv[0][1]; tau_0 = cv[len(GRID)//2][1]; tau_p3 = cv[-1][1]
        print(f"{txt:<34}{int(m.sum()):>4}{b:>20}{tau_0:>8.3f}{tau_m3:>9.3f}"
              f"{tau_p3:>9.3f}{('%.2f' % d) if np.isfinite(d) else '> 2.0':>10}  {b} -> {v}")
        rows.append(dict(rule=txt, n=int(m.sum()), base=b, tip_v=v, tip_abs=d))

    pd.DataFrame(rows).to_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "gate5_tipping.csv"), index=False)
    print("\ntipping |delta| = smallest |delta| at which the verdict changes from "
          "its delta=0 value.\nA tick means the verdict is stable over the whole "
          "grid; '> 3.0' means no change within the evaluated range.")


if __name__ == "__main__":
    main()

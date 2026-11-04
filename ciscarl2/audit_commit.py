"""Audit - is Gate 19's commitment decision in-sample?

Gate 19 estimates the high leaf's AIPW effect and SE on the TEST patients, uses
that same estimate to decide whether to commit, and evaluates the decision on
the same patients. If the decision is in-sample, the null-leaf rate and
efficiency are optimistic.

This script compares:
    full-sample : decide on all test patients, assess on all test patients
    honest split: decide on a random half of test, assess on the other half

Run:  python -m ciscarl2.audit_commit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import crossfit_nuisances, phi_on_holdout, truncate_weights
from .gate2 import pseudo_outcome
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate11 import _cart
from .gate14 import _calibrate, _config, _region, _simulate
from .gate19 import _leaf_table, _phi_marginal
from .rules import eval_rule

START = 220100
CELLS = ((3.0, "marginal"),)


def _if(phi, mask):
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan
    e = float(phi[mask].mean())
    se = float(np.sqrt(((phi[mask] - e) ** 2).sum()) / n)
    return e, se


def run_rep(rep, kappa, cen, cfg, lam_c0):
    pool, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, lam0 = cfg
    seed = START + rep
    rng = np.random.RandomState(seed)
    n = 8000
    X = pool[rng.choice(len(pool), n, replace=True)]
    R = _region(X, cfg)
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    eff = np.where(R, _delta_for(EFFECT, s0), 0.0)
    A = rng.binomial(1, 0.5, n)
    sim = _simulate(X, A, lam_c0, lam0, rng, eff, R, kappa, cfg)
    T, E = sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    tr, te = train_test_split(np.arange(n), test_size=0.30,
                              random_state=seed, stratify=A)
    XD, XE = X[tr], X[te]
    AD, TD, ED = A[tr], T[tr], E[tr]
    AE, TE = A[te], T[te]
    truth = tau_i[te]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    if cen == "conditional":
        G = truncate_weights(GD, fid, fcaps)[0]
        p1, p0 = phi_on_holdout(XD, AD, TD, ED, XE, AE, TE, TSTAR, seed=seed)
    else:
        caps = {}
        for k in np.unique(fid):
            idx = np.where(fid == k)[0]
            ar = idx[TD[idx] > TSTAR]
            caps[int(k)] = {0.99: float(np.quantile(1.0 / Gmarg[ar], 0.99))
                            if ar.size > 5 else np.inf}
        G = truncate_weights(Gmarg, fid, caps)[0]
        p1, p0 = _phi_marginal(XD, AD, TD, ED, XE, AE, TE)
    dphi = p1 - p0
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
    leaves = _cart(psi, XD, 3)[1]
    j_hi = max(range(len(leaves)),
               key=lambda j: float(psi[leaves[j]["mask"]].mean()))
    hi_cond = leaves[j_hi]["cond"]

    rows = []
    # full-sample decision
    hi_all = eval_rule(hi_cond, XE)
    e_all, se_all = _if(dphi, hi_all)
    commit_full = bool(np.isfinite(e_all) and e_all > norm.ppf(0.95) * se_all)
    rows.append(_metrics("full", commit_full, hi_all, truth, dphi, e_all))

    # honest split: decide on half, assess on the other half
    half = np.random.RandomState(seed + 1).permutation(len(XE))
    dec, asm = half[: len(half) // 2], half[len(half) // 2:]
    hi_dec = hi_all[dec]
    e_dec, se_dec = _if(dphi[dec], hi_dec)
    commit_split = bool(np.isfinite(e_dec) and e_dec > norm.ppf(0.95) * se_dec)
    hi_asm = hi_all[asm]
    rows.append(_metrics("split", commit_split, hi_asm, truth[asm],
                         dphi[asm], e_dec))
    return rows


def _metrics(kind, committed, mask, truth, dphi, e_used):
    k = int(mask.sum())
    if committed and k:
        per = float(truth[mask].mean())
        oracle = float(np.sort(truth)[::-1][:k].mean())
        capture = float(truth[mask].sum() / truth.sum()) if truth.sum() > 0 else np.nan
        rows = dict(kind=kind, committed=True, coverage=float(mask.mean()),
                    per_treated=per, oracle=oracle,
                    efficiency=per / oracle if oracle else np.nan,
                    captured=capture, null=bool(per <= 0.01),
                    beneficial=bool(per > 0), e_used=float(e_used))
    else:
        rows = dict(kind=kind, committed=False, coverage=0.0, per_treated=np.nan,
                    oracle=np.nan, efficiency=np.nan, captured=0.0,
                    null=False, beneficial=False, e_used=float(e_used))
    return rows


def main():
    cfg = _config()
    lam_c = {k: _calibrate(cfg, k) for k, _ in CELLS}
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=6)(
        delayed(run_rep)(rep, k, c, cfg, lam_c[k])
        for k, c in CELLS for rep in range(20))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    print("=== commitment decision audit (joint cell, 20 reps) ===")
    print(f"{'decision':>10}{'commit%':>9}{'coverage':>10}{'per-treated':>13}"
          f"{'efficiency':>11}{'captured':>10}{'null%':>7}{'benef%':>8}")
    for kind, g in df.groupby("kind"):
        tr = g[g.committed]
        print(f"{kind:>10}{100*g.committed.mean():>8.0f}%"
              f"{g.coverage.mean():>10.3f}{tr.per_treated.mean():>+13.3f}"
              f"{tr.efficiency.mean():>11.2f}{g.captured.mean():>10.3f}"
              f"{100*tr.null.mean() if len(tr) else np.nan:>7.0f}"
              f"{100*tr.beneficial.mean() if len(tr) else np.nan:>8.0f}")
    df.to_csv("results/audit_commitment.csv", index=False)
    print("\nIf the split row has a higher null rate / lower efficiency, the "
          "full-sample commitment is optimistic.")
    return df


if __name__ == "__main__":
    main()

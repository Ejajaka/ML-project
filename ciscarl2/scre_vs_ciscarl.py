"""Compare SCRE's rule table with CISCaRL/C200 predictions on ACTG175.

Two outputs:
  (A) C200 leaf table: every CART leaf with discovery support, mean psi,
      validation effect/SE/LCB, assessment effect/CI, commitment.
  (B) SCRE's 16 rules re-estimated with our AIPW tau(730): HR versus tau.

Descriptive only: the SCRE rule thresholds are transcribed from its published
table; our tau is computed on ACTG175 with cross-fitted nuisances.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import pseudo_outcome
from .gate11 import _cart
from .gate13 import _holdout
from .gate22 import _stats
from .rules import eval_rule

from .paths import data_path

DATA = data_path()
TSTAR = 730.0
SEED = 2024

SCRE_RULES = [
    ("wtkg>=80.63 & cd80>=1524", 0.21, 0.04, 100.00),
    ("cd40<=266.5 & age<=39.5", 0.72, 0.14, 39.05),
    ("cd40>=298.5 & age>=41.5", 0.72, 0.14, 36.93),
    ("homo<=0.5 & race<=0.5 & cd40<359", 0.71, 0.11, 34.63),
    ("cd80>=817 & age>=48.5", 0.71, 0.04, 32.54),
    ("cd40<=268 & age<39", 0.75, 0.13, 32.42),
    ("symptom<=0.5 & cd80>=790.5 & age>=25", 0.83, 0.46, 30.87),
    ("wtkg<97.26 & cd80>=814 & cd40>=335", 0.82, 0.30, 30.12),
    ("symptom<=0.5 & cd80>=790.5 & age>=28.5", 0.86, 0.39, 24.82),
    ("homo<=0.5 & cd40>=380.5 & cd80>=1353", 0.74, 0.05, 22.44),
    ("wtkg>=75.05 & karnof>=95", 0.88, 0.29, 18.88),
    ("str2<=0.5 & wtkg>=78.25 & cd40<322.5", 0.87, 0.07, 11.27),
    ("cd40<268 & karnof>=85 & cd80>=647", 0.93, 0.12, 7.87),
    ("wtkg<97.26 & cd80>=814", 0.96, 0.55, 6.85),
    ("karnof>=85", 0.91, 0.96, 6.33),
    ("cd40<415.5", 1.00, 0.78, 0.17),
]


def load():
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])].reset_index(drop=True)
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    sd[sd == 0] = 1.0
    X = (Xr - mu) / sd
    return dict(X=X, Xr=Xr, mu=mu, sd=sd,
                A=(raw.arms == 1).to_numpy(int),
                T=raw.days.to_numpy(float), E=raw.cens.to_numpy(int))


def parse(rule):
    out = []
    for part in rule.split("&"):
        m = re.match(r"\s*(\w+)\s*(>=|<=|>|<)\s*([0-9.]+)", part)
        if not m:
            continue
        out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


def mask_rule(rule, d):
    X, mu, sd = d["X"], d["mu"], d["sd"]
    m = np.ones(len(X), bool)
    for name, op, v in parse(rule):
        j = dgp.COVARIATES.index(name)
        z = (v - mu[j]) / sd[j]
        if op == ">=":
            m &= X[:, j] >= z
        elif op == "<=":
            m &= X[:, j] <= z
        elif op == ">":
            m &= X[:, j] > z
        else:
            m &= X[:, j] < z
    return m


def ci(phi, mask):
    x = phi[mask]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan, n
    e = float(x.mean())
    se = float(np.sqrt(((x - e) ** 2).sum()) / n)
    q = float(tdist.ppf(0.975, n - 1))
    return e, e - q * se, e + q * se, n


def main():
    d = load()
    X, A, T, E = d["X"], d["A"], d["T"], d["E"]

    print("=" * 92)
    print("A. C200 leaf table on ACTG175 (527 discovery / 263 validation / 264 assessment)")
    print("=" * 92)
    idx = np.arange(len(X))
    disc, hold = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=A)
    val, asm = train_test_split(hold, test_size=0.5, random_state=SEED + 1,
                                stratify=A[hold])
    XD, AD, TD, ED = X[disc], A[disc], T[disc], E[disc]
    XV, AV, TV, EV = X[val], A[val], T[val], E[val]
    XA, AA, TA, EA = X[asm], A[asm], T[asm], E[asm]
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=SEED, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, GD)
    leaves = _cart(psi, XD, 3)[1]
    means, ses, sup = _stats(psi, leaves)
    j_sel = int(np.nanargmax(np.where(sup >= 200, means, -np.inf))) \
        if (sup >= 200).any() else int(np.nanargmax(means))

    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=SEED)
    dphi_v = p1v - p0v
    p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, TSTAR, seed=SEED)
    dphi_a = p1a - p0a

    print(f"  {'#':<3}{'rule':<48}{'disc':>5}{'psi':>7}{'val n':>6}"
          f"{'val tau':>8}{'LCB':>7}{'comm':>6}{'asm n':>6}{'asm tau':>8}"
          f"{'95% CI':>18}")
    print("  " + "-" * 112)
    for j, r in enumerate(leaves):
        cond = r["cond"]
        txt = " & ".join(f"{dgp.COVARIATES[f]} {op} {thr*d['sd'][f]+d['mu'][f]:.1f}"
                         for f, op, thr in cond)
        mv = eval_rule(cond, XV)
        ev, lo, hi, nv = ci(dphi_v, mv)
        lcb = (ev - 1.645 * (hi - ev) / 1.96) if np.isfinite(ev) else np.nan
        comm = "YES" if (np.isfinite(lcb) and lcb > 0) else "no"
        ma = eval_rule(cond, XA)
        ea, loa, hia, na = ci(dphi_a, ma)
        mark = "*" if j == j_sel else " "
        print(f" {mark}{j+1:<3}{txt:<48}{int(sup[j]):>5}{means[j]:>+7.3f}{nv:>6}"
              f"{ev:>+8.3f}{lcb:>+7.3f}{comm:>6}{na:>6}{ea:>+8.3f}"
              f"   [{loa:+.3f},{hia:+.3f}]")
    print("  * selected by C200 (max discovery mean among leaves with support >=200)")
    print(f"  support >=200 met: {'yes' if (sup>=200).any() else 'NO (fallback)'}")

    print()
    print("=" * 92)
    print("B. SCRE's 16 rules re-estimated with our AIPW tau(730) on ACTG175")
    print("=" * 92)
    mu, GD, _, fid, fcaps, _ = crossfit_nuisances(
        X, A, T, E, TSTAR, seed=SEED, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    _, _, p1, p0 = aipw(X, A, T, TSTAR, mu, GD)
    phi = p1 - p0
    print(f"  {'#':<3}{'SCRE rule':<48}{'HR':>6}{'supp':>6}{'n':>5}"
          f"{'our tau':>9}{'95% CI':>20}{'verdict':>13}")
    print("  " + "-" * 112)
    for i, (rule, hr, supp, imp) in enumerate(SCRE_RULES, 1):
        m = mask_rule(rule, d)
        e, lo, hi, n = ci(phi, m)
        v = "TREAT" if lo > 0 else "AVOID" if hi < 0 else "INSUFFICIENT"
        print(f"  {i:<3}{rule:<48}{hr:>6.2f}{supp:>6.2f}{n:>5}"
              f"{e:>+9.3f}   [{lo:+.3f},{hi:+.3f}]{v:>13}")
    print("\n  HR = SCRE's hazard ratio (from its table); tau = our 730-day")
    print("  survival-probability contrast. Different scales; not directly")
    print("  comparable. Descriptive re-analysis, not a head-to-head benchmark.")


if __name__ == "__main__":
    main()

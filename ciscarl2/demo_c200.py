"""Final C200 illustration on ACTG175 (not evidence of clinical benefit).

Discovery 50 % / validation 25 % / assessment 25 %. The rule is selected on
discovery, committed on validation, and evaluated on the untouched assessment
partition. The true subgroup is unknown in real data; this is an output-format
illustration.

Run:  python -m ciscarl2.demo_c200
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (crossfit_nuisances, ipcw_outcome, phi_on_holdout,
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
SUP = 200


def _verdict(lo, hi):
    return "TREAT" if lo > 0 else "AVOID" if hi < 0 else "INSUFFICIENT"


def _ci(phi, mask, q=0.975):
    m = phi[mask]
    n = len(m)
    if n < 5:
        return np.nan, np.nan, np.nan, n
    t = float(m.mean())
    se = float(np.sqrt(((m - t) ** 2).sum()) / n)
    qq = float(tdist.ppf(q, n - 1))
    return t, t - qq * se, t + qq * se, n


def main():
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])].reset_index(drop=True)
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    sd[sd == 0] = 1.0
    X = (Xr - mu) / sd
    A = (raw.arms == 1).to_numpy(int)
    T = raw.days.to_numpy(float)
    E = raw.cens.to_numpy(int)

    idx = np.arange(len(X))
    disc, hold = train_test_split(idx, test_size=0.5, random_state=SEED,
                                  stratify=A)
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
    ok = sup >= SUP
    j = int(np.nanargmax(np.where(ok, means, -np.inf))) if ok.any() \
        else int(np.nanargmax(means))
    cond = leaves[j]["cond"]
    rule = " & ".join(f"{dgp.COVARIATES[f]} {op} {thr * sd[f] + mu[f]:.1f}"
                      for f, op, thr in cond)

    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=SEED)
    dphi_v = p1v - p0v
    p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, TSTAR, seed=SEED)
    dphi_a = p1a - p0a

    t_r, lo_r, hi_r, n_v = _ci(dphi_v, eval_rule(cond, XV))
    commit = bool(np.isfinite(t_r) and t_r > norm.ppf(0.95) *
                  (float(np.sqrt(((dphi_v[eval_rule(cond, XV)] - t_r) ** 2).sum()) /
                         max(n_v, 1))))
    ta, loa, hia, n_a = _ci(dphi_a, eval_rule(cond, XA))
    tall, loall, hiall, nall = _ci(dphi_a, np.ones(len(XA), bool))

    print("\nCISCaRL C200  |  ACTG175 arms 0 vs 1")
    print("=" * 78)
    print("Endpoint : progression-free (CD4 decline >=50, AIDS progression, or death)")
    print("Horizon  : 730 days; treatment 0 = AZT, 1 = AZT + ddI")
    print(f"Partition: discovery {len(disc)} / validation {len(val)} / "
          f"assessment {len(asm)}")
    print("\nOVERALL (assessment partition)")
    print(f"  tau(730) = {tall:+.3f}  95% CI [{loall:+.3f}, {hiall:+.3f}]  "
          f"-> {_verdict(loall, hiall)}  (n={nall})")
    print("\nSELECTED RULE (C200: max discovery mean, support >=200)")
    print(f"  {rule}")
    if ok.any():
        print(f"  support constraint >=200: met (largest eligible leaf "
              f"{int(sup[ok].max())})")
    else:
        print(f"  support constraint >=200: NOT MET on discovery "
              f"(max leaf support {int(sup.max())}); fell back to max mean")
    print(f"  discovery support {int(sup[j])} / {len(XD)}   "
          f"validation n {n_v}   assessment n {n_a}")
    print(f"  validation effect {t_r:+.3f}  -> commitment "
          f"{'YES' if commit else 'NO'}")
    print(f"  assessment effect {ta:+.3f}  95% CI [{loa:+.3f}, {hia:+.3f}]  "
          f"-> {_verdict(loa, hia)}")
    if not commit:
        print("  DECISION: INSUFFICIENT EVIDENCE (no treatment recommendation)")

    # delta-tilt sensitivity on the assessment rule effect
    mu_h, G_h, cap = _holdout(XD, AD, TD, ED, XA, SEED)
    print("\nSENSITIVITY (censoring-weight tilt, rule effect on assessment)")
    print(f"  {'delta':>6}{'tau':>9}{'95% CI':>22}")
    for delta in (-2.0, -1.0, 0.0, 1.0, 2.0):
        Gt = np.maximum(G_h ** np.exp(delta), 1.0 / cap)
        Y = ipcw_outcome(TA, TSTAR, Gt)
        p1 = mu_h[:, 1] + (AA == 1) / 0.5 * (Y - mu_h[:, 1])
        p0 = mu_h[:, 0] + (AA == 0) / 0.5 * (Y - mu_h[:, 0])
        d = p1 - p0
        t, lo, hi, _ = _ci(d, eval_rule(cond, XA))
        print(f"  {delta:>+6.1f}{t:>+9.3f}   [{lo:+.3f}, {hi:+.3f}]")

    print("\nInterpretation: illustration only. The real data contain no known")
    print("ground truth, the endpoint is a composite, follow-up appears to end at")
    print("discontinuation, and interval coverage is ~93%, not 95%.")


if __name__ == "__main__":
    main()

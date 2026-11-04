"""Gate 2e - decision-layer diagnosis and commitment-threshold sweep.

Question
--------
Why is TREAT precision 67 % in the sign-flip DGP, and can a stricter commitment
threshold restore it without destroying utility?

Pre-registered acceptance criteria (fixed before running)
--------------------------------------------------------
The decision layer is SALVAGEABLE only if some threshold k achieves BOTH
    - TREAT precision >= 90 %
    - at least one TREAT verdict in >= 50 % of replications
Otherwise the decision layer is not salvageable at n = 589.

No re-simulation is needed: verdicts are a function of (tau_hat, SE, k) and both
are already stored, so the sweep is exactly paired at the rule level.

Usage
-----
    python -m ciscarl2.gate2e
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

K_GRID = (1.96, 2.2, 2.5, 3.0, 3.5, 4.0)
PRECISION_TARGET = 90.0
UTILITY_TARGET = 50.0


def load(signflip=True):
    name = ("gate2d_replications_signflip.csv" if signflip
            else "gate2d_replications.csv")
    d = pd.read_csv(os.path.join(RESULTS, name))
    d = d[d.null == 0].copy()
    d["beneficial"] = d.tau_true > 0
    return d


def diagnosе(d):
    """Why is precision low? Support, selection, stability, replication."""
    print("\n=== 1. Coverage and precision by support quartile ===")
    d = d.copy()
    d["q"] = pd.qcut(d.support, 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
    g = d.groupby("q", observed=True).agg(
        n=("support", "size"), support=("support", "mean"),
        coverage=("covered", lambda s: 100 * s.mean()),
        mean_se=("se", "mean"),
        mean_abs_effect=("tau_true", lambda s: s.abs().mean()))
    print(g.round(3).to_string())

    print("\n=== 2. Winner's curse: is the committed estimate biased upward? ===")
    for v in ("TREAT", "AVOID", "INSUFFICIENT"):
        s = d[d.verdict == v]
        if len(s) == 0:
            continue
        print(f"  {v:13s} n={len(s):5d}  mean(tau_hat)={s.tau_hat.mean():+.4f}  "
              f"mean(tau_true)={s.tau_true.mean():+.4f}  "
              f"inflation={s.tau_hat.mean()-s.tau_true.mean():+.4f}  "
              f"mean |tau_true|={s.tau_true.abs().mean():.4f}")

    print("\n=== 3. Is the 67 % stable across replications? ===")
    per = d.groupby("rep").apply(
        lambda s: pd.Series({
            "n_treat": (s.verdict == "TREAT").sum(),
            "n_correct": ((s.verdict == "TREAT") & s.beneficial).sum(),
        }), include_groups=False)
    per = per[per.n_treat > 0]
    prec = 100 * per.n_correct / per.n_treat
    print(f"  reps with >=1 TREAT: {len(per)}/{d.rep.nunique()}")
    print(f"  per-rep TREAT precision: median={prec.median():.0f} %  "
          f"IQR=[{prec.quantile(.25):.0f}, {prec.quantile(.75):.0f}]  "
          f"min={prec.min():.0f}  max={prec.max():.0f}")

    print("\n=== 4. Does rule stability predict accuracy? (selection optimism) ===")
    d["err"] = d.tau_hat - d.tau_true
    for q in (0.25, 0.5, 0.75):
        hi = d[d.freq >= d.freq.quantile(q)]
        lo = d[d.freq < d.freq.quantile(q)]
        print(f"  freq >= {d.freq.quantile(q):.2f}: n={len(hi):4d} "
              f"|err|={hi.err.abs().mean():.4f}  coverage={100*hi.covered.mean():.1f} %"
              f"   | freq < : |err|={lo.err.abs().mean():.4f} "
              f"coverage={100*lo.covered.mean():.1f} %")

    print("\n=== 5. Coverage conditional on the estimate being far from zero ===")
    # coverage should be uniform across |tau_hat|/se if intervals are valid
    d["z"] = (d.tau_hat / d.se).abs()
    bins = pd.cut(d.z, [0, 1, 1.96, 2.5, 3.5, np.inf],
                  labels=["<1", "1-1.96", "1.96-2.5", "2.5-3.5", ">3.5"])
    print(d.groupby(bins, observed=True).agg(
        n=("z", "size"), coverage=("covered", lambda s: 100 * s.mean()),
        precision_oracle=("beneficial", lambda s: 100 * s.mean())).round(1).to_string())


def sweep(d):
    print("\n=== 6. Commitment-threshold sweep (exactly paired) ===")
    print(f"{'k':>5} {'TREAT n':>8} {'prec %':>8} {'lift':>6} {'commit %':>9} "
          f"{'reps>=1':>8} {'AVOID n':>8} {'prec %':>8}")
    rows = []
    n_reps = d.rep.nunique()
    base_treat = 100 * d.beneficial.mean()
    for k in K_GRID:
        lo = d.tau_hat - k * d.se
        hi = d.tau_hat + k * d.se
        t = d[lo > 0]
        a = d[hi < 0]
        tp = 100 * t.beneficial.mean() if len(t) else np.nan
        ap = 100 * (~a.beneficial).mean() if len(a) else np.nan
        reps_t = t.rep.nunique() if len(t) else 0
        rows.append(dict(k=k, n_treat=len(t), treat_prec=tp,
                         lift=(tp / base_treat) if len(t) else np.nan,
                         commit=100 * len(t) / len(d),
                         reps_pct=100 * reps_t / n_reps,
                         n_avoid=len(a), avoid_prec=ap))
        print(f"{k:5.2f} {len(t):8d} {tp:8.1f} {tp/base_treat:6.2f} "
              f"{100*len(t)/len(d):9.1f} {100*reps_t/n_reps:8.0f} "
              f"{len(a):8d} {ap:8.1f}")
    out = pd.DataFrame(rows)
    ok = out[(out.treat_prec >= PRECISION_TARGET) &
             (out.reps_pct >= UTILITY_TARGET)]
    print(f"\nCriteria: TREAT precision >= {PRECISION_TARGET:.0f} % AND "
          f">= {UTILITY_TARGET:.0f} % of reps with a TREAT verdict.")
    if len(ok):
        best = ok.sort_values("k").iloc[0]
        print(f"  -> SALVAGEABLE at k = {best.k:.2f}: precision {best.treat_prec:.0f} %, "
              f"commitment in {best.reps_pct:.0f} % of reps, "
              f"{best.n_treat} TREAT verdicts, lift {best.lift:.2f}x")
    else:
        print("  -> NOT SALVAGEABLE under these criteria at n = 589.")
    return out


def main():
    d = load(signflip=True)
    print(f"sign-flip replications: {len(d)} rules over {d.rep.nunique()} reps")
    print(f"base rate: {100*d.beneficial.mean():.1f} % of rules truly beneficial")
    diagnosе(d)
    s = sweep(d)
    s.to_csv(os.path.join(RESULTS, "gate2e_sweep.csv"), index=False)


if __name__ == "__main__":
    main()

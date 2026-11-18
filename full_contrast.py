"""Full ACTG175 cohort, contrast arm 0 vs arms 1-3 (any alternative regimen).

A = 1 for arms 1/2/3, 0 for arm 0. With four randomised arms the propensity is
e = 0.75 (three of four arms are 'alternative'), not 0.5. The CISCaRL machinery
is run with the correct e over 20 splits.

Run:  py -3.13 full_contrast.py --reps 20
Writes: results/full_contrast.csv, .json, .md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ciscarl2 import dgp
from ciscarl2.estimators import (crossfit_nuisances, phi_on_holdout,
                                 truncate_weights)
from ciscarl2.gate2 import pseudo_outcome
from ciscarl2.gate11 import _cart
from ciscarl2.gate22 import _stats
from ciscarl2.paths import data_path
from ciscarl2.rules import eval_rule, jaccard

ROOT = Path(__file__).resolve().parent
H = 730.0
SUP = 200
Z = 1.645


def raw_rule(cond, names, mu, sd):
    out = []
    for f, op, thr in cond:
        nm = names[f]
        out.append(f"{nm} {op} {thr * sd[f] + mu[f]:.1f}")
    return " & ".join(out)


def run_split(seed, X, A, T, E, e, names):
    idx = np.arange(len(A))
    strat = A * 2 + E
    disc, hold = train_test_split(idx, test_size=0.5, random_state=seed,
                                  stratify=strat)
    val, asm = train_test_split(hold, test_size=0.5, random_state=seed + 1,
                                stratify=strat[hold])

    mu_x, sd_x = X[disc].mean(0), X[disc].std(0)
    sd_x[sd_x == 0] = 1.0
    Xs = (X - mu_x) / sd_x

    XD, XV, XA = Xs[disc], Xs[val], Xs[asm]
    AD, TD, ED = A[disc], T[disc], E[disc]
    AV, TV, EV = A[val], T[val], E[val]
    AA, TA, EA = A[asm], T[asm], E[asm]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, H, seed=seed, return_folds=True)
    G, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(AD, TD, H, muD, G, e=e)

    _, leaves = _cart(psi, XD, 3)
    means, ses, sup = _stats(psi, leaves)
    ok = sup >= SUP
    fallback = not bool(ok.any())
    j = (int(np.nanargmax(np.where(ok, means, -np.inf))) if ok.any()
         else int(np.nanargmax(means)))
    cond = leaves[j]["cond"]
    mask_full = eval_rule(cond, Xs)

    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, H, e=e, seed=seed)
    dphi_v = p1v - p0v
    p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, H, e=e, seed=seed)
    dphi_a = p1a - p0a

    mv = eval_rule(cond, XV)
    ma = eval_rule(cond, XA)
    eff = float(dphi_v[mv].mean()) if mv.any() else np.nan
    se = (float(np.sqrt(((dphi_v[mv] - eff) ** 2).sum()) / mv.sum())
          if mv.any() else np.nan)
    lcb = eff - Z * se
    asm = float(dphi_a[ma].mean()) if ma.any() else np.nan

    return dict(
        seed=int(seed), n=int(len(A)), e=float(e),
        n_disc=int(len(disc)), n_val=int(len(val)),
        support=int(sup[j]), fallback=bool(fallback),
        rule=raw_rule(cond, names, mu_x, sd_x),
        mean_psi=float(means[j]),
        val_effect=float(eff), val_se=float(se), val_lcb=float(lcb),
        asm_effect=float(asm), committed=bool(lcb > 0),
        n_rule_val=int(mv.sum()), mask_full=mask_full,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--start", type=int, default=2024)
    args = ap.parse_args()

    df = pd.read_csv(data_path())
    df = df[df["arms"].isin([0, 1, 2, 3])].reset_index(drop=True)
    X = df[dgp.COVARIATES].to_numpy(float)
    A = (df["arms"] != 0).to_numpy(int)
    T = df["days"].to_numpy(float)
    E = df["cens"].to_numpy(int)
    e = float(A.mean())
    names = dgp.COVARIATES

    print(f"n={len(A)}  e={e:.3f}  events={int(E.sum())}", flush=True)
    rows = []
    t0 = time.time()
    for r in range(args.reps):
        seed = args.start + r
        row = run_split(seed, X, A, T, E, e, names)
        rows.append(row)
        print(f"[{r+1}/{args.reps}] seed={seed} support={row['support']} "
              f"val={row['val_effect']:+.3f} lcb={row['val_lcb']:+.3f} "
              f"commit={row['committed']} ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.drop(columns=["mask_full"]).to_csv(
        ROOT / "results" / "full_contrast.csv", index=False)

    masks = res["mask_full"].tolist()
    jac = [jaccard(masks[0], m) for m in masks]
    committed = res[res.committed]
    summary = {
        "dataset": "ACTG175, four arms",
        "contrast": "arm 0 (ZDV) vs arms 1-3 (any alternative)",
        "e": e,
        "n": int(len(A)),
        "events": int(E.sum()),
        "reps": int(args.reps),
        "commit_rate": float(100 * res.committed.mean()),
        "abstain_rate": float(100 * (1 - res.committed.mean())),
        "fallback_rate": float(100 * res.fallback.mean()),
        "mean_lcb": float(res.val_lcb.mean()),
        "mean_val_effect": float(res.val_effect.mean()),
        "mean_assessment_committed": (float(committed.asm_effect.mean())
                                      if len(committed) else None),
        "mean_support": float(res.support.mean()),
        "mean_jaccard_vs_canonical": float(np.mean(jac)),
        "unique_rules": int(res.rule.nunique()),
        "canonical": {k: v for k, v in rows[0].items() if k != "mask_full"},
    }
    with open(ROOT / "results" / "full_contrast.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md = [
        "# ACTG175 full cohort: arm 0 vs arms 1-3",
        "",
        "Contrast: ZDV monotherapy vs any alternative regimen (arms 1/2/3).",
        f"Propensity e = {e:.3f} (known by four-arm randomisation).",
        f"n = {len(A):,}; events = {int(E.sum()):,}; splits = {args.reps}.",
        "",
        f"- commit = {summary['commit_rate']:.1f}% "
        f"({int(res.committed.sum())}/{args.reps})",
        f"- abstain = {summary['abstain_rate']:.1f}%",
        f"- fallback = {summary['fallback_rate']:.1f}%",
        f"- mean validation LCB = {summary['mean_lcb']:+.3f}",
        f"- mean validation effect = {summary['mean_val_effect']:+.3f}",
        f"- mean support = {summary['mean_support']:.0f}",
        f"- unique rules = {summary['unique_rules']}/{args.reps}",
        f"- mean Jaccard vs canonical = {summary['mean_jaccard_vs_canonical']:.3f}",
    ]
    if summary["mean_assessment_committed"] is not None:
        md.append(f"- committed mean assessment = "
                  f"{summary['mean_assessment_committed']:+.3f}")
    md += [
        "",
        "## Canonical split (seed 2024)",
        "",
        f"- rule: `{rows[0]['rule']}`",
        f"- support: {rows[0]['support']} (fallback: {rows[0]['fallback']})",
        f"- validation: {rows[0]['val_effect']:+.3f} "
        f"(SE {rows[0]['val_se']:.3f}, LCB {rows[0]['val_lcb']:+.3f})",
        f"- assessment: {rows[0]['asm_effect']:+.3f}",
        f"- decision: {'TREAT' if rows[0]['committed'] else 'INSUFFICIENT'}",
    ]
    (ROOT / "results" / "full_contrast.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

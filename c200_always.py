"""C200-always: the C200 pipeline with the confidence (LCB) layer removed.

The selected leaf is always treated. There is no confidence/LCB layer in this
file. SCRE is kept as the comparator.

Run:  py -3.13 c200_always.py --reps 40 --kinds single,two
Writes: results/always_treat.csv, _summary.csv, .md, .json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ciscarl2 import dgp
from ciscarl2.estimators import aipw, crossfit_nuisances, phi_on_holdout
from ciscarl2.gate2 import pseudo_outcome
from ciscarl2.gate4 import scre_select_alpha
from ciscarl2.gate4d import TSTAR, _delta_for
from ciscarl2.gate11 import _cart
from ciscarl2.gate19 import _assign_effs, _leaf_table
from ciscarl2.gate20 import _capped_G
from ciscarl2.gate22 import _stats
from ciscarl2.gate23 import E1, E2, _config23, _regions
from ciscarl2.gate26 import _calibrate_kind, _sim_kind
from ciscarl2.gate28 import _metrics
from ciscarl2.rules import eval_rule

try:
    from ciscarl2.external.scre import SurvivalCausalRuleEnsemble
except Exception:  # noqa: BLE001
    SurvivalCausalRuleEnsemble = None

ROOT = Path(__file__).resolve().parent
START = 300100
SUP = 200


def regions_kind(X, cfg, kind):
    R1, R2 = _regions(X, cfg)
    if kind == "single":
        R2 = np.zeros(len(X), bool)
    return R1, R2


def run_rep(rep, kind, cfg, lam_c0):
    seed = START + rep
    rng = np.random.RandomState(seed)
    n = 8000
    X = cfg[0][rng.choice(len(cfg[0]), n, replace=True)]
    R1, R2 = regions_kind(X, cfg, kind)
    union = R1 | R2
    lp0 = X @ dgp.BETA
    s0 = np.exp(-cfg[5] * TSTAR * np.exp(lp0))
    e1 = np.where(R1, _delta_for(E1, s0), 0.0)
    e2 = np.where(R2, _delta_for(E2, s0), 0.0)
    A = rng.binomial(1, 0.5, n)
    sim = _sim_kind(X, A, lam_c0, cfg[5], rng, e1, e2)
    T, E = sim["Tobs"], sim["E"]
    truth = sim["S1"] - sim["S0"]

    ii = np.arange(n)
    disc, hold = train_test_split(ii, test_size=0.5, random_state=seed,
                                  stratify=A)
    val, asm = train_test_split(hold, test_size=0.5, random_state=seed + 1,
                                stratify=A[hold])
    XD, AD, TD, ED = X[disc], A[disc], T[disc], E[disc]
    XV, AV, TV, EV = X[val], A[val], T[val], E[val]
    XA, AA, TA, EA = X[asm], A[asm], T[asm], E[asm]
    uniA, truthA = union[asm], truth[asm]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    G = _capped_G(muD, GD, Gmarg, fid, fcaps, "conditional", TD)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
    const = float(aipw(XD, AD, TD, TSTAR, muD, G)[0])

    leaves = _cart(psi, XD, 3)[1]
    means, ses_d, sup = _stats(psi, leaves)
    ok = sup >= SUP
    j = (int(np.nanargmax(np.where(ok, means, -np.inf))) if ok.any()
         else int(np.nanargmax(means)))

    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=seed)
    dphi_v = p1v - p0v
    tbl_v = _leaf_table(leaves, XV, dphi_v)
    effs = np.array([t["eff"] for t in tbl_v], float)
    ses = np.array([t["se"] for t in tbl_v], float)
    effs = np.where(np.isfinite(effs), effs, const)

    masks_a = [eval_rule(r["cond"], XA) for r in leaves]
    tbl_a = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                  eff=effs[t], se=ses[t]) for t in range(len(leaves))]
    pred_a = _assign_effs(tbl_a, XA, effs, const)
    treated_a = masks_a[j]

    mc = _metrics(pred_a, treated_a, truthA, uniA)
    rows = [dict(rep=rep, kind=kind, method="C200-always",
                 support=int(sup[j]), **mc)]

    tau_s = np.full(len(XA), np.nan)
    n_rules_s = 0
    ok_s = True
    try:
        if SurvivalCausalRuleEnsemble is None:
            raise RuntimeError("SCRE unavailable")
        s = SurvivalCausalRuleEnsemble(
            alpha=0.5, pred_n_estimators=100, progn_n_estimators=100,
            rule_min_support=20, max_rules=200, max_rule_conditions=3,
            max_selected_rules=10, random_state=42)
        with contextlib.redirect_stdout(io.StringIO()):
            s.fit(XD, psi, np.isfinite(psi),
                  feature_names=[f"x{i}" for i in range(XD.shape[1])],
                  time=TD, event=ED.astype(bool), treatment=AD)
            pick = scre_select_alpha(s, XV, AV, TV, EV.astype(bool), TSTAR)
        if pick is not None:
            s.alpha_ = s.alpha_final_ = pick[1]
            s.coef_ = pick[2]
        s.t_star_ = TSTAR
        with contextlib.redirect_stdout(io.StringIO()):
            tau_s = np.asarray(s.predict(XA), float)
        n_rules_s = len(getattr(s, "selected_rules_", []))
    except Exception:
        ok_s = False

    if ok_s and np.all(np.isfinite(tau_s)):
        ms = _metrics(tau_s, tau_s > 0, truthA, uniA)
        rows.append(dict(rep=rep, kind=kind, method="SCRE",
                         support=np.nan, **ms))
    else:
        rows.append(dict(rep=rep, kind=kind, method="SCRE",
                         support=np.nan, mae=np.nan, spearman=np.nan,
                         capture=np.nan, purity=np.nan, per_treated=np.nan,
                         efficiency=np.nan, null=np.nan, coverage=np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--kinds", default="single,two")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]

    cfg = _config23()
    lam = {k: _calibrate_kind(cfg, k) for k in kinds}

    from joblib import Parallel, delayed
    jobs = [(r, k) for k in kinds for r in range(args.reps)]
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_rep)(r, k, cfg, lam[k]) for r, k in jobs)
    df = pd.DataFrame([x for ch in out for x in ch])
    df.to_csv(ROOT / "results" / "always_treat.csv", index=False)

    agg = (df.groupby(["kind", "method"])
           .agg(reps=("rep", "size"), mae=("mae", "mean"),
                purity=("purity", "mean"), capture=("capture", "mean"),
                per_treated=("per_treated", "mean"),
                coverage=("coverage", "mean"),
                null_rate=("null", lambda s: 100 * s.mean()))
           .reset_index())
    agg.to_csv(ROOT / "results" / "always_treat_summary.csv", index=False)
    with open(ROOT / "results" / "always_treat.json", "w", encoding="utf-8") as fh:
        json.dump(json.loads(agg.to_json(orient="records")), fh, indent=2)

    lines = [
        "# C200-always vs SCRE",
        "",
        f"{args.reps} reps per DGP, n = 8,000, seeds {START}-{START+args.reps-1}.",
        "C200-always has no confidence/LCB layer: the selected leaf is always treated.",
        "",
        "| DGP | method | reps | MAE | purity | capture | per-treated | coverage | null |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r.kind} | {r.method} | {int(r.reps)} | {r.mae:.4f} | "
            f"{r.purity:.3f} | {r.capture:.3f} | {r.per_treated:+.3f} | "
            f"{r.coverage:.3f} | {r.null_rate:.1f}% |")
    (ROOT / "results" / "always_treat.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== C200-always vs SCRE ===")
    print(agg.round(4).to_string(index=False))
    print(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""Ablation: remove the commitment (confidence) layer from C200.

Three methods on identical simulated data and assessment patients:
  C200-commit  : selected leaf treated only if validation LCB > 0 (production)
  C200-always  : selected leaf always treated (no confidence layer)
  SCRE         : comparator

Run:  py -3.13 always_treat_ablation.py --reps 20 --kinds single,two
Writes: results/always_treat.csv, .json, .md
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
from scipy.stats import norm
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
Z = 1.645


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
    truth = (sim["S1"] - sim["S0"])

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
    mv = eval_rule(leaves[j]["cond"], XV)
    kv = int(mv.sum())
    hi_eff = float(dphi_v[mv].mean()) if kv else np.nan
    hi_se = (float(np.sqrt(((dphi_v[mv] - hi_eff) ** 2).sum()) / kv)
             if kv else np.nan)
    committed = bool(np.isfinite(hi_eff) and np.isfinite(hi_se)
                     and hi_eff > Z * hi_se)

    masks_a = [eval_rule(r["cond"], XA) for r in leaves]

    # production: commitment layer
    use_c = effs.copy()
    if not committed:
        use_c[j] = const
    tbl_c = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                  eff=use_c[t], se=ses[t]) for t in range(len(leaves))]
    pred_c = _assign_effs(tbl_c, XA, use_c, const)
    treated_c = masks_a[j] if committed else np.zeros(len(XA), bool)

    # ablation: no confidence layer
    tbl_a = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                  eff=effs[t], se=ses[t]) for t in range(len(leaves))]
    pred_a = _assign_effs(tbl_a, XA, effs, const)
    treated_a = masks_a[j]

    rows = []
    for method, pred, treated in (
            ("C200-commit", pred_c, treated_c),
            ("C200-always", pred_a, treated_a)):
        mc = _metrics(pred, treated, truthA, uniA)
        rows.append(dict(rep=rep, kind=kind, method=method,
                         committed=bool(committed), support=int(sup[j]),
                         **mc))

    # SCRE
    tau_s = np.full(len(XA), np.nan)
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
    except Exception:
        ok_s = False

    if ok_s and np.all(np.isfinite(tau_s)):
        ms = _metrics(tau_s, tau_s > 0, truthA, uniA)
        rows.append(dict(rep=rep, kind=kind, method="SCRE",
                         committed=True, support=np.nan, **ms))
    else:
        rows.append(dict(rep=rep, kind=kind, method="SCRE",
                         committed=True, support=np.nan, mae=np.nan,
                         spearman=np.nan, capture=np.nan, purity=np.nan,
                         per_treated=np.nan, efficiency=np.nan, null=np.nan,
                         coverage=np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
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

    rows = []
    for (kind, method), g in df.groupby(["kind", "method"]):
        rows.append({
            "kind": kind, "method": method, "reps": len(g),
            "mae": float(g.mae.mean()),
            "purity": float(g.purity.mean()),
            "capture": float(g.capture.mean()),
            "per_treated": float(g.per_treated.mean()),
            "coverage": float(g.coverage.mean()),
            "null_rate": float(100 * g["null"].mean()),
        })
    summ = pd.DataFrame(rows).sort_values(["kind", "method"])
    summ.to_csv(ROOT / "results" / "always_treat_summary.csv", index=False)

    lines = [
        "# Ablation: removing the commitment (confidence) layer",
        "",
        f"{args.reps} reps per DGP, n = 8,000, seeds {START}-{START+args.reps-1}.",
        "C200-commit = production (treat only if validation LCB > 0);",
        "C200-always = selected leaf always treated; SCRE = comparator.",
        "",
        "| DGP | method | MAE | purity | capture | per-treated | coverage | null |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"| {r.kind} | {r.method} | {r.mae:.4f} | {r.purity:.3f} | "
            f"{r.capture:.3f} | {r.per_treated:+.3f} | {r.coverage:.3f} | "
            f"{r.null_rate:.1f}% |")
    (ROOT / "results" / "always_treat.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    with open(ROOT / "results" / "always_treat.json", "w", encoding="utf-8") as fh:
        json.dump(json.loads(summ.to_json(orient="records")), fh, indent=2)

    print("\n=== ablation summary ===")
    print(summ.round(4).to_string(index=False))
    print(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

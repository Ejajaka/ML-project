"""Gate 2c - stability-threshold sweep (discovery quality vs calibration).

Design
------
Thresholds are NESTED: a rule at frequency f passes every threshold <= f. So the
stability frequencies are computed ONCE per replication and all four thresholds
are evaluated on the same candidate set. This gives exact within-replication
pairing (the review's requirement) and makes the sweep ~4x cheaper than running
each threshold separately.

Thresholds: 0.40, 0.50, 0.60 (baseline), 0.70.

Pre-registered discovery-quality metrics (defined BEFORE running)
----------------------------------------------------------------
Let tau_i be the true effect for patient i on the evaluation split, sd(tau) its
standard deviation, and r the selected rule.

  separation(r) = [mean(tau | r) - mean(tau | not r)] / sd(tau)
        Continuous, needs no arbitrary region definition, and directly measures
        "does this rule isolate a distinct-benefit subgroup".

  R*_75            = the top-quartile-benefit patients on the evaluation split
  precision(r)     = |r and R*| / |r|
  recall(r)        = |r and R*| / |R*|
  IoU(r)           = |r and R*| / |r or R*|

Per replication we record BOTH the rule with the highest stability (the one a
user would read first) and the best rule by separation, so the summary is not
driven by cherry-picking a favourable rule after the fact.

Null replications are recorded as discovery failures and are EXCLUDED from the
coverage denominator (there is no interval to evaluate). Coverage is conditional
on a rule being discovered, as in Gate 2b.

Usage
-----
    python -m ciscarl2.gate2c --reps 100
    python -m ciscarl2.gate2c --confirm          # fresh seeds, chosen threshold
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
from .estimators import aipw_eif, crossfit_nuisances, truncate_weights
from .gate2 import CELLS, TAU, pseudo_outcome
from .rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts_gate2c")
from .paths import data_path

DEFAULT_DATA = data_path()

THRESHOLDS = (0.40, 0.50, 0.60, 0.70)
BASELINE = 0.60
TRUE_REGION_Q = 75      # top-quartile true benefit


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def run_one(job):
    (ci, rep, kind, lam0, lam_c0, pool, t_star, n, stab_B, min_support,
     offset) = job
    rng = np.random.RandomState(50000 * ci + rep + offset)
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate(X, kind, lam_c0, lam0, rng, t_star, tau=TAU)
    A, Tobs, E, S0, S1 = sim["A"], sim["Tobs"], sim["E"], sim["S0"], sim["S1"]

    idx = np.arange(n)
    D, Ev = train_test_split(idx, test_size=0.5, random_state=rep, stratify=A)
    XD, AD, TD, ED = X[D], A[D], Tobs[D], E[D]
    XE, AE, TE, EE = X[Ev], A[Ev], Tobs[Ev], E[Ev]

    # ---- discovery (once) ----
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, t_star, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(AD, TD, t_star, muD, GD)

    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=min_support, random_state=0,
                               n_jobs=-1).fit(XD, psi)
    cands = dedup(rules_from_ensemble(rf, XD, min_support, max_trees=30))
    subs = []
    for _ in range(stab_B):
        ii = rng.choice(len(XD), len(XD) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, min_support // 2),
                                  random_state=0,
                                  n_jobs=-1).fit(XD[ii], psi[ii])
        subs.append(rules_from_ensemble(t, XD, 5, max_trees=15))
    freq = stability_scores(cands, subs, XD) if cands else np.array([])

    # ---- evaluation (once) ----
    out = aipw_eif(XE, AE, TE, EE, t_star, seed=rep, n_splits=5)
    dphi = out["phi1"] - out["phi0"]
    tau_i = (S1 - S0)[Ev]
    sd_tau = float(np.std(tau_i)) or 1.0
    rstar = tau_i >= np.percentile(tau_i, TRUE_REGION_Q)

    rows = []
    for thr in THRESHOLDS:
        sel = [dict(cond=c, mask=m, score=f)
               for (c, m), f in zip(cands, freq) if f >= thr]
        sel = prune(sel, XD, max_rules=5, overlap=0.8) if sel else []
        if not sel:
            rows.append(dict(cell=ci, rep=rep, mechanism=kind, threshold=thr,
                             null=1, n_rules=0))
            continue

        per_rule = []
        for r in sel:
            mE = eval_rule(r["cond"], XE)
            nr = int(mE.sum())
            if nr < 10:
                continue
            tau = float(dphi[mE].mean())
            se = float(np.sqrt(((dphi[mE] - tau) ** 2).sum()) / nr)
            tq = float(tdist.ppf(0.975, max(nr - 1, 1)))
            sep = (tau_i[mE].mean() - tau_i[~mE].mean()) / sd_tau
            inter = np.sum(mE & rstar)
            prec = inter / max(nr, 1)
            rec = inter / max(int(rstar.sum()), 1)
            iou = inter / max(int((mE | rstar).sum()), 1)
            per_rule.append(dict(
                freq=float(r["score"]), support=nr,
                tau_hat=tau, tau_true=float(tau_i[mE].mean()),
                covered=float(abs(tau - tau_i[mE].mean()) <= tq * se),
                width=2 * tq * se,
                separation=float(sep), precision=float(prec),
                recall=float(rec), iou=float(iou)))
        if not per_rule:
            rows.append(dict(cell=ci, rep=rep, mechanism=kind, threshold=thr,
                             null=1, n_rules=0))
            continue

        best_stab = max(per_rule, key=lambda d: d["freq"])
        best_sep = max(per_rule, key=lambda d: d["separation"])
        rows.append(dict(cell=ci, rep=rep, mechanism=kind, threshold=thr,
                         null=0, n_rules=len(per_rule),
                         top_freq=best_stab["freq"],
                         top_support=best_stab["support"],
                         top_tau=best_stab["tau_hat"],
                         top_covered=best_stab["covered"],
                         top_width=best_stab["width"],
                         top_sep=best_stab["separation"],
                         top_iou=best_stab["iou"],
                         top_prec=best_stab["precision"],
                         top_rec=best_stab["recall"],
                         mean_sep=float(np.mean([d["separation"] for d in per_rule])),
                         mean_iou=float(np.mean([d["iou"] for d in per_rule])),
                         mean_freq=float(np.mean([d["freq"] for d in per_rule])),
                         mean_cover=float(np.mean([d["covered"] for d in per_rule])),
                         best_sep=best_sep["separation"],
                         best_iou=best_sep["iou"],
                         best_width=best_sep["width"]))
    return rows


def summarise(df):
    out = []
    for (ci, m, thr), g in df.groupby(["cell", "mechanism", "threshold"]):
        ok = g[g.null == 0]
        out.append(dict(
            cell=int(ci), mechanism=m, threshold=thr, reps=len(g),
            null_rate=100 * g.null.mean(),
            rules=ok.n_rules.mean() if len(ok) else 0.0,
            top_freq=ok.top_freq.mean() if len(ok) else np.nan,
            separation=ok.top_sep.mean() if len(ok) else np.nan,
            iou=ok.top_iou.mean() if len(ok) else np.nan,
            precision=ok.top_prec.mean() if len(ok) else np.nan,
            recall=ok.top_rec.mean() if len(ok) else np.nan,
            coverage=100 * ok.top_covered.mean() if len(ok) else np.nan,
            width=ok.top_width.mean() if len(ok) else np.nan,
            mean_sep=ok.mean_sep.mean() if len(ok) else np.nan,
            mean_iou=ok.mean_iou.mean() if len(ok) else np.nan,
            mean_freq=ok.mean_freq.mean() if len(ok) else np.nan,
            mean_cover=100 * ok.mean_cover.mean() if len(ok) else np.nan,
            n_cov=int(ok.top_covered.notna().sum())))
    return pd.DataFrame(out).sort_values(["mechanism", "threshold"])


def paired(df):
    """Paired-by-replication differences vs the 0.60 baseline."""
    rows = []
    for (ci, m), g in df.groupby(["cell", "mechanism"]):
        piv = g.pivot_table(index="rep", columns="threshold",
                            values=["null", "top_sep", "top_iou",
                                    "top_covered"], aggfunc="first")
        for thr in THRESHOLDS:
            if thr == BASELINE:
                continue
            dn = piv[("null", thr)] - piv[("null", BASELINE)]
            ds = piv[("top_sep", thr)] - piv[("top_sep", BASELINE)]
            di = piv[("top_iou", thr)] - piv[("top_iou", BASELINE)]
            n = int(dn.notna().sum())
            rows.append(dict(
                cell=int(ci), mechanism=m, threshold=thr, reps=n,
                d_null=100 * dn.mean(),
                d_null_se=100 * dn.std(ddof=1) / np.sqrt(n),
                d_sep=ds.mean(), d_sep_se=ds.std(ddof=1) / np.sqrt(n),
                d_iou=di.mean(), d_iou_se=di.std(ddof=1) / np.sqrt(n)))
    return pd.DataFrame(rows)


def to_markdown(s, p):
    L = ["# Gate 2c - stability-threshold sweep\n",
         "Discovery metrics were defined before the run. Coverage is conditional "
         "on a rule being discovered; null replications are counted as discovery "
         "failures, not as coverage zeros.\n",
         "| cell | thr | null | rules | mean freq | top separation | mean "
         "separation | mean IoU | coverage (top) | coverage (all rules) | width |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.mechanism} | {r.threshold:.2f} | {r.null_rate:.0f} % | "
                 f"{r.rules:.1f} | {r.mean_freq:.2f} | {r.separation:.2f} | "
                 f"{r.mean_sep:.2f} | {r.mean_iou:.2f} | "
                 f"{r.coverage:.1f} % | {r.mean_cover:.1f} % | {r.width:.3f} |")
    L.append("\n## Paired differences vs the 0.60 baseline (same replications)\n")
    L.append("| cell | thr | d null (pp) | d separation | d IoU |")
    L.append("|---|---|---|---|---|")
    for _, r in p.iterrows():
        L.append(f"| {r.mechanism} | {r.threshold:.2f} | "
                 f"{r.d_null:+.1f} +/- {r.d_null_se:.1f} | "
                 f"{r.d_sep:+.3f} +/- {r.d_sep_se:.3f} | "
                 f"{r.d_iou:+.3f} +/- {r.d_iou_se:.3f} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--t-star", type=float, default=dgp.TSTAR)
    ap.add_argument("--data", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stab-b", type=int, default=20)
    ap.add_argument("--min-support", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0,
                    help="seed offset; use for the confirmation run")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool)
    cells, reps = list(CELLS), args.reps
    if args.quick:
        cells, reps = cells[:1], min(reps, 6)
    lam_c = {c: dgp.calibrate_lam_c(c[0], c[1], pool, lam0, args.t_star,
                                    gen_kwargs={"tau": TAU}) for c in cells}

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    from joblib import Parallel, delayed
    jobs = [(ci, rep, c[0], lam0, lam_c[c], pool, args.t_star, args.n,
             args.stab_b, args.min_support, args.offset)
            for ci, c in enumerate(cells) for rep in range(reps)]
    out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
    df = pd.DataFrame([r for ch in out for r in ch if r])
    df.to_csv(os.path.join(RESULTS, f"gate2c_replications_{args.tag}.csv"),
              index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, f"gate2c_summary_{args.tag}.csv"), index=False)
    p = paired(df)
    p.to_csv(os.path.join(RESULTS, f"gate2c_paired_{args.tag}.csv"), index=False)
    md = to_markdown(s, p)
    with open(os.path.join(RESULTS, f"gate2c_summary_{args.tag}.md"),
              "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

"""Gate 2b - do EIF-corrected intervals restore nominal coverage?

One focused experiment. Same pipeline as Gate 2 (same DGP, same discovery, same
evaluation split), but the interval is computed analytically three ways:

    IF      plain AIPW influence function (nuisances treated as fixed)
    EIF     IF variance + fold-jackknife variance over the nuisance fits
    EIF-t   EIF with a t-quantile at n_r - 1 df instead of 1.96

No refit bootstrap, so this runs in seconds per replication and can afford
100 replications per cell.

Target: 95 % conditional-on-rule coverage.

Usage
-----
    python -m ciscarl2.gate2b --reps 100
    python -m ciscarl2.gate2b --quick --reps 6
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import aipw_eif, crossfit_nuisances, truncate_weights
from .gate2 import CELLS, TAU, discover, pseudo_outcome
from .rules import eval_rule

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts_gate2b")
from .paths import data_path

DEFAULT_DATA = data_path()


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def run_one(job):
    (ci, rep, kind, lam0, lam_c0, pool, t_star, n, stab_B, min_support, n_splits) = job
    rng = np.random.RandomState(50000 * ci + rep)      # SAME seeds as Gate 2
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate(X, kind, lam_c0, lam0, rng, t_star, tau=TAU)
    A, Tobs, E, S0, S1 = sim["A"], sim["Tobs"], sim["E"], sim["S0"], sim["S1"]

    idx = np.arange(n)
    D, Ev = train_test_split(idx, test_size=0.5, random_state=rep, stratify=A)
    XD, AD, TD, ED = X[D], A[D], Tobs[D], E[D]
    XE, AE, TE, EE = X[Ev], A[Ev], Tobs[Ev], E[Ev]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, t_star, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psiD = pseudo_outcome(AD, TD, t_star, muD, GD)
    rules, _ = discover(XD, psiD, np.random.RandomState(rep + 1),
                        min_support, max_rules=5, stab_B=stab_B, stab_thr=0.6)
    if not rules:
        return [dict(cell=ci, rep=rep, mechanism=kind, n_rules=0)]

    out = aipw_eif(XE, AE, TE, EE, t_star, seed=rep, n_splits=n_splits)
    dphi = out["phi1"] - out["phi0"]
    true_eff = S1[Ev] - S0[Ev]
    K = len(out["fold_phi"])

    rows = []
    for r in rules:
        mE = eval_rule(r["cond"], XE)
        nr = int(mE.sum())
        if nr < 10:
            continue
        tau = float(dphi[mE].mean())
        tau_true = float(true_eff[mE].mean())

        se_if = float(np.sqrt(((dphi[mE] - tau) ** 2).sum()) / nr)
        tk = np.array([float((p1 - p0)[mE].mean()) for p1, p0 in out["fold_phi"]])
        var_jack = (K - 1) / K * float(((tk - tk.mean()) ** 2).sum())
        se_eif = float(np.sqrt(se_if ** 2 + var_jack))

        tq = float(tdist.ppf(0.975, max(nr - 1, 1)))

        rows.append(dict(
            cell=ci, rep=rep, mechanism=kind, n_rules=len(rules),
            support=nr, tau_true=tau_true, tau_hat=tau, freq=float(r["score"]),
            se_if=se_if, se_eif=se_eif,
            cov_if=float(abs(tau - tau_true) <= 1.96 * se_if),
            cov_eif=float(abs(tau - tau_true) <= 1.96 * se_eif),
            cov_eift=float(abs(tau - tau_true) <= tq * se_eif),
            width_if=2 * 1.96 * se_if, width_eif=2 * 1.96 * se_eif,
            width_eift=2 * tq * se_eif,
            jack_share=100 * var_jack / max(se_if ** 2 + var_jack, 1e-12),
        ))
    return rows


def summarise(df):
    ok = df[df.n_rules > 0]
    out = []
    for (ci, m), g in ok.groupby(["cell", "mechanism"]):
        total = int(df[(df.cell == ci) & (df.mechanism == m)].rep.nunique())
        n = len(g)
        row = dict(cell=int(ci), mechanism=m, reps=total, rules=n,
                   no_rule_reps=total - int(g.rep.nunique()),
                   support=g.support.mean(), freq=g.freq.mean(),
                   jack_share=g.jack_share.mean(),
                   bias=(g.tau_hat - g.tau_true).mean())
        for k in ("if", "eif", "eift"):
            c = g[f"cov_{k}"].mean()
            row[f"cover_{k}"] = 100 * c
            row[f"mc_{k}"] = 100 * np.sqrt(c * (1 - c) / max(n, 1))
            row[f"width_{k}"] = g[f"width_{k}"].mean()
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(s):
    L = ["# Gate 2b - EIF-corrected subgroup intervals\n",
         "Same pipeline and seeds as Gate 2; analytic intervals only, so no "
         "bootstrap noise. Target: 95 %.\n",
         "| cell | no-rule reps | rules | bias | jackknife share of variance | "
         "cover IF | **cover EIF** | cover EIF-t | width EIF |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.mechanism} | {r.no_rule_reps}/{r.reps} | {r.rules} | "
                 f"{r.bias:+.4f} | {r.jack_share:.0f} % | "
                 f"{r.cover_if:.1f} % +/- {r.mc_if:.1f} | "
                 f"**{r.cover_eif:.1f} % +/- {r.mc_eif:.1f}** | "
                 f"{r.cover_eift:.1f} % +/- {r.mc_eift:.1f} | {r.width_eif:.3f} |")
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
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool)
    cells, reps = list(CELLS), args.reps
    if args.quick:
        cells, reps = cells[:1], min(reps, 6)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    lam_c = {c: dgp.calibrate_lam_c(c[0], c[1], pool, lam0, args.t_star,
                                    gen_kwargs={"tau": TAU}) for c in cells}

    from joblib import Parallel, delayed
    frames, t0 = [], time.time()
    for ci, cell in enumerate(cells):
        part = os.path.join(PARTS, f"g2b_{cell[0]}_{int(round(100*cell[1]))}_k{args.n_splits}.csv")
        if os.path.exists(part) and not args.force:
            d = pd.read_csv(part)
            if d.rep.nunique() >= reps:
                print(f"[skip] {cell}", flush=True)
                frames.append(d)
                continue
        jobs = [(ci, rep, cell[0], lam0, lam_c[cell], pool, args.t_star, args.n,
                 args.stab_b, args.min_support, args.n_splits) for rep in range(reps)]
        out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
        d = pd.DataFrame([r for ch in out for r in ch if r])
        d.to_csv(part, index=False)
        frames.append(d)
        print(f"[cell] {cell[0]:12s} {100*cell[1]:4.0f}%  rows={len(d)}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(RESULTS, "gate2b_replications.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, "gate2b_summary.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, "gate2b_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

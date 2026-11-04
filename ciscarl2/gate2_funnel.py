"""Gate 2-funnel - why does discovery return no rule in 16-21% of replications?

Discovery only: no effect estimation, so this is ~1 s per replication. Records
where candidate rules are lost:

    raw       leaf rules extracted from the RF ensemble
    dedup     after de-duplicating identical condition sets
    support   after the min_support filter
    stable    after the stability >= 0.6 filter
    final     after pruning / the max-5 cap

Also records, for every replication, the largest stability score achieved and
the support of the best candidate, so a null can be attributed.

Usage
-----
    python -m ciscarl2.gate2_funnel --reps 100
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import crossfit_nuisances, truncate_weights
from .gate2 import CELLS, TAU, pseudo_outcome
from .rules import dedup, rules_from_ensemble, stability_scores

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
from .paths import data_path

DEFAULT_DATA = data_path()


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def run_one(job):
    (ci, rep, kind, lam0, lam_c0, pool, t_star, n, stab_B, min_support) = job
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.RandomState(50000 * ci + rep)      # same seeds as Gate 2
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate(X, kind, lam_c0, lam0, rng, t_star, tau=TAU)
    A, Tobs, E = sim["A"], sim["Tobs"], sim["E"]

    idx = np.arange(n)
    D, _ = train_test_split(idx, test_size=0.5, random_state=rep, stratify=A)
    XD, AD, TD, ED = X[D], A[D], Tobs[D], E[D]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, t_star, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(AD, TD, t_star, muD, GD)

    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=min_support, random_state=0,
                               n_jobs=-1).fit(XD, psi)
    raw = rules_from_ensemble(rf, XD, min_support, max_trees=30)
    cands = dedup(raw)

    subs = []
    for _ in range(stab_B):
        ii = rng.choice(len(XD), len(XD) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, min_support // 2),
                                  random_state=0,
                                  n_jobs=-1).fit(XD[ii], psi[ii])
        subs.append(rules_from_ensemble(t, XD, 5, max_trees=15))

    if cands:
        freq = stability_scores(cands, subs, XD)
        n_stable = int((freq >= 0.6).sum())
        best_freq = float(freq.max())
        best_support = int(max(m.sum() for _, m in cands))
        supports = np.array([m.sum() for _, m in cands])
    else:
        freq = np.array([])
        n_stable, best_freq, best_support = 0, 0.0, 0
        supports = np.array([0])

    return [dict(cell=ci, rep=rep, mechanism=kind, n_raw=len(raw),
                 n_cand=len(cands), n_stable=n_stable,
                 n_final=min(n_stable, 5), best_freq=best_freq,
                 best_support=best_support,
                 median_support=float(np.median(supports)),
                 n_events_rule_arm0=int(ED[TD > t_star].sum()),
                 psi_sd=float(np.std(psi)))]


def summarise(df):
    out = []
    for (ci, m), g in df.groupby(["cell", "mechanism"]):
        null = g.n_final == 0
        # attribute each null to the first stage that zeroed it
        cause = np.where(g.n_cand == 0, "no candidates",
                np.where(g.n_stable == 0, "stability filter", "found"))
        out.append(dict(
            cell=int(ci), mechanism=m, reps=len(g),
            null_rate=100 * null.mean(),
            cause_no_cand=100 * (cause == "no candidates").mean(),
            cause_stability=100 * (cause == "stability filter").mean(),
            mean_cand=g.n_cand.mean(), mean_stable=g.n_stable.mean(),
            best_freq_mean=g.best_freq.mean(),
            best_freq_p10=g.best_freq.quantile(0.10),
            median_support=g.median_support.mean(),
            psi_sd=g.psi_sd.mean(),
        ))
    return pd.DataFrame(out)


def to_markdown(s):
    L = ["# Gate 2-funnel - attribution of empty rule lists\n",
         "| cell | reps | null rate | no candidates | stability filter | "
         "mean candidates | mean stable | best freq (mean) | best freq (p10) | "
         "median support |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.mechanism} | {r.reps} | **{r.null_rate:.0f} %** | "
                 f"{r.cause_no_cand:.0f} % | {r.cause_stability:.0f} % | "
                 f"{r.mean_cand:.1f} | {r.mean_stable:.1f} | "
                 f"{r.best_freq_mean:.2f} | {r.best_freq_p10:.2f} | "
                 f"{r.median_support:.0f} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--t-star", type=float, default=dgp.TSTAR)
    ap.add_argument("--data", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--stab-b", type=int, default=20)
    ap.add_argument("--min-support", type=int, default=60)
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool)
    cells, reps = list(CELLS), args.reps
    if args.quick:
        cells, reps = cells[:1], min(reps, 6)
    lam_c = {c: dgp.calibrate_lam_c(c[0], c[1], pool, lam0, args.t_star,
                                    gen_kwargs={"tau": TAU}) for c in cells}

    from joblib import Parallel, delayed
    jobs = [(ci, rep, c[0], lam0, lam_c[c], pool, args.t_star, args.n,
             args.stab_b, args.min_support)
            for ci, c in enumerate(cells) for rep in range(reps)]
    out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
    df = pd.DataFrame([r for ch in out for r in ch if r])
    df.to_csv(os.path.join(RESULTS, "gate2_funnel_replications.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, "gate2_funnel_summary.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, "gate2_funnel_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

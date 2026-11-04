"""Gate 1 - estimator validation under four censoring mechanisms.

Protocol (pre-registered)
-------------------------
n = 600 covariates resampled from the prepared ACTG175 matrix, A ~ Bern(0.5),
Cox event model with a known analytic truth, t* = 730 days.

Censoring intensities are calibrated to hit 10 %, 25 % and 50 % censoring
before t* - the ACTG175 operating range (2 y, intermediate, 3 y).

Estimators
----------
aipw_cond    AIPW, conditional (per-arm Cox on X) censoring model   [PRIMARY]
aipw_marg    AIPW, marginal Kaplan-Meier censoring                  [sensitivity]
aipw_oracle  AIPW with the true G(t*|X)                             [ceiling]
cox_ph       Cox T-learner comparator (SCRE's estimator family)     [comparator]

Checkpointing
-------------
Each (mechanism, intensity) cell is written to results/parts/ as soon as it
finishes, so an interrupted run resumes instead of restarting. Use --force to
recompute everything.

Usage
-----
    python -m ciscarl2.gate1 --reps 200
    python -m ciscarl2.gate1 --reps 8 --quick       # smoke test
    python -m ciscarl2.gate1 --force                # ignore checkpoints
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import dgp
from .estimators import (aipw, cox_bootstrap_se, cox_tlearner_contrast,
                         crossfit_nuisances, weight_diagnostics)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts")

from .paths import data_path

DEFAULT_DATA = data_path()

FRACTIONS = (0.10, 0.25, 0.50)
ESTIMATORS = ("aipw_cond", "aipw_marg", "aipw_oracle", "cox_ph")

# The Cox comparator needs a bootstrap SE (no analytic one). To keep the run
# affordable it is bootstrapped on the first COMP_BOOT_REPS reps per cell;
# its point estimate is still computed on every rep.
COMP_BOOT_REPS = 60
COMP_B = 50


def _find_data(explicit):
    candidates = [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
                  os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "ACTG175.csv not found. Pass --data or set ACTG175_PATH.")


def run_one(job):
    """One replication: simulate, fit nuisances once, evaluate 4 estimators."""
    (mi, frac, rep, kind, lam0, lam_c0, pool, t_star, n) = job
    rng = np.random.RandomState(10000 * mi + int(round(100 * frac)) + rep)

    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate(X, kind, lam_c0, lam0, rng, t_star)
    A, Tobs, E = sim["A"], sim["Tobs"], sim["E"]
    tau_true = sim["tau_true"]
    base = dict(mechanism=kind, cens_frac=frac, rep=rep, tau_true=tau_true,
                cens_before=sim["cens_before"])

    rows = []
    try:
        mu, G_cond, G_marg = crossfit_nuisances(X, A, Tobs, E, t_star, seed=rep)
    except Exception as exc:
        for name in ESTIMATORS:
            rows.append(dict(base, estimator=name, tau_hat=np.nan, se_hat=np.nan,
                             covered=np.nan, width=np.nan, w_max=np.nan,
                             w_p99=np.nan, ess=np.nan, pct_trunc=np.nan,
                             error=str(exc)))
        return rows

    G_oracle = dgp.oracle_G(kind, lam_c0, X, t_star)

    for name, G in (("aipw_cond", G_cond), ("aipw_marg", G_marg),
                    ("aipw_oracle", G_oracle)):
        if G is None:
            rows.append(dict(base, estimator=name, tau_hat=np.nan, se_hat=np.nan,
                             covered=np.nan, width=np.nan, w_max=np.nan,
                             w_p99=np.nan, ess=np.nan, pct_trunc=np.nan,
                             error="no oracle for this mechanism"))
            continue
        tau, se, _, _ = aipw(X, A, Tobs, t_star, mu, G, e=0.5)
        wd = weight_diagnostics(G, Tobs, t_star)
        rows.append(dict(base, estimator=name, tau_hat=tau, se_hat=se,
                         covered=float(abs(tau - tau_true) <= 1.96 * se),
                         width=2 * 1.96 * se, error="", **wd))

    tau = cox_tlearner_contrast(X, A, Tobs, E, t_star)
    if rep < COMP_BOOT_REPS:
        se = cox_bootstrap_se(X, A, Tobs, E, t_star, B=COMP_B, seed=rep)
        cov = float(abs(tau - tau_true) <= 1.96 * se) if np.isfinite(se) else np.nan
    else:
        se, cov = np.nan, np.nan
    rows.append(dict(base, estimator="cox_ph", tau_hat=tau, se_hat=se,
                     covered=cov,
                     width=2 * 1.96 * se if np.isfinite(se) else np.nan,
                     w_max=np.nan, w_p99=np.nan, ess=np.nan, pct_trunc=np.nan,
                     error=""))
    return rows


def summarise(df):
    ok = df["tau_hat"].notna()
    out = []
    for (mech, frac, est), g in df[ok].groupby(["mechanism", "cens_frac", "estimator"]):
        err = g["tau_hat"].to_numpy() - g["tau_true"].to_numpy()
        R = len(g)
        cov_all = g["covered"].to_numpy(dtype=float)
        cov = cov_all[np.isfinite(cov_all)]
        Rc = len(cov)
        out.append(dict(
            mechanism=mech, cens_frac=frac, estimator=est, reps=R, cov_reps=Rc,
            true_tau=g["tau_true"].mean(),
            bias=err.mean(),
            mc_se_bias=err.std(ddof=1) / np.sqrt(R) if R > 1 else np.nan,
            rmse=float(np.sqrt(np.mean(err ** 2))),
            bias_pct=100 * err.mean() / g["tau_true"].mean(),
            cover=100 * cov.mean() if Rc else np.nan,
            mc_se_cover=(100 * np.sqrt(cov.mean() * (1 - cov.mean()) / Rc)
                         if Rc > 1 else np.nan),
            width=g["width"].mean(),
            w_max=g["w_max"].max() if g["w_max"].notna().any() else np.nan,
            w_p99=g["w_p99"].max() if g["w_p99"].notna().any() else np.nan,
            ess=g["ess"].mean() if g["ess"].notna().any() else np.nan,
            pct_trunc=g["pct_trunc"].mean() if g["pct_trunc"].notna().any() else np.nan,
            cens_realised=100 * g["cens_before"].mean(),
        ))
    return pd.DataFrame(out).sort_values(["mechanism", "cens_frac", "estimator"])


def to_markdown(s):
    lines = ["# Gate 1 - estimator validation\n",
             f"Reps per cell: {int(s['reps'].max())}. "
             "Truth = full-data tau; coverage = 95 % nominal.\n"]
    titles = {"independent": "1. Independent censoring (baseline)",
              "covariate": "2. Censoring depends on X (ACTG-like)",
              "event": "3. Censoring depends on latent T",
              "frailty": "4. Shared unmeasured frailty"}
    for mech in ("independent", "covariate", "event", "frailty"):
        sub = s[s.mechanism == mech]
        if sub.empty:
            continue
        lines.append(f"\n## {titles[mech]}\n")
        lines.append("| cens | estimator | bias | MC SE | bias % | RMSE | "
                     "cover 95 % | width | max wt | ESS |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            cov_txt = ("n/a" if not np.isfinite(r.cover)
                       else f"{r.cover:.1f} % +/- {r.mc_se_cover:.1f} (n={int(r.cov_reps)})")
            lines.append(
                f"| {100*r.cens_frac:.0f} % | {r.estimator} | {r.bias:+.4f} | "
                f"{r.mc_se_bias:.4f} | {r.bias_pct:+.1f} % | {r.rmse:.4f} | "
                f"{cov_txt} | {r.width:.3f} | {r.w_max:.2f} | {r.ess:.0f} |")
    lines.append("\n## Pre-registered criteria\n")
    lines.append("- PASS if, under mechanisms 1-2 with `aipw_cond`: "
                 "|bias| < 0.01 and coverage in [93 %, 97 %].\n"
                 "- Monotone bias growth under mechanisms 3-4 is a "
                 "pre-declared empirical hypothesis, not a theorem.\n"
                 "- Coverage carries its Monte Carlo SE (approx. 1.5 pp at "
                 "R = 200); a single 92.5 % observation is not a definitive "
                 "failure.\n"
                 "- The weight tail (max wt, ESS) is a result, not a nuisance.\n")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--t-star", type=float, default=dgp.TSTAR)
    ap.add_argument("--data", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="2 mechanisms, 2 intensities, few reps")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    args = ap.parse_args(argv)

    data_path = _find_data(args.data)
    pool = dgp.load_covariate_pool(data_path)
    lam0 = dgp.lambda0_for_target_survival(pool)
    print(f"covariate pool: {pool.shape}  from {data_path}")
    print(f"baseline hazard lam0 = {lam0:.6g} (mean S0(t*) = 0.70)")

    mechanisms = list(dgp.MECHANISMS)
    fractions = list(FRACTIONS)
    reps = args.reps
    if args.quick:
        mechanisms, fractions, reps = mechanisms[:2], fractions[::2], min(reps, 8)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)

    print("\ncalibrating censoring intensity ...")
    lam_c = {}
    for m in mechanisms:
        for f in fractions:
            lam_c[(m, f)] = dgp.calibrate_lam_c(m, f, pool, lam0, args.t_star)
            print(f"  {m:12s} target {100*f:4.0f}%  -> lam_c0 = {lam_c[(m, f)]:.6g}")

    from joblib import Parallel, delayed

    frames, t0 = [], time.time()
    for mi, m in enumerate(mechanisms):
        for f in fractions:
            part = os.path.join(PARTS, f"gate1_{m}_{int(round(100*f))}.csv")
            if os.path.exists(part) and not args.force:
                d = pd.read_csv(part)
                if len(d) >= reps * len(ESTIMATORS):
                    print(f"[skip] {m} {100*f:.0f}% (checkpoint)")
                    frames.append(d)
                    continue
            jobs = [(mi, f, rep, m, lam0, lam_c[(m, f)], pool, args.t_star, args.n)
                    for rep in range(reps)]
            out = Parallel(n_jobs=args.jobs, verbose=0)(
                delayed(run_one)(j) for j in jobs)
            d = pd.DataFrame([r for ch in out for r in ch])
            d.to_csv(part, index=False)
            frames.append(d)
            print(f"[cell] {m:12s} {100*f:4.0f}%  rows={len(d)}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    rep_path = os.path.join(RESULTS, "gate1_replications.csv")
    df.to_csv(rep_path, index=False)

    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, "gate1_summary.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, "gate1_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)

    print("\n" + md)
    print(f"\nwrote {rep_path}")
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

"""Gate 1c - non-proportional hazards: is the Cox comparator fairly specified?

Every Gate 1 / 1b mechanism used Cox event times, so a Cox comparator was
correctly specified and the gate could only show CISCaRL's cost, never its
benefit. This gate adds a crossing-hazards DGP.

Important structural point
--------------------------
With crossing hazards, a Cox *T-learner* (one baseline per arm) is still
correctly specified: two separate baselines can represent any difference in
timing between arms. A *pooled* Cox with a single baseline cannot - it forces a
constant log-hazard-ratio. That pooled model is the fair structural analogue of
a shared-basis Cox rule ensemble (SCRE), so it is the comparator that this gate
is about.

Estimators compared
-------------------
aipw_cox     truncated AIPW, Cox outcome model (correct here)
aipw_rsf     truncated AIPW, RSF outcome model (frozen ACTG175 spec)
cox_tlearner per-arm Cox baselines      (flexible; expected unbiased)
cox_pooled   single-baseline Cox        (SCRE-like; expected biased)

Usage
-----
    python -m ciscarl2.gate1c --reps 200
    python -m ciscarl2.gate1c --quick --reps 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import dgp
from .estimators import (cox_bootstrap_se, cox_pooled_contrast,
                         cox_tlearner_contrast, fit_ate)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts_gate1c")
from .paths import data_path

DEFAULT_DATA = data_path()

MECHS = ("independent", "covariate")
FRACTIONS = (0.10, 0.25, 0.50)
COMP_BOOT_REPS = 60
COMP_B = 40


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def run_one(job):
    mi, frac, rep, kind, lam0, lam_c0, pool, t_star, n = job
    rng = np.random.RandomState(40000 * mi + int(round(100 * frac)) + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate_nonph(X, kind, lam_c0, lam0, rng, t_star)
    A, Tobs, E, truth = sim["A"], sim["Tobs"], sim["E"], sim["tau_true"]
    base = dict(mechanism=kind, cens_frac=frac, rep=rep, tau_true=truth,
                cens_before=sim["cens_before"])

    rows = []
    for name, mu_model in (("aipw_cox", "cox"), ("aipw_rsf", "rsf")):
        try:
            r = fit_ate(X, A, Tobs, E, t_star, seed=rep, mu_model=mu_model,
                        truncate=True)
            rows.append(dict(base, estimator=name, tau_hat=r["tau"], se=r["se"],
                             covered=float(abs(r["tau"] - truth) <= 1.96 * r["se"]),
                             w_max=r["w_max"], ess=r["ess"],
                             pct_capped=r["pct_capped"]))
        except Exception as exc:                        # pragma: no cover
            rows.append(dict(base, estimator=name, tau_hat=np.nan, se=np.nan,
                             covered=np.nan, w_max=np.nan, ess=np.nan,
                             pct_capped=np.nan, error=str(exc)))

    for name, fn in (("cox_tlearner", cox_tlearner_contrast),
                     ("cox_pooled", cox_pooled_contrast)):
        tau = fn(X, A, Tobs, E, t_star)
        if rep < COMP_BOOT_REPS:
            se = cox_bootstrap_se(X, A, Tobs, E, t_star, B=COMP_B, seed=rep)
            cov = float(abs(tau - truth) <= 1.96 * se) if np.isfinite(se) else np.nan
        else:
            se, cov = np.nan, np.nan
        rows.append(dict(base, estimator=name, tau_hat=tau, se=se, covered=cov,
                         w_max=np.nan, ess=np.nan, pct_capped=np.nan))
    return rows


def summarise(df):
    ok = df.tau_hat.notna()
    out = []
    for (m, f, e), g in df[ok].groupby(["mechanism", "cens_frac", "estimator"]):
        err = g.tau_hat.to_numpy() - g.tau_true.to_numpy()
        R = len(g)
        cov = g.covered.to_numpy(float)
        cov = cov[np.isfinite(cov)]
        Rc = len(cov)
        out.append(dict(mechanism=m, cens_frac=f, estimator=e, reps=R,
                        cov_reps=Rc, true_tau=g.tau_true.mean(),
                        bias=err.mean(), mc_se=err.std(ddof=1) / np.sqrt(R),
                        rmse=float(np.sqrt(np.mean(err ** 2))),
                        bias_pct=100 * err.mean() / g.tau_true.mean(),
                        cover=100 * cov.mean() if Rc else np.nan,
                        mc_cover=(100 * np.sqrt(cov.mean() * (1 - cov.mean()) / Rc)
                                  if Rc > 1 else np.nan),
                        w_max=g.w_max.max() if g.w_max.notna().any() else np.nan,
                        ess=g.ess.mean() if g.ess.notna().any() else np.nan,
                        pct_capped=(g.pct_capped.mean()
                                    if g.pct_capped.notna().any() else np.nan)))
    return pd.DataFrame(out).sort_values(["mechanism", "cens_frac", "estimator"])


def to_markdown(s):
    L = ["# Gate 1c - crossing hazards (non-PH)\n",
         f"Reps per cell: {int(s.reps.max())}. Truth = full-data tau. "
         "Hazards cross at 180 d; treated worse early, better late.\n",
         "A Cox **T-learner** has one baseline per arm and stays correctly "
         "specified; a **pooled** Cox (single baseline, SCRE-like) does not.\n"]
    for m in MECHS:
        sub = s[s.mechanism == m]
        if sub.empty:
            continue
        L.append(f"\n## {m}\n")
        L.append("| cens | estimator | bias | MC SE | bias % | RMSE | cover | "
                 "max wt | ESS |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            cov = ("n/a" if not np.isfinite(r.cover)
                   else f"{r.cover:.1f} % +/- {r.mc_cover:.1f} (n={int(r.cov_reps)})")
            L.append(f"| {100*r.cens_frac:.0f} % | {r.estimator} | {r.bias:+.4f} | "
                     f"{r.mc_se:.4f} | {r.bias_pct:+.1f} % | {r.rmse:.4f} | "
                     f"{cov} | "
                     f"{r.w_max if np.isfinite(r.w_max) else float('nan'):.1f} | "
                     f"{r.ess if np.isfinite(r.ess) else float('nan'):.0f} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--t-star", type=float, default=dgp.TSTAR)
    ap.add_argument("--data", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool)
    mech, fracs, reps = list(MECHS), list(FRACTIONS), args.reps
    if args.quick:
        mech, fracs, reps = mech[:1], [0.50], min(reps, 6)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    lam_c = {(m, f): dgp.calibrate_lam_c(m, f, pool, lam0, args.t_star,
                                         gen=dgp.generate_nonph)
             for m in mech for f in fracs}

    from joblib import Parallel, delayed
    frames, t0 = [], time.time()
    for mi, m in enumerate(mech):
        for f in fracs:
            part = os.path.join(PARTS, f"g1c_{m}_{int(round(100*f))}.csv")
            if os.path.exists(part) and not args.force:
                d = pd.read_csv(part)
                if len(d) >= reps * 4:
                    print(f"[skip] {m} {100*f:.0f}%", flush=True)
                    frames.append(d)
                    continue
            jobs = [(mi, f, rep, m, lam0, lam_c[(m, f)], pool, args.t_star, args.n)
                    for rep in range(reps)]
            out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
            d = pd.DataFrame([r for ch in out for r in ch])
            d.to_csv(part, index=False)
            frames.append(d)
            print(f"[cell] {m:12s} {100*f:4.0f}%  rows={len(d)}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(RESULTS, "gate1c_replications.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, "gate1c_summary.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, "gate1c_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

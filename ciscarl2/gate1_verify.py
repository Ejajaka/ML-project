"""Gate 1-verify: does the v1.1 cap survive the outer cross-fitting pipeline?

The Gate 1b cap was validated with a single 5-fold cross-fit. The production
pipeline is outer 5-fold (discovery/evaluation separation) x inner cross-fit,
so the cap must be re-checked there: in the outer loop the cap threshold is
derived from an inner cross-fit of the *training* rows only, never from the
held-out rows it is applied to.

Compares, per (mechanism, intensity):
    v1.0          inner cross-fit, untruncated
    v1.1-inner    inner cross-fit, cap = min(Q_0.99 of fold-training weights, 20)
    v1.1-outer    outer cross-fit, cap derived from the outer-training rows

Pass criterion: v1.1-outer reproduces v1.1-inner to within Monte Carlo error
(|Δbias| < 2 MC SE, |Δcoverage| < 4 pp).

Usage
-----
    python -m ciscarl2.gate1_verify --reps 100
    python -m ciscarl2.gate1_verify --quick --reps 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import dgp
from .estimators import fit_ate, outer_crossfit_ate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts_verify")
from .paths import data_path

DEFAULT_DATA = data_path()

MECHS = ("independent", "covariate")
FRACTIONS = (0.10, 0.25, 0.50)


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def run_one(job):
    mi, frac, rep, kind, lam0, lam_c0, pool, t_star, n = job
    rng = np.random.RandomState(30000 * mi + int(round(100 * frac)) + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate(X, kind, lam_c0, lam0, rng, t_star)
    A, Tobs, E = sim["A"], sim["Tobs"], sim["E"]
    truth = sim["tau_true"]

    rows = []
    for name, kw in (
            ("v1.0", dict(truncate=False)),
            ("v1.1-inner", dict(truncate=True))):
        try:
            r = fit_ate(X, A, Tobs, E, t_star, seed=rep, **kw)
            rows.append(dict(mechanism=kind, cens_frac=frac, rep=rep,
                             estimator=name, tau_true=truth, tau_hat=r["tau"],
                             se=r["se"], covered=float(abs(r["tau"] - truth) <= 1.96 * r["se"]),
                             w_max=r["w_max"], ess=r["ess"],
                             pct_capped=r["pct_capped"], cap=r["cap_median"]))
        except Exception as exc:                       # pragma: no cover
            rows.append(dict(mechanism=kind, cens_frac=frac, rep=rep,
                             estimator=name, tau_true=truth, tau_hat=np.nan,
                             se=np.nan, covered=np.nan, w_max=np.nan,
                             ess=np.nan, pct_capped=np.nan, cap=np.nan))

    try:
        r = outer_crossfit_ate(X, A, Tobs, E, t_star, seed=rep, truncate=True)
        rows.append(dict(mechanism=kind, cens_frac=frac, rep=rep,
                         estimator="v1.1-outer", tau_true=truth,
                         tau_hat=r["tau"], se=r["se"],
                         covered=float(abs(r["tau"] - truth) <= 1.96 * r["se"]),
                         w_max=np.nan, ess=np.nan, pct_capped=np.nan,
                         cap=float(np.median(r["caps"]))))
    except Exception as exc:                           # pragma: no cover
        rows.append(dict(mechanism=kind, cens_frac=frac, rep=rep,
                         estimator="v1.1-outer", tau_true=truth, tau_hat=np.nan,
                         se=np.nan, covered=np.nan, w_max=np.nan, ess=np.nan,
                         pct_capped=np.nan, cap=np.nan))
    return rows


def summarise(df):
    ok = df.tau_hat.notna()
    out = []
    for (m, f, e), g in df[ok].groupby(["mechanism", "cens_frac", "estimator"]):
        err = g.tau_hat.to_numpy() - g.tau_true.to_numpy()
        R = len(g)
        cov = g.covered.to_numpy(float)
        out.append(dict(mechanism=m, cens_frac=f, estimator=e, reps=R,
                        bias=err.mean(), mc_se=err.std(ddof=1) / np.sqrt(R),
                        rmse=float(np.sqrt(np.mean(err ** 2))),
                        cover=100 * cov.mean(),
                        mc_cover=100 * np.sqrt(cov.mean() * (1 - cov.mean()) / R),
                        width=g.width.mean() if "width" in g else 2 * 1.96 * g.se.mean(),
                        w_max=g.w_max.max(), ess=g.ess.mean(),
                        pct_capped=g.pct_capped.mean(), cap=g["cap"].mean()))
    return pd.DataFrame(out).sort_values(["mechanism", "cens_frac", "estimator"])


def to_markdown(s):
    L = ["# Gate 1-verify - cap through the outer pipeline\n",
         f"Reps per cell: {int(s.reps.max())}. Pass: |dbias| < 2 MC SE, "
         "|dcover| < 4 pp between v1.1-inner and v1.1-outer.\n"]
    for m in MECHS:
        sub = s[s.mechanism == m]
        if sub.empty:
            continue
        L.append(f"\n## {m}\n")
        L.append("| cens | estimator | bias | MC SE | RMSE | cover | width | "
                 "max wt | ESS | capped | cap |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            L.append(f"| {100*r.cens_frac:.0f} % | {r.estimator} | {r.bias:+.4f} | "
                     f"{r.mc_se:.4f} | {r.rmse:.4f} | {r.cover:.1f} % +/- {r.mc_cover:.1f} | "
                     f"{r.width:.3f} | {r.w_max:.1f} | {r.ess:.0f} | "
                     f"{r.pct_capped:.2f} % | {r.cap:.1f} |")
    L.append("\n## Inner vs outer agreement\n")
    L.append("| mechanism | cens | dbias | 2xMC | dcover | verdict |")
    L.append("|---|---|---|---|---|---|")
    for (m, f), g in s.groupby(["mechanism", "cens_frac"]):
        try:
            a = g[g.estimator == "v1.1-inner"].iloc[0]
            b = g[g.estimator == "v1.1-outer"].iloc[0]
        except IndexError:
            continue
        db = b.bias - a.bias
        tol = 2 * np.sqrt(a.mc_se ** 2 + b.mc_se ** 2)
        dc = b.cover - a.cover
        ok = abs(db) <= tol and abs(dc) < 4
        L.append(f"| {m} | {100*f:.0f} % | {db:+.4f} | {tol:.4f} | "
                 f"{dc:+.1f} pp | {'PASS' if ok else 'FAIL'} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
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
    lam_c = {(m, f): dgp.calibrate_lam_c(m, f, pool, lam0, args.t_star)
             for m in mech for f in fracs}

    from joblib import Parallel, delayed
    frames, t0 = [], time.time()
    for mi, m in enumerate(mech):
        for f in fracs:
            part = os.path.join(PARTS, f"ver_{m}_{int(round(100*f))}.csv")
            if os.path.exists(part) and not args.force:
                d = pd.read_csv(part)
                if len(d) >= reps * 3:
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
    df.to_csv(os.path.join(RESULTS, "gate1_verify_replications.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, "gate1_verify_summary.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, "gate1_verify_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

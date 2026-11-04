"""Gate 1b - stabilise the inverse-probability-of-censoring weights.

Gate 1 found that under covariate-dependent censoring the bias stays small but
the RMSE explodes (0.148 at 25 % censoring against a bias of +0.0025), driven
by a handful of replications where a single fitted G(t*|X) is far below the
training support and 1/G reaches 500-700.

This gate compares weight-handling strategies and freezes one. All caps are
estimated from the *fitting* rows of each fold, so no cap sees the held-out
patients it is applied to.

Strategies
----------
none     untruncated (Gate 1 baseline)
cap10    weight capped at 10          (fixed)
cap20    weight capped at 20          (fixed)
cap50    weight capped at 50          (fixed)
q99      weight capped at the 99th percentile of the fold's training weights
q95      weight capped at the 95th percentile of the fold's training weights
oracle   true G(t*|X), uncapped       (reference, not a usable procedure)

Decision rule (pre-registered): among the *usable* strategies (everything
except `oracle`), choose the one that minimises RMSE subject to |bias| < 0.01
and coverage in [93 %, 97 %], pooling over mechanisms 1-2 and intensities
25 % and 50 % - i.e. the cells where Gate 1 struggled.

Usage
-----
    python -m ciscarl2.gate1b --reps 200
    python -m ciscarl2.gate1b --quick --reps 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import dgp
from .estimators import aipw, crossfit_nuisances

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts_gate1b")
from .paths import data_path

DEFAULT_DATA = data_path()

FRACTIONS = (0.10, 0.25, 0.50)
STRATEGIES = ("none", "cap10", "cap20", "cap50", "q99", "q95", "oracle")


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def _cap_vector(G0, fold_id, fold_caps, q=None, fixed=None):
    if fixed is not None:
        cap = np.full(len(G0), float(fixed))
    else:
        cap = np.array([fold_caps[int(k)][q] for k in fold_id], float)
    return np.maximum(G0, 1.0 / cap), cap


def run_one(job):
    (mi, frac, rep, kind, lam0, lam_c0, pool, t_star, n) = job
    rng = np.random.RandomState(20000 * mi + int(round(100 * frac)) + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate(X, kind, lam_c0, lam0, rng, t_star)
    A, Tobs, E = sim["A"], sim["Tobs"], sim["E"]
    tau_true = sim["tau_true"]
    base = dict(mechanism=kind, cens_frac=frac, rep=rep, tau_true=tau_true,
                cens_before=sim["cens_before"])

    mu, G0, _, fold_id, fold_caps, g_train_min = crossfit_nuisances(
        X, A, Tobs, E, t_star, seed=rep, return_folds=True)

    # extrapolation diagnostic: held-out G below the smallest value the
    # censoring model ever produced on its own training rows
    gmin = np.array([g_train_min[int(k)] for k in fold_id])
    extrap = float(np.mean(G0 < gmin))
    w0 = np.where(Tobs > t_star, 1.0, 0.0) / G0
    nz0 = w0[w0 > 0]

    variants = {}
    variants["none"] = (G0, np.full(len(G0), np.inf))
    for name, fixed in (("cap10", 10.0), ("cap20", 20.0), ("cap50", 50.0)):
        variants[name] = _cap_vector(G0, fold_id, fold_caps, fixed=fixed)
    for name, q in (("q99", 0.99), ("q95", 0.95)):
        variants[name] = _cap_vector(G0, fold_id, fold_caps, q=q)
    G_or = dgp.oracle_G(kind, lam_c0, X, t_star)
    if G_or is not None:
        variants["oracle"] = (G_or, np.full(len(G_or), np.inf))

    rows = []
    for name in STRATEGIES:
        if name not in variants:
            continue
        G, cap = variants[name]
        tau, se, _, _ = aipw(X, A, Tobs, t_star, mu, G, e=0.5)
        w = np.where(Tobs > t_star, 1.0, 0.0) / G
        nz = w[w > 0]
        ess = float((w.sum() ** 2) / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0
        capped = (float(np.mean(nz0 > cap[0])) if np.isfinite(cap[0])
                  and nz0.size else 0.0)
        rows.append(dict(
            base, strategy=name, tau_hat=tau, se_hat=se,
            covered=float(abs(tau - tau_true) <= 1.96 * se),
            width=2 * 1.96 * se,
            w_max=float(nz.max()) if nz.size else 0.0,
            w_p99=float(np.percentile(nz, 99)) if nz.size else 0.0,
            ess=ess, pct_capped=100 * capped, extrap=100 * extrap,
            rmse_rep=float((tau - tau_true) ** 2)))
    return rows


def summarise(df):
    out = []
    for (mech, frac, s), g in df.groupby(["mechanism", "cens_frac", "strategy"]):
        err = g["tau_hat"].to_numpy() - g["tau_true"].to_numpy()
        R = len(g)
        cov = g["covered"].to_numpy(dtype=float)
        out.append(dict(
            mechanism=mech, cens_frac=frac, strategy=s, reps=R,
            bias=err.mean(), mc_se_bias=err.std(ddof=1) / np.sqrt(R),
            rmse=float(np.sqrt(np.mean(err ** 2))),
            bias_pct=100 * err.mean() / g["tau_true"].mean(),
            cover=100 * cov.mean(),
            mc_se_cover=100 * np.sqrt(cov.mean() * (1 - cov.mean()) / R),
            width=g["width"].mean(),
            w_max=g["w_max"].max(), w_p99=g["w_p99"].mean(),
            ess=g["ess"].mean(), pct_capped=g["pct_capped"].mean(),
            extrap=g["extrap"].mean(),
        ))
    return pd.DataFrame(out).sort_values(["mechanism", "cens_frac", "strategy"])


def choose(s):
    """Pre-registered selection: minimise RMSE s.t. bias/coverage criteria."""
    usable = s[~s.strategy.isin(["oracle"])]
    cells = usable[usable.mechanism.isin(["independent", "covariate"])
                   & usable.cens_frac.isin([0.25, 0.50])]
    rows = []
    for strat, g in cells.groupby("strategy"):
        rows.append(dict(strategy=strat,
                         max_abs_bias=g.bias.abs().max(),
                         min_cover=g.cover.min(),
                         mean_rmse=g.rmse.mean(),
                         mean_width=g.width.mean(),
                         mean_ess=g.ess.mean(),
                         mean_capped=g.pct_capped.mean()))
    t = pd.DataFrame(rows)
    t["eligible"] = (t.max_abs_bias < 0.01) & t.min_cover.between(93, 97)
    good = t[t.eligible].sort_values("mean_rmse")
    pick = good.iloc[0].strategy if len(good) else None
    return t, pick


def to_markdown(s, t, pick):
    L = ["# Gate 1b - weight stabilisation\n",
         f"Reps per cell: {int(s.reps.max())}. Strategy chosen a priori by the "
         "decision rule at the bottom.\n"]
    for mech in ("independent", "covariate", "event", "frailty"):
        sub = s[s.mechanism == mech]
        if sub.empty:
            continue
        L.append(f"\n## {mech}\n")
        L.append("| cens | strategy | bias | RMSE | cover | width | max wt | "
                 "p99 wt | ESS | capped | extrap |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            L.append(f"| {100*r.cens_frac:.0f} % | {r.strategy} | {r.bias:+.4f} | "
                     f"{r.rmse:.4f} | {r.cover:.1f} % | {r.width:.3f} | "
                     f"{r.w_max:.1f} | {r.w_p99:.2f} | {r.ess:.0f} | "
                     f"{r.pct_capped:.2f} % | {r.extrap:.1f} % |")
    L.append("\n## Selection (mechanisms 1-2, 25 % and 50 % censoring)\n")
    L.append("| strategy | max abs bias | min cover | mean RMSE | mean width | "
             "mean ESS | capped | eligible |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in t.iterrows():
        L.append(f"| {r.strategy} | {r.max_abs_bias:.4f} | {r.min_cover:.1f} % | "
                 f"{r.mean_rmse:.4f} | {r.mean_width:.3f} | {r.mean_ess:.0f} | "
                 f"{r.mean_capped:.2f} % | {'yes' if r.eligible else 'no'} |")
    L.append(f"\n**Chosen: {pick if pick else 'none eligible — specification needs revisiting'}**\n")
    L.append("Selection rule was fixed before the run: minimise mean RMSE subject to "
             "|bias| < 0.01 and coverage in [93 %, 97 %] in the cells where Gate 1 "
             "struggled. `oracle` is a reference and is not eligible.\n")
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
    mech, fracs, reps = list(dgp.MECHANISMS), list(FRACTIONS), args.reps
    if args.quick:
        mech, fracs, reps = mech[:2], [0.50], min(reps, 8)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)

    lam_c = {}
    for m in mech:
        for f in fracs:
            lam_c[(m, f)] = dgp.calibrate_lam_c(m, f, pool, lam0, args.t_star)

    from joblib import Parallel, delayed
    frames, t0 = [], time.time()
    for mi, m in enumerate(mech):
        for f in fracs:
            part = os.path.join(PARTS, f"g1b_{m}_{int(round(100*f))}.csv")
            if os.path.exists(part) and not args.force:
                d = pd.read_csv(part)
                if len(d) >= reps * 6:
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
    df.to_csv(os.path.join(RESULTS, "gate1b_replications.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, "gate1b_summary.csv"), index=False)
    t, pick = choose(s)
    t.to_csv(os.path.join(RESULTS, "gate1b_selection.csv"), index=False)
    md = to_markdown(s, t, pick)
    with open(os.path.join(RESULTS, "gate1b_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

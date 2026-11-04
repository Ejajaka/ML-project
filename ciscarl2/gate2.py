"""Gate 2 - rule-level inference, abstention and stability.

The reframed claim rests on three things the estimator gates cannot test:
a small reproducible rule list, subgroup intervals that actually cover, and
honest abstention. This gate evaluates all three on data with known subgroup
effects.

Design
------
n = 800, split 50/50 into discovery (D) and evaluation (E).

Discovery (D only):
  * cross-fitted nuisances -> DR pseudo-outcome psi
  * shallow regression tree on psi -> candidate leaf rules
  * stability: refit the tree on 30 half-samples of D, match rules by Jaccard
  * keep rules with frequency >= 0.6, prune near-duplicates, cap at 5
  * NOTE: the tree is refit on subsamples but psi is held fixed, so this
    measures TREE stability, not full-pipeline stability (documented).

Estimation (E only), three interval methods:
  if_analytic   influence-function SE
  multiplier    patient bootstrap reusing fitted phi (nuisances held fixed)
  refit         patient bootstrap refitting nuisances (3-fold) each replicate

Truth = mean(S1 - S0) over the E patients matching the rule.

Coverage, width, abstention and verdict accuracy are reported per interval
method, plus rule-selection frequency and the Holm-adjusted decision.

Usage
-----
    python -m ciscarl2.gate2 --reps 50
    python -m ciscarl2.gate2 --quick --reps 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor

from . import dgp
from .estimators import aipw, crossfit_nuisances, truncate_weights
from .rules import (dedup, eval_rule, prune, rules_from_ensemble,
                    stability_scores)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts_gate2")
from .paths import data_path

DEFAULT_DATA = data_path()

CELLS = (("independent", 0.10), ("covariate", 0.25))
EPS = 1e-12

# Gate 2 needs an effect size at which discovery is possible at all. The Gate 1
# default (tau ~ 0.08) sits below the pseudo-outcome noise floor (sd ~ 0.9) and
# no tree can recover a subgroup. 0.8/0.5 gives tau ~ 0.15 with a 0.07-0.21
# range across age, which is the ACTG175 scale.
TAU = (0.8, 0.5)


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def pseudo_outcome(A, Tobs, t_star, mu, G, e=0.5):
    Y = np.where(Tobs > t_star, 1.0, 0.0) / G
    mu_a = np.where(A == 1, mu[:, 1], mu[:, 0])
    return (A - e) / (e * (1 - e)) * (Y - mu_a) + (mu[:, 1] - mu[:, 0])


def _nuisance_phi(X, A, Tobs, E, t_star, seed=0, n_splits=3,
                  cap_q=0.99, cap_ceil=20.0):
    mu, G, _, fid, fcaps, _ = crossfit_nuisances(
        X, A, Tobs, E, t_star, seed=seed, n_splits=n_splits, return_folds=True)
    G, _ = truncate_weights(G, fid, fcaps, quantile=cap_q, ceiling=cap_ceil)
    _, _, phi1, phi0 = aipw(X, A, Tobs, t_star, mu, G)
    return phi1, phi0


def discover(XD, psiD, rng, min_support, max_rules, stab_B=30, stab_thr=0.6):
    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=min_support, random_state=0,
                               n_jobs=-1).fit(XD, psiD)
    cands = dedup(rules_from_ensemble(rf, XD, min_support, max_trees=30))
    if not cands:
        return [], np.array([])

    subs = []
    n = len(XD)
    for _ in range(stab_B):
        idx = rng.choice(n, n // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, min_support // 2),
                                  random_state=0, n_jobs=-1).fit(XD[idx], psiD[idx])
        subs.append(rules_from_ensemble(t, XD, 5, max_trees=15))

    freq = stability_scores(cands, subs, XD)
    keep = [dict(cond=c, mask=m, score=f)
            for (c, m), f in zip(cands, freq) if f >= stab_thr]
    return prune(keep, XD, max_rules=max_rules), freq


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def verdict(lo, hi):
    if lo > 0:
        return "TREAT"
    if hi < 0:
        return "AVOID"
    return "INSUFFICIENT"


def run_one(job):
    (ci, rep, kind, lam0, lam_c0, pool, t_star, n, stab_B, boot_B, mult_B,
     min_support) = job
    rng = np.random.RandomState(50000 * ci + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate(X, kind, lam_c0, lam0, rng, t_star, tau=TAU)
    A, Tobs, E, S0, S1 = sim["A"], sim["Tobs"], sim["E"], sim["S0"], sim["S1"]

    idx = np.arange(n)
    D, Ev = train_test_split(idx, test_size=0.5, random_state=rep,
                              stratify=A)
    XD, AD, TD, ED = X[D], A[D], Tobs[D], E[D]
    XE, AE, TE, EE = X[Ev], A[Ev], Tobs[Ev], E[Ev]

    # ---- discovery on D ----
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, t_star, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psiD = pseudo_outcome(AD, TD, t_star, muD, GD)
    rules, freqs = discover(XD, psiD, np.random.RandomState(rep + 1),
                            min_support, max_rules=5, stab_B=stab_B,
                            stab_thr=0.6)
    if not rules:
        return [dict(cell=ci, rep=rep, mechanism=kind, n_rules=0, rule="",
                     support=0, tau_true=np.nan, freq=np.nan)]

    # ---- estimation on E ----
    phi1, phi0 = _nuisance_phi(XE, AE, TE, EE, t_star, seed=rep)
    dphi = phi1 - phi0
    true_eff = S1[Ev] - S0[Ev]

    rows = []
    for r in rules:
        mE = eval_rule(r["cond"], XE)
        nr = int(mE.sum())
        if nr < 10:
            continue
        tau = float(dphi[mE].mean())
        tau_true = float(true_eff[mE].mean())

        se_if = float(np.sqrt(((dphi[mE] - tau) ** 2).sum()) / nr)
        lo_if, hi_if = tau - 1.96 * se_if, tau + 1.96 * se_if

        # multiplier bootstrap (nuisances fixed) - resample the RULE members
        sub = dphi[mE]
        mb = np.array([sub[np.random.RandomState(1000 + b).choice(nr, nr)].mean()
                       for b in range(mult_B)])
        lo_mb, hi_mb = np.percentile(mb, [2.5, 97.5])

        # refit bootstrap
        rb = []
        for b in range(boot_B):
            ii = np.random.RandomState(9000 + b).choice(len(Ev), len(Ev))
            try:
                p1, p0 = _nuisance_phi(XE[ii], AE[ii], TE[ii], EE[ii], t_star,
                                       seed=b + 1, n_splits=3)
            except Exception:
                continue
            mm = eval_rule(r["cond"], XE[ii])
            if mm.sum() >= 5:
                rb.append(float((p1 - p0)[mm].mean()))
        if len(rb) >= max(10, boot_B // 2):
            lo_rb, hi_rb = np.percentile(rb, [2.5, 97.5])
            sd_rb = float(np.std(rb, ddof=1))
            lo_rbn, hi_rbn = tau - 1.96 * sd_rb, tau + 1.96 * sd_rb
        else:
            lo_rb = hi_rb = lo_rbn = hi_rbn = np.nan

        rows.append(dict(
            cell=ci, rep=rep, mechanism=kind, n_rules=len(rules),
            rule=" & ".join(f"{f}{op}{t:.1f}" for f, op, t in r["cond"]),
            support=nr, tau_true=tau_true, tau_hat=tau, freq=float(r["score"]),
            lo_if=lo_if, hi_if=hi_if, cov_if=float(lo_if <= tau_true <= hi_if),
            lo_mb=lo_mb, hi_mb=hi_mb, cov_mb=float(lo_mb <= tau_true <= hi_mb),
            lo_rb=lo_rb, hi_rb=hi_rb,
            cov_rb=float(lo_rb <= tau_true <= hi_rb) if np.isfinite(lo_rb) else np.nan,
            width_if=hi_if - lo_if, width_mb=hi_mb - lo_mb,
            width_rb=(hi_rb - lo_rb) if np.isfinite(lo_rb) else np.nan,
            lo_rbn=lo_rbn, hi_rbn=hi_rbn,
            cov_rbn=float(lo_rbn <= tau_true <= hi_rbn) if np.isfinite(lo_rbn) else np.nan,
            width_rbn=(hi_rbn - lo_rbn) if np.isfinite(lo_rbn) else np.nan,
            verdict=verdict(lo_rb, hi_rb) if np.isfinite(lo_rb) else "NA",
            verdict_n=verdict(lo_rbn, hi_rbn) if np.isfinite(lo_rbn) else "NA",
            p_if=float(2 * (1 - norm.cdf(abs(tau / max(se_if, EPS))))),
            sign_ok=float((tau > 0) == (tau_true > 0)),
        ))

    if rows:
        adj = holm([r["p_if"] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = float(a)
    return rows


def summarise(df):
    ok = df[df.n_rules > 0]
    out = []
    for (cell, m), g in ok.groupby(["cell", "mechanism"]):
        total_reps = int(df[(df.cell == cell) & (df.mechanism == m)].rep.nunique())
        with_rule = int(g.rep.nunique())
        out.append(dict(
            cell=int(cell), mechanism=m, reps=total_reps,
            rules_per_rep=len(g) / max(total_reps, 1),
            support=g.support.mean(), freq=g.freq.mean(),
            width_if=g.width_if.mean(), width_mb=g.width_mb.mean(),
            width_rb=g.width_rb.mean(), width_rbn=g.width_rbn.mean(),
            cover_if=100 * g.cov_if.mean(), cover_mb=100 * g.cov_mb.mean(),
            cover_rb=100 * g.cov_rb.mean(), cover_rbn=100 * g.cov_rbn.mean(),
            n_rules=int(g.cov_rb.notna().sum()),
            no_rule_reps=total_reps - with_rule,
            abstain=100 * (g.verdict == "INSUFFICIENT").mean(),
            abstain_n=100 * (g.verdict_n == "INSUFFICIENT").mean(),
            treat=100 * (g.verdict == "TREAT").mean(),
            avoid=100 * (g.verdict == "AVOID").mean(),
            sign_ok=100 * g.sign_ok.mean(),
            holm_sig=100 * (g.get("p_holm", pd.Series(np.nan)) < 0.05).mean(),
        ))
    return pd.DataFrame(out)


def to_markdown(s):
    L = ["# Gate 2 - rule-level inference, abstention, stability\n",
         "Coverage is conditional on the rule being selected. `refit` is the "
         "frozen production interval.\n",
         "| cell | rules/rep | support | freq | no-rule reps | cover IF | "
         "cover mult | cover refit-pct | **cover refit-normal** | width | "
         "abstain | TREAT | sign ok |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.mechanism} | {r.rules_per_rep:.1f} | {r.support:.0f} | "
                 f"{r.freq:.2f} | {r.no_rule_reps:.0f}/{r.reps} | "
                 f"{r.cover_if:.1f} % | {r.cover_mb:.1f} % | {r.cover_rb:.1f} % | "
                 f"**{r.cover_rbn:.1f} %** | {r.width_rbn:.3f} | {r.abstain:.0f} % | "
                 f"{r.treat:.0f} % | {r.sign_ok:.0f} % |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--t-star", type=float, default=dgp.TSTAR)
    ap.add_argument("--data", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stab-b", type=int, default=30)
    ap.add_argument("--boot-b", type=int, default=40)
    ap.add_argument("--mult-b", type=int, default=200)
    ap.add_argument("--min-support", type=int, default=60)
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool)
    cells = list(CELLS)
    reps = args.reps
    if args.quick:
        cells, reps = cells[:1], min(reps, 4)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    lam_c = {c: dgp.calibrate_lam_c(c[0], c[1], pool, lam0, args.t_star,
                                     gen_kwargs={"tau": TAU})
             for c in cells}

    from joblib import Parallel, delayed
    frames, t0 = [], time.time()
    for ci, cell in enumerate(cells):
        part = os.path.join(PARTS, f"g2_{cell[0]}_{int(round(100*cell[1]))}.csv")
        if os.path.exists(part) and not args.force:
            d = pd.read_csv(part)
            if d.rep.nunique() >= reps:
                print(f"[skip] {cell}", flush=True)
                frames.append(d)
                continue
        jobs = [(ci, rep, cell[0], lam0, lam_c[cell], pool, args.t_star, args.n,
                 args.stab_b, args.boot_b, args.mult_b, args.min_support)
                for rep in range(reps)]
        out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
        d = pd.DataFrame([r for ch in out for r in ch if r])
        d.to_csv(part, index=False)
        frames.append(d)
        print(f"[cell] {cell[0]:12s} {100*cell[1]:4.0f}%  rows={len(d)}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(RESULTS, "gate2_replications.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, "gate2_summary.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, "gate2_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

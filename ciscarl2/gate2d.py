"""Gate 2d - application-matched sample size (ACTG175 scale).

Every earlier discovery/inference gate ran at n = 2000 with tau ~ 0.2. The real
analysis has 589 training patients, S0(t*) ~ 0.80, ~10 % censoring at 730 days
and tau ~ 0.13. This gate mirrors those conditions instead of shrinking the
simulation.

Matched to ACTG175
------------------
n               = 589          (the notebook's training partition)
target S0(t*)   = 0.80         (ACTG175: S0 = 0.735, S1 = 0.863)
tau(t*)         ~ 0.13         (ACTG175 KM contrast at 730 d = 0.128)
censoring at t* = 10 %         (ACTG175: 103/1054 censored before 730 d)
pipeline        = outer 5-fold: discover on 4/5, evaluate on the held-out 1/5
                                 (the frozen spec's discovery/evaluation split,
                                  not a 50/50 carve-up)
threshold       = 0.50, min_support 60, Jaccard 0.8, max 5 rules (v1.2)

The point is not to make the method look strong. It is to find out whether it
produces anything usable at the size it will actually be applied to, and if not,
to record that as the operating limit.

Usage
-----
    python -m ciscarl2.gate2d --reps 100
    python -m ciscarl2.gate2d --quick --reps 5
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
from sklearn.model_selection import KFold

from . import dgp
from .estimators import (crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import pseudo_outcome
from .rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts_gate2d")
from .paths import data_path

DEFAULT_DATA = data_path()

TARGET_S0 = 0.735       # ACTG175 control survival at 730 d
TAU = (0.74, 0.30)      # chosen so tau(730) ~ 0.128 at S0 = 0.735
CENS = 0.10             # censored before t* in ACTG175
MECH = "covariate"      # ACTG175-like (censoring depends on age/cd40/cd80/race)
THR = 0.50
MIN_SUPPORT = 60
# sign-flip DGP: effect = SIGN_SLOPE * cd40_z, so it changes sign at the median
SIGNFLIP = False
SIGN_SLOPE = 0.60
SIGN_IDX = 0
MAX_RULES = 99          # v1.4: no rule-count cap; Jaccard pruning retained


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def _discover(XD, psiD, rng, stab_B):
    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=MIN_SUPPORT, random_state=0,
                               n_jobs=-1).fit(XD, psiD)
    cands = dedup(rules_from_ensemble(rf, XD, MIN_SUPPORT, max_trees=30))
    if not cands:
        return [], np.array([])
    subs = []
    for _ in range(stab_B):
        ii = rng.choice(len(XD), len(XD) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, MIN_SUPPORT // 2),
                                  random_state=0,
                                  n_jobs=-1).fit(XD[ii], psiD[ii])
        subs.append(rules_from_ensemble(t, XD, 5, max_trees=15))
    freq = stability_scores(cands, subs, XD)
    sel = [dict(cond=c, mask=m, score=f)
           for (c, m), f in zip(cands, freq) if f >= THR]
    return (prune(sel, XD, max_rules=MAX_RULES, overlap=0.8) if sel else []), freq


def run_one(job):
    (rep, lam0, lam_c0, pool, t_star, n, stab_B) = job
    rng = np.random.RandomState(70000 + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    eff = SIGN_SLOPE * X[:, SIGN_IDX] if SIGNFLIP else None
    sim = dgp.generate(X, MECH, lam_c0, lam0, rng, t_star, tau=TAU,
                       effect_vec=eff)
    A, Tobs, E, S0, S1 = sim["A"], sim["Tobs"], sim["E"], sim["S0"], sim["S1"]
    tau_i = S1 - S0

    rows = []
    for k, (fit, hold) in enumerate(KFold(5, shuffle=True,
                                          random_state=rep).split(X)):
        # ---- discovery on the other four folds ----
        muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
            X[fit], A[fit], Tobs[fit], E[fit], t_star, seed=rep, return_folds=True)
        GD, _ = truncate_weights(GD, fid, fcaps)
        psi = pseudo_outcome(A[fit], Tobs[fit], t_star, muD, GD)
        rules, _ = _discover(X[fit], psi, np.random.RandomState(rep + k), stab_B)

        if not rules:
            rows.append(dict(rep=rep, fold=k, null=1, n_rules=0))
            continue

        # ---- evaluation on the held-out fold ----
        h1, h0 = phi_on_holdout(X[fit], A[fit], Tobs[fit], E[fit],
                                X[hold], A[hold], Tobs[hold], t_star, seed=rep)
        dphi = h1 - h0
        t_hold = tau_i[hold]
        sd_tau = float(np.std(t_hold)) or 1.0
        rstar = t_hold >= np.percentile(t_hold, 75)

        for r in rules:
            m = eval_rule(r["cond"], X[hold])
            nr = int(m.sum())
            if nr < 8:
                continue
            tau = float(dphi[m].mean())
            se = float(np.sqrt(((dphi[m] - tau) ** 2).sum()) / nr)
            tq = float(tdist.ppf(0.975, max(nr - 1, 1)))
            tt = float(t_hold[m].mean())
            lo, hi = tau - tq * se, tau + tq * se
            verdict = ("TREAT" if lo > 0 else "AVOID" if hi < 0
                       else "INSUFFICIENT")
            inter = np.sum(m & rstar)
            rows.append(dict(
                rep=rep, fold=k, null=0, n_rules=len(rules),
                support=nr, freq=float(r["score"]),
                tau_hat=tau, tau_true=tt, se=se,
                covered=float(lo <= tt <= hi), width=hi - lo,
                verdict=verdict,
                sign_ok=float((tau > 0) == (tt > 0)),
                iou=float(inter / max(int((m | rstar).sum()), 1))))
    return rows


def summarise(df):
    per_rep = df.groupby("rep").agg(
        any_rule=("null", lambda s: int((s == 0).any())),
        rules=("null", lambda s: int((s == 0).sum()))).reset_index()
    ok = df[df.null == 0]
    return dict(
        reps=len(per_rep),
        null_reps=100 * (1 - per_rep.any_rule.mean()),
        rules_per_rep=per_rep.rules.mean(),
        rules_evaluated=len(ok),
        support=ok.support.mean(),
        freq=ok.freq.mean(),
        bias=(ok.tau_hat - ok.tau_true).mean(),
        rmse=float(np.sqrt(((ok.tau_hat - ok.tau_true) ** 2).mean())),
        coverage=100 * ok.covered.mean(),
        width=ok.width.mean(),
        abstain=100 * (ok.verdict == "INSUFFICIENT").mean(),
        treat=100 * (ok.verdict == "TREAT").mean(),
        sign_ok=100 * ok.sign_ok.mean(),
        treat_prec=100 * (ok[ok.verdict == "TREAT"].tau_true > 0).mean(),
        avoid_prec=(100 * (ok[ok.verdict == "AVOID"].tau_true < 0).mean()
                    if (ok.verdict == "AVOID").any() else float("nan")),
        n_treat=int((ok.verdict == "TREAT").sum()),
        n_avoid=int((ok.verdict == "AVOID").sum()),
        harmful_rules=100 * (ok.tau_true < 0).mean(),
        iou=ok.iou.mean(),
        tau_true=ok.tau_true.mean(),
    )


def to_markdown(m, ref=None):
    L = ["# Gate 2d - application-matched sample size (n = 589)\n",
         f"Matched to ACTG175: n=589, S0(t*)=0.735, tau~0.13, {int(CENS*100)} % "
         "censoring, outer 5-fold discovery/evaluation, v1.2 thresholds.\n",
         "| metric | n = 589 |" + (" n = 2000 (Gate 2c) |" if ref else ""),
         "|---|---|" + ("---|" if ref else "")]
    rows = [
        ("replications", f"{m['reps']}", None),
        ("null rate (no rule in any fold)", f"{m['null_reps']:.0f} %", "2–10 %"),
        ("rules per replication", f"{m['rules_per_rep']:.1f}", "2.9–3.0"),
        ("rules evaluated", f"{m['rules_evaluated']}", "—"),
        ("mean support", f"{m['support']:.0f}", "~127"),
        ("mean stability", f"{m['freq']:.2f}", "0.61–0.63"),
        ("bias", f"{m['bias']:+.4f}", "−0.012"),
        ("RMSE", f"{m['rmse']:.4f}", "—"),
        ("95 % coverage", f"{m['coverage']:.1f} %", "94.5–94.7 %"),
        ("mean CI width", f"{m['width']:.3f}", "0.35–0.55"),
        ("effect size tau", f"{m['tau_true']:.3f}", "0.21"),
        ("width / effect", f"{m['width']/max(m['tau_true'],1e-9):.1f}x", "1.7–2.6x"),
        ("abstain", f"{m['abstain']:.0f} %", "—"),
        ("TREAT", f"{m['treat']:.0f} %", "—"),
        ("sign accuracy (all rules)", f"{m['sign_ok']:.0f} %", "85–92 %"),
        ("TREAT verdicts", f"{m['n_treat']}", "—"),
        ("TREAT precision", f"{m['treat_prec']:.0f} %", "—"),
        ("AVOID verdicts", f"{m['n_avoid']}", "—"),
        ("AVOID precision", f"{m['avoid_prec']:.0f} %", "—"),
        ("rules with truly harmful effect", f"{m['harmful_rules']:.0f} %", "0 %"),
        ("mean IoU", f"{m['iou']:.2f}", "0.19–0.26"),
    ]
    for name, val, refv in rows:
        L.append(f"| {name} | {val} |" + (f" {refv if refv else '—'} |" if ref else ""))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=589)
    ap.add_argument("--t-star", type=float, default=dgp.TSTAR)
    ap.add_argument("--data", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stab-b", type=int, default=20)
    ap.add_argument("--signflip", action="store_true")
    args = ap.parse_args(argv)

    global SIGNFLIP
    SIGNFLIP = args.signflip
    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool, target=TARGET_S0)
    lam_c0 = dgp.calibrate_lam_c(MECH, CENS, pool, lam0, args.t_star,
                                 gen_kwargs={"tau": TAU})
    reps = min(args.reps, 8) if args.quick else args.reps

    # sanity: the matched DGP should reproduce ACTG175-scale quantities
    chk = dgp.generate(pool[:4000], MECH, lam_c0, lam0,
                       np.random.RandomState(1), args.t_star, tau=TAU,
                       effect_vec=(SIGN_SLOPE * pool[:4000, SIGN_IDX]
                                   if SIGNFLIP else None))
    print(f"matched DGP: tau={chk['tau_true']:.3f} (ACTG175 0.128), "
          f"censored-before-t*={100*chk['cens_before']:.1f} % (ACTG175 9.8 %), "
          f"mean S0={chk['S0'].mean():.3f} (ACTG175 0.735)", flush=True)

    from joblib import Parallel, delayed
    jobs = [(rep, lam0, lam_c0, pool, args.t_star, args.n, args.stab_b)
            for rep in range(reps)]
    out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
    df = pd.DataFrame([r for ch in out for r in ch if r])
    df.to_csv(os.path.join(RESULTS, ("gate2d_replications_signflip.csv" if SIGNFLIP
               else "gate2d_replications.csv")), index=False)
    m = summarise(df)
    pd.DataFrame([m]).to_csv(os.path.join(RESULTS, ("gate2d_summary_signflip.csv" if SIGNFLIP
                      else "gate2d_summary.csv")),
                             index=False)
    md = to_markdown(m, ref=True)
    with open(os.path.join(RESULTS, ("gate2d_summary_signflip.md" if SIGNFLIP
              else "gate2d_summary.md")), "w",
              encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return m


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

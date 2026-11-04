"""Gate 4c - crossing hazards against the ACTUAL SCRE implementation.

Question
--------
Gate 1c found a pooled / shared-basis Cox is biased 16-45 % under crossing
hazards. That was measured against a pooled-Cox *proxy*. This gate asks whether
the real SCRE implementation inherits the bias, and whether that bias is
structural or an artefact of one tuning choice.

Unit of analysis
----------------
The contrast is the **average treatment effect at t\* = 730**,

    tau = E[ S_1(t*|X) - S_0(t*|X) ],

estimated once per replication by each method. This is deliberately NOT a
per-patient comparison: CISCaRL's per-patient influence contributions have
sd ~ 0.9 by construction and are not a prediction function, so comparing them to
SCRE's model prediction would be meaningless. Patient-level accuracy is Gate 4's
job; the structural bias question is an ATE question.

Fairness protocol (fixed before running)
----------------------------------------
SCRE is tuned by validation concordance on a 25 % slice of the training data.
That criterion targets prognosis, not HTE, so one selected alpha cannot separate
"structurally biased" from "badly tuned for this task". Hence:

  * primary comparison = SCRE at the validation-selected alpha
  * alpha sweep        = up to 10 alphas spanning the admissible range
                         (1..10 rule-interaction terms), evenly spaced
  * no oracle alpha    (no deployable analogue)

Pre-registered prediction
------------------------
A Cox model with time-constant covariates forces a constant log-hazard ratio, so
the bias should PERSIST across the sweep:

    share of admissible alphas with |bias| / tau >= 0.10 should be high.

If some admissible alpha removes the bias, the structural claim is dropped and
the earlier proxy result is recorded as a tuning artefact.

Usage
-----
    python -m ciscarl2.gate4c --reps 30 --n 589
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import dgp
from .gate2 import TAU, pseudo_outcome
from .gate4 import REPO, _scre_design, scre_select_alpha
from .estimators import crossfit_nuisances, phi_on_holdout, truncate_weights

if REPO not in sys.path:
    sys.path.insert(0, REPO)
from scre import SurvivalCausalRuleEnsemble          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
from .paths import data_path

DEFAULT_DATA = data_path()
TSTAR = 730.0
SWEEP = 10
BIAS_PERSIST = 0.10


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def _scre_tau_at(scre, X, alpha, t_star):
    cox = scre.model_
    d1 = _scre_design(scre, X, np.ones(len(X)))
    d0 = _scre_design(scre, X, np.zeros(len(X)))
    u = cox.unique_times_.astype(float)
    out = []
    for d in (d1, d0):
        sf = cox.predict_survival_function(d, alpha=alpha, return_array=True)
        out.append(np.array([np.interp(t_star, u, sf[i]) for i in range(len(d))]))
    return out[0] - out[1]


def run_one(job):
    (rep, lam0, lam_c0, pool, n) = job
    rng = np.random.RandomState(90000 + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate_nonph(X, "covariate", lam_c0, lam0, rng, TSTAR)
    A, T, E = sim["A"], sim["Tobs"], sim["E"]

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=rep, stratify=A)
    inner, val = train_test_split(np.arange(len(tr)), test_size=0.25,
                                  random_state=rep, stratify=A[tr])
    f, v = tr[inner], tr[val]

    true_ate = float((sim["S1"] - sim["S0"])[te].mean())

    # ---- CISCaRL: AIPW ATE on the held-out patients ----
    p1, p0 = phi_on_holdout(X[tr], A[tr], T[tr], E[tr],
                            X[te], A[te], T[te], TSTAR, seed=rep)
    cis_ate = float((p1 - p0).mean())

    # ---- SCRE: fit, select alpha, predict ----
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        X[f], A[f], T[f], E[f], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A[f], T[f], TSTAR, muD, GD)

    scre = SurvivalCausalRuleEnsemble(
        alpha=0.5, pred_n_estimators=100, progn_n_estimators=100,
        rule_min_support=20, max_rules=200, max_rule_conditions=3,
        max_selected_rules=10, random_state=42)
    scre.fit(X[f], psi, np.isfinite(psi),
             feature_names=[f"x{i}" for i in range(X.shape[1])],
             time=T[f], event=E[f].astype(bool), treatment=A[f])

    pick = scre_select_alpha(scre, X[v], A[v], T[v], E[v].astype(bool), TSTAR)
    if pick is None:
        return []
    _, alpha_sel, _ = pick

    cox = scre.model_
    n_rules = len(scre.candidates_)
    p_ = X.shape[1]
    start = 2 * p_ + n_rules
    blk = slice(start, start + n_rules)
    admissible = []
    for a in cox.alphas_:
        coef, _ = cox._get_coef(a)
        k = int(np.sum(np.abs(coef[blk]) > 1e-6)) if n_rules else 0
        if 1 <= k <= 10:
            admissible.append(a)
    if not admissible:
        admissible = [alpha_sel]
    idx = np.linspace(0, len(admissible) - 1, min(SWEEP, len(admissible))).astype(int)
    sweep = sorted({admissible[i] for i in idx})

    # per-replication audit trail for the alpha claim
    coef_sel, _ = cox._get_coef(alpha_sel)
    n_terms_sel = int(np.sum(np.abs(coef_sel[blk]) > 1e-6)) if n_rules else 0
    in_adm = int(any(np.isclose(alpha_sel, a) for a in admissible))

    rows = [dict(rep=rep, n=n, method="CISCaRL", alpha=np.nan,
                 est=cis_ate, truth=true_ate, err=cis_ate - true_ate,
                 selected=1, n_terms=np.nan, in_adm=in_adm,
                 k_sel=n_terms_sel)]
    for a in sorted(set(sweep) | {alpha_sel}):
        try:
            t_ = _scre_tau_at(scre, X[te], a, TSTAR)
        except Exception:
            continue
        est = float(t_.mean())
        rows.append(dict(rep=rep, n=n, method="SCRE", alpha=float(a),
                         est=est, truth=true_ate, err=est - true_ate,
                         selected=int(a == alpha_sel),
                         n_terms=(n_terms_sel if a == alpha_sel else np.nan)))
    return rows


def summarise(df):
    """Marginals and the paired contrast are computed on the SAME rep subset.

    d = MAE(SCRE) - MAE(CISCaRL). Positive => SCRE has the larger absolute
    error => CISCaRL is closer.
    """
    out = []
    for n, g in df.groupby("n"):
        cis = g[g.method == "CISCaRL"].set_index("rep")
        scr = g[g.method == "SCRE"]
        sel = scr[scr.selected == 1].set_index("rep")

        per = scr.assign(rel=lambda d: d.err.abs() / d.truth.abs()) \
                 .groupby("rep").rel.apply(
                     lambda x: float(np.mean(x >= BIAS_PERSIST)))

        common = sorted(set(sel.index.tolist()) & set(cis.index.tolist()))
        a = cis.loc[common, "err"].abs().to_numpy()
        b = sel.loc[common, "err"].abs().to_numpy()
        d = b - a
        rs = np.random.RandomState(0)
        bs = [rs.choice(d, len(d), replace=True).mean() for _ in range(5000)]

        cis_p = cis.loc[common]
        sel_p = sel.loc[common]
        out.append(dict(
            n=n, reps=len(common), reps_total=int(cis.shape[0]),
            zero_term_share=100 * float(
                (sel.n_terms.dropna() == 0).mean()) if sel.n_terms.notna().any() else np.nan,
            adm_share=100 * float(cis_p.in_adm.mean()) if "in_adm" in cis_p else np.nan,
            k_min=float(cis_p.k_sel.min()) if "k_sel" in cis_p else np.nan,
            k_max=float(cis_p.k_sel.max()) if "k_sel" in cis_p else np.nan,
            truth=float(cis_p.truth.mean()),
            cis_bias=float(cis_p.err.mean()),
            cis_mae=float(cis_p.err.abs().mean()),
            scre_bias=float(sel_p.err.mean()),
            scre_mae=float(sel_p.err.abs().mean()),
            scre_bias_sweep_min=float(scr.groupby("rep").err.apply(
                lambda x: x.abs().min()).mean()),
            persist_frac=float(per.mean()),
            d_abs_err=float(d.mean()),
            marginal_check=float(sel_p.err.abs().mean() -
                                 cis_p.err.abs().mean()),
            d_lo=float(np.percentile(bs, 2.5)),
            d_hi=float(np.percentile(bs, 97.5))))
    return pd.DataFrame(out)


def to_markdown(s):
    L = ["# Gate 4c - crossing hazards vs the actual SCRE implementation\n",
         "Contrast = average treatment effect at t* = 730, estimated once per "
         "replication. **All MAEs are on the same paired rep subset**, so "
         "`MAE(SCRE) - MAE(CISCaRL)` reconciles exactly with d.\n",
         "d = MAE(SCRE) - MAE(CISCaRL): **positive favours CISCaRL**.\n",
         "| n | reps (of total) | zero-term reps | true tau | CISCaRL bias | "
         "CISCaRL MAE | SCRE bias | SCRE MAE | min abs bias over sweep | "
         "persist frac | d [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.n} | {r.reps} ({r.reps_total}) | "
                 f"{r.zero_term_share:.0f} % | {r.truth:.4f} | "
                 f"{r.cis_bias:+.4f} | {r.cis_mae:.4f} | {r.scre_bias:+.4f} | "
                 f"{r.scre_mae:.4f} | {r.scre_bias_sweep_min:.4f} | "
                 f"{100*r.persist_frac:.0f} % | "
                 f"{r.d_abs_err:+.4f} [{r.d_lo:+.4f}, {r.d_hi:+.4f}] |")
    L.append("")
    L.append(f"persist frac = share of admissible alphas with |bias| / |tau| "
             f">= {BIAS_PERSIST:.0%}. zero-term reps = share where the "
             "validation-selected alpha retained no treatment-interaction term.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n", type=int, default=589)
    ap.add_argument("--data", default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen=dgp.generate_nonph)

    from joblib import Parallel, delayed
    t0 = time.time()
    jobs = [(rep, lam0, lam_c0, pool, args.n) for rep in range(args.reps)]
    out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
    df = pd.DataFrame([r for ch in out for r in ch if r])
    tag = f"n{args.n}"
    df.to_csv(os.path.join(RESULTS, f"gate4c_replications_{tag}.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, f"gate4c_summary_{tag}.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, f"gate4c_summary_{tag}.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)
    print(f"\n{args.reps} reps in {time.time()-t0:.0f}s")
    print(md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

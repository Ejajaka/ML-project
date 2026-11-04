"""Gate 4 - CISCaRL v1.4 vs the actual SCRE implementation.

Matched conditions: same simulated dataset, same train/test split, same horizon
(t* = 730 d), same evaluation patients. Two DGPs (all-beneficial and sign-flip)
x two sample sizes.

Two questions are reported separately
-------------------------------------
A. **Estimation accuracy** - patient-level error in tau_i. SCRE is expected to
   win here: it is a penalised Cox likelihood using all event times, while CISCaRL
   averages inverse-probability-weighted residuals.

B. **Decision reliability** - can the method say "I don't know"? SCRE returns a
   point estimate for every patient and therefore commits 100 % of the time.
   CISCaRL returns TREAT / AVOID / INSUFFICIENT per rule, so its coverage is
   below 100 % by construction. Precision is reported against the base rate.

The comparison is only meaningful if both are evaluated on the same patients.

Usage
-----
    python -m ciscarl2.gate4 --reps 20 --n 589
    python -m ciscarl2.gate4 --reps 20 --n 2000 --signflip
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (crossfit_nuisances, ipcw_outcome, phi_on_holdout,
                         truncate_weights)
from .gate2 import TAU, pseudo_outcome
from .rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

# The SCRE baseline is an external package. Point SCRE_PATH at the directory
# that contains it (the folder holding `scre/`), or install it on sys.path.
_CANDIDATES = [os.environ.get("SCRE_PATH", ""),
               os.path.join(os.path.expanduser("~"), "Downloads",
                            "Machine-Learning-Project", "python")]
for _c in _CANDIDATES:
    if _c and os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)
# kept for other modules that import it
REPO = next((c for c in _CANDIDATES if c and os.path.isdir(c)), _CANDIDATES[-1])
try:                                                        # noqa: E402
    from scre import SurvivalCausalRuleEnsemble, _eval_rule
    _SCRE_IMPORT_ERROR = None
except Exception as _exc:                                   # noqa: BLE001
    SurvivalCausalRuleEnsemble = None
    _eval_rule = None
    _SCRE_IMPORT_ERROR = _exc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
from .paths import data_path

DEFAULT_DATA = data_path()
TSTAR = 730.0
THR, MIN_SUPPORT = 0.50, 60
SIGN_SLOPE, SIGN_IDX = 0.60, 0


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def verdict(lo, hi):
    return "TREAT" if lo > 0 else "AVOID" if hi < 0 else "INSUFFICIENT"



def _scre_design(scre, X, a):
    """Rebuild SCRE's design the same way scre.predict does (rules re-evaluated)."""
    Xs = (np.asarray(X, float) - scre.linear_mean_) / scre.linear_std_
    R = np.column_stack([_eval_rule(cond, np.asarray(X, float)).astype(float)
                         for cond, _ in scre.candidates_])
    z = np.asarray(a, float).reshape(-1, 1)
    return np.hstack([Xs, R, Xs * z, R * z])


def scre_select_alpha(scre, Xv, Av, Tv, Ev, t_star, max_terms=10):
    """Choose the Coxnet alpha by validation C-index instead of the path heuristic.

    `scre.py` walks the path from the MOST regularised end and stops at the first
    model retaining >= 1 rule-interaction term. At that end every other term is
    already zero, so the selected model can predict "no treatment effect at all".
    Validation concordance is the fairer and more standard choice.
    """
    try:
        from sksurv.metrics import concordance_index_censored
    except Exception:
        return None
    cox = scre.model_
    p_ = np.asarray(Xv).shape[1]
    n_rules = len(scre.candidates_)
    dv = _scre_design(scre, Xv, Av)
    if not np.all(np.isfinite(dv)):
        return None
    start = 2 * p_ + n_rules
    blk = slice(start, start + n_rules)
    u = cox.unique_times_.astype(float)

    best = None
    for a in cox.alphas_:
        try:
            coef, _ = cox._get_coef(a)
        except Exception:
            continue
        n_terms = int(np.sum(np.abs(coef[blk]) > 1e-6)) if n_rules else 0
        if n_rules and not (1 <= n_terms <= max_terms):
            continue
        try:
            sf = cox.predict_survival_function(dv, alpha=a, return_array=True)
            s_obs = np.array([np.interp(t_star, u, sf[i]) for i in range(len(dv))])
            ci = concordance_index_censored(Ev.astype(bool), Tv, 1.0 - s_obs)[0]
        except Exception:
            continue
        if best is None or ci > best[0]:
            best = (ci, a, coef)
    return best

def run_one(job):
    (rep, kind, signflip, lam0, lam_c0, pool, n, stab_B) = job
    rng = np.random.RandomState(80000 + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    eff = SIGN_SLOPE * X[:, SIGN_IDX] if signflip else None
    sim = dgp.generate(X, kind, lam_c0, lam0, rng, TSTAR, tau=TAU, effect_vec=eff)
    A, T, E = sim["A"], sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=rep, stratify=A)

    # ---- discovery on train ----
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        X[tr], A[tr], T[tr], E[tr], TSTAR, seed=rep, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    psi = pseudo_outcome(A[tr], T[tr], TSTAR, muD, GD)

    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=MIN_SUPPORT, random_state=0,
                               n_jobs=-1).fit(X[tr], psi)
    cands = dedup(rules_from_ensemble(rf, X[tr], MIN_SUPPORT, max_trees=30))
    subs = []
    r2 = np.random.RandomState(rep + 1)
    for _ in range(stab_B):
        ii = r2.choice(len(tr), len(tr) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, MIN_SUPPORT // 2),
                                  random_state=0, n_jobs=-1).fit(X[tr][ii], psi[ii])
        subs.append(rules_from_ensemble(t, X[tr], 5, max_trees=15))
    freq = stability_scores(cands, subs, X[tr]) if cands else np.array([])
    sel = [dict(cond=c, mask=m, score=f)
           for (c, m), f in zip(cands, freq) if f >= THR]
    sel = prune(sel, X[tr], max_rules=99, overlap=0.8) if sel else []

    # ---- CISCaRL evaluation on test ----
    p1, p0 = phi_on_holdout(X[tr], A[tr], T[tr], E[tr],
                            X[te], A[te], T[te], TSTAR, seed=rep)
    dphi = p1 - p0
    tau_true = tau_i[te]

    default = float(dphi.mean())
    dse = float(np.sqrt(((dphi - default) ** 2).sum()) / len(te))
    dq = float(tdist.ppf(0.975, len(te) - 1))
    cis_tau = np.full(len(te), default)
    cis_verdict = np.full(len(te), verdict(default - dq * dse, default + dq * dse),
                          dtype=object)
    assigned = np.zeros(len(te), bool)          # most-specific rule wins
    for r in sorted(sel, key=lambda d: -len(d["cond"])):
        m = eval_rule(r["cond"], X[te])
        nr = int(m.sum())
        if nr < 5:
            continue
        t_r = float(dphi[m].mean())
        s_r = float(np.sqrt(((dphi[m] - t_r) ** 2).sum()) / nr)
        q = float(tdist.ppf(0.975, max(nr - 1, 1)))
        take = m & ~assigned
        cis_tau[take] = t_r
        cis_verdict[take] = verdict(t_r - q * s_r, t_r + q * s_r)
        assigned |= take

    # ---- SCRE on the same split (fair alpha selection on a validation slice) ----
    inner, val = train_test_split(np.arange(len(tr)), test_size=0.25,
                                  random_state=rep, stratify=A[tr])
    f, v = tr[inner], tr[val]
    try:
        scre = SurvivalCausalRuleEnsemble(
            alpha=0.5, pred_n_estimators=100, progn_n_estimators=100,
            rule_min_support=20, max_rules=200, max_rule_conditions=3,
            max_selected_rules=10, random_state=42)
        scre.fit(X[f], psi[inner], np.isfinite(psi[inner]),
                 feature_names=[f"x{i}" for i in range(X.shape[1])],
                 time=T[f], event=E[f].astype(bool), treatment=A[f])
        pick = scre_select_alpha(scre, X[v], A[v], T[v], E[v].astype(bool), TSTAR)
        if pick is not None:
            scre.val_cindex_, alpha_pick, coef_pick = pick
            scre.alpha_ = alpha_pick
            scre.alpha_final_ = alpha_pick
            scre.coef_ = coef_pick
        scre.t_star_ = TSTAR
        scre_tau = np.asarray(scre.predict(X[te]), float)
        n_scre_rules = len(getattr(scre, "selected_rules_", []))
    except Exception as exc:
        scre_tau = np.full(len(te), np.nan)
        n_scre_rules = np.nan
        print(f"    [SCRE failed: {exc}]")

    def acc(pred):
        ok = np.isfinite(pred)
        e = pred[ok] - tau_true[ok]
        sp = (spearmanr(pred[ok], tau_true[ok])[0]
              if np.std(pred[ok]) > 1e-12 else 0.0)
        return dict(mae=float(np.abs(e).mean()),
                    rmse=float(np.sqrt((e ** 2).mean())),
                    bias=float(e.mean()), spearman=float(sp),
                    sign_acc=float(np.mean((pred[ok] > 0) == (tau_true[ok] > 0))))

    common = dict(rep=rep, n=n, signflip=int(signflip), mechanism=kind,
                  n_rules=len(sel), scre_rules=n_scre_rules,
                  base_rate=float(np.mean(tau_true > 0)))
    rows = []
    for meth, pred in (("CISCaRL", cis_tau), ("SCRE", scre_tau)):
        rows.append(dict(common, method=meth, **acc(pred)))

    comm = cis_verdict != "INSUFFICIENT"
    pos = cis_verdict == "TREAT"
    for r in rows:
        r.update(
            cis_coverage=100 * comm.mean(),
            cis_precision=(100 * np.mean(tau_true[pos] > 0)) if pos.any() else np.nan,
            cis_n_commit=int(comm.sum()))
    return rows


def summarise(df):
    out = []
    for (n, sf, m), g in df.groupby(["n", "signflip", "mechanism"]):
        r = dict(n=n, signflip=int(sf), mechanism=m, reps=int(g.rep.nunique()),
                 base_rate=100 * g.base_rate.mean(),
                 rules=g.n_rules.mean(), scre_rules=g.scre_rules.mean())
        for meth in ("CISCaRL", "SCRE"):
            s = g[g.method == meth]
            for k in ("mae", "rmse", "bias", "spearman", "sign_acc"):
                r[f"{meth}_{k}"] = s[k].mean()
        r["cis_coverage"] = g.cis_coverage.mean()
        r["cis_precision"] = g.cis_precision.mean()
        out.append(r)
    return pd.DataFrame(out)


def to_markdown(s):
    L = ["# Gate 4 - CISCaRL v1.4 vs SCRE (actual implementation)\n",
         "Matched splits, horizon, and evaluation patients. Two questions separated.\n",
         "## A. Estimation accuracy (patient-level tau)\n",
         "| n | DGP | method | MAE | RMSE | bias | Spearman | sign acc | rules |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        dgp_name = "sign-flip" if r.signflip else "all-beneficial"
        for meth in ("CISCaRL", "SCRE"):
            L.append(f"| {r.n} | {dgp_name} | {meth} | {r[f'{meth}_mae']:.4f} | "
                     f"{r[f'{meth}_rmse']:.4f} | {r[f'{meth}_bias']:+.4f} | "
                     f"{r[f'{meth}_spearman']:+.2f} | "
                     f"{100*r[f'{meth}_sign_acc']:.0f} % | "
                     f"{r.rules if meth=='CISCaRL' else r.scre_rules:.1f} |")
    L.append("\n## B. Decision reliability\n")
    L.append("| n | DGP | base rate | CISCaRL coverage | CISCaRL precision | "
             "SCRE coverage | SCRE sign acc |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in s.iterrows():
        dgp_name = "sign-flip" if r.signflip else "all-beneficial"
        L.append(f"| {r.n} | {dgp_name} | {r.base_rate:.0f} % | "
                 f"{r.cis_coverage:.1f} % | {r.cis_precision:.0f} % | "
                 f"100 % | {100*r.SCRE_sign_acc:.0f} % |")
    L.append("\nSCRE commits for every patient by construction (no interval). "
             "CISCaRL coverage is the share receiving TREAT or AVOID.\n")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n", type=int, default=589)
    ap.add_argument("--data", default=None)
    ap.add_argument("--signflip", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--stab-b", type=int, default=20)
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})
    from joblib import Parallel, delayed
    jobs = [(rep, "covariate", args.signflip, lam0, lam_c0, pool, args.n,
             args.stab_b) for rep in range(args.reps)]
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
    df = pd.DataFrame([r for ch in out for r in ch if r])
    tag = f"n{args.n}{'_signflip' if args.signflip else ''}"
    df.to_csv(os.path.join(RESULTS, f"gate4_replications_{tag}.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, f"gate4_summary_{tag}.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, f"gate4_summary_{tag}.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)
    print(f"\n{args.reps} reps in {time.time()-t0:.0f}s")
    print(md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

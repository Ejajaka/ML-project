"""Gate 3 - delta sensitivity analysis under a known MNAR mechanism.

Two pre-registered checks
-------------------------
1. **delta = 0 identity.** The adjusted estimator must reproduce the baseline to
   machine precision. Otherwise the sensitivity parameter is entangled with the
   primary analysis.
2. **Known-MNAR recovery.** Under a censoring mechanism whose departure from
   MAR is controlled by psi, the delta that recovers the truth must (a) be
   approximately 0 when psi = 0, (b) move monotonically with psi, and (c) be
   stable across rules and replications. A sensitivity analysis that merely
   produces a range of effects is not validated by producing a range.

MNAR mechanism
--------------
Patients whose latent event time exceeds t* have their censoring hazard
multiplied by exp(psi). The tilt uses the latent T, so no function of observed
covariates can recover the correct weights. lambda_c0 is recalibrated at every
psi so that the marginal fraction censored before t* stays at 25 %: psi is the
only thing that varies.

Adjustment (v1.1 specification, arm-specific)
---------------------------------------------
    phi_a(δ) = phi_a(0) + 1{i in C, A_i = a} * (p_tilde_a - p_hat_a)
    p_tilde_a = expit(logit(p_hat_a) + δ),  p_hat_a = S_a(t*|X)

This is a first-order imputation perturbation, NOT an EIF-derived estimator
under MNAR. The gate exists to find out whether that is good enough.

Usage
-----
    python -m ciscarl2.gate3 --reps 200
    python -m ciscarl2.gate3 --quick --reps 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t as tdist

from . import dgp
from .estimators import (aipw, crossfit_nuisances, ipcw_outcome,
                         truncate_weights)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "parts_gate3")
from .paths import data_path

DEFAULT_DATA = data_path()

CELLS = ("covariate", "independent")
PSI = (0.0, 0.5, 1.0, 1.5)
GRID = np.round(np.arange(-2.0, 2.0001, 0.05), 3)
TARGET_CENS = 0.25
TAU = (0.8, 0.5)


def _find_data(explicit):
    for c in [explicit, os.environ.get("ACTG175_PATH"), DEFAULT_DATA,
              os.path.join(ROOT, "data", "ACTG175.csv"), "ACTG175.csv"]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("ACTG175.csv not found; pass --data.")


def phi_at(X, A, Tobs, E, t_star, mu, G, delta):
    """Per-patient phi for both arms at a given delta."""
    Y = ipcw_outcome(Tobs, t_star, G)
    phi1 = mu[:, 1] + (A == 1) / 0.5 * (Y - mu[:, 1])
    phi0 = mu[:, 0] + (A == 0) / 0.5 * (Y - mu[:, 0])
    cens = (Tobs <= t_star) & (E == 0)
    out = []
    for a, phi in ((1, phi1), (0, phi0)):
        sel = cens & (A == a)
        if sel.any() and delta != 0.0:
            p = expit(logit(np.clip(mu[sel, a], 1e-6, 1 - 1e-6)) + delta)
            phi = phi.copy()
            phi[sel] = phi[sel] + (p - mu[sel, a])
        out.append(phi)
    return out[0], out[1]



def phi_g_tilt(X, A, Tobs, t_star, mu, G, delta):
    """Alternative parameterisation: tilt the CENSORING WEIGHTS, not the outcome.

    G_eff(t*|X) = G(t*|X) ** exp(delta), i.e. a proportional-hazard tilt of the
    censoring process. delta = 0 gives the baseline; delta = psi recovers the
    truth exactly when the MNAR mechanism is the psi-tilt used by the DGP.
    """
    Gs = np.clip(G ** np.exp(delta), 1e-6, 1.0)
    Y = ipcw_outcome(Tobs, t_star, Gs)
    phi1 = mu[:, 1] + (A == 1) / 0.5 * (Y - mu[:, 1])
    phi0 = mu[:, 0] + (A == 0) / 0.5 * (Y - mu[:, 0])
    return phi1, phi0


def curve(phi1, phi0, mask):
    """tau(delta) and SE on a fixed mask for a precomputed phi pair."""
    d = (phi1 - phi0)[mask]
    nr = int(mask.sum())
    tau = float(d.mean())
    se = float(np.sqrt(((d - tau) ** 2).sum()) / nr)
    return tau, se, nr


def run_one(job):
    (ci, pj, rep, kind, psi, lam0, lam_c0, pool, t_star, n) = job
    rng = np.random.RandomState(60000 * ci + 1000 * pj + rep)
    X = pool[rng.choice(len(pool), n, replace=True)]
    sim = dgp.generate_mnar(X, kind, lam_c0, lam0, rng, t_star, tau=TAU, psi=psi)
    A, Tobs, E, S0, S1 = sim["A"], sim["Tobs"], sim["E"], sim["S0"], sim["S1"]

    mu, G, _, fid, fcaps, _ = crossfit_nuisances(
        X, A, Tobs, E, t_star, seed=rep, return_folds=True)
    G, _ = truncate_weights(G, fid, fcaps)

    masks = {
        "all": np.ones(len(X), bool),
        "age>40": X[:, dgp.COVARIATES.index("age")] > np.median(X[:, dgp.COVARIATES.index("age")]),
        "cd40<median": X[:, 0] < np.median(X[:, 0]),
    }
    tau_i = S1 - S0
    tau_base, _, _, _ = aipw(X, A, Tobs, t_star, mu, G)   # reference for delta=0

    rows = []
    for name, mask in masks.items():
        if mask.sum() < 50:
            continue
        t_true = float(tau_i[mask].mean())
        nr = int(mask.sum())

        # --- delta = 0 identity check (independent recomputation) ---
        p1_0, p0_0 = phi_at(X, A, Tobs, E, t_star, mu, G, 0.0)
        tau0, se0, _ = curve(p1_0, p0_0, mask)
        q = float(tdist.ppf(0.975, max(nr - 1, 1)))

        taus = np.empty(len(GRID))
        for gi, dl in enumerate(GRID):
            p1, p0 = phi_at(X, A, Tobs, E, t_star, mu, G, dl)
            taus[gi] = (p1 - p0)[mask].mean()

        taus_g = np.empty(len(GRID))
        for gi, dl in enumerate(GRID):
            q1, q0 = phi_g_tilt(X, A, Tobs, t_star, mu, G, dl)
            taus_g[gi] = (q1 - q0)[mask].mean()

        k = int(np.argmin(np.abs(taus - t_true)))
        d_truth = float(GRID[k])
        kg = int(np.argmin(np.abs(taus_g - t_true)))
        d_truth_g = float(GRID[kg])
        tau_g, se_g, _ = curve(*phi_g_tilt(X, A, Tobs, t_star, mu, G, d_truth_g), mask)
        tau_at_truth, se_at_truth, _ = curve(*phi_at(X, A, Tobs, E, t_star, mu, G, d_truth), mask)

        # delta at which the estimate crosses zero (the practically relevant tip)
        d_null = np.nan
        s = np.sign(taus)
        if s[0] != s[-1]:
            idx = np.where(s != s[0])[0]
            if len(idx):
                j = int(idx[0])
                d_null = float(GRID[j - 1] + (0 - taus[j - 1]) *
                               (GRID[j] - GRID[j - 1]) / (taus[j] - taus[j - 1]))

        rows.append(dict(
            cell=ci, mechanism=kind, psi=psi, rep=rep, rule=name,
            support=nr, tau_true=t_true,
            tau0=tau0, se0=se0, bias0=tau0 - t_true,
            identity_err=abs(tau0 - tau_base) if name == "all" else np.nan,
            covered0=float(abs(tau0 - t_true) <= q * se0),
            d_truth=d_truth, tau_dtruth=tau_at_truth, se_dtruth=se_at_truth,
            covered_dtruth=float(abs(tau_at_truth - t_true) <= q * se_at_truth),
            err_dtruth=tau_at_truth - t_true,
            d_null=d_null,
            slope=float((taus[-1] - taus[0]) / (GRID[-1] - GRID[0])),
            d_truth_g=d_truth_g,
            slope_g=float((taus_g[-1] - taus_g[0]) / (GRID[-1] - GRID[0])),
            err_g=tau_g - t_true,
            covered_g=float(abs(tau_g - t_true) <= q * se_g),
            cens_before=sim["cens_before"]))
    return rows


def summarise(df):
    out = []
    for (ci, m, psi, rule), g in df.groupby(["cell", "mechanism", "psi", "rule"]):
        out.append(dict(
            cell=int(ci), mechanism=m, psi=psi, rule=rule, reps=len(g),
            cens=100 * g.cens_before.mean(),
            bias0=g.bias0.mean(), mc0=g.bias0.std(ddof=1) / np.sqrt(len(g)),
            cover0=100 * g.covered0.mean(),
            d_truth=g.d_truth.mean(), d_truth_sd=g.d_truth.std(ddof=1),
            err_dtruth=g.err_dtruth.mean(),
            cover_dtruth=100 * g.covered_dtruth.mean(),
            slope=g.slope.mean(), slope_g=g.slope_g.mean(),
            d_truth_g=g.d_truth_g.mean(), d_truth_g_sd=g.d_truth_g.std(ddof=1),
            err_g=g.err_g.mean(), cover_g=100 * g.covered_g.mean(),
            identity_err=float(g.identity_err.max()) if g.identity_err.notna().any() else 0.0,
            d_null=g.d_null.mean(), null_frac=100 * g.d_null.notna().mean()))
    return pd.DataFrame(out).sort_values(["mechanism", "psi", "rule"])


def to_markdown(s):
    L = ["# Gate 3 - delta sensitivity under known MNAR\n",
         "psi multiplies the censoring hazard for patients whose latent T > t*. "
         "Censoring is recalibrated to 25 % at every psi, so psi is the only "
         "thing that varies.\n",
         "| cell | psi | rule | cens | bias at delta=0 | cover delta=0 | "
         "**delta that recovers truth** | sd | residual err | cover at that delta | "
         "slope (mu tilt) | **slope (G tilt)** | **delta* (G tilt)** | err | cover |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    worst = s.identity_err.max()
    L.append(f"\n**delta = 0 identity check: max |tau(0) - baseline AIPW| = {worst:.3e}** "
             "(must be machine precision).\n")
    for _, r in s.iterrows():
        L.append(f"| {r.mechanism} | {r.psi:.1f} | {r.rule} | {r.cens:.0f} % | "
                 f"{r.bias0:+.4f} +/- {r.mc0:.4f} | {r.cover0:.0f} % | "
                 f"**{r.d_truth:+.2f}** | {r.d_truth_sd:.2f} | "
                 f"{r.err_dtruth:+.4f} | {r.cover_dtruth:.0f} % | {r.slope:+.3f} | "
                 f"**{r.slope_g:+.2f}** | **{r.d_truth_g:+.2f}** | "
                 f"{r.err_g:+.4f} | {r.cover_g:.0f} % |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--t-star", type=float, default=dgp.TSTAR)
    ap.add_argument("--data", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(_find_data(args.data))
    lam0 = dgp.lambda0_for_target_survival(pool)
    cells = list(CELLS)
    psis = list(PSI)
    reps = args.reps
    if args.quick:
        cells, psis, reps = cells[:1], psis[:2], min(reps, 10)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    print("calibrating censoring at 25 % for each psi ...", flush=True)
    lam_c = {}
    for c in cells:
        for psi in psis:
            lam_c[(c, psi)] = dgp.calibrate_lam_c(
                c, TARGET_CENS, pool, lam0, args.t_star,
                gen=dgp.generate_mnar,
                gen_kwargs={"tau": TAU, "psi": psi})
            print(f"  {c:12s} psi={psi:.1f} -> lam_c0={lam_c[(c, psi)]:.6g}",
                  flush=True)

    from joblib import Parallel, delayed
    rows, t0 = [], time.time()
    for ci, c in enumerate(cells):
        for pj, psi in enumerate(psis):
            part = os.path.join(PARTS, f"g3_{c}_psi{psi}.csv")
            if os.path.exists(part) and not args.force:
                d = pd.read_csv(part)
                if d.rep.nunique() >= reps:
                    print(f"[skip] {c} psi={psi}", flush=True)
                    rows.append(d)
                    continue
            jobs = [(ci, pj, rep, c, psi, lam0, lam_c[(c, psi)], pool,
                     args.t_star, args.n) for rep in range(reps)]
            out = Parallel(n_jobs=args.jobs)(delayed(run_one)(j) for j in jobs)
            d = pd.DataFrame([r for ch in out for r in ch if r])
            d.to_csv(part, index=False)
            rows.append(d)
            print(f"[cell] {c:12s} psi={psi:.1f}  rows={len(d)}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(RESULTS, "gate3_replications.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(RESULTS, "gate3_summary.csv"), index=False)
    md = to_markdown(s)
    with open(os.path.join(RESULTS, "gate3_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return s


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

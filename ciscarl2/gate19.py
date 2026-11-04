"""Gate 19 - leaf-level commitment vs shrinkage.

Pre-registration: `results/gate19_PREREGISTRATION.md` (frozen before this ran).

Arms (paired on the same data within a replication):
    A  no commitment   leaf effect everywhere; always treat the high leaf
    B  hard commitment prediction: leaf effect only if it differs from the
                       marginal (paired IF for tau_L - tau); otherwise constant.
                       treatment: high leaf only if tau_L > 1.645 * SE_L
    C  EB shrinkage    leaf effects shrunk toward the marginal; no abstention

Cells: (kappa=0, conditional censoring) and (kappa=3, marginal KM).
n = 8000, 40 fresh seeds, seeds 210100-210139.

Usage
-----
    python -m ciscarl2.gate19 --reps 2 --start 210900   # smoke, off-seed
    python -m ciscarl2.gate19 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (aipw, crossfit_nuisances, ipcw_outcome, km_survival,
                         phi_on_holdout, truncate_weights)
from .gate2 import pseudo_outcome
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate8 import best_jaccard
from .gate11 import _cart, _hi_leaf
from .gate14 import _calibrate, _config, _region, _simulate
from .rules import eval_rule

from .paths import data_path

DATA = data_path()
CELLS = ((0.0, "conditional", 210100), (3.0, "marginal", 210100))
ARMS = ("A", "B", "C")


# ---------------------------------------------------------------------------
def _leaf_table(leaves, XE, dphi):
    out = []
    for r in leaves:
        m = eval_rule(r["cond"], XE)
        ns = int(m.sum())
        if ns < 5:
            out.append(dict(cond=r["cond"], mask=m, eff=np.nan, se=np.nan, n=ns))
            continue
        eff = float(dphi[m].mean())
        se = float(np.sqrt(((dphi[m] - eff) ** 2).sum()) / ns)
        out.append(dict(cond=r["cond"], mask=m, eff=eff, se=se, n=ns))
    return out


def _assign_effs(tbl, XE, effs, default):
    tau = np.full(len(XE), default)
    assigned = np.zeros(len(XE), bool)
    for j in sorted(range(len(tbl)), key=lambda j: -len(tbl[j]["cond"])):
        m = tbl[j]["mask"]
        if m.sum() < 5:
            continue
        take = m & ~assigned
        tau[take] = effs[j]
        assigned |= take
    return tau


def _contrast(dphi, mask):
    """D = mean_S(phi) - mean(phi), with its paired influence-function SE."""
    n = len(dphi)
    ns = int(mask.sum())
    if ns < 5:
        return np.nan, np.nan
    p = ns / n
    psi = dphi * (np.where(mask, 1.0 / p, 0.0) - 1.0)
    D = float(psi.mean())
    se = float(np.sqrt(((psi - D) ** 2).sum()) / n)
    return D, se


def _phi_marginal(XD, AD, TD, ED, XE, AE, TE):
    Gtr = km_survival(TD, (ED == 0).astype(int), TSTAR)
    w = np.where(TD > TSTAR, 1.0, 0.0) / max(Gtr, 1e-6)
    w = w[w > 0]
    cap = min(float(np.quantile(w, 0.99)), 20.0) if w.size else 20.0
    G = max(Gtr, 1.0 / cap)
    from .estimators import _fit_predict_nuisances
    mu, _ = _fit_predict_nuisances(XD, AD, TD, ED, XE, TSTAR)
    Y = ipcw_outcome(TE, TSTAR, np.full(len(TE), G))
    phi1 = mu[:, 1] + (AE == 1) / 0.5 * (Y - mu[:, 1])
    phi0 = mu[:, 0] + (AE == 0) / 0.5 * (Y - mu[:, 0])
    return phi1, phi0


def run_rep(rep, kappa, cen, start, cfg, lam_c0):
    pool, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, lam0 = cfg
    seed = start + rep
    rng = np.random.RandomState(seed)
    n = 8000
    X = pool[rng.choice(len(pool), n, replace=True)]
    R = _region(X, cfg)
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    eff = np.where(R, _delta_for(EFFECT, s0), 0.0)
    A = rng.binomial(1, 0.5, n)
    sim = _simulate(X, A, lam_c0, lam0, rng, eff, R, kappa, cfg)
    T, E = sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    tr, te = train_test_split(np.arange(n), test_size=0.30,
                              random_state=seed, stratify=A)
    XD, XE = X[tr], X[te]
    AD, TD, ED = A[tr], T[tr], E[tr]
    AE, TE = A[te], T[te]
    truth = tau_i[te]
    regE = R[te]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    if cen == "conditional":
        G = truncate_weights(GD, fid, fcaps)[0]
        p1, p0 = phi_on_holdout(XD, AD, TD, ED, XE, AE, TE, TSTAR, seed=seed)
    else:
        # fold-specific marginal caps mirror the conditional path
        caps = {}
        for k in np.unique(fid):
            idx = np.where(fid == k)[0]
            ar = idx[TD[idx] > TSTAR]
            caps[int(k)] = {0.99: float(np.quantile(1.0 / Gmarg[ar], 0.99))
                            if ar.size > 5 else np.inf}
        G = truncate_weights(Gmarg, fid, caps)[0]
        p1, p0 = _phi_marginal(XD, AD, TD, ED, XE, AE, TE)
    dphi = p1 - p0
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
    _, _, q1, q0 = aipw(XD, AD, TD, TSTAR, muD, G)
    const = float((q1 - q0).mean())
    const_mae = float(np.abs(const - truth).mean())

    leaves = _cart(psi, XD, 3)[1]
    tbl = _leaf_table(leaves, XE, dphi)
    j_hi = max(range(len(leaves)),
               key=lambda j: float(psi[leaves[j]["mask"]].mean()))
    hi_mask = tbl[j_hi]["mask"]
    hi_true = float(truth[hi_mask].mean()) if hi_mask.any() else np.nan
    tau_all = float(dphi.mean())
    dfree = max(int(hi_mask.sum()) - 1, 1)
    tq = float(tdist.ppf(0.975, dfree))

    effs = np.array([t["eff"] for t in tbl], float)
    ses = np.array([t["se"] for t in tbl], float)
    ok = np.isfinite(effs)
    t2 = max(0.0, float(np.var(effs[ok]) - np.mean(ses[ok] ** 2))) \
        if ok.sum() > 1 else 0.0
    shrunk = np.where(ok, const + (effs - const) * t2 / (t2 + ses ** 2), const)

    # arm A
    predA = _assign_effs(tbl, XE, effs, const)
    # arm B: prediction commitment vs the marginal (paired IF)
    effB = effs.copy()
    n_commit = 0
    for j, t_ in enumerate(tbl):
        D, seD = _contrast(dphi, t_["mask"])
        if np.isfinite(D) and np.isfinite(seD) and seD > 0 and abs(D) > tq * seD:
            n_commit += 1
        else:
            effB[j] = const
    predB = _assign_effs(tbl, XE, effB, const)
    # arm C
    predC = _assign_effs(tbl, XE, shrunk, const)

    # treatment decision for the high leaf
    hi_eff = effs[j_hi]
    hi_se = ses[j_hi]
    commit_treat = bool(np.isfinite(hi_eff) and np.isfinite(hi_se) and
                        hi_eff > float(norm.ppf(0.95)) * hi_se)
    k = int(hi_mask.sum())
    oracle_k = float(np.sort(truth)[::-1][:k].mean()) if k else np.nan

    rows = []
    for arm, pred, treated in (("A", predA, True),
                               ("B", predB, commit_treat),
                               ("C", predC, True)):
        mae = float(np.abs(pred - truth).mean())
        cov = float(hi_mask.mean()) if treated else 0.0
        per = hi_true if treated else np.nan
        capt = (float(truth[hi_mask].sum() / truth.sum())
                if treated and truth.sum() > 0 else 0.0)
        eff_ratio = (per / oracle_k) if treated and oracle_k else np.nan
        rows.append(dict(
            rep=rep, n=n, kappa=kappa, censoring=cen, arm=arm,
            mae=mae, const_mae=const_mae, delta=mae - const_mae,
            treated=bool(treated), commit_treat=bool(commit_treat),
            n_commit=n_commit, n_leaves=len(tbl),
            coverage=cov, per_treated=per, oracle_at_k=oracle_k,
            efficiency=eff_ratio, captured=capt,
            hi_true=hi_true,
            null_flag=bool(treated and np.isfinite(hi_true) and hi_true <= 0.01),
            beneficial_flag=bool(treated and np.isfinite(hi_true) and hi_true > 0),
            hi_share=float(hi_mask.mean()),
            jaccard=best_jaccard(leaves, XE, regE)))
    return rows


def _boot(d, seed=0, B=5000):
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for (kappa, cen, arm), g in df.groupby(["kappa", "censoring", "arm"]):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        ntr = int(g.treated.sum())
        out.append(dict(
            kappa=kappa, censoring=cen, arm=arm, reps=len(g),
            mae=g.mae.mean(), const_mae=g.const_mae.mean(), delta=d.mean(),
            lo=lo, hi=hi, frac_better=100.0 * float((d < 0).mean()),
            abstention=100.0 * (1.0 - g.treated.mean()),
            coverage=g.coverage.mean(),
            per_treated=(g.loc[g.treated, "per_treated"].mean()
                         if ntr else np.nan),
            efficiency=(g.loc[g.treated, "efficiency"].mean()
                        if ntr else np.nan),
            captured=g.captured.mean(),
            null_rate=(100.0 * g.null_flag.sum() / ntr if ntr else np.nan),
            beneficial_rate=(100.0 * g.beneficial_flag.sum() / ntr
                             if ntr else np.nan),
            commit_frac=(100.0 * g.commit_treat.mean() if arm == "B" else np.nan),
            leaf_commit_frac=(100.0 * g.n_commit.mean() / g.n_leaves.mean()
                              if arm == "B" else np.nan)))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    cells = CELLS if args.start is None else \
        tuple((k, c, args.start) for k, c, _ in CELLS)
    lam_c = {k: _calibrate(cfg, k) for k, _, _ in cells}

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, k, c, s, cfg, lam_c[k])
        for k, c, s in cells for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate19_commit.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate19_summary.csv"), index=False)

    print(f"\n=== Gate 19, n=8000, {args.reps} reps per cell ===")
    print(f"{'kappa':>6}{'cens':>12}{'arm':>4}{'MAE':>9}{'const':>8}{'delta':>9}"
          f"{'95% CI':>21}{'%bet':>6}{'abst%':>7}{'cover':>7}{'per-tr':>8}"
          f"{'effic':>7}{'capt':>7}{'null%':>7}{'benef%':>8}")
    for _, r in res.sort_values(["kappa", "censoring", "arm"]).iterrows():
        print(f"{r.kappa:>6.0f}{r.censoring:>12}{r.arm:>4}{r.mae:>9.4f}"
              f"{r.const_mae:>8.4f}{r.delta:>+9.4f}"
              f"   [{r.lo:+.4f},{r.hi:+.4f}]{r.frac_better:>5.0f}%"
              f"{r.abstention:>6.0f}%{r.coverage:>7.3f}{r.per_treated:>+8.3f}"
              f"{r.efficiency:>7.2f}{r.captured:>7.2f}{r.null_rate:>7.0f}"
              f"{r.beneficial_rate:>8.0f}")

    joint = res[(res.kappa == 3.0) & (res.censoring == "marginal")]
    base = joint[joint.arm == "A"].iloc[0]
    print("\nPre-registered checks (joint cell, arms B and C):")
    adopted = []
    for arm in ("B", "C"):
        r = joint[joint.arm == arm].iloc[0]
        checks = {
            "null-leaf rate <= 10%": bool(r.null_rate <= 10.0),
            "mean delta < 0 and CI below 0": bool(r.delta < 0 and r.hi < 0),
            "efficiency >= 0.60": bool(r.efficiency >= 0.60),
            "beneficial rate within 5 pp of A": bool(
                r.beneficial_rate >= base.beneficial_rate - 5.0),
            "coverage >= 0.05": bool(r.coverage >= 0.05),
            "captured >= 0.15": bool(r.captured >= 0.15),
        }
        print(f"  arm {arm}:")
        for k, v in checks.items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if all(checks.values()):
            adopted.append(arm)
    print(f"\n-> {'ADOPT ' + ', '.join(adopted) if adopted else 'No commitment arm adopted'}")
    return df


if __name__ == "__main__":
    main()

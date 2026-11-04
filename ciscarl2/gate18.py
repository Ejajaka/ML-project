"""Gate 18 - joint outcome + censoring nuisance misspecification (2x2).

Pre-registration: `results/gate18_PREREGISTRATION.md`.

Cells (n = 8000, same simulated data within a replication):
    outcome  in {correct, kappa=3 (Gate 14 violation)}
    censoring in {conditional Cox (correct), marginal Kaplan-Meier (misspecified)}

Evaluation is not only MAE. For every cell we report:
  estimation   ATE estimate vs truth; high-leaf effect vs its true effect
  discovery    best-leaf Jaccard, purity, capture
  rule quality high-leaf true effect, sign, absolute error
  stability    half-sample split stability
  decision     coverage, per-treated benefit, oracle benefit at the same
               coverage, efficiency, fraction of total benefit captured

Usage
-----
    python -m ciscarl2.gate18 --verify --reps 10     # pre-freeze check
    python -m ciscarl2.gate18 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (_fit_predict_nuisances, aipw, crossfit_nuisances,
                         ipcw_outcome, km_survival, phi_on_holdout,
                         truncate_weights)
from .gate2 import pseudo_outcome
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate8 import assign, best_jaccard
from .gate11 import _cart, _hi_leaf, _leaf_stats, _split_stability
from .gate14 import _calibrate, _config, _region, _simulate
from .rules import eval_rule

from .paths import data_path

DATA = data_path()
KAPPAS = (0.0, 3.0)
START = 200100          # same seed range for both kappa blocks (paired by rep)
CENSORINGS = ("conditional", "marginal")


# ---------------------------------------------------------------------------
# Marginal-KM censoring path
# ---------------------------------------------------------------------------
def _marginal_caps(Gmarg, fid, Tobs, t_star, quantile=0.99):
    caps = {}
    for k in np.unique(fid):
        idx = np.where(fid == k)[0]
        ar = idx[Tobs[idx] > t_star]
        if ar.size > 5:
            caps[int(k)] = {quantile: float(np.quantile(1.0 / Gmarg[ar], quantile))}
        else:
            caps[int(k)] = {quantile: np.inf}
    return caps


def phi_holdout_marginal(XD, AD, TD, ED, XE, AE, TE, t_star, e=0.5,
                         cap_q=0.99, cap_ceil=20.0):
    """Held-out AIPW with a marginal (covariate-free) censoring curve."""
    Gtr = km_survival(TD, (ED == 0).astype(int), t_star)
    ar = TD > t_star
    w = 1.0 / Gtr
    cap = min(float(np.quantile(w, cap_q)), cap_ceil) if ar.any() else cap_ceil
    G = max(Gtr, 1.0 / cap)
    mu, _ = _fit_predict_nuisances(XD, AD, TD, ED, XE, t_star)
    Y = ipcw_outcome(TE, t_star, np.full(len(TE), G))
    phi1 = mu[:, 1] + (AE == 1) / e * (Y - mu[:, 1])
    phi0 = mu[:, 0] + (AE == 0) / (1 - e) * (Y - mu[:, 0])
    return phi1, phi0


# ---------------------------------------------------------------------------
def run_rep(rep, kappa, cfg, lam_c0):
    pool, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, lam0 = cfg
    seed = START + rep
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
    regE, truth = R[te], tau_i[te]
    regD = R[tr]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    gm_caps = _marginal_caps(Gmarg, fid, TD, TSTAR)

    rows = []
    for cen in CENSORINGS:
        if cen == "conditional":
            G = truncate_weights(GD, fid, fcaps)[0]
            p1, p0 = phi_on_holdout(XD, AD, TD, ED, XE, AE, TE, TSTAR, seed=seed)
        else:
            G = truncate_weights(Gmarg, fid, gm_caps)[0]
            p1, p0 = phi_holdout_marginal(XD, AD, TD, ED, XE, AE, TE, TSTAR)
        psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
        dphi = p1 - p0
        _, _, q1, q0 = aipw(XD, AD, TD, TSTAR, muD, G)
        const = float((q1 - q0).mean())
        const_mae = float(np.abs(const - truth).mean())

        _, leaves = _cart(psi, XD, 3)
        hat = assign(leaves, XE, dphi, const)
        hi = _hi_leaf(leaves, psi, XD)
        st = _leaf_stats(hi["cond"], XE, regE, dphi) if hi else {}
        if hi is not None:
            hi_mask = eval_rule(hi["cond"], XE)
        else:
            hi_mask = np.zeros(len(XE), bool)

        k = int(hi_mask.sum())
        cov = float(hi_mask.mean())
        per_treated = float(truth[hi_mask].mean()) if k else np.nan
        oracle_k = (float(np.sort(truth)[::-1][:k].mean()) if k else np.nan)
        eff_ratio = per_treated / oracle_k if k and oracle_k else np.nan
        captured = (float(truth[hi_mask].sum() / truth.sum())
                    if truth.sum() > 0 else np.nan)
        pos = truth > 1e-6
        sens = float((hi_mask & pos).sum() / max(pos.sum(), 1))
        spec = float(((~hi_mask) & (~pos)).sum() / max((~pos).sum(), 1))
        hi_true = float(truth[hi_mask].mean()) if k else np.nan

        mae = float(np.abs(hat - truth).mean())
        rows.append(dict(
            rep=rep, n=n, kappa=kappa, censoring=cen,
            ate_est=const, ate_true=float(truth.mean()),
            ate_bias=const - float(truth.mean()),
            mae=mae, const_mae=const_mae, delta=mae - const_mae,
            target_sd=float(np.std(psi)),
            target_corr=float(np.corrcoef(psi, tau_i[tr])[0, 1]),
            n_rules=len(leaves), rules=len(leaves),
            best_jaccard=best_jaccard(leaves, XE, regE),
            hi_purity=st.get("purity", np.nan),
            hi_capture=st.get("capture", np.nan),
            hi_effect=st.get("effect", np.nan),
            hi_se=st.get("se", np.nan),
            hi_true_effect=hi_true,
            hi_effect_err=(st.get("effect", np.nan) - hi_true),
            hi_positive=bool(hi_true > 0) if k else False,
            stability=_split_stability(psi, XD, XE, hi["cond"], 3, seed + 7)
            if hi else np.nan,
            coverage=cov, per_treated=per_treated,
            oracle_at_k=oracle_k, efficiency=eff_ratio,
            benefit_captured=captured,
            decision_sens=sens, decision_spec=spec))
    return rows


def _boot(d, seed=0, B=5000):
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for (kappa, cen), g in df.groupby(["kappa", "censoring"]):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        out.append(dict(
            kappa=kappa, censoring=cen, reps=len(g),
            ate_est=g.ate_est.mean(), ate_true=g.ate_true.mean(),
            ate_bias=g.ate_bias.mean(), mae=g.mae.mean(),
            const_mae=g.const_mae.mean(), delta=d.mean(), lo=lo, hi=hi,
            frac_better=100.0 * float((d < 0).mean()),
            rel_gain=100.0 * (1.0 - g.mae.mean() / g.const_mae.mean()),
            target_corr=g.target_corr.mean(),
            jaccard=g.best_jaccard.mean(), purity=g.hi_purity.mean(),
            capture=g.hi_capture.mean(),
            hi_effect=g.hi_effect.mean(), hi_true=g.hi_true_effect.mean(),
            hi_err=g.hi_effect_err.mean(),
            hi_positive=100.0 * float(g.hi_positive.mean()),
            stability=g.stability.mean(), coverage=g.coverage.mean(),
            per_treated=g.per_treated.mean(),
            oracle_at_k=g.oracle_at_k.mean(),
            efficiency=g.efficiency.mean(),
            captured=g.benefit_captured.mean(),
            sens=g.decision_sens.mean(), spec=g.decision_spec.mean()))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--verify", action="store_true",
                    help="pre-freeze check: bias and isolation only")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    lam_c = {k: _calibrate(cfg, k) for k in KAPPAS}

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, k, cfg, lam_c[k])
        for k in KAPPAS for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate18_joint.csv"), index=False)
    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate18_summary.csv"), index=False)

    print(f"\n=== Gate 18 {'VERIFY' if args.verify else ''}, "
          f"n=8000, {args.reps} reps ===")
    print(f"{'kappa':>6}{'cens':>13}{'ATE est':>9}{'true':>7}{'bias':>9}"
          f"{'MAE':>8}{'const':>8}{'delta':>9}{'%bet':>6}{'gain':>8}")
    for _, r in res.iterrows():
        print(f"{r.kappa:>6.0f}{r.censoring:>13}{r.ate_est:>+9.3f}"
              f"{r.ate_true:>+7.3f}{r.ate_bias:>+9.4f}{r.mae:>8.4f}"
              f"{r.const_mae:>8.4f}{r.delta:>+9.4f}{r.frac_better:>5.0f}%"
              f"{r.rel_gain:>+7.1f}%")
    print(f"\n{'kappa':>6}{'cens':>13}{'corr':>7}{'Jacc':>7}{'pure':>6}"
          f"{'capt':>6}{'hi eff':>8}{'hi true':>9}{'hi err':>8}{'stab':>6}")
    for _, r in res.iterrows():
        print(f"{r.kappa:>6.0f}{r.censoring:>13}{r.target_corr:>7.3f}"
              f"{r.jaccard:>7.2f}{r.purity:>6.2f}{r.capture:>6.2f}"
              f"{r.hi_effect:>+8.3f}{r.hi_true:>+9.3f}{r.hi_err:>+8.3f}"
              f"{r.stability:>6.2f}")
    print(f"\n{'kappa':>6}{'cens':>13}{'cover':>7}{'per-treated':>12}"
          f"{'oracle@k':>10}{'effic':>7}{'captured':>9}{'sens':>6}{'spec':>6}")
    for _, r in res.iterrows():
        print(f"{r.kappa:>6.0f}{r.censoring:>13}{r.coverage:>7.3f}"
              f"{r.per_treated:>+12.3f}{r.oracle_at_k:>10.3f}"
              f"{r.efficiency:>7.2f}{r.captured:>9.2f}{r.sens:>6.2f}"
              f"{r.spec:>6.2f}")

    d = res[(res.kappa == 3.0) & (res.censoring == "marginal")].iloc[0]
    checks = {
        "primary: mean delta < 0": bool(d.delta < 0),
        "primary: 95% CI entirely below 0": bool(d.hi < 0),
        "primary: Jaccard >= 0.107": bool(d.jaccard >= 0.107),
        "primary: >= 70% of reps better": bool(d.frac_better >= 70),
        "quality: |ATE bias| <= 0.015": bool(abs(d.ate_bias) <= 0.015),
        "quality: purity >= 0.50": bool(d.purity >= 0.50),
        "quality: efficiency >= 0.50": bool(d.efficiency >= 0.50),
        "quality: high leaf beneficial in >= 90% of reps": bool(
            d.hi_positive >= 90.0),
    }
    print("\nPre-registered checks (cell D: kappa=3, marginal KM):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if all(checks.values()):
        print("\n-> USABLE under joint misspecification (all criteria met)")
    else:
        print("\n-> NOT usable as a subgroup model under joint "
              "misspecification; see failing criteria")
    return df


if __name__ == "__main__":
    main()

"""Gate 14 - does the v1.5 rule model survive a misspecified Cox outcome model?

Pre-registration: `results/gate14_PREREGISTRATION.md` (revised before running;
the pre-run verification and its reason are recorded there).

Misspecification: inside the region, the CONTROL arm's hazard is non-proportional
with a conservative multiplier that leaves S(730) exactly unchanged:

    control, t <= 365:  h0 = lam0 exp(lp0) * e^{k zR} / cosh(k zR)
    control, t >  365:  h0 = lam0 exp(lp0) * e^{-k zR} / cosh(k zR)
    zR = 0.5*(age_z - cd40_z), region R = 1{cd40 < median AND age > median}
    treated arm:        h1 = lam0 exp(lp0 - delta*R)   (correct)

Because e^x + e^{-x} = 2 cosh(x), the integral of h0 over [0, 730] is unchanged,
so the true tau is identical to the correctly specified DGP and is exactly zero
outside the region. Only the hazard *path* is wrong, and only in the control arm.

The censoring model stays correctly specified, so AIPW remains consistent; the
only thing the violation can damage is the plug-in discovery target.

Arms: k = 3 (primary, target correlation ~0.02 in pre-run checks),
      k = 1 (mild, ~0.57), k = 0 (control, ~0.66).

Usage
-----
    python -m ciscarl2.gate14 --reps 2 --start 160900   # smoke, off-seed
    python -m ciscarl2.gate14 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import TAU
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate8 import assign, best_jaccard
from .gate11 import _cart, _hi_leaf, _leaf_stats, _split_stability

from .paths import data_path

DATA = data_path()
TAU_C = 365.0
ARMS = (("k3", 3.0, 160100), ("k1", 1.0, 160300), ("k0", 0.0, 160200))
PRIMARY = "k3"


def _region(X, cfg):
    _, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, _ = cfg
    return (X[:, i_cd4] < (cd4_cut - mu_[i_cd4]) / sd_[i_cd4]) & \
           (X[:, i_age] > (age_cut - mu_[i_age]) / sd_[i_age])


def _multipliers(X, R, kappa, cfg):
    """Conservative early/late multiplier; sum over both periods == 2."""
    _, _, _, i_cd4, i_age, _, _, _ = cfg
    zr = 0.5 * (X[:, i_age] - X[:, i_cd4])
    a = kappa * zr
    mf = np.where(R, np.exp(a) / np.cosh(a), 1.0)
    ms = np.where(R, np.exp(-a) / np.cosh(a), 1.0)
    return mf, ms


def _simulate(X, A, lam_c0, lam0, rng, eff, R, kappa, cfg):
    lp0 = X @ dgp.BETA
    lp1 = lp0 - eff
    mf, ms = _multipliers(X, R, kappa, cfg)
    base = lam0 * np.exp(np.where(A == 1, lp1, lp0))
    use = (A == 0)                       # only the control arm is misspecified
    lf = base * np.where(use, mf, 1.0)
    ls = base * np.where(use, ms, 1.0)

    V = rng.exponential(1.0, len(X))
    Hc = lf * TAU_C
    T = np.where(V <= Hc, V / np.maximum(lf, 1e-12),
                 TAU_C + (V - Hc) / np.maximum(ls, 1e-12))

    # true S(730): (mf + ms) * 365 == 730 by construction, so tau is unchanged
    S0 = np.exp(-lam0 * np.exp(lp0) * (mf + ms) * TAU_C)
    S1 = np.exp(-lam0 * np.exp(lp1) * 2.0 * TAU_C)

    lam_c = lam_c0 * np.exp(X @ dgp.GAMMA)
    C = rng.exponential(1.0 / np.maximum(lam_c, 1e-12))
    Tobs = np.minimum(T, C)
    E = (T <= C).astype(int)
    return dict(T=T, C=C, Tobs=Tobs, E=E, S0=S0, S1=S1)


def _calibrate(cfg, kappa, target=0.10, n_pilot=6000, seed=99):
    pool, _, _, _, _, _, _, lam0 = cfg
    rng0 = np.random.RandomState(seed)
    Xp = pool[rng0.choice(len(pool), n_pilot, replace=True)]
    R = _region(Xp, cfg)
    lp0 = Xp @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    eff = np.where(R, _delta_for(EFFECT, s0), 0.0)

    def frac(lc):
        rng = np.random.RandomState(seed)
        A = rng.binomial(1, 0.5, len(Xp))
        d = _simulate(Xp, A, lc, lam0, rng, eff, R, kappa, cfg)
        return float(np.mean((d["Tobs"] <= TSTAR) & (d["E"] == 0)))

    lo, hi = 1e-7, 5e-2
    for _ in range(45):
        mid = float(np.sqrt(lo * hi))
        if frac(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def run_rep(rep, arm, kappa, start, cfg, lam_c0):
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
    regE, truth = R[te], tau_i[te]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=seed, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    _, _, q1, q0 = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const = float((q1 - q0).mean())
    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=seed)
    dphi = p1 - p0
    tau_hat = muD[:, 1] - muD[:, 0]

    _, leaves = _cart(tau_hat, XD, 3)
    hat = assign(leaves, XE, dphi, const)
    hi = _hi_leaf(leaves, tau_hat, XD)
    st = _leaf_stats(hi["cond"], XE, regE, dphi) if hi else {}

    mae = float(np.abs(hat - truth).mean())
    const_mae = float(np.abs(const - truth).mean())
    regD = R[tr]
    return dict(
        rep=rep, n=n, arm=arm, kappa=kappa, rules=len(leaves),
        mae=mae, const_mae=const_mae, delta=mae - const_mae,
        target_sd=float(np.std(tau_hat)),
        target_corr=float(np.corrcoef(tau_hat, tau_i[tr])[0, 1]),
        target_contrast=float(tau_hat[regD].mean() - tau_hat[~regD].mean()),
        true_contrast=float(tau_i[tr][regD].mean() - tau_i[tr][~regD].mean()),
        const_ate=const, true_ate=float(tau_i[tr].mean()),
        best_jaccard=best_jaccard(leaves, XE, regE),
        hi_share=st.get("share", np.nan),
        hi_purity=st.get("purity", np.nan),
        hi_capture=st.get("capture", np.nan),
        hi_effect=st.get("effect", np.nan),
        hi_se=st.get("se", np.nan),
        stability=_split_stability(tau_hat, XD, XE, hi["cond"], 3, seed + 7)
        if hi else np.nan,
        region_share=100.0 * float(regE.mean()))


def _boot(d, seed=0, B=5000):
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for arm, g in df.groupby("arm"):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        out.append(dict(
            arm=arm, kappa=g.kappa.iloc[0], reps=len(g), rules=g.rules.mean(),
            mae=g.mae.mean(), const_mae=g.const_mae.mean(), delta=d.mean(),
            lo=lo, hi=hi, frac_better=100.0 * float((d < 0).mean()),
            rel_gain=100.0 * (1.0 - g.mae.mean() / g.const_mae.mean()),
            target_sd=g.target_sd.mean(), target_corr=g.target_corr.mean(),
            target_contrast=g.target_contrast.mean(),
            true_contrast=g.true_contrast.mean(),
            const_ate=g.const_ate.mean(), true_ate=g.true_ate.mean(),
            best_jaccard=g.best_jaccard.mean(), hi_share=g.hi_share.mean(),
            hi_purity=g.hi_purity.mean(), hi_capture=g.hi_capture.mean(),
            hi_effect=g.hi_effect.mean(), hi_se=g.hi_se.mean(),
            stability=g.stability.mean()))
    return pd.DataFrame(out)


def decide(res):
    r = res[res.arm == PRIMARY].iloc[0]
    return {
        "mean delta < 0": bool(r.delta < 0),
        "95% CI entirely below 0": bool(r.hi < 0),
        "Jaccard >= 0.107": bool(r.best_jaccard >= 0.107),
        ">= 70% of reps better": bool(r.frac_better >= 70),
    }


def _config():
    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    cd4_cut, age_cut = float(raw["cd40"].median()), float(raw["age"].median())
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    return (pool, mu, sd, dgp.COVARIATES.index("cd40"),
            dgp.COVARIATES.index("age"), cd4_cut, age_cut, lam0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    arms = ARMS if args.start is None else \
        tuple((a, k, args.start) for a, k, _ in ARMS)
    lam = {a: _calibrate(cfg, k) for a, k, _ in arms}
    for a, k, _ in arms:
        print(f"  {a:>3}: kappa={k}, lam_c0={lam[a]:.3e}", flush=True)

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, a, k, s, cfg, lam[a])
        for a, k, s in arms for rep in range(args.reps))
    df = pd.DataFrame([r for r in out if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate14_misspec.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate14_summary.csv"), index=False)
    checks = decide(res)

    print(f"\n=== Gate 14, n=8000, {args.reps} reps per arm ===")
    print(f"{'arm':>5}{'k':>4}{'rules':>7}{'MAE':>9}{'const':>9}{'delta':>10}"
          f"{'95% CI':>21}{'%bet':>6}{'gain':>7}")
    for _, r in res.iterrows():
        print(f"{r.arm:>5}{r.kappa:>4.0f}{r.rules:>7.1f}{r.mae:>9.4f}"
              f"{r.const_mae:>9.4f}{r.delta:>+10.4f}"
              f"   [{r.lo:+.4f},{r.hi:+.4f}]{r.frac_better:>5.0f}%"
              f"{r.rel_gain:>+6.1f}%")

    print("\nDiagnostics (does the violation damage the target?):")
    print(f"{'arm':>5}{'tgt sd':>9}{'corr':>7}{'tgt contr':>11}"
          f"{'true contr':>11}{'ATE est':>9}{'ATE true':>9}")
    for _, r in res.iterrows():
        print(f"{r.arm:>5}{r.target_sd:>9.4f}{r.target_corr:>7.3f}"
              f"{r.target_contrast:>11.3f}{r.true_contrast:>11.3f}"
              f"{r.const_ate:>+9.3f}{r.true_ate:>+9.3f}")
    print(f"\n{'arm':>5}{'purity':>8}{'capture':>9}{'hi eff':>8}{'hi se':>7}"
          f"{'jacc':>7}{'stab':>7}{'prev':>7}")
    for _, r in res.iterrows():
        print(f"{r.arm:>5}{r.hi_purity:>8.2f}{r.hi_capture:>9.2f}"
              f"{r.hi_effect:>+8.3f}{r.hi_se:>7.3f}{r.best_jaccard:>7.2f}"
              f"{r.stability:>7.2f}{r.hi_share:>7.2f}")

    print(f"\nPre-registered decision checks (primary arm {PRIMARY}, kappa=3):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n-> {'ROBUST to this misspecification' if all(checks.values()) else 'NOT robust'}")
    return df


if __name__ == "__main__":
    main()

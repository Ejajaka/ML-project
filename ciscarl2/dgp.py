"""Simulation DGPs for CISCaRL Gate 1 (estimator validation).

Frozen decisions implemented here
---------------------------------
* t* = 730 days (primary horizon).
* Treatment is randomized: A ~ Bernoulli(0.5)  -> known e = 0.5.
* Covariates are RESAMPLED from the real ACTG175 matrix so the correlation
  structure and marginal distributions match the application.
* The full-data truth (S_a(t*|X)) is computed analytically from the DGP,
  never from the observed (censored) sample.

Censoring mechanisms (Gate 1 protocol)
--------------------------------------
1. 'independent' : C _|_ (T, X).  Baseline correctness check.
2. 'covariate'   : C depends on X.  Conditional G should fix it; marginal G
                   should bias (the ACTG175-like case).
3. 'event'       : C depends on the LATENT event time T (beyond X).
                   Informative; not fixable by conditioning on X.
4. 'frailty'     : unmeasured frailty U enters BOTH hazards (Gamma).  This is
                   the discontinuation analogue: unmeasured time-varying health.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TSTAR = 730.0

COVARIATES = ["cd40", "cd80", "age", "wtkg", "karnof", "hemo",
              "homo", "race", "gender", "drugs", "str2", "symptom"]

# ---- event model ----------------------------------------------------------
# Standardised covariate effects on the log-hazard.
BETA = np.zeros(12)
BETA[0] = -0.40   # cd40      (higher CD4  -> lower hazard)
BETA[1] = 0.30    # cd80
BETA[2] = 0.50    # age
BETA[4] = -0.20   # karnof
BETA[7] = 0.20    # race
BETA[11] = 0.30   # symptom

TAU0, TAU1 = 0.40, 0.30   # beneficial effect: log-HR = -(tau0 + tau1 * age_z)
AGE = COVARIATES.index("age")

# ---- censoring model ------------------------------------------------------
# Mechanism 2 uses the SAME covariates that drive the outcome, which is what
# makes the strong-covariance regime realistic for ACTG175.
GAMMA = np.zeros(12)
GAMMA[2] = 0.40
GAMMA[0] = -0.30
GAMMA[1] = 0.30
GAMMA[7] = 0.30

EVENT_CENS_PSI = 0.70     # mechanism 3: dependence on latent log T
FRAILTY_SHAPE = 1.0       # mechanism 4: U ~ Gamma(k, 1/k), mean 1, var 1/k

MECHANISMS = ("independent", "covariate", "event", "frailty")


def load_covariate_pool(path: str) -> np.ndarray:
    """Load ACTG175 arms 0/1 and return standardised covariates."""
    df = pd.read_csv(path)
    df = df[df["arms"].isin([0, 1])]
    X = df[COVARIATES].to_numpy(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def lambda0_for_target_survival(pool: np.ndarray, target: float = 0.70,
                                t_star: float = TSTAR) -> float:
    """Choose the baseline hazard so mean S_0(t*) == target."""
    lp = pool @ BETA
    return -np.log(target) / (t_star * np.exp(lp).mean())


def _draw_censoring(kind, lam_c0, Xs, T_latent, U, rng):
    n = len(Xs)
    if kind == "independent":
        lam = np.full(n, lam_c0)
    elif kind == "covariate":
        lam = lam_c0 * np.exp(Xs @ GAMMA)
    elif kind == "event":
        lt = np.log(T_latent)
        z = (lt - lt.mean()) / (lt.std() + 1e-9)
        lam = lam_c0 * np.exp(EVENT_CENS_PSI * z)
    elif kind == "frailty":
        lam = lam_c0 * U
    else:
        raise ValueError(f"unknown mechanism {kind!r}")
    return rng.exponential(1.0 / lam)


def generate(Xs, kind, lam_c0, lam0, rng, t_star=TSTAR, tau=None,
             effect_vec=None):
    """Simulate one dataset. Returns observed data + full-data truth.

    tau=(tau0, tau1) overrides the default treatment effect, which is useful
    when a gate needs a clinically realistic effect size for discovery.
    """
    tau0, tau1 = tau if tau is not None else (TAU0, TAU1)
    n = len(Xs)
    A = rng.binomial(1, 0.5, n)

    U = np.ones(n)
    if kind == "frailty":
        U = rng.gamma(FRAILTY_SHAPE, 1.0 / FRAILTY_SHAPE, n)

    lp0 = Xs @ BETA
    if effect_vec is not None:
        lp1 = lp0 - effect_vec                 # per-patient log-HR offset
    else:
        lp1 = lp0 - (tau0 + tau1 * Xs[:, AGE])   # benefit: lower hazard
    lam_T = lam0 * np.exp(np.where(A == 1, lp1, lp0)) * U
    T = rng.exponential(1.0 / lam_T)

    C = _draw_censoring(kind, lam_c0, Xs, T, U, rng)
    Tobs = np.minimum(T, C)
    E = (T <= C).astype(int)

    S0 = np.exp(-lam0 * t_star * np.exp(lp0))
    S1 = np.exp(-lam0 * t_star * np.exp(lp1))
    return dict(A=A, T=T, C=C, Tobs=Tobs, E=E, S0=S0, S1=S1, U=U,
                tau_true=float(np.mean(S1 - S0)),
                cens_before=float(np.mean((Tobs <= t_star) & (E == 0))))


def generate_mnar(Xs, kind, lam_c0, lam0, rng, t_star=TSTAR, tau=None,
                  psi=0.0):
    """MNAR censoring (Gate 3).

    Patients whose latent event time exceeds t* have their censoring hazard
    multiplied by exp(psi). psi=0 recovers the MAR/MCAR mechanisms used in
    Gates 1-2, so the family nests the null case.

    Because the tilt uses the latent T, no function of the observed covariates
    can recover the correct weights: this is a genuine departure from
    conditional independent censoring, and the sensitivity parameter delta has
    to absorb it.
    """
    tau0, tau1 = tau if tau is not None else (TAU0, TAU1)
    n = len(Xs)
    A = rng.binomial(1, 0.5, n)
    U = np.ones(n)
    if kind == "frailty":
        U = rng.gamma(FRAILTY_SHAPE, 1.0 / FRAILTY_SHAPE, n)

    lp0 = Xs @ BETA
    lp1 = lp0 - (tau0 + tau1 * Xs[:, AGE])
    lam_T = lam0 * np.exp(np.where(A == 1, lp1, lp0)) * U
    T = rng.exponential(1.0 / lam_T)

    tilt = np.exp(psi * (T > t_star).astype(float))
    lam_c = np.full(n, lam_c0)
    if kind == "covariate":
        lam_c = lam_c0 * np.exp(Xs @ GAMMA)
    elif kind == "frailty":
        lam_c = lam_c0 * U
    C = rng.exponential(1.0 / (lam_c * tilt))

    Tobs = np.minimum(T, C)
    E = (T <= C).astype(int)
    S0 = np.exp(-lam0 * t_star * np.exp(lp0))
    S1 = np.exp(-lam0 * t_star * np.exp(lp1))
    return dict(A=A, T=T, C=C, Tobs=Tobs, E=E, S0=S0, S1=S1, U=U,
                tau_true=float(np.mean(S1 - S0)),
                cens_before=float(np.mean((Tobs <= t_star) & (E == 0))),
                dgp="mnar", psi=psi)


def oracle_G(kind, lam_c0, Xs, t_star=TSTAR):
    """True P(C > t* | X) where analytically available (else None)."""
    if kind == "independent":
        return np.full(len(Xs), np.exp(-lam_c0 * t_star))
    if kind == "covariate":
        return np.exp(-lam_c0 * t_star * np.exp(Xs @ GAMMA))
    if kind == "frailty":
        k = FRAILTY_SHAPE
        return np.full(len(Xs), (1.0 + lam_c0 * t_star / k) ** (-k))
    return None   # 'event': depends on latent T, not identified from X


def calibrate_lam_c(kind, target_frac, pool, lam0, t_star=TSTAR,
                    n_pilot=6000, seed=99, gen=None, gen_kwargs=None):
    """Bisect lam_c0 so that the censored-before-t* fraction hits target."""
    gen = gen or generate
    Xp = pool[np.random.RandomState(seed).choice(len(pool), n_pilot, replace=True)]

    def frac(lam_c0):
        rng = np.random.RandomState(seed)
        d = gen(Xp, kind, lam_c0, lam0, rng, t_star, **(gen_kwargs or {}))
        return d["cens_before"]

    lo, hi = 1e-7, 5e-2
    for _ in range(45):
        mid = float(np.sqrt(lo * hi))
        if frac(mid) < target_frac:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ---------------------------------------------------------------------------
# Non-proportional-hazards DGP (Gate 1c)
# ---------------------------------------------------------------------------
CROSS_TAU = 180.0      # hazards cross here
CROSS_DELTA = 0.60     # treated hazard is (1+d) below CROSS_TAU, (1-d) above


def generate_nonph(Xs, kind, lam_c0, lam0, rng, t_star=TSTAR):
    """Crossing hazards: the treated group is worse early, better late.

    Within each arm the covariate effects are still proportional, so a Cox
    T-learner (one baseline per arm) can represent this. A pooled Cox model
    with a single baseline cannot: it forces a constant log-hazard-ratio.
    """
    n = len(Xs)
    A = rng.binomial(1, 0.5, n)
    lp = Xs @ BETA
    lam_i = lam0 * np.exp(lp)

    V = -np.log(rng.uniform(size=n))          # Exp(1)
    w = V / lam_i                             # H_0-weighted time

    a1, a2 = 1.0 + CROSS_DELTA, 1.0 - CROSS_DELTA
    thr = a1 * CROSS_TAU
    T1 = np.where(w <= thr, w / a1, CROSS_TAU + (w - thr) / a2)
    T0 = w
    T = np.where(A == 1, T1, T0)

    C = _draw_censoring(kind, lam_c0, Xs, T, np.ones(n), rng)
    Tobs = np.minimum(T, C)
    E = (T <= C).astype(int)

    g1 = a1 * min(t_star, CROSS_TAU) + a2 * max(0.0, t_star - CROSS_TAU)
    S0 = np.exp(-lam_i * t_star)
    S1 = np.exp(-lam_i * g1)
    return dict(A=A, T=T, C=C, Tobs=Tobs, E=E, S0=S0, S1=S1, U=np.ones(n),
                tau_true=float(np.mean(S1 - S0)),
                cens_before=float(np.mean((Tobs <= t_star) & (E == 0))),
                dgp="nonph")

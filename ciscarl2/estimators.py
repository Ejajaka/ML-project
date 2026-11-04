"""Frozen-specification estimators.

Implements exactly the pieces Gate 1 needs:

* cross-fitted nuisances (per-arm survival model + censoring model)
* the AIPW subgroup estimator (NOT a mean of patient-level pseudo-outcomes)
* a Cox T-learner comparator (allows effect modification; SCRE's family)

Nothing here does rule discovery - that is Gate 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.model_selection import KFold

from lifelines import CoxPHFitter

W_CAP = 10.0      # weight cap used only for diagnostics


# ---------------------------------------------------------------------------
# Step-function evaluation (KM / Cox baseline are right-continuous steps;
# np.interp between events gives the wrong value and biases every nuisance)
# ---------------------------------------------------------------------------
def step_value(times: np.ndarray, surv: np.ndarray, t: float) -> float:
    times = np.asarray(times, float)
    surv = np.asarray(surv, float)
    if times.size == 0:
        return 1.0
    j = int(np.searchsorted(times, t, side="right")) - 1
    if j < 0:
        return 1.0
    return float(surv[j])


def km_survival(t: np.ndarray, event: np.ndarray, t_star: float) -> float:
    """Product-limit estimate of the 'event' process at t_star."""
    order = np.argsort(t)
    t = np.asarray(t, float)[order]
    d = np.asarray(event)[order]
    if len(t) == 0:
        return 1.0
    s = 1.0
    for tt in np.unique(t[d == 1]):
        if tt > t_star:
            break
        at_risk = (t >= tt).sum()
        deaths = ((t == tt) & (d == 1)).sum()
        if at_risk > 0:
            s *= 1.0 - deaths / at_risk
    return float(max(min(s, 1.0), 1e-6))


# ---------------------------------------------------------------------------
# Cox helper
# ---------------------------------------------------------------------------
def _fit_cox(df: pd.DataFrame, tcol: str, dcol: str):
    last = None
    for pen in (0.01, 0.05, 0.1, 0.5, 2.0, 10.0):
        try:
            return CoxPHFitter(penalizer=pen).fit(df, tcol, dcol)
        except Exception as exc:                      # pragma: no cover
            last = exc
    raise last


def _surv_fallback(T, D, t_star):
    """Marginal product-limit fallback when a Cox fit cannot converge."""
    order = np.argsort(T)
    t = np.asarray(T, float)[order]
    d = np.asarray(D)[order]
    s = 1.0
    for tt in np.unique(t[d == 1]):
        if tt > t_star:
            break
        at_risk = (t >= tt).sum()
        deaths = ((t == tt) & (d == 1)).sum()
        if at_risk > 0:
            s *= 1.0 - deaths / at_risk
    return float(max(min(s, 1.0), 1e-3))


def _cox_fit(X, T, D, t_star):
    """Return (S0(t*), beta, covariance) for a Cox model."""
    df = pd.DataFrame(X)
    df["_T"] = T
    df["_D"] = D
    try:
        cph = _fit_cox(df, "_T", "_D")
    except Exception:                                 # pragma: no cover
        return _surv_fallback(T, D, t_star), np.zeros(X.shape[1]), np.eye(X.shape[1])
    bh = cph.baseline_survival_
    s0 = step_value(bh.index.values, bh.iloc[:, 0].values, t_star)
    beta = np.asarray(cph.params_.values, float)
    V = np.asarray(cph.variance_matrix_, float)
    return s0, beta, V


# ---------------------------------------------------------------------------
# Nuisances: 5-fold cross-fitted, out-of-fold only
# ---------------------------------------------------------------------------
def crossfit_nuisances(X, A, Tobs, E, t_star, n_splits=5, seed=0,
                       mu_model="cox", return_folds=False,
                       cap_quantiles=(0.95, 0.99)):
    """Return mu (n x 2), G_conditional, G_marginal (all out-of-fold).

    mu_model='cox' keeps the outcome model correctly specified for these Cox
    DGPs. That is deliberate: Gate 1 isolates the effect of the *censoring*
    model, so outcome-model error must not be confounded into the comparison.

    With return_folds=True also returns (fold_id, fold_caps, g_train_min), where
    fold_caps[k][q] is the q-th percentile of 1/G computed on the *fitting* rows
    of fold k (so a quantile weight cap uses training data only), and
    g_train_min[k] is the smallest G seen on those fitting rows (used to detect
    out-of-sample extrapolation below the training support).
    """
    n = len(X)
    mu = np.zeros((n, 2))
    G_cond = np.ones(n)
    G_marg = np.ones(n)
    fold_id = np.full(n, -1, dtype=int)
    fold_caps = {}
    g_train_min = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (fit_idx, hold_idx) in enumerate(kf.split(X)):
        fold_id[hold_idx] = k
        for arm in (0, 1):
            m = fit_idx[A[fit_idx] == arm]
            if E[m].sum() < 2:                        # degenerate fold
                mu[hold_idx, arm] = np.exp(-1e-3 * t_star)
                continue
            if mu_model == "cox":
                s0, beta, _ = _cox_fit(X[m], Tobs[m], E[m], t_star)
                mu[hold_idx, arm] = s0 ** np.exp(X[hold_idx] @ beta)
            else:
                from sksurv.ensemble import RandomSurvivalForest
                from sksurv.util import Surv
                y = Surv.from_arrays(E[m].astype(bool), Tobs[m])
                rsf = RandomSurvivalForest(n_estimators=150, max_depth=5,
                                           min_samples_leaf=10,
                                           random_state=seed, n_jobs=-1).fit(X[m], y)
                sf = rsf.predict_survival_function(X[hold_idx], return_array=True)
                u = rsf.unique_times_.astype(float)
                mu[hold_idx, arm] = np.array(
                    [step_value(u, sf[i], t_star) for i in range(len(hold_idx))])

        # conditional censoring model (Cox on X)
        s0c, betac, _ = _cox_fit(X[fit_idx], Tobs[fit_idx],
                                 (E[fit_idx] == 0).astype(int), t_star)
        G_cond[hold_idx] = s0c ** np.exp(X[hold_idx] @ betac)

        # training-side diagnostics (used by Gate 1b; never for prediction)
        g_fit = np.clip(s0c ** np.exp(X[fit_idx] @ betac), 1e-3, 1.0)
        at_risk = Tobs[fit_idx] > t_star
        if at_risk.sum() > 5:
            w_fit = 1.0 / g_fit[at_risk]
            fold_caps[k] = {q: float(np.quantile(w_fit, q)) for q in cap_quantiles}
        else:                                          # pragma: no cover
            fold_caps[k] = {q: np.inf for q in cap_quantiles}
        g_train_min[k] = float(g_fit.min())

        # marginal censoring (deliberate misspecification / sensitivity row)
        G_marg[hold_idx] = km_survival(Tobs[fit_idx],
                                       (E[fit_idx] == 0).astype(int), t_star)

    out = (mu, np.clip(G_cond, 1e-3, 1.0), np.clip(G_marg, 1e-3, 1.0))
    if return_folds:
        return out + (fold_id, fold_caps, g_train_min)
    return out


# ---------------------------------------------------------------------------
# AIPW estimator
# ---------------------------------------------------------------------------
def ipcw_outcome(Tobs, t_star, G):
    """Y_IPCW = 1{Tobs > t*} / G(t*|X).

    Zero for early events AND for patients censored before the horizon. The
    zero is a weighting device, not an observed status: it is what makes
    E[Y_IPCW | X, A] = S_a(t*|X) under the stated censoring assumption.
    """
    return np.where(Tobs > t_star, 1.0, 0.0) / G


def aipw(X, A, Tobs, t_star, mu, G, e=0.5):
    """Return (tau, se, phi1, phi0)."""
    n = len(X)
    Y = ipcw_outcome(Tobs, t_star, G)
    phi1 = mu[:, 1] + (A == 1) / e * (Y - mu[:, 1])
    phi0 = mu[:, 0] + (A == 0) / (1.0 - e) * (Y - mu[:, 0])
    tau = float(phi1.mean() - phi0.mean())
    ph = phi1 - phi0
    se = float(np.sqrt(((ph - tau) ** 2).sum()) / n)
    return tau, se, phi1, phi0


def aipw_subgroup(phi1, phi0, mask):
    """Rule-level AIPW effect on the same target population/normalisation."""
    ph = (phi1 - phi0)[mask]
    n_r = int(mask.sum())
    if n_r < 2:
        return np.nan, np.nan
    tau = float(ph.mean())
    se = float(np.sqrt(((ph - tau) ** 2).sum()) / n_r)
    return tau, se


# ---------------------------------------------------------------------------
# Cox T-learner comparator (allows effect modification)
# ---------------------------------------------------------------------------
def cox_tlearner_contrast(X, A, Tobs, E, t_star):
    """tau(t*) = E[S(t*|X,A=1) - S(t*|X,A=0)].

    Arm-specific Cox fits so the comparator is not structurally forbidden from
    expressing heterogeneity. NOTE: no analytic SE is returned. The delta
    method that conditions on the Breslow baseline understates the variance by
    ~3x because the baseline, not the coefficients, dominates a survival
    contrast at a fixed horizon. Use `cox_bootstrap_se` for intervals.
    """
    fits = []
    for arm in (0, 1):
        m = A == arm
        s0, beta, _ = _cox_fit(X[m], Tobs[m], E[m], t_star)
        fits.append((s0, beta))

    (s0_0, b0), (s0_1, b1) = fits
    S0 = s0_0 ** np.exp(X @ b0)
    S1 = s0_1 ** np.exp(X @ b1)
    return float((S1 - S0).mean())


def cox_bootstrap_se(X, A, Tobs, E, t_star, B=50, seed=0):
    """Non-parametric bootstrap SE for the Cox T-learner contrast."""
    rng = np.random.RandomState(seed)
    n = len(X)
    vals = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        try:
            vals.append(cox_tlearner_contrast(X[i], A[i], Tobs[i], E[i], t_star))
        except Exception:                              # pragma: no cover
            continue
    return float(np.std(vals, ddof=1)) if len(vals) > 5 else np.nan


def cox_pooled_contrast(X, A, Tobs, E, t_star):
    """Pooled Cox: treatment + covariates, ONE baseline (SCRE-like structure).

    Forces a constant log-hazard-ratio over time, so it is misspecified when
    hazards cross. This - not the T-learner - is the fair structural analogue
    of a shared-basis Cox rule ensemble.
    """
    df = pd.DataFrame(np.column_stack([X, A]))
    df["_T"] = Tobs
    df["_D"] = E
    cph = _fit_cox(df, "_T", "_D")
    bh = cph.baseline_survival_
    s0 = step_value(bh.index.values, bh.iloc[:, 0].values, t_star)
    beta = np.asarray(cph.params_.values, float)
    lp = X @ beta[:-1]
    return float((s0 ** np.exp(lp + beta[-1]) - s0 ** np.exp(lp)).mean())


# ---------------------------------------------------------------------------
# Weight diagnostics
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# EIF-corrected variance
# ---------------------------------------------------------------------------
def aipw_eif(X, A, Tobs, E, t_star, e=0.5, n_splits=5, seed=0,
             cap_quantile=0.99, cap_ceiling=20.0):
    """AIPW with a nuisance-aware (EIF-style) standard error.

    The plain AIPW influence function treats the nuisances as fixed, which
    ignores the variability of mu_hat and G_hat. Because G_hat enters the
    inverse-probability weight it is not a second-order term in practice, and
    Gate 2 showed the resulting SEs under-cover by 15-20 %.

    This function adds the missing component by a **fold jackknife over the
    nuisance fits**: each of the K cross-fitting folds yields a model, each is
    applied to the whole sample to give theta_hat_k, and the jackknife variance
    of those K estimates measures how much the target moves when the nuisance
    is refit. Total variance = IF variance + jackknife variance.

    Returns tau, se_if, se_eif, phi1, phi0.
    """
    n = len(X)
    mu = np.zeros((n, 2))
    G = np.ones(n)
    fold_id = np.full(n, -1, dtype=int)
    fold_caps = {}
    fold_mu, fold_G = [], []

    for k, (fit, hold) in enumerate(KFold(n_splits, shuffle=True,
                                          random_state=seed).split(X)):
        fold_id[hold] = k
        mu_k = np.zeros((n, 2))

        for arm in (0, 1):
            m = fit[A[fit] == arm]
            s0, beta, _ = _cox_fit(X[m], Tobs[m], E[m], t_star)
            pred = s0 ** np.exp(X @ beta)
            mu_k[:, arm] = pred
        s0c, betac, _ = _cox_fit(X[fit], Tobs[fit],
                                 (E[fit] == 0).astype(int), t_star)
        G_k = np.clip(s0c ** np.exp(X @ betac), 1e-3, 1.0)

        mu[hold] = mu_k[hold]
        G[hold] = G_k[hold]

        g_fit = np.clip(s0c ** np.exp(X[fit] @ betac), 1e-3, 1.0)
        at_risk = Tobs[fit] > t_star
        w_fit = 1.0 / g_fit[at_risk] if at_risk.any() else np.ones(1)
        fold_caps[k] = {q: float(np.quantile(w_fit, q))
                        for q in (cap_quantile,)}

        fold_mu.append(mu_k)
        fold_G.append(G_k)

    cap = np.array([min(fold_caps[int(k)][cap_quantile], cap_ceiling)
                    for k in fold_id])
    G = np.maximum(G, 1.0 / cap)

    tau, se_if, phi1, phi0 = aipw(X, A, Tobs, t_star, mu, G, e=e)

    fold_phi = []
    taus = []
    for mu_k, G_k in zip(fold_mu, fold_G):
        G_k = np.maximum(G_k, 1.0 / cap)
        _, _, p1k, p0k = aipw(X, A, Tobs, t_star, mu_k, G_k, e=e)
        fold_phi.append((p1k, p0k))
        taus.append(float((p1k - p0k).mean()))
    taus = np.asarray(taus)
    K = len(taus)
    var_jack = (K - 1) / K * float(((taus - taus.mean()) ** 2).sum())

    se_eif = float(np.sqrt(se_if ** 2 + var_jack))
    return dict(tau=tau, se_if=se_if, se_eif=se_eif, phi1=phi1, phi0=phi0,
                fold_phi=fold_phi, cap=cap)


def weight_diagnostics(G, Tobs, t_star, cap=W_CAP):
    w = ipcw_outcome(Tobs, t_star, G)
    nz = w[w > 0]
    ess = float((w.sum() ** 2) / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0
    return dict(
        w_max=float(nz.max()) if nz.size else 0.0,
        w_p99=float(np.percentile(nz, 99)) if nz.size else 0.0,
        ess=ess,
        pct_trunc=float(np.mean(nz > cap) * 100) if nz.size else 0.0,
    )


# ---------------------------------------------------------------------------
# v1.1 amendment: truncated-IPCW AIPW
# ---------------------------------------------------------------------------
def truncate_weights(G, fold_id, fold_caps, quantile=0.99, ceiling=20.0):
    """Cap the weight at min(Q_quantile of the fold's TRAINING weights, ceiling).

    Capping the weight w = 1/G at c is equivalent to flooring G at 1/c.

    This changes the estimator: with a fixed ceiling it targets the truncated
    functional unless the cap binds on a vanishing fraction of patients, i.e.
    unless the strengthened positivity condition G(t*|X) >= 1/ceiling holds.
    """
    cap = np.array([min(fold_caps[int(k)][quantile], ceiling) for k in fold_id],
                   dtype=float)
    return np.maximum(G, 1.0 / cap), cap


def fit_ate(X, A, Tobs, E, t_star, e=0.5, mu_model="cox", seed=0,
            truncate=True, cap_quantile=0.99, cap_ceiling=20.0):
    """The v1.1 production estimator for the ATE.

    Returns tau, its influence-style SE, per-patient phi (for subgroup use) and
    diagnostics. With truncate=False this reproduces the v1.0 estimator.
    """
    mu, G_raw, _, fold_id, fold_caps, g_train_min = crossfit_nuisances(
        X, A, Tobs, E, t_star, seed=seed, mu_model=mu_model, return_folds=True)

    cap = np.full(len(G_raw), np.inf)
    G = G_raw
    if truncate:
        G, cap = truncate_weights(G_raw, fold_id, fold_caps,
                                  quantile=cap_quantile, ceiling=cap_ceiling)

    gmin_vec = np.array([g_train_min[int(k)] for k in fold_id])
    extrap = float(np.mean(G_raw < gmin_vec))

    tau, se, phi1, phi0 = aipw(X, A, Tobs, t_star, mu, G, e=e)
    w = ipcw_outcome(Tobs, t_star, G)
    nz = w[w > 0]
    w_raw = ipcw_outcome(Tobs, t_star, G_raw)
    at_risk = w_raw > 0
    diag = dict(
        w_max=float(nz.max()) if nz.size else 0.0,
        w_p99=float(np.percentile(nz, 99)) if nz.size else 0.0,
        ess=float((w.sum() ** 2) / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0,
        pct_capped=float(np.mean(w_raw[at_risk] > cap[at_risk]) * 100) if at_risk.any() else 0.0,
        extrap=extrap,
        cap_median=float(np.median(cap[np.isfinite(cap)])) if np.isfinite(cap).any() else np.inf,
    )
    return dict(tau=tau, se=se, phi1=phi1, phi0=phi0, G=G, cap=cap, **diag)


# ---------------------------------------------------------------------------
# Outer cross-fitting: discovery/evaluation separation
# ---------------------------------------------------------------------------
def _fit_predict_nuisances(Xtr, Atr, Ttr, Etr, Xte, t_star, mu_model="cox"):
    """Fit nuisances on the training rows, predict on held-out rows."""
    n_te = len(Xte)
    mu = np.zeros((n_te, 2))
    for arm in (0, 1):
        m = Atr == arm
        if Etr[m].sum() < 2:
            mu[:, arm] = np.exp(-1e-3 * t_star)
            continue
        if mu_model == "cox":
            s0, beta, _ = _cox_fit(Xtr[m], Ttr[m], Etr[m], t_star)
            mu[:, arm] = s0 ** np.exp(Xte @ beta)
        else:
            from sksurv.ensemble import RandomSurvivalForest
            from sksurv.util import Surv
            y = Surv.from_arrays(Etr[m].astype(bool), Ttr[m])
            rsf = RandomSurvivalForest(n_estimators=150, max_depth=5,
                                       min_samples_leaf=10, random_state=0,
                                       n_jobs=-1).fit(Xtr[m], y)
            sf = rsf.predict_survival_function(Xte, return_array=True)
            u = rsf.unique_times_.astype(float)
            mu[:, arm] = np.array([step_value(u, sf[i], t_star)
                                   for i in range(n_te)])
    s0c, betac, _ = _cox_fit(Xtr, Ttr, (Etr == 0).astype(int), t_star)
    G = np.clip(s0c ** np.exp(Xte @ betac), 1e-3, 1.0)
    return mu, G


def phi_on_holdout(Xfit, Afit, Tfit, Efit, Xhold, Ahold, Thold, t_star, e=0.5,
                   seed=0, cap_quantile=0.99, cap_ceiling=20.0):
    """AIPW phi on held-out rows; nuisances fitted on the training rows only.

    The weight cap is derived from an inner cross-fit of the TRAINING rows, so
    no held-out patient influences its own threshold.
    """
    _, G_in, _, fid, fcaps, _ = crossfit_nuisances(
        Xfit, Afit, Tfit, Efit, t_star, seed=seed, return_folds=True)
    w_in = ipcw_outcome(Tfit, t_star, G_in)
    w_in = w_in[w_in > 0]
    cap_val = min(float(np.quantile(w_in, cap_quantile)), cap_ceiling) \
        if w_in.size else np.inf

    mu, G = _fit_predict_nuisances(Xfit, Afit, Tfit, Efit, Xhold, t_star)
    G = np.maximum(G, 1.0 / cap_val)
    Y = ipcw_outcome(Thold, t_star, G)
    phi1 = mu[:, 1] + (Ahold == 1) / e * (Y - mu[:, 1])
    phi0 = mu[:, 0] + (Ahold == 0) / (1 - e) * (Y - mu[:, 0])
    return phi1, phi0


def outer_crossfit_ate(X, A, Tobs, E, t_star, e=0.5, n_outer=5, seed=0,
                       mu_model="cox", truncate=True, cap_quantile=0.99,
                       cap_ceiling=20.0):
    """ATE estimated with rules/nuisances never evaluated on their own fold.

    Outer split -> nuisances fitted on the outer-training rows and predicted on
    the held-out rows; the weight cap is derived from an INNER cross-fit of the
    outer-training rows only, so no held-out patient influences its own cap.
    """
    n = len(X)
    phi1 = np.full(n, np.nan)
    phi0 = np.full(n, np.nan)
    caps = []

    for k, (fit, hold) in enumerate(
            KFold(n_outer, shuffle=True, random_state=seed).split(X)):
        # ---- cap from an inner cross-fit of the training rows only ----
        _, G_in, _, fid, fcaps, _ = crossfit_nuisances(
            X[fit], A[fit], Tobs[fit], E[fit], t_star, seed=seed + 1 + k,
            mu_model="cox", return_folds=True)
        w_in = np.where(Tobs[fit] > t_star, 1.0, 0.0) / G_in
        w_in = w_in[w_in > 0]
        cap_val = min(float(np.quantile(w_in, cap_quantile)), cap_ceiling) \
            if w_in.size else np.inf
        caps.append(cap_val)

        # ---- nuisances fitted on all outer-training rows ----
        mu_h, G_h = _fit_predict_nuisances(X[fit], A[fit], Tobs[fit], E[fit],
                                           X[hold], t_star, mu_model)
        if truncate:
            G_h = np.maximum(G_h, 1.0 / cap_val)

        Y = np.where(Tobs[hold] > t_star, 1.0, 0.0) / G_h
        Ah = A[hold]
        phi1[hold] = mu_h[:, 1] + (Ah == 1) / e * (Y - mu_h[:, 1])
        phi0[hold] = mu_h[:, 0] + (Ah == 0) / (1 - e) * (Y - mu_h[:, 0])

    ok = np.isfinite(phi1)
    n_ok = int(ok.sum())
    tau = float(phi1[ok].mean() - phi0[ok].mean())
    ph = phi1[ok] - phi0[ok]
    se = float(np.sqrt(((ph - tau) ** 2).sum()) / n_ok)
    return dict(tau=tau, se=se, phi1=phi1, phi0=phi0, caps=caps, n_eval=n_ok)

import numpy as np
from sklearn.model_selection import KFold

def compute_ipcw(time, event, treatment, X=None, t_star=None):
    """Compute Inverse Probability of Censoring Weights.

    Outcome is Y = I(T > t*) with the censoring-adjusted status at t*.
    The outcome is KNOWN only for patients who were NOT censored before t*:
      - time > t*              -> alive at t*  (Y = 1)
      - time <= t* & event == 1 -> died before t* (Y = 0)
      - time <= t* & event == 0 -> censored before t* (outcome unknown)
    """
    if t_star is None:
        t_star = np.percentile(time[event == 1], 50)  # median survival
    obs_time = np.minimum(time, t_star)
    known = (time > t_star) | ((time <= t_star) & (event == 1))

    # KM of the censoring process: treat censoring as the "event".
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    censored = (event == 0).astype(bool)
    kmf.fit(time, censored)
    surv = kmf.predict(obs_time)
    w_c = 1.0 / np.maximum(surv, 0.01)

    return w_c, known, t_star


def _outcome_at_tstar(time, event, t_star):
    """Return (y_obs, known) for Y = I(T > t*).

    y_obs is 1 for patients alive at t* (time > t*), 0 for patients known
    dead before t* (time <= t* & event == 1). Censored-before-t* patients
    (time <= t* & event == 0) have an unknown outcome and are excluded via
    `known`. Previously the code set y_obs = event for time <= t*, which
    flipped dead patients to 1, and marked censored-before-t* patients as
    known -- corrupting the pseudo-ITE target.
    """
    n = len(time)
    known = (time > t_star) | ((time <= t_star) & (event == 1))
    y_obs = (time > t_star).astype(float)
    return y_obs, known


def compute_pseudo_ite_dr(time, event, treatment, X, t_star, 
                           propensity, surv_func_0, surv_func_1):
    """DR-learner pseudo-ITE."""
    n = len(time)
    obs_time = np.minimum(time, t_star)
    y_obs, known = _outcome_at_tstar(time, event, t_star)
    
    e_hat = propensity
    mu0 = surv_func_0
    mu1 = surv_func_1
    
    y_star = np.zeros(n) * np.nan
    for i in range(n):
        if known[i]:
            a = treatment[i]
            e = e_hat[i]
            y = y_obs[i]
            mu_a = mu1[i] if a == 1 else mu0[i]
            mu_diff = mu1[i] - mu0[i]
            y_star[i] = (a - e) / (e * (1 - e) + 1e-8) * (y - mu_a) + mu_diff
    
    return y_star, known


def compute_pseudo_ite_r(time, event, treatment, X, t_star,
                          propensity, surv_func):
    """R-learner pseudo-ITE."""
    n = len(time)
    y_obs, known = _outcome_at_tstar(time, event, t_star)
    
    y_star = np.zeros(n) * np.nan
    for i in range(n):
        if known[i]:
            e = propensity[i]
            mu = surv_func[i]
            y = y_obs[i]
            a = treatment[i]
            y_star[i] = (y - mu) / (a - e + 1e-8)
    
    return y_star, known


def compute_pseudo_ite_dea(time, event, treatment, X, t_star, surv_func):
    """DEA-learner pseudo-ITE (efficiency-augmented D-learner)."""
    n = len(time)
    y_obs, known = _outcome_at_tstar(time, event, t_star)
    
    y_star = np.zeros(n) * np.nan
    for i in range(n):
        if known[i]:
            a = treatment[i]
            mu = surv_func[i]
            y = y_obs[i]
            y_star[i] = 2 * (2 * a - 1) * (y - mu)
    
    return y_star, known


def evaluate_cate(true_cate, pred_cate):
    """Evaluate CATE prediction performance."""
    valid = ~(np.isnan(true_cate) | np.isnan(pred_cate))
    true_c = true_cate[valid]
    pred_c = pred_cate[valid]
    
    bias = np.mean(pred_c - true_c)
    mse = np.mean((pred_c - true_c) ** 2)
    
    from scipy.stats import spearmanr
    corr, _ = spearmanr(pred_c, true_c)
    
    # Treatment recommendation accuracy
    rec = (pred_c > 0).astype(int)
    actual = (true_c > 0).astype(int)
    acc = np.mean(rec == actual)
    
    return {"bias": bias, "mse": mse, "spearman": corr, "accuracy": acc, "n": len(valid)}

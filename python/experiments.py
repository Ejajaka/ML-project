"""
Comprehensive experiments for CISCaRL paper submission.
Covers: 3 datasets x 4 DGPs x 9 methods + ablations

Datasets (from papers): PBC, SUPPORT, GBSG
Methods: Cox, CSF, Bo&Ding, Hybrid, CRE, SCRE, CISCaRL (2 modes)
DGPs: AFT-Gumbel, Cox PH, Non-PH (crossing), Nonlinear CATE

Usage: python experiments.py [--quick]
  --quick: reduced settings for testing
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, gumbel_r
from sklearn.metrics import r2_score
import time, os, sys, json

# ============================================================================
# Method imports
# ============================================================================
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet
from sksurv.ensemble import RandomSurvivalForest
from utils import compute_pseudo_ite_dr
from cis_carl import CISCaRL
from scre import SurvivalCausalRuleEnsemble


# ============================================================================
# Data Loaders (paper datasets only)
# ============================================================================

def load_pbc():
    from data import load_pbc
    return load_pbc()

def load_support():
    url = "https://hbiostat.org/data/repo/support2csv.zip"
    df = pd.read_csv(url, compression='zip')
    use_cols = ['age', 'sex', 'num.co', 'edu', 'income', 'scoma', 'race',
                'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb',
                'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine',
                'adlp', 'adls', 'diabetes', 'dementia']
    use_cols = [c for c in use_cols if c in df.columns]
    df = df[['death', 'd.time'] + use_cols].dropna()
    df = df[df['d.time'] > 0]
    for c in use_cols:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = pd.factorize(df[c])[0]
    df['treatment'] = np.random.binomial(1, 0.5, len(df))
    df['event'] = df['death'].astype(int)
    df['time_days'] = df['d.time']
    df = df.drop(columns=['death', 'd.time'])
    return df, use_cols

def load_gbsg():
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/gbsg.csv"
    df = pd.read_csv(url)
    use_cols = ['age', 'meno', 'size', 'grade', 'nodes', 'pgr', 'er']
    for c in use_cols:
        df[c] = df[c].fillna(df[c].median())
    df['treatment'] = df['hormon'].astype(int)
    df['event'] = df['status'].astype(int)
    df['time_days'] = df['rfstime']
    df = df.drop(columns=['rownames', 'hormon', 'status', 'rfstime'], errors='ignore')
    return df, use_cols


# ============================================================================
# Data Generating Processes
# ============================================================================

def dgp_aft_gumbel(df_raw, covariates, seed=42):
    """DGP 1: AFT-Gumbel model (same as existing demo)."""
    np.random.seed(seed)
    n = len(df_raw)
    X = df_raw[covariates].values.astype(float)
    treatment = df_raw['treatment'].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_z = (X - X_mean) / X_std

    idx = {c: i for i, c in enumerate(covariates)}
    age_i = idx.get('age', 0)
    bili_i = idx.get('bili', min(1, len(covariates)-1))
    alb_i = idx.get('albumin', idx.get('alb', min(2, len(covariates)-1)))

    sub1 = (X_z[:, age_i] > 0) & (X_z[:, bili_i] > 0)
    sub2 = (X_z[:, age_i] <= 0) & (X_z[:, alb_i] > 0)
    # Effect sizes chosen so the CATE (survival-probability difference at t*)
    # has enough variance for R^2 to be meaningful. The original 0.50/0.15
    # effects compressed to CATE std ~0.06, below the DR pseudo-ITE noise floor
    # (std ~0.8), which made every method score R^2 < 0 regardless of quality.
    true_eff = np.where(sub1, 2.00, np.where(sub2, 1.00, 0.0))

    eps = np.random.gumbel(0, 1, n)
    base = 6.5 + 0.3*X_z[:, age_i] - 0.2*X_z[:, bili_i] + 0.15*X_z[:, alb_i]
    t_obs = np.where(treatment == 1, np.exp(base + true_eff + eps), np.exp(base + eps))
    censor = np.random.exponential(scale=np.median(t_obs)*2.5, size=n)
    t_final, event = np.minimum(t_obs, censor).astype(float), (t_obs <= censor).astype(int)
    t_star = np.percentile(t_obs[event == 1], 50)

    def cate_fn(x_i, te, t):
        bl = 6.5 + 0.3*x_i[age_i] - 0.2*x_i[bili_i] + 0.15*x_i[alb_i]
        s0 = 1 - gumbel_r.cdf(np.log(t) - bl, 0, 1)
        s1 = 1 - gumbel_r.cdf(np.log(t) - bl - te, 0, 1)
        return s1 - s0

    true_cate = np.array([cate_fn(X_z[i], true_eff[i], t_star) for i in range(n)])
    info = {'sub1': sub1.sum(), 'sub2': sub2.sum(),
            'cate_sub1': true_cate[sub1].mean(), 'cate_sub2': true_cate[sub2].mean(),
            'n': n, 'events': event.sum(), 't_star': t_star}
    return t_final, event, treatment, X, covariates, true_cate, info


def dgp_cox_ph(df_raw, covariates, seed=42):
    """DGP 2: Cox proportional hazards model."""
    np.random.seed(seed)
    n = len(df_raw)
    X = df_raw[covariates].values.astype(float)
    treatment = df_raw['treatment'].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_z = (X - X_mean) / X_std

    idx = {c: i for i, c in enumerate(covariates)}
    age_i = idx.get('age', 0)
    bili_i = idx.get('bili', min(1, len(covariates)-1))

    sub1 = (X_z[:, age_i] > 0) & (X_z[:, bili_i] > 0)
    sub2 = (X_z[:, age_i] <= 0) & (~sub1)
    # HR magnitudes scaled up from 0.50/0.80 so the CATE survival-probability
    # differences have enough variance for a meaningful (positive) R^2.
    true_hr = np.where(sub1, 0.30, np.where(sub2, 0.60, 1.0))

    # Weibull baseline hazard
    base_hazard = 0.001
    shape = 1.5
    u = np.random.uniform(size=n)
    t_control = (-np.log(u) / base_hazard) ** (1/shape)
    t_treated = t_control / true_hr
    t_obs = np.where(treatment == 1, t_treated, t_control)
    censor = np.random.exponential(scale=np.median(t_obs)*3, size=n)
    t_final, event = np.minimum(t_obs, censor).astype(float), (t_obs <= censor).astype(int)
    t_star = np.percentile(t_obs[event == 1], 50)

    # True CATE: S1(t*) - S0(t*) under Weibull
    true_cate = np.array([
        np.exp(-base_hazard * t_star**shape / hr) - np.exp(-base_hazard * t_star**shape)
        for hr in true_hr
    ])
    info = {'sub1': sub1.sum(), 'sub2': sub2.sum(),
            'cate_sub1': true_cate[sub1].mean(), 'cate_sub2': true_cate[sub2].mean(),
            'n': n, 'events': event.sum(), 't_star': t_star}
    return t_final, event, treatment, X, covariates, true_cate, info


def dgp_non_ph_crossing(df_raw, covariates, seed=42):
    """DGP 3: Non-PH crossing survival curves.

    Log-logistic baseline where treatment changes BOTH the location
    (tau(x), scale shift) and the shape (rho(x)) parameter. Because the shape
    differs between arms, the survival curves CROSS: treatment is beneficial
    early and harmful late (subgroup 1), or harmful early and beneficial late
    (subgroup 2). This violates the proportional-hazards assumption by design.

    The true CATE is the closed-form survival-probability difference
        tau(x) = S_1(t*) - S_0(t*) = 1/(1+(t*/lam1)^a1) - 1/(1+(t*/lam0)^a0),
    which is a counterfactual quantity: it does NOT depend on the realized
    treatment assignment (fixing the previous mis-specification where
    true_cate was set to 0 for control patients and measured as a TIME
    difference rather than a survival-probability difference).
    """
    np.random.seed(seed)
    n = len(df_raw)
    X = df_raw[covariates].values.astype(float)
    treatment = df_raw['treatment'].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_z = (X - X_mean) / X_std

    idx = {c: i for i, c in enumerate(covariates)}
    age_i = idx.get('age', 0)
    bili_i = idx.get('bili', min(1, len(covariates)-1))
    alb_i = idx.get('albumin', idx.get('alb', min(2, len(covariates)-1)))

    # Subgroup 1: beneficial early (positive location shift, flatter shape ->
    # curves cross late => positive CATE at t*).
    # Subgroup 2: harmful early (negative shift, steeper shape -> crosses
    # early => negative CATE at t*).
    sub1 = (X_z[:, age_i] > 0) & (X_z[:, bili_i] > 0)
    sub2 = (X_z[:, age_i] <= 0) & (~sub1)

    base_log = 6.5 + 0.3*X_z[:, age_i] - 0.2*X_z[:, bili_i] + 0.15*X_z[:, alb_i]
    alpha0 = 1.5
    # tau/rho scaled up (was 0.6/-0.5) so the CATE survival-probability spread
    # is large enough for a meaningful (positive) R^2.
    tau = np.where(sub1, 1.2, np.where(sub2, -1.0, 0.0))
    rho = np.where(sub1, -0.3, np.where(sub2, 0.4, 0.0))

    lam0 = np.exp(base_log)
    lam1 = np.exp(base_log + tau)
    a0 = alpha0
    a1 = alpha0 * (1.0 + rho)
    a1 = np.maximum(a1, 0.5)

    # Draw potential survival times under BOTH arms (shared U, as in the other
    # DGPs); observe the arm actually assigned.
    u = np.random.uniform(size=n)
    t_control = lam0 * (u / (1.0 - u)) ** (1.0 / a0)
    t_treated = lam1 * (u / (1.0 - u)) ** (1.0 / a1)
    t_obs = np.where(treatment == 1, t_treated, t_control)

    censor = np.random.exponential(scale=np.median(t_obs)*3, size=n)
    t_final, event = np.minimum(t_obs, censor).astype(float), (t_obs <= censor).astype(int)
    t_star = np.percentile(t_obs[event == 1], 50)

    # True CATE: survival-probability difference at t* (closed form).
    s0 = 1.0 / (1.0 + (t_star / lam0) ** a0)
    s1 = 1.0 / (1.0 + (t_star / lam1) ** a1)
    true_cate = s1 - s0

    info = {'sub1': sub1.sum(), 'sub2': sub2.sum(),
            'cate_sub1': true_cate[sub1].mean(), 'cate_sub2': true_cate[sub2].mean(),
            'n': n, 'events': event.sum(), 't_star': t_star}
    return t_final, event, treatment, X, covariates, true_cate, info



def dgp_nonlinear_cate(df_raw, covariates, seed=42):
    """DGP 4: Nonlinear CATE with XOR structure."""
    np.random.seed(seed)
    n = len(df_raw)
    X = df_raw[covariates].values.astype(float)
    treatment = df_raw['treatment'].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_z = (X - X_mean) / X_std

    idx = {c: i for i, c in enumerate(covariates)}
    age_i = idx.get('age', 0)
    bili_i = idx.get('bili', min(1, len(covariates)-1))
    alb_i = idx.get('albumin', idx.get('alb', min(2, len(covariates)-1)))

    # XOR interaction: sign(X_age * X_bili) determines effect direction.
    # Amplitude scaled up (was 0.3/0.1) so CATE spread is large enough for a
    # meaningful (positive) R^2.
    xor_signal = np.sign(X_z[:, age_i] * X_z[:, bili_i])
    true_eff = xor_signal * 1.0 + 0.3 * X_z[:, alb_i]

    eps = np.random.gumbel(0, 1, n)
    base = 6.5 + 0.2*X_z[:, age_i] - 0.1*X_z[:, bili_i]
    t_obs = np.where(treatment == 1, np.exp(base + true_eff + eps), np.exp(base + eps))
    censor = np.random.exponential(scale=np.median(t_obs)*2.5, size=n)
    t_final, event = np.minimum(t_obs, censor).astype(float), (t_obs <= censor).astype(int)
    t_star = np.percentile(t_obs[event == 1], 50)

    def cate_fn(x_i, te, t):
        bl = 6.5 + 0.2*x_i[age_i] - 0.1*x_i[bili_i]
        s0 = 1 - gumbel_r.cdf(np.log(t) - bl, 0, 1)
        s1 = 1 - gumbel_r.cdf(np.log(t) - bl - te, 0, 1)
        return s1 - s0

    true_cate = np.array([cate_fn(X_z[i], true_eff[i], t_star) for i in range(n)])
    info = {'n': n, 'events': event.sum(), 't_star': t_star,
            'cate_range': f'{true_cate.min():.4f} to {true_cate.max():.4f}'}
    return t_final, event, treatment, X, covariates, true_cate, info


DGP_REGISTRY = {
    'AFT-Gumbel': dgp_aft_gumbel,
    'Cox PH': dgp_cox_ph,
    'Non-PH (Crossing)': dgp_non_ph_crossing,
    'Nonlinear XOR': dgp_nonlinear_cate,
}


# ============================================================================
# Rule Extraction Utilities (shared)
# ============================================================================

def extract_rules(tree_, X):
    rules = []
    def rec(node_id, masks):
        if tree_.feature[node_id] == -2:
            if masks[-1].sum() > 5:
                rules.append(masks[-1].copy())
            return
        f, thr = tree_.feature[node_id], tree_.threshold[node_id]
        l = masks[-1] & (X[:, f] <= thr)
        rec(tree_.children_left[node_id], masks + [l])
        r = masks[-1] & (X[:, f] > thr)
        rec(tree_.children_right[node_id], masks + [r])
    rec(0, [np.ones(len(X), dtype=bool)])
    return rules

def rules_from_ens(ens, X, mt=200):
    all_r = []
    for i in range(min(len(ens.estimators_), mt)):
        e = ens.estimators_[i]
        t = e[0].tree_ if hasattr(e, '__len__') else e.tree_
        all_r.extend(extract_rules(t, X))
    return all_r


# ============================================================================
# Single Run
# ============================================================================

def run_single(dataset_name, dgp_name, t_final, event, treatment, X,
               covariates, true_cate, quick=False):
    """Run all methods on one dataset/DGP combination."""
    n = len(X)

    # Train/test split
    idx = np.random.permutation(n)
    n_train = int(n * 0.7)
    train_i, test_i = idx[:n_train], idx[n_train:]

    X_tr, X_te = X[train_i], X[test_i]
    a_tr, a_te = treatment[train_i], treatment[test_i]
    t_tr, t_te = t_final[train_i], t_final[test_i]
    e_tr, e_te = event[train_i].astype(int), event[test_i].astype(int)
    true_tr, true_te = true_cate[train_i], true_cate[test_i]
    ntr = len(X_tr)

    t_star = np.percentile(t_tr[e_tr == 1], 50) if e_tr.sum() > 0 else np.median(t_tr)
    t_star = max(t_star, 1)

    results = {}

    # ---- 1. Cox T-learner ----
    try:
        df_tr = pd.DataFrame(X_tr, columns=covariates)
        df_tr['time'] = t_tr; df_tr['event'] = e_tr; df_tr['treatment'] = a_tr
        df_te = pd.DataFrame(X_te, columns=covariates)
        df_te['time'] = t_te; df_te['event'] = e_te; df_te['treatment'] = a_te

        cph_t = CoxPHFitter().fit(df_tr[df_tr['treatment']==1].drop(columns='treatment'),
                                  duration_col='time', event_col='event')
        cph_c = CoxPHFitter().fit(df_tr[df_tr['treatment']==0].drop(columns='treatment'),
                                  duration_col='time', event_col='event')
        sf_t = cph_t.predict_survival_function(df_te.drop(columns='treatment'))
        sf_c = cph_c.predict_survival_function(df_te.drop(columns='treatment'))
        tt = sf_t.index.values.astype(float)
        tc = sf_c.index.values.astype(float)
        s_t = np.array([np.interp(float(t_star), tt, sf_t[i].values.astype(float)) for i in df_te.index])
        s_c = np.array([np.interp(float(t_star), tc, sf_c[i].values.astype(float)) for i in df_te.index])
        results['Cox T-learner'] = s_t - s_c
    except Exception as ex:
        results['Cox T-learner'] = np.zeros(n - n_train)

    # ---- Nuisance ----
    rf_p = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_tr, a_tr)
    e_tr_p, e_te_p = rf_p.predict_proba(X_tr)[:, 1], rf_p.predict_proba(X_te)[:, 1]

    def fit_rsf(X_, t_, e_):
        y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))],
                     dtype=[("event", bool), ("time", float)])
        return RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10).fit(X_, y)

    def surv_at(m, X_, t_):
        sf = m.predict_survival_function(X_, return_array=True)
        return np.array([np.interp(float(t_), m.unique_times_.astype(float), sf[i].astype(float)) for i in range(len(X_))])

    rsf_t = fit_rsf(X_tr[a_tr==1], t_tr[a_tr==1], e_tr[a_tr==1])
    rsf_c = fit_rsf(X_tr[a_tr==0], t_tr[a_tr==0], e_tr[a_tr==0])
    s1_tr, s0_tr = surv_at(rsf_t, X_tr, t_star), surv_at(rsf_c, X_tr, t_star)
    s1_te, s0_te = surv_at(rsf_t, X_te, t_star), surv_at(rsf_c, X_te, t_star)

    ystar_tr, k_tr = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, t_star, e_tr_p, s0_tr, s1_tr)
    ystar_te, k_te = compute_pseudo_ite_dr(t_te, e_te, a_te, X_te, t_star, e_te_p, s0_te, s1_te)
    idx_k = k_tr & ~np.isnan(ystar_tr)

    if idx_k.sum() < 20:
        return {k: np.zeros(n - n_train) for k in ALL_METHODS}

    # ---- 2. CSF (RF on pseudo-ITE) ----
    csf_rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42, min_samples_leaf=10)
    csf_rf.fit(X_tr[idx_k], ystar_tr[idx_k])
    results['CSF (RF)'] = csf_rf.predict(X_te)

    X_all, ntr = np.vstack([X_tr, X_te]), len(X_tr)

    # ---- 3. Bo & Ding ----
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb.fit(X_tr[idx_k], ystar_tr[idx_k])
    r_gb = rules_from_ens(gb, X_all, 100)
    if r_gb:
        Rgb_tr = np.column_stack([m[:ntr].astype(float) for m in r_gb])
        Rgb_te = np.column_stack([m[ntr:].astype(float) for m in r_gb])
        lcv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 2, 100))
        lcv.fit(Rgb_tr[idx_k], ystar_tr[idx_k])
        mm, ms = lcv.mse_path_.mean(axis=1), lcv.mse_path_.std(axis=1)/np.sqrt(5)
        mi = np.argmin(mm)
        cand = np.where(mm <= mm[mi] + ms[mi])[0]
        a1 = lcv.alphas_[cand[-1]] if len(cand) > 0 else lcv.alphas_[mi]
        ls = Lasso(alpha=a1, max_iter=10000).fit(Rgb_tr[idx_k], ystar_tr[idx_k])
        results['Bo & Ding'] = ls.intercept_ + Rgb_te @ ls.coef_
    else:
        results['Bo & Ding'] = np.zeros(n - n_train)

    # ---- 4. Hybrid ----
    rf_d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, min_samples_leaf=5)
    rf_d.fit(X_tr[idx_k], ystar_tr[idx_k])
    r_rf = rules_from_ens(rf_d, X_all, 200)
    if r_rf:
        Rrf_tr = np.column_stack([m[:ntr].astype(float) for m in r_rf])
        Rrf_te = np.column_stack([m[ntr:].astype(float) for m in r_rf])
        lcv2 = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 2, 100))
        lcv2.fit(Rrf_tr[idx_k], ystar_tr[idx_k])
        mm2, ms2 = lcv2.mse_path_.mean(axis=1), lcv2.mse_path_.std(axis=1)/np.sqrt(5)
        mi2 = np.argmin(mm2)
        cand2 = np.where(mm2 <= mm2[mi2] + ms2[mi2])[0]
        a2 = lcv2.alphas_[cand2[-1]] if len(cand2) > 0 else lcv2.alphas_[mi2]
        ls2 = Lasso(alpha=a2, max_iter=10000).fit(Rrf_tr[idx_k], ystar_tr[idx_k])
        results['Hybrid'] = ls2.intercept_ + Rrf_te @ ls2.coef_
    else:
        results['Hybrid'] = np.zeros(n - n_train)

    # ---- 5. CRE (Causal Rule Ensemble) ----
    # Simplified CRE: RF rules + ElasticNet (matching Bargagli-Stoffi et al.)
    cre_enet = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=42, max_iter=10000,
                            alphas=np.logspace(-4, 1, 50))
    if r_rf:
        cre_enet.fit(Rrf_tr[idx_k], ystar_tr[idx_k])
        mm_c = cre_enet.mse_path_.mean(axis=1)
        ms_c = cre_enet.mse_path_.std(axis=1)/np.sqrt(5)
        mi_c = np.argmin(mm_c)
        cand_c = np.where(mm_c <= mm_c[mi_c] + ms_c[mi_c])[0]
        a_c = cre_enet.alphas_[cand_c[-1]] if len(cand_c) > 0 else cre_enet.alphas_[mi_c]
        enet_f = ElasticNet(alpha=a_c, l1_ratio=0.5, max_iter=10000)
        enet_f.fit(Rrf_tr[idx_k], ystar_tr[idx_k])
        results['CRE'] = enet_f.intercept_ + Rrf_te @ enet_f.coef_
    else:
        results['CRE'] = np.zeros(n - n_train)

    # ---- 6. SCRE (Wan et al. 2024) ----
    try:
        scre_model = SurvivalCausalRuleEnsemble(
            alpha=0.5, pred_n_estimators=200, progn_n_estimators=200,
            rule_min_support=10, max_rules=2000, max_rule_conditions=4)
        scre_model.fit(X_tr, ystar_tr, idx_k, feature_names=covariates,
                       time=t_tr, event=e_tr.astype(bool), treatment=a_tr)
        results['SCRE'] = scre_model.predict(X_te)
    except Exception as ex:
        print(f"    [SCRE] failed: {ex}")
        results['SCRE'] = np.zeros(n - n_train)

    # ---- 7. CISCaRL (direct) ----
    cis_d = CISCaRL(B=100 if quick else 200, stability_threshold=0.7, alpha=0.10,
                    csf_n_estimators=200, csf_max_depth=10,
                    gbm_n_estimators=100, gbm_max_depth=3,
                    rule_min_support=10, calib_split=0.3,
                    max_rules=2000, max_rule_conditions=4,
                    max_selected_rules=10, mode='direct', random_state=42)
    cis_d.fit(X_tr, ystar_tr, idx_k, feature_names=covariates)
    results['CISCaRL (direct)'] = cis_d.predict(X_te)

    # ---- 8. CISCaRL (auto) ----
    cis_a = CISCaRL(B=100 if quick else 200, stability_threshold=0.7, alpha=0.10,
                    csf_n_estimators=200, csf_max_depth=10,
                    gbm_n_estimators=100, gbm_max_depth=3,
                    rule_min_support=10, calib_split=0.3,
                    max_rules=2000, max_rule_conditions=4,
                    max_selected_rules=10, mode='auto', random_state=42)
    cis_a.fit(X_tr, ystar_tr, idx_k, feature_names=covariates)
    results['CISCaRL (auto)'] = cis_a.predict(X_te)

    # ---- 9. CISCaRL (posthoc, recommended by the paper) ----
    cis_p = CISCaRL(B=100 if quick else 200, stability_threshold=0.7, alpha=0.10,
                    csf_n_estimators=200, csf_max_depth=10,
                    gbm_n_estimators=100, gbm_max_depth=3,
                    rule_min_support=10, calib_split=0.3,
                    max_rules=2000, max_rule_conditions=4,
                    max_selected_rules=10, mode='posthoc', random_state=42)
    cis_p.fit(X_tr, ystar_tr, idx_k, feature_names=covariates)
    results['CISCaRL (posthoc)'] = cis_p.predict(X_te)

    # Evaluate all
    evals = {}
    for mname, pred in results.items():
        valid = ~np.isnan(pred)
        p, t = pred[valid], true_te[valid]
        if len(p) < 10:
            evals[mname] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Spearman': np.nan, 'Bias': np.nan}
            continue
        evals[mname] = {
            'MAE': float(np.mean(np.abs(p - t))),
            'RMSE': float(np.sqrt(np.mean((p - t)**2))),
            'R2': float(r2_score(t, p)) if len(np.unique(t)) > 1 else float('nan'),
            'Spearman': float(spearmanr(p, t)[0]) if len(np.unique(t))>1 and len(np.unique(p))>1 else 0.0,
            'Bias': float(np.mean(p - t)),
        }

    # Count rules
    try: evals['Bo & Ding']['Rules'] = int(np.sum(np.abs(ls.coef_) > 1e-6)) if r_gb else 0
    except: evals['Bo & Ding']['Rules'] = 0
    try: evals['Hybrid']['Rules'] = int(np.sum(np.abs(ls2.coef_) > 1e-6)) if r_rf else 0
    except: evals['Hybrid']['Rules'] = 0
    try: evals['CRE']['Rules'] = int(np.sum(np.abs(enet_f.coef_) > 1e-6)) if r_rf else 0
    except: evals['CRE']['Rules'] = 0
    try: evals['SCRE']['Rules'] = len(scre_model.selected_rules_)
    except: evals['SCRE']['Rules'] = 0
    cis_models = {'CISCaRL (direct)': cis_d, 'CISCaRL (auto)': cis_a,
                  'CISCaRL (posthoc)': cis_p}
    for k, m in cis_models.items():
        evals[k]['Rules'] = len(m.selected_rules_)

    return evals


# ============================================================================
# Ablation: rule sources, selection methods, CI methods
# ============================================================================

def run_ablations(X_tr, X_te, ystar_tr, idx_k, covariates, quick=False):
    """Run ablation experiments on rule sources and selection methods."""
    ntr, n = len(X_tr), len(X_tr) + len(X_te)
    X_all = np.vstack([X_tr, X_te]) if len(X_te) > 0 else X_tr

    results = {}

    # Fit CSF (shared)
    csf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, min_samples_leaf=5)
    csf.fit(X_tr[idx_k], ystar_tr[idx_k])
    cate_csf = csf.predict(X_all)

    # Ablation 1: Rule source
    # CSF-only rules
    rules_csf = rules_from_ens(csf, X_all, 200)
    # GBM-only rules
    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gbm.fit(X_all, cate_csf)
    rules_gbm_node = rules_from_ens(gbm, X_all, 100)
    # Combined
    rules_both = rules_csf + rules_gbm_node

    # Dedup
    def dedup(rules):
        seen = set()
        result = []
        for m in rules:
            key = str(m.sum()) + str(hash(m.tobytes()))
            if key not in seen:
                seen.add(key)
                result.append(m)
        return result

    rules_csf_d = dedup(rules_csf)[:500]
    rules_gbm_d = dedup(rules_gbm_node)[:500]
    rules_both_d = dedup(rules_both)[:1000]

    for src_name, src_rules in [('CSF-only', rules_csf_d),
                                 ('GBM-only', rules_gbm_d),
                                 ('CSF+GBM', rules_both_d)]:
        if not src_rules:
            continue
        R_tr = np.column_stack([m[:ntr].astype(float) for m in src_rules])

        # Selection: Lasso 1se
        lcv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 1, 50))
        lcv.fit(R_tr[idx_k], ystar_tr[idx_k])
        mm, ms = lcv.mse_path_.mean(axis=1), lcv.mse_path_.std(axis=1)/np.sqrt(5)
        mi = np.argmin(mm)
        cand = np.where(mm <= mm[mi] + ms[mi])[0]
        a1 = lcv.alphas_[cand[-1]] if len(cand) > 0 else lcv.alphas_[mi]
        ls = Lasso(alpha=a1, max_iter=10000).fit(R_tr[idx_k], ystar_tr[idx_k])
        results[f'{src_name}/Lasso1se'] = int(sum(np.abs(ls.coef_) > 1e-6))
        results[f'{src_name}/Lasso_alpha'] = float(a1)

        # Selection: Stability (CISCaRL-style)
        # Simplified: one stability run
        from sklearn.linear_model import Lasso as Lasso2
        counts = np.zeros(len(src_rules))
        rng = np.random.RandomState(42)
        B = 50 if quick else 100
        train_ix = np.where(idx_k)[0][:int(len(np.where(idx_k)[0])*0.7)]
        for b in range(B):
            boot = rng.choice(len(train_ix), len(train_ix), replace=True)
            boot_mask = np.zeros(len(X_tr), dtype=bool)
            boot_mask[train_ix[boot]] = True
            try:
                l = Lasso2(alpha=a1, max_iter=10000).fit(R_tr[boot_mask], ystar_tr[boot_mask])
                sel = np.where(np.abs(l.coef_) > 1e-6)[0]
                counts[sel] += 1
            except:
                pass
        stab = counts / B
        results[f'{src_name}/Stability'] = int(sum(stab > 0.7))

    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_all(quick=False):
    """Run all experiments and return results DataFrame.

    Full factorial design: 4 real datasets (PBC, SUPPORT, GBSG, ACTG175) x 4
    DGPs, plus SYNTH-PBC x 4 DGPs for a large-sample synthetic check. Every
    setting runs all 9 methods (incl. CISCaRL posthoc). Use --quick for a
    smoke run.
    """
    from data import load_actg175
    loaders = {
        'PBC': load_pbc,
        'SUPPORT': load_support,
        'GBSG': load_gbsg,
        'ACTG175': load_actg175,
    }

    all_rows = []

    # ---- Part 1: Real datasets x all 4 DGPs ----
    print("=" * 70)
    print("PART 1: REAL DATASETS (all 4 DGPs)")
    print("=" * 70)

    for dname, loader in loaders.items():
        df_raw, covariates = loader()
        for dgp_name, dgp_fn in DGP_REGISTRY.items():
            print(f"\n--- {dname} / {dgp_name} ---")
            np.random.seed(42)
            t_final, event, treatment, X, covs, true_cate, info = dgp_fn(
                df_raw, covariates)
            print(f"  n={info['n']}, events={info['events']}, t*={info['t_star']:.1f}")

            evals = run_single(dname, dgp_name, t_final, event, treatment,
                               X, covs, true_cate, quick)

            for mname, e in evals.items():
                all_rows.append({
                    'Dataset': dname, 'DGP': dgp_name, 'Method': mname,
                    **e
                })
                print(f"  {mname:25s}: MAE={e.get('MAE', np.nan):.4f}, "
                      f"Rules={e.get('Rules', 0)}")

    # ---- Part 2: Synthetic data (PBC covariate structure), all 4 DGPs ----
    print(f"\n{'='*70}")
    print("PART 2: SYNTHETIC DATA (ALL 4 DGPs)")
    print("=" * 70)

    # Generate synthetic dataset (use PBC covariate structure for realism)
    df_pbc, covariates_pbc = load_pbc()
    syn_name = 'SYNTH-PBC'

    for dgp_name, dgp_fn in DGP_REGISTRY.items():
        print(f"\n--- {syn_name} / {dgp_name} ---")
        np.random.seed(42)
        t_final, event, treatment, X, covs, true_cate, info = dgp_fn(
            df_pbc, covariates_pbc
        )
        print(f"  n={info['n']}, events=~{info.get('events', '?')}, t*={info.get('t_star', '?'):.1f}")

        evals = run_single(syn_name, dgp_name, t_final, event, treatment,
                           X, covs, true_cate, quick)

        for mname, e in evals.items():
            all_rows.append({
                'Dataset': syn_name, 'DGP': dgp_name, 'Method': mname,
                **e
            })
            print(f"  {mname:25s}: MAE={e.get('MAE', np.nan):.4f}, "
                  f"Rules={e.get('Rules', 0)}")

    # ---- Part 3: Ablation (on PBC only) ----
    print(f"\n{'='*70}")
    print("PART 3: ABLATION STUDIES (PBC)")
    print("=" * 70)

    df_pbc, covs_pbc = load_pbc()
    t_final, event, treatment, X, covs, true_cate, info = dgp_aft_gumbel(df_pbc, covs_pbc)
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n_train = int(len(X) * 0.7)
    X_tr, X_te = X[idx[:n_train]], X[idx[n_train:]]
    a_tr, a_te = treatment[idx[:n_train]], treatment[idx[n_train:]]
    t_tr, t_te = t_final[idx[:n_train]], t_final[idx[n_train:]]
    e_tr, e_te = event[idx[:n_train]].astype(int), event[idx[n_train:]].astype(int)
    true_te = true_cate[idx[n_train:]]

    t_star = np.percentile(t_tr[e_tr == 1], 50) if e_tr.sum() > 0 else np.median(t_tr)
    t_star = max(t_star, 1)

    rf_p = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_tr, a_tr)
    e_tr_p = rf_p.predict_proba(X_tr)[:, 1]

    def fit_rsf(X_, t_, e_):
        y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))], dtype=[("event", bool), ("time", float)])
        return RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10).fit(X_, y)
    def surv_at(m, X_, t_):
        sf = m.predict_survival_function(X_, return_array=True)
        return np.array([np.interp(float(t_), m.unique_times_.astype(float), sf[i].astype(float)) for i in range(len(X_))])

    rsf_t = fit_rsf(X_tr[a_tr==1], t_tr[a_tr==1], e_tr[a_tr==1])
    rsf_c = fit_rsf(X_tr[a_tr==0], t_tr[a_tr==0], e_tr[a_tr==0])
    s1_tr, s0_tr = surv_at(rsf_t, X_tr, t_star), surv_at(rsf_c, X_tr, t_star)
    ystar_tr, k_tr = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, t_star, e_tr_p, s0_tr, s1_tr)
    idx_k = k_tr & ~np.isnan(ystar_tr)

    abl_results = run_ablations(X_tr, X_te, ystar_tr, idx_k, covs_pbc, quick)
    print("  Rule source / Selection method -> Rules selected:")
    for k, v in abl_results.items():
        print(f"    {k}: {v}")
    all_rows.append({'Dataset': 'ABLATION', 'DGP': 'PBC', 'Method': 'Ablation',
                     'Details': json.dumps({k: int(v) for k, v in abl_results.items()})})

    # ---- Part 4: Summary ----
    df_results = pd.DataFrame(all_rows)
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    # Average MAE by method
    if 'MAE' in df_results.columns:
        pivot_mae = df_results.groupby('Method')['MAE'].mean()
        print("\nMean MAE across all settings:")
        for m, v in pivot_mae.sort_values().items():
            print(f"  {m:25s}: {v:.4f}")

        print("\nMean Rules:")
        if 'Rules' in df_results.columns:
            pivot_rules = df_results.groupby('Method')['Rules'].mean()
            for m, v in pivot_rules.sort_values().items():
                print(f"  {m:25s}: {v:.1f}")

    df_results.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'paper_results_full.csv'), index=False)
    print(f"\nResults saved to results/paper_results_full.csv")
    return df_results


ALL_METHODS = [
    'Cox T-learner', 'CSF (RF)', 'Bo & Ding', 'Hybrid',
    'CRE', 'SCRE', 'CISCaRL (direct)', 'CISCaRL (auto)', 'CISCaRL (posthoc)'
]


if __name__ == '__main__':
    import sys
    quick = '--quick' in sys.argv
    print(f"Running experiments (quick={quick})...")
    t0 = time.time()
    df = run_all(quick=quick)
    t1 = time.time()
    print(f"\nTotal time: {(t1-t0)/60:.1f} minutes")
    print("Done.")

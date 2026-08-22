"""Real-data ACTG175 clinical showcase: NO simulated outcome.

ACTG175 is a real randomized HIV trial (Hammer et al. 1996): combination therapy
vs zidovudine monotherapy, time to AIDS/death. We fit CISCaRL on the REAL
treatment/outcome data and display the interpretable rule list a clinician
would see -- the actual deliverable of the method. This is the anti-benchmark
slop demonstration: real RCT, real censoring, real clinical covariates.

(No ground-truth CATE exists, so we do NOT report MAE/R2 here -- we report the
interpretable output and its coverage-valid intervals. This is the use case.)
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\sanje\OneDrive\Desktop\Amrita\Sem 5\ML\project_ML\bio-ml-project-ideas\python')
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest
from utils import compute_pseudo_ite_dr
from cis_carl import CISCaRL
from data import load_actg175

df, covs = load_actg175()
n = len(df)
X = df[covs].values.astype(float)
t = df['time_days'].values.astype(float)
e = df['event'].values.astype(int)
a = df['treatment'].values.astype(int)
print(f"ACTG175 real RCT: n={n}, events={e.sum()} ({e.sum()/n:.0%}), "
      f"treated={a.sum()} ({a.mean():.0%})")

# 70/30 split
idx = np.random.RandomState(42).permutation(n); n_tr = int(0.7*n)
X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
a_tr, a_te = a[idx[:n_tr]], a[idx[n_tr:]]
t_tr, t_te = t[idx[:n_tr]], t[idx[n_tr:]]
e_tr, e_te = e[idx[:n_tr]], e[idx[n_tr:]]
t_star = np.percentile(t_tr[e_tr == 1], 50)
print(f"t* (horizon) = {t_star:.0f} days (~median event time)")

# nuisance: propensity (should be ~0.75 treated since treat=1 is combination)
rf_p = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42).fit(X_tr, a_tr)
e_tr_p = rf_p.predict_proba(X_tr)[:, 1]
def fit_rsf(X_, t_, e_):
    y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))],
                 dtype=[("event", bool), ("time", float)])
    return RandomSurvivalForest(n_estimators=150, max_depth=5, random_state=42,
                                min_samples_leaf=10).fit(X_, y)
def surv_at(m, X_, t_):
    sf = m.predict_survival_function(X_, return_array=True)
    return np.array([np.interp(float(t_), m.unique_times_.astype(float), sf[i])
                     for i in range(len(X_))])
rsf_t = fit_rsf(X_tr[a_tr == 1], t_tr[a_tr == 1], e_tr[a_tr == 1])
rsf_c = fit_rsf(X_tr[a_tr == 0], t_tr[a_tr == 0], e_tr[a_tr == 0])
s1, s0 = surv_at(rsf_t, X_tr, t_star), surv_at(rsf_c, X_tr, t_star)
ystar, k = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, t_star, e_tr_p, s0, s1)
idx_k = k & ~np.isnan(ystar)
print(f"known-outcome training patients: {idx_k.sum()}")

m = CISCaRL(B=300, stability_threshold=0.7, alpha=0.10, csf_n_estimators=200, csf_max_depth=10,
            gbm_n_estimators=100, gbm_max_depth=3, rule_min_support=10, calib_split=0.3,
            max_rules=2000, max_rule_conditions=4, max_selected_rules=10, min_calib_support=5,
            shrinkage=0.5, mode='posthoc', random_state=42)
m.fit(X_tr, ystar, idx_k, feature_names=covs)
m.print_rule_list()

# coverage / interpretability stats on test set
pred, rids = m.predict(X_te, return_details=True)
cov_te = np.mean(rids != -1)
print(f"\nTest coverage (patients assigned a rule, not default): {cov_te:.1%}")
print(f"Rules selected: {len(m.selected_rules_)}")

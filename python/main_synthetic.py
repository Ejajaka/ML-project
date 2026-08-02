"""
FULLY SYNTHETIC experiment with known ground truth.
Large n, clear subgroup structure, modest censoring.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

np.random.seed(42)

# ============================================================
# GENERATE DATA
# ============================================================
n = 2000
p = 10

X = np.random.randn(n, p)
# Create binary covariates from some of them
X_bin = (X[:, :3] > 0).astype(float)

# True subgroup structure:
# Subgroup 1: X1 > 0 AND X2 > 0 -> effect = +0.30 (large benefit)
# Subgroup 2: X1 <= 0 AND X3 > 0 -> effect = +0.15 (moderate benefit)  
# Subgroup 3: X1 > 0 AND X2 <= 0 AND X3 > 0 -> effect = -0.10 (harm)
# Everyone else: effect = 0

sub1 = (X[:, 0] > 0) & (X[:, 1] > 0)
sub2 = (X[:, 0] <= 0) & (X[:, 2] > 0)
sub3 = (X[:, 0] > 0) & (X[:, 1] <= 0) & (X[:, 2] > 0)
sub4 = ~(sub1 | sub2 | sub3)

true_effect = np.zeros(n)
true_effect[sub1] = 0.30
true_effect[sub2] = 0.15
true_effect[sub3] = -0.10

# Create treatment assignment (RCT: 50/50)
treatment = np.random.binomial(1, 0.5, n)

# Simulate survival times: Weibull model
# log(T) = beta*X + tau*X*Z + noise
baseline = 2.0 + 0.5*X[:, 0] - 0.3*X[:, 1] + 0.2*X[:, 2]
log_t_control = baseline + np.random.gumbel(0, 1, n)
log_t_treated = baseline + true_effect + np.random.gumbel(0, 1, n)

t_control = np.exp(log_t_control)
t_treated = np.exp(log_t_treated)

t_obs = np.where(treatment == 1, t_treated, t_control)

# Add censoring (30%)
censor_time = np.random.exponential(scale=np.median(t_obs) * 3, size=n)
t_final = np.minimum(t_obs, censor_time)
event = (t_obs <= censor_time).astype(bool)

t_star = np.percentile(t_obs[event], 50)  # median survival

# True CATE at t_star (survival probability difference)
from scipy.stats import gumbel_r
# True CATE = P(T(1) > t) - P(T(0) > t) = S1(t) - S0(t)
def true_cate_at(x, t):
    b = 2.0 + 0.5*x[0] - 0.3*x[1] + 0.2*x[2]
    s0 = 1 - gumbel_r.cdf(np.log(t) - b, 0, 1)
    te = np.where((x[0] > 0) & (x[1] > 0), 0.30,
                  np.where((x[0] <= 0) & (x[2] > 0), 0.15,
                           np.where((x[0] > 0) & (x[1] <= 0) & (x[2] > 0), -0.10, 0.0)))
    s1 = 1 - gumbel_r.cdf(np.log(t) - b - te, 0, 1)
    return s1 - s0

true_cate = np.array([true_cate_at(X[i], t_star) for i in range(n)])

feature_names = [f"X{i}" for i in range(p)]

print("=" * 60)
print("SYNTHETIC EXPERIMENT")
print(f"  n={n}, p={p}, events={event.sum()}, t*={t_star:.2f}")
print(f"  Sub1 (X1>0 & X2>0, effect=+0.30): {sub1.sum()}")
print(f"  Sub2 (X1<=0 & X3>0, effect=+0.15): {sub2.sum()}")
print(f"  Sub3 (X1>0 & X2<=0 & X3>0, effect=-0.10): {sub3.sum()}")
print(f"  Sub4 (rest, effect=0): {sub4.sum()}")
print(f"  True CATE range: [{true_cate.min():.4f}, {true_cate.max():.4f}]")

# Train/test split
idx = np.random.permutation(n)
n_train = int(n * 0.7)
train_idx = idx[:n_train]
test_idx = idx[n_train:]

X_train, X_test = X[train_idx], X[test_idx]
a_train, a_test = treatment[train_idx], treatment[test_idx]
t_train, t_test = t_final[train_idx], t_final[test_idx]
e_train, e_test = event[train_idx].astype(int), event[test_idx].astype(int)
true_train, true_test = true_cate[train_idx], true_cate[test_idx]

print(f"\nTrain: {n_train}, Test: {n - n_train}")

# ============================================================
# METHOD 1: COX T-LEARNER
# ============================================================
print("\n" + "-" * 50)
print("METHOD 1: Cox T-learner")
from lifelines import CoxPHFitter

df_train = pd.DataFrame(X_train, columns=feature_names)
df_train["time"] = t_train
df_train["event"] = e_train
df_train["treatment"] = a_train

df_test = pd.DataFrame(X_test, columns=feature_names)
df_test["time"] = t_test
df_test["event"] = e_test
df_test["treatment"] = a_test

try:
    cph_t = CoxPHFitter()
    cph_t.fit(df_train[df_train["treatment"]==1].drop(columns="treatment"), 
              duration_col="time", event_col="event")
    sf_t = cph_t.predict_survival_function(df_test.drop(columns="treatment"))
    
    cph_c = CoxPHFitter()
    cph_c.fit(df_train[df_train["treatment"]==0].drop(columns="treatment"), 
              duration_col="time", event_col="event")
    sf_c = cph_c.predict_survival_function(df_test.drop(columns="treatment"))
    
    times_t = sf_t.index.values.astype(float)
    times_c = sf_c.index.values.astype(float)
    s_t = np.array([np.interp(float(t_star), times_t, sf_t[i].values.astype(float)) for i in df_test.index])
    s_c = np.array([np.interp(float(t_star), times_c, sf_c[i].values.astype(float)) for i in df_test.index])
    cate_cox = s_t - s_c
except Exception as ex:
    print(f"  Cox failed: {ex}")
    cate_cox = np.zeros(len(df_test))

# ============================================================
# NUISANCE FUNCTIONS
# ============================================================
print("NUISANCE functions...")
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor

rf_p = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_p.fit(X_train, a_train)
e_tr = rf_p.predict_proba(X_train)[:, 1]
e_te = rf_p.predict_proba(X_test)[:, 1]

from sksurv.ensemble import RandomSurvivalForest

def fit_rsf(X, t, e):
    y = np.array([(bool(e[i]), float(t[i])) for i in range(len(t))],
                 dtype=[("event", bool), ("time", float)])
    m = RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10)
    m.fit(X, y)
    return m

def surv_at(m, X, t):
    sf = m.predict_survival_function(X, return_array=True)
    times = m.unique_times_.astype(float)
    return np.array([np.interp(float(t), times, sf[i].astype(float)) for i in range(len(X))])

rsf_all = fit_rsf(X_train, t_train, e_train)
rsf_t = fit_rsf(X_train[a_train==1], t_train[a_train==1], e_train[a_train==1])
rsf_c = fit_rsf(X_train[a_train==0], t_train[a_train==0], e_train[a_train==0])

s1_tr = surv_at(rsf_t, X_train, t_star)
s0_tr = surv_at(rsf_c, X_train, t_star)
s1_te = surv_at(rsf_t, X_test, t_star)
s0_te = surv_at(rsf_c, X_test, t_star)

# Pseudo-ITE
from utils import compute_pseudo_ite_dr
ystar_tr, k_tr = compute_pseudo_ite_dr(t_train, e_train, a_train, X_train, t_star, e_tr, s0_tr, s1_tr)
ystar_te, k_te = compute_pseudo_ite_dr(t_test, e_test, a_test, X_test, t_star, e_te, s0_te, s1_te)

idx_k = k_tr & ~np.isnan(ystar_tr)
print(f"  Known: {idx_k.sum()}/{n_train}")

# ============================================================
# METHOD 2: CSF
# ============================================================
print("-" * 50)
print("METHOD 2: CSF (RF on pseudo-ITE)")
rf_csf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42, min_samples_leaf=10)
rf_csf.fit(X_train[idx_k], ystar_tr[idx_k])
cate_csf = rf_csf.predict(X_test)

# ============================================================
# RULE EXTRACTION
# ============================================================
def extract_rules(tree_, X):
    rules = []
    def rec(node_id, masks):
        if tree_.feature[node_id] == -2:
            if masks[-1].sum() > 10:
                rules.append(masks[-1].copy())
            return
        f = tree_.feature[node_id]
        thr = tree_.threshold[node_id]
        l = masks[-1] & (X[:, f] <= thr)
        rec(tree_.children_left[node_id], masks + [l])
        r = masks[-1] & (X[:, f] > thr)
        rec(tree_.children_right[node_id], masks + [r])
    rec(0, [np.ones(len(X), dtype=bool)])
    return rules

def rules_from_ens(ens, X, max_trees=200):
    all_r = []
    for i in range(min(len(ens.estimators_), max_trees)):
        e = ens.estimators_[i]
        tree = e[0].tree_ if hasattr(e, '__len__') else e.tree_
        all_r.extend(extract_rules(tree, X))
    return all_r

X_all = np.vstack([X_train, X_test])
n_tr = len(X_train)

# ============================================================
# METHOD 3: BO & DING
# ============================================================
print("-" * 50)
print("METHOD 3: Bo & Ding (GB + Lasso)")
from sklearn.linear_model import Lasso, LassoCV

gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X_train[idx_k], ystar_tr[idx_k])
rules_gb = rules_from_ens(gb, X_all, max_trees=100)
print(f"  Rules: {len(rules_gb)}")

if rules_gb:
    R_tr = np.column_stack([m[:n_tr].astype(float) for m in rules_gb])
    R_te = np.column_stack([m[n_tr:].astype(float) for m in rules_gb])
    
    lcv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-2, 2, 100))
    lcv.fit(R_tr[idx_k], ystar_tr[idx_k])
    
    # 1se rule
    mse_mean = lcv.mse_path_.mean(axis=1)
    mse_se = lcv.mse_path_.std(axis=1) / np.sqrt(5)
    min_i = np.argmin(mse_mean)
    thresh = mse_mean[min_i] + mse_se[min_i]
    cand = np.where(mse_mean <= thresh)[0]
    a1se = lcv.alphas_[cand[-1]] if len(cand) > 0 else lcv.alphas_[min_i]
    
    ls = Lasso(alpha=a1se, max_iter=10000)
    ls.fit(R_tr[idx_k], ystar_tr[idx_k])
    sel = np.where(abs(ls.coef_) > 1e-6)[0]
    print(f"  Selected: {len(sel)} rules (1se alpha={a1se:.4f})")
    
    intercept = ls.intercept_
    cate_bd = intercept + R_te @ ls.coef_
    
    if len(sel) > 0:
        top = sel[np.argsort(-np.abs(ls.coef_[sel]))[:10]]
        print("  Top rules:")
        for j in top:
            print(f"    [{ls.coef_[j]:+.4f}] support={rules_gb[j].mean():.3f}")
else:
    cate_bd = np.zeros(len(X_test))
    sel = np.array([])

# ============================================================
# METHOD 4: HYBRID
# ============================================================
print("-" * 50)
print("METHOD 4: Hybrid (RF + Lasso)")
rf_d = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, min_samples_leaf=5)
rf_d.fit(X_train[idx_k], ystar_tr[idx_k])
rules_rf = rules_from_ens(rf_d, X_all, max_trees=200)
print(f"  Rules: {len(rules_rf)}")

if rules_rf:
    R_tr2 = np.column_stack([m[:n_tr].astype(float) for m in rules_rf])
    R_te2 = np.column_stack([m[n_tr:].astype(float) for m in rules_rf])
    
    lcv2 = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-2, 2, 100))
    lcv2.fit(R_tr2[idx_k], ystar_tr[idx_k])
    
    mse_mean = lcv2.mse_path_.mean(axis=1)
    mse_se = lcv2.mse_path_.std(axis=1) / np.sqrt(5)
    min_i = np.argmin(mse_mean)
    thresh = mse_mean[min_i] + mse_se[min_i]
    cand = np.where(mse_mean <= thresh)[0]
    a1se = lcv2.alphas_[cand[-1]] if len(cand) > 0 else lcv2.alphas_[min_i]
    
    ls2 = Lasso(alpha=a1se, max_iter=10000)
    ls2.fit(R_tr2[idx_k], ystar_tr[idx_k])
    sel2 = np.where(abs(ls2.coef_) > 1e-6)[0]
    print(f"  Selected: {len(sel2)} rules (1se alpha={a1se:.4f})")
    
    cate_hybrid = ls2.intercept_ + R_te2 @ ls2.coef_
    
    if len(sel2) > 0:
        top = sel2[np.argsort(-np.abs(ls2.coef_[sel2]))[:10]]
        print("  Top rules:")
        for j in top:
            print(f"    [{ls2.coef_[j]:+.4f}] support={rules_rf[j].mean():.3f}")
else:
    cate_hybrid = np.zeros(len(X_test))
    sel2 = np.array([])

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION AGAINST GROUND TRUTH")
print("=" * 60)

methods = {
    "Cox T-learner": cate_cox,
    "CSF (RF)": cate_csf,
    "Bo & Ding": cate_bd,
    "Hybrid": cate_hybrid,
}

results = []
for name, pred in methods.items():
    valid = ~np.isnan(pred)
    p = pred[valid]
    t = true_test[valid]
    
    bias = np.mean(p - t)
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t)**2))
    corr, _ = spearmanr(p, t) if len(np.unique(t)) > 1 and len(np.unique(p)) > 1 else (0, 1)
    
    rec = (p > np.median(p)).astype(int)
    true_rec = (t > np.median(t)).astype(int)
    acc = np.mean(rec == true_rec)
    
    n_r = 0
    try: n_r = len(sel) if "Bo" in name else (len(sel2) if "Hybrid" in name else 0)
    except: n_r = 0
    
    results.append({"Method": name, "Bias": f"{bias:.4f}", "MAE": f"{mae:.4f}",
                    "RMSE": f"{rmse:.4f}", "Spearman": f"{corr:.3f}",
                    "Acc": f"{acc:.3f}", "Rules": n_r})

for r in results:
    print(f"  {r['Method']:15s}: Bias={r['Bias']}, MAE={r['MAE']}, RMSE={r['RMSE']}, "
          f"Spearman={r['Spearman']}, Acc={r['Acc']}, Rules={r['Rules']}")

print("\n" + "-" * 50)
print("SUMMARY")
print(pd.DataFrame(results).to_string(index=False))
print("\nDone!")

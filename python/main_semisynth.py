"""
HTE comparison with SEMI-SYNTHETIC data on PBC.
We keep real covariates + treatment, simulate survival times with known subgroups.
This lets us evaluate against GROUND TRUTH CATE.
"""
import numpy as np
import pandas as pd
from data import load_pbc
from lifelines import CoxPHFitter

print("=" * 60)
print("SEMI-SYNTHETIC EXPERIMENT")
print("=" * 60)

# Load real data
df, covariates = load_pbc()
print(f"Real PBC data: {len(df)} patients, {len(covariates)} covariates")

# ============================================================
# SIMULATE SURVIVAL TIMES WITH KNOWN SUBGROUPS
# ============================================================
np.random.seed(42)
n = len(df)
X = df[covariates].values
treatment = df["treatment"].values

# Create 4 true subgroups with different treatment effects
# Subgroup 1: age > median AND bili > median -> large positive effect
# Subgroup 2: age <= median AND albumin > median -> moderate positive effect
# Subgroup 3: bili > median AND albumin <= median -> negative effect (harm)
# Subgroup 4: everyone else -> no effect

age_med = df["age"].median()
bili_med = df["bili"].median()
alb_med = df["albumin"].median()

sub1 = (df["age"] > age_med) & (df["bili"] > bili_med)
sub2 = (df["age"] <= age_med) & (df["albumin"] > alb_med)
sub3 = (df["bili"] > bili_med) & (df["albumin"] <= alb_med)
sub4 = ~(sub1 | sub2 | sub3)

true_effects = np.zeros(n)
true_effects[sub1] = 0.25  # +25% survival probability
true_effects[sub2] = 0.12  # +12%
true_effects[sub3] = -0.15  # -15%
true_effects[sub4] = 0.0    # none

print(f"\nTrue subgroups:")
print(f"  Sub1 (age>median & bili>median, effect=+0.25): {sub1.sum()} patients")
print(f"  Sub2 (age<=median & albumin>median, effect=+0.12): {sub2.sum()} patients")
print(f"  Sub3 (bili>median & albumin<=median, effect=-0.15): {sub3.sum()} patients")
print(f"  Sub4 (rest, effect=0): {sub4.sum()} patients")

# Simulate survival times using Weibull model (like Bo & Ding paper)
# T ~ Weibull(baseline + treatment * true_effect)
baseline_hazard = 7.0  # scale
shape = 1.5

# Base survival time (control)
u = np.random.uniform(size=n)
t_control = (-np.log(u) / np.exp(baseline_hazard)) ** (1/shape)
t_control = t_control * 1000  # scale to days

# Treatment survival time
t_treated = (-np.log(u) / np.exp(baseline_hazard + true_effects * 1.5)) ** (1/shape)
t_treated = t_treated * 1000

# Observed time (based on actual treatment assignment)
t_obs = np.where(treatment == 1, t_treated, t_control)

# Add independent censoring (30% censoring rate like paper)
censor_time = np.random.exponential(scale=2000, size=n)
t_final = np.minimum(t_obs, censor_time)
event = (t_obs <= censor_time).astype(bool)

# True CATE at median survival time
t_star_true = np.median(t_final[event])
true_cate = np.array([np.exp(-(t_star_true/np.exp(baseline_hazard + 1.5*te))**shape) 
                      - np.exp(-(t_star_true/np.exp(baseline_hazard))**shape) 
                      for te in true_effects])

print(f"\nSimulated data: {event.sum()} events, {n-event.sum()} censored")
print(f"t* = {t_star_true:.0f}")

# Replace real outcomes with simulated
df_sim = df.copy()
df_sim["time_days"] = t_final
df_sim["event"] = event.astype(int)

# ============================================================
# RUN ALL METHODS
# ============================================================

# Train/test split
idx = np.random.permutation(n)
n_train = int(n * 0.7)
train_df = df_sim.iloc[idx[:n_train]].copy()
test_df = df_sim.iloc[idx[n_train:]].copy()
print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

X_train = train_df[covariates].values
X_test = test_df[covariates].values
a_train = train_df["treatment"].values
a_test = test_df["treatment"].values
true_cate_train = true_cate[idx[:n_train]]
true_cate_test = true_cate[idx[n_train:]]

t_star = t_star_true

# ---- METHOD 1: Cox T-learner ----
print("\n" + "-" * 50)
print("METHOD 1: Cox T-learner")
cox_train = train_df[["time_days", "event", "treatment"] + covariates].copy()
cox_test = test_df[["time_days", "event", "treatment"] + covariates].copy()

train_trt = cox_train[cox_train["treatment"] == 1].drop(columns=["treatment"])
train_ctrl = cox_train[cox_train["treatment"] == 0].drop(columns=["treatment"])

cph_trt = CoxPHFitter()
cph_trt.fit(train_trt, duration_col="time_days", event_col="event")
cph_ctrl = CoxPHFitter()
cph_ctrl.fit(train_ctrl, duration_col="time_days", event_col="event")

sf_trt = cph_trt.predict_survival_function(cox_test.drop(columns=["treatment"]))
sf_ctrl = cph_ctrl.predict_survival_function(cox_test.drop(columns=["treatment"]))
times_trt = sf_trt.index.values.astype(float)
times_ctrl = sf_ctrl.index.values.astype(float)
idx_test = cox_test.index
s_trt = np.array([np.interp(float(t_star), times_trt, sf_trt[i].values.astype(float)) for i in idx_test])
s_ctrl = np.array([np.interp(float(t_star), times_ctrl, sf_ctrl[i].values.astype(float)) for i in idx_test])
cate_cox = s_trt - s_ctrl

# ---- ESTIMATE NUISANCE FUNCTIONS ----
print("\nEstimating nuisance functions...")
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor

rf_prop = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_prop.fit(X_train, a_train)
e_train = rf_prop.predict_proba(X_train)[:, 1]
e_test = rf_prop.predict_proba(X_test)[:, 1]

def fit_rsf(X, y_df, name):
    y_list = [(bool(y_df.iloc[i]["event"]), float(y_df.iloc[i]["time_days"])) for i in range(len(y_df))]
    y_arr = np.array(y_list, dtype=[("event", bool), ("time", float)])
    rsf = RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10)
    rsf.fit(X, y_arr)
    return rsf

y_train = train_df[["time_days", "event"]]
y_test = test_df[["time_days", "event"]]

rsf_all = fit_rsf(X_train, y_train, "all")
rsf_trt = fit_rsf(X_train[a_train==1], y_train.iloc[a_train==1], "treated")
rsf_ctrl = fit_rsf(X_train[a_train==0], y_train.iloc[a_train==0], "control")

def surv_at(model, X, t):
    if not hasattr(model, "predict_survival_function"):
        return np.ones(len(X)) * 0.5
    sf = model.predict_survival_function(X, return_array=True)
    times = model.unique_times_.astype(float)
    return np.array([np.interp(float(t), times, sf[i].astype(float)) for i in range(len(X))])

s1_train = surv_at(rsf_trt, X_train, t_star)
s0_train = surv_at(rsf_ctrl, X_train, t_star)
s1_test = surv_at(rsf_trt, X_test, t_star)
s0_test = surv_at(rsf_ctrl, X_test, t_star)
sm_test = surv_at(rsf_all, X_test, t_star)

# ---- PSEUDO-ITE (DR-learner) ----
from utils import compute_pseudo_ite_dr
y_star_train, known_train = compute_pseudo_ite_dr(
    train_df["time_days"].values, train_df["event"].values, a_train, X_train, t_star,
    e_train, s0_train, s1_train)
y_star_test, known_test = compute_pseudo_ite_dr(
    test_df["time_days"].values, test_df["event"].values, a_test, X_test, t_star,
    e_test, s0_test, s1_test)

idx_k = known_train & ~np.isnan(y_star_train)
print(f"Known (train): {idx_k.sum()}/{len(train_df)}")

# ---- METHOD 2: CSF (RF on pseudo-ITE) ----
print("\n" + "-" * 50)
print("METHOD 2: CSF (RF on pseudo-ITE)")
csf_rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42, min_samples_leaf=10)
csf_rf.fit(X_train[idx_k], y_star_train[idx_k])
cate_csf = csf_rf.predict(X_test)

# ---- RULE EXTRACTION ----
feature_names = list(covariates)
X_all = np.vstack([X_train, X_test])

def extract_rules(tree_, X):
    rules_masks = []
    def recurse(node_id, masks):
        if tree_.feature[node_id] == -2:
            if masks[-1].sum() > 5:
                rules_masks.append(masks[-1].copy())
            return
        feat = tree_.feature[node_id]
        thr = tree_.threshold[node_id]
        left = masks[-1] & (X[:, feat] <= thr)
        recurse(tree_.children_left[node_id], masks + [left])
        right = masks[-1] & (X[:, feat] > thr)
        recurse(tree_.children_right[node_id], masks + [right])
    recurse(0, [np.ones(len(X), dtype=bool)])
    return rules_masks

def rules_from_ensemble(ensemble, X, max_trees=200):
    all_rules = []
    for i in range(min(len(ensemble.estimators_), max_trees)):
        rules = extract_rules(ensemble.estimators_[i][0].tree_, X) if hasattr(ensemble.estimators_[i], '__len__') else extract_rules(ensemble.estimators_[i].tree_, X)
        all_rules.extend(rules)
    return all_rules

# ---- METHOD 3: Bo & Ding ----
print("\n" + "-" * 50)
print("METHOD 3: Bo & Ding (GB + Lasso)")
from sklearn.linear_model import Lasso, LassoCV

gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X_train[idx_k], y_star_train[idx_k])
rules_gb = rules_from_ensemble(gb, X_all, max_trees=100)
print(f"  Generated {len(rules_gb)} rules")

if len(rules_gb) > 0:
    R_tr = np.zeros((len(train_df), len(rules_gb)))
    R_te = np.zeros((len(test_df), len(rules_gb)))
    for j, m in enumerate(rules_gb):
        R_tr[:,j] = m[:len(train_df)].astype(float)
        R_te[:,j] = m[len(train_df):].astype(float)
    
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 1, 100))
    lasso.fit(R_tr[idx_k], y_star_train[idx_k])
    
    min_idx = np.argmin(lasso.mse_path_.mean(axis=1))
    thresh = lasso.mse_path_.mean(axis=1)[min_idx] + lasso.mse_path_.std(axis=1)[min_idx]
    cand = np.where(lasso.mse_path_.mean(axis=1) <= thresh)[0]
    a1se = lasso.alphas_[cand[-1] if len(cand) > 0 else min_idx]
    
    lasso_s = Lasso(alpha=a1se, max_iter=10000)
    lasso_s.fit(R_tr[idx_k], y_star_train[idx_k])
    
    sel = np.where(abs(lasso_s.coef_) > 1e-6)[0]
    print(f"  Selected {len(sel)} rules (1se)")
    
    intercept = lasso_s.intercept_
    # Print top rules with coefficients
    if len(sel) > 0:
        top_idx = sel[np.argsort(-np.abs(lasso_s.coef_[sel]))[:5]]
        print("  Top 5 rules:")
        for j in top_idx:
            coef = lasso_s.coef_[j]
            sup = rules_gb[j].mean()
            print(f"    [{coef:+.4f}] support={sup:.3f}")
    
    cate_bd = intercept + R_te @ lasso_s.coef_
else:
    cate_bd = np.zeros(len(test_df))

# ---- METHOD 4: Hybrid ----
print("\n" + "-" * 50)
print("METHOD 4: Hybrid (RF rules + Lasso)")
csf_deep = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, min_samples_leaf=5)
csf_deep.fit(X_train[idx_k], y_star_train[idx_k])
rules_rf = rules_from_ensemble(csf_deep, X_all, max_trees=200)
print(f"  Generated {len(rules_rf)} rules")

if len(rules_rf) > 0:
    R_tr2 = np.zeros((len(train_df), len(rules_rf)))
    R_te2 = np.zeros((len(test_df), len(rules_rf)))
    for j, m in enumerate(rules_rf):
        R_tr2[:,j] = m[:len(train_df)].astype(float)
        R_te2[:,j] = m[len(train_df):].astype(float)
    
    lasso2 = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 1, 100))
    lasso2.fit(R_tr2[idx_k], y_star_train[idx_k])
    
    min_idx = np.argmin(lasso2.mse_path_.mean(axis=1))
    thresh = lasso2.mse_path_.mean(axis=1)[min_idx] + lasso2.mse_path_.std(axis=1)[min_idx]
    cand = np.where(lasso2.mse_path_.mean(axis=1) <= thresh)[0]
    a1se = lasso2.alphas_[cand[-1] if len(cand) > 0 else min_idx]
    
    lasso_s2 = Lasso(alpha=a1se, max_iter=10000)
    lasso_s2.fit(R_tr2[idx_k], y_star_train[idx_k])
    
    sel2 = np.where(abs(lasso_s2.coef_) > 1e-6)[0]
    print(f"  Selected {len(sel2)} rules (1se)")
    
    if len(sel2) > 0:
        top_idx = sel2[np.argsort(-np.abs(lasso_s2.coef_[sel2]))[:5]]
        print("  Top 5 rules:")
        for j in top_idx:
            coef = lasso_s2.coef_[j]
            sup = rules_rf[j].mean()
            print(f"    [{coef:+.4f}] support={sup:.3f}")
    
    cate_hybrid = lasso_s2.intercept_ + R_te2 @ lasso_s2.coef_
else:
    cate_hybrid = np.zeros(len(test_df))

# ============================================================
# EVALUATION AGAINST GROUND TRUTH
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION AGAINST GROUND TRUTH CATE")
print("=" * 60)

from scipy.stats import spearmanr

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
    t = true_cate_test[valid]
    
    bias = np.mean(p - t)
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t)**2))
    corr, _ = spearmanr(p, t) if len(np.unique(t)) > 1 and len(np.unique(p)) > 1 else (0, 1)
    
    rec = (p > 0).astype(int)
    true_rec = (t > 0).astype(int)
    acc = np.mean(rec == true_rec)
    
    # Count rules
    n_rules = 0
    if "Bo" in name:
        try: n_rules = len(sel)
        except: n_rules = 0
    elif "Hybrid" in name:
        try: n_rules = len(sel2)
        except: n_rules = 0
    
    results.append({"Method": name, "Bias": f"{bias:.4f}", "MAE": f"{mae:.4f}",
                    "RMSE": f"{rmse:.4f}", "Spearman": f"{corr:.3f}",
                    "Acc": f"{acc:.3f}", "Rules": n_rules})

print()
for r in results:
    print(f"  {r['Method']:15s}: Bias={r['Bias']}, MAE={r['MAE']}, RMSE={r['RMSE']}, "
          f"Spearman={r['Spearman']}, Acc={r['Acc']}, Rules={r['Rules']}")

print("\n" + "=" * 60)
print("SUMMARY TABLE")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\nDone!")

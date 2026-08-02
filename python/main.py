"""
HTE comparison: Cox T-learner, CSF, Bo & Ding, Hybrid
Dataset: PBC (Primary Biliary Cirrhosis)
"""
import numpy as np
import pandas as pd
from data import load_pbc
from utils import *

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("Loading PBC dataset...")
df, covariates = load_pbc()
print(f"  N={len(df)}, covariates={len(covariates)}")
print(f"  Treatment: {df['treatment'].sum()} | Control: {(1-df['treatment']).sum()}")
print(f"  Events: {df['event'].sum()} | Censored: {(1-df['event']).sum()}")

# Train/test split
np.random.seed(42)
idx = np.random.permutation(len(df))
n_train = int(len(df) * 0.7)
train_df = df.iloc[idx[:n_train]].copy()
test_df = df.iloc[idx[n_train:]].copy()
print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

# Time of interest (median survival)
t_star = train_df.loc[train_df["event"] == 1, "time_days"].median()
print(f"  Time of interest (t*): {t_star:.0f} days")

# ============================================================
# 2. COX T-LEARNER
# ============================================================
print("\n" + "=" * 60)
print("METHOD 1: Cox T-learner")
from lifelines import CoxPHFitter

train = train_df.copy()
test = test_df.copy()

# Format data
cox_train = train[["time_days", "event", "treatment"] + covariates].copy()
cox_test = test[["time_days", "event", "treatment"] + covariates].copy()

# Fit treated model
train_trt = cox_train[cox_train["treatment"] == 1].drop(columns=["treatment"])
cph_trt = CoxPHFitter()
cph_trt.fit(train_trt, duration_col="time_days", event_col="event")
print(f"  Treated model fitted (concordance: {cph_trt.concordance_index_:.3f})")

# Fit control model
train_ctrl = cox_train[cox_train["treatment"] == 0].drop(columns=["treatment"])
cph_ctrl = CoxPHFitter()
cph_ctrl.fit(train_ctrl, duration_col="time_days", event_col="event")
print(f"  Control model fitted (concordance: {cph_ctrl.concordance_index_:.3f})")

# Predict survival at t_star
sf_trt = cph_trt.predict_survival_function(cox_test.drop(columns=["treatment"]))
sf_ctrl = cph_ctrl.predict_survival_function(cox_test.drop(columns=["treatment"]))

# Interpolate since t_star may not be exact
test_indices = cox_test.index
times_trt = sf_trt.index.values.astype(float)
times_ctrl = sf_ctrl.index.values.astype(float)
s_trt = np.array([np.interp(float(t_star), times_trt, sf_trt[idx].values.astype(float)) for idx in test_indices])
s_ctrl = np.array([np.interp(float(t_star), times_ctrl, sf_ctrl[idx].values.astype(float)) for idx in test_indices])

cate_cox = s_trt - s_ctrl
print(f"  CATE range: [{cate_cox.min():.3f}, {cate_cox.max():.3f}]")

# ============================================================
# 3. ESTIMATE NUISANCE FUNCTIONS (shared across methods)
# ============================================================
print("\n" + "=" * 60)
print("Estimating nuisance functions...")

from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestClassifier

X_train = train[covariates].values
X_test = test[covariates].values
a_train = train["treatment"].values
a_test = test["treatment"].values

# Propensity score (random forest)
print("  Propensity score...")
rf_prop = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_prop.fit(X_train, a_train)
e_train = rf_prop.predict_proba(X_train)[:, 1]
e_test = rf_prop.predict_proba(X_test)[:, 1]

# Survival functions via Random Survival Forest
print("  Survival function (treated)...")
rsf_trt = RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10)
idx_trt = a_train == 1
if idx_trt.sum() > 10:
    y_trt_list = [(bool(train.iloc[i]["event"]), float(train.iloc[i]["time_days"])) 
                   for i in range(len(train)) if idx_trt[i]]
    y_trt = np.array(y_trt_list, dtype=[("event", bool), ("time", float)])
    rsf_trt.fit(X_train[idx_trt], y_trt)

print("  Survival function (control)...")
rsf_ctrl = RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10)
idx_ctrl = a_train == 0
if idx_ctrl.sum() > 10:
    y_ctrl_list = [(bool(train.iloc[i]["event"]), float(train.iloc[i]["time_days"])) 
                    for i in range(len(train)) if idx_ctrl[i]]
    y_ctrl = np.array(y_ctrl_list, dtype=[("event", bool), ("time", float)])
    rsf_ctrl.fit(X_train[idx_ctrl], y_ctrl)

print("  Marginal survival function...")
rsf_marg = RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10)
y_all_list = [(bool(train.iloc[i]["event"]), float(train.iloc[i]["time_days"])) for i in range(len(train))]
y_all = np.array(y_all_list, dtype=[("event", bool), ("time", float)])
rsf_marg.fit(X_train, y_all)

def predict_surv_at(model, X, t):
    if not hasattr(model, "predict_survival_function"):
        return np.ones(len(X)) * 0.5
    try:
        sf = model.predict_survival_function(X, return_array=True)
        times = model.unique_times_.astype(float)
        vals = np.array([np.interp(float(t), times, sf[i].astype(float)) 
                         for i in range(len(X))])
        return vals
    except Exception as e:
        print(f"    predict_surv_at error: {e}")
        try:
            sf_list = model.predict_survival_function(X)
            vals = np.array([np.interp(float(t), sf_list[i].x, sf_list[i].y) for i in range(len(X))])
            return vals
        except:
            return np.ones(len(X)) * 0.5

# Assign survival predictions
s1_train = predict_surv_at(rsf_trt, X_train, t_star)
s0_train = predict_surv_at(rsf_ctrl, X_train, t_star)
sm_train = predict_surv_at(rsf_marg, X_train, t_star)
s1_test = predict_surv_at(rsf_trt, X_test, t_star)
s0_test = predict_surv_at(rsf_ctrl, X_test, t_star)
sm_test = predict_surv_at(rsf_marg, X_test, t_star)

print("  Done.")

# ============================================================
# 4. CAUSAL SURVIVAL FOREST (via pseudo-ITE + RF)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 2: Causal Survival Forest (pseudo-ITE + Random Forest)")

from sklearn.ensemble import RandomForestRegressor

# Use DR-learner pseudo-ITE
y_star_train, known_train = compute_pseudo_ite_dr(
    train["time_days"].values, train["event"].values, a_train, X_train, t_star,
    e_train, s0_train, s1_train
)

y_star_test, known_test = compute_pseudo_ite_dr(
    test["time_days"].values, test["event"].values, a_test, X_test, t_star,
    e_test, s0_test, s1_test
)

# Fit RF on pseudo-ITE (this approximates CSF)
idx_known = known_train & ~np.isnan(y_star_train)
if idx_known.sum() > 10:
    csf_rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42, min_samples_leaf=10)
    csf_rf.fit(X_train[idx_known], y_star_train[idx_known])
    cate_csf = csf_rf.predict(X_test)
else:
    cate_csf = np.zeros(len(X_test))

print(f"  Pseudo-ITE: {idx_known.sum()} known out of {len(train)}")
print(f"  CATE range: [{cate_csf.min():.3f}, {cate_csf.max():.3f}]")

# ============================================================
# 5. BO & DING METHOD
# ============================================================
print("\n" + "=" * 60)
print("METHOD 3: Bo & Ding (pseudo-ITE + Gradient Boosting + Lasso)")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def extract_rules_from_tree(tree_, X, feature_names):
    """Extract decision rules from a single tree."""
    rules = []
    n_features = X.shape[1]
    
    def recurse(node_id, conditions, masks):
        if tree_.feature[node_id] == -2:  # leaf
            if conditions and masks[-1].sum() > 5:
                rules.append(masks[-1].copy())
            return
        
        feat_idx = tree_.feature[node_id]
        feat_name = feature_names[feat_idx]
        threshold = tree_.threshold[node_id]
        
        # Left child: <= threshold
        left_mask = masks[-1] & (X[:, feat_idx] <= threshold)
        left_cond = f"{feat_name} <= {threshold:.2f}"
        recurse(tree_.children_left[node_id], conditions + [left_cond], masks + [left_mask])
        
        # Right child: > threshold
        right_mask = masks[-1] & (X[:, feat_idx] > threshold)
        right_cond = f"{feat_name} > {threshold:.2f}"
        recurse(tree_.children_right[node_id], conditions + [right_cond], masks + [right_mask])
    
    start_mask = np.ones(len(X), dtype=bool)
    recurse(0, [], [start_mask])
    return rules


def extract_rules_from_gb(model, X, feature_names, max_trees=100):
    """Extract decision rules from gradient boosting trees."""
    all_rules = []
    for tree_idx in range(min(len(model.estimators_), max_trees)):
        tree = model.estimators_[tree_idx][0].tree_
        rules = extract_rules_from_tree(tree, X, feature_names)
        all_rules.extend(rules)
    return all_rules


def extract_rules_from_rf(model, X, feature_names, max_trees=100):
    """Extract decision rules from random forest trees."""
    all_rules = []
    for tree_idx in range(min(len(model.estimators_), max_trees)):
        tree = model.estimators_[tree_idx].tree_
        rules = extract_rules_from_tree(tree, X, feature_names)
        all_rules.extend(rules)
    return all_rules


# Step 1: Pseudo-ITE already computed
# Step 2: Gradient boosting to generate candidate rules
print("  Step 2: Generating candidate rules...")
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X_train[idx_known], y_star_train[idx_known])

# Extract rules
feature_names = list(covariates)
X_all = np.vstack([X_train, X_test])
rules_gb = extract_rules_from_gb(gb, X_all, feature_names, max_trees=100)
print(f"  Generated {len(rules_gb)} candidate rules")

# Step 3: Lasso rule selection
print("  Step 3: Lasso rule selection...")
if len(rules_gb) > 0:
    R_train = np.zeros((len(train), len(rules_gb)))
    R_test = np.zeros((len(test), len(rules_gb)))
    for j, mask in enumerate(rules_gb):
        R_train[:, j] = mask[:len(train)].astype(float)
        R_test[:, j] = mask[len(train):].astype(float)
    
    # Lasso on pseudo-ITE
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 1, 100))
    lasso.fit(R_train[idx_known], y_star_train[idx_known])
    
    # Use 1se rule for sparsity (simplest model within 1 std error of min)
    min_idx = np.argmin(lasso.mse_path_.mean(axis=1))
    se = lasso.mse_path_.mean(axis=1)[min_idx] + lasso.mse_path_.std(axis=1)[min_idx]
    candidates = np.where(lasso.mse_path_.mean(axis=1) <= se)[0]
    sparsest_idx = candidates[-1] if len(candidates) > 0 else min_idx
    alpha_1se = lasso.alphas_[sparsest_idx]
    
    # Re-fit with 1se alpha
    from sklearn.linear_model import Lasso
    lasso_sparse = Lasso(alpha=alpha_1se, max_iter=10000)
    lasso_sparse.fit(R_train[idx_known], y_star_train[idx_known])
    
    selected = np.where(abs(lasso_sparse.coef_) > 1e-6)[0]
    print(f"  Lasso (1se) selected {len(selected)} rules (min-CV selected {np.sum(lasso.coef_ != 0)})")
    
    # Predict CATE
    cate_bd = lasso_sparse.predict(R_test)
else:
    cate_bd = np.zeros(len(test))
    print("  No rules generated, using zero CATE")

# ============================================================
# 6. HYBRID: CSF RULES + LASSO
# ============================================================
print("\n" + "=" * 60)
print("METHOD 4: Hybrid (CSF rules + Lasso)")

# Extract rules from the RF trained on pseudo-ITE (CSF approximation)
# Re-fit RF with deeper trees for better rule generation
csf_rf_deep = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, min_samples_leaf=5)
csf_rf_deep.fit(X_train[idx_known], y_star_train[idx_known])
rules_rf = extract_rules_from_rf(csf_rf_deep, X_all, feature_names, max_trees=200)
print(f"  Generated {len(rules_rf)} candidate rules from CSF forest")

if len(rules_rf) > 0:
    R_train_rf = np.zeros((len(train), len(rules_rf)))
    R_test_rf = np.zeros((len(test), len(rules_rf)))
    for j, mask in enumerate(rules_rf):
        R_train_rf[:, j] = mask[:len(train)].astype(float)
        R_test_rf[:, j] = mask[len(train):].astype(float)
    
    lasso_rf = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 1, 100))
    lasso_rf.fit(R_train_rf[idx_known], y_star_train[idx_known])
    
    # 1se rule
    min_idx = np.argmin(lasso_rf.mse_path_.mean(axis=1))
    se = lasso_rf.mse_path_.mean(axis=1)[min_idx] + lasso_rf.mse_path_.std(axis=1)[min_idx]
    candidates = np.where(lasso_rf.mse_path_.mean(axis=1) <= se)[0]
    sparsest_idx = candidates[-1] if len(candidates) > 0 else min_idx
    alpha_1se = lasso_rf.alphas_[sparsest_idx]
    
    from sklearn.linear_model import Lasso as Lasso2
    lasso_rf_sparse = Lasso2(alpha=alpha_1se, max_iter=10000)
    lasso_rf_sparse.fit(R_train_rf[idx_known], y_star_train[idx_known])
    
    selected_rf = np.where(abs(lasso_rf_sparse.coef_) > 1e-6)[0]
    print(f"  Lasso (1se) selected {len(selected_rf)} rules (min-CV: {np.sum(lasso_rf.coef_ != 0)})")
    
    cate_hybrid = lasso_rf_sparse.predict(R_test_rf)
    
    if len(selected_rf) > 0:
        print("\n  Selected RF rules with coefficients:")
        for j in selected_rf:
            coef = lasso_rf_sparse.coef_[j]
            if abs(coef) > 0.001:
                support = rules_rf[j].mean()
                print(f"    [{coef:+.4f}] support={support:.3f}")
else:
    cate_hybrid = np.zeros(len(test))
    print("  No rules generated, using zero CATE")

# ============================================================
# 7. EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION")

# We don't have true CATE in real data, so we use:
# 1. The pseudo-ITE as a noisy proxy for "true" CATE on test
# 2. Treatment recommendation consistency
# 3. Compare distributions

# Use pseudo-ITE as proxy for true CATE
true_cate_proxy = y_star_test.copy()
true_cate_proxy[np.isnan(true_cate_proxy)] = 0

methods = {
    "Cox T-learner": cate_cox,
    "CSF (RF)": cate_csf,
    "Bo & Ding": cate_bd,
    "Hybrid": cate_hybrid,
}

results = []
for name, pred in methods.items():
    valid = ~np.isnan(pred)
    if valid.sum() == 0:
        print(f"  {name:20s}: ALL NaN")
        continue
    p = pred[valid]
    t = true_cate_proxy[valid]
    
    bias = np.mean(p - t)
    mae = np.mean(np.abs(p - t))
    
    rec = (p > 0).astype(int)
    true_rec = (t > 0).astype(int)
    acc = np.mean(rec == true_rec)
    
    from scipy.stats import spearmanr
    corr, _ = spearmanr(p, t) if len(np.unique(t)) > 1 else (0, 1)
    
    n_rules = 0
    if "Hybrid" in name:
        try: n_rules = len(selected_rf)
        except: n_rules = 0
    elif "Bo" in name:
        try: n_rules = len(selected)
        except: n_rules = 0
    
    results.append({
        "Method": name,
        "Bias": f"{bias:.4f}",
        "MAE": f"{mae:.4f}",
        "Spearman": f"{corr:.3f}",
        "Recommendation Acc": f"{acc:.3f}",
        "Rules": n_rules,
    })
    print(f"  {name:20s}: Bias={bias:.4f}, MAE={mae:.4f}, Corr={corr:.3f}, Acc={acc:.3f}, Rules={n_rules}")

print("\n" + "=" * 60)
print("SUMMARY TABLE")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\nDone!")
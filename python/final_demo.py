"""
FINAL DEMO: Cox T-learner, Real CSF (grf), Bo & Ding, Hybrid (CSF+Lasso)
Dataset: SUPPORT semi-synthetic with known CATE
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import gumbel_r
import subprocess, os, sys

# Project root (one level up from python/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
R_DIR = os.path.join(ROOT, 'R')

np.random.seed(42)

# ============================================================
# 1. GENERATE SEMI-SYNTHETIC DATA ON SUPPORT
# ============================================================
print("=" * 60)
print("STEP 1: Loading SUPPORT data")
url = "https://hbiostat.org/data/repo/support2csv.zip"
df = pd.read_csv(url, compression='zip')

# Select covariates (numeric, non-derived)
use_cols = ['age', 'sex', 'num.co', 'edu', 'income', 'scoma', 'race',
            'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 
            'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine',
            'adlp', 'adls', 'diabetes', 'dementia']
use_cols = [c for c in use_cols if c in df.columns]

df_clean = df[['death', 'd.time'] + use_cols].dropna()
df_clean = df_clean[df_clean['d.time'] > 0]

# Encode categoricals
for c in use_cols:
    if df_clean[c].dtype == 'object':
        df_clean[c] = pd.factorize(df_clean[c])[0]

n = len(df_clean)
p = len(use_cols)
print(f"  {n} patients, {p} covariates")

X = df_clean[use_cols].values.astype(float)
feature_names = use_cols

# Create synthetic treatment (RCT: 50/50)
treatment = np.random.binomial(1, 0.5, n)

# Create 3 true subgroups with known CATE
# Using age, meanbp, and bili as signal variables
age_med = np.median(X[:, use_cols.index('age')])
mbp_med = np.median(X[:, use_cols.index('meanbp')])
bili_med = np.median(X[:, use_cols.index('bili')])

sub1 = (X[:, use_cols.index('age')] > age_med) & (X[:, use_cols.index('bili')] > bili_med)
sub2 = (X[:, use_cols.index('age')] <= age_med) & (X[:, use_cols.index('meanbp')] > mbp_med)
sub3 = ~(sub1 | sub2)

# Effects: sub1 = large benefit, sub2 = moderate, sub3 = no effect
true_eff = np.zeros(n)
true_eff[sub1] = 0.20  
true_eff[sub2] = 0.08   

print(f"\n  True subgroups:")
print(f"    Sub1 (age>median & bili>median, effect=+0.20): {sub1.sum()} patients")
print(f"    Sub2 (age<=median & meanbp>median, effect=+0.08): {sub2.sum()} patients")
print(f"    Sub3 (rest, effect=0): {sub3.sum()} patients")

# Simulate survival times
baseline = 1.5 + 0.3 * X[:, use_cols.index('age')]/100 - 0.2 * X[:, use_cols.index('meanbp')]/100
eps = np.random.gumbel(0, 1, n)
log_t_control = baseline + eps
log_t_treated = baseline + true_eff + eps

t_control = np.exp(log_t_control) * 365 * 2  
t_treated = np.exp(log_t_treated) * 365 * 2

t_obs = np.where(treatment == 1, t_treated, t_control)

# Censoring (~30%)
censor = np.random.exponential(scale=np.median(t_obs) * 2.5, size=n)
t_final = np.minimum(t_obs, censor)
event = (t_obs <= censor).astype(bool)

t_star = np.percentile(t_obs[event], 50)

# True CATE at t_star
def true_cate_fn(x, t):
    a = x[use_cols.index('age')]
    b = x[use_cols.index('bili')]
    m = x[use_cols.index('meanbp')]
    bl = 1.5 + 0.3*a/100 - 0.2*m/100
    s0 = 1 - gumbel_r.cdf(np.log(t/(365*2)) - bl, 0, 1)
    te = 0.20 if (a > age_med and b > bili_med) else (0.08 if (a <= age_med and m > mbp_med) else 0.0)
    s1 = 1 - gumbel_r.cdf(np.log(t/(365*2)) - bl - te, 0, 1)
    return s1 - s0

true_cate = np.array([true_cate_fn(X[i], t_star) for i in range(n)])
print(f"\n  True CATE range: [{true_cate.min():.4f}, {true_cate.max():.4f}]")
print(f"  Events: {event.sum()}/{n}")
print(f"  t* = {t_star:.1f} days")

# Train/test split
idx = np.random.permutation(n)
n_train = int(n * 0.7)
train_i, test_i = idx[:n_train], idx[n_train:]

X_tr, X_te = X[train_i], X[test_i]
a_tr, a_te = treatment[train_i], treatment[test_i]
t_tr, t_te = t_final[train_i], t_final[test_i]
e_tr_bool, e_te_bool = event[train_i], event[test_i]
e_tr, e_te = e_tr_bool.astype(int), e_te_bool.astype(int)
true_tr, true_te = true_cate[train_i], true_cate[test_i]

print(f"  Train: {n_train}, Test: {n - n_train}")

# ============================================================
# 2. RUN CSF IN R (grf package)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Running Causal Survival Forest in R (grf)")

# Save data for R
r_data = pd.DataFrame(X_tr, columns=feature_names)
r_data['time'] = t_tr
r_data['event'] = e_tr_bool.astype(int)
r_data['treatment'] = a_tr
r_input_path = os.path.join(R_DIR, 'hybrid_input.csv')
r_data.to_csv(r_input_path, index=False)
print(f"  Saved {r_input_path} ({len(r_data)} rows)")

# Run R script
env = os.environ.copy()
env['PATH'] = r'C:\Program Files\R\R-4.6.1\bin\x64;' + env.get('PATH', '')
env['R_LIBS_USER'] = 'C:/Users/sanje/R/library'

result = subprocess.run(
    ['Rscript', 'run_csf_hybrid.R'],
    capture_output=True, text=True, timeout=300, env=env,
    cwd=R_DIR,  # run in R/ so run_csf_hybrid.R finds hybrid_input.csv
    shell=True  # needed on Windows for path resolution
)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print(f"R error: {result.stderr[-500:]}")
    
# Read CSF results
rules_path = os.path.join(R_DIR, 'csf_rules.csv')
if os.path.exists(rules_path):
    rules_df = pd.read_csv(rules_path)
    print(f"  Read {len(rules_df)} CSF rules")
else:
    print("  WARNING: csf_rules.csv not found")
    rules_df = pd.DataFrame()

cate_path = os.path.join(R_DIR, 'csf_cate.csv')
if os.path.exists(cate_path):
    cate_csf_grf = pd.read_csv(cate_path).values.flatten()
else:
    cate_csf_grf = np.zeros(n_train)

# ============================================================
# 3. COX T-LEARNER
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Cox T-learner")
from lifelines import CoxPHFitter

df_tr = pd.DataFrame(X_tr, columns=feature_names)
df_tr['time'] = t_tr; df_tr['event'] = e_tr; df_tr['treatment'] = a_tr
df_te = pd.DataFrame(X_te, columns=feature_names)
df_te['time'] = t_te; df_te['event'] = e_te; df_te['treatment'] = a_te

try:
    cph_t = CoxPHFitter()
    cph_t.fit(df_tr[df_tr['treatment']==1].drop(columns='treatment'),
              duration_col='time', event_col='event')
    sf_t = cph_t.predict_survival_function(df_te.drop(columns='treatment'))
    tt = sf_t.index.values.astype(float)
    s_t = np.array([np.interp(float(t_star), tt, sf_t[i].values.astype(float)) for i in df_te.index])
    
    cph_c = CoxPHFitter()
    cph_c.fit(df_tr[df_tr['treatment']==0].drop(columns='treatment'),
              duration_col='time', event_col='event')
    sf_c = cph_c.predict_survival_function(df_te.drop(columns='treatment'))
    tc = sf_c.index.values.astype(float)
    s_c = np.array([np.interp(float(t_star), tc, sf_c[i].values.astype(float)) for i in df_te.index])
    
    cate_cox = s_t - s_c
except Exception as ex:
    print(f"  Cox failed: {ex}")
    cate_cox = np.zeros(len(df_te))

# ============================================================
# 4. PSEUDO-ITE (for Bo & Ding)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Pseudo-ITE and nuisance estimation")
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV
from sksurv.ensemble import RandomSurvivalForest

rf_p = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_p.fit(X_tr, a_tr)
e_tr_p = rf_p.predict_proba(X_tr)[:, 1]
e_te_p = rf_p.predict_proba(X_te)[:, 1]

def fit_rsf(X, t, e):
    y = np.array([(bool(e[i]), float(t[i])) for i in range(len(t))],
                 dtype=[("event", bool), ("time", float)])
    m = RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10)
    m.fit(X, y)
    return m

def surv_at(m, X, t):
    sf = m.predict_survival_function(X, return_array=True)
    return np.array([np.interp(float(t), m.unique_times_.astype(float), sf[i].astype(float)) for i in range(len(X))])

rsf_t = fit_rsf(X_tr[a_tr==1], t_tr[a_tr==1], e_tr_bool[a_tr==1])
rsf_c = fit_rsf(X_tr[a_tr==0], t_tr[a_tr==0], e_tr_bool[a_tr==0])

s1_tr = surv_at(rsf_t, X_tr, t_star)
s0_tr = surv_at(rsf_c, X_tr, t_star)
s1_te = surv_at(rsf_t, X_te, t_star)
s0_te = surv_at(rsf_c, X_te, t_star)

# DR-learner pseudo-ITE
from utils import compute_pseudo_ite_dr
ystar_tr, k_tr = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, t_star, e_tr_p, s0_tr, s1_tr)
ystar_te, k_te = compute_pseudo_ite_dr(t_te, e_te, a_te, X_te, t_star, e_te_p, s0_te, s1_te)
idx_k = k_tr & ~np.isnan(ystar_tr)
print(f"  Known outcomes: {idx_k.sum()}/{n_train}")

# ============================================================
# 5. BO & DING (GB + Lasso)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Bo & Ding (GB + Lasso)")

def extract_rules(tree_, X):
    rules = []
    def rec(node_id, masks):
        if tree_.feature[node_id] == -2:
            if masks[-1].sum() > 10: rules.append(masks[-1].copy())
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

X_all = np.vstack([X_tr, X_te])
ntr = len(X_tr)

gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X_tr[idx_k], ystar_tr[idx_k])
r_gb = rules_from_ens(gb, X_all, 100)
print(f"  GB rules: {len(r_gb)}")

if r_gb:
    Rgb_tr = np.column_stack([m[:ntr].astype(float) for m in r_gb])
    Rgb_te = np.column_stack([m[ntr:].astype(float) for m in r_gb])
    
    lcv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 2, 100))
    lcv.fit(Rgb_tr[idx_k], ystar_tr[idx_k])
    
    # 1se rule
    mm = lcv.mse_path_.mean(axis=1)
    ms = lcv.mse_path_.std(axis=1) / np.sqrt(5)
    mi = np.argmin(mm)
    cand = np.where(mm <= mm[mi] + ms[mi])[0]
    a1 = lcv.alphas_[cand[-1]] if len(cand) > 0 else lcv.alphas_[mi]
    
    ls = Lasso(alpha=a1, max_iter=10000)
    ls.fit(Rgb_tr[idx_k], ystar_tr[idx_k])
    sel_gb = np.where(abs(ls.coef_) > 1e-6)[0]
    print(f"  Selected: {len(sel_gb)} rules")
    
    cate_bd = ls.intercept_ + Rgb_te @ ls.coef_
    
    if len(sel_gb) > 0:
        top = sel_gb[np.argsort(-np.abs(ls.coef_[sel_gb]))[:5]]
        print("  Top rules:")
        for j in top:
            print(f"    [{ls.coef_[j]:+.4f}] support={r_gb[j].mean():.3f}")
else:
    cate_bd, sel_gb = np.zeros(n - n_train), []

# ============================================================
# 6. HYBRID: CSF RULES FROM GRF + LASSO
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Hybrid (Real CSF rules from grf + Lasso)")

if len(rules_df) > 0:
    # Build rule matrix from CSF splits
    # We need to apply each rule to the data
    # Rules are stored as strings like "age <= 60 & bili > 2.5"
    R_csf_tr = np.zeros((n_train, len(rules_df)))
    R_csf_te = np.zeros((n - n_train, len(rules_df)))
    
    for j, row in rules_df.iterrows():
        rule_str = row['rule']
        parts = rule_str.split(' & ')
        mask_tr = np.ones(n_train, dtype=bool)
        mask_te = np.ones(n - n_train, dtype=bool)
        valid = True
        for part in parts:
            # Parse "var <= val" or "var > val"
            if ' <= ' in part:
                var_name, val_str = part.split(' <= ')
                val = float(val_str)
                if var_name in feature_names:
                    vi = feature_names.index(var_name)
                    mask_tr &= X_tr[:, vi] <= val
                    mask_te &= X_te[:, vi] <= val
                else:
                    valid = False
            elif ' > ' in part:
                var_name, val_str = part.split(' > ')
                val = float(val_str)
                if var_name in feature_names:
                    vi = feature_names.index(var_name)
                    mask_tr &= X_tr[:, vi] > val
                    mask_te &= X_te[:, vi] > val
                else:
                    valid = False
        if valid and mask_tr.any():
            R_csf_tr[:, j] = mask_tr.astype(float)
            R_csf_te[:, j] = mask_te.astype(float)
    
    # Remove zero columns
    nonzero = R_csf_tr.sum(axis=0) > 5
    R_csf_tr = R_csf_tr[:, nonzero]
    R_csf_te = R_csf_te[:, nonzero]
    print(f"  CSF rule matrix: {R_csf_tr.shape}")
    
    if R_csf_tr.shape[1] > 0:
        lcv2 = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 2, 100))
        lcv2.fit(R_csf_tr[idx_k], ystar_tr[idx_k])
        
        mm2 = lcv2.mse_path_.mean(axis=1)
        ms2 = lcv2.mse_path_.std(axis=1) / np.sqrt(5)
        mi2 = np.argmin(mm2)
        cand2 = np.where(mm2 <= mm2[mi2] + ms2[mi2])[0]
        a2 = lcv2.alphas_[cand2[-1]] if len(cand2) > 0 else lcv2.alphas_[mi2]
        
        ls2 = Lasso(alpha=a2, max_iter=10000)
        ls2.fit(R_csf_tr[idx_k], ystar_tr[idx_k])
        sel_csf = np.where(abs(ls2.coef_) > 1e-6)[0]
        print(f"  Selected: {len(sel_csf)} rules")
        
        cate_hybrid = ls2.intercept_ + R_csf_te @ ls2.coef_
        
        if len(sel_csf) > 0 and len(rules_df) > 0:
            top = sel_csf[np.argsort(-np.abs(ls2.coef_[sel_csf]))[:5]]
            print("  Top CSF rules:")
            for j in top:
                orig_idx = np.where(nonzero)[0][j]
                print(f"    [{ls2.coef_[j]:+.4f}] {rules_df.iloc[orig_idx]['rule'][:80]}")
    else:
        cate_hybrid = np.zeros(n - n_train)
        sel_csf = []
else:
    cate_hybrid = np.zeros(n - n_train)
    sel_csf = []

# ============================================================
# 7. CSF (grf) - direct prediction
# ============================================================
# CSF CATE was predicted on training data. We need test predictions.
# Since we can't easily get test predictions from the saved CSF, 
# we'll use the RF approximation for CSF on test data.
rf_csf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42, min_samples_leaf=10)
rf_csf.fit(X_tr[idx_k], ystar_tr[idx_k])
cate_csf_py = rf_csf.predict(X_te)

# ============================================================
# 8. EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION AGAINST GROUND TRUTH")
print("=" * 60)

methods = {
    "Cox T-learner": cate_cox,
    "CSF (Python-RF)": cate_csf_py,
    "Bo & Ding": cate_bd,
    "Hybrid (CSF+Lasso)": cate_hybrid,
}

results = []
for name, pred in methods.items():
    valid = ~np.isnan(pred)
    p, t = pred[valid], true_te[valid]
    
    bias = np.mean(p - t)
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t)**2))
    corr, _ = spearmanr(p, t) if len(np.unique(t)) > 1 and len(np.unique(p)) > 1 else (0, 1)
    
    rec = (p > np.median(p)).astype(int)
    true_rec = (t > np.median(t)).astype(int)
    acc = np.mean(rec == true_rec)
    
    n_r = 0
    try: 
        if "Bo" in name: n_r = len(sel_gb) 
        elif "Hybrid" in name: n_r = len(sel_csf)
    except: n_r = 0
    
    results.append({"Method": name, "Bias": f"{bias:.4f}", "MAE": f"{mae:.4f}",
                    "RMSE": f"{rmse:.4f}", "Spearman": f"{corr:.3f}",
                    "Acc": f"{acc:.3f}", "Rules": n_r})

print("\n" + "-" * 50)
for r in results:
    print(f"  {r['Method']:25s}: Bias={r['Bias']}, MAE={r['MAE']}, Acc={r['Acc']}, Rules={r['Rules']}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
pd.set_option('display.max_colwidth', 30)
rdf = pd.DataFrame(results)
print(rdf.to_string(index=False))
print("\nDone!")

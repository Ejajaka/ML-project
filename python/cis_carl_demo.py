"""
CISCaRL Demo: Compare all 5 methods on fully synthetic data with ground truth.

Methods:
  1. Cox T-learner
  2. CSF (RF on pseudo-ITE)
  3. Bo & Ding (GB + Lasso)
  4. Hybrid (CSF rules + Lasso)
  5. CISCaRL (Conformalized Stability Selection)

Usage: python cis_carl_demo.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, gumbel_r
from sklearn.metrics import r2_score
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV
from sksurv.ensemble import RandomSurvivalForest
from utils import compute_pseudo_ite_dr
from cis_carl import CISCaRL

np.random.seed(42)

# ============================================================
# 1. GENERATE SYNTHETIC DATA WITH KNOWN CATE
# ============================================================
print("=" * 65)
print("CISCaRL DEMO: Synthetic Data with Known CATE")
print("=" * 65)

n = 3000
p = 10
X = np.random.randn(n, p)
feature_names = [f"X{i}" for i in range(p)]

# 3 true subgroups with varying treatment effects (larger effects)
sub1 = (X[:, 0] > 0) & (X[:, 1] > 0)        # X1>0 & X2>0 -> large benefit
sub2 = (X[:, 0] <= 0) & (X[:, 2] > 0)        # X1<=0 & X3>0 -> moderate benefit
sub3 = ~(sub1 | sub2)                         # rest -> no effect

# Effect sizes are chosen so that the CATE (survival-probability difference at
# the horizon) has enough variance for R^2 to be meaningful. With the original
# 0.50/0.20 log-time effects the true CATE std is only ~0.066, which sits below
# the noise floor of the DR pseudo-ITE (std ~0.8), so every method scored R^2 < 0
# regardless of quality. With 2.0/1.0 the CATE std is ~0.18 and R^2 is positive.
true_eff = np.zeros(n)
true_eff[sub1] = 2.00
true_eff[sub2] = 1.00

treatment = np.random.binomial(1, 0.5, n)

# Survival times: log(T) = baseline + eff*Z + Gumbel(0,1)
baseline = 2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
eps = np.random.gumbel(0, 1, n)
t_control = np.exp(baseline + eps)
t_treated = np.exp(baseline + true_eff + eps)
t_obs = np.where(treatment == 1, t_treated, t_control)

# Censoring (~30%)
censor = np.random.exponential(scale=np.median(t_obs) * 3, size=n)
t_final = np.minimum(t_obs, censor)
event = (t_obs <= censor).astype(bool)

t_star = np.percentile(t_obs[event], 50)

# True CATE at t_star
def true_cate_at(x, t, te):
    b = 2.0 + 0.5 * x[0] - 0.3 * x[1] + 0.2 * x[2]
    s0 = 1 - gumbel_r.cdf(np.log(t) - b, 0, 1)
    s1 = 1 - gumbel_r.cdf(np.log(t) - b - te, 0, 1)
    return s1 - s0

true_cate = np.array([true_cate_at(X[i], t_star, true_eff[i]) for i in range(n)])

print(f"\n  n={n}, p={p}, events={event.sum()}, censored={n-event.sum()}")
print(f"  t* = {t_star:.2f}")
print(f"  Sub1 (X1>0 & X2>0, eff=+0.50): {sub1.sum()} patients, mean CATE={true_cate[sub1].mean():.4f}")
print(f"  Sub2 (X1<=0 & X3>0, eff=+0.20): {sub2.sum()} patients, mean CATE={true_cate[sub2].mean():.4f}")
print(f"  Sub3 (rest, eff=0): {sub3.sum()} patients, mean CATE={true_cate[sub3].mean():.4f}")
print(f"  True CATE range: [{true_cate.min():.4f}, {true_cate.max():.4f}]")

# Train/test split
idx = np.random.permutation(n)
n_train = int(n * 0.7)
train_i, test_i = idx[:n_train], idx[n_train:]

X_tr, X_te = X[train_i], X[test_i]
a_tr, a_te = treatment[train_i], treatment[test_i]
t_tr, t_te = t_final[train_i], t_final[test_i]
e_tr, e_te = event[train_i].astype(int), event[test_i].astype(int)
true_tr, true_te = true_cate[train_i], true_cate[test_i]

print(f"\n  Train: {n_train}, Test: {n - n_train}")

# ============================================================
# 2. COX T-LEARNER
# ============================================================
print("\n" + "-" * 65)
print("METHOD 1: Cox T-learner")

df_tr = pd.DataFrame(X_tr, columns=feature_names)
df_tr["time"] = t_tr; df_tr["event"] = e_tr; df_tr["treatment"] = a_tr
df_te = pd.DataFrame(X_te, columns=feature_names)
df_te["time"] = t_te; df_te["event"] = e_te; df_te["treatment"] = a_te

cph_t = CoxPHFitter()
cph_t.fit(df_tr[df_tr["treatment"] == 1].drop(columns="treatment"),
          duration_col="time", event_col="event")
cph_c = CoxPHFitter()
cph_c.fit(df_tr[df_tr["treatment"] == 0].drop(columns="treatment"),
          duration_col="time", event_col="event")

sf_t = cph_t.predict_survival_function(df_te.drop(columns="treatment"))
sf_c = cph_c.predict_survival_function(df_te.drop(columns="treatment"))
tt = sf_t.index.values.astype(float)
tc = sf_c.index.values.astype(float)
s_t = np.array([np.interp(float(t_star), tt, sf_t[i].values.astype(float)) for i in df_te.index])
s_c = np.array([np.interp(float(t_star), tc, sf_c[i].values.astype(float)) for i in df_te.index])
cate_cox = s_t - s_c
print(f"  CATE range: [{cate_cox.min():.4f}, {cate_cox.max():.4f}]")

# ============================================================
# 3. NUISANCE FUNCTIONS
# ============================================================
print("\n" + "-" * 65)
print("NUISANCE functions (shared)")

rf_p = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_p.fit(X_tr, a_tr)
e_tr_p = rf_p.predict_proba(X_tr)[:, 1]
e_te_p = rf_p.predict_proba(X_te)[:, 1]

def fit_rsf(X_, t_, e_):
    y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))],
                 dtype=[("event", bool), ("time", float)])
    m = RandomSurvivalForest(n_estimators=200, max_depth=5,
                             random_state=42, min_samples_leaf=10)
    m.fit(X_, y)
    return m

def surv_at(m, X_, t_):
    sf = m.predict_survival_function(X_, return_array=True)
    return np.array([np.interp(float(t_), m.unique_times_.astype(float),
                               sf[i].astype(float)) for i in range(len(X_))])

rsf_t = fit_rsf(X_tr[a_tr == 1], t_tr[a_tr == 1], e_tr[a_tr == 1])
rsf_c = fit_rsf(X_tr[a_tr == 0], t_tr[a_tr == 0], e_tr[a_tr == 0])

s1_tr = surv_at(rsf_t, X_tr, t_star)
s0_tr = surv_at(rsf_c, X_tr, t_star)
s1_te = surv_at(rsf_t, X_te, t_star)
s0_te = surv_at(rsf_c, X_te, t_star)

ystar_tr, k_tr = compute_pseudo_ite_dr(
    t_tr, e_tr, a_tr, X_tr, t_star, e_tr_p, s0_tr, s1_tr)
ystar_te, k_te = compute_pseudo_ite_dr(
    t_te, e_te, a_te, X_te, t_star, e_te_p, s0_te, s1_te)

idx_k = k_tr & ~np.isnan(ystar_tr)
print(f"  Known outcomes (train): {idx_k.sum()}/{n_train}")

# ============================================================
# 4. CSF (RF on pseudo-ITE)
# ============================================================
print("\n" + "-" * 65)
print("METHOD 2: CSF (RF on pseudo-ITE)")

csf_rf = RandomForestRegressor(n_estimators=500, max_depth=5,
                               random_state=42, min_samples_leaf=10)
csf_rf.fit(X_tr[idx_k], ystar_tr[idx_k])
cate_csf = csf_rf.predict(X_te)

# ============================================================
# RULE EXTRACTION UTILITIES
# ============================================================
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

X_all = np.vstack([X_tr, X_te])
ntr = len(X_tr)

# ============================================================
# 5. BO & DING (GB + Lasso)
# ============================================================
print("\n" + "-" * 65)
print("METHOD 3: Bo & Ding (GB + Lasso)")

cate_bd = np.zeros(len(X_te))
sel_gb = []

gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X_tr[idx_k], ystar_tr[idx_k])
r_gb = rules_from_ens(gb, X_all, 100)
print(f"  GB rules: {len(r_gb)}")

if r_gb:
    Rgb_tr = np.column_stack([m[:ntr].astype(float) for m in r_gb])
    Rgb_te = np.column_stack([m[ntr:].astype(float) for m in r_gb])

    lcv = LassoCV(cv=5, random_state=42, max_iter=10000,
                  alphas=np.logspace(-3, 2, 100))
    lcv.fit(Rgb_tr[idx_k], ystar_tr[idx_k])

    mm = lcv.mse_path_.mean(axis=1)
    ms = lcv.mse_path_.std(axis=1) / np.sqrt(5)
    mi = np.argmin(mm)
    cand = np.where(mm <= mm[mi] + ms[mi])[0]
    a1 = lcv.alphas_[cand[-1]] if len(cand) > 0 else lcv.alphas_[mi]

    ls = Lasso(alpha=a1, max_iter=10000)
    ls.fit(Rgb_tr[idx_k], ystar_tr[idx_k])
    sel_gb = np.where(abs(ls.coef_) > 1e-6)[0]
    print(f"  Selected: {len(sel_gb)} rules (1se alpha={a1:.4f})")

    cate_bd = ls.intercept_ + Rgb_te @ ls.coef_

    if len(sel_gb) > 0:
        top = sel_gb[np.argsort(-np.abs(ls.coef_[sel_gb]))[:5]]
        print("  Top 5 rules:")
        for j in top:
            print(f"    [{ls.coef_[j]:+.4f}] support={r_gb[j].mean():.3f}")

# ============================================================
# 6. HYBRID (CSF rules + Lasso)
# ============================================================
print("\n" + "-" * 65)
print("METHOD 4: Hybrid (CSF rules + Lasso)")

cate_hybrid = np.zeros(len(X_te))
sel_rf = []

rf_d = RandomForestRegressor(n_estimators=200, max_depth=10,
                             random_state=42, min_samples_leaf=5)
rf_d.fit(X_tr[idx_k], ystar_tr[idx_k])
r_rf = rules_from_ens(rf_d, X_all, 200)
print(f"  RF rules: {len(r_rf)}")

if r_rf:
    Rrf_tr = np.column_stack([m[:ntr].astype(float) for m in r_rf])
    Rrf_te = np.column_stack([m[ntr:].astype(float) for m in r_rf])

    lcv2 = LassoCV(cv=5, random_state=42, max_iter=10000,
                   alphas=np.logspace(-3, 2, 100))
    lcv2.fit(Rrf_tr[idx_k], ystar_tr[idx_k])

    mm2 = lcv2.mse_path_.mean(axis=1)
    ms2 = lcv2.mse_path_.std(axis=1) / np.sqrt(5)
    mi2 = np.argmin(mm2)
    cand2 = np.where(mm2 <= mm2[mi2] + ms2[mi2])[0]
    a2 = lcv2.alphas_[cand2[-1]] if len(cand2) > 0 else lcv2.alphas_[mi2]

    ls2 = Lasso(alpha=a2, max_iter=10000)
    ls2.fit(Rrf_tr[idx_k], ystar_tr[idx_k])
    sel_rf = np.where(abs(ls2.coef_) > 1e-6)[0]
    print(f"  Selected: {len(sel_rf)} rules (1se alpha={a2:.4f})")

    cate_hybrid = ls2.intercept_ + Rrf_te @ ls2.coef_

    if len(sel_rf) > 0:
        top = sel_rf[np.argsort(-np.abs(ls2.coef_[sel_rf]))[:5]]
        print("  Top 5 rules:")
        for j in top:
            print(f"    [{ls2.coef_[j]:+.4f}] support={r_rf[j].mean():.3f}")

# ============================================================
# 7. CISCaRL
# ============================================================
print("\n" + "-" * 65)
print("METHOD 5: CISCaRL (Conformalized Stability Selection)")

cis_carl = CISCaRL(
    B=500,
    stability_threshold=0.7,
    alpha=0.10,
    csf_n_estimators=200,
    csf_max_depth=10,
    gbm_n_estimators=100,
    gbm_max_depth=3,
    rule_min_support=10,
    calib_split=0.3,
    max_rules=2000,
    max_rule_conditions=4,
    max_selected_rules=10,
    mode='posthoc',   # paper-recommended mode: explains the smoothed CSF CATE
    random_state=42,
)

cis_carl.fit(X_tr, ystar_tr, idx_k, feature_names=feature_names)
cis_carl.print_rule_list()

cate_cis, rids = cis_carl.predict(X_te, return_details=True)

# ============================================================
# 8. EVALUATION
# ============================================================
print("\n" + "=" * 65)
print("EVALUATION AGAINST GROUND TRUTH CATE")
print("=" * 65)

methods = {
    "Cox T-learner": (cate_cox, 0),
    "CSF (RF)":      (cate_csf, 0),
    "Bo & Ding":     (cate_bd, len(sel_gb)),
    "Hybrid":        (cate_hybrid, len(sel_rf)),
    "CISCaRL":       (cate_cis, len(cis_carl.selected_rules_)),
}

results = []
for name, (pred, n_rules) in methods.items():
    valid = ~np.isnan(pred)
    p, t = pred[valid], true_te[valid]

    bias = np.mean(p - t)
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t) ** 2))
    r2 = r2_score(t, p) if len(np.unique(t)) > 1 else float("nan")
    corr, _ = spearmanr(p, t) if (len(np.unique(t)) > 1 and
                                   len(np.unique(p)) > 1) else (0, 1)

    rec = (p > np.median(p)).astype(int)
    tr = (t > np.median(t)).astype(int)
    acc = np.mean(rec == tr)

    results.append({
        "Method": name,
        "Bias": bias,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": corr,
        "Rec. Acc": acc,
        "Rules": n_rules,
    })

rdf = pd.DataFrame(results)
print("\n" + "-" * 72)
print("COMPARISON TABLE")
print("-" * 72)
print(f"{'Method':20s} {'Bias':>8s} {'MAE':>8s} {'RMSE':>8s} {'R2':>7s} {'Spearman':>9s} {'Acc':>6s} {'Rules':>6s}")
print("-" * 72)
for _, r in rdf.iterrows():
    print(f"{r['Method']:20s} {r['Bias']:>+8.4f} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>7.3f} {r['Spearman']:>9.3f} {r['Rec. Acc']:>6.3f} {int(r['Rules']):>6d}")

print("\n" + "-" * 65)
print("CISCaRL Rule Assignment (test set):")
rule_counts = pd.Series(rids).value_counts().sort_index()
for rid, cnt in rule_counts.items():
    if rid == -1:
        print(f"  Default: {cnt} patients  -> CATE={cis_carl.default_cate_['mean']:.4f} [{cis_carl.default_cate_['ci_low']:.4f}, {cis_carl.default_cate_['ci_high']:.4f}]")
    else:
        r = cis_carl.selected_rules_[rid]
        print(f"  Rule {rid+1}: {cnt} patients  -> CATE={r['mean_cate']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

print(f"\nTest MAE ranking (lower is better):")
for _, r in rdf.sort_values("MAE").iterrows():
    marker = " <<<" if r['MAE'] == rdf['MAE'].min() else ""
    print(f"  {r['Method']:20s}: MAE={r['MAE']:.4f}, Rules={int(r['Rules'])}{marker}")

best = rdf.loc[rdf['MAE'].idxmin(), 'Method']
print(f"\nBest method: {best} (MAE={rdf['MAE'].min():.4f})")
print("=" * 65)
print("Done.")

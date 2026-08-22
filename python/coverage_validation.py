"""Empirical coverage of CISCaRL's conformal prediction intervals.

CISCaRL claims each rule's 90% conformal interval covers the TRUE rule-level
CATE with probability >= 0.90 (finite-sample, distribution-free). This is the
one thing SCRE does not provide. Here we verify it empirically on synthetic
data with known CATE, under both effect regimes.

Coverage is evaluated two ways:
  (a) rule-level: does the interval contain the mean true CATE of the rule's
      calibration/test members?
  (b) the stated guarantee is for a NEW patient in the subgroup, so we also
      report per-patient coverage of the rule CATE against each member's true
      CATE (marginal, not conditional -- expected to be lower).
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\sanje\OneDrive\Desktop\Amrita\Sem 5\ML\project_ML\bio-ml-project-ideas\python')
import numpy as np
from scipy.stats import gumbel_r
from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest
from utils import compute_pseudo_ite_dr
from cis_carl import CISCaRL

def gen(n=3000, p=10, eff_a=2.0, eff_b=1.0, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, p)
    sub1 = (X[:,0]>0) & (X[:,1]>0); sub2 = (X[:,0]<=0) & (X[:,2]>0)
    te = np.zeros(n); te[sub1]=eff_a; te[sub2]=eff_b
    trt = np.random.binomial(1, 0.5, n)
    base = 2.0 + 0.5*X[:,0]-0.3*X[:,1]+0.2*X[:,2]
    eps = np.random.gumbel(0,1,n)
    t_obs = np.where(trt==1, np.exp(base+te+eps), np.exp(base+eps))
    cen = np.random.exponential(scale=np.median(t_obs)*3, size=n)
    t = np.minimum(t_obs, cen); ev = (t_obs<=cen).astype(bool)
    ts = np.percentile(t_obs[ev], 50)
    def cate(x, te, t):
        b = 2.0 + 0.5*x[0]-0.3*x[1]+0.2*x[2]
        return (1-gumbel_r.cdf(np.log(t)-b-te)) - (1-gumbel_r.cdf(np.log(t)-b))
    true_cate = np.array([cate(X[i], te[i], ts) for i in range(n)])
    return X, trt, t, ev, true_cate, ts

def run(eff_a, eff_b, label, n_seeds=6):
    cover_rule = []
    cover_patient = []
    n_rules = []
    for seed in range(n_seeds):
        X, trt, t, ev, true_cate, ts = gen(eff_a=eff_a, eff_b=eff_b, seed=seed)
        idx = np.random.RandomState(seed).permutation(len(X)); n_tr = int(0.7*len(X))
        X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
        a_tr, a_te = trt[idx[:n_tr]], trt[idx[n_tr:]]
        t_tr, t_te = t[idx[:n_tr]], t[idx[n_tr:]]
        e_tr, e_te = ev[idx[:n_tr]].astype(int), ev[idx[n_tr:]].astype(int)
        true_te = true_cate[idx[n_tr:]]
        rf_p = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42).fit(X_tr, a_tr)
        e_tr_p = rf_p.predict_proba(X_tr)[:,1]
        def fit_rsf(X_, t_, e_):
            y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))], dtype=[('event',bool),('time',float)])
            return RandomSurvivalForest(n_estimators=120, max_depth=5, random_state=42, min_samples_leaf=10).fit(X_, y)
        def surv_at(m, X_, t_):
            sf = m.predict_survival_function(X_, return_array=True)
            return np.array([np.interp(float(t_), m.unique_times_.astype(float), sf[i]) for i in range(len(X_))])
        rsf_t = fit_rsf(X_tr[a_tr==1], t_tr[a_tr==1], e_tr[a_tr==1]); rsf_c = fit_rsf(X_tr[a_tr==0], t_tr[a_tr==0], e_tr[a_tr==0])
        s1, s0 = surv_at(rsf_t, X_tr, ts), surv_at(rsf_c, X_tr, ts)
        ystar, k = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, ts, e_tr_p, s0, s1)
        idx_k = k & ~np.isnan(ystar)
        m = CISCaRL(B=100, stability_threshold=0.7, alpha=0.10, csf_n_estimators=120, csf_max_depth=10,
                    gbm_n_estimators=60, gbm_max_depth=3, rule_min_support=10, calib_split=0.3,
                    max_rules=1200, max_rule_conditions=4, max_selected_rules=10, min_calib_support=20,
                    shrinkage=0.5, mode='posthoc', random_state=42)
        m.fit(X_tr, ystar, idx_k, feature_names=[f'X{i}' for i in range(10)])
        pred, rids = m.predict(X_te, return_details=True)
        v = ~np.isnan(pred)
        # rule-level: for each rule (and default), does CI contain mean true CATE of its members?
        ok_rule = 0; tot_rule = 0
        ok_pat = 0; tot_pat = 0
        rules = m.selected_rules_ + [{'mean_cate': m.default_cate_['mean'],
                                      'ci_low': m.default_cate_['ci_low'],
                                      'ci_high': m.default_cate_['ci_high']}]
        for rid in range(len(rules)):
            mask = (rids == rid) & v
            if mask.sum() < 5:
                continue
            t_true = true_te[mask]
            lo, hi = rules[rid]['ci_low'], rules[rid]['ci_high']
            # rule-level coverage: interval vs mean true CATE of members
            ok_rule += int(lo <= t_true.mean() <= hi); tot_rule += 1
            # patient-level: interval vs each member's true CATE
            ok_pat += int(((t_true >= lo) & (t_true <= hi)).sum()); tot_pat += mask.sum()
        cover_rule.append(ok_rule / max(tot_rule, 1))
        cover_patient.append(ok_pat / max(tot_pat, 1))
        n_rules.append(len(m.selected_rules_))
    print(f"  [{label}] rules={np.mean(n_rules):.1f}  "
          f"rule-level coverage={np.mean(cover_rule)*100:.1f}%  "
          f"patient-level coverage={np.mean(cover_patient)*100:.1f}%  "
          f"(target: >=90% rule-level; patient-level is marginal, expected lower)")

print("=== EMPIRICAL COVERAGE of CISCaRL 90% conformal intervals ===")
print("(finite-sample guarantee is for the rule's CATE; patient-level shown for context)")
run(2.0, 1.0, 'rescaled')
run(0.5, 0.15, 'original (harder)')
"""
Multi-dataset benchmark comparing CISCaRL with all methods on real-world datasets.
Datasets: PBC, SUPPORT, GBSG (from the papers)

Usage: python run_benchmark.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, gumbel_r
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV
from sksurv.ensemble import RandomSurvivalForest
from utils import compute_pseudo_ite_dr
from cis_carl import CISCaRL

np.random.seed(42)


# ============================================================================
# Data Loading
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
    df['treatment'] = np.random.binomial(1, 0.5, len(df))  # RCT simulation
    df['event'] = df['death'].astype(int)
    df['time_days'] = df['d.time']
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
    return df, use_cols


# ============================================================================
# Synthetic semi-simulation (to get ground truth CATE)
# ============================================================================

def make_semisynthetic(df, covariates):
    """Create semi-synthetic survival with known CATE."""
    n = len(df)
    X = df[covariates].values.astype(float)
    treatment = df['treatment'].values

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_z = (X - X_mean) / X_std

    # Try to find age, bili, albumin indices
    idx_map = {c: i for i, c in enumerate(covariates)}
    age_i = idx_map.get('age', 0)
    bili_i = idx_map.get('bili', min(1, len(covariates)-1))
    alb_i = idx_map.get('albumin', idx_map.get('alb', min(2, len(covariates)-1)))

    sub1 = (X_z[:, age_i] > 0) & (X_z[:, bili_i] > 0)
    sub2 = (X_z[:, age_i] <= 0) & (X_z[:, alb_i] > 0)
    true_eff = np.zeros(n)
    true_eff[sub1] = 0.50
    true_eff[sub2] = 0.15

    eps = np.random.gumbel(0, 1, n)
    base_log = 6.5 + 0.3 * X_z[:, age_i] - 0.2 * X_z[:, bili_i] + 0.15 * X_z[:, alb_i]
    t_control = np.exp(base_log + eps)
    t_treated = np.exp(base_log + true_eff + eps)
    t_obs = np.where(treatment == 1, t_treated, t_control)
    censor = np.random.exponential(scale=np.median(t_obs) * 2.5, size=n)
    t_final = np.minimum(t_obs, censor)
    event = (t_obs <= censor).astype(bool)
    t_star = np.percentile(t_obs[event], 50)

    def compute_true(x_z_i, te, t):
        bl = 6.5 + 0.3 * x_z_i[age_i] - 0.2 * x_z_i[bili_i] + 0.15 * x_z_i[alb_i]
        s0 = 1 - gumbel_r.cdf(np.log(t) - bl, 0, 1)
        s1 = 1 - gumbel_r.cdf(np.log(t) - bl - te, 0, 1)
        return s1 - s0

    true_cate = np.array([compute_true(X_z[i], true_eff[i], t_star) for i in range(n)])

    df_out = df.copy()
    df_out['time_days'] = t_final
    df_out['event'] = event.astype(int)
    df_out['true_cate'] = true_cate

    info = {
        'n': n, 'p': len(covariates),
        'sub1': sub1.sum(), 'sub2': sub2.sum(),
        'eff_sub1': true_cate[sub1].mean(), 'eff_sub2': true_cate[sub2].mean(),
        'events': event.sum(), 't_star': t_star,
    }
    return df_out, covariates, info


# ============================================================================
# Single-run pipeline for one dataset
# ============================================================================

def run_one_dataset(name, df, covariates, n_runs=1):
    """Run all methods on a dataset and return results."""
    all_rows = []
    n = len(df)

    for run in range(n_runs):
        # Train/test split
        idx = np.random.permutation(n)
        n_train = int(n * 0.7)
        train_i, test_i = idx[:n_train], idx[n_train:]

        X_tr = df[covariates].values[train_i].astype(float)
        X_te = df[covariates].values[test_i].astype(float)
        a_tr = df['treatment'].values[train_i]
        a_te = df['treatment'].values[test_i]
        t_tr = df['time_days'].values[train_i]
        t_te = df['time_days'].values[test_i]
        e_tr = df['event'].values[train_i].astype(int)
        e_te = df['event'].values[test_i].astype(int)
        true_tr = df['true_cate'].values[train_i]
        true_te = df['true_cate'].values[test_i]

        t_star = np.percentile(t_tr[e_tr == 1], 50) if e_tr.sum() > 0 else np.median(t_tr)
        t_star = max(t_star, 1)

        # --- Cox T-learner ---
        df_tr_pd = pd.DataFrame(X_tr, columns=covariates)
        df_tr_pd['time'] = t_tr; df_tr_pd['event'] = e_tr; df_tr_pd['treatment'] = a_tr
        df_te_pd = pd.DataFrame(X_te, columns=covariates)
        df_te_pd['time'] = t_te; df_te_pd['event'] = e_te; df_te_pd['treatment'] = a_te

        cate_cox = np.zeros(len(X_te))
        try:
            cph_t = CoxPHFitter()
            cph_t.fit(df_tr_pd[df_tr_pd['treatment'] == 1].drop(columns='treatment'),
                      duration_col='time', event_col='event')
            cph_c = CoxPHFitter()
            cph_c.fit(df_tr_pd[df_tr_pd['treatment'] == 0].drop(columns='treatment'),
                      duration_col='time', event_col='event')
            sf_t = cph_t.predict_survival_function(df_te_pd.drop(columns='treatment'))
            sf_c = cph_c.predict_survival_function(df_te_pd.drop(columns='treatment'))
            tt = sf_t.index.values.astype(float)
            tc = sf_c.index.values.astype(float)
            s_t = np.array([np.interp(float(t_star), tt, sf_t[i].values.astype(float)) for i in df_te_pd.index])
            s_c = np.array([np.interp(float(t_star), tc, sf_c[i].values.astype(float)) for i in df_te_pd.index])
            cate_cox = s_t - s_c
        except Exception as ex:
            print(f"    [Cox] failed: {ex}")

        # --- Nuisance functions ---
        rf_p = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        rf_p.fit(X_tr, a_tr)
        e_tr_p = rf_p.predict_proba(X_tr)[:, 1]
        e_te_p = rf_p.predict_proba(X_te)[:, 1]

        def fit_rsf(X_, t_, e_):
            y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))],
                         dtype=[("event", bool), ("time", float)])
            m = RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=10)
            m.fit(X_, y)
            return m

        def surv_at(m, X_, t_):
            sf = m.predict_survival_function(X_, return_array=True)
            return np.array([np.interp(float(t_), m.unique_times_.astype(float), sf[i].astype(float)) for i in range(len(X_))])

        rsf_t = fit_rsf(X_tr[a_tr == 1], t_tr[a_tr == 1], e_tr[a_tr == 1])
        rsf_c = fit_rsf(X_tr[a_tr == 0], t_tr[a_tr == 0], e_tr[a_tr == 0])
        s1_tr = surv_at(rsf_t, X_tr, t_star)
        s0_tr = surv_at(rsf_c, X_tr, t_star)
        s1_te = surv_at(rsf_t, X_te, t_star)
        s0_te = surv_at(rsf_c, X_te, t_star)

        ystar_tr, k_tr = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, t_star, e_tr_p, s0_tr, s1_tr)
        ystar_te, k_te = compute_pseudo_ite_dr(t_te, e_te, a_te, X_te, t_star, e_te_p, s0_te, s1_te)
        idx_k = k_tr & ~np.isnan(ystar_tr)

        if idx_k.sum() < 20:
            print(f"    [SKIP] Too few known outcomes: {idx_k.sum()}")
            continue

        # --- CSF (RF on pseudo-ITE) ---
        csf_rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42, min_samples_leaf=10)
        csf_rf.fit(X_tr[idx_k], ystar_tr[idx_k])
        cate_csf = csf_rf.predict(X_te)

        # --- Rule utilities ---
        def extract_rules(tree_, X_):
            rules = []
            def rec(node_id, masks):
                if tree_.feature[node_id] == -2:
                    if masks[-1].sum() > 5:
                        rules.append(masks[-1].copy())
                    return
                f, thr = tree_.feature[node_id], tree_.threshold[node_id]
                l = masks[-1] & (X_[:, f] <= thr)
                rec(tree_.children_left[node_id], masks + [l])
                r = masks[-1] & (X_[:, f] > thr)
                rec(tree_.children_right[node_id], masks + [r])
            rec(0, [np.ones(len(X_), dtype=bool)])
            return rules

        def rules_from_ens(ens, X_, mt=200):
            all_r = []
            for i in range(min(len(ens.estimators_), mt)):
                e = ens.estimators_[i]
                t = e[0].tree_ if hasattr(e, '__len__') else e.tree_
                all_r.extend(extract_rules(t, X_))
            return all_r

        X_all = np.vstack([X_tr, X_te])
        ntr = len(X_tr)

        # --- Bo & Ding ---
        cate_bd = np.zeros(len(X_te))
        sel_gb = []
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gb.fit(X_tr[idx_k], ystar_tr[idx_k])
        r_gb = rules_from_ens(gb, X_all, 100)
        if r_gb:
            Rgb_tr = np.column_stack([m[:ntr].astype(float) for m in r_gb])
            Rgb_te = np.column_stack([m[ntr:].astype(float) for m in r_gb])
            lcv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 2, 100))
            lcv.fit(Rgb_tr[idx_k], ystar_tr[idx_k])
            mm = lcv.mse_path_.mean(axis=1); ms = lcv.mse_path_.std(axis=1) / np.sqrt(5)
            mi = np.argmin(mm); cand = np.where(mm <= mm[mi] + ms[mi])[0]
            a1 = lcv.alphas_[cand[-1]] if len(cand) > 0 else lcv.alphas_[mi]
            ls = Lasso(alpha=a1, max_iter=10000)
            ls.fit(Rgb_tr[idx_k], ystar_tr[idx_k])
            sel_gb = np.where(abs(ls.coef_) > 1e-6)[0]
            cate_bd = ls.intercept_ + Rgb_te @ ls.coef_

        # --- Hybrid ---
        cate_hybrid = np.zeros(len(X_te))
        sel_rf = []
        rf_d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, min_samples_leaf=5)
        rf_d.fit(X_tr[idx_k], ystar_tr[idx_k])
        r_rf = rules_from_ens(rf_d, X_all, 200)
        if r_rf:
            Rrf_tr = np.column_stack([m[:ntr].astype(float) for m in r_rf])
            Rrf_te = np.column_stack([m[ntr:].astype(float) for m in r_rf])
            lcv2 = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 2, 100))
            lcv2.fit(Rrf_tr[idx_k], ystar_tr[idx_k])
            mm2 = lcv2.mse_path_.mean(axis=1); ms2 = lcv2.mse_path_.std(axis=1) / np.sqrt(5)
            mi2 = np.argmin(mm2); cand2 = np.where(mm2 <= mm2[mi2] + ms2[mi2])[0]
            a2 = lcv2.alphas_[cand2[-1]] if len(cand2) > 0 else lcv2.alphas_[mi2]
            ls2 = Lasso(alpha=a2, max_iter=10000)
            ls2.fit(Rrf_tr[idx_k], ystar_tr[idx_k])
            sel_rf = np.where(abs(ls2.coef_) > 1e-6)[0]
            cate_hybrid = ls2.intercept_ + Rrf_te @ ls2.coef_

        # --- CISCaRL ---
        cis = CISCaRL(B=200, stability_threshold=0.7, alpha=0.10,
                      csf_n_estimators=200, csf_max_depth=10,
                      gbm_n_estimators=100, gbm_max_depth=3,
                      rule_min_support=10, calib_split=0.3,
                      max_rules=2000, max_rule_conditions=4,
                      max_selected_rules=10, mode='posthoc', random_state=42)
        cis.fit(X_tr, ystar_tr, idx_k, feature_names=covariates)
        cate_cis = cis.predict(X_te)

        # --- Evaluate ---
        methods = {
            'Cox T-learner': (cate_cox, 0),
            'CSF (RF)': (cate_csf, 0),
            'Bo & Ding': (cate_bd, len(sel_gb)),
            'Hybrid': (cate_hybrid, len(sel_rf)),
            'CISCaRL': (cate_cis, len(cis.selected_rules_)),
        }

        for mname, (pred, n_rules) in methods.items():
            valid = ~np.isnan(pred)
            p, t = pred[valid], true_te[valid]
            if len(p) < 10:
                continue
            bias = np.mean(p - t)
            mae = np.mean(np.abs(p - t))
            rmse = np.sqrt(np.mean((p - t) ** 2))
            corr, _ = spearmanr(p, t) if len(np.unique(t)) > 1 and len(np.unique(p)) > 1 else (0, 1)
            rec = (p > np.median(p)).astype(int)
            tr = (t > np.median(t)).astype(int)
            acc = np.mean(rec == tr)

            all_rows.append({
                'Dataset': name, 'Method': mname,
                'Bias': bias, 'MAE': mae, 'RMSE': rmse,
                'Spearman': corr, 'Rec_Acc': acc, 'Rules': n_rules,
            })

    return pd.DataFrame(all_rows)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    datasets = {
        'PBC': load_pbc,
        'SUPPORT': load_support,
        'GBSG': load_gbsg,
    }

    all_results = []

    for dname, loader in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dname}")
        print(f"{'='*70}")

        df_raw, covariates = loader()
        print(f"  Raw: {len(df_raw)} patients, {len(covariates)} covariates")

        df_sim, covariates, info = make_semisynthetic(df_raw, covariates)
        print(f"  Simulated: {info['n']} patients, {info['events']} events")
        print(f"  t* = {info['t_star']:.1f}")
        print(f"  Sub1 (effect={info['eff_sub1']:.4f}): {info['sub1']} patients")
        print(f"  Sub2 (effect={info['eff_sub2']:.4f}): {info['sub2']} patients")

        results = run_one_dataset(dname, df_sim, covariates, n_runs=1)
        all_results.append(results)

        if len(results) > 0:
            print(f"\n  Results:")
            for _, r in results.sort_values('MAE').iterrows():
                marker = ' <<<' if r['MAE'] == results['MAE'].min() else ''
                print(f"    {r['Method']:20s}: MAE={r['MAE']:.4f}, Rules={int(r['Rules'])}, Spearman={r['Spearman']:.3f}{marker}")

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\n\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        pivot = combined.pivot_table(
            values=['MAE', 'Rules', 'Spearman'],
            index='Method', columns='Dataset', aggfunc='first'
        )
        print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))

        print(f"\n  MAE rank across datasets (lower is better):")
        for dname in ['PBC', 'SUPPORT', 'GBSG']:
            if dname in combined['Dataset'].values:
                dres = combined[combined['Dataset'] == dname].sort_values('MAE')
                print(f"    {dname}:")
                for _, r in dres.iterrows():
                    print(f"      {r['Method']:20s}: MAE={r['MAE']:.4f}, Rules={int(r['Rules'])}")

    print(f"\n{'='*70}")
    print("Done.")

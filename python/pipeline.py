"""
Shared modelling pipeline for CISCaRL experiments and the live webapp.

SINGLE code path for method fitting + evaluation, mirroring
experiments.run_single() exactly, so that offline benchmark results and the
live demo are identical for the same dataset / DGP / seed / regime.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score, r2_score)
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

from utils import compute_pseudo_ite_dr
from cis_carl import CISCaRL
from scre import SurvivalCausalRuleEnsemble

METHODS = ['Cox T-learner', 'CSF (RF)', 'Bo & Ding', 'Hybrid', 'CRE', 'SCRE',
           'CISCaRL (direct)', 'CISCaRL (auto)', 'CISCaRL (posthoc)']
CIS_MODES = [('direct', 'cis_direct'), ('auto', 'cis_auto'),
             ('posthoc', 'cis_posthoc')]


# ---------------------------------------------------------------------------
# rule extraction (conditions + masks) mirroring run_single's `extract_rules`
# ---------------------------------------------------------------------------
def _extract_rules_cond(tree_, X, min_support=5):
    rules = []
    def rec(nid, conds, masks):
        if tree_.feature[nid] == -2:
            if masks[-1].sum() > min_support:      # run_single uses `> 5`
                rules.append((list(conds), masks[-1].copy()))
            return
        f, thr = tree_.feature[nid], tree_.threshold[nid]
        rec(tree_.children_left[nid], conds + [(f, '<=', thr)],
            masks + [masks[-1] & (X[:, f] <= thr)])
        rec(tree_.children_right[nid], conds + [(f, '>', thr)],
            masks + [masks[-1] & (X[:, f] > thr)])
    rec(0, [], [np.ones(len(X), dtype=bool)])
    return rules


def _rules_cond_ens(ens, X, mt=200, min_support=5):
    out = []
    for i in range(min(len(ens.estimators_), mt)):
        e = ens.estimators_[i]
        t = e[0].tree_ if hasattr(e, '__len__') else e.tree_
        out.extend(_extract_rules_cond(t, X, min_support))
    return out


def _1se(alphas, mm, ms):
    mi = int(np.argmin(mm))
    cand = np.where(mm <= mm[mi] + ms[mi])[0]
    return alphas[cand[-1]] if len(cand) > 0 else alphas[mi]


# ---------------------------------------------------------------------------
# data + split
# ---------------------------------------------------------------------------
def make_data(dataset='ACTG175', dgp='AFT-Gumbel', seed=1039, regime='rescaled'):
    """Generate the semi-synthetic data exactly as the offline benchmark does."""
    import experiments
    from data import load_actg175
    experiments.EFFECT_REGIME = regime
    loaders = {'PBC': experiments.load_pbc, 'SUPPORT': experiments.load_support,
               'GBSG': experiments.load_gbsg, 'ACTG175': load_actg175,
               'SYNTH-PBC': experiments.load_pbc}
    df, covs = loaders[dataset]()
    np.random.seed(seed)
    t, e, a, X, covs, true_cate, info = experiments.DGP_REGISTRY[dgp](df, covs, seed=seed)
    return t, e, a, X, covs, true_cate, info


def split_indices(n, frac=0.7):
    idx = np.random.permutation(n)
    ntr = int(n * frac)
    return idx[:ntr], idx[ntr:]


# ---------------------------------------------------------------------------
# fit all methods (mirrors run_single exactly)
# ---------------------------------------------------------------------------
def fit_all(X_tr, t_tr, e_tr, a_tr, covariates, X_te, quick=False):
    ntr, nte = len(X_tr), len(X_te)
    t_star = np.percentile(t_tr[e_tr == 1], 50) if e_tr.sum() > 0 else np.median(t_tr)
    t_star = max(t_star, 1)
    preds, models = {}, {'t_star': float(t_star), 'covariates': list(covariates)}

    # 1. Cox T-learner (per-arm: drop constant cols, escalating ridge)
    try:
        df_tr = pd.DataFrame(X_tr, columns=covariates)
        df_tr['time'], df_tr['event'], df_tr['treatment'] = t_tr, e_tr, a_tr
        df_te = pd.DataFrame(X_te, columns=covariates)

        def _cox_arm(mask):
            sub = df_tr[mask]
            k = [c for c in covariates if sub[c].std() > 1e-8]
            last = None
            for pen in (0.1, 0.5, 1.0, 2.0, 5.0):
                try:
                    mdl = CoxPHFitter(penalizer=pen).fit(
                        sub[k + ['time', 'event']], 'time', 'event')
                    return mdl, k
                except Exception as ex:
                    last = ex
            raise last if last else RuntimeError('cox arm failed')

        cph_t, keep_t = _cox_arm(df_tr['treatment'] == 1)
        cph_c, keep_c = _cox_arm(df_tr['treatment'] == 0)
        sf_t = cph_t.predict_survival_function(df_te[keep_t], times=[float(t_star)])
        sf_c = cph_c.predict_survival_function(df_te[keep_c], times=[float(t_star)])
        preds['Cox T-learner'] = sf_t.values[0] - sf_c.values[0]
        models['cox'] = (cph_t, cph_c)
        models['cox_keep'] = (keep_t, keep_c)
    except Exception:
        preds['Cox T-learner'] = np.zeros(nte); models['cox'] = None
        models['cox_keep'] = None

    # nuisance (propensity + RSF survival)
    rf_p = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_tr, a_tr)
    e_tr_p = rf_p.predict_proba(X_tr)[:, 1]

    def fit_rsf(X_, t_, e_):
        y = np.array([(bool(e_[i]), float(t_[i])) for i in range(len(t_))],
                     dtype=[("event", bool), ("time", float)])
        return RandomSurvivalForest(n_estimators=200, max_depth=5, random_state=42,
                                    min_samples_leaf=10).fit(X_, y)

    def surv_at(m, X_, ts):
        sf = m.predict_survival_function(X_, return_array=True)
        return np.array([np.interp(float(ts), m.unique_times_.astype(float), sf[i].astype(float))
                         for i in range(len(X_))])

    rsf_t = fit_rsf(X_tr[a_tr == 1], t_tr[a_tr == 1], e_tr[a_tr == 1])
    rsf_c = fit_rsf(X_tr[a_tr == 0], t_tr[a_tr == 0], e_tr[a_tr == 0])
    s1_tr, s0_tr = surv_at(rsf_t, X_tr, t_star), surv_at(rsf_c, X_tr, t_star)
    ystar_tr, k_tr = compute_pseudo_ite_dr(t_tr, e_tr, a_tr, X_tr, t_star, e_tr_p, s0_tr, s1_tr)
    idx_k = k_tr & ~np.isnan(ystar_tr)
    models.update({'ystar_tr': ystar_tr, 'idx_k': idx_k})

    if idx_k.sum() < 20:
        for m in METHODS:
            preds.setdefault(m, np.zeros(nte))
        return models, preds

    # 2. CSF
    csf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42,
                                min_samples_leaf=10).fit(X_tr[idx_k], ystar_tr[idx_k])
    preds['CSF (RF)'] = csf.predict(X_te); models['csf'] = csf

    X_all = np.vstack([X_tr, X_te])

    # 3. Bo & Ding
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_tr[idx_k], ystar_tr[idx_k])
    r_gb = _rules_cond_ens(gb, X_all, 100)
    if r_gb:
        Rgb_tr = np.column_stack([m[:ntr].astype(float) for _, m in r_gb])
        Rgb_te = np.column_stack([m[ntr:].astype(float) for _, m in r_gb])
        lcv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 2, 100)).fit(Rgb_tr[idx_k], ystar_tr[idx_k])
        a1 = _1se(lcv.alphas_, lcv.mse_path_.mean(axis=1), lcv.mse_path_.std(axis=1) / np.sqrt(5))
        ls = Lasso(alpha=a1, max_iter=10000).fit(Rgb_tr[idx_k], ystar_tr[idx_k])
        preds['Bo & Ding'] = ls.intercept_ + Rgb_te @ ls.coef_
        models['bd'] = {'rules': r_gb, 'coef': ls.coef_, 'intercept': float(ls.intercept_)}
    else:
        preds['Bo & Ding'] = np.zeros(nte); models['bd'] = None

    # 4/5. Hybrid + CRE share RF rules
    rf_d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42,
                                 min_samples_leaf=5).fit(X_tr[idx_k], ystar_tr[idx_k])
    r_rf = _rules_cond_ens(rf_d, X_all, 200)
    if r_rf:
        Rrf_tr = np.column_stack([m[:ntr].astype(float) for _, m in r_rf])
        Rrf_te = np.column_stack([m[ntr:].astype(float) for _, m in r_rf])
        lcv2 = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-3, 2, 100)).fit(Rrf_tr[idx_k], ystar_tr[idx_k])
        a2 = _1se(lcv2.alphas_, lcv2.mse_path_.mean(axis=1), lcv2.mse_path_.std(axis=1) / np.sqrt(5))
        ls2 = Lasso(alpha=a2, max_iter=10000).fit(Rrf_tr[idx_k], ystar_tr[idx_k])
        preds['Hybrid'] = ls2.intercept_ + Rrf_te @ ls2.coef_
        models['hybrid'] = {'rules': r_rf, 'coef': ls2.coef_, 'intercept': float(ls2.intercept_)}

        cre = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 1, 50)).fit(Rrf_tr[idx_k], ystar_tr[idx_k])
        a_c = _1se(cre.alphas_, cre.mse_path_.mean(axis=1), cre.mse_path_.std(axis=1) / np.sqrt(5))
        ef = ElasticNet(alpha=a_c, l1_ratio=0.5, max_iter=10000).fit(Rrf_tr[idx_k], ystar_tr[idx_k])
        preds['CRE'] = ef.intercept_ + Rrf_te @ ef.coef_
        models['cre'] = {'rules': r_rf, 'coef': ef.coef_, 'intercept': float(ef.intercept_)}
    else:
        preds['Hybrid'] = np.zeros(nte); models['hybrid'] = None
        preds['CRE'] = np.zeros(nte); models['cre'] = None

    # 6. SCRE
    try:
        scre = SurvivalCausalRuleEnsemble(alpha=0.5, pred_n_estimators=200,
                                          progn_n_estimators=200, rule_min_support=10,
                                          max_rules=2000, max_rule_conditions=4)
        scre.fit(X_tr, ystar_tr, idx_k, feature_names=covariates, time=t_tr,
                 event=e_tr.astype(bool), treatment=a_tr)
        preds['SCRE'] = scre.predict(X_te); models['scre'] = scre
    except Exception:
        preds['SCRE'] = np.zeros(nte); models['scre'] = None

    # 7/8/9. CISCaRL (direct/auto/posthoc)
    for mode, key in CIS_MODES:
        m = CISCaRL(B=100 if quick else 200, stability_threshold=0.7, alpha=0.10,
                    csf_n_estimators=200, csf_max_depth=10, gbm_n_estimators=100,
                    gbm_max_depth=3, rule_min_support=10, calib_split=0.3,
                    max_rules=2000, max_rule_conditions=4, max_selected_rules=10,
                    mode=mode, random_state=42)
        m.fit(X_tr, ystar_tr, idx_k, feature_names=covariates)
        preds[f'CISCaRL ({mode})'] = m.predict(X_te)
        models[key] = m

    return models, preds


def evaluate(preds, true_te, models):
    """Metric dict per method (identical formulas to run_single)."""
    evals = {}
    for name, pred in preds.items():
        valid = ~np.isnan(pred)
        p, t = pred[valid], true_te[valid]
        if len(p) < 10:
            evals[name] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Spearman': np.nan,
                           'Bias': np.nan, 'Acc': np.nan, 'Prec': np.nan, 'Rec': np.nan,
                           'F1': np.nan, 'AUC': np.nan}
            continue
        evals[name] = {
            'MAE': float(np.mean(np.abs(p - t))),
            'RMSE': float(np.sqrt(np.mean((p - t) ** 2))),
            'R2': float(r2_score(t, p)) if len(np.unique(t)) > 1 else float('nan'),
            'Spearman': float(spearmanr(p, t)[0]) if len(np.unique(t)) > 1 and len(np.unique(p)) > 1 else 0.0,
            'Bias': float(np.mean(p - t)),
        }
        yp = (p > 0).astype(int); yt = (t > 0).astype(int)
        if len(np.unique(yt)) > 1:
            evals[name].update({
                'Acc': float(accuracy_score(yt, yp)),
                'Prec': float(precision_score(yt, yp, zero_division=0)),
                'Rec': float(recall_score(yt, yp, zero_division=0)),
                'F1': float(f1_score(yt, yp, zero_division=0)),
                'AUC': float(roc_auc_score(yt, p)) if len(np.unique(p)) > 1 else float('nan'),
            })
        else:
            evals[name].update({'Acc': np.nan, 'Prec': np.nan, 'Rec': np.nan, 'F1': np.nan, 'AUC': np.nan})

    def _nrules(key):
        mdl = models.get(key)
        if mdl is None:
            return 0
        return int(np.sum(np.abs(mdl['coef']) > 1e-6))
    try: evals['Bo & Ding']['Rules'] = _nrules('bd')
    except Exception: pass
    try: evals['Hybrid']['Rules'] = _nrules('hybrid')
    except Exception: pass
    try: evals['CRE']['Rules'] = _nrules('cre')
    except Exception: pass
    try: evals['SCRE']['Rules'] = len(models['scre'].selected_rules_) if models.get('scre') is not None else 0
    except Exception: pass
    for mode, key in CIS_MODES:
        try: evals[f'CISCaRL ({mode})']['Rules'] = len(models[key].selected_rules_)
        except Exception: pass
    return evals

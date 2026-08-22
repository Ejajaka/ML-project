"""
Fair benchmark: every method runs on the SAME train/test splits, the SAME
datasets, and the SAME number of repetitions. Reports mean +/- std over reps,
plus pairwise paired-significance tests (paired because all methods share the
same splits within a rep).

Usage:
  python fair_benchmark.py [--quick] [--reps N] [--regime rescaled|original]

--regime selects the DGP effect-size regime (see experiments.EFFECT_REGIME).
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats

import experiments
from experiments import (load_pbc, load_support, load_gbsg, DGP_REGISTRY,
                         run_single, EFFECT_REGIME)
from data import load_actg175

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--regime', choices=['rescaled', 'original'], default='rescaled')
    args = ap.parse_args()
    REPS = args.reps
    experiments.EFFECT_REGIME = args.regime

    loaders = {
        'PBC': load_pbc,
        'SUPPORT': load_support,
        'GBSG': load_gbsg,
        'ACTG175': load_actg175,
    }

    all_rows = []
    t0 = time.time()
    print(f"regime={args.regime} reps={REPS} (effect regime from experiments.EFFECT_REGIME)", flush=True)

    for dname, loader in loaders.items():
        df_raw, covariates = loader()
        for dgp_name, dgp_fn in DGP_REGISTRY.items():
            for rep in range(REPS):
                seed = 1000 + rep * 7 + (list(loaders).index(dname) * 13 + list(DGP_REGISTRY).index(dgp_name))
                np.random.seed(seed)
                t_final, event, treatment, X, covs, true_cate, info = dgp_fn(
                    df_raw, covariates, seed=seed)
                evals = run_single(dname, dgp_name, t_final, event, treatment,
                                   X, covs, true_cate, args.quick)
                for mname, e in evals.items():
                    all_rows.append({
                        'Dataset': dname, 'DGP': dgp_name, 'Rep': rep,
                        'Method': mname, **e
                    })
                print(f"  [{dname}/{dgp_name}] rep {rep+1}/{REPS} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(all_rows)
    out = os.path.join(ROOT, 'results', f'fair_benchmark_{args.regime}.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved {out} ({len(df)} rows)")

    # ---- Summary table: mean +/- std of MAE over reps per method ----
    print(f"\n=== MAE by dataset (mean +/- std over {REPS} reps) ===")
    g = df.groupby(['Dataset', 'Method'])['MAE']
    summ = g.agg(['mean', 'std', 'count']).reset_index()
    summ['se'] = summ['std'] / np.sqrt(summ['count'])
    piv = summ.pivot_table(index='Dataset', columns='Method', values='mean')
    piv_sd = summ.pivot_table(index='Dataset', columns='Method', values='std')
    piv_se = summ.pivot_table(index='Dataset', columns='Method', values='se')
    for d in piv.index:
        row = []
        for m in piv.columns:
            if pd.notna(piv.loc[d, m]):
                row.append(f"{piv.loc[d, m]:.3f}±{piv_se.loc[d, m]:.3f}")
            else:
                row.append("—")
        print(f"  {d:10s}: " + "  ".join(row))
    print(f"  columns: {list(piv.columns)}")

    # ---- Overall mean MAE ----
    print("\n=== Overall MAE (mean over all reps/settings) ===")
    overall = df.groupby('Method')['MAE'].agg(['mean', 'std']).sort_values('mean')
    print(overall.round(4).to_string())

    # ---- Paired significance: is each method better than CISCaRL-posthoc? ----
    print(f"\n=== Paired t-test vs CISCaRL-posthoc (same splits, {REPS*len(loaders)*len(DGP_REGISTRY)} pairs) ===")
    base = 'CISCaRL (posthoc)'
    methods = sorted(df['Method'].unique())
    # pair by (Dataset, DGP, Rep)
    pair = df.pivot_table(index=['Dataset', 'DGP', 'Rep'], columns='Method', values='MAE')
    others = [m for m in methods if m != base]
    for m in others:
        if m not in pair.columns:
            continue
        a = pair[base].values
        b = pair[m].values
        v = ~(np.isnan(a) | np.isnan(b))
        if v.sum() < 3:
            print(f"  {m:22s}: too few pairs")
            continue
        t, p = stats.ttest_rel(b[v], a[v])  # does m beat base? negative t = m better
        d = (b[v] - a[v]).mean()
        print(f"  {m:22s}: delta={d:+.4f}  t={t:+.2f}  p={p:.4f}  "
              f"{'SIG better' if (p<0.05 and d<0) else 'SIG worse' if (p<0.05 and d>0) else 'ns'}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
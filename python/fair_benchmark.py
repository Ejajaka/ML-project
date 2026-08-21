"""
Fair benchmark: every method runs on the SAME train/test splits, the SAME
datasets, and the SAME number of repetitions. Reports mean +/- std over reps.

Usage: python fair_benchmark.py [--quick] [--reps N]
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from experiments import (load_pbc, load_support, load_gbsg, DGP_REGISTRY,
                         run_single)
from data import load_actg175

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--reps', type=int, default=3)
    args = ap.parse_args()
    REPS = args.reps

    loaders = {
        'PBC': load_pbc,
        'SUPPORT': load_support,
        'GBSG': load_gbsg,
        'ACTG175': load_actg175,
    }

    all_rows = []
    t0 = time.time()

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
    out = os.path.join(ROOT, 'results', 'fair_benchmark.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved {out} ({len(df)} rows)")

    # Aggregate: mean +/- std per method per dataset/DGP
    key = ['Dataset', 'DGP', 'Method']
    agg = df.groupby(key).agg(
        MAE=('MAE', 'mean'), MAE_sd=('MAE', 'std'),
        RMSE=('RMSE', 'mean'), R2=('R2', 'mean'),
        Spearman=('Spearman', 'mean'), Rules=('Rules', 'mean'),
    ).reset_index()

    # Summary tables
    for metric in ['MAE', 'RMSE', 'R2', 'Rules']:
        print(f"\n=== {metric} by dataset (mean over {REPS} reps) ===")
        piv = agg.pivot_table(index='Dataset', columns='Method',
                              values=metric, aggfunc='mean')
        piv = piv[sorted(piv.columns, key=lambda c: piv[c].mean()
                         if metric != 'Rules' else -piv[c].mean())]
        print(piv.round(3).to_string())

    print(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
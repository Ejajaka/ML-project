"""Gate 4f - does larger n make the full subgroup reproducibly discoverable?

Pre-specified endpoints (fixed before running):

  PRIMARY    half-sample stability of the full subgroup: fraction of 20 trees,
             each grown on a 50 % half-sample, whose best leaf has Jaccard >= 0.8
             with the true region. Same split fraction, same 20 resamples, same
             rule format, same threshold as Gate 4e.
  RECOVERY   Jaccard of the full-sample best leaf against the true region.
  SECONDARY  precision, recall, leaf count.

Three methods at every n, so the signal limit and the pipeline limit can be
separated:
    oracle  tree on the TRUE treatment effect   (representability ceiling)
    psi     tree on the DR pseudo-outcome       (signal limit)
    rf      CISCaRL's current RF candidate search (pipeline)

The claim this gate can support is only about the sample sizes tested.

Usage
-----
    python -m ciscarl2.gate4f --reps 5 --n 8000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import dgp
from .gate2 import TAU
from .gate4e import run_reps

from .paths import data_path

DATA = data_path()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n", type=int, nargs="+", default=[4000, 8000])
    args = ap.parse_args(argv)

    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    i_cd4, i_age = dgp.COVARIATES.index("cd40"), dgp.COVARIATES.index("age")
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, 730.0,
                                 gen_kwargs={"tau": TAU})

    rows = []
    for n in args.n:
        rows += run_reps(n, args.reps, pool, lam0, lam_c0, mu, sd, i_cd4, i_age)
        print(f"  done n={n}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "gate4f_large_n.csv")
    df.to_csv(out, index=False)

    g = df.groupby(["n", "method"]).agg(
        reps=("rep", "nunique"),
        leaves=("n_leaves_full", "mean"),
        jaccard_full=("best_jaccard_full", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        stability=("stability", "mean")).round(2)
    print("\n=== full-subgroup recoverability vs sample size ===")
    print(g.to_string())
    print("\nPRIMARY = stability (Jaccard >= 0.8 in >= 50 % of half-samples).")
    print("oracle = representability ceiling; psi = signal limit; rf = pipeline.")
    return df


if __name__ == "__main__":
    main()

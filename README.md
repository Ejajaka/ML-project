# CISCaRL: Conformalized Interpretable Survival Causal Rule Lists

**CISCaRL** is a framework for interpretable heterogeneous treatment effect (HTE)
estimation with right-censored survival outcomes. It produces a short,
human-readable rule list where every rule comes with a valid, distribution-free
confidence interval for its conditional average treatment effect (CATE), plus a
stability score indicating how reliably the rule is rediscovered across
bootstrap resamples.

This repo compares CISCaRL against 7 competing methods across 3 real-world
datasets (PBC, SUPPORT, GBSG) and 4 data-generating processes.

## Key Ideas

1. **Multi-source rule generation** — candidate rules are extracted from both
   a causal survival forest (CSF) fit on DR-learner pseudo-ITE and a gradient
   boosting model fit on CSF-predicted CATE, capturing complementary signals.
2. **Conformalized stability selection** — replaces Lasso's single-point 1se
   rule selection with bootstrap aggregation: each rule gets a stability score
   = frequency of selection across `B` bootstrap iterations. Rules above a
   threshold are kept, with FDR control in the spirit of Meinshausen & Buhlmann
   (2010).
3. **Weighted conformal prediction intervals** — each selected rule gets a 90%
   finite-sample-valid interval for its CATE, adjusted for censoring via IPCW
   (Tibshirani et al. 2019 style).
4. **Post-hoc mode** — in `mode='posthoc'` (recommended), rules explain the
   smoothed CSF CATE surface rather than the noisy pseudo-ITE, closing the
   accuracy gap to black-box models.

## Repository Layout

```
hte_project/
├── python/            # All Python source code
│   ├── cis_carl.py    #   CISCaRL core algorithm
│   ├── scre.py        #   Survival Causal Rule Ensemble (Wan et al. 2024)
│   ├── experiments.py #   Full paper experiments (8 methods x 4 DGPs)
│   ├── run_benchmark.py, cis_carl_demo.py, final_demo.py
│   ├── main*.py       #   Earlier HTE comparison pipelines
│   ├── utils.py       #   Pseudo-ITE learners (DR, R, DEA)
│   └── data.py        #   PBC data loader
├── R/                 # R scripts (grf CSF, rule extraction, GB survival)
├── data/              # Dataset files (PBC/SUPPORT/GBSG sources)
├── results/           # Output tables and figures
├── docs/              # Project documentation and weekly reports
├── PAPER.md           # Full paper draft (theory + methods + results)
└── requirements.txt
```

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the CISCaRL demo (synthetic data with known CATE)
cd python
python cis_carl_demo.py

# Run the 3-dataset benchmark
python run_benchmark.py

# Run full paper experiments (8 methods x 4 DGPs)
python experiments.py          # full run
python experiments.py --quick  # reduced bootstraps for a quick check
```

### R dependency

Some scripts (e.g. `final_demo.py`, the R pipeline) use the `grf` package for a
real causal survival forest. Install it with:

```r
install.packages("grf")
```

## Using CISCaRL

```python
from cis_carl import CISCaRL

model = CISCaRL(B=500, stability_threshold=0.7, alpha=0.10, mode='auto')
model.fit(X_train, pseudo_ite, known_mask, feature_names=covariates)
model.print_rule_list()          # interpretable rule list + 90% CIs
cate, rule_ids = model.predict(X_test, return_details=True)
```

Modes:
- `mode='direct'` — rules fitted to the raw pseudo-ITE.
- `mode='posthoc'` (default) — rules fitted to the smoothed CSF CATE surface.
- `mode='auto'` — picks `posthoc` when the known-outcome sample is small
  (n < 200), otherwise `direct`; falls back to `posthoc` if `direct` finds
  too few stable rules.

## Datasets

| Dataset | Source | Used in |
|---------|--------|---------|
| PBC | [Rdatasets survival](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv) | Fleming & Harrington (1991) |
| SUPPORT | [HBiostat](https://hbiostat.org/data/repo/support2csv.zip) | Connors et al. (1995) |
| GBSG | [Rdatasets survival](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/gbsg.csv) | Schumacher et al. (1994) |

## Citation

If you use this work, please cite the paper draft in `PAPER.md` (pending
submission).

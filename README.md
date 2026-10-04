# CISCaRL: Conformalized Interpretable Survival Causal Rule Lists

**Live demo:** https://ejajaka.github.io/ML-project/ — enter a patient, run every
method, and read the rules each one discovered.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Ejajaka/ML-project)

**CISCaRL** is a framework for interpretable heterogeneous treatment effect (HTE)
estimation with right-censored survival outcomes. It produces a short,
human-readable rule list where every rule comes with a valid, distribution-free
confidence interval for its conditional average treatment effect (CATE), plus a
stability score indicating how reliably the rule is rediscovered across
bootstrap resamples.

This repo compares CISCaRL against 8 competing methods across 4 real-world
datasets (PBC, SUPPORT, GBSG, ACTG175) and 4 data-generating processes.

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
│   ├── experiments.py #   Full paper experiments (9 methods x 4 DGPs x 4 datasets)
│   ├── run_benchmark.py, cis_carl_demo.py, final_demo.py
│   ├── main*.py       #   Earlier HTE comparison pipelines
│   ├── utils.py       #   Pseudo-ITE learners (DR, R, DEA)
│   └── data.py        #   PBC / ACTG175 data loaders
├── R/                 # R scripts (grf CSF, rule extraction, GB survival)
├── data/              # Dataset files (PBC/SUPPORT/GBSG/ACTG175 sources)
├── results/           # Output tables and figures
├── deliverables/      # Reports, literature, and lab assignments
│   ├── weekly-reports/          # Typst reports (weekly + final)
│   ├── literature/              # Paper PDFs and IEEE survey
│   └── qn3.typ, qn3.pdf         # Lab assignment (kNN)
├── PAPER.md           # Full paper draft (theory + methods + results)
└── requirements.txt
```

Weekly reports share one bibliography at `deliverables/weekly-reports/references.bib`.
Compile them from the repo root with the `--root` flag:

```bash
typst compile --root deliverables/weekly-reports \
  deliverables/weekly-reports/2026-08-16/main_report.typ \
  deliverables/weekly-reports/2026-08-16/main_report.pdf
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

# Run full paper experiments (9 methods x 4 datasets x 4 DGPs)
python experiments.py          # full run
python experiments.py --quick  # reduced bootstraps for a quick check

# Fair benchmark (all methods, identical splits, N reps; either effect regime)
python fair_benchmark.py --quick --reps 3 --regime original
python fair_benchmark.py --quick --reps 3 --regime rescaled
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
| ACTG175 | `speff2trial` R package (local copy in `data/`) | Hammer et al. (1996); used by the survival-RuleFit/SCRE papers for interpretable HTE |

## Fixes on the `cis-carl` branch

This branch fixes several issues found in a review of the original committed code:

- **Faithful SCRE**: `scre.py` previously fit an RF + ElasticNet on pseudo-ITE,
  which is *not* the Wan et al. (2024) method. It is now a shared-basis penalized
  Cox RuleFit: main-effect linear + rule terms and treatment-interaction terms
  are fitted jointly, and HTE is the survival-probability difference
  S(t*|x, z=1) - S(t*|x, z=0).
- **CISCaRL posthoc mode** is now included in the benchmark (was missing from
  `experiments.py`, despite being the recommended mode in the paper).
- **All 4 DGPs now run on all real datasets** (previously only DGP 1 ran on
  real data; the 4 DGP comparison was synthetic-only).
- **Corrected crossing DGP ground truth**: `dgp_non_ph_crossing` previously set
  `true_cate` as a function of *treatment assignment* and measured it as a time
  difference. It is now a closed-form survival-probability difference from a
  log-logistic shared-basis model, independent of the realized treatment.
- **Bootstrap multiplicity** in `cis_carl.py` stability selection: bootstrap
  resampling was collapsed to a boolean mask (turning with-replacement sampling
  into without-replacement). It now uses per-point multiplicities as
  `sample_weight`.
- **R scripts**: fixed crashes/bugs in `clean_run.R` (missing `ap$rules`),
  `run_boost.R` (unloaded `gbm`), `paper_final.R` (broken
  `variable_importance` indexing), `complete_pipeline.R` (wrong fallback length
  in GBSG section), `run_csf_hybrid.R` (off-by-one tree traversal + missing
  `horizon` argument), and `paper_refined.R` (rep-count header mismatch).
- **Reproducibility**: `results/paper_results_full.csv` is no longer gitignored,
  so committed results can be checked.

## Citation

If you use this work, please cite the paper draft in `PAPER.md` (pending
submission).

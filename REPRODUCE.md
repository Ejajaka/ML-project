# CISCaRL — Step-by-Step Reproduction Manual

**Project:** CISCaRL — Conformalized Interpretable Survival Causal Rule Lists
**Repository:** https://github.com/nithitsuki/bio-ml-project-ideas
**Branch to use:** `ciscarl-updated`
**Goal of this document:** anyone, on a fresh machine, can follow these steps top
to bottom and reproduce every number, table, and figure in the project.

> If you only have five minutes, read §1–§3, then run §5.1 (demo) and §5.6 (R
> test). That proves the install is correct and the pipeline runs.

---

## 0. What the project does (30-second version)

We estimate **which patient subgroups benefit from a treatment**, for
**right-censored survival data**, and output a short **interpretable rule list**
where each rule has a treatment effect, a **finite-sample-valid 90% conformal
interval**, a bootstrap **stability score**, and a treat/avoid recommendation.

Pipeline stages:
1. Build a doubly-robust pseudo-outcome for each patient (handles confounding + censoring).
2. Generate candidate rules from two sources (causal forest + gradient boosting).
3. Keep only rules that survive bootstrap **stability selection**.
4. Attach **IPCW-weighted conformal intervals** to each surviving rule.

---

## 1. Prerequisites

| Requirement | Version used | Notes |
|---|---|---|
| Python | 3.13 (3.10+ should work) | tested on Windows |
| R | 4.6.1 | only needed for the R bridge + regression test |
| Git | any | to clone |

Python packages (`requirements.txt`):

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
lifelines>=0.28
scikit-survival>=0.22
```

Additional packages used by specific scripts but not pinned in
`requirements.txt`: `pypdf` (only if you re-extract PDFs; not needed to run).

---

## 2. Get the code

```bash
git clone https://github.com/nithitsuki/bio-ml-project-ideas.git
cd bio-ml-project-ideas
git checkout ciscarl-updated
```

### Repository layout

```
bio-ml-project-ideas/
├── python/                     # all Python source
│   ├── cis_carl.py             # CISCaRL core (3 stages)
│   ├── scre.py                 # faithful SCRE baseline (shared-basis Cox RuleFit)
│   ├── experiments.py          # benchmark engine, 9 methods, 4 DGPs
│   ├── fair_benchmark.py       # fair multi-rep benchmark (both regimes)  <-- main
│   ├── cis_carl_demo.py        # synthetic demo with known CATE
│   ├── coverage_validation.py  # conformal-interval coverage check
│   ├── actg_showcase.py        # real-data (ACTG175) rule-list showcase
│   ├── utils.py                # DR/R/DEA pseudo-ITE + IPCW
│   ├── data.py                 # PBC / ACTG175 loaders
│   ├── run_benchmark.py, main*.py, final_demo.py   # earlier/auxiliary pipelines
├── R/                          # R scripts (grf CSF bridge) + regression test
│   ├── test_grf_traversal.R    # regression test for the grf off-by-one fix
│   └── (other working scripts)
├── data/                       # PBC/SUPPORT remote; ACTG175 local copy
│   └── ACTG175.csv
├── results/                    # committed result CSVs
│   ├── fair_benchmark_rescaled.csv
│   ├── fair_benchmark_original.csv
│   └── paper_results_full.csv
├── deliverables/               # literature PDFs, reports
├── PROJECT_STATUS.md           # reviewer-facing evidence summary
├── PAPER.md                    # the paper draft
└── requirements.txt
```

---

## 3. Install dependencies

### 3.1 Python

```bash
pip install -r requirements.txt
# optional (only for PDF inspection):
pip install pypdf
```

Verify:

```bash
python -c "import numpy, pandas, scipy, sklearn, lifelines, sksurv; print('ok')"
```

> **Interpreter gotcha (Windows).** If you have multiple Pythons installed,
> the bare `python` command may point to an install *without* the packages.
> Check with `python -c "import sksurv"`. If it fails, use `py -3.13`
> (or whichever interpreter has the packages) for every command below:
> `py -3.13 run_all.py`. To install into a specific interpreter:
> `py -3.13 -m pip install -r requirements.txt`.

### 3.2 R (needed for the CSF bridge and the regression test)

`grf` is required. It may not be in a writable system library, so install to a
user library (Windows example path shown; use your own on Linux/macOS):

```bash
Rscript -e '.libPaths(c("C:/Users/<you>/R/library", .libPaths())); \
  dir.create("C:/Users/<you>/R/library", recursive=TRUE, showWarnings=FALSE); \
  install.packages("grf", repos="https://cloud.r-project.org", \
                   lib="C:/Users/<you>/R/library")'
```

Note: the R scripts begin with `.libPaths(c('C:/Users/sanje/R/library',
.libPaths()))`. **Edit that path** to your own user library, or remove it if
`grf` is installed system-wide.

---

## 4. Data

| Dataset | Source | How it is loaded | Real or semi-synthetic? |
|---|---|---|---|
| PBC | `Rdatasets` GitHub URL | `data.load_pbc()` | real covariates, **simulated** outcome |
| SUPPORT | HBiostat URL | `experiments.load_support()` | real covariates, **simulated** outcome |
| GBSG | `Rdatasets` GitHub URL | `experiments.load_gbsg()` | real covariates, **simulated** outcome |
| **ACTG175** | local `data/ACTG175.csv` | `data.load_actg175()` | **fully real RCT** |

- The first three download at runtime (internet needed).
- ACTG175 is committed locally (`data/ACTG175.csv`) so it always works offline.
- In the **benchmark**, all four use semi-synthetic outcomes (simulated survival
  times with known true CATE, so we can compute accuracy). In
  `actg_showcase.py`, ACTG175 is used **fully raw** (real treatment + outcomes).

Quick check:

```bash
cd python
python data.py
# prints loaded PBC and ACTG175 shapes, treatment/event counts
```

---

## 5. Run everything (step by step)

All commands are run from the `python/` directory unless noted.

### 5.0 ONE-FILE DEMO (run this to show everything)

If you only want a single command that demonstrates the whole project:

```bash
py -3.13 run_all.py
```

This runs six steps and prints a PASS/FAIL summary:

1. Loads the datasets (ACTG175 is local; PBC skips gracefully if offline).
2. Synthetic example with known effect — shows CISCaRL gets low error **with**
   a short rule list (Bo & Ding needs ~300 rules; the black-box CSF has none).
3. Fits CISCaRL on the **real** ACTG175 trial and prints the interpretable
   rule list (CATEs, 90% intervals, stability, recommendations).
4. Measures empirical coverage of the 90% conformal intervals.
5. Prints the headline benchmark tables (read from committed `results/`).
6. Runs the R grf regression test.

Runtime ≈ 2–3 minutes. This is the recommended demonstration for a live review.

### 5.1 Demo — synthetic, known CATE (≈5–10 min)

```bash
python cis_carl_demo.py
```

- What it does: generates synthetic survival data with **known** CATE, runs 5
  methods, prints a comparison table (with R²), and prints CISCaRL's rule list.
- Look for: `COMPARISON TABLE` and `CISCaRL Rule Assignment`.
- Expected: CISCaRL MAE ~0.13–0.17 with ~7–9 rules; Cox ~0.07–0.14.

### 5.2 Full paper benchmark — one pass, 9 methods (≈45 min)

```bash
python experiments.py --quick
```

- Runs 4 datasets × 4 DGPs × 9 methods, writes
  `../results/paper_results_full.csv`.
- `--quick` reduces CISCaRL bootstraps (B=100 instead of 200). Drop `--quick`
  for the full run (≈1.5–2 h).

### 5.3 Fair benchmark — the main result (≈2 h per regime)

```bash
python fair_benchmark.py --quick --reps 3 --regime original
python fair_benchmark.py --quick --reps 3 --regime rescaled
```

- What it does: 3 reps × 4 datasets × 4 DGPs × 9 methods, **identical splits for
  all methods**, paired t-tests, Holm correction, effect sizes. Writes
  `../results/fair_benchmark_original.csv` and `..._rescaled.csv`.
- Flags:
  - `--reps N`  number of repetitions (default 3)
  - `--regime {original,rescaled}`  effect-size regime
  - `--quick`   fewer bootstraps (faster)
- Expected runtime: ~2 h per regime (16 settings × 3 reps).
- **This is the experiment that supports the paper's claims.** §6 explains the
  columns.

### 5.4 Conformal coverage validation (≈2–4 min)

```bash
python coverage_validation.py
```

- What it does: on known-CATE synthetic data, measures the fraction of rules
  whose 90% interval contains the true effect. 6 seeds, both regimes.
- Expected: rule-level coverage 100% (guarantee is ≥90%); intervals are
  conservative (~4.6× the CATE spread).

### 5.5 Real-data clinical showcase (≈1–2 min)

```bash
python actg_showcase.py
```

- What it does: fits CISCaRL on the **real** ACTG175 HIV trial (no simulated
  outcome), prints the interpretable rule list with intervals, stability scores,
  and recommendations.
- Note: uses `min_calib_support=5` (real subgroups are small); with the default
  20, all rules are dropped. This is a documented, real finding.

### 5.6 R regression test (≈10 s)

```bash
cd ../R
Rscript test_grf_traversal.R
```

- What it does: asserts the corrected **1-based** grf tree walk passes and the
  old **0-based** walk FAILS (locking the off-by-one fix).
- Expected final line: `TEST PASSED`.

### 5.7 (Optional) R CSF bridge — earlier pipeline

```bash
cd ../R
Rscript run_csf_hybrid.R     # needs hybrid_input.csv produced by final_demo.py
```

Note: `final_demo.py` writes `hybrid_input.csv` and calls this script via a
hard-coded R path; edit the path there if you run it.

---

## 6. Understanding the results files

### `results/fair_benchmark_<regime>.csv`

One row per `(Dataset, DGP, Rep, Method)` = 4 × 4 × 3 × 9 = **432 rows**.

| Column | Meaning |
|---|---|
| `Dataset`, `DGP`, `Rep` | which setting / repetition |
| `Method` | one of 9 methods |
| `MAE` | mean absolute error vs true CATE (primary) |
| `RMSE` | root mean squared error (= PEHE) |
| `R2` | coefficient of determination (often negative on real data) |
| `Spearman` | rank correlation (patient ranking) |
| `Acc`,`Prec`,`Rec`,`F1`,`AUC` | decision metrics (recommend-treat = predicted CATE>0) |
| `Rules` | number of rules emitted |
| `Bias` | mean(pred − true) |

### How to read the two regimes

- `rescaled`: larger effects (CATE std ~0.13–0.32). R² can be positive.
- `original`: the harder, original small effects (CATE std ~0.06–0.10). R² is
  negative for **every** method (effect below the DR pseudo-ITE noise floor);
  this is inherent CATE difficulty, not a method defect (an oracle scores +0.75–1.00).

### `results/paper_results_full.csv`

Output of `experiments.py`. One row per setting × method with the same style of
metrics, plus an `Ablation` row.

---

## 7. Statistical procedures (what the code actually computes)

1. **Splits.** Train/test 70/30. Within CISCaRL, known-outcome patients are
   further split 70/30 into rule-training and conformal **calibration**.
2. **Repetitions.** `fair_benchmark.py` repeats each setting `--reps` times with
   different seeds; **all methods share the same split** within a rep.
3. **Paired tests.** Paired t-test of per-setting MAE differences vs
   CISCaRL-posthoc (paired because splits are shared).
4. **Multiple comparisons.** Holm–Bonferroni across the 8 competitors *per
   metric per regime* (family = one metric's 8 comparisons).
5. **Effect sizes.** Cohen's d (`delta / sd(delta)`) and 95% CI on the mean
   difference.
6. **Aggregation.** Mean ± SE over the 48 settings.

---

## 8. Bugs that were fixed (and how to verify each)

These were real correctness bugs in the original repo. The repo now contains the
fixes; the notes below let a reader confirm them.

| # | Bug | Where | Fix | Verify |
|---|---|---|---|---|
| 1 | **Pseudo-ITE outcome mis-coded**: `Y=I(T>t*)` inverted for died-before-t* patients; censored-before-t* wrongly marked known | `utils.py` `_outcome_at_tstar` | `known = (time>t*) \| ((time<=t*) & (event==1))`; `Y = (time>t*)` | inspect `utils.py`; MAE improved 2–3× across methods |
| 2 | **SCRE baseline was fake** (RF+ElasticNet, not Wan et al.) | `scre.py` | rewrote as shared-basis penalized Cox RuleFit | inspect `scre.py` docstring + `fit()` |
| 3 | **Bootstrap multiplicity** collapsed with-replacement to withOUT | `cis_carl.py` `_stability_selection` | uses per-point `sample_weight` multiplicities | inspect the `bincount` weight line |
| 4 | **Crossing-DGP ground truth** was a function of treatment assignment & a time-difference | `experiments.py` `dgp_non_ph_crossing` | log-logistic closed-form `S1(t*)-S0(t*)` | inspect function; CATE no longer depends on assignment |
| 5 | **R scripts** crashed / failed silently | `R/*.R` | fixed `clean_run` (`ap$rules`), `run_boost` (`gbm`), `paper_final` (VI indexing), `complete_pipeline` (fallback length), `run_csf_hybrid` (see #6) | run `test_grf_traversal.R`; inspect diffs |
| 6 | **grf tree traversal off-by-one** (`nodes[[node_id+1]]` with 0-based root) | `R/run_csf_hybrid.R` | 1-based: root `nodes[[1]]`, child indices used directly | `Rscript R/test_grf_traversal.R` → PASSED |
| 7 | **Negative R² for every method** | DGP effect sizes | added `EFFECT_REGIME` (original vs rescaled) | `fair_benchmark.py --regime rescaled` shows positive R² on synthetic |

---

## 9. Quick verification checklist (for a reviewer)

```bash
# 1. install
pip install -r requirements.txt

# 2. data sanity
cd python && python data.py

# 3. pipeline runs
python cis_carl_demo.py                 # demo + R2
python coverage_validation.py           # coverage ~100%

# 4. real-data output
python actg_showcase.py                 # 6 interpretable rules

# 5. R fix locked
cd ../R && Rscript test_grf_traversal.R # TEST PASSED

# 6. main benchmark (long)
cd ../python
python fair_benchmark.py --quick --reps 3 --regime original
```

If steps 2–5 pass, the pipeline is correct. Step 6 reproduces the headline tables.

---

## 10. Environment used for the committed results

- OS: Windows 11
- Python: 3.13
- Key libs: numpy 2.2, pandas 2.3, scipy 1.16, scikit-learn 1.9,
  lifelines 0.30, scikit-survival 0.28
- R: 4.6.1, grf (from CRAN)
- Branch: `ciscarl-updated`
- Committed result files were produced with `--quick --reps 3`.

---

## 11. Known limitations / gotchas

- **Runtime**: the full fair benchmark is ~2 h per regime; use `--quick` first.
- **R library path**: scripts hard-code `C:/Users/sanje/R/library`; edit to yours.
- **ACTG175 min support**: the showcase uses `min_calib_support=5`; with 20 it
  returns zero rules (a real property of small real-data subgroups).
- **Remaining negative R² on real-covariate data is expected** (all methods,
  oracle positive).
- **Auxiliary scripts** (`main*.py`, `final_demo.py`, `run_benchmark.py`) are
  earlier/side pipelines, kept for history; the reproducible path is §5.

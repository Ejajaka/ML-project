# CISCaRL — support-constrained interpretable subgroup discovery for survival outcomes

An interpretable survival model that finds a **high-benefit subgroup** for a
730-day composite progression-free endpoint, and **abstains** when the evidence
is not strong enough. This repository contains the model, the notebook that
produces every reported number, the simulation evidence, and the IEEE paper.

The method is CISCaRL. Its frozen configuration is **C200**: the 200 is the
minimum number of discovery patients a candidate subgroup must contain, not a
count of trees or iterations.

---

## What it does

1. **Estimate** each patient's 730-day survival-probability contrast with a
   doubly robust (AIPW) estimator, adjusting for censoring with a conditional
   Cox model and a weight cap, cross-fitted over 5 folds.
2. **Discover** a depth-3 CART on those pseudo-outcomes. Each leaf is a readable
   rule of at most three conditions.
3. **Select** the leaf with the highest discovery mean **subject to at least 200
   discovery patients** — a pre-registered guard against the winner's curse.
4. **Validate** the selected rule on a disjoint partition and **commit only if**
   the one-sided 95 % lower confidence bound is above zero. Otherwise the
   verdict is `INSUFFICIENT` and nobody is treated.
5. **Score** the frozen model on a third, untouched partition.

Discovery / validation / assessment are always disjoint. The estimand is
τ(730 | x) = S₁(730 | x) − S₀(730 | x), on the survival-probability scale — not
a hazard ratio.

## Headline results

**Simulation** (ACTG175 covariates, simulated outcomes, 80 paired replications,
two DGPs):

| | CISCaRL (C200-S) | SCRE |
|---|---|---|
| MAE, one region | 0.0525 | 0.0514 (tied, p = 0.48) |
| MAE, two regions | 0.0680 | 0.0676 (tied, p = 0.76) |
| purity | **0.890 / 0.858** | 0.437 / 0.599 |
| benefit per treated patient | **+0.174 / +0.165** | +0.085 / +0.101 |
| coverage | 13.3 % / 11.7 % | 54.8 % / 56.3 % |
| null-benefit treatment | **0 %** | 0 % |

C200-S is the frozen model with empirical-Bayes shrinkage of leaf effects used
for prediction only; it changes no treatment decision.

**ACTG175** (arms 0 vs 1, n = 1054, 527/263/264 split): the overall contrast is
**+0.143 [0.025, 0.262]**, but the support floor cannot be met at that sample
size (largest leaf 174). The model falls back to a 62-patient rule, its
validation lower bound is −0.226, and it returns **INSUFFICIENT**. That is the
intended behaviour at the sample-size boundary, not a failure.

## Repository layout

```
ciscarl2/        model, estimators, and one module per experiment (gate*.py)
notebooks/       CISCaRL_final.ipynb — every reported number, executed
paper/           IEEEtran source, compiled PDF, and manuscript
results/         pre-registrations, verdicts, and per-replication tables
data/            ACTG175.csv (the exact file used by every run)
```

Useful entry points:

| file | what it is |
|---|---|
| `notebooks/CISCaRL_final.ipynb` | the whole pipeline, ACTG175 end to end, 48 cells |
| `paper/CISCaRL_IEEE_conference.pdf` | the 9-page IEEE conference paper |
| `results/FINAL_AUDIT.md` | consistency audit and known issues |
| `results/CLAIMS_v1.16.md` | exactly what may and may not be claimed |
| `results/gate30_verdict.md` | the knob sweep and the calibration result |

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/CISCaRL_final.ipynb
```

The notebook finds the data at `data/ACTG175.csv` automatically. Running it end
to end takes a few minutes; the last simulation cell re-runs one replication
live (about a minute).

To re-execute without opening the UI:

```bash
jupyter nbconvert --to notebook --execute notebooks/CISCaRL_final.ipynb --inplace
```

## The SCRE baseline

The comparator is the survival causal rule ensemble of Wan, Tanioka and
Shimokawa. It is **not vendored here**. Clone or install that package and point
the environment at it:

```bash
export SCRE_PATH=/path/to/directory/containing/scre      # Linux/macOS
setx SCRE_PATH "C:\path\to\directory\containing\scre"    # Windows
```

Without it, the notebook still runs: the SCRE cells report that the comparator
is unavailable and every CISCaRL number is unaffected.

## Rebuilding the paper

The paper is standard IEEEtran and compiles with any TeX installation:

```bash
cd paper
tectonic -X compile ieee_conference.tex
# or: pdflatex ieee_conference.tex
```

`paper/figs/` is generated from the CSV files in `results/`.

## Data

`data/ACTG175.csv` is the AIDS Clinical Trials Group Protocol 175 dataset
(arms 0 and 1, 1054 patients, 12 baseline covariates), distributed with the
`speff2trial` R package and used here for reproducibility. No new patient data
were collected; the simulations reuse this covariate matrix and generate
outcomes.

## Scope and honest limitations

- The model is **not more accurate than SCRE**. It ties after a post-hoc
  calibration and is significantly worse before it.
- It captures **less total benefit** (0.28–0.47 vs 0.82–0.95) because it treats
  far fewer patients. That is the design, not a defect that was missed.
- One leaf cannot recover two separated regions.
- Interval coverage is about **93 %**, not 95 %: the intervals condition on the
  selected rule.
- ACTG175 is an illustration. The true subgroup is unknown there, so accuracy
  metrics are computed only in simulation.

## License

Code: MIT (see `LICENSE`). The paper text and figures: CC BY 4.0. The dataset
retains the terms of its original distribution (`speff2trial`).

## Citation

```bibtex
@misc{ciscarl2026,
  title  = {CISCaRL: support-constrained interpretable subgroup discovery for survival outcomes},
  author = {George, Roshna},
  year   = {2026},
  note   = {Configuration C200},
  url    = {https://github.com/RoshnaGeorge/ciscarl}
}
```

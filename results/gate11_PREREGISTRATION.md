# Gate 11 pre-registration — sample size, subgroup purity, and rule-model MAE

Frozen 2026-11-01, before execution. Not edited after the run.

## Question

At n = 2000 no interpretable rule model beat the marginal ATE: Gate 7
(min_support sweep), Gate 8 (RMST), Gate 9 (plug-in targets) and Gate 10
(depth-1 CART) all failed. Gate 10 also quantified why: the best single split
had ~49 % purity and the leaf's effect error cancelled against the marginal
exactly at break-even.

Two distinct limitations could produce that, and this gate separates them in
principle even if it cannot fully identify them:

* **statistical** — finite samples make the effect surface noisy, and a larger n
  can improve the estimated boundary and the precision of leaf effects;
* **approximation** — a depth-2 tree may fail to represent the true conjunction
  if the plug-in target has graded level sets or the split criterion prefers a
  different partition.

Additional data can reduce the first. It does not automatically remove the
second.

**Prediction (a statement to test, not a claim):** increasing sample size is
expected to improve subgroup purity and may enable a depth-2 rule model to
outperform the marginal ATE. The experiment tests whether this occurs at the
specified sample sizes.

## Frozen specification

| item | value |
|---|---|
| sample sizes | n ∈ {4000, 8000, 12000} |
| replications | 40 fresh seeds per sample size |
| seeds | 120100–120139 (n=4000), 120200–120239 (n=8000), 120300–120339 (n=12000) |
| DGP | unchanged from Gates 4d–10: region = CD4 < median(340) AND age > median(34), effect +0.20 in S(730), 10 % covariate censoring |
| split | 70 / 30 stratified by A, `random_state = seed` |
| discovery target | cross-fitted Cox plug-in τ̂(X) = μ̂₁(X) − μ̂₀(X), TRAIN only |
| **primary model** | CART depth 2, min_samples_leaf 60, on τ̂ |
| descriptive model | CART depth 3, otherwise identical |
| rule effects | held-out AIPW φ, nuisances fitted on TRAIN only |
| default | TRAIN marginal ATE, one per replication and sample-size cell |
| comparator | same-replication TRAIN marginal ATE |
| evaluation | the same held-out patients for every method within a replication |

## Leakage audit

Identical to Gate 10 and unchanged: the split search, thresholds, leaf
conditions and default are fitted on TRAIN only; TEST outcomes, TEST covariates
and TEST φ never enter any model choice. The TEST fold is used only to score the
frozen model.

## Decision rule (frozen)

The depth-2 model is adopted at a sample size only if ALL of:

1. mean paired Δ = MAE(model) − MAE(constant) < 0;
2. the 95 % paired bootstrap CI (5000 resamples) lies entirely below 0;
3. at least 70 % of replications have Δ < 0.

Purity, capture and Jaccard are **reported, not used** to select or modify the
primary model. The depth-3 model is descriptive only and can never be adopted.

## Mandatory diagnostics per sample size and model

* high-leaf prevalence on TEST;
* high-leaf purity = P(true region | high leaf) on TEST;
* fraction of true-region patients captured = P(high leaf | true region);
* high-leaf AIPW effect and standard error;
* MAE against the true individual τ on TEST;
* paired MAE difference versus the TRAIN constant, with 95 % CI and the
  fraction of replications better;
* best-leaf Jaccard against the true region;
* split stability: 20 half-sample refits of the same CART on TRAIN, Jaccard of
  the refit high leaf against the full-data high leaf, evaluated on TEST.

## The break-even purity is DGP-specific, not universal

Under this DGP, with a regional effect of 0.195, a marginal ATE of ~0.05, 25 %
prevalence and mean-absolute-error loss, a leaf of purity p assigned its own
unbiased effect has

    MAE_leaf = 0.39 p (1 − p),      MAE_constant = 0.05 + 0.095 p

so break-even is at **p ≈ 0.50**. This is an *estimated* break-even for these
effect sizes, prevalence, loss function and estimator variance. It is reported
as a reference point, not as a requirement for rule-based treatment-effect
models in general. The exact threshold will differ under other DGPs, loss
functions and leaf-effect estimators.

## If it succeeds

Claim, scoped to the tested simulation:

"Under the specified DGP, an interpretable depth-2 treatment-effect rule model
improves held-out MAE over the marginal ATE at sample sizes where subgroup
recovery becomes sufficiently accurate."

It does not establish robustness to Cox misspecification, does not extend
beyond n = 12 000, and does not establish full recovery of the true region.

## If it fails at all three sample sizes

Do **not** conclude that rule-based subgroup analysis has no advantage at any
feasible sample size. The tested range ends at 12 000, the DGP is one specific
piecewise-constant effect with one censoring mechanism, and the representation
is one tree family. Record the failure with its measured purity/capture and stop.

## Secondary analyses

None before the primary result. If an arm is adopted, the MNAR δ-tilt and the
n-sweep extension follow in a separate pre-registered gate.

# Gate 10 pre-registration — conservative rule extraction on the plug-in target

Frozen 2026-11-01, before execution. This file is not edited after the run.

## Rationale

Gate 9 showed the plug-in target is a genuine discovery improvement (Jaccard
0.107 → 0.62, stability 0.57 → 0.99, correlation with true τ 0.08 → 0.56) but
does not improve MAE, because the RF partitions the entire covariate space and
assigns a noisy AIPW effect to the 75 % of patients whose true effect is zero.

On the Gate 9 seeds (pilot, 30 reps), with the same clean target:

| extractor | rules | MAE | Δ vs constant |
|---|---|---|---|
| smooth τ̂, no rules | — | 0.0593 | −0.0167 |
| CART depth 1 | 2 | 0.0692 | −0.0068 |
| CART depth 2 | 3.9 | 0.0779 | +0.0019 |
| CART depth 3 | 6.9 | 0.0832 | +0.0073 |
| RF + stability | 26.1 | 0.0807 | +0.0047 |

This gate confirms the depth-1 arm on independent seeds. The MAE loss is
asymmetric (a false positive costs ≈0.11, a false negative ≈0.05), which is why
a conservative partition is the candidate.

## Frozen specification

| item | value |
|---|---|
| sample size | n = 2000 |
| replications | 80 |
| DGP seeds | 110100–110179 (indices rep = 100..179); seeds 110000–110029 used in Gate 9 are excluded |
| discovery target | cross-fitted Cox plug-in τ̂(X) = μ̂₁(X) − μ̂₀(X) |
| primary extractor | `DecisionTreeRegressor(max_depth=1, min_samples_leaf=60, random_state=0)` |
| rule conditions | root-to-leaf paths of that tree, `min_support = 60` |
| rule-effect estimator | held-out AIPW φ (`phi_on_holdout`), nuisances fitted on TRAIN only, cap = min(Q0.99, 20) from an inner cross-fit of TRAIN |
| default prediction | training-only marginal ATE |
| comparator | same-replication constant, training-only marginal ATE |
| robustness arm | RSF plug-in target, identical extraction |
| primary metric | paired Δ = MAE(CISCaRL) − MAE(constant) on the same held-out patients |

## Leakage audit (frozen; nothing below uses held-out data)

| quantity | fitted on | applied to |
|---|---|---|
| Cox per-arm outcome models for τ̂ | TRAIN, 5-fold cross-fitted (out-of-fold predictions only) | TRAIN rows, to form the discovery target |
| CART split feature + threshold | (X_TRAIN, τ̂_TRAIN) only | TRAIN, to read off leaf conditions |
| leaf membership conditions | TRAIN tree structure | TEST, by evaluating the thresholds |
| marginal ATE (default + comparator) | TRAIN | TEST |
| AIPW φ for rule effects | TRAIN nuisances → TEST predictions | TEST |
| weight cap | inner cross-fit of TRAIN | TEST |

The TEST outcomes, TEST covariates, and TEST φ values never enter the split
search, the threshold, the leaf conditions, the default, or any model choice.

## Documentation correction (not a design change)

The frozen table says "default prediction = training-only constant". With a
depth-1 tree the two leaves partition the covariate space, so **every** test
patient matches exactly one leaf, and both leaves receive their own held-out
AIPW effect. The constant is the fallback only when no rule matches (it is
therefore almost never used here). The pilot MAE of 0.0692 was produced by
exactly this behaviour, so the specification and the pilot are consistent; the
wording above is what needs correcting.

Consequence for interpretation: the model is a **two-level step function**, not
"treatment for one group, marginal for everyone else". The low-τ leaf's AIPW
effect is itself a noisy near-marginal estimate. This is recorded now so the
result is not described as "one rule plus abstention" later.

## Decision rule (fixed before running)

The primary arm is adopted only if ALL of:

1. mean paired Δ < 0;
2. the 95 % paired bootstrap CI (5000 resamples) lies entirely below 0;
3. mean best-leaf Jaccard ≥ 0.107 (the frozen AIPW-ψ production baseline);
4. at least 70 % of replications have Δ < 0.

The RSF arm is assessed against the same four criteria. It cannot rescue a
failed primary arm; it only tests Cox-dependence.

## Descriptive arms (never eligible for adoption)

CART depth 2, CART depth 3, RF + stability on the same target, and the smooth
τ̂ predictor. Reported to show the shape of the loss curve on fresh seeds.

## Power caveat

sd(Δ) ≈ 0.020 from the pilot implies a half-width of ≈0.0043 at 80 reps. This is
a planning approximation only; the frozen seeds may give a different sd, and the
80-replication result determines the decision.

## Claim if adopted

"Under the tested DGP at n = 2000, an interpretable two-rule treatment-effect
model estimated with a cross-fitted plug-in target and held-out AIPW effects
achieves lower held-out MAE than the marginal ATE."

It does **not** establish full subgroup recovery (best-leaf Jaccard ≈0.34), does
**not** establish robustness to Cox misspecification, and says nothing about
n = 589.

## Secondary analyses (only after a primary adoption)

MNAR censoring δ-tilt on the winning rule set, and an n-sweep (589, 4000).

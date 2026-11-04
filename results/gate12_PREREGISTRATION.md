# Gate 12 pre-registration — depth-3 confirmation on fresh seeds

Frozen 2026-11-01, before execution. Not edited after the run.

## Question

Gate 11 adopted the **depth-2** rule model at n = 8000 and n = 12000. The
descriptive depth-3 arm was stronger on the same seeds:

| n | depth-3 Δ | reps better | purity | best Jaccard |
|---|---|---|---|---|
| 8000 | −0.0219 | 88 % | 0.93 | 0.49 |
| 12000 | −0.0247 | 100 % | 0.93 | 0.48 |

Depth-3 was not eligible for adoption there. This gate tests it on independent
seeds under the same four-part rule. It is a confirmation, not a promotion of
the same result.

## Frozen specification

| item | value |
|---|---|
| sample sizes | n ∈ {8000, 12000} |
| replications | 40 fresh seeds per cell |
| seeds | 130100–130139 (n=8000), 130200–130239 (n=12000); no overlap with 110xxx, 120xxx |
| DGP | unchanged: region = CD4 < median AND age > median, +0.20 in S(730), 10 % covariate censoring |
| split | 70 / 30 stratified by A, `random_state = seed` |
| discovery target | cross-fitted Cox plug-in τ̂, TRAIN only |
| **primary model** | CART depth 3, min_samples_leaf 60 |
| descriptive model | CART depth 2 on the same seeds (reproduction check) |
| rule effects | held-out AIPW φ |
| default / comparator | TRAIN marginal ATE |
| evaluation | same held-out patients within a replication |

## Decision rule (frozen, identical to Gate 11)

The depth-3 model is adopted at a sample size only if ALL of:

1. mean paired Δ < 0;
2. 95 % paired bootstrap CI (5000 resamples) entirely below 0;
3. best-leaf Jaccard ≥ 0.107;
4. at least 70 % of replications have Δ < 0.

No depth, threshold, support, pruning or adoption rule may be changed after
seeing the result.

## Mandatory diagnostics

High-leaf prevalence, purity, capture, leaf AIPW effect and SE, MAE, paired Δ
with CI, best-leaf Jaccard, split stability (20 half-sample refits).

## If it passes

The v1.5 extraction is depth 3 at n ≥ 8000, and the claim is the same as Gate
11's but with the stronger arm. If it fails at either sample size, depth 2
remains the adopted model and the Gate 11 depth-3 numbers are reported as
seed-specific.

## Scope

Unchanged from Gate 11: one DGP, correctly specified Cox nuisance, n ≤ 12 000,
partial region recovery. No claim of Cox-misspecification robustness or
generalisation beyond the tested range.

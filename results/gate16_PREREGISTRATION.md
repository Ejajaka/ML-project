# Gate 16 pre-registration — confirm the AIPW target at n = 8000 / 12000

Frozen 2026-11-01, before execution. Not edited after the run.

## Question

Gate 15's descriptive ψ-only arm beat the plug-in at every κ and passed all four
checks at κ=3 (Δ = −0.0248, 90 % of reps, +34.1 %). It was not the
pre-registered primary and used one DGP.

This gate asks whether that replicates on fresh seeds and whether it holds at
both adopted sample sizes, under both a correct nuisance and Gate 14's severe
violation. If it does, ψ becomes the v1.6 discovery target and the
outcome-model dependence in the claim is removed for discovery.

## Frozen specification

| item | value |
|---|---|
| sample sizes | n ∈ {8000, 12000} |
| nuisance states | κ = 0 (correct) and κ = 3 (Gate 14 severe violation) |
| replications | 40 fresh seeds per cell |
| seeds | (8000,0): 180100–180139; (12000,0): 180200–180239; (8000,3): 180300–180339; (12000,3): 180400–180439 |
| primary target | **AIPW pseudo-outcome ψ** |
| comparator | cross-fitted Cox plug-in τ̂, same exact replications |
| extractor | depth-3 CART, min_samples_leaf 60 |
| rule effects | held-out AIPW φ |
| default / comparator baseline | TRAIN marginal ATE |
| metric | paired held-out Δ MAE = MAE(rule model) − MAE(constant) |

## Decision rule (unchanged)

ψ is adopted at a cell only if ALL of:

1. mean paired Δ < 0;
2. the 95 % paired bootstrap CI lies entirely below 0;
3. best-leaf Jaccard ≥ 0.107;
4. at least 70 % of replications have Δ < 0.

The v1.6 model requires ψ to pass at **κ=0 for performance** and at **κ=3 for
robustness**, at n ≥ 8000:

* both pass → v1.6 adopts ψ; the Gate 14 dependence clause is removed for
  discovery and replaced by the double-robustness condition.
* κ=0 passes, κ=3 fails → v1.6 adopts ψ for performance only, and the Gate 14
  caveat remains.
* κ=0 fails → ψ is not adopted; the plug-in (v1.5) stays, and Gate 15's result
  is recorded as a seed-specific descriptive finding.

No depth, support, threshold, or extraction change is permitted in this gate.

## Secondary, pre-specified

* paired ψ vs plug-in MAE on the identical replications, per cell;
* the Gate 15 anomaly: ψ's gain at κ=3 exceeded its gain at κ=0. Report the
  marginal ATE estimate, censoring fraction, event fraction, and target sd by κ
  so the anomaly can be explained or explicitly left unexplained.

## Mandatory diagnostics

Target correlation with true τ on TRAIN; target sd; regional target contrast;
high-leaf prevalence, purity, capture; leaf AIPW effect and SE; best-leaf
Jaccard; split stability; paired Δ with CI; marginal ATE estimate versus truth.

## Scope

One DGP family, one extractor, two sample sizes. The censoring model is correct
in every cell, so this is **not** a joint-misspecification test; that is the
subject of the next gate if ψ is adopted.

# Gate 15 pre-registration — can a plugin + AIPW blend rescue a collapsed target?

Frozen 2026-11-01, before execution. Not edited after the run.

## Question

Gate 14 showed the v1.5 gain collapses when the outcome nuisance no longer
preserves the ordering of τ (κ=3: target correlation 0.09, MAE gain −4.1 %),
while the marginal ATE stays unbiased.

This gate tests the obvious mitigation: blend the plug-in target with the AIPW
pseudo-outcome, so that when the plug-in collapses the pseudo-outcome carries
the ordering.

## Scale correction (settled before freezing)

A raw convex combination is meaningless here: the plug-in sd is 0.078 under
κ=0 and **0.031** under κ=3, while ψ has sd ≈ 0.97. A raw `0.5*plugin + 0.5*ψ`
is ≈ 100 % ψ, not a 50/50 blend.

The blend is therefore defined on **TRAIN-standardised** targets:

    z_plugin = (τ̂_plugin − mean)/sd     (mean, sd computed on TRAIN only)
    z_psi    = (ψ − mean)/sd            (TRAIN only)
    blend_λ  = λ z_plugin + (1−λ) z_psi

λ = 0.5 is the primary. λ = 1 (plugin) and λ = 0 (ψ) are the endpoints and
serve as controls, so a blend that fails can be compared with both of its
ingredients on identical replications.

## Pre-run pilot (separate seeds 170000–170019, 10 reps)

Recorded before the confirmatory run so the result is not over-interpreted:

| κ | corr(plugin, τ) | corr(ψ, τ) | corr(raw λ=.5) | corr(std λ=.5) | corr(std λ=.75) |
|---|---|---|---|---|---|
| 0 | 0.655 | 0.083 | 0.135 | 0.508 | 0.637 |
| 3 | 0.126 | 0.082 | 0.086 | 0.146 | 0.145 |

ψ's correlation with the true τ is **0.08 whether or not the nuisance is
violated**. It is not a complementary signal; it is a noisy one. The pilot
therefore predicts that no convex blend restores ordering at κ=3. The
confirmatory run tests that prediction rather than assuming it.

## Frozen specification

| item | value |
|---|---|
| sample size | n = 8000 |
| replications | 40 fresh seeds per cell |
| seeds | κ=3: 170100–170139; κ=0: 170200–170239; κ=1: 170300–170339 |
| primary cell | **κ = 3, λ = 0.5** (standardised blend) |
| primary comparator | same-replication TRAIN marginal ATE |
| secondary comparator | the λ = 1 plug-in target on the identical seeds |
| descriptive cells | κ=0 and κ=1 at λ ∈ {0, 0.5, 1}; κ=3 at λ ∈ {0, 1} |
| extractor | depth-3 CART, min_samples_leaf 60 (unchanged) |
| rule effects | held-out AIPW φ (unchanged) |
| default | TRAIN marginal ATE (unchanged) |
| DGP | Gate 14's control-arm non-PH violation, S(730) exactly unchanged |

## Decision rule (identical to Gates 11–14, not redefined)

The primary cell (κ=3, λ=0.5) passes only if ALL of:

1. mean paired Δ = MAE(blend) − MAE(constant) < 0;
2. the 95 % paired bootstrap CI lies entirely below 0;
3. best-leaf Jaccard ≥ 0.107;
4. at least 70 % of replications have Δ < 0.

Secondary, reported regardless: paired Δ against the λ=1 plug-in on the same
seeds, which is the direct test of whether blending helps at all.

## Mandatory diagnostics

Target correlation with the true τ on TRAIN for every λ; target sd; regional
target contrast; high-leaf prevalence, purity, capture; leaf AIPW effect and SE;
best-leaf Jaccard; split stability; paired Δ MAE with CI; marginal ATE estimate
versus the true ATE.

These are read together. A better MAE without better target correlation would
indicate a different mechanism; restored correlation without a better MAE would
show that ordering alone is not sufficient.

## Scope and non-claims

* One DGP family, one violation, one sample size, one blending rule.
* The censoring model is deliberately correct. This is **not** a joint
  outcome-plus-censoring misspecification test.
* A negative result does not establish that no target construction can rescue
  discovery; it establishes that this convex blend with this pseudo-outcome does
  not.
* A positive result would not establish that ψ is a reliable rescue target in
  general; it would be specific to this setting where the censoring model is
  correct.

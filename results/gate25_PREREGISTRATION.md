# Gate 25 pre-registration — two-leaf rule list on the two-region DGP

Frozen 2026-11-03, before execution. **Final architectural experiment.**

## Question

Gate 23 showed a single leaf captures at most 0.364 of a two-region truth.
Does adding a second, low-overlap high-support leaf improve captured benefit or
ground-truth recovery without violating purity, support, or decision safety?

A larger selected set captures more benefit trivially. The success criteria are
therefore relative to the frozen C200 single-leaf method, with purity and
per-treated benefit as explicit guardrails.

## Rule selection (fixed; no tuning, max 2 leaves)

1. **First leaf:** highest discovery mean ψ among leaves with discovery support
   ≥ 200 (the frozen C200 rule).
2. **Second leaf:** highest discovery mean ψ among remaining leaves with support
   ≥ 200 and Jaccard overlap with the first leaf **< 0.20**.
3. If no second leaf is eligible, **fall back to the single first leaf** and
   report the fallback.
4. "Highest mean" is the **discovery** estimate. Validation is used only for the
   frozen commitment rule.

Because both leaves are root-to-leaf paths of one depth-3 CART, they partition
the discovery space and are disjoint by construction after the `simplify` fix.
The Jaccard threshold is an eligibility safeguard; the **actual** overlap is
reported on the assessment partition. The rules are not described as
"non-overlapping" without that measurement.

## Assignment (deterministic)

Patients are assigned to the **most specific matching selected leaf** (longest
condition); ties are broken by selection order. Patients matching no selected
leaf receive the marginal-ATE default. Support and captured benefit are computed
on the **assigned** groups, never by double-counting a patient.

## Data roles and conditions

| item | value |
|---|---|
| n | 8000 |
| partitions | 4000 discovery / 2000 validation / 2000 assessment |
| replications | 40 fresh seeds |
| seeds | 280100–280139 |
| conditions | **clean** two-region DGP (κ=0, conditional censoring); **stress** cell (κ=3 in R1, marginal KM) |
| models | A (max mean), C200 (single leaf), C400 (single leaf, sensitivity), **LIST (two leaves)** |
| discovery | depth-3 CART on ψ, min_samples_leaf 60 |
| leaf effects | AIPW on validation |
| commitment | per selected leaf, `τ̂_L > 1.645·SE_L`; uncommitted leaves fall back to the constant |
| prediction | all leaves' validation effects; uncommitted selected leaves → constant |
| assessment | 2000 untouched patients |

## Primary outcomes (LIST vs C200, clean condition, paired)

1. **Captured benefit** on the selected set (uncommitted selection quality).
2. **Ground-truth recovery** (Jaccard with R1 ∪ R2).
3. **Purity** (P(true union | selected set)).
4. **Reference-leaf recovery** (subsample refit of the same procedure vs the
   full-data selected set).
5. **Decision safety**: commitment, coverage, null-treatment, efficiency,
   captured benefit under commitment, per-treated benefit.

## Success criteria (fixed before the run)

The LIST is adopted only if ALL of:

**Representation gain**
1. captured benefit ≥ C200 + 0.05 **or** ground-truth recovery ≥ C200 + 0.05,
   with the paired 95 % bootstrap CI on the difference excluding zero;

**Purity**
2. purity ≥ C200 − 0.05;

**No coverage inflation**
3. per-treated benefit ≥ C200 − 0.02 (among committed);

**Decision safety**
4. null-treatment rate ≤ 5 %;
5. efficiency ≥ 0.70;

**Prediction**
6. MAE Δ vs the constant < 0 with the 95 % CI below zero;
7. MAE ≤ C200 + 0.005.

C400 remains sensitivity only and cannot be adopted. The stress condition is a
robustness check: the LIST must not violate purity (≥ C200 − 0.05), null ≤ 5 %,
or efficiency ≥ 0.70 there either.

## Failure interpretation (fixed)

| result | interpretation |
|---|---|
| LIST improves capture/recovery and keeps purity | the single-leaf limit is partly representational; adopt |
| capture improves but purity falls beyond tolerance | the second leaf buys coverage at the cost of precision; do not adopt |
| no second leaf eligible in most reps | the CART rarely has two high-support leaves; the representation is data-limited |
| LIST no better than C200 | the second leaf adds nothing; the single-leaf limit is structural |
| LIST works clean but fails stress | the improvement is not robust to joint nuisance error |

## Stability

The LIST stability refits the **two-leaf selection procedure** on 20 disjoint
discovery half-samples (support scaled to 100, Jaccard <0.20), forms the
assigned union, and compares it to the full-data selected union. Mean Jaccard
and recovery frequency at J > 0.8 are reported, alongside C200's single-leaf
stability.

## Scope

Two DGPs, two conditions, one sample size, max two leaves, one Jaccard
eligibility threshold (0.20). No further leaves, thresholds, or variants after
this run.

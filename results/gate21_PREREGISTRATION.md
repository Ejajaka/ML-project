# Gate 21 pre-registration — high-leaf selection under the winner's curse

Frozen 2026-11-02, before execution. Only the **selection rule** changes. The
discovery/validation/assessment roles, commitment procedure, and evaluation
metrics are frozen from Gate 20.

## Question

Gate 20 isolated the remaining problem: the commitment filter can reject null
leaves, but the max-mean selection rule picks small extreme leaves, limiting
captured benefit. Does changing the selection criterion improve capture or
stability without damaging purity, prediction, or decision quality?

## Four selection rules (fixed; no fifth rule added after inspection)

Let `ψ̄_L` be the mean of ψ on discovery for leaf L, `SE_L = sd(ψ_L)/√n_L`, and
`n_L` its discovery support.

| arm | rule |
|---|---|
| **A** | `argmax ψ̄_L` (current) |
| **B** | `argmax (ψ̄_L − 1.645·SE_L)` — one-sided 95 % lower confidence bound |
| **C** | `argmax ψ̄_L` among leaves with `n_L ≥ 200`; if none, fall back to A |
| **D** | `argmax (freq_L · ψ̄_L)`, where `freq_L` is the fraction of 20 discovery half-samples whose max-mean leaf matches L at Jaccard > 0.8 |

Arm D parameters, frozen: 20 subsamples, 50 % of discovery without replacement,
Jaccard threshold 0.8, subsample seed `rep + 11`.

## Non-circular stability outcome

The stability **outcome** for every arm is computed on a disjoint set of 20
discovery half-samples (seed `rep + 21`): refit the baseline max-mean CART on
each half-sample; the arm's leaf is "recovered" if Jaccard > 0.8. This is a
common yardstick across arms and is independent of Arm D's selection subsamples.

## Frozen design

| item | value |
|---|---|
| n | 8000 |
| partitions | 4000 discovery / 2000 validation / 2000 assessment, stratified by A |
| replications | 40 fresh seeds per cell |
| seeds | 240100–240139 |
| cells | κ=0 conditional Cox; κ=3 marginal KM |
| discovery | depth-3 CART on ψ, min_samples_leaf 60 |
| leaf effects | AIPW on validation |
| commitment | `τ̂_L > 1.645·SE_L` on validation; no treatment otherwise |
| prediction | validation leaf effects; discovery marginal ATE fallback |
| assessment | 2000 untouched patients, identical across arms |

## Primary outcomes (paired, same seeds and partitions)

1. **Captured benefit (selection):** fraction of the total true τ in the
   assessment captured by the selected leaf if always treated. This isolates
   selection from commitment.
2. **Purity:** P(true region | selected leaf) on the assessment.
3. **Discovery stability:** the non-circular outcome defined above.

## Guardrails

Predictive MAE and Δ vs the constant; coverage; commitment rate;
null-treatment rate; decision efficiency; captured benefit under the fixed
commitment. Per-treated benefit is reported descriptively, not as a paired
comparison across arms with different abstention rates.

## Contrasts and multiplicity

* **Primary contrast:** C vs A on captured benefit.
* **Secondary contrasts:** B vs A and D vs A, Holm-adjusted across the two.
* All contrasts are paired within replication; 95 % bootstrap CIs (5000
  resamples) are reported unadjusted, with Holm-adjusted p-values for the
  secondary family.

## Decision rule (fixed before the run)

A method is a candidate for adoption only if ALL of:

**Selection**
1. captured benefit ≥ Arm A + 0.03 **or** stability ≥ Arm A + 0.10;
2. purity ≥ Arm A − 0.05;
3. stability not lower than Arm A − 0.05.

**Prediction**
4. MAE Δ vs the constant < 0 with the 95 % CI below 0;
5. MAE ≤ Arm A + 0.005.

**Decision**
6. null-treatment rate ≤ 5 %;
7. decision efficiency ≥ 0.70.

Arm A cannot be "adopted"; it is the reference. The primary confirmatory claim
is C vs A; B and D are secondary.

## Interpretation rules

* C passes → support-constrained selection reduces the winner's curse; adopted.
* B or D passes but C does not → report as secondary evidence, not a
  confirmatory result.
* None passes → changing the selection criterion alone does not fix
  capture/stability; the next step is to reconsider the rule representation,
  not to add a fifth heuristic.

## Cautions recorded before the run

* The support constraint can exclude a genuinely small, high-benefit subgroup;
  that is the trade-off being measured.
* The LCB penalises uncertainty and may over-penalise small but real subgroups.
* Stability-weighted selection can be circular; the disjoint evaluation
  subsamples are the safeguard.
* The cells are two nuisance states at one sample size; no generalisation.

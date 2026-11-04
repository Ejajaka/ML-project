# Gate 22 pre-registration — valid stability measurement

Frozen 2026-11-02, before execution. **Measurement-validation gate.** No
selection rule is added, removed, or tuned. The support threshold 200 remains
the primary rule; 100 and 400 are sensitivity analyses only.

## Question

Gate 21 improved captured benefit (0.356 → 0.548), purity (0.69 → 0.81) and
capture (0.35 → 0.55) with support-constrained selection. Its stability outcome
was invalid: the yardstick refit the **max-mean** rule, itself winner's-cursed,
so every arm scored 0.02–0.05.

This gate measures stability by refitting **each arm's own selection procedure**
on disjoint discovery subsamples, and reports both reproducibility of the
full-data leaf and recovery of the **true** simulated region.

## Configurations

| config | selection procedure |
|---|---|
| A | max discovery mean ψ |
| B | max lower confidence bound, `ψ̄ − 1.645·SE` |
| C100 | max mean among leaves with discovery support ≥ 100 |
| **C200 (primary)** | max mean among leaves with discovery support ≥ 200 |
| C400 | max mean among leaves with discovery support ≥ 400 |
| D | stability-weighted: `freq × mean`, 20 inner half-samples, Jaccard 0.8 |

**Support scaling for subsample refits:** the threshold is multiplied by the
subsample fraction. A half-sample therefore uses 50 for C100, **100 for C200**,
and 200 for C400. This preserves the relative support requirement.

## Data roles (unchanged from Gates 20–21)

| partition | size | use |
|---|---|---|
| discovery | 4000 | full-data selection; all subsampling |
| validation | 2000 | not used in this gate (no commitment) |
| assessment | 2000 | evaluation of leaf masks and true-region recovery |

The assessment partition is never used to select or refit anything.

## Subsample design

* **Selection subsamples** (used only by Arm D's weights): 20 half-samples of
  discovery, seed `rep + 31`.
* **Evaluation subsamples** (disjoint): 20 half-samples of discovery, seed
  `rep + 41`.
* The **full-data reference leaf** is selected once per configuration on the
  full discovery set and frozen before any evaluation subsample is drawn.

## Stability measurements (both reported, not merged)

For configuration g with full-data leaf `L_g` and subsample leaf `L_b`:

1. **Reference-leaf recovery**

       J_full(g) = Jaccard(L_b, L_g)   evaluated on the assessment partition

   Reported as mean, median, 25th/75th percentiles, minimum, and the full
   distribution; **not only the mean**.

2. **Selection frequency**

       freq_full(g) = fraction of evaluation subsamples with J_full > 0.8

3. **Ground-truth recovery** (the check that reproducibility ≠ correctness)

       J_true(g) = Jaccard(L_b, true region) on the assessment
       freq_true(g) = fraction with J_true > 0.8

   Also report the full-data leaf's own `J(L_g, true region)` and its capture of
   the true region, so the stability of a *wrong* leaf cannot be mistaken for
   success.

## Frozen design

| item | value |
|---|---|
| n | 8000 |
| partitions | 4000 discovery / 2000 validation / 2000 assessment |
| replications | 40 fresh seeds per cell |
| seeds | 250100–250139 |
| cells | κ=0 conditional Cox; κ=3 marginal KM |
| discovery | depth-3 CART on ψ, min_samples_leaf 60 |
| evaluation subsamples | 20 per replication per configuration |

## Primary contrast

**C200 vs A** on `J_full` (mean Jaccard against the full-data reference leaf),
paired within replication, 95 % bootstrap CI. Secondary: B, C100, C400, D vs A,
Holm-adjusted across the four secondary configs.

## Decision rule (measurement gate)

There is no adoption criterion. The gate reports:

1. whether C200's capture gain is accompanied by higher reference-leaf recovery
   and higher ground-truth recovery than A;
2. whether C200's stability is materially different from A (paired CI);
3. whether the capture gain survives at support 100 and 400 (sensitivity, not
   selection);
4. whether any configuration is stable in the reference sense but *not* in the
   ground-truth sense — which would show that reproducibility alone is
   insufficient.

Pre-specified interpretation:

* C200 higher than A on both `J_full` and `J_true`, CIs excluding zero → the
  capture improvement is accompanied by genuine reproducibility.
* C200 higher on `J_full` but not `J_true` → the procedure reproduces its own
  choice without reproducing the truth; the capture gain stands but stability
  claims must be qualified.
* No config differs from A → the capture improvement is real but the selection
  is not more reproducible.
* C100/C400 preserve capture → the gain is not specific to 200. C100 or C400
  better → report, but **do not switch the primary rule** in this gate.

## Scope

One sample size, two nuisance cells, one subsample fraction (50 %), 20
evaluation subsamples. The stability estimate is conditional on the discovery
procedure and the DGP.

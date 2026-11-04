# Gate 23 pre-registration — independent DGP confirmation (two-region effect)

Frozen 2026-11-02, before execution. **The method is frozen at v1.10.** Only the
treatment-effect structure changes.

## Question

v1.10 improved capture, purity, reference-leaf recovery and ground-truth recovery
in a single-region DGP. Does the support-constrained gain travel to a
**two-region** treatment effect, which the single-leaf representation cannot
fully capture?

A two-region DGP is a stronger test than renaming the covariates: it directly
challenges the assumption that one high leaf is an adequate representation.

## DGP (only the effect structure changes)

| region | definition | effect in S(730) |
|---|---|---|
| R1 | `cd40 < median(340)` AND `age > median(34)` | +0.20 |
| R2 | `karnof < median` AND `wtkg < median` | +0.15 |
| overlap | R1 ∩ R2 | +0.20 (the larger offset) |
| outside | — | 0 |

Event times remain exponential with proportional hazards given X; censoring is
the same covariate-dependent mechanism calibrated to 10 %; nuisances are the same
per-arm Cox and conditional censoring Cox. **No censoring or nuisance change in
the primary gate.**

The true region for recovery is `R1 ∪ R2`. A single leaf can capture at most one
region, so absolute capture is expected to be well below 1. The primary test is
**C200 vs A**, not absolute recovery.

## Frozen method (v1.10)

* depth-3 CART on ψ, min_samples_leaf 60, discovery partition;
* high-leaf selection: max mean among leaves with ≥200 discovery patients;
* validation leaf effects; commitment `τ̂_L > 1.645·SE_L`; no treatment otherwise;
* stability measured by refitting each rule's own procedure on 20 disjoint
  discovery half-samples, with the support threshold scaled.

## Frozen design

| item | value |
|---|---|
| n | 8000 |
| partitions | 4000 discovery / 2000 validation / 2000 assessment |
| replications | 40 fresh seeds |
| seeds | 260100–260139 |
| nuisance state | κ=0, conditional Cox censoring (fixed) |
| configs | A, B, C100, C200, C400, D |

## Primary outcomes (paired, same seeds and partitions)

1. **Ground-truth recovery:** Jaccard of the selected leaf with `R1 ∪ R2`, and
   the full-data leaf's Jaccard with the union.
2. **Captured benefit:** fraction of total true τ captured by the selected leaf.
3. **Reference-leaf stability:** mean Jaccard of subsample-refit leaves with the
   full-data leaf; plus the recovery frequency at J > 0.8.
4. **Purity:** P(true union | selected leaf).

Guardrails: predictive MAE and Δ vs constant; commitment rate; coverage;
null-treatment rate; decision efficiency; captured benefit under commitment.
Per-treated benefit reported descriptively.

## Primary contrast

**C200 vs A**, paired within replication, 95 % bootstrap CIs. Secondary:
B, C100, C400, D vs A, Holm-adjusted across the four.

## Success criteria (fixed)

C200 is declared to **generalise** only if:

1. captured benefit > A with the 95 % CI excluding zero;
2. ground-truth recovery (Jaccard with the union) > A with the CI excluding zero;
3. purity ≥ A − 0.05;
4. prediction guardrail: MAE Δ vs constant < 0 with the CI below zero;
5. decision guardrails: null rate ≤ 5 %, efficiency ≥ 0.70.

Absolute capture is reported but is **not** a success criterion: a single leaf
cannot capture a two-region truth.

## Failure interpretation (fixed before the run)

| result | interpretation |
|---|---|
| C200 improves capture and recovery | the support constraint generalises beyond the single-region DGP |
| capture improves, ground-truth recovery does not | the method selects a useful but different population |
| stability improves, capture falls | reproducibility purchased through conservatism |
| C200 fails on two-region truth | the single-leaf representation is the limiting factor, not the selection rule |
| all methods fail | the DGP may require a different discovery architecture |

## Scope

One alternative DGP, one sample size, one nuisance state, two regions. If C200
passes, a separate pre-registered robustness cell (marginal-KM censoring) follows;
it is **not** part of this gate.

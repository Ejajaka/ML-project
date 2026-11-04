# Gate 30 pre-registration — knob sweep for C200

Frozen 2026-11-03, before execution.

## Question

The frozen C200 loses to SCRE on patient-level MAE (0.0562 vs 0.0519 single,
0.0715 vs 0.0685 two). Gate 28 also shows most of that gap comes from
abstention: on committed replications C200 is +0.0037 (single) and +0.0012
(two). Which, if any, of the remaining knobs materially improves the model
without damaging purity, the null-treatment rate or commitment?

## Design (fixed before the run)

| item | value |
|---|---|
| n | 8000 |
| partitions | 4000 discovery / 2000 validation / 2000 assessment |
| replications | 40 fresh seeds, 310100–310139 |
| designs | single region, two regions |
| nuisance | unchanged (per-arm Cox, 5-fold cross-fit, conditional censoring Cox) |
| target | unchanged (AIPW psi, e = 0.5) |

## Variants (all paired on the same splits)

| id | change | why |
|---|---|---|
| **V0** | frozen C200: depth 3, `min_leaf` 60, support ≥ 200, z = 1.645 | baseline |
| **V2** | V0 + empirical-Bayes shrinkage of leaf effects toward the marginal ATE, used for prediction only (commitment still uses the unshrunk effect) | reduce variance of step predictions |
| **V3a** | V0 with commitment z = 1.282 (one-sided 90 %) | looser policy |
| **V3b** | V0 with commitment z = 1.960 (one-sided 97.5 %) | stricter policy |
| **V4a** | `min_samples_leaf` 30 | finer leaves |
| **V4b** | `min_samples_leaf` 100 | steadier leaves |
| **V5** | tree depth 4 | better approximation |

V3a/V3b are a policy curve, not candidate replacements: they are reported, not
adopted.

## Primary outcome

**MAE**, paired against V0 within replication.

## Guardrails (a variant may replace C200 only if ALL hold)

1. MAE improves: paired 95 % bootstrap CI entirely below zero.
2. Purity falls by no more than 0.02.
3. Null-treatment rate stays 0 %.
4. Commitment rate does not fall.
5. Capture does not fall by more than 0.02.

Multiplicity: Holm adjustment across the five candidate variants (V2, V3a, V3b,
V4a, V4b, V5) on the MAE p-value.

## Secondary outcomes

MAE conditional on commitment, capture, ground-truth Jaccard, coverage,
per-treated benefit, decision efficiency.

## Small-n grid (separate question)

The support floor is absolute (200). On ACTG175 (527 discovery) it is
unreachable. Grid:

| item | value |
|---|---|
| n | 1000, 2000, 4000 |
| design | single region |
| seeds | 310100–310139 |
| variants | `abs200` (floor 200) vs `prop5` (floor max(60, 5 % of discovery)) |

Outcome: commitment rate, purity, capture, MAE vs the constant. No adoption
decision; this is a feasibility map.

## Confirmation

Any variant that passes the guardrails is re-run on **fresh seeds
320100–320139** before it is reported as an improvement. A variant that fails
confirmation is reported as a tuning artefact.

## Reporting

Whatever the outcome, both the tuning and the confirmation results are written
to `results/gate30_summary.csv` / `gate30_confirm.csv` and quoted in the paper.
Negative results are reported with the same prominence as positive ones.

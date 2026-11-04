# Gate 12 verdict — depth-3 confirmed on fresh seeds

Pre-registration: `gate12_PREREGISTRATION.md`. 40 fresh seeds per cell
(130100–130139, 130200–130239). Depth-2 reproduced on the same seeds as a check.

## Result

| n | model | rules | MAE | constant | Δ | 95 % CI | reps better | gain | adopted |
|---|---|---|---|---|---|---|---|---|---|
| 8000 | **cart_d3 (primary)** | 7.9 | 0.0538 | 0.0727 | **−0.0189** | **[−0.0229, −0.0147]** | **90 %** | **+26.0 %** | **yes** |
| 8000 | cart_d2 (reproduction) | 4.0 | 0.0615 | 0.0727 | −0.0112 | [−0.0146, −0.0077] | 82 % | +15.4 % | — |
| 12000 | **cart_d3 (primary)** | 8.0 | 0.0482 | 0.0727 | **−0.0245** | **[−0.0282, −0.0208]** | **100 %** | **+33.7 %** | **yes** |
| 12000 | cart_d2 (reproduction) | 4.0 | 0.0564 | 0.0727 | −0.0163 | [−0.0193, −0.0133] | 100 % | +22.4 % | — |

Depth-3 passed all four pre-registered checks at both sample sizes. Depth-2 also
reproduced on the same fresh seeds, so the Gate 11 result was not seed-specific.

## Diagnostics

| n | prevalence | purity | capture | leaf effect | SE | best Jaccard | stability |
|---|---|---|---|---|---|---|---|
| 8000 | 0.114 | 0.87 | 0.37 | +0.192 | 0.073 | 0.46 | 0.60 |
| 12000 | 0.111 | 0.95 | 0.39 | +0.182 | 0.063 | 0.50 | 0.63 |

The depth-3 model isolates a **small, very pure** subgroup: ~11 % of patients,
87–95 % of whom are truly in the region, with an AIPW effect of +0.18–0.19
against a true +0.195. Capture is 0.37–0.39, so it finds about two-fifths of the
region and ignores the rest. Jaccard 0.46–0.50.

Split stability is 0.60–0.63, lower than depth-2 (0.84–0.85). The deeper rule
list is less reproducible as a partition, but its effect estimates are strong
enough that the MAE gain survives.

## Comparison of the two adopted models

| n | depth-2 gain | depth-3 gain |
|---|---|---|
| 8000 | +15.4 % | **+26.0 %** |
| 12000 | +22.4 % | **+33.7 %** |

Depth-3 dominates on MAE at both sizes. The trade-off is interpretability and
stability: 8 rules instead of 4, stability 0.60–0.63 instead of 0.84–0.85.

**v1.5 extraction is therefore depth 3 at n ≥ 8000**, with depth 2 retained as
the more stable alternative when rule-list reproducibility matters more than
point accuracy.

## Scope

Unchanged: one DGP, correctly specified Cox nuisance, n ≤ 12 000, partial region
recovery. No claim of robustness to Cox misspecification or generalisation
beyond the tested range.

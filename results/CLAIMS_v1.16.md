# CISCaRL — final claims after v1.16 (Gate 30 knob sweep)

**Status:** C200 remains the frozen model and the object of every pre-registered
result. Gate 30 found one admissible improvement — empirical-Bayes shrinkage of
leaf effects for prediction only — reported as `C200-S` and confirmed on fresh
seeds. Nothing else in the sweep improved the model.

Supersedes `CLAIMS_v1.15.md`.

---

## 1. The model

| layer | specification |
|---|---|
| Estimand | P(T¹>730\|X) − P(T⁰>730\|X), composite progression-free endpoint |
| Estimation | per-arm Cox nuisances, 5-fold cross-fitted; conditional censoring Cox; cap min(Q₀.₉₉, 20); AIPW |
| Discovery | depth-3 CART on ψ, min_samples_leaf 60, discovery partition |
| Selection | max discovery mean ψ among leaves with ≥200 discovery patients |
| Leaf effects | AIPW on validation |
| Prediction | validation leaf effects, optionally shrunk toward the marginal ATE |
| Decision | TREAT only if `τ̂_L > 1.645·SE_L` on validation; else abstain |
| Roles | 4000 discovery / 2000 validation / 2000 assessment, disjoint |

## 2. Gate 30 — knob sweep (pre-registered, tune + confirm)

40 tuning seeds (310100–310139), 40 fresh confirmation seeds (320100–320139),
two DGPs, paired within replication. Pooled over both blocks (80 reps):

| change | one region ΔMAE | two regions ΔMAE | verdict |
|---|---|---|---|
| **shrink leaf effects** | **−0.0051 [−0.0077, −0.0025]** | **−0.0026 [−0.0048, −0.0005]** | **adopted (prediction only)** |
| min_leaf 100 | +0.0002 [−0.0011, +0.0015] | −0.0016 [−0.0026, −0.0006] | rejected (design-specific) |
| min_leaf 30 | −0.0002 [−0.0014, +0.0011] | +0.0003 [−0.0008, +0.0013] | rejected |
| depth 4 | +0.0099 [+0.0080, +0.0117] | +0.0083 [+0.0067, +0.0100] | rejected (worse) |
| z = 1.28 | −0.0001 [−0.0004, +0.0000] | −0.0001 [−0.0004, +0.0000] | policy dial, not adopted |
| z = 1.96 | +0.0005 [+0.0001, +0.0010] | +0.0004 [+0.0000, +0.0008] | policy dial, not adopted |

Shrinkage leaves every policy quantity unchanged on the confirmation block:
purity 0.8719/0.8777, capture 0.4510/0.2562, commitment 72.5 %/62.5 %, and
Spearman 0.522/0.444 are identical between C200 and C200-S.

## 3. Head-to-head with SCRE (80 paired replications)

| design | method | MAE | vs SCRE [95 % CI] | p | purity | capture | per-treated | coverage |
|---|---|---|---|---|---|---|---|---|
| single | C200 | 0.0576 | +0.0062 [+0.0031, +0.0094] | <0.001 | 0.890 | 0.465 | +0.174 | 0.133 |
| single | C200-S | 0.0525 | +0.0011 [−0.0019, +0.0042] | 0.48 | 0.890 | 0.465 | +0.174 | 0.133 |
| single | SCRE | 0.0514 | — | — | 0.437 | 0.954 | +0.085 | 0.548 |
| two | C200 | 0.0707 | +0.0031 [+0.0001, +0.0059] | 0.044 | 0.858 | 0.277 | +0.165 | 0.117 |
| two | C200-S | 0.0680 | +0.0004 [−0.0025, +0.0033] | 0.76 | 0.858 | 0.277 | +0.165 | 0.117 |
| two | SCRE | 0.0676 | — | — | 0.599 | 0.823 | +0.101 | 0.563 |

## 4. Small-sample support floor

| n | absolute 200: purity / capture / commit | 5 % proportional: purity / capture / commit |
|---|---|---|
| 1000 | 0.614 / 0.105 / 27.5 % | 0.624 / 0.102 / 27.5 % |
| 2000 | 0.468 / 0.096 / 17.5 % | **0.728 / 0.127 / 30.0 %** |
| 4000 | 0.710 / 0.311 / 55 % | 0.804 / 0.281 / 55 % |

The absolute floor is 20 % of discovery at n = 2000 and behaves badly there; a
proportional floor is the right specification for small samples. Not
retro-fitted.

## 5. Claims we can defend

> With empirical-Bayes shrinkage of leaf effects used for prediction, the model
> is statistically tied with a survival causal rule ensemble on patient-level
> MAE over 80 paired replications (difference +0.0011, p = 0.48 single region;
> +0.0004, p = 0.76 two regions), while treating 12–13 % of patients against
> 55–56 %, delivering about twice the benefit per treated patient, and abstaining
> when the validation lower bound is not positive.

Shrinkage is a post-hoc calibration confirmed on fresh seeds; the frozen
pre-registered model remains C200.

## 6. Claims we cannot make

- ❌ "CISCaRL beats SCRE on accuracy." Tied after calibration, not better.
- ❌ "CISCaRL captures more benefit." Capture 0.28–0.47 vs 0.82–0.95.
- ❌ "Depth 4 / min_leaf tuning helps." Measured; depth 4 is clearly worse.
- ❌ "The support floor 200 is optimal." Proportional is better at small n.
- ❌ "Shrinkage was pre-registered." It was chosen after Gate 28 and confirmed
  separately; say so.
- ❌ "Validated on ACTG175." The model abstains there.

## 7. Remaining

- Paper: §V-F and §V-G updated; abstract, discussion and conclusion updated.
- Notebook: Gate 30 section added; ACTG175 output unchanged (abstention).
- Future work: proportional floor, diagonal geometry, selection-aware inference.

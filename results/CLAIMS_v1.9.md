# CISCaRL — consolidated claims after v1.9

**Status:** v1.8 honest decision layer + support-constrained high-leaf selection
(Gate 21). All numbers use the corrected rule extractor and the honest
4000/2000/2000 partition.

Supersedes `CLAIMS_v1.8.md`.

---

## 1. The model (v1.9)

| layer | specification |
|---|---|
| Estimand | progression-free probability at day 730 (composite) |
| Estimation | per-arm Cox nuisances, 5-fold cross-fitted; conditional censoring Cox; cap min(Q₀.₉₉, 20); AIPW |
| Discovery | depth-3 CART on ψ, min_samples_leaf 60, fitted on discovery |
| **High-leaf selection** | **maximum discovery mean ψ among leaves with ≥200 discovery patients** |
| Leaf effects | AIPW on the validation partition |
| Prediction | validation leaf effects; discovery marginal ATE fallback |
| Decision | treat the high leaf only if `τ̂_L > 1.645·SE_L` on validation; otherwise no treatment |

## 2. Why the selection change matters (Gate 21)

| | max mean (v1.8) | **support ≥200 (v1.9)** |
|---|---|---|
| captured benefit (selection) | 0.356 | **0.548** |
| purity | 0.69 | **0.81** |
| capture of true region | 0.35 | **0.55** |
| Jaccard | 0.33 | **0.51** |
| commitment rate | 55 % | **80 %** |
| coverage | 0.082 | **0.134** |
| captured benefit (committed) | 0.288 | **0.474** |
| null-treatment rate | 5 % | **0 %** |
| decision efficiency | 0.83 | 0.85 |
| predictive MAE | 0.0601 | 0.0602 |

Primary contrast C − A on captured benefit: **+0.192 [+0.113, +0.278],
p < 0.001**. The LCB rule also improves capture (+0.070, Holm p = 0.017); the
stability-weighted rule fails (purity 0.41, null 13 %).

## 3. Honest decision performance (v1.9, joint cell)

Commitment 80 %, coverage 0.134, per-treated benefit +0.16, efficiency 0.85,
captured benefit 0.474, null-treatment 0 %, beneficial 100 %. Prediction
MAE 0.0602 against a constant 0.0702 (~14 % gain), CI below zero.

## 4. What the model recovers

A **substantial but partial** region: purity 0.81, capture 0.51–0.55, Jaccard
0.46–0.51. Not full recovery, but roughly double the capture of v1.8.

## 5. Stability — unresolved and currently unmeasured

The pre-registered stability yardstick refits the **max-mean** rule on disjoint
discovery halves. Because that refit is itself winner's-cursed, it matches the
selected leaf in only 2–5 % of halves for *every* selection rule. The metric
cannot discriminate and must not be read as "the rule list is stable."

A valid stability measure must refit each arm's own selection rule (e.g.
support-constrained refits for v1.9) on disjoint subsamples. That is the next
experiment.

## 6. Claims we can defend

> Under the specified simulation DGP at n = 8000 with a 4000/2000/2000 honest
> partition, a depth-3 rule model on the AIPW pseudo-outcome with
> support-constrained high-leaf selection captures 55 % of the true regional
> benefit at 81 % purity, and commits to treatment in 80 % of replications with
> a 0 % null-treatment rate, 0.134 coverage, 0.474 captured benefit and 0.85
> decision efficiency. Predictive MAE beats the constant by ~14 % under joint
> outcome-plus-censoring misspecification.

## 7. Claims we cannot make

- ❌ "Recovers the full region." Capture 0.51–0.55.
- ❌ "Stable rule list." The current stability metric is invalid; do not quote
  0.04 as stability.
- ❌ "The predictive gain is 35 %." It is ~14 % in the honest 50/25/25 partition;
  35 % requires a 70/30 discovery split.
- ❌ "The support threshold 200 is optimal." One frozen value, not tuned.
- ❌ "Validated beyond n = 8000, two nuisance cells, one partition ratio."
- ❌ "CISCaRL outperforms SCRE." Still untested fairly.
- ❌ "95 % confidence intervals." ≈ 93 %.

## 8. Next experiment

**A valid stability measure.** For each selection rule, refit that same rule on
disjoint discovery subsamples and measure Jaccard recovery of the full-data leaf.
For the support-constrained rule the subsample support threshold must scale
(e.g. 100 on a 2000-patient subsample). This will answer whether v1.9 is stable,
and whether stability can be improved without losing capture.

Secondary: a support-threshold sensitivity analysis (100/200/400) to check that
the gain is not specific to 200. This is a robustness check, not a new selection
rule.

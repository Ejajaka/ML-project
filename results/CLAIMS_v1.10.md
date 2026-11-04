# CISCaRL — consolidated claims after v1.10

**Status:** v1.9 support-constrained selection, now with a **valid** stability
measurement (Gate 22). All numbers use the corrected rule extractor, the honest
4000/2000/2000 partition, and each configuration's own refit procedure.

Supersedes `CLAIMS_v1.9.md`.

---

## 1. The model (v1.10)

| layer | specification |
|---|---|
| Estimand | progression-free probability at day 730 (composite) |
| Estimation | per-arm Cox nuisances, 5-fold cross-fitted; conditional censoring Cox; cap min(Q₀.₉₉, 20); AIPW |
| Discovery | depth-3 CART on ψ, min_samples_leaf 60, discovery partition |
| **High-leaf selection** | **max discovery mean ψ among leaves with ≥200 discovery patients** |
| Leaf effects | AIPW on validation |
| Prediction | validation leaf effects; discovery marginal ATE fallback |
| Decision | treat the high leaf only if `τ̂_L > 1.645·SE_L` on validation; else no treatment |

## 2. What the model recovers (Gate 22, joint cell)

| metric | max mean (v1.8) | **support ≥200 (v1.10)** |
|---|---|---|
| capture of true region | 0.343 | **0.623** |
| full-data leaf Jaccard with truth | 0.317 | **0.522** |
| reference-leaf recovery (J_full) | 0.199 | **0.318** (+0.119 [+0.081, +0.159]) |
| ground-truth recovery (J_true) | 0.240 | **0.334** (+0.094 [+0.076, +0.116]) |
| half-samples recovering the leaf at J>0.8 | 4 % | **7 %** |
| purity | 0.69 | **0.81** |

Threshold sensitivity (descriptive): capture 0.500 at 100, 0.623 at 200,
0.660 at 400. The gain is not specific to 200; 200 remains the primary rule.

## 3. Honest decision performance (unchanged from Gate 21)

Commitment 80 %, coverage 0.134, per-treated +0.16, efficiency 0.85, captured
benefit 0.474, null-treatment 0 %, beneficial 100 %. Predictive MAE 0.0602
against a constant 0.0702 (~14 % gain, CI below zero).

## 4. Stability — measured, improved, still modest

Mean reference-leaf Jaccard 0.318 (median 0.28, q25 0.15, q75 0.44); 7 % of
half-samples recover the full-data leaf at Jaccard > 0.8. This is a **real
improvement over the max-mean rule** (+0.119) and it co-occurs with better
ground-truth recovery (+0.094), so it is not reproducibility of a wrong leaf.
It is **not** a stable rule list in absolute terms.

## 5. Claims we can defend

> Under the specified simulation DGP at n = 8000 with a 4000/2000/2000 honest
> partition, a depth-3 rule model on the AIPW pseudo-outcome with
> support-constrained leaf selection captures 62 % of the true regional benefit
> at 81 % purity. Its selected leaf is closer to the true region than the
> max-mean leaf (Jaccard 0.52 vs 0.32), is more reproducible across discovery
> half-samples (mean Jaccard 0.32 vs 0.20; ground-truth recovery +0.09), and
> yields an 80 % commitment rate, 0.134 coverage, 0.474 captured benefit, 0 %
> null-treatment and 0.85 decision efficiency. Prediction beats the constant by
> ~14 % under joint outcome-plus-censoring misspecification.

## 6. Claims we cannot make

- ❌ "Stable rule list." Mean recovery Jaccard 0.32; only 7 % of half-samples
  recover the leaf at J>0.8. Better than v1.8, not stable.
- ❌ "Recovers the full region." Capture 0.62, not 1.0.
- ❌ "Support threshold 200 is optimal." 400 is descriptively better; 200 was
  frozen and is not switched.
- ❌ "The predictive gain is 35 %." It is ~14 % in the honest 50/25/25
  partition; 35 % requires a 70/30 discovery split.
- ❌ "Validated beyond one DGP family, n = 8000, two nuisance cells."
- ❌ "CISCaRL outperforms SCRE." Still untested fairly.
- ❌ "95 % confidence intervals." ≈ 93 %.

## 7. Next experiment

**Independent confirmation on a genuinely different DGP.** Freeze v1.10 and
re-run it on a new generating mechanism: a different region definition (e.g. a
threshold on other covariates), a different effect shape (e.g. smooth or
multi-region), and a different censoring mechanism. Fresh seeds, same four-part
rule and honest partition. This tests whether v1.10 is a method or a fit to one
simulation.

Secondary: the real-data ACTG175 illustration under v1.10 (output format only,
not evidence).

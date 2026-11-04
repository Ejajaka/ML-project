# CISCaRL — consolidated claims after v1.7

**Status:** v1.6 discovery (ψ + depth-3 CART) plus a hard-commitment decision
layer (Gate 19). All numbers here are from the **corrected** rule extractor
(`results/CORRECTION_rules_simplify.md`); pre-fix figures are not used.

Supersedes `CLAIMS_v1.6.md`. That document remains the record of the v1.6
discovery result; its numbers should be read together with the correction note.

---

## 1. The model

| layer | specification |
|---|---|
| Estimand | progression-free probability at day 730 (composite endpoint) |
| Estimation | per-arm Cox nuisances, 5-fold cross-fitted; conditional censoring Cox; cap min(Q₀.₉₉, 20); AIPW |
| Discovery | depth-3 CART on the AIPW pseudo-outcome ψ, min_samples_leaf 60 |
| Prediction | leaf AIPW effects; TRAIN marginal ATE fallback |
| **Decision (new)** | **treat the high leaf only if τ̂_L > 1.645·SE_L; otherwise abstain** |
| Prediction commitment | assign a leaf effect only if it differs from the marginal (paired IF); else constant |

## 2. Predictive performance (corrected, Gates 16–18)

| n | κ | censoring | ψ gain | reps better |
|---|---|---|---|---|
| 8000 | 0 | conditional | +35.7 % | 98 % |
| 8000 | 3 | conditional | +30.9 % | 92 % |
| 12000 | 0 | conditional | +46.8 % | 100 % |
| 12000 | 3 | conditional | +51.9 % | 100 % |
| 8000 | 3 | marginal KM | +29.1 % | 92 % |

All four pre-registered checks pass in every cell. The marginal ATE is unbiased
in the correct-outcome cells and small-bias under joint misspecification
(−0.0047).

**Target comparison, corrected:** the plug-in is better at n = 4000–6000 (+11 %
to +26 % vs ψ +4.5 % to +21.7 %); at n = 8000/κ=0 the two are statistically tied
(paired Δ −0.0023 [−0.0069, +0.0026]); at n = 12000 and under κ=3, ψ is
significantly better. **Operating range: ψ from n ≈ 6000; plug-in from
n ≈ 4000. Crossover ≈ 8000.**

## 3. What the model recovers (corrected)

The high leaf is a **small, high-purity slice**:

* purity 0.67–0.83;
* capture of the true region 0.23–0.25 (not 0.30–0.43 as the pre-fix inflated
  masks suggested);
* best-leaf Jaccard 0.61–0.66;
* split stability 0.16–0.20.

It is not subgroup recovery. Capture has been revised downward; the earlier
"high-purity subset" claim stands, but the subset is narrower than reported.

## 4. Decision quality (Gate 19, joint cell)

| | no commitment | **hard commitment** |
|---|---|---|
| null-benefit treated reps | 12 % | **0 %** |
| beneficial treated reps | 88 % | **100 %** |
| per-treated benefit | +0.150 | **+0.183** |
| efficiency vs oracle | 0.75 | **0.93** |
| coverage | 0.100 | 0.088 |
| captured benefit | 0.36 | 0.32 |
| abstention | 0 % | **45 %** |
| MAE gain | +21.1 % | +14.8 % |

Commitment is **decision-positive and prediction-negative**. Shrinkage (arm C)
does not fix the null rate and slightly worsens MAE; it is rejected.

## 5. Robustness

* **Censoring-weight tilt:** Δ remains negative with CI below zero over
  δ ∈ [−2, 2] (Gate 13; pre-fix numbers, qualitative result only).
* **Outcome misspecification:** ψ survives κ=3 (+30.9 % at 8000, +51.9 % at
  12000) while the plug-in collapses (−4.9 %, +5.3 %).
* **Joint outcome + censoring misspecification:** prediction survives (+29.1 %),
  but the subgroup bar fails (beneficial in 70 % vs 90 % required). Commitment
  fixes the null-leaf rate; the prediction and subgroup claims remain separate.

## 6. Claims we can defend

> Under the specified simulation DGP, a depth-3 rule model fitted to the AIPW
> pseudo-outcome improves held-out MAE over the marginal ATE by 29–52 % at
> n = 8000–12 000, in 92–100 % of replications, under a correct outcome model,
> a severe within-arm non-proportional-hazards violation, and (for prediction)
> joint outcome-plus-censoring misspecification. Adding a leaf-level commitment
> rule eliminates null-benefit treatment (12 % → 0 %) and raises efficiency from
> 0.75 to 0.93 of the oracle, at the cost of 45 % abstention and an 11 % loss of
> captured benefit. The model identifies a small high-purity subset
> (purity 0.67–0.83, capture 0.23–0.25), not the full region.

## 7. Claims we cannot make

- ❌ "Recovers the true subgroup." Capture 0.23–0.25; Jaccard 0.61–0.66.
- ❌ "Stable rule list." Split stability 0.16–0.20.
- ❌ "Beats the constant at ACTG175 scale." It does not at n = 589.
- ❌ "ψ dominates the plug-in at all n." It ties at n = 8000/κ=0 and loses at
  4000–6000; the crossover is ≈8000.
- ❌ "Commitment improves prediction." It costs MAE (+21.1 % → +14.8 %) and
  captured benefit (0.36 → 0.32). It improves decisions.
- ❌ "Validated beyond n = 12 000 or beyond this DGP family."
- ❌ "CISCaRL outperforms SCRE." Still untested fairly.
- ❌ "95 % confidence intervals." ≈ 93 %.

## 8. Open items

1. Re-run Gates 10–15 on the corrected extractor if their numbers are to be
   quoted; their qualitative conclusions are consistent with Gates 16–19.
2. Fixed-capacity policy frontier: commitment abstains 45 %; if a minimum
   coverage is required, report the coverage/capture trade-off rather than the
   null rate alone.
3. Rule-list stability remains the weakest dimension.
4. Real-data section remains an output-format illustration, not evidence.

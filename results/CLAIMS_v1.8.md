# CISCaRL — consolidated claims after v1.8

**Status:** v1.7 discovery + an **honestly validated** commitment layer
(4000 discovery / 2000 validation / 2000 assessment). All numbers use the
corrected rule extractor (`CORRECTION_rules_simplify.md`).

Supersedes `CLAIMS_v1.7.md`. The v1.7 decision figures (test-set reuse) are
withdrawn and replaced by the Gate 20 honest figures.

---

## 1. The model (v1.8)

| layer | specification |
|---|---|
| Estimand | progression-free probability at day 730 (composite) |
| Estimation | per-arm Cox nuisances, 5-fold cross-fitted; conditional censoring Cox; cap min(Q₀.₉₉, 20); AIPW |
| Discovery | depth-3 CART on ψ, min_samples_leaf 60, fitted on the discovery partition |
| Leaf effects | AIPW on the **validation** partition |
| Prediction | validation leaf effects; discovery marginal ATE fallback |
| Decision | treat the high leaf only if `τ̂_L > 1.645·SE_L` on validation; otherwise **no treatment** |

Data roles are disjoint: discovery fits the CART and nuisances; validation
estimates leaf effects and the commitment decision; assessment is used only to
score. The assessment partition never affects a decision.

## 2. Honest decision performance (Gate 20, joint cell, V-full)

| metric | no commitment | **honest commitment** |
|---|---|---|
| commitment rate | 100 % | **50 %** |
| coverage | 0.113 | 0.092 |
| per-treated benefit | +0.127 | **+0.161** |
| decision efficiency | 0.68 | **0.88** |
| captured benefit | 0.304 | 0.258 |
| null-treatment rate | 17.5 % | **0 %** |
| beneficial treated runs | 82.5 % | **100 %** |
| predictive MAE | 0.0574 | 0.0571 |
| Δ vs constant | −0.0129 | −0.0131 [−0.0183, −0.0081] |

Paired: null rate −17.5 pp [−30.0, −7.5]; captured −4.7 pp [−9.3, −1.7]; MAE
unchanged. Commitment is decision-positive, prediction-neutral, and costs ~5 pp
of captured benefit.

**Validation size matters:** with 1000 validation patients the quality is
unchanged (null 0 %, efficiency 0.87) but coverage falls to 0.069, captured to
0.184, and the prediction no longer beats the constant (Δ = +0.0083, CI above
zero). **At n = 8000 the honest design requires the full 2000-patient validation
set.**

## 3. Predictive performance and the partition trade-off

| design | discovery n | ψ MAE gain |
|---|---|---|
| 70/30 split (Gate 16) | 5600 | +35.7 % (n=8000), +46.8 % (n=12000) |
| 50/25/25 honest (Gate 20) | 4000 | **+18.7 %** (n=8000, joint cell) |

The honest three-way partition costs about half the predictive gain because the
discovery set shrinks. This is a genuine budget trade-off: discovery wants
patients, validation needs enough to estimate leaf effects, assessment must be
untouched.

## 4. What the model recovers

High-purity but **narrow**: purity 0.64–0.68, capture 0.24–0.30, Jaccard
0.53–0.57, split stability 0.16–0.17. Not region recovery.

## 5. The remaining diagnosed weaknesses

1. **Winner's curse in leaf selection.** The high leaf is chosen by maximum
   discovery mean ψ, which selects small extreme leaves (diagnostic: a 64-patient
   leaf at ψ = 0.434 over a 321-patient leaf at ψ = 0.335). This is why the
   no-commitment null rate is 17.5 % and capture is only ~0.24–0.30.
2. **Low capture** of the true region (0.24–0.30).
3. **Low split stability** (0.16–0.17).

These are the next experimental targets; Gate 20 was a correctness gate and did
not address them.

## 6. Claims we can defend

> Under the specified simulation DGP at n = 8000, a depth-3 rule model on the AIPW
> pseudo-outcome, with leaf effects estimated on a disjoint validation partition,
> beats the marginal ATE by 18.7 % in held-out MAE under joint outcome-plus-
> censoring misspecification. An honest leaf-level commitment rule reduces
> null-benefit treatment from 17.5 % to 0 % and raises decision efficiency from
> 0.68 to 0.88, with no prediction loss, at the cost of 50 % abstention and
> 4.7 pp of captured benefit. The model identifies a narrow high-purity subset
> (purity ≈ 0.65, capture ≈ 0.25–0.30), not the full region.

## 7. Claims we cannot make

- ❌ "Gate 19's decision numbers." Withdrawn; use Gate 20's honest figures.
- ❌ "Recovers the true subgroup." Capture 0.24–0.30.
- ❌ "Stable rule list." Split stability 0.16–0.17.
- ❌ "Beats the constant at ACTG175 scale." It does not at n = 589.
- ❌ "The predictive gain is 35 % in the honest design." It is 18.7 % with the
  50/25/25 partition; 35.7 % requires a 70/30 discovery split.
- ❌ "Validated beyond n = 8000, two nuisance cells, one partition ratio."
- ❌ "95 % confidence intervals." ≈ 93 %.
- ❌ "CISCaRL outperforms SCRE." Still untested fairly.

## 8. Next experiment (diagnosed, not scheduled)

**Selection under the winner's curse.** Compare high-leaf selection rules on
capture, purity and stability, with prediction and decision metrics as
guardrails:

* max discovery mean ψ (current);
* max lower confidence bound (mean − z·SE);
* max mean among leaves with ≥200 discovery patients;
* stability-weighted selection across discovery subsamples.

Fresh seeds; the discovery/validation/assessment roles are unchanged.

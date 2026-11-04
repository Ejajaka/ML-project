# CISCaRL — consolidated claims after v1.6

**Status:** the v1.5 plug-in discovery target is superseded by the AIPW
pseudo-outcome at n ≥ 8000. The estimator and uncertainty layers are unchanged
from v1.3. The application-level decision layer remains stopped at ACTG175 size.

Supersedes `CLAIMS_v1.5.md`. That document remains the correct record of the
plug-in line of work (Gates 10–14) and of the estimand and uncertainty sections.

---

## 1. What changed in v1.6

Gate 16 confirmed on fresh seeds that a depth-3 CART fitted to the **AIPW
pseudo-outcome ψ** beats both the marginal ATE and the v1.5 plug-in target, at
n = 8000 and n = 12000, under a correct outcome model **and** under Gate 14's
severe outcome-model violation.

| n | κ | ψ gain | plug-in gain | ψ reps better | plug-in reps better |
|---|---|---|---|---|---|
| 8000 | 0 | **+37.8 %** | +28.2 % | 100 % | 92 % |
| 8000 | 3 | **+32.7 %** | −4.0 % | 95 % | 38 % |
| 12000 | 0 | **+48.0 %** | +32.5 % | 100 % | 98 % |
| 12000 | 3 | **+53.4 %** | +6.5 % | 100 % | 57 % |

Paired ψ vs plug-in MAE is negative in every cell (65–95 % of replications).

The model is therefore **simpler** (no cross-fitted plug-in), **more accurate**,
and **robust to the outcome-model misspecification that broke v1.5**.

## 2. The estimand (unchanged from v1.3)

> τ(730) = contrast in the probability of remaining free of the composite
> endpoint (CD4 decline ≥ 50, AIDS progression, or death) through day 730, under
> the observed follow-up process.

Observed-follow-up contrast, not ITT, not per-protocol.

## 3. Why ψ works — the leaf-level argument

ψ has sd ≈ 0.97 and an individual-level correlation with true τ of only ≈ 0.08,
yet its discovered rules reach Jaccard 0.65–0.84. The reason is that a tree uses
**leaf means**, not individual values:

* a depth-3 leaf at n = 8000 holds ~700 training patients, so its mean ψ has
  SE ≈ 0.97/√700 ≈ 0.037 against a regional contrast of 0.195 (≈ 5σ);
* at n = 2000 the same leaf holds ~175 patients (SE ≈ 0.073) and the tree is
  selecting among many noisy splits — the regime where the old RF + stability
  pipeline failed.

ψ is also **doubly robust**: under a wrong outcome model but a correct censoring
model it remains unbiased, which is why it survives κ=3 while the plug-in does
not.

**Consequence for diagnostics:** individual-level target correlation is not a
valid predictor of discovery performance. Leaf-level signal-to-noise is.

### 3.1 Operating range and the target crossover (Gate 17)

40 fresh seeds per cell, κ=0:

| n | plug-in gain | ψ gain | oracle gain | better target |
|---|---|---|---|---|
| 2000 | −17.9 % | −22.8 % | **+40.1 %** | neither works |
| 4000 | **+16.6 %** | +8.1 % | +59.5 % | plug-in |
| 6000 | +24.2 % | +23.0 % | +61.3 % | tie |
| 8000 | +28.2 % | **+37.8 %** | — | ψ |
| 12000 | +32.5 % | **+48.0 %** | — | ψ |

Three consequences:

1. **The v1.6 operating range starts at n ≈ 4000**, not 8000. The old 8000
   threshold was measured with the RF + stability extractor; the single CART
   moves it down by about one doubling.
2. **The best target depends on n, with a crossover at ~6000.** Below it, the
   plug-in's lower variance wins; above it, ψ's unbiasedness plus leaf-level
   averaging wins.
3. **At n = 2000 the barrier is discovery, not information.** The oracle (true
   membership, same AIPW effects) beats the constant by 40.1 % at n = 2000, so
   the data contain the signal; the discovered rules are what fail.

### 3.2 Joint nuisance misspecification (Gate 18)

n = 8000, 40 fresh seeds, 2×2 design: outcome correct vs κ=3, censoring
conditional Cox vs marginal Kaplan–Meier (both applied to the same data).

| cell | ATE bias | MAE gain | purity | high-leaf true τ | null-leaf rate | efficiency |
|---|---|---|---|---|---|---|
| correct / conditional | −0.0012 | +30.5 % | 0.768 | +0.151 | 12.5 % | 0.766 |
| correct / marginal | −0.0044 | +30.6 % | 0.687 | +0.134 | 25 % | 0.687 |
| κ=3 / conditional | −0.0013 | +30.1 % | 0.823 | +0.162 | 10 % | 0.830 |
| **κ=3 / marginal** | **−0.0047** | **+31.0 %** | **0.659** | **+0.129** | **30 %** | **0.660** |

The joint cell passes every predictive and estimation criterion but **fails the
pre-registered subgroup bar**: the high leaf is genuinely beneficial in 70 % of
replications rather than the required 90 %. It is **never harmful** (0 % negative
in every cell); the failure is a tripled rate of null leaves, i.e. wasted
treatment, not harm.

Double robustness therefore protects the **average effect** even when both
nuisances are wrong, but not the **identity of the subgroup**. Prediction and
subgroup identification must be claimed separately.

## 4. The cost: partition stability

| target | split stability | Jaccard | purity | capture |
|---|---|---|---|---|
| ψ | **0.20–0.25** | 0.65–0.84 | 0.69–0.73 | 0.30–0.43 |
| plug-in | 0.61–0.68 | 0.25–0.48 (κ=3) | 0.34–0.35 (κ=3) | 0.15–0.19 (κ=3) |

ψ produces better predictions and recovers the region better, but its rule list
is **markedly less reproducible** across half-samples. If the paper's goal is a
stable rule list rather than predictive accuracy, the plug-in or a resampled
ψ-based rule list may still be preferable. This is a genuine trade-off, not a
defect to be hidden.

## 5. Estimation and uncertainty (unchanged from v1.3)

- Bias ≈ −0.003 to −0.012 in the tested DGPs.
- Consistent under a correct censoring model; **not** doubly robust for the ATE
  in the sense of tolerating an arbitrary censoring model — it needs *at least
  one* nuisance correct.
- Marginal interval coverage 93.1–93.4 % (333 rules) against 95 % nominal.
- Near-null false-commitment rate 7.5 % (nominal 5 %).

## 6. Decision layer (unchanged from v1.3)

At n = 589: TREAT precision 66.7 % [54.8, 79.3], abstention 90 %, ≈460 patients
per rule for 80 % power. Power-limited at ACTG175 scale; the v1.6 improvement
requires n ≥ 8000 and does not change this.

## 7. Frozen specification — v1.6

| Component | Value |
|---|---|
| Horizon | 730 days |
| Propensity | known design value, e = 0.5 |
| Outcome nuisance | per-arm Cox, 5-fold cross-fitted |
| Censoring nuisance | conditional Cox on X, 5-fold cross-fitted |
| Weight cap | min(Q₀.₉₉ of fold-training weights, 20) |
| Subgroup estimator | AIPW; consistent under correct Ĝ; doubly robust in ψ |
| Interval | IF + t-quantile, conditional on the selected rule list |
| **Discovery target (v1.6)** | **AIPW pseudo-outcome ψ** |
| **Extraction (v1.6)** | **CART depth 3, min_samples_leaf 60** |
| Rule effects | held-out AIPW φ |
| Default | TRAIN marginal ATE |
| Sensitivity | weight tilt $\hat G^{\exp(\delta)}$; δ = 0 reproduces baseline |
| Multiplicity | Holm within the rule list |

The v1.5 plug-in target is retained only as a documented alternative: it was
necessary at n = 2000 (Gates 11–12) but is dominated by ψ at n ≥ 8000 and is not
robust to the Gate 14 violation.

## 8. The claim we can defend

> CISCaRL provides interpretable subgroup estimation for a progression-free
> composite outcome with uncertainty quantification and explicit abstention. In
> simulation the estimator has low bias and approximately nominal marginal
> coverage (≈ 93 %). At ACTG175 scale the decision layer is power-limited
> (≈ 460 patients per rule for 80 % power). Separately, under the specified
> simulation DGP at n = 8000–12 000, a depth-3 rule model fitted to the AIPW
> pseudo-outcome and scored with held-out AIPW effects achieves lower held-out
> MAE than the training-only marginal ATE: +38 % and +48 % under a correct
> outcome model, and +33 % and +53 % under a severe within-arm
> non-proportional-hazards violation, in 95–100 % of replications. The
> operating range begins at n ≈ 4000 (+8 % marginal, +23 % at 6000), and below
> ~6000 the plug-in target is the better choice. The model isolates a high-purity
> subset (purity 0.69–0.73, capture 0.30–0.43), not the full treatment-effect
> region, and its rule list is less stable (0.20–0.25) than the plug-in
> alternative (0.61–0.68).

## 9. Claims we cannot make

- ❌ "CISCaRL recovers the true subgroup." Best-leaf Jaccard 0.65–0.84, capture 0.30–0.43.
- ❌ "The rule model beats the constant at ACTG175 scale." It does not at n = 589.
- ❌ "Robust to joint nuisance misspecification." Measured: prediction survives (+31 %, 92 % of reps), but the high leaf is genuinely beneficial in only 70 % of replications (90 % required), with null leaves tripled (10 % → 30 %). It is never harmful; the failure is wasted treatment.
- ❌ "The n = 8000 threshold is intrinsic." Measured instead: the v1.6 model works from ≈4000, the old pipeline from ≈8000, and the plug-in is better between 4000 and 6000.
- ❌ "Generalises beyond n = 12 000 or beyond this DGP."
- ❌ "Stable rule list." Split stability 0.20–0.25.
- ❌ "CISCaRL outperforms SCRE on HTE accuracy." Still untested fairly.
- ❌ "95 % confidence intervals." ≈ 93 %.
- ❌ "Survival benefit" / "intention-to-treat effect." See v1.3 §2.

## 10. Remaining work, in order

1. ~~**Gate 17** — locate the lower-n boundary for the v1.6 extractor.~~ **Done:
   ≈4000, with a target crossover at ≈6000.**
2. ~~**Gate 18** — joint misspecification (wrong outcome **and** wrong censoring).~~ **Done: prediction survives; subgroup reliability fails the 90 % bar.**
3. **Gate 19 (next)** — leaf-level commitment: assign a leaf's effect only if its
   interval excludes zero, otherwise fall back to the constant. Target the
   null-leaf rate found in Gate 18.
4. **Rule-list stability** — if a stable list matters more than MAE, test a
   resampled/stability-selected ψ rule list at n ≥ 8000.
5. Rewrite the paper against v1.6 terminology and scope.

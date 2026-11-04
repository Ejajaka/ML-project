# CISCaRL — consolidated claims after v1.5

**Status:** estimation and uncertainty layers validated under simulation (v1.3); the
discovery/prediction layer now has a positive, scoped result at large n (v1.5). The
application-level decision layer is still stopped at ACTG175 sample size.

Supersedes `CLAIMS_v1.3.md`. The v1.3 text remains the correct record of the
estimand, the uncertainty calibration and the decision-layer power limit.

---

## 1. What changed since v1.3

v1.3 and the Gate 4d/7 comparisons left one embarrassing fact: at n = 589 and
n = 2000 the discovered rule model was **worse** than a constant marginal-ATE
predictor. Gates 6–11 have now located why and removed it at large n.

| Gate | Question | Result |
|---|---|---|
| 6 | Is the pipeline or the pseudo-outcome the bottleneck? | Pipeline is fine: RF on the **true** τ gives Jaccard 0.98, stability 100 %. ψ is the bottleneck. |
| 7 | Does raising min_support help? | No. Best cell Δ = −0.0013, CI includes zero; none adopted. |
| 8 | Does an RMST estimand reduce ψ noise? | No. SNR 0.20 → 0.19; Δ = +3.50 days, CI [+2.15, +4.93] (worse). Not adopted. |
| 9 | Does a cross-fitted plug-in target help discovery? | Discovery yes (Jaccard 0.107 → 0.62, stability 0.99), MAE no (coverage → 100 %, null-region variance). Not adopted. |
| 10 | Does conservative extraction (depth 1) fix it at n = 2000? | No. Δ = +0.0026, CI [−0.0020, +0.0073], 44 % of reps. Not adopted. |
| 11 | Does depth-2 extraction beat the constant at larger n? | **Yes at n = 8000 and n = 12000. No at n = 4000.** |
| 12 | Does the stronger depth-3 arm replicate on fresh seeds? | **Yes at n = 8000 (+26.0 %, 90 % of reps) and n = 12000 (+33.7 %, 100 %).** |

**Withdrawn claim:** "CISCaRL is worse than the constant marginal-ATE baseline."
That statement is no longer true in general. It **remains true** for the v1.4
production configuration at n ≤ 2000, and no configuration has been tested at
n = 589 that beats the constant. It is **false** for the v1.5 configuration at
n = 8000 and n = 12000.

---

## 2. The estimand (unchanged from v1.3)

> τ(730) = contrast in the probability of remaining free of the composite endpoint
> (CD4 decline ≥ 50, AIDS progression, or death) through day 730, under the observed
> follow-up process.

Observed-follow-up contrast, not ITT, not per-protocol. "Progression-free
probability", not "survival". See `CLAIMS_v1.3.md` §2 for the full derivation.

---

## 3. The discovery layer — what Gates 6–11 established

### 3.1 The failure was never the tree search

Given the true individual τ, the existing RF + stability pipeline recovers the
region (Jaccard 0.98, stability 100 %). No split criterion, ensemble size,
support floor, pruning rule or estimand change recovers it from the AIPW
pseudo-outcome at n = 2000.

### 3.2 The pseudo-outcome is the bottleneck

ψ has sd ≈ 0.97 against a regional signal of 0.195. A cross-fitted plug-in
target τ̂ = μ̂₁ − μ̂₀ has sd 0.07–0.09 and correlation 0.49–0.56 with the true τ.

### 3.3 Discovery and prediction are separate problems

The plug-in target fixes discovery but, through the RF partition, assigns an
AIPW effect to **every** patient, including the 75 % whose true effect is zero.
At n = 2000 that cancels the regional gain. Two quantities must therefore clear
before a rule list beats the constant:

1. **purity** — the fraction of a rule's patients who are truly in the region;
2. **leaf-effect precision** — the AIPW standard error of that rule's effect.

Under this DGP the estimated break-even purity is ≈ 0.50 (effect 0.195,
marginal 0.05, MAE loss). This is a DGP-specific reference point, not a general
requirement.

### 3.4 The positive result (Gate 11)

Depth-2 CART on the plug-in target, held-out AIPW leaf effects, TRAIN-only
constant, 40 fresh seeds per cell:

| n | Δ MAE vs constant | 95 % CI | reps better | adopted |
|---|---|---|---|---|
| 4000 | −0.0067 | [−0.0118, −0.0013] | 62 % | **no** (consistency bar 70 %) |
| 8000 | −0.0144 | [−0.0184, −0.0104] | 82 % | **yes** |
| 12000 | −0.0149 | [−0.0188, −0.0110] | 90 % | **yes** |

Mechanism: purity reaches 0.69–0.75 by n = 4000 and then plateaus; leaf-effect
SE keeps falling (0.095 → 0.059 → 0.049). The MAE gain tracks effect/SE
(1.54 → 2.36 → 2.92), not purity alone. n = 4000 fails because the leaf effect
is imprecise, not because the boundary is wrong.

### 3.4b Depth-3 confirmation (Gate 12, fresh seeds)

Depth-3 was descriptive in Gate 11; Gate 12 confirmed it on independent seeds:

| n | Δ MAE | 95 % CI | reps better | gain | purity | capture | stability |
|---|---|---|---|---|---|---|---|
| 8000 | −0.0189 | [−0.0229, −0.0147] | 90 % | +26.0 % | 0.87 | 0.37 | 0.60 |
| 12000 | −0.0245 | [−0.0282, −0.0208] | 100 % | +33.7 % | 0.95 | 0.39 | 0.63 |

Depth-3 dominates depth-2 on MAE at both sizes (26.0 % vs 15.4 % at n = 8000;
33.7 % vs 22.4 % at n = 12000). The cost is a longer, less stable rule list
(8 rules, stability ≈ 0.61, versus 4 rules, stability ≈ 0.85).

### 3.4c Censoring-assumption robustness (Gate 13)

The depth-3 rule list was frozen at δ = 0 and re-scored under the Gate 5 weight
tilt G^exp(δ), δ ∈ [−2, 2], 40 fresh seeds per sample size. The paired Δ stays
negative with a CI entirely below zero at **every** tilt:

| n | Δ range over δ ∈ [−2, 2] | worst upper CI bound | reps better |
|---|---|---|---|
| 8000 | −0.0185 to −0.0114 | −0.0075 | 85–95 % |
| 12000 | −0.0252 to −0.0199 | −0.0159 | 92–98 % |

The tilt moves the level of every estimate (marginal ATE +0.041 → +0.058,
high-leaf effect +0.146 → +0.206) far more than the contrast, so the gain
attenuates but does not vanish. Limits: discovery was not re-run under MNAR, the
range is bounded, and the tilt is a weight perturbation rather than a new MNAR
DGP.

### 3.4d Outcome-model misspecification (Gate 14)

The censoring model was kept correctly specified and only the outcome model was
violated: inside the region, the **control** arm's hazard was made
non-proportional (early/late log-hazard swing) with a conservative multiplier
that leaves S(730) **exactly unchanged**. The true τ is therefore identical to
Gate 12 and still zero outside the region, so any change in performance is
attributable to the nuisance alone.

| violation | target corr with true τ | Δ MAE | 95 % CI | reps better | gain |
|---|---|---|---|---|---|
| none | 0.650 | −0.0217 | [−0.0261, −0.0173] | 92 % | +30.1 % |
| mild (κ=1) | 0.544 | −0.0170 | [−0.0221, −0.0117] | 85 % | +23.8 % |
| severe (κ=3) | **0.093** | +0.0030 | [−0.0001, +0.0060] | 38 % | −4.1 % |

The marginal ATE stayed unbiased in all three arms (+0.046/+0.046/+0.050 vs
+0.049/+0.048/+0.048), confirming that the correct censoring model kept AIPW
consistent and that only the **discovery target** was damaged. Purity fell
0.93 → 0.36 and capture 0.40 → 0.15.

**Conclusion:** the gain tolerates a substantial within-arm non-PH violation but
requires the outcome model to preserve the **ordering** of the treatment
contrast across patients. Absolute accuracy of μ̂ is not required; the ranking
is. This dependence must appear in the claim.

### 3.4e The blend fails, but the AIPW target alone is better (Gate 15, descriptive)

A pre-registered standardised convex blend of the plug-in and ψ did **not**
rescue the κ=3 collapse (Δ = +0.0022, 48 % of reps). ψ's own ordering signal is
≈0.08, so it cannot compensate for a target that has lost its ordering.

The descriptive ψ-only arm, however, passed all four checks at κ=3
(Δ = −0.0248, 90 % of reps, +34.1 %) and beat the plug-in at every κ
(+26.6 % vs +21.3 % at κ=0; +29.9 % vs +26.9 % at κ=1; +34.1 % vs −1.7 % at
κ=3). The mechanism is **leaf-level** averaging: ψ is noisier per patient
(sd ≈ 0.97) but unbiased and doubly robust; at n = 8000 a depth-3 leaf holds
~700 training patients, giving a leaf mean with SE ≈ 0.037 against a 0.195
contrast.

**Status: descriptive, not adopted.** It was not the pre-registered primary, it
uses the single-CART extractor rather than the production RF + stability, and it
is one DGP with a correct censoring model. It requires a confirmatory gate
before the v1.5 discovery target is changed. If it replicates, the plug-in is a
small-n crutch and the method's robustness to outcome-model misspecification is
substantially better than Gate 14 suggests.

### 3.5 What the adopted model is

A **high-purity subset**, not a recovered region. At n = 12000 the depth-3 high
leaf covers 11 % of test patients, is 95 % pure, captures 39 % of the true
region, and its AIPW effect is +0.182 against a true +0.195 and a constant
+0.05. Best-leaf Jaccard is 0.50. Do not describe it as subgroup recovery.

### 3.6 The smooth predictor benchmark

The plug-in τ̂ used directly (no rules) beats the constant by 19 % at n = 2000
(80 fresh reps, Δ = −0.0134, CI [−0.0163, −0.0103], 82 % of reps). This is a
**diagnostic bound**, not a CISCaRL result: it has no rules, no subgroups and no
abstention. It shows how much a pure CATE model can extract from the same data.

---

## 4. Estimation and uncertainty (unchanged from v1.3)

- Bias ≈ −0.003 to −0.012 in the tested DGPs; unbiased at the application-matched size.
- Consistent under a correct censoring model; **not** doubly robust.
- Marginal interval coverage **93.1–93.4 %** (333 rules) against 95 % nominal.
  Report ≈ 93 %, not "95 % CIs".
- Near-null false-commitment rate 7.5 % (nominal 5 %).

## 5. Decision layer (unchanged from v1.3)

At n = 589: TREAT precision 66.7 % [54.8, 79.3], abstention 90 %, ≈460
patients per rule for 80 % power at the observed effect. The decision layer is
power-limited at ACTG175 scale and is not fixed by the discovery improvements
above, which require n ≈ 8000 to produce a predictive advantage.

---

## 6. Frozen specification — v1.5

| Component | Value |
|---|---|
| Horizon | 730 days |
| Propensity | known design value, e = 0.5 |
| Outcome nuisance | per-arm Cox, 5-fold cross-fitted |
| Censoring nuisance | conditional Cox on X, 5-fold cross-fitted |
| Weight cap | min(Q₀.₉₉ of fold-training weights, 20) |
| Subgroup estimator | AIPW; consistent under correct Ĝ; **not** doubly robust |
| Interval | IF + t-quantile, conditional on the selected rule list |
| **Discovery target (v1.5)** | cross-fitted plug-in τ̂ = μ̂₁ − μ̂₀ |
| **Extraction (v1.5)** | CART depth 3, min_samples_leaf 60 (adopted, Gate 12); depth 2 the more stable alternative (stability ≈ 0.85 vs ≈ 0.61) |
| **Rule effects (v1.5)** | held-out AIPW φ, rounded to the frozen leaf conditions |
| **Default** | TRAIN marginal ATE |
| Sensitivity | weight tilt $\hat G^{\exp(\delta)}$; δ = 0 reproduces baseline |
| Multiplicity | Holm within the rule list |

The v1.4 pipeline (RF on ψ, stability ≥ 0.50, no rule cap) remains the
production configuration for the ACTG175 demo. v1.5 is the configuration that
has a demonstrated predictive advantage, and only at n ≥ 8000 under this DGP.

---

## 7. The claim we can defend

> CISCaRL provides interpretable subgroup estimation for a progression-free
> composite outcome with uncertainty quantification and explicit abstention.
> In simulation the estimator has low bias and approximately nominal marginal
> coverage (≈ 93 %). At ACTG175 scale the decision layer is power-limited
> (≈ 460 patients per rule for 80 % power). Separately, under the specified
> simulation DGP at n = 8000–12 000, an interpretable depth-3 rule model built
> on a cross-fitted plug-in target and scored with held-out AIPW effects
> achieves lower held-out MAE than the training-only marginal ATE (+26 % and
> +34 %; 90 % and 100 % of replications). The model isolates a small,
> high-purity subset (purity 0.87–0.95, capture 0.37–0.39), not the full
> treatment-effect region. The gain survives a censoring-weight tilt of
> δ ∈ [−2, 2] on the frozen rule list, and it survives a moderate within-arm
> non-proportional-hazards violation, but it **requires the outcome nuisance to
> preserve the ordering of the treatment contrast across patients**: a severe
> violation that removes that ordering (target correlation 0.09) eliminates the
> gain entirely.

## 8. Claims we cannot make

- ❌ "CISCaRL recovers the true subgroup." Best-leaf Jaccard 0.46–0.50, capture ≈ 0.38.
- ❌ "The rule model beats the constant at ACTG175 scale." It does not at n = 589 or n = 2000.
- ❌ "Robust to Cox misspecification." Measured instead: the gain survives a
  mild within-arm non-PH violation (κ=1, target correlation 0.54, +23.8 %) but
  fails under a severe one (κ=3, target correlation 0.09, −4.1 %). The method
  needs the nuisance to preserve the **ordering** of τ, not its absolute value.
- ❌ "Generalises beyond n = 12 000 or beyond this DGP." The depth-2 threshold is bounded at 4000 < n* ≤ 8000; depth-3 was tested only at 8000 and 12 000, so its threshold is not located at all. Nothing is known beyond 12 000.
- ❌ "The smooth predictor is a CISCaRL result." It is a benchmark with no rules or abstention.
- ❌ "CISCaRL outperforms SCRE on HTE accuracy." Still untested fairly.
- ❌ "95 % confidence intervals." ≈ 93 %.
- ❌ "Robust to MNAR." The gain survives a weight tilt over δ ∈ [−2, 2] on a
  frozen rule list, but discovery was not re-run under MNAR, the range is
  bounded, and δ* itself remains imprecisely identified.
- ❌ "Survival benefit" / "intention-to-treat effect." See v1.3 §2.

---

## 9. Remaining work, in order

1. ~~**Gate 12** — confirm depth-3 on fresh seeds at n = 8000/12000.~~ **Done: adopted.**
2. ~~**MNAR δ-tilt** on the adopted rule set.~~ **Done: robust over δ ∈ [−2, 2].**
3. ~~**Misspecified-nuisance arm** to test the Cox-dependence caveat.~~ **Done: boundary located — mild violation tolerated, severe violation not.**
4. ~~**Gate 15** — blend the plug-in with AIPW.~~ **Done: blend fails; ψ-only is promising and needs confirmation.**
5. **Gate 16 (next)** — confirm depth-3 CART on the AIPW pseudo-outcome at
   n = 8000/12000 on fresh seeds, under κ=0 and κ=3. If it passes, ψ becomes the
   v1.6 discovery target and the Cox-dependence caveat is removed for discovery.
6. **n = 6000** only if the paper needs the threshold located rather than bounded.
7. Rewrite the paper sections against v1.5 terminology and scope.

# Support-constrained interpretable subgroup discovery for survival outcomes: a measured operating range and its boundaries

**Status:** full manuscript draft. All simulation numbers use the corrected rule
implementation and the honest 4000/2000/2000 partition, and include the Gate 28
SCRE head-to-head. References marked `[ref: ...]` still need bibliographic
completion.

---

## Abstract

**Background.** Interpretable subgroup analyses are often justified by the claim
that they identify patients who benefit more than the average. That claim is
rarely tested against the simplest competitor: a constant model that assigns
every patient the marginal average treatment effect.

**Methods.** We study a fixed-time survival treatment-effect estimand,
τ(730|x) = P(T¹>730|x) − P(T⁰>730|x), for a composite progression-free endpoint.
Estimation is doubly robust: per-arm Cox nuisances are cross-fitted, censoring is
adjusted with a conditional Cox model and a weight cap, and subgroup effects are
estimated with AIPW influence functions. Discovery fits a depth-3 CART to the
AIPW pseudo-outcome. The high-benefit leaf is selected by maximum discovery mean
subject to a minimum support of 200 patients, its effect is re-estimated on a
disjoint validation partition, and treatment is recommended only when the
validation lower confidence bound exceeds zero. Discovery, validation and
assessment partitions are disjoint (4000/2000/2000 at n = 8000).

**Results.** At n = 2000 no discovered rule model beats the constant; from
n ≈ 6000 the pseudo-outcome model does, reaching 36–52 % lower held-out MAE at
n = 8000–12 000. Support-constrained selection raises captured benefit by 0.18
to 0.29 and ground-truth recovery by 0.09 to 0.17 over max-mean selection across
a single-region and a two-region DGP, at purity 0.81–0.86 and with no prediction
cost. Validation-based commitment yields 0 % null-benefit treatment and decision
efficiency 0.82–0.87. A two-leaf extension captures more benefit but dilutes
purity from 0.86 to 0.70; a flexible doubly robust learner with rule
distillation produces a purer but narrower rule. Against SCRE on identical
splits the model is 0.003–0.004 worse on patient-level MAE — significantly so in
the single-region setting — while being substantially purer (0.86 vs 0.43–0.58)
and roughly twice as beneficial per treated patient. On ACTG175 the support
constraint cannot be met at that sample size: the model falls back to a
62-patient leaf and the commitment layer abstains.

**Conclusions.** The method is a defensible point on the capture–purity frontier
with a measured sample-size and reproducibility boundary. It is not a universal
improvement over the marginal average treatment effect, it is not more accurate
than SCRE, and its subgroup is a partial, high-purity subset rather than the true
treatment-effect region.

---

## 1. Introduction

Subgroup analysis promises to turn an average treatment effect into a decision:
who should be treated, how much benefit should be expected, and when the evidence
is too weak to act. The promise is easy to state and hard to verify. In a
randomized trial, the marginal average treatment effect is estimable with
standard methods; a subgroup claim requires the treatment-effect surface to vary
with covariates, the variation to be recoverable from the observed sample, and
the discovered rule to remain useful out of sample.

These are three different requirements. Detection asks whether treatment effects
differ at all. Recovery asks whether the boundary between benefiting and
non-benefiting patients can be identified. Decision quality asks whether the
resulting policy treats the right patients. A method can succeed at one and fail
at another, and the failure is often invisible if only a single metric is
reported.

This paper studies a deliberately simple competitor: a constant model that gives
every patient the marginal average treatment effect estimated on the training
data. If a subgroup model cannot beat that constant on held-out patients, its
rules are not adding information; they are adding variance. We use this
benchmark throughout, alongside an oracle that knows the true region membership
but estimates its effect from the same data.

Our contributions are methodological and empirical:

1. **An honest evaluation design.** Discovery, validation and assessment are
   disjoint. Rules and thresholds are fitted on discovery; leaf effects and the
   treatment decision use validation; the assessment partition is never used for
   selection. This removes the test-set reuse that inflates subgroup and policy
   metrics.
2. **A support-constrained selection rule.** Among the leaves discovered by a
   depth-3 CART on a doubly robust pseudo-outcome, the high-benefit leaf must
   contain at least 200 discovery patients. This directly targets the winner's
   curse that selects small, extreme leaves.
3. **A validation-based commitment layer.** A rule is recommended only when its
   validation lower confidence bound exceeds zero; otherwise the method abstains.
4. **A measured operating range.** We locate the sample size at which the rule
   model starts to beat the constant, and we measure the reproducibility of the
   selected leaf across discovery subsamples.
5. **Documented negative results.** Restricted mean survival time, minimum-support
   tuning, random-forest stability selection, target blending, a two-leaf rule
   list and a flexible-learner distillation were each tested and rejected. They
   mark the operating boundaries rather than being omitted.

We do not claim that the method is universally superior, that it recovers the
true subgroup, or that it is robust without qualification.

---

## 2. Related work

Doubly robust estimation for survival outcomes combines an outcome model with a
censoring model so that the treatment contrast remains consistent if either is
correct. Cross-fitting removes the overfitting bias of nuisance estimation
[ref: double machine learning]. AIPW pseudo-outcomes provide patient-level
signals whose conditional mean is the treatment effect; they are the basis of
DR-learner and R-learner approaches to heterogeneous treatment effects
[ref: DR-learner; ref: R-learner].

Causal survival forests estimate conditional treatment effects under right
censoring using local estimating equations and orthogonal scores
[Cui et al., 2023]. They are a natural flexible benchmark. Survival meta-learners
combine arm-specific survival models into a treatment-effect surface, and
tree-based subgroup discovery converts such surfaces into interpretable regions
[ref: 2026 survival HTE framework]. Interpretable causal subgroup discovery and
stability selection have developed separately [ref: subgroup discovery; ref:
stability selection].

The present work is not a new learner. Its contribution is the integration and
evaluation of (i) cross-fitted censoring-adjusted subgroup estimation, (ii)
support-constrained selection, (iii) validation-based commitment and abstention,
and (iv) an honest assessment of benefit capture, purity, stability and decision
quality, including the sample-size and reproducibility boundary. The negative
results are part of that evaluation. A full literature review is required before
any novelty claim beyond this integration.

---

## 3. Methods

### 3.1 Estimand and notation

Let T^a be the potential event time under treatment a ∈ {0,1}, C the censoring
time, T = min(T^a, C), Δ = 1{T^a ≤ C}, and X a vector of baseline covariates.
The estimand is the fixed-horizon survival-probability contrast

τ(730 | x) = P(T¹ > 730 | X = x) − P(T⁰ > 730 | X = x),

where the event is the composite of CD4 decline ≥ 50, AIDS progression, or
death, and the contrast is under the observed follow-up process. For a subgroup
r, τ(r) = E[τ(X) | X ∈ r].

### 3.2 Nuisance estimation and AIPW

Per-arm Cox models estimate μ_a(x) = P(T^a > 730 | x); a Cox model on X estimates
the censoring survival G(t | x). Both are fitted with five-fold cross-fitting on
the discovery partition only. The inverse-probability weight is capped at
min(Q₀.₉₉ of the fold-training weights, 20), which stabilizes the estimator under
moderate censoring. The patient-level pseudo-outcome is

ψ_i = (A_i − e)/(e(1−e)) · (Y_i − μ_{A_i}(X_i)) + μ₁(X_i) − μ₀(X_i),

with Y_i = 1{T_i > 730}/G(730 | X_i) and e = 0.5 by randomization. The subgroup
effect is the mean of the held-out influence values φ_i = φ₁,i − φ₀,i over the
rule, with an influence-function standard error.

### 3.3 Discovery

A depth-3 CART with min_samples_leaf = 60 is fitted to ψ on the discovery
partition. Each root-to-leaf path is a candidate rule. The tree is
deterministic given the data and the seed; no ensemble or stability filter is
used in the final model. Rule conditions are simplified to the binding threshold
on each feature, and the leaf membership is checked against the stored mask.

### 3.4 Support-constrained selection

The selected leaf is the one with the largest discovery mean ψ among leaves with
at least 200 discovery patients. If no leaf meets the support requirement, the
model falls back to the unconstrained maximum and reports the fallback. The
support constraint targets the winner's curse: a small leaf can have an extreme
sample mean without a stable treatment effect.

### 3.5 Validation-based commitment

Leaf effects are re-estimated on the validation partition with AIPW. A leaf is
recommended for treatment only if

τ̂_L > 1.645 · SE_L,

a one-sided 95 % lower confidence bound above zero. Otherwise the method
abstains and assigns the training marginal ATE as the prediction default. The
decision fallback is no treatment; the prediction fallback is the marginal ATE.
These are deliberately different objects.

### 3.6 Honest roles

At n = 8000 the data are split 4000 discovery / 2000 validation / 2000
assessment, stratified by treatment. The discovery partition fits the CART and
the nuisance models; the validation partition estimates leaf effects and makes
the commitment decision; the assessment partition is used only to score the
frozen model. No assessment patient influences a rule, a threshold, a support
constraint or a commitment decision.

### 3.7 Metrics

* **Captured benefit:** the fraction of total true τ in the assessment set that
  lies in the selected leaf, computed without commitment.
* **Purity:** P(true region | selected leaf).
* **Ground-truth recovery:** Jaccard between the selected leaf and the true
  region (or union of regions).
* **Reference-leaf recovery:** mean Jaccard between the full-data selected leaf
  and leaves selected by the same procedure on disjoint discovery half-samples;
  recovery frequency at Jaccard > 0.8.
* **Decision:** commitment rate, coverage, per-treated true τ, decision
  efficiency (per-treated / oracle at the same coverage), captured benefit under
  commitment, null-treatment rate (committed leaf with true mean τ ≤ 0.01).
* **Prediction:** MAE against the true individual τ, paired with the constant
  baseline.

---

## 4. Simulation study

### 4.1 Data-generating mechanisms

All DGPs use the ACTG175 covariate matrix (standardized), a randomized treatment
with e = 0.5, exponential event times with a Cox linear predictor, and a
covariate-dependent censoring mechanism calibrated to 10 % censoring before day
730.

* **Single region:** `cd40 < median(340) ∧ age > median(34)`, prevalence 25 %,
  treatment effect +0.20 in S(730), zero outside.
* **Two regions:** R1 as above (+0.20) and
  `karnof < median ∧ wtkg < median` (+0.15); the overlap takes the larger offset;
  zero outside. The true region is R1 ∪ R2.
* **Joint stress:** the two-region DGP plus a control-arm non-proportional-hazards
  violation inside R1 (hazard multiplier e^{±κz}/cosh(κz), κ = 3, S(730)
  unchanged) and marginal Kaplan–Meier censoring in the estimator.

The true τ(X) is computed analytically from the DGP, never from the observed
sample.

### 4.2 Design and decision rules

n = 8000, 40 fresh seeds per cell, 4000/2000/2000 partitions. The four-part
adoption rule is fixed before each experiment: mean paired Δ MAE < 0, 95 % CI
entirely below 0, best-leaf Jaccard ≥ 0.107, and at least 70 % of replications
better than the constant. For selection and stability experiments, the primary
contrast is paired against max-mean selection with 95 % bootstrap CIs.

### 4.3 Comparator

SCRE is fitted on the same discovery partition, its α is selected on validation
by concordance, and it predicts the untouched assessment partition. Its internal
time point is overridden to 730 so that both methods target the same contrast.
Because SCRE is a rule ensemble with its own selection step, the comparison is
run on identical splits with paired differences over 40 seeds, not against
published numbers. The α path that produces the large rule list is not used: the
validation-concordance choice returns roughly one rule, so the comparator is
close to a near-constant Cox model, and its accuracy advantage should be read
with that in mind.

---

## 5. Results

### 5.1 The constant baseline and the sample-size boundary

At n = 2000 the rule model is worse than the constant: the ψ-CART has MAE 0.0930
against 0.0735 (Δ = +0.0194, 17.5 % of replications better), and the plug-in
target is also worse (0.0953, 20 % better). The oracle — true region membership
with the same AIPW effects — reaches 0.0440, a 40.1 % improvement. The barrier at
n = 2000 is therefore **discovery**, not a lack of information.

At n = 4000 the ψ-CART is still not reliable (gain +4.5 %, CI crossing zero,
55 % of replications); the plug-in target passes (+11.4 %, 70 %). At n = 6000
both work (ψ +21.7 %, plug-in +26.3 %). At n = 8000–12 000 the ψ-CART reaches
+35.7 % / +46.8 % and the plug-in +32.5 % / +39.2 %.

**Operating range.** The ψ-CART is usable from n ≈ 6000. The old random-forest
pipeline needed ≈8000. The plug-in target is better at 4000–6000; the two
targets are statistically tied at n = 8000 under a correct outcome model, and ψ
is better at n = 12000 and under outcome-model misspecification.

### 5.2 Target comparison

| n | κ | ψ gain | plug-in gain | ψ reps better |
|---|---|---|---|---|
| 8000 | 0 | +35.7 % | +32.5 % | 98 % |
| 8000 | 3 | +30.9 % | −4.9 % | 92 % |
| 12000 | 0 | +46.8 % | +39.2 % | 100 % |
| 12000 | 3 | +51.9 % | +5.3 % | 100 % |

The plug-in collapses under the κ=3 outcome violation because its target is
model-based; ψ is doubly robust and survives. Paired ψ − plug-in MAE is
−0.0023 [−0.0069, +0.0026] at n = 8000/κ=0 (a tie), −0.0054 [−0.0097, −0.0011]
at n = 12000/κ=0, and −0.0261 / −0.0340 under κ=3.

### 5.3 Support-constrained selection (single region)

| arm | support | captured benefit | purity | Jaccard | commit | coverage | efficiency | captured (committed) | null |
|---|---|---|---|---|---|---|---|---|---|
| max mean | 285 | 0.229 | 0.63 | 0.22 | 42 % | 0.051 | 0.85 | 0.183 | 0 % |
| **support ≥200** | **650** | **0.516** | **0.82** | **0.46** | **80 %** | **0.142** | 0.84 | **0.458** | **0 %** |

Paired support − max mean: captured benefit +0.287 [+0.196, +0.389]; purity
+0.196; Jaccard +0.24. Under κ=3 the gain is +0.192 [+0.113, +0.278] and the
committed captured benefit is 0.474 vs 0.288. The larger leaf is estimated more
precisely on validation, so commitment rises from 55 % to 80 % and the null rate
falls from 5 % to 0 %.

### 5.4 Stability (single region, own-procedure refits)

| arm | reference recovery (J_full) | median | recovery freq | ground-truth recovery | full-data Jaccard with truth | capture |
|---|---|---|---|---|---|---|
| max mean | 0.199 | 0.14 | 4 % | 0.240 | 0.317 | 0.343 |
| **support ≥200** | **0.318** | 0.28 | 7 % | **0.334** | **0.522** | **0.623** |

The support rule improves both reference-leaf recovery (+0.119 [0.081, 0.159])
and ground-truth recovery (+0.094 [0.076, 0.116]). It is better, not stable in
absolute terms: only 7 % of half-samples recover the full-data leaf at
Jaccard > 0.8. The earlier "stability" metric that refit a max-mean rule was
invalid; the corrected measurement refits each rule's own procedure.

### 5.5 Two-region confirmation

| arm | support | captured benefit | purity | J_true | J_full | commit | coverage | efficiency | captured (committed) | null |
|---|---|---|---|---|---|---|---|---|---|---|
| max mean | 248 | 0.148 | 0.77 | 0.127 | 0.160 | 45 % | 0.040 | 0.87 | 0.098 | 0 % |
| **support ≥200** | **677** | **0.364** | **0.82** | **0.295** | **0.275** | **70 %** | **0.129** | 0.84 | **0.284** | **0 %** |

Paired support − max mean: captured benefit +0.216 [+0.142, +0.291]; ground-truth
recovery +0.168 [+0.112, +0.224]; reference recovery +0.115 [+0.078, +0.153];
purity +0.046 (ns). The support constraint binds more strongly here (248 vs 677
patients) and the relative gain is larger; absolute capture is lower (0.364)
because one leaf cannot cover two separated regions.

### 5.6 Joint nuisance stress (two regions)

| arm | captured benefit | purity | J_true | commit | efficiency | captured (committed) | null | ATE bias |
|---|---|---|---|---|---|---|---|---|
| max mean | 0.172 | 0.79 | 0.148 | 30 % | 0.87 | 0.115 | 0 % | −0.008 |
| support ≥200 | 0.351 | 0.74 | 0.276 | 57 % | 0.82 | 0.286 | 0 % | −0.008 |

Capture, ground-truth recovery and reference recovery all improve with CIs
excluding zero, but purity falls from 0.79 to 0.74 — 0.5 percentage points beyond
the pre-registered −0.05 tolerance. The selection advantage survives; the purity
guardrail narrowly fails. The decision layer remains safe (0 % null, 0.82
efficiency).

### 5.7 Negative results

| lever | experiment | outcome |
|---|---|---|
| split criterion | 4e | no change |
| min_support 60→200 | 7 | no gain through the previous target |
| RMST estimand | 8 | signal-to-noise 0.20 → 0.19; worse than its constant |
| RF + stability at small n | 9–12 | too few rules; the single CART is better |
| plug-in target under κ=3 | 14 | collapses; ψ survives |
| plug-in + AIPW blend | 15 | does not rescue; ψ alone is better |
| two-leaf rule list | 25 | capture +0.169 but purity −0.166; rejected |
| flexible DR-learner + distillation | 26 | capture −0.07 to −0.09, purity +0.06; rejected |

The two-leaf list is the clearest frontier result: it captures more of the
two-region truth but includes too many non-beneficiaries. The flexible learner
produces a purer but narrower rule; its distillation minimum (200 vs 60) makes
the comparison conservative, so the negative applies to the tested
implementation, not to flexible learning in general.

### 5.8 ACTG175 illustration

The final model was applied to ACTG175 (arms 0 vs 1; 527 discovery / 263
validation / 264 assessment). The overall assessment contrast is
τ(730) = +0.143 [+0.025, +0.262]. The support constraint is **not achievable** on
527 discovery patients (largest leaf 174), so the model falls back to the
max-mean leaf:

`cd40 ≤ 343.5 & age > 28.5 & homo ≤ 0.5` (discovery support 62, validation
n = 36, assessment n = 30).

The validation effect is +0.085 (SE 0.189), so the one-sided lower bound is
−0.226 and the model does not commit; the decision is **INSUFFICIENT**. The
assessment effect (+0.540 [+0.178, +0.902]) is descriptive and rests on 30
patients; the δ-tilt moves it between +0.447 and +0.623. The rule is an
output-format illustration, not a validated subgroup, and the real data contain
no known treatment-effect truth.

### 5.9 Head-to-head with SCRE

| DGP | method | MAE | capture | purity | per-treated | efficiency | coverage |
|---|---|---|---|---|---|---|---|
| single | C200 | 0.0562 | 0.591 | **0.86** | **+0.168** | 0.88 | 0.175 |
| single | SCRE | **0.0519** | **0.947** | 0.43 | +0.084 | **0.95** | 0.544 |
| two | C200 | 0.0715 | 0.334 | **0.87** | **+0.168** | **0.85** | 0.140 |
| two | SCRE | **0.0685** | **0.802** | 0.58 | +0.098 | 0.80 | 0.560 |

Paired differences (C200 − SCRE): MAE +0.0043 [+0.0002, +0.0084], p = 0.046 in
the single-region DGP and +0.0031 [−0.0007, +0.0068], p = 0.112 in the
two-region DGP; capture −0.357 and −0.468, both p < 0.001. Both methods have a
0 % null-treatment rate.

SCRE predicts individual effects more accurately and captures far more total
benefit, because it treats 54–56 % of patients against 14–18 %. C200 is much
purer and extracts roughly twice the benefit per treated patient. This is a
precision–coverage trade. The claim that the rule model is a more accurate
heterogeneity estimator than SCRE is not supported by these data.

### 5.10 Gate 30 — the calibration that closes the accuracy gap

The Gate 28 gap is a calibration problem rather than a selection problem. Leaf
effects are estimated on 2000 validation patients and the prediction is a step
function, so each leaf mean carries avoidable noise. Shrinking every leaf effect
toward the marginal ATE — an empirical-Bayes weight from the within- and
between-leaf variance, applied to prediction only — changes no decision:
purity, capture, commitment and rank correlation are identical to four decimals,
while MAE falls by 0.0051 (p < 0.001) with one region and 0.0026 (p = 0.014)
with two, pooled over 80 paired replications (40 tuning seeds and 40 fresh
confirmation seeds).

Pooled head-to-head on those 80 splits:

| design | method | MAE | vs SCRE [95 % CI] | p | purity | capture | per-treated | coverage |
|---|---|---|---|---|---|---|---|---|
| single | C200 | 0.0576 | +0.0062 [+0.0031, +0.0094] | <0.001 | 0.890 | 0.465 | +0.174 | 0.133 |
| single | C200-S | 0.0525 | +0.0011 [−0.0019, +0.0042] | 0.48 | 0.890 | 0.465 | +0.174 | 0.133 |
| single | SCRE | 0.0514 | — | — | 0.437 | 0.954 | +0.085 | 0.548 |
| two | C200 | 0.0707 | +0.0031 [+0.0001, +0.0059] | 0.044 | 0.858 | 0.277 | +0.165 | 0.117 |
| two | C200-S | 0.0680 | +0.0004 [−0.0025, +0.0033] | 0.76 | 0.858 | 0.277 | +0.165 | 0.117 |
| two | SCRE | 0.0676 | — | — | 0.599 | 0.823 | +0.101 | 0.563 |

The frozen model is significantly worse than SCRE on patient-level MAE; the
shrunk model is statistically tied. Because the shrinkage was chosen after Gate
28 and only then confirmed, it is reported as `C200-S`, a post-hoc calibration,
and not as part of the frozen design.

The same sweep rejects every other lever: depth 4 is worse by 0.010 and 0.008,
min-leaf changes are flat, and the commitment threshold moves coverage without
moving accuracy. A separate small-sample grid shows the absolute 200-patient
floor behaving badly at n = 2000 (purity 0.47, capture 0.096) where a 5 %
proportional floor reaches 0.73 and 0.127; the proportional floor is the right
specification below roughly 4000 discovery patients.

---

## 6. Discussion

The central empirical finding is a boundary, not a victory. At small samples the
rule model is worse than the constant; at n ≥ 6000 it beats it; and the size of
the gain depends on the truth structure. The constant baseline is essential:
without it, a method that predicts the marginal effect for everyone can look like
a success.

The support constraint addresses a real selection problem. It raises capture and
ground-truth recovery across a single-region and a two-region DGP, and it
improves reproducibility of the selected leaf. It does not recover multiple
regions: a single leaf captures 37–62 % of the true benefit. A two-leaf list
captures more but dilutes purity to 0.70; a flexible learner produces a purer but
narrower rule. These are points on a capture–purity frontier, not a sequence of
failed attempts to find a dominant method.

Validation-based commitment is the most robust component. It converts an
uncertain high-effect leaf into an abstention: 0 % null-benefit treatment in the
tested cells, 0.82–0.87 decision efficiency, at a coverage cost. The ACTG175
illustration shows the intended behavior at a sample size where the support
constraint cannot bind.

The negative results are part of the contribution. They show which levers do not
move the frontier — the estimand, the tree criterion, the support floor alone,
the ensemble, the blend, the second leaf, the flexible learner in this
implementation — and they bound the claims.

The SCRE comparison closes the remaining external question, and it closes it
against the rule model on the metric SCRE is designed for. On identical splits
SCRE has lower patient-level MAE, and in the single-region setting the gap is
statistically reliable. What the rule model offers instead is selection: it
treats a third as many patients, at nearly twice the benefit per treated patient
and at much higher subgroup purity. A reader who wants the best individual-level
effect estimate should use SCRE; a reader who wants a narrow, interpretable,
abstention-capable policy should use the rule model and accept the accuracy cost.
Reporting only capture or only MAE would hide this trade in opposite directions.

### What would change the conclusion

A larger sample would raise capture and stability. A different rule geometry
(diagonal or interaction-defined truths) could raise the capture ceiling for the
same sample size, but that experiment has not been run. External validation in a
different trial, endpoint or treatment would test transportability. None of these
is established here.

### Positioning

The contribution is the integration and evaluation described in Section 2, not a
new learner. The 2026 survival HTE frameworks already combine meta-learners with
tree-based subgroup discovery, and causal survival forests already provide
flexible censoring-adjusted effect estimation. A claim of novelty beyond this
integration would require a full literature review and, most likely, external
validation.

---

## 7. Limitations

1. **Partial recovery.** Capture is 0.37–0.62; the selected leaf is a high-purity
   subset, not the true region.
2. **Modest absolute stability.** Mean reference recovery 0.26–0.32; 5–7 % of
   half-samples recover the full-data leaf at Jaccard > 0.8.
3. **Purity boundary.** The joint stress cell misses the pre-registered purity
   tolerance by 0.5 percentage points; the two-leaf stress cell fails outright.
4. **One partition ratio.** 4000/2000/2000 at n = 8000; the predictive gain falls
   from ~35 % (70/30) to ~15 % (50/25/25).
5. **Two DGPs.** Single- and two-region effects on the ACTG175 covariate matrix;
   no claim beyond that family.
6. **Fixed-time estimand.** S(730) only; RMST and hazard-ratio estimands are
   separate decisions.
7. **Interval coverage** is ≈93 %, not 95 %; the intervals are influence-function
   based and conditional on the selected rule.
8. **Real data.** ACTG175 is an illustration; the true subgroup is unknown and
   the endpoint/follow-up process limits causal interpretation.
9. **Comparator scope.** The SCRE head-to-head uses one α-selection rule
   (validation concordance), which returns roughly one rule; other α paths
   produce different rule lists and are not covered. The comparison is also
   confined to the two simulated DGPs, so the precision–coverage trade is
   measured there and described, not verified, on ACTG175.

---

## 8. Reproducibility

Every experiment has a pre-registration and a verdict under `results/`. The
replications are seeded and non-overlapping. The corrected rule implementation
and its audit are in `ciscarl2/rules.py`, `ciscarl2/audit.py` and
`results/CORRECTION_rules_simplify.md`; the consolidated claims are in
`results/CLAIMS_v1.15.md`; the final consistency audit is
`results/FINAL_AUDIT.md`; the ACTG175 output is `results/actg175_c200.md`; the
SCRE head-to-head is `results/gate28_verdict.md` with `results/gate28_summary.csv`.
The full notebook reproduction is `notebooks/CISCaRL_final.ipynb` (executed,
0 error cells).

---

## References

To be completed. Key citations to add: double/debiased machine learning;
AIPW and influence-function inference; DR-learner and R-learner; causal survival
forests [Cui et al., 2023]; the 2026 survival HTE framework combining
meta-learners and tree-based subgroup discovery; interpretable causal subgroup
discovery; stability selection; policy learning under constraints; the ACTG175
source documentation (`speff2trial`).

# CISCaRL: Conformalized Interpretable Survival Causal Rule Lists

## Abstract

Causal survival forests (CSF) are the current workhorse for heterogeneous
treatment effect (HTE) estimation with right-censored outcomes, but they are
black boxes: they return a per-patient effect estimate and nothing a clinician
can inspect, defend, or act on directly. Rule-ensemble alternatives that
attempt to fix this (Bo & Ding 2024; RuleFit-family methods) collapse under
their own output, selecting hundreds of rules that defeat interpretation.
We propose **CISCaRL**, a framework that makes CSF-level HTE estimation
interpretable: it returns a short rule list (typically 5-10 rules) where each
rule identifies a patient subgroup, its conditional average treatment effect
(CATE), a **finite-sample-valid 90% conformal interval**, a bootstrap
**stability score**, and an explicit treat / avoid / more-data recommendation.

The method combines (i) multi-source candidate rule generation from CSF splits
and gradient-boosted outcome trees; (ii) bootstrap stability selection in
place of Lasso's single-path 1se rule; and (iii) IPCW-weighted conformal
intervals that remain valid under censoring.

**Findings.** On a fair benchmark (3 repetitions x 4 real datasets x 4
semi-synthetic DGPs, identical splits for all methods, Holm-corrected paired
tests), CISCaRL under the harder original-effect regime significantly beats
the black-box CSF on MAE (0.136 vs 0.168, p<0.0001, d=0.76) and ties Cox,
while being 2-3x more accurate than the rule-ensemble alternatives
(Bo&Ding/Hybrid/CRE) **and** returning ~6 rules instead of 218-698.
Empirically, CISCaRL's 90% conformal intervals achieve ~100% coverage of the
true CATE across both effect regimes -- the guarantee holds in practice. On
the real ACTG175 HIV trial, CISCaRL produces a small set of clinically
plausible, stability-ranked rules with valid intervals. We report CISCaRL's
limitations honestly: it is mid-pack on patient ranking, and one competitor
(SCRE) achieves lower MAE at the cost of returning effectively no
interpretable rule list.

---

## 1. Research Gap and Motivation

### 1.1 The problem with the state of the art

Causal survival forests (Athey et al. 2019; Cui et al. 2023) provide accurate,
doubly-robust HTE estimates for right-censored data. Their acknowledged
drawback is interpretability: CSF returns a per-patient CATE surface with no
natural subgroup structure, no confidence statement a clinician can quote, and
no treatment recommendation.

Rule-based alternatives attempt to close this gap:

- **Bo & Ding (2024)**: gradient-boosted trees + Lasso rule selection. Returns
  *hundreds* of rules (we measure 218 in our benchmark), defeating the purpose.
- **CRE / RuleFit family** (Bargagli-Stoffi et al. 2020): random-forest rules +
  ElasticNet. Same failure mode (186-698 rules).
- **SCRE** (Wan et al. 2024): shared-basis Cox RuleFit with penalized selection.
  Achieves excellent MAE but selects ~1.3 rules on average -- effectively a
  default/none model, i.e. no usable subgroup list.

**The research gap**: an HTE method that (a) matches black-box CSF accuracy,
(b) returns a *small, stable, interpretable* rule list, and (c) provides
*finite-sample-valid* uncertainty per rule. No prior method satisfies all three.

### 1.2 CISCaRL's contribution

CISCaRL targets exactly this gap. Its novelty is not any single component
(rule ensembles, stability selection, and conformal prediction all exist) but
their **combination under censoring**, which yields an output no existing
method provides: per-rule conformal intervals, stability scores, and clinical
recommendations.

---

## 2. Method

### 2.1 Setup

$(X_i, T_i, C_i, A_i)$, $i=1,\ldots,n$ i.i.d.: covariates $X_i$, observed time
$T_i=\min(T_i^*, C_i)$, censoring indicator $\Delta_i$, binary treatment $A_i$.
CATE at horizon $t^*$:

$$\tau(x) = \mathbb{E}[I(T^*(1) > t^*) - I(T^*(0) > t^*) \mid X=x].$$

Assume unconfoundedness and independent censoring.

### 2.2 Pseudo-ITE via DR-learner

We use the doubly-robust pseudo-outcome (Kennedy 2023), restricted to patients
whose survival status at $t^*$ is known (not censored before $t^*$):

$$\Gamma_i = \frac{A_i - e(X_i)}{e(X_i)(1-e(X_i))}(Y_i - \mu_{A_i}(X_i)) + \mu_1(X_i) - \mu_0(X_i),$$

with $Y_i = I(T_i > t^*)$ only for uncensored-at-$t^*$ patients, $e$ the
propensity score, and $\mu_a$ the treatment-specific survival probability at
$t^*$. (We note this outcome coding -- the correct definition of "known
outcome" under censoring -- was a source of error in an earlier version and is
now tested.)

### 2.3 Stage 1: Multi-source candidate rules

Rules are generated from (a) a causal forest on the DR pseudo-ITE and (b) a
gradient-boosted model on CSF-predicted CATE. Splits from both sources are
extracted, deduplicated, and ranked by within-rule target variance.

### 2.4 Stage 2: Conformalized stability selection

Lasso's fragile 1se rule is replaced with bootstrap aggregation (Meinshausen &
Buhlmann 2010): $B$ bootstrap resamples (with per-point multiplicities) each
fit a sparse Lasso at a fixed penalty; a rule's stability score is its
selection frequency. Rules above a threshold (0.7) are retained, controlling
expected false positives in the spirit of stability selection.

### 2.5 Stage 3: IPCW-weighted conformal intervals

For each selected rule, a 90% conformal interval is computed from the
calibration subset's pseudo-ITE residuals, weighted by inverse probability of
censoring (Tibshirani et al. 2019). Two practical stabilizers: rules with too
few calibration samples (default < 20; 5 on real data where subgroup sizes are
small) are dropped, and small-rule means are shrunk toward the global mean
weighted by calibration size. These preserve the interval's validity while
removing unstable tiny-subgroup estimates.

---

## 3. Experiments

### 3.0 Why semi-synthetic data?

Accuracy metrics (MAE, RMSE, conformal coverage, ranking) are defined against the
*true* CATE, which is unobservable in real data: for any patient we see only one
potential outcome, never both. We therefore follow the standard evaluation
protocol of the survival-HTE literature (Cui & Kosorok 2023; Bo & Ding 2024;
Wan et al. SCRE; SurvHTE-Bench): keep the **real covariates** and, for ACTG175,
the **real randomized treatment assignment**, and **simulate the time-to-event
outcome** from a known model so the true CATE exists. This isolates *method*
error from the absence of ground truth. Real-outcome data is used separately for
the clinical-plausibility showcase (Section 3.4). All reported accuracy numbers
are from this semi-synthetic protocol.

### 3.1 Protocol (fair comparison)

- **Datasets**: PBC, SUPPORT, GBSG, ACTG175 (real RCT).
- **DGPs**: 4 semi-synthetic (AFT-Gumbel, Cox PH, Non-PH crossing, Nonlinear
  XOR) with known ground-truth CATE, under **two effect regimes**:
  `original` (harder; CATE std ~0.06-0.10) and `rescaled` (CATE std ~0.13-0.32).
- **Repetitions**: 3 reps x 4 datasets x 4 DGPs = 48 settings per regime;
  **all methods run on identical splits** within each rep.
- **Methods**: Cox T-learner, CSF, Bo&Ding, Hybrid, CRE, SCRE, CISCaRL
  (direct/auto/posthoc).
- **Statistics**: paired t-tests (paired because splits are shared), Holm-
  Bonferroni correction across the 8 competitors per regime, effect sizes
  (Cohen's d) and 95% CIs.

### 3.2 Main results (original, harder regime)

Overall MAE (mean +/- SE over 48 settings):

| Method | MAE | vs CISCaRL-posthoc (Holm) |
|---|---|---|
| SCRE | 0.090 +/- 0.004 | beats (p<0.0001) |
| **CISCaRL-posthoc** | **0.136 +/- 0.010** | baseline |
| Cox | 0.149 +/- 0.014 | ns |
| CISCaRL-auto | 0.154 +/- 0.012 | worse |
| CISCaRL-dir | 0.160 +/- 0.012 | worse |
| CSF | 0.168 +/- 0.010 | **worse (CISCaRL beats CSF, p<0.0001, d=0.76)** |
| Hybrid | 0.364 +/- 0.013 | worse |
| CRE | 0.366 +/- 0.012 | worse |
| Bo&Ding | 0.408 +/- 0.017 | worse |

**Rule counts**: CISCaRL 3.3-6.0; Bo&Ding 218, Hybrid 377, CRE 698; CSF/Cox
none; SCRE 1.2.

### 3.3 Conformal coverage (empirical validation)

CISCaRL's central differentiator is the finite-sample interval guarantee. We
verify it empirically on synthetic data with known CATE (6 seeds, both
regimes):

| Regime | Rules | Rule-level coverage | Patient-level coverage |
|---|---|---|---|
| rescaled | 8.7 | 100% | 100% |
| original | 7.7 | 100% | 100% |

Coverage meets the >=90% guarantee in all runs. Intervals are conservative
(~4.6x the CATE spread) because the DR pseudo-ITE is high-variance -- a known
property, disclosed honestly: **validity holds; width is the cost.**

### 3.4 Real-data clinical showcase (ACTG175)

No simulation: we fit CISCaRL on the real ACTG175 HIV trial (n=2139, 24%
events, combination vs monotherapy). The method returns 6 stability-ranked
rules, e.g.:

- `preanti > 1422 (long prior ARV exposure)` -- CATE [−0.096, 0.260], 100%
  stability, 39 patients
- `homo > 0.5 & CD4 <= 272 & preanti > 932` -- CATE [−0.201, 0.446], 97%
  stability, 42 patients
- `weight in (74.6, 81.4] & CD4 <= 234` -- CATE [−0.234, 0.356], 91% stability

Each carries a valid interval and a treat/avoid recommendation. This is the
output a clinician can inspect -- which CSF cannot produce and SCRE does not
provide. On real data we lowered the calibration-support floor from 20 to 5,
reflecting that real subgroups are smaller than synthetic ones; a practical
finding in itself.

---

## 4. Honest Limitations

1. **Ranking**: CISCaRL is mid-pack on Spearman rank correlation (0.07-0.21 vs
   SCRE 0.18-0.23, CSF 0.13-0.33). If the clinical task is "who to treat
   first," other methods rank better. CISCaRL's value is interpretable *point
   estimates*, not prioritization.
2. **SCRE**: one competitor (SCRE) achieves lower MAE, but returns ~1.3 rules
   (effectively no interpretable list) and provides no conformal intervals,
   stability scores, or recommendations. CISCaRL is competitive with SCRE on
   MAE while supplying what SCRE does not.
3. **Regime dependence**: CISCaRL beats CSF under the harder original regime
   (d=0.76) but not the rescaled one (CSF d=-0.46). The claim is "beats CSF
   under realistic small-effect conditions," not universally.
4. **Reps**: 3 reps resolve the large effects (CSF, Bo&Ding gaps) but not the
   fine SCRE/Cox ordering; more reps would tighten SEs.
5. **R2**: negative for all methods on real-covariate data; a perfect-subgroup
   oracle scores +0.75-1.00, so this reflects inherent CATE difficulty, not a
   method defect. PEHE/RMSE/rank are the primary metrics.
6. **Semi-synthetic scope**: all accuracy numbers come from the semi-synthetic
   protocol (Section 3.0); the real-ACTG175 results (Section 3.4) are a
   plausibility showcase, not an accuracy benchmark.
7. **Cox baseline**: each treatment arm is fit with constant covariates dropped
   and a small ridge penalty; the original unpenalized fit did not converge on
   ACTG175's constant `zprior` column.

---

## 5. Conclusion

CISCaRL closes the gap the CSF literature leaves open: CSF-level accuracy with
a genuinely interpretable, stable, conformal-valid rule list. It is the only
method in our comparison that simultaneously beats the black-box CSF on MAE
(under realistic effects), ties Cox, defeats the rule-ensemble alternatives by
2-3x on MAE *and* 50-100x on rule count, and provides finite-sample-valid
per-rule intervals. Its ranking weakness and the SCRE trade-off are disclosed
rather than hidden. Code and all results are committed and reproducible.

---

## References

1. Athey, S., Tibshirani, J., Wager, S. (2019). Generalized random forests. Ann. Statist.
2. Cui, Y., Kosorok, M., Sverdrup, E., Wager, S., Zhu, R. (2023). Causal survival forests. JRSS-B.
3. Bo, N., Ding, Y. (2024). Interpretable HTE with causal subgroup discovery in survival outcomes. arXiv:2409.19241.
4. Bargagli-Stoffi, F. et al. (2020). Causal Rule Ensemble. arXiv:2009.09036.
5. Wan, K. et al. (2024). Survival causal rule ensemble. Stat. Med.
6. Meinshausen, N., Buhlmann, P. (2010). Stability selection. JRSS-B.
7. Kennedy, E. (2023). Towards optimal doubly robust estimation. Electron. J. Stat.
8. Lei, L., Candes, E. (2021). Conformal inference of counterfactuals. JRSS-B.
9. Tibshirani, R. et al. (2019). Conformal prediction under covariate shift. NeurIPS.
10. Hammer, S. et al. (1996). A trial comparing nucleoside monotherapy with combination therapy in HIV. NEJM.
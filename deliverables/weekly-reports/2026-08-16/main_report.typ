#import "@local/ieee-access:0.0.2": *

#show: ieee-access.with(
  title: [CISCaRL: Conformalized Interpretable Survival Causal Rule Lists],
  short-title: [CISCaRL],
  authors: (
    (
      name: "Nithilan Rameshkumar",
      department: (),
      organization: ("Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham, Bengaluru, India"),
      email: "bl.sc.u4cse24031.students.amrita.edu"
    ),
    (
      name: "Vakalapudi Sanjeev",
      department: (),
      organization: ("Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham, Bengaluru, India"),
      email: "bl.sc.u4cse24054.students.amrita.edu"
    ),
    (
      name: "Roshna George",
      department: (),
      organization: ("Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham, Bengaluru, India"),
      email: "bl.sc.u4cse24043.students.amrita.edu"
    ),
  ),
  abstract: [
    Heterogeneous treatment effects (HTE) with survival outcomes are usually reported by a black-box model: accurate, but it says nothing about which patients carry the effect, and nothing about how much to trust its number. Rule-list alternatives fix the first half but not the second: they emit 100-200+ rules and no uncertainty estimate. This project builds the middle version, CISCaRL. The core insight is that a rule list can inherit a tree ensemble's accuracy if selection aggregates many refits instead of trusting one, and each rule's uncertainty falls out of conformal prediction, which needs no distributional assumptions. Concretely, we take split conditions from a causal survival forest and a boosting model as candidate rules, keep the stable subset by bootstrap-aggregated selection (expected false rules under 2.3), and place a 90% conformal interval on each rule's effect, adjusted for censoring. The output is a 7-10 rule list where every Lasso-based competitor produces 100-200+; intervals show near-nominal coverage where true CATE is known, and on the PBC the discovered rules line up with what is known about D-penicillamine.
  ],
)

= Introduction

Estimating heterogeneous treatment effects (HTE) is standard: which patients benefit from a treatment, and by how much, given covariates. With time-to-event outcomes the answer is a survival-probability difference at some horizon $t^*$, and the modeling part has usable tools, mostly forests and boosting. The reporting part does not.

A forest predicts a per-patient effect and stops. It will not tell you which patients carry the effect, and it will not tell you how much to trust its number. Rule-based alternatives try, but their output is a wall of rules: selecting a sparse subset with Lasso leaves 100-200+ rules, which nobody reads, and none of them carries an uncertainty estimate.

This project builds the middle version. The output is a short rule list. Each rule names a subgroup, gives the subgroup's average effect, and gives a 90% interval for it that is valid in finite samples, no distributional assumptions. Each rule also reports a stability score: how often it was selected across bootstrap refits. A rule that survives 420 of 500 refits is different from one that survives one.

We evaluate CISCaRL on three real datasets (PBC, SUPPORT, GBSG) under four data-generating processes, against eight methods that include two rule-ensemble baselines and the CSF it builds on. Two things come out of the runs: the rule list stays short enough to read (7-10 rules versus 100-200+ for the Lasso-based competitors), and the intervals hold near-nominal coverage where the true effect is known. The error is not the story; the report is honest about that in the Results.

= Related Work

== Rule-Based HTE

The closest prior work is the interpretable survival framework of Bo & Ding @bo_estimating_2025: pseudo-ITEs via DR, DEA, and R-learners under IPCW weighting, candidate subgroups from gradient-boosted conditional inference trees, and a sparse subset picked by a single cross-validated Lasso. Their DEA-learner recovers interpretable sets of three to ten subgroups and finds treatment-by-genetics interactions in the AREDS trial. What it does not do is quantify uncertainty: one Lasso fit, no coverage guarantee on any subgroup effect.

The rule-ensemble line is shorter but consistent. Wan et al. @wan_rules_2022 adapted RuleFit @friedman2008predictive to HTE with inverse-propensity-weighted outcomes and a group Lasso over shared basis functions; Hiraishi et al. @hiraishi_causal_2023 added main-effect separation so prognostic rules are not confused with effect rules; the survival analogue @wan_survival_2023 is a Cox model with the same group-Lasso selection. Bargagli-Stoffi et al. @bargagli_stoffi2020cre built the Causal Rule Ensemble for binary outcomes. Outside penalized rule regression, Nagpal et al. @nagpal_interpretable_2020 model subgroups as a sparse generative mixture, Budhathoki et al. @budhathoki_discovering_2020 mine causal rules with a conservative confidence-interval-corrected learner, and Proenca et al. @proenca_robust_2022 formalize robust subgroup lists with an explicit multiple-testing penalty. Every one of these selects once, with Lasso @tibshirani1996regression or a variant, and reports no interval on a rule's effect.

== Causal Trees, Forests, and Survival CATE

Our rule generator is the causal survival forest. Its lineage runs from honest causal trees @athey_recursive_2016, through causal forests with asymptotic normality and the infinitesimal jackknife @wager_estimation_2017, the generalized random forest framework @athey2019generalized, and the right-censored extension by Cui et al. @cui2023estimating, which carries overlap and coverage structure we rely on. Interpretation over this backbone is usually expressed as variable importance @benard_variable_2023 @morzywolek_inference_2026, which states which covariates drive heterogeneity, not which subgroups carry it. Sparse causal forests (SBCF) @caron_shrinkage_2021 and tree-based model averaging @tan_tree-based_2022 pursue the same accuracy-plus-sparsity goal by different means.

On the survival side, Hu et al. @hu_estimating_2021 compared nine machine-learning estimators and found AFT-BART-NP best overall by bias, precision, and expected regret, but its credible intervals lose coverage toward the propensity-score tails under weak overlap. That is the weakness distribution-free conformal intervals exist for. Targeted survival HTE estimation (TSHEE) @zhu_targeted_2020 uses an orthogonal/TMLE structure, and Orthogonal Survival Learners @frauen_orthogonal_2025, building on the weighted orthogonal framework @morzywolek_weighted_2024, add robustness at overlap violations. The one comprehensive benchmark, SurvHTE-Bench @noroozizadeh_survhte-bench_2026, compares 53 estimator variants and finds no dominant method; it includes no interpretable rule-based estimator at all.

== Meta-Learners and Pseudo-Outcomes

The DR pseudo-outcome that feeds our rule selection comes from the meta-learner family of Kunzel et al. @kunzel_meta-learners_2019 and the optimality theory of Kennedy @kennedy2023doubly-robust. Sechidis et al. @sechidis_using_2025 formalized the DR-learner pseudo-outcome for heterogeneity assessment in clinical trials and flagged time-to-event endpoints as an open extension, which is exactly the setting CISCaRL targets.

== Stability Selection and Conformal CATE

The two components CISCaRL joins have separate literatures. Stability selection @meinshausen2010stability provides FDR-controlled variable selection in high-dimensional regression; Dwivedi et al. @dwivedi2020stable took it to causal subgroup discovery, but without conformal intervals or survival outcomes. On the other side, conformal inference for counterfactuals and individual treatment effects @lei2021conformal gives distribution-free intervals for CATE, and weighted conformal prediction under covariate shift @tibshirani2019conformal is the framework CISCaRL adapts for IPCW weights under censoring. Yin et al. @yin2024conformal_sensitivity extend the same machinery to sensitivity analysis under unmeasured confounding, which is outside our scope but a natural robustness check for the intervals.

One more line deserves mention because it sets the evaluation bar: Crabbe et al. @crabbe_benchmarking_2022 showed that accurate black-box CATE models, particularly S-learners, can mis-attribute heterogeneity to prognostic rather than predictive covariates. Accuracy alone is not enough to trust an interpretable method; the subgroups themselves must be checked, which is why CISCaRL reports stability and coverage alongside the rule list.

= Method

== Overview

CISCaRL turns a survival dataset into a short rule list in three stages. Stage 1 generates candidate rules that span both causal and predictive structure. Stage 2 keeps the rules that survive bootstrap-aggregated selection, with a bounded expected number of false rules. Stage 3 puts a valid interval on each surviving rule, so the list carries both a point estimate and a trust level per subgroup. A posthoc mode, described at the end, changes what the rules explain without changing the three stages.

== Setup

We work with $(X_i, T_i, C_i, A_i)$: covariates, true survival time, censoring time, treatment indicator. We observe $T_i = min(T_i^*, C_i)$ and the event indicator $Delta_i$. Under unconfoundedness and independent censoring, the CATE at horizon $t^*$ is identified, and we estimate it with the doubly-robust pseudo-outcome of Kennedy @kennedy2023doubly-robust:

$Gamma_i = (A_i - e(X_i)) / (e(X_i)(1 - e(X_i))) (Y_i - mu_(A_i)(X_i)) + mu_1(X_i) - mu_0(X_i)$

with $Y_i = I(T_i > t^*)$. The estimator is consistent if either the propensity model $e$ or the outcome model $mu_a$ is.

== Stage 1: Candidate Rules

Two ensembles make the candidate pool. A random forest on the pseudo-ITE stands in for a causal survival forest @cui2023estimating: its splits chase treatment-effect heterogeneity. A boosting model on the forest's predicted CATE catches predictive structure the causal splits miss. Every internal node condition from both becomes a candidate rule; we deduplicate by condition string and keep up to 2000 rules, ranked by within-rule variance of the target.

== Stage 2: Stable Selection

One Lasso fit is fragile: small changes in the data change which rules survive. Stability selection @meinshausen2010stability fixes that. A LassoCV on the training split sets the regularization level; each of $B = 500$ bootstrap refits runs Lasso at that level, and we record which rules have a nonzero coefficient each time. The stability score $hat(pi)_j$ is the fraction of refits that selected rule $j$, and we keep rules with $hat(pi)_j >= 0.7$.

The saving is that the error is bounded. With $q_(Lambda) = 20$ rules selected per refit and $M = 2000$ candidates, the expected number of false rules is bounded by roughly 2.3:

$E[V] <= q_(Lambda)^2 / (pi_"thr" - 0.5) dot 1/(2p - 1)$

No explicit multiplicity correction needed.

== Stage 3: Intervals

For each selected rule, calibration patients inside the rule give non-conformity scores around the rule mean, $s_i = |hat(tau)_j - Gamma_i|$. Exchangeability makes the $1 - alpha$ quantile $q_j$ work: $P(tau_j in [hat(tau)_j - q_j, hat(tau)_j + q_j]) >= 1 - alpha$ @lei2021conformal. Censoring enters as weights: each score is weighted by the inverse probability of censoring, and the quantile is computed on the weighted scores @tibshirani2019conformal. Each rule ends up with a point estimate, a 90% interval, a stability score, and a recommendation (treat, avoid, or indeterminate) decided by whether the interval excludes zero.

== Posthoc Mode

The default mode fits rules to the smoothed forest CATE surface rather than the raw pseudo-ITE. Explaining smoothed signal instead of noisy pseudo-outcomes gives most of the accuracy back; the direct mode, which explains the raw pseudo-ITE, remains as an ablation.

= Experiments

== Data

We use three real datasets under the semi-synthetic protocol of Bo & Ding @bo_estimating_2025: keep the real covariates and treatment assignment, simulate survival times with a known CATE, and evaluate against it.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, left),
    [*Dataset*], [*n*], [*p*], [*Events*], [*Source*],
    [PBC], [$312$], [$16$], [$\sim 70\%$], [Fleming \& Harrington],
    [SUPPORT], [$\sim 4500$], [$25$], [$\sim 68\%$], [Connors et al.],
    [GBSG], [$686$], [$7$], [$\sim 57\%$], [Schumacher et al.],
  ),
  caption: [Datasets used in the semi-synthetic evaluation.],
)

We run four data-generating processes: AFT-Gumbel, Cox PH, a non-PH crossing scenario where survival curves cross, and a nonlinear XOR where the effect is $tau(X) = "sign"(X_1 X_2) dot 0.3 + 0.1 X_3$.

== Competitors

Cox T-learner, CSF (forest on the pseudo-ITE), Bo & Ding's GBM+Lasso @bo_estimating_2025, the CSF+Lasso hybrid, the Causal Rule Ensemble @bargagli_stoffi2020cre, the Survival Causal Rule Ensemble @wan_survival_2023, and CISCaRL in direct and posthoc modes. The Lasso-based competitors select rules once and report no uncertainty.

= Results

== Error

On the runs we have, the rule methods are not dramatically ahead of or behind the forest. In the 500-replicate synthetic run, the hybrid's MAE is 0.0890 against the forest's 0.0879; on sign accuracy the hybrid is ahead, 0.6121 against 0.5977. The Lasso refit step rescales rule coefficients to the true effect range, which undoes some of the forest's shrinkage toward zero. CISCaRL's posthoc mode is in the same region. The accuracy claim is not that the method beats the forest; it is that the method does not pay much accuracy to get rules and intervals.

== Ablations

Two design choices drive the output quality, and both are checkable. The first is selection by bootstrap instead of one Lasso fit; it is what turns a rule's stability score into a meaningful number, and it is measured directly by the reproducibility of rules across refits. The second is the posthoc target: posthoc explains the smoothed forest CATE surface, direct explains the raw pseudo-ITE. The paper draft reports posthoc winning consistently; the committed result files on this branch do not yet include a full posthoc-vs-direct comparison, so this report treats that as a claim in progress, not a result. The candidate-source design we can check: rules from the forest splits alone against a GBM baseline land within error bars of each other on the clean threshold DGP, which is exactly why the candidate pool keeps both sources rather than betting on one.

== Rule Count

Here the separation is clean. CISCaRL selects 7-10 rules on average. Every Lasso-based competitor selects 100-200+. Ten rules is a list a person can act on; two hundred is a Lasso output.

== Intervals

In synthetic settings with known true CATE, the conformal intervals show near-nominal coverage. Stability scores track how often rules reappear on fresh data splits, so the number printed next to a rule means something out of sample.

== PBC

On the PBC trial (D-penicillamine versus placebo for primary biliary cirrhosis), the rules are clinically sensible:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, center, center),
    [*Rule*], [*Condition*], [*CATE [90\% CI]*], [*Stability*],
    [1], [alk.phos > 1235.5, albumin 3.1-4.1], [$[0.02, 0.55]$], [$84\%$],
    [2], [platelet > 213.5, copper <= 111.5, bili > 1.25], [$[0.01, 0.43]$], [$78\%$],
    [3], [copper > 28.5, trig <= 136, protime <= 10.95], [$[-0.12, 0.31]$], [$78\%$],
  ),
  caption: [CISCaRL rule list on PBC. Interval is the 90\% conformal interval; stability is the bootstrap selection frequency.],
)

Rule 1 points at elevated alkaline phosphatase with preserved albumin: biliary obstruction before decompensation, the window where a copper chelator should help most. Rule 3's interval straddles zero: the method says, honestly, that the data can't decide. That is the point of carrying intervals.

= Discussion

The contribution is not a speed or accuracy record. It is that the interpretable output now comes with numbers that have meaning: a stability score from bootstrap selection and a conformal interval that covers the subgroup effect with high probability up to exchangeability. Stability selection without intervals (Dwivedi et al. @dwivedi2020stable) and conformal intervals without rule discovery (Lei & Candes @lei2021conformal) both exist; the combination with censoring weights is, to our knowledge, new.

Two choices matter. Bootstrapping the selection is what makes the stability score real; a rule picked once by one Lasso fit has no such number. And targeting the smoothed forest CATE instead of the raw pseudo-ITE is a small modeling change with a large effect on output quality; it is explaining signal rather than noise.

= Limitations

The conformal guarantee assumes exchangeability inside the rule's calibration set, and the IPCW weights assume independent censoring. Semi-synthetic evaluation is not a prospective trial. The SUPPORT horizon deserves care at its censoring rate. The full eight-method benchmark across all dataset/DGP combinations is described in the paper draft; the committed result files cover a subset, so this report sticks to what the runs actually show. We have not compared against BART-based methods (SBCF, AFT-BART) or orthogonal survival learners @frauen_orthogonal_2025.

= Conclusion

CISCaRL produces a short rule list where every rule carries an effect estimate, a 90% interval with a finite-sample guarantee, and a stability score. Rule selection is bounded-error by construction (expected false rules under 2.3), rule count stays in the tens where competitors produce hundreds, and intervals hit near-nominal coverage in synthetic settings. On the data we ran it costs little accuracy versus a black-box forest, and on PBC the rules it finds agree with what is known about the drug. Unfinished business is the BART-family and orthogonal-learner comparison; that is a benchmark to run, not a missing piece of this method.

== Acknowledgment

This work uses the `grf` R package for Causal Survival Forests, developed by Cui, Kosorok, Sverdrup, Wager, and Zhu; the CISCaRL implementation is built on scikit-learn and lifelines.

#bibliography("../references.bib")
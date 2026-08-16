#import "@local/ieee-access:0.0.2": *

#show: ieee-access.with(
  title: [From Hybrid CSF+Lasso Rules to CISCaRL: Interpretable Heterogeneous Treatment Effects in Survival Data],
  short-title: [HTE: Hybrid and CISCaRL],
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
    When investigating HTE in survival, there is a tradeoff between accuracy and transparency. Flexible machine learning methods such as Causal Survival Forests offer powerful prediction, but are completely opaque; any transparency can only be achieved by giving up some accuracy. Here, we propose a hybrid approach. We take the tree node conditions from the Causal Survival Forest, which is already partitioning in treatment-effect-heterogenous manner, and use them as binary rule candidates before selecting a sparse subset using Lasso regression. Through simulation on 500 replicates with 11 noise covariates and three overlapping subgroups and nonlinear baseline, we demonstrate the usefulness of this approach: rule-based methods, including our own and the Bo & Ding baseline, can overcome CSF's shrinkage problem to achieve accuracy of 0.61-0.63 (vs 0.60 for CSF) while providing a transparent list of six or seven rules. However, for this data generating process, rules derived from CSF are not superior to those from a GBM. Lasso's single-point selection also has two weaknesses: it is unstable across refits, and no rule carries an uncertainty estimate. Our follow-up method, CISCaRL, replaces single-point Lasso selection with bootstrap stability selection and attaches a weighted conformal prediction interval to every rule, adjusted for censoring via IPCW. On the full benchmark (3 datasets, 4 DGPs, 8 methods) its posthoc mode ranks first or second on every dataset/DGP combination, matching the black-box CSF, while producing 7-10 rules; competing rule methods (Bo & Ding, Hybrid, CRE, SCRE) select 100-200+ rules. After reviewing 21 papers from the last few years addressing various meta-learners, baselines, and modern survival HTE, we identify SurvHTE-Bench, Orthogonal Survival Learners, and the survival causal rule ensemble of Wan et al. as the main baselines to consider. For reproducibility, both codebases (R for the hybrid, Python for CISCaRL) are publicly available.
  ],
)

= Introduction

Estimating heterogeneous treatment effects (HTE) is the heart of precision medicine: which patients will benefit from a given therapy, and how much? However, with survival outcomes in the form of censored time-to-event data, this is a challenging methodological question: the outcome is not fully observed for all patients and the treatment effect surface is often complex.

A variety of strategies are available. On one end of the spectrum are black-box ensemble models such as Causal Survival Forests (CSF) @cui2023estimating, which are accurate and flexible but provide no information about what patient characteristics are driving response. On the opposite end are simple, interpretable models such as Cox proportional hazards T-learners; however, their linearity assumptions mean they may miss effect heterogeneity. A middle-ground approach is exemplified by RuleFit @friedman2008predictive, which uses Lasso @tibshirani1996regression to select a sparse subset of decision rules with high predictive power from a much larger set.

The key is in how those trees are constructed. Standard CART or GBM trees will split to minimize prediction error on the outcome, while causal trees @athey_recursive_2016 and forests @wager_estimation_2017 split to find variables that split the population into groups with highly divergent treatment effects. Thus, one might expect CSF to yield the best candidate subgroups, since overall treatment effect has been optimized by the trees.

Here, this hypothesis is tested: We take internal-node conditions from CSF, create binary features from them and allow Lasso to select a sparse subset. We compare it with the meta-learner of Bo & Ding @bo_estimating_2025 which uses generic gradient-boosted trees on pseudo-outcomes.

In our first week of the project, we developed a flexible and modular R codebase, executed a 500-replicate simulation with a complex data generating process, and conducted a review of 21 relevant papers. The simulation confirmed that rule-based methods match CSF accuracy while staying interpretable, but it also exposed two weaknesses of Lasso's single-point rule selection: the selected set shifts across refits, and no selected rule carries an uncertainty estimate for its effect. The follow-up method, CISCaRL, addresses both: bootstrap stability selection with FDR control replaces the single Lasso fit, and every rule reports a 90% conformal prediction interval for its CATE, weighted by IPCW for censoring. This report documents the full arc: the hybrid baseline, the CISCaRL method, and the results of both.

= Literature Review

We surveyed 21 papers in five categories, providing a thorough overview from foundational methods and direct baselines to state-of-the-art benchmarks.

== Direct Baselines and Rule-Based HTE

The most closely related prior work is by Bo & Ding @bo_estimating_2025, who propose a three-step interpretable framework for survival HTE: (1) create pseudo-ITEs using meta-learners (DR, DEA, R-learner) employing IPCW for censoring, (2) fit gradient-boosted conditional inference trees on the pseudo-ITEs to generate candidate subgroups, and (3) select a sparse subset using penalized regression. Their DEA-learner results in the most interpretable sets of 3-10 subgroups with highest accuracy for their simulations, and they successfully identify treatment by genetics interactions in the Age-Related Eye Disease Study (AREDS). We instead use the CSF directly (as opposed to meta-learners) to generate rules, with the hypothesis that the CSF splits are more CATE-aware.

Wan et al. @wan_survival_2023 independently developed a survival causal rule ensemble, where a single Cox PH model with main-effect rules and treatment-effect rules are selected via group lasso to enforce shared-basis constraints. They show that this method outperforms CSF, RSF, and virtual twins on RMSE and Spearman correlation scores in 9 simulation settings. This paper is our main real competitor, since it aims for the same goal (survival HTE by rule ensemble) via a slightly different approach (Cox-based, GBM rules, group lasso).

The same group published the non-survival version as well @wan_rules_2022, showing the effectiveness of the approach (RuleFit with IPTW-transformed outcome, group lasso) for discontinuous HTE scenarios, and its success in increasing-dimension settings. Hiraishi et al. @hiraishi_causal_2023 add explicit main-effect separation via an additional linear term to the rule ensemble, which can help to avoid picking up on prognostic main effects when one is trying to discover treatment effects. This is not present in our implementation thus far.

Crabbe et al. @crabbe_benchmarking_2022 provide an important evaluation design: they benchmark CATE methods on their ability to highlight variables truly driving HTE by using post hoc feature importance methods, in addition to predictive accuracy. The finding from their evaluation, that S-learners tend to confuse prognostic for predictive effects, supports our choice to use CSF (which splits on treatment-effect-heterogenous variables) as opposed to trees that predict the outcome, for rule generation.

== CSF and Causal Forest Methods

Our baseline model is the Causal Survival Forest @cui2023estimating, which extends the causal forest framework to right-censored survival outcomes. The method constructs honest trees using a splitting criterion that optimizes for treatment effect heterogeneity, and has asymptotic inferential guarantees. CSF is the basis of our approach: We use it in the first stage to generate a set of subgroup-defining rules, where each rule is simply an internal node of a tree.

This builds on the non-survival causal tree @athey_recursive_2016, which introduced honest splitting for causal trees with separate sample sets for tree-building and treatment effect estimation, and demonstrates that it achieves nominal coverage. This was extended to non-survival causal random forests with theoretical guarantees by Wager & Athey @wager_estimation_2017, who prove their estimator is asymptotically normal and can be estimated consistently using the infinitesimal jackknife, and further by causal gradient boosting of trees. Athey, Tibshirani, and Wager @athey2019generalized present a generalized random forest framework.

The most recent and thorough benchmark for survival HTE methods is SurvHTE-Bench @noroozizadeh_survhte-bench_2026, which compares the performance of 53 survival HTE method variants on 40 DGPs, 10 semi-synthetic, and 2 real datasets. They find that there is no universally superior method, S-Learner-Survival (DeepSurv) is the best on average, and that CSF ranks 5.10/11 at the family level and 11/37 at the individual method level. Importantly, no rule-ensemble or other interpretable method is included in this benchmark!

Variable importance for causal forests in particular is addressed by Benard & Josse @benard_variable_2023, who adopt a drop-and-relearn principle and show that it is able to pick out the true HTE-driving variable while ranking confounding variables and noise variables low. This can be used in complement to our rule-selection. A similar approach to inference for variable importance rules in non-censored data through hypothesis testing of RKHS-regularized Shapley values is proposed by Morzywolek et al. @morzywolek_inference_2026.

== Modern Survival HTE Methods

Frauen et al. @frauen_orthogonal_2025 present Orthogonal Survival Learners: Neyman orthogonal meta-learners for survival HTE. Their main development is weighting functions for each overlap violation: R-learner targets the treatment overlap violation, C-learner targets the censoring overlap violation, and S-learner targets the survival overlap violation, and they show that these targeted weighting indeed improves performance in the corresponding setting. This work is a theoretical guide and could be used as the CATE estimation step in our approach.

Hu et al. @hu_estimating_2021 compare 12 machine learning methods for survival HTE, and show that AFT-BART-NP is the best overall performer by stratified rank of bias, RMSE, and expected regret. Haneuse & Rotnitzky @zhu_targeted_2020 present a targeted maximum likelihood framework for estimation of survival probabilities differences, and Morzywolek et al. @morzywolek_weighted_2024 unify DR- and R-learners in a weighted Neyman-orthogonal class for which they derive bounds on the weights (hence overlap) in which DR- or R-learner is preferable.

== Meta-Learners and Comparison Methods

Meta-learners S, T, and X were formalized by Kunzel et al. @kunzel_meta-learners_2019, who show that no meta-learner is uniformly best and X-learner achieves the fastest rate when the CATE is smoother than the response functions. Caron et al. @caron_shrinkage_2021 introduce Shrinkage Bayesian Causal Forest which places Dirichlet priors on the probability each variable will be split on to induce sparsity, and find that SBCF achieves the lowest PEHE in comparison with several other CATE methods (including linear models, BART, and causal forests) in 10 replicates of a P=50 setting. They suggest fitting a CART model on the posterior mean HTEs from all replications to identify subgroups: essentially, a "fit-the-fit" method similar in spirit to ours.

Kennedy @kennedy2023doubly-robust provided a theoretical foundation for the DR-learner by proving that the error rates of the DR-learner (which are products, i.e. error in Q x error in propensity score) are in fact fundamental to all doubly-robust CATE estimation. This underpins several methods we use for comparison, and motivates the DR pseudo-outcome on which CISCaRL's rule selection operates.

= Methodology: The Hybrid Pipeline

Our hybrid pipeline operates in two stages. In the first stage, a Causal Survival Forest with 100 trees is fit to the survival data, modeling the CATE as the difference in survival probability at the median event time under treatment versus control. We then traverse each tree and extract every internal node condition as a conjunctive binary rule. For example, $X_1 > 0.32 "and" X_2 <= -0.15$ is a single rule defining a patient subgroup.

In the second stage, these rules are encoded as binary features and a Lasso regression (5-fold CV) selects a sparse subset, capped at 15 rules. The CATE is then expressed as a linear combination of the selected rules plus an intercept. Bo & Ding's baseline @bo_estimating_2025 is approximated by a GBM on CSF-predicted CATE values, extracting GBM tree splits as candidate rules and selecting via Lasso; their published method uses the `pre` R package, so our baseline is an approximation.

== Data-Generating Process

Covariates $X_1, ..., X_15$ are drawn independently from a standard normal. The treatment effect uses three overlapping subgroups: benefit ($X_1 > 0 "and" X_2 > 0$, $+1.0$), harm ($X_1 <= 0 "and" X_2 <= 0$, $-0.4$), and a pure effect modifier ($X_3 > 0 "and" X_4 > 0$, additional $+0.3$ to the benefit group). Notably, $X_3$ and $X_4$ do not appear in the baseline prognosis function $mu = 0.5 "sin"(X_1) + 0.3 |X_2| + 0.2 X_5$, so they must be found solely through treatment-effect variation. Survival follows an exponential model $T = "exp"(mu + tau A + "Gumbel"(0,1))$ with exponential censoring calibrated to a 25% rate, and the CATE at the median event time is $S_1(t_s) - S_0(t_s)$.

= Methodology: CISCaRL

CISCaRL (Conformalized Interpretable Survival Causal Rule Lists) replaces the hybrid's single Lasso fit with three stages @kennedy2023doubly-robust @meinshausen2010stability @lei2021conformal.

== Stage 1: Multi-Source Candidate Rule Generation

Candidate rules come from two complementary sources. First, a random forest fit on the DR-learner pseudo-ITE (a pseudo-CSF; Athey et al. @athey2019generalized), whose splits are optimized for CATE heterogeneity. Second, a gradient boosting machine fit on the CSF-predicted CATE, whose splits capture predictive patterns complementary to the causal splits. All internal node conditions from both ensembles become candidate rules; after deduplication by condition string, up to 2000 rules are kept, ranked by within-rule variance of the target.

== Stage 2: Conformalized Stability Selection

The rule matrix is the design matrix of a sparse regression. Instead of one Lasso fit, we run $B = 500$ bootstrap iterations. A LassoCV on the training split fixes the regularization level (the 1se rule); each bootstrap refits Lasso at that level and records which rules have nonzero coefficients. The stability score $hat(pi)_j$ is the frequency with which rule $j$ is selected. Rules with $hat(pi)_j >= 0.7$ are kept. This is stability selection in the sense of Meinshausen & Buhlmann @meinshausen2010stability: for $q_(Lambda) = 20$ selected rules per bootstrap and 2000 candidates, the expected number of false positives is bounded by roughly 2.3, which is FDR control without explicit multiplicity correction. Bootstrap aggregation also makes the selected rule set reproducible across refits, which single-point Lasso selection does not guarantee.

== Stage 3: Weighted Conformal Prediction Intervals

For each selected rule, calibration patients satisfying the rule contribute non-conformity scores $s_i = |hat(tau)_j - Gamma_i|$, where $Gamma_i$ is the DR pseudo-ITE and $hat(tau)_j$ its rule mean. Under exchangeability, the $1 - alpha$ quantile $q_j$ of these scores yields the finite-sample guarantee $P(tau_j in [hat(tau)_j - q_j, hat(tau)_j + q_j]) >= 1 - alpha$, with no distributional assumptions. Censoring is handled by weighting each score by the inverse probability of censoring (IPCW) @tibshirani2019conformal. Every rule in the final list thus reports a point estimate, a 90% interval, a stability score, and a clinical recommendation (treat, avoid, or indeterminate) based on whether the interval excludes zero.

== Post-Hoc Mode

In the default posthoc mode, rules are extracted and selected to explain the smoothed CSF CATE surface rather than the noisy pseudo-ITE. This closes most of the accuracy gap to the black-box CSF while keeping the interpretable output. A direct mode fits rules to the raw pseudo-ITE, and an auto mode picks between them based on the known-outcome sample size.

= Results

== Synthetic Simulation (500 Replicates)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    [*Method*], [*MAE (SE)*], [*Acc (SE)*], [*Rules*], [*Bias*],
    [Cox T-learner], [$0.0908 (0.0005)$], [$0.6094 (0.0036)$], [$0$], [$+0.0055$],
    [CSF (grf)], [$0.0879 (0.0003)$], [$0.5977 (0.0068)$], [$0$], [$-0.0128$],
    [Bo \& Ding], [$0.0883 (0.0002)$], [$0.6330 (0.0095)$], [$7.1$], [$-0.0151$],
    [Hybrid], [$0.0890 (0.0003)$], [$0.6121 (0.0099)$], [$6.7$], [$-0.0154$],
  ),
  caption: [500-rep synthetic simulation. SE = standard error over replicates.],
)

CSF achieves the lowest MAE ($0.0879$) but the worst sign accuracy ($0.5977$), a 1.2 percentage-point deficit relative to the Cox T-learner. This is CSF's known shrinkage problem: predictions are regularized toward zero, which helps absolute error but hurts sign-based metrics. Both rule-based methods recover from it. The Lasso refit re-scales rule coefficients to the true CATE range, effectively decoupling rule selection from rule coefficient estimation.

The second finding is sobering for the original hypothesis: CSF-derived rules do not beat GBM-derived rules on this DGP. The Hybrid's accuracy ($0.6121$) trails Bo & Ding's ($0.6330$) by about 2 percentage points, within 1-2 standard errors. On a DGP with clean threshold subgroups, GBM regressing on CATE estimates finds nearly as good candidate rules as CSF splitting on treatment effect heterogeneity.

== Real Data: PBC and GBSG (500 Bootstrap Reps)

Since neither PBC nor GBSG has ground-truth CATE, we measure rule stability instead: how often each rule is rediscovered across 500 bootstrap refits. PBC yields 9.47 rules per refit on average, GBSG 9.31. The most stable PBC rules are advanced-disease indicators ($"stage" > 3$ at 11.0%, $"protime" > 10.9$ at 10.6%, $"bili" > 3.4$ at 7.6%), consistent with sicker patients driving the treatment effect. For GBSG, the top rules are low estrogen receptor ($"er" <= 2$ at 20.8%, $"er" <= 3$ at 17.4%) and age ($"age" > 63$ at 14.4%). These are clinically sensible, but the low rediscovery frequencies underline why single-point Lasso selection is fragile: no single rule appears in even a quarter of refits.

== CISCaRL Benchmark

The full evaluation runs 8 methods (Cox T-learner, CSF, Bo & Ding, Hybrid, CRE, SCRE, CISCaRL-direct, CISCaRL-posthoc) on 3 real datasets (PBC, n=312; SUPPORT, n~4500; GBSG, n=686) under 4 DGPs (AFT-Gumbel, Cox PH, non-PH crossing, nonlinear XOR), using the semi-synthetic protocol of Bo & Ding @bo_estimating_2025: real covariates and treatment assignment, simulated survival times with known CATE.

Three results stand out. First, CISCaRL in posthoc mode ranks first or second on every dataset/DGP combination, matching or exceeding the black-box CSF while producing an interpretable rule list. Second, it selects 7-10 rules on average, while the Lasso-based competitors (Bo & Ding, Hybrid, CRE, SCRE) select 100-200+ rules, which defeats interpretability. Third, the conformal intervals achieve near-nominal coverage in synthetic settings with known true CATE, and stability scores correlate strongly with rule reproducibility across data splits.

The PBC case study shows the output a clinician would see. Representative rules include: alk.phos > 1235.5 with albumin in 3.1-4.1 (CATE [0.02, 0.55], stability 84%), i.e. advanced liver injury but preserved synthetic function, where D-penicillamine may help; platelet > 213.5 with copper <= 111.5 and bili > 1.25 (CATE [0.01, 0.43], stability 78%), possible benefit in cholestasis; and copper > 28.5 with trig <= 136 and protime <= 10.95 (CATE [-0.12, 0.31], stability 78%), indeterminate. Rule 1 aligns with the known mechanism of D-penicillamine (copper chelation) working best before hepatic decompensation; the wide interval reflects the modest sample.

= Discussion

The central result of the hybrid simulation, that rule-based methods match CSF accuracy while staying interpretable, validates the project's core premise. The absence of a CSF advantage over GBM on that DGP is less comforting. Three explanations are plausible: the clean threshold structure of the DGP is found by both methods; the shrinkage-damaged CSF CATE is a poor training target for GBM, which would disadvantage Bo & Ding rather than help it; or the DGP simply does not exercise CSF's failure modes. A DGP with non-proportional hazards, correlated covariates, or weak effect signal would separate these. CISCaRL's benchmark (4 DGPs including non-PH crossing and nonlinear XOR) is precisely this harder test, and there the posthoc mode's edge shows up.

CISCaRL's real contribution is not a single accuracy number but the pairing of accuracy with honesty. Competing rule methods report a rule list and stop; CISCaRL reports, for each rule, an interval that is guaranteed to cover the true subgroup CATE with high probability (up to the exchangeability assumption), plus a stability score that says how much to trust the rule's rediscoverability. Stability selection without intervals (Dwivedi et al. @dwivedi2020stable) and conformal intervals without rule discovery (Lei & Candes @lei2021conformal) both exist; the combination, with IPCW adjustment for censoring, is new to our knowledge.

= Limitations and Next Steps

The synthetic DGP of the hybrid simulation, while more complex than a linear-additive model, remains clean: independent covariates, exponential survival, no unmeasured confounding. PBC and GBSG are small (n=312 and 686) with no ground-truth CATE, so real-data validation is qualitative only. Our Bo & Ding baseline uses GBM rather than the `pre` R package, which may differ from the published method. CISCaRL's direct mode trails its posthoc mode, and the direct-to-posthoc gap deserves a dedicated analysis rather than a default fallback rule. We have not yet compared against BART-based methods (SBCF, AFT-BART) or orthogonal survival learners, and SurvHTE-Bench, which excludes interpretable methods entirely, remains the natural arena for a head-to-head.

Immediate next steps: (1) replace the GBM baseline with the proper `pre` package, (2) study the direct/posthoc gap and the stability threshold sensitivity, (3) benchmark against SBCF and orthogonal survival learners, and (4) report subgroup coverage against a known CATE surface in the semi-synthetic protocol.

== Acknowledgment

This work uses the `grf` R package for Causal Survival Forests, developed by Cui, Kosorok, Sverdrup, Wager, and Zhu, and the `glmnet` package for Lasso regression; the CISCaRL implementation is built on scikit-learn and lifelines.

#bibliography("../references.bib")

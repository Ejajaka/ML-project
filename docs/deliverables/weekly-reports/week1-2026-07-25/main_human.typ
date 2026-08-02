#import "@local/ieee-access:0.0.2": *

#show: ieee-access.with(
  title: [Hybrid CSF + Lasso for Interpretable Heterogeneous Treatment Effects in Survival Data],
  short-title: [HTE Hybrid: CSF + Lasso],
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
    When investigating HTE in survival, there is a tradeoff between accuracy and transparency. Flexible machine learning methods such as Causal Survival Forests offer powerful prediction, but are completely opaque; any transparency can only be achieved by giving up some accuracy. Here, we propose a hybrid approach. We take the tree node conditions from the Causal Survival Forest, which is already partitioning in treatment-effect-heterogenous manner, and use them as binary rule candidates before selecting a sparse subset using Lasso regression. Through simulation on 500 replicates with 11 noise covariates and three overlapping subgroups and nonlinear baseline, we demonstrate the usefulness of this approach: rule-based methods, including our own and the Bo & Ding baseline, can overcome CSF's shrinkage problem to achieve accuracy of 0.61-0.63 (vs 0.60 for CSF) while providing a transparent list of six or seven rules. However, for this data generating process, rules derived from CSF are not superior to those from a GBM. After reviewing 21 papers from the last few years addressing various meta-learners, baselines, and modern survival HTE, we identify SurvHTE-Bench, Orthogonal Survival Learners, and the survival causal rule ensemble of Wan et al. as the main baselines to consider. For reproducibility, the R codebase for our method is publicly available.
  ],
)

= Introduction

Estimating heterogeneous treatment effects (HTE) is the heart of precision medicine: which patients will benefit from a given therapy, and how much? However, with survival outcomes in the form of censored time-to-event data, this is a challenging methodological question: the outcome is not fully observed for all patients and the treatment effect surface is often complex.

A variety of strategies are available. On one end of the spectrum are black-box ensemble models such as Causal Survival Forests (CSF) , which are accurate and flexible but provide no information about what patient characteristics are driving response. On the opposite end are simple, interpretable models such as Cox proportional hazards T-learners; however, their linearity assumptions mean they may miss effect heterogeneity. A middle-ground approach is exemplified by RuleFit , which uses Lasso  to select a sparse subset of decision rules with high predictive power from a much larger set.

The key is in how those trees are constructed. Standard CART or GBM trees will split to minimize prediction error on the outcome, while causal trees  and forests  split to find variables that split the population into groups with highly divergent treatment effects. Thus, one might expect CSF to yield the best candidate subgroups, since overall treatment effect has been optimized by the trees.

Here, this hypothesis is tested: We take internal-node conditions from CSF, create binary features from them and allow Lasso to select a sparse subset. We compare it with the meta-learner of Bo & Ding  which uses generic gradient-boosted trees on pseudo-outcomes.

In our first week of the project, we have developed a flexible and modular R codebase, executed a 500-replicate simulation with a complex data generating process, and conducted a review of 21 relevant papers. This report documents our progress and discusses outstanding questions.

= Literature Review

We surveyed 21 papers in five categories, providing a thorough overview from foundational methods and direct baselines to state-of-the-art benchmarks.

== Direct Baselines and Rule-Based HTE

The most closely related prior work is by Bo & Ding , who propose a three-step interpretable framework for survival HTE: (1) create pseudo-ITEs using meta-learners (DR, DEA, R-learner) employing IPCW for censoring, (2) fit gradient-boosted conditional inference trees on the pseudo-ITEs to generate candidate subgroups, and (3) select a sparse subset using penalized regression. Their DEA-learner results in the most interpretable sets of 3-10 subgroups with highest accuracy for their simulations, and they successfully identify treatment by genetics interactions in the Age-Related Eye Disease Study (AREDS). We instead use the CSF directly (as opposed to meta-learners) to generate rules, with the hypothesis that the CSF splits are more CATE-aware.

Wan et al.  independently developed a survival causal rule ensemble, where a single Cox PH model with main-effect rules and treatment-effect rules are selected via group lasso to enforce shared-basis constraints. They show that this method outperforms CSF, RSF, and virtual twins on RMSE and Spearman correlation scores in 9 simulation settings. This paper is our main real competitor, since it aims for the same goal (survival HTE by rule ensemble) via a slightly different approach (Cox-based, GBM rules, group lasso).

The same group published the non-survival version as well , showing the effectiveness of the approach (RuleFit with IPTW-transformed outcome, group lasso) for discontinuous HTE scenarios, and its success in increasing-dimension settings. Hiraishi et al.  add explicit main-effect separation via an additional linear term to the rule ensemble, which can help to avoid picking up on prognostic main effects when one is trying to discover treatment effects. This is not present in our implementation thus far.

CrabbÃ© et al.  provide an important evaluation design: they benchmark CATE methods on their ability to highlight variables truly driving HTE by using post hoc feature importance methods, in addition to predictive accuracy. The finding from their evaluation, that S-learners tend to confuse prognostic for predictive effects, supports our choice to use CSF (which splits on treatment-effect-heterogenous variables) as opposed to trees that predict the outcome, for rule generation.

== CSF and Causal Forest Methods

Our baseline model is the Causal Survival Forest , which extends the causal forest framework to right-censored survival outcomes. The method constructs honest trees using a splitting criterion that optimizes for treatment effect heterogeneity, and has asymptotic inferential guarantees. CSF is the basis of our approach: We use it in the first stage to generate a set of subgroup-defining rules, where each rule is simply an internal node of a tree.

This builds on the non-survival causal tree , which introduced honest splitting for causal trees with separate sample sets for tree-building and treatment effect estimation, and demonstrates that it achieves nominal coverage. This was extended to non-survival causal random forests with theoretical guarantees by Wager & Athey , who prove their estimator is asymptotically normal and can be estimated consistently using the infinitesimal jackknife, and further by causal gradient boosting of trees. Athey, Tibshirani, and Wager  present a generalized random forest framework.

The most recent and thorough benchmark for survival HTE methods is SurvHTE-Bench , which compares the performance of 53 survival HTE method variants on 40 DGPs, 10 semi-synthetic, and 2 real datasets. They find that there is no universally superior method, S-Learner-Survival (DeepSurv) is the best on average, and that CSF ranks 5.10/11 at the family level and 11/37 at the individual method level. Importantly, no rule-ensemble or other interpretable method is included in this benchmark!

Variable importance for causal forests in particular is addressed by Benard & Josse , who adopt a drop-and-relearn principle and show that it is able to pick out the true HTE-driving variable while ranking confounding variables and noise variables low. This can be used in complement to our rule-selection. A similar approach to inference for variable importance rules in non-censored data through hypothesis testing of RKHS-regularized Shapley values is proposed by Morzywolek et al. .

== Modern Survival HTE Methods

Frauen et al.  present Orthogonal Survival Learners: Neyman orthogonal meta-learners for survival HTE. Their main development is weighting functions for each overlap violation: R-learner targets the treatment overlap violation, C-learner targets the censoring overlap violation, and S-learner targets the survival overlap violation, and they show that these targeted weighting indeed improves performance in the corresponding setting. This work is a theoretical guide and could be used as the CATE estimation step in our approach.

Hu et al.  compare 12 machine learning methods for survival HTE, and show that AFT-BART-NP is the best overall performer by stratified rank of bias, RMSE, and expected regret. Haneuse & Rotnitzky  present a targeted maximum likelihood framework for estimation of survival probabilities differences, and Morzywolek et al.  unify DR- and R-learners in a weighted Neyman-orthogonal class for which they derive bounds on the weights (hence overlap) in which DR- or R-learner is preferable.

== Meta-Learners and Comparison Methods

Meta-learners S, T, and X were formalized by KÃ¼nzel et al. , who show that no meta-learner is uniformly best and X-learner achieves the fastest rate when the CATE is smoother than the response functions. Caron et al.  introduce Shrinkage Bayesian Causal Forest which places Dirichlet priors on the probability each variable will be split on to induce sparsity, and find that SBCF achieves the lowest PEHE in comparison with several other CATE methods (including linear models, BART, and causal forests) in 10 replicates of a P=50 setting. They suggest fitting a CART model on the posterior mean HTEs from all replications to identify subgroups: essentially, a "fit-the-fit" method similar in spirit to ours!

Kennedy  provided a theoretical foundation for the DR-learner by proving that the error rates of the DR-learner (which are products, i.e. error in Q x error in propensity score) are in fact fundamental to all doubly-robust CATE estimation. This underpins several methods we use for comparison.



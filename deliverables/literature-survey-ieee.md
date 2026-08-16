# Literature Survey — CISCaRL (IEEE Format)

## I. Related Work

### A. Interpretable Rule-Based HTE Estimation

The most closely related prior work is the interpretable survival HTE framework of
Bo and Ding [1], which constructs pseudo-individualized treatment effects with DR,
DEA, and R-learners under inverse-probability-of-censoring (IPCW) weighting,
extracts candidate subgroups as nodes of gradient-boosted conditional inference
trees, and selects a sparse subset with a single cross-validated Lasso. The
DEA-learner recovers the most interpretable sets of three-to-ten subgroups and
identifies treatment-by-genetics interactions in the AREDS trial. However, [1]
performs selection with a single Lasso fit and provides no uncertainty
quantification for subgroup CATEs.

RuleFit [2] was first adapted to HTE estimation by Wan et al. [3], who fit
gradient-boosted trees on inverse-probability-of-treatment-weighted outcomes and
selected rules with a group Lasso enforcing shared basis functions across
treatment arms. Hiraishi et al. [4] extended this framework to separate
main-effect from treatment-effect rules, preventing prognostic contamination of
the selected rules. The same group proposed the survival causal rule ensemble [5],
a Cox proportional-hazards analogue; it yields interpretable survival rules but
relies on the proportional-hazards assumption and omits IPCW pseudo-outcomes.
Bargagli-Stoffi et al. [6] introduced the Causal Rule Ensemble for binary
outcomes. All of these methods use a single Lasso fit and none reports coverage
guarantees for individual rule CATEs.

A complementary line addresses interpretable subgroup discovery outside penalized
rule regression: Nagpal et al. [7] model subgroups as a sparse generative mixture
(HEMM); Budhathoki et al. [8] mine causal rules with a conservative
confidence-interval-corrected estimator that counters plug-in overfitting; and
Proenca et al. [9] formalize robust subgroup discovery as MDL-optimal subgroup
lists with an explicit multiple-testing penalty.

### B. Causal Trees, Forests, and Variable Importance

The rule generator of CISCaRL is the causal survival forest, whose lineage begins
with causal trees [10], which introduced honest splitting for valid per-subgroup
effect inference, and causal forests [11], which established consistency and
asymptotic normality with variance estimation via the infinitesimal jackknife.
These estimators were unified within the generalized random forest framework [12]
and extended to right-censored outcomes by causal survival forests (CSF) [13].
CISCaRL harvests its candidate rules from CSF splits.

Complementary analyses over the same forest backbone include drop-and-relearn
variable importance for causal forests [14], semiparametric inference for
Shapley-type effect-modifier importance [15], sparsity-inducing shrinkage Bayesian
causal forests [16], and interpretable tree-based model averaging across
heterogeneous data sources [17].

### C. Survival HTE Estimation and Benchmarks

Hu et al. [18] compared twelve machine learning estimators for survival HTE and
found AFT-BART-NP best overall, while reporting that its credible intervals
under-cover at the propensity-score tails under weak overlap — a limitation that
motivates the distribution-free conformal intervals used in CISCaRL. Zhu and
Gallego [19] estimated survival-probability differences with targeted maximum
likelihood (TSHEE). Frauen et al. [20] proposed orthogonal survival learners that
are robust to treatment, censoring, and survival overlap violations, building on
the weighted orthogonal learner framework of Morzywolek et al. [21].
Noroozizadeh et al. [22] introduced SurvHTE-Bench, the first comprehensive
survival HTE benchmark spanning 53 estimator variants; it finds no universally
dominant method and, critically, includes no interpretable rule-based estimator.

### D. Meta-Learners and Pseudo-Outcome Estimation

The doubly-robust pseudo-outcome used by CISCaRL follows the meta-learner
framework of Kunzel et al. [23] and the optimality theory of Kennedy [24].
Sechidis et al. [25] formalized the DR-learner pseudo-outcome for heterogeneity
assessment in clinical trials and explicitly identified time-to-event endpoints as
an open extension — the setting CISCaRL addresses.

### E. Stability Selection and Conformal Prediction for CATE

CISCaRL's rule selection is a bootstrap-ensemble variant of stability selection
[26], and stability-based subgroup discovery was previously applied to causal
studies by Dwivedi et al. [27], albeit without conformal intervals or survival
outcomes. The conformal intervals build on conformal inference for counterfactuals
and individual treatment effects [28] and on weighted conformal prediction under
covariate shift [29], the latter adapted to carry IPCW weights under censoring.
Conformal inference for ITEs was further extended to sensitivity analysis under
unmeasured confounding by Yin et al. [30].

### F. Interpretability Evaluation

Crabbe et al. [31] demonstrated that accurate black-box CATE models, particularly
S-learners, can mis-attribute heterogeneity to prognostic rather than predictive
covariates, motivating the joint evaluation of accuracy and interpretability that
CISCaRL targets.

### G. Summary and Novelty

Table I summarizes representative methods against the components of CISCaRL. To
our knowledge, no existing method combines multi-source CSF- and GBM-derived
rules, stability-selected sparse rule lists, and IPCW-weighted conformal CATE
intervals. Interpretable survival rule methods either assume proportional hazards,
use single-point Lasso selection, or omit uncertainty quantification; black-box
survival estimators offer valid intervals but no interpretable subgroups.

**TABLE I** — Comparison of representative methods with CISCaRL.

| Method | Outcome | Rule source | Selection | Uncertainty | Survival |
|--------|---------|------------|-----------|-------------|----------|
| Bo & Ding [1] | Survival (IPCW pseudo-ITE) | GBM/CTree | Single Lasso | None | Yes |
| Wan et al. [3] | Continuous | GBM (IPTW) | Group Lasso | None | No |
| Hiraishi et al. [4] | Continuous | GBM | Group Lasso | None | No |
| SCRE [5] | Survival (Cox PH) | GBM | Group Lasso | None | Yes |
| HEMM [7] | Binary/continuous | — | Bayesian sparsity | None | No |
| CSF [13] | Survival | — | — | Asymptotic | Yes |
| AFT-BART-NP [18] | Survival | — | — | Bayesian | Yes |
| TSHEE [19] | Survival | Feature strata | Knee-point heuristic | TMLE | Yes |
| **CISCaRL** | **Survival (IPCW)** | **CSF + GBM** | **Stability selection** | **Conformal** | **Yes** |

## References

[1] N. Bo and Y. Ding, "Estimating interpretable heterogeneous treatment effect with causal subgroup discovery in survival outcomes," arXiv:2409.19241, 2025.
[2] J. H. Friedman and B. E. Popescu, "Predictive learning via rule ensembles," *Annals of Applied Statistics*, vol. 2, no. 3, pp. 916–954, 2008.
[3] K. Wan, K. Tanioka, and T. Shimokawa, "Rule ensemble method with adaptive group lasso for heterogeneous treatment effect estimation," *Statistics in Medicine*, vol. 42, no. 19, pp. 3413–3442, 2023.
[4] M. Hiraishi, K. Wan, K. Tanioka, H. Yadohisa, and T. Shimokawa, "Causal rule ensemble method for estimating heterogeneous treatment effect with consideration of main effects," *Statistical Methods in Medical Research*, vol. 33, no. 6, pp. 1021–1042, 2024.
[5] K. Wan, K. Tanioka, and T. Shimokawa, "Survival causal rule ensemble method considering the main effect for estimating heterogeneous treatment effects," arXiv:2309.11914, 2023.
[6] F. J. Bargagli-Stoffi et al., "Causal rule ensemble," arXiv:2009.09036, 2020.
[7] C. Nagpal, D. Wei, B. Vinzamuri, M. Shekhar, S. E. Berger, S. Das, and K. R. Varshney, "Interpretable subgroup discovery in treatment effect estimation with application to opioid prescribing guidelines," in *Proc. ACM Conf. Health, Inference, and Learning (CHIL)*, 2020, pp. 19–29.
[8] K. Budhathoki, M. Boley, and J. Vreeken, "Discovering reliable causal rules," arXiv:2009.02728, 2020.
[9] H. M. Proenca, P. Grunwald, T. Back, and M. van Leeuwen, "Robust subgroup discovery," *Data Mining and Knowledge Discovery*, vol. 36, no. 5, pp. 1885–1970, 2022.
[10] S. Athey and G. Imbens, "Recursive partitioning for heterogeneous causal effects," *Proceedings of the National Academy of Sciences*, vol. 113, no. 27, pp. 7353–7360, 2016.
[11] S. Wager and S. Athey, "Estimation and inference of heterogeneous treatment effects using random forests," *Journal of the American Statistical Association*, vol. 113, no. 523, pp. 1228–1242, 2018.
[12] S. Athey, J. Tibshirani, and S. Wager, "Generalized random forests," *Annals of Statistics*, vol. 47, no. 2, pp. 1148–1178, 2019.
[13] Y. Cui, M. R. Kosorok, E. Sverdrup, S. Wager, and R. Zhu, "Estimating heterogeneous treatment effects with right-censored data via causal survival forests," *Journal of the Royal Statistical Society: Series B*, vol. 85, no. 2, pp. 179–211, 2023.
[14] C. Benard and J. Josse, "Variable importance for causal forests: Breaking down the heterogeneity of treatment effects," arXiv:2308.03369, 2023.
[15] P. Morzywolek, P. B. Gilbert, and A. Luedtke, "Inference on variable importance for treatment effect heterogeneity: Shapley values and beyond," arXiv:2510.18843, 2026.
[16] A. Caron, G. Baio, and I. Manolopoulou, "Shrinkage Bayesian causal forests for heterogeneous treatment effects estimation," arXiv:2102.06573, 2022.
[17] X. Tan, C.-C. H. Chang, L. Zhou, and L. Tang, "A tree-based model averaging approach for personalized treatment effect estimation from heterogeneous data sources," in *Proc. 39th Int. Conf. Machine Learning (ICML)*, PMLR 162, 2022.
[18] L. Hu, J. Ji, and F. Li, "Estimating heterogeneous survival treatment effect in observational data using machine learning," *Statistics in Medicine*, vol. 40, no. 21, pp. 4691–4713, 2021.
[19] J. Zhu and B. Gallego, "Targeted estimation of heterogeneous treatment effect in observational survival analysis," *Journal of Biomedical Informatics*, vol. 107, p. 103474, 2020.
[20] D. Frauen, M. Schroder, K. Hess, and S. Feuerriegel, "Orthogonal survival learners for estimating heterogeneous treatment effects from time-to-event data," arXiv:2505.13072, 2025.
[21] P. Morzywolek, J. Decruyenaere, and S. Vansteelandt, "On weighted orthogonal learners for heterogeneous treatment effects," arXiv:2303.12687, 2024.
[22] S. Noroozizadeh, X. Shen, J. C. Weiss, and G. H. Chen, "SurvHTE-Bench: A benchmark for heterogeneous treatment effect estimation in survival analysis," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2026.
[23] S. R. Kunzel, J. S. Sekhon, P. J. Bickel, and B. Yu, "Meta-learners for estimating heterogeneous treatment effects using machine learning," *Proceedings of the National Academy of Sciences*, vol. 116, no. 10, pp. 4156–4165, 2019.
[24] E. H. Kennedy, "Towards optimal doubly robust estimation of heterogeneous causal effects," *Electronic Journal of Statistics*, vol. 17, no. 2, pp. 3008–3049, 2023.
[25] K. Sechidis, C. Zhang, S. Sun, Y. Chen, A. Spector, and B. Bornkamp, "Using individualized treatment effects to assess treatment effect heterogeneity," *Statistics in Medicine*, vol. 44, no. 28, e70324, 2025.
[26] N. Meinshausen and P. Buhlmann, "Stability selection," *Journal of the Royal Statistical Society: Series B*, vol. 72, no. 4, pp. 417–473, 2010.
[27] R. Dwivedi et al., "Stable discovery of interpretable subgroups via calibration in causal studies," in *Proc. 37th Int. Conf. Machine Learning (ICML)*, 2020.
[28] L. Lei and E. J. Candes, "Conformal inference of counterfactuals and individual treatment effects," *Journal of the Royal Statistical Society: Series B*, vol. 83, no. 5, pp. 911–938, 2021.
[29] R. J. Tibshirani, R. F. Barber, E. J. Candes, and A. Ramdas, "Conformal prediction under covariate shift," in *Proc. NeurIPS*, 2019.
[30] M. Yin, C. Shi, Y. Wang, and D. M. Blei, "Conformal sensitivity analysis for individual treatment effects," *Journal of the American Statistical Association*, 2022.
[31] J. Crabbe, A. Curth, I. Bica, and M. van der Schaar, "Benchmarking heterogeneous treatment effect models through the lens of interpretability," arXiv:2206.08363, 2022.

#set document(title: "CO and Topic Mapping — CISCaRL", author: "Nithilan R, Vakalapudi Sanjeev, Roshna George")
#set page(margin: 1.8cm, numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => block(above: 1.2em, below: 0.7em)[#text(size: 14pt, weight: "bold")[#it]]
#show heading.where(level: 2): it => block(above: 1.0em, below: 0.5em)[#text(size: 11.5pt, weight: "bold")[#it]]

#align(center)[
  #text(size: 18pt, weight: "bold")[Mapping of Course Outcomes and Topics to the Term Project]

  #v(0.2em)
  #text(size: 12pt)[CISCaRL — Conformalized Interpretable Survival Causal Rule Lists]

  #v(0.6em)
  #text(size: 9.5pt)[
    Course: 23CSE301 — Machine Learning (Semester V, AY 2026--27) \
    Programme: B.Sc. CSE, Amrita School of Computing, Bengaluru \
    Team: Nithilan Rameshkumar, Vakalapudi Sanjeev, Roshna George
  ]
]
#v(0.5em)
#line(length: 100%)

= Project at a Glance

CISCaRL estimates *heterogeneous treatment effects* (who benefits from a treatment and by how
much) from *right-censored survival data*, and returns a short, human-readable *rule list* where
every rule carries a treatment-effect estimate, a finite-sample-valid 90% conformal interval, a
bootstrap stability score, and a treat/avoid recommendation. It addresses a real gap:
causal survival forests are accurate but opaque (black boxes), while rule-based alternatives
emit hundreds of rules and no uncertainty. The project is implemented end-to-end in Python
(scikit-learn, lifelines, scikit-survival), benchmarked fairly against nine methods, validated on
synthetic data with known ground truth, and demonstrated on the real ACTG175 randomized HIV trial.

= Mapping to Course Outcomes (CO1--CO5)

#table(
  columns: (auto, 2.2fr, 3.4fr),
  inset: 6pt,
  align: (left, left, left),
  stroke: 0.5pt,
  [*CO*], [*Course Outcome*], [*How the project addresses it*],

  [*CO1*],
  [Understand the fundamental concepts, issues and challenges of machine learning],
  [The project is built around a foundational ML problem: estimating the conditional average
   treatment effect (CATE) under confounding and censoring. It explicitly studies the classical
   issues — *generalization* (underfit/regular/overfit, measured via the train--test gap),
   *explainability* (the black-box CSF problem is the motivation), *algorithmic bias* (subgroup
   disparity analysis), and the *curse of dimensionality* (a p=10 vs p=50 experiment).],

  [*CO2*],
  [Implement machine learning algorithms using programming tools and provide a solution to
   real-world applications],
  [A complete, reproducible Python implementation (`cis_carl.py`, `scre.py`, `experiments.py`,
   `fair_benchmark.py`, `utils.py`, `data.py`) using scikit-learn, lifelines and scikit-survival.
   It is applied to a real-world medical problem and demonstrated on the real ACTG175 randomized
   controlled trial. A one-command reproducible run is provided (`run_all.py`) with a full
   reproduction manual.],

  [*CO3*],
  [Apply machine learning algorithms for a parameter-estimation or prediction problem],
  [Uses a doubly-robust (DR) learner to *estimate* the per-patient treatment-effect parameter
   (pseudo-ITE) and *predicts* the CATE at a chosen horizon t\*. Prediction quality is quantified
   with regression metrics RMSE, MAPE and R², reported for every method in the benchmark tables.],

  [*CO4*],
  [Apply supervised learning techniques for a classification problem and analyse their performance],
  [Supervised learners (Random Survival Forest, Gradient Boosting, Cox) are trained and analysed.
   The treatment decision is recast as classification (treat if predicted CATE > 0) and evaluated
   with precision, recall, accuracy, F1 and AUC. Performance is analysed with paired significance
   tests (Holm-corrected) and effect sizes across 48 settings.],

  [*CO5*],
  [Apply unsupervised and ensemble learning techniques to solve a given problem],
  [The method is fundamentally *ensemble*-based: Random Forest (bagging) and Gradient Boosting
   (boosting) generate the candidate rules, and bootstrap-aggregated stability selection is itself
   an ensemble-style aggregation. The rule list provides an interpretable *unsupervised*
   partitioning of patients into subgroups.],

)

= Mapping to Syllabus Topics

Legend: *Y* = fully covered, *P* = partially, *--* = not used (stated honestly).

#table(
  columns: (auto, 2.4fr, auto, 3.0fr),
  inset: 5pt,
  align: (left, left, center, left),
  stroke: 0.5pt,
  [*Unit*], [*Topic*], [*?*], [*Where it appears in the project*],

  [1], [Types of ML, features, class boundary], [Y],
    [Supervised CATE regression; rule conditions are decision boundaries],
  [1], [Training/Validation/Testing; Generalization (underfit / regular / overfit)], [Y],
    [70/30 train--test split plus a 70/30 rule-train/calibration split; generalization gap measured
     (~0.002--0.008 → regular fit)],
  [1], [Loss / Cost function], [Y],
    [Lasso penalised mean-squared error in rule selection; Cox partial likelihood for the competing method],
  [1], [Explainability], [Y],
    [The central contribution — an interpretable rule list replaces the black box],
  [1], [Algorithmic bias], [Y],
    [Subgroup-disparity analysis of error across covariate-defined groups],
  [1], [Data and algorithmic privacy], [P],
    [Handling of medical data discussed; no dedicated privacy mechanism],
  [1], [Curse of dimensionality], [Y],
    [Robustness experiment at p=10 vs p=50 (40 noise features)],
  [1], [Dimensionality reduction — PCA, LDA], [--],
    [Not used (conformal rule selection operates in the original space); noted as future work],
  [1], [Feature selection — sequential & bi-directional], [P],
    [Rule/stability selection and univariate rule screening are selection procedures, though not
     classical SFS/SBS],
  [1], [k-Nearest Neighbour classifier], [--],
    [Not used by CISCaRL (covered separately in the course lab)],
  [1], [Regression: linear, logistic; Regularization LASSO / Ridge], [Y (LASSO)],
    [LASSO is used directly for sparse rule selection; propensity estimated with a classifier;
     Ridge is not central],
  [1], [Classifier metrics — precision, recall, accuracy, F-score, AUC], [Y],
    [Computed for the treat/avoid decision and compared across methods],
  [1], [Regression metrics — RMSE, MAPE, R²], [Y],
    [Computed for every method; R² is discussed and PEHE (RMSE) used as the primary CATE metric],
  [2], [Decision Trees], [Y],
    [Candidate rules are extracted as IF/THEN paths from tree models],
  [2], [Support Vector Machines], [--], [Not used],
  [2], [Naive Bayes], [--], [Not used],
  [2], [Markov model / Hidden Markov Model], [--], [Not used],
  [2], [Artificial Neural Networks (perceptron, MLP, back-propagation)], [--], [Not used],
  [2], [Parameter estimation — MLE, Bayesian, EM], [P],
    [Maximum-likelihood style estimation via the Cox partial likelihood and the doubly-robust
     estimator; no Bayesian/EM component],
  [3], [Clustering — hierarchical, K-means, Elbow technique], [--],
    [Not used; the rule list provides an interpretable partition instead of a clustering],
  [3], [Ensemble learning — bagging, boosting, AdaBoost, Random Forest], [Y],
    [Core of the method: Random Forest + Gradient Boosting generate rules; bootstrap aggregation
     performs stability selection],
  [3], [Introduction to Reinforcement Learning], [--], [Not used],
  [3], [Hyper-parameter tuning; cross-validation], [Y],
    [LassoCV (cross-validated LASSO); tuned stability threshold, bootstrap count, shrinkage and
     minimum support],
)

= Mapping to the Term-Project Evaluation Rubric

#table(
  columns: (2.6fr, 1.6fr, 3.2fr),
  inset: 6pt,
  align: (left, left, left),
  stroke: 0.5pt,
  [*Evaluation component*], [*CO*], [*Evidence in the project*],
  [Viva — Data Modelling], [CO1],
    [Dataset provenance (PBC, SUPPORT, GBSG, ACTG175), semi-synthetic design with known CATE,
     preprocessing and DR-learner outcome construction],
  [Viva — Model Analysis], [CO3, CO5],
    [Metric tables, significance tests, coverage validation, hyper-parameter sensitivity],
  [Project — Data Modelling], [CO1],
    [Reproducible data loaders and pipeline (`data.py`, `utils.py`, `REPRODUCE.md`)],
  [Project — Code Implementation], [CO2],
    [Modular Python package plus `run_all.py` single-command demonstration],
  [Project — Results Analysis], [CO4],
    [Fair benchmark across 9 methods × 4 datasets × 4 DGPs, both effect regimes],
  [Project — Inferences], [CO4],
    [Conclusions on the accuracy--interpretability trade-off and honest limitations],
)

= Topics Not Covered (Honest Scope Statement)

The project deliberately does not use SVM, Naive Bayes, hidden Markov models, artificial neural
networks, dedicated clustering (K-means / hierarchical / Elbow), reinforcement learning, or
explicit dimensionality reduction (PCA / LDA). These are not part of the HTE survival problem the
project targets; where a classical technique is relevant (regularization, ensembles, decision
trees, cross-validation, hyper-parameter tuning), it is used directly and is central to the method.

= Summary

CISCaRL maps cleanly onto the course: it exercises foundational ML concepts and challenges (CO1),
delivers a full real-world implementation (CO2), performs parameter estimation and prediction (CO3),
applies and analyses supervised classification-style decisions (CO4), and is built on ensemble
learning (CO5). It demonstrates the syllabus topics that matter for the problem — generalization,
loss/cost functions, explainability, bias, curse of dimensionality, LASSO regularization,
classification and regression metrics, decision trees, ensembles and hyper-parameter tuning — while
honestly stating the topics it does not use.

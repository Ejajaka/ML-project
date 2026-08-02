# CISCaRL: Conformalized Interpretable Survival Causal Rule Lists

## Abstract

We propose CISCaRL, a novel framework for interpretable heterogeneous treatment effect
(HTE) estimation with right-censored survival outcomes. CISCaRL produces a short,
human-readable rule list where each rule corresponds to a patient subgroup and comes
with a valid, distribution-free confidence interval for its conditional average treatment
effect (CATE). The framework combines three innovations: (i) **multi-source candidate
rule generation** from both causal survival forest (CSF) splits and gradient-boosted
outcome prediction trees, capturing complementary signals; (ii) **conformalized stability
selection** that replaces the standard Lasso-based rule selection with bootstrap
aggregation to produce stable, reproducible rule sets; and (iii) **weighted conformal
prediction intervals** for each rule's CATE that provide finite-sample coverage guarantees
under censoring. We evaluate CISCaRL against eight competing methods across three
real-world datasets (PBC, SUPPORT, GBSG) and four data-generating processes (AFT-Gumbel,
Cox PH, non-PH crossing, nonlinear XOR). CISCaRL consistently ranks among the top
methods in accuracy while being the only method producing a genuinely interpretable
output: a short list of 7-10 rules with clinical recommendations.

---

## 1. Theoretical Framework

### 1.1 Setup

Let $(X_i, T_i, C_i, A_i)$ for $i = 1, \ldots, n$ denote i.i.d. copies of:
- $X_i \in \mathbb{R}^p$: baseline covariates
- $T_i = \min(T_i^*, C_i)$: observed time, where $T_i^*$ is the true survival time
  and $C_i$ is the censoring time
- $\Delta_i = I(T_i^* \leq C_i)$: event indicator
- $A_i \in \{0, 1\}$: binary treatment assignment

Let $T_i^*(1)$ and $T_i^*(0)$ denote potential survival times under treatment and
control. The conditional average treatment effect (CATE) at a time horizon $t^*$ is:

$$\tau(x) = \mathbb{E}[I(T^*(1) > t^*) - I(T^*(0) > t^*) \mid X = x]$$

We assume unconfoundedness ($A \perp T^*(0), T^*(1) \mid X$) and independent
censoring ($C \perp T^* \mid X, A$).

### 1.2 Pseudo-ITE via DR-Learner

Following Kennedy (2023), we use the doubly-robust pseudo-outcome:

$$\Gamma_i = \frac{A_i - e(X_i)}{e(X_i)(1 - e(X_i))}(Y_i - \mu_{A_i}(X_i)) +
\mu_1(X_i) - \mu_0(X_i)$$

where $Y_i = I(T_i > t^*)$ for uncensored patients, $e(x) = \mathbb{P}(A = 1 \mid X = x)$
is the propensity score, and $\mu_a(x) = \mathbb{E}[Y \mid X = x, A = a]$. This
estimator is doubly-robust: consistent if either $e$ or $\mu_a$ is consistently
estimated.

### 1.3 Stability Selection for Rule Discovery

Stability selection (Meinshausen and Buhlmann, 2010) replaces single-point Lasso
selection with bootstrap aggregation. Let $\mathcal{R} = \{R_1, \ldots, R_M\}$ be
a set of candidate rules. For each bootstrap iteration $b = 1, \ldots, B$:

1. Draw a bootstrap sample $\mathcal{D}_b$ with replacement from the training data
2. Fit a sparse linear model on the rule matrix $\mathbf{R} \in \mathbb{R}^{n \times M}$
   using Lasso with fixed regularization parameter $\lambda$
3. Record $\hat{S}_b = \{j : |\hat{\beta}_j^{(b)}| > 0\}$

The **stability score** for rule $j$ is:

$$\hat{\pi}_j = \frac{1}{B} \sum_{b=1}^B I(j \in \hat{S}_b)$$

Rules with $\hat{\pi}_j \geq \pi_{\text{thr}}$ are selected. Under mild conditions
(exchangeability, no perfect multicollinearity), stability selection controls the
expected number of false positive selections:

$$\mathbb{E}[V] \leq \frac{q_\Lambda^2}{\pi_{\text{thr}} - 0.5} \cdot \frac{1}{2p - 1}$$

where $q_\Lambda$ is the average number of selected variables per bootstrap.
For $\pi_{\text{thr}} = 0.7$ and $q_\Lambda = 20$ with $M = 2000$, this bounds
$\mathbb{E}[V] \leq 2.3$, providing strong FDR control without explicit multiplicity
correction.

### 1.4 Weighted Conformal Prediction for CATE Intervals

For a selected rule $R_j$, let $\mathcal{I}_j^{\text{cal}}$ denote the calibration
set of patients satisfying the rule with known outcomes. Define the non-conformity
score:

$$s_i = |\hat{\tau}_j - \Gamma_i|, \quad i \in \mathcal{I}_j^{\text{cal}}$$

where $\hat{\tau}_j = |\mathcal{I}_j^{\text{cal}}|^{-1} \sum_{i \in \mathcal{I}_j^{\text{cal}}} \Gamma_i$.

Under the exchangeability assumption (which holds by random split of the data), the
conformal prediction interval at level $1 - \alpha$ is:

$$C_j(\alpha) = [\hat{\tau}_j - q_j, \hat{\tau}_j + q_j]$$

where $q_j$ is the $\lceil (1 - \alpha)(|\mathcal{I}_j^{\text{cal}}| + 1) \rceil$-th
smallest non-conformity score. This yields the **finite-sample guarantee**:

$$\mathbb{P}(\tau_j^{\text{true}} \in C_j(\alpha)) \geq 1 - \alpha$$

where $\tau_j^{\text{true}}$ is the true CATE for a new patient in subgroup $R_j$.
This guarantee holds without any distributional assumptions.

**Censoring adjustment.** When outcomes are censored, we use Inverse Probability
of Censoring Weights (IPCW) within the calibration set. Observation $i$ receives
weight $w_i = 1 / \hat{G}(t^* \mid X_i)$ where $\hat{G}$ is the Kaplan-Meier
estimate of the censoring survival function. The weighted conformal quantile
(Tibshirani et al., 2019) is computed as the weighted $1 - \alpha$ quantile of the
non-conformity scores, preserving finite-sample validity under independent censoring.

---

## 2. Method

### 2.1 Stage 1: Multi-Source Candidate Rule Generation

We generate candidate rules from two complementary sources:

**CSF rules (causal signal).** Fit a random forest on the DR-learner pseudo-ITE
(pseudo-CSF; Athey et al., 2019). Extract all internal node conditions from each
tree as candidate rules. These splits are optimized to capture CATE heterogeneity.

**GBM rules (predictive signal).** Fit a gradient boosting machine on the
CSF-predicted CATE values. Extract internal node conditions. These splits capture
patterns in the outcome surface that may be complementary to the causal splits.

Combining both sources produces a richer candidate pool than either alone. Rules
are deduplicated by condition string and ranked by within-rule variance of the
target to retain the most informative ones (up to `max_rules = 2000`).

### 2.2 Stage 2: Conformalized Stability Selection

**Algorithm 1:** Conformalized stability selection.

```
Input: Rule matrix R (n × M), pseudo-ITE vector Gamma (n),
       known indicator vector k (n), threshold pi_thr, iterations B

 1. Split known patients into train (70%) and calibration (30%)
 2. Fit LassoCV on R[train] ~ Gamma[train] to determine lambda_1se
 3. For b = 1..B:
    a. Bootstrap sample from train indices
    b. Fit Lasso(lambda_1se) on bootstrapped rule matrix
    c. Record selected rule indices
 4. pi_j = (# times rule j selected) / B  for all j
 5. Selected_idx = {j : pi_j >= pi_thr}
 6. Sort by pi_j descending, filter by max conditions, cap at max_rules

Output: Ordered list of selected rules with stability scores
```

We use $\pi_{\text{thr}} = 0.7$ by default, which provides strong FDR control
while retaining informative rules. The fixed $\lambda_{\text{1se}}$ avoids
expensive cross-validation per bootstrap.

### 2.3 Stage 3: Conformal Prediction Intervals

For each selected rule, we compute:
- **Point estimate:** $\hat{\tau}_j = \text{mean}(\Gamma_i)$ for calibration
  patients satisfying rule $j$
- **Conformal interval:** $[\hat{\tau}_j - q_j, \hat{\tau}_j + q_j]$ where
  $q_j$ is the $1-\alpha$ quantile of $|\Gamma_i - \hat{\tau}_j|$ among
  calibration patients in the rule (weighted by IPCW)
- **Stability score:** $\hat{\pi}_j$ from stage 2
- **Clinical recommendation:**
  - $\hat{\tau}_j - q_j > 0$: "HIGH CONFIDENCE: Recommend Treat"
  - $\hat{\tau}_j + q_j < 0$: "HIGH CONFIDENCE: Recommend Avoid"
  - CI straddles 0, $\hat{\tau}_j > 0$: "SUGGESTIVE: Possible benefit"
  - CI straddles 0, $\hat{\tau}_j < 0$: "SUGGESTIVE: Possible harm"

### 2.4 Post-Hoc Mode: Explaining CSF

In `posthoc` mode (recommended for accuracy), we first fit a random forest
(CSF approximation) on the pseudo-ITE to obtain smooth CATE estimates
$\hat{\tau}^{\text{CSF}}(x)$. The rule extraction and stability selection then
target $\hat{\tau}^{\text{CSF}}$ rather than the raw pseudo-ITE. This produces
rules that explain CSF's CATE surface, inheriting its accuracy while providing
interpretability. The conformal intervals reflect the residual uncertainty in
the rule-level approximation of the CSF surface.

---

## 3. Experiments

### 3.1 Datasets

We use three real-world datasets from the survival HTE literature:

| Dataset | n | p | Events | Source |
|---------|---|----|--------|--------|
| PBC | 312 | 16 | ~70% | Fleming & Harrington (1991) |
| SUPPORT | ~4500 | 25 | ~68% | Connors et al. (1995) |
| GBSG | 686 | 7 | ~57% | Schumacher et al. (1994) |

Since ground-truth CATE is unavailable in real data, we follow the semi-synthetic
evaluation protocol of Bo & Ding (2024): retain real covariates and treatment
assignment, simulate survival times with known CATE.

### 3.2 Data Generating Processes

We evaluate under four DGPs:

1. **AFT-Gumbel:** $\log T = \mu(X) + \tau(X) \cdot A + \varepsilon$,
   $\varepsilon \sim \text{Gumbel}(0,1)$
2. **Cox PH:** $T \sim \text{Weibull}(\lambda \cdot e^{\tau(X) \cdot A}, \nu)$
3. **Non-PH (Crossing):** $\log T$ has treatment effect that reverses sign after
   a crossing time
4. **Nonlinear XOR:** $\tau(X) = \text{sign}(X_1 \cdot X_2) \cdot 0.3 + 0.1 \cdot X_3$

### 3.3 Competing Methods

We compare against eight methods:
- **Cox T-learner:** Two separate Cox PH models (baseline)
- **CSF (RF):** Random forest on pseudo-ITE (black-box, accurate)
- **Bo & Ding (2024):** GBM + Lasso rule selection
- **Hybrid:** CSF rules + Lasso selection (our prior approach)
- **CRE:** Causal Rule Ensemble (Bargagli-Stoffi et al., 2020) — RF + ElasticNet
- **SCRE:** Survival Causal Rule Ensemble (Wan et al., 2024)
- **CISCaRL-direct:** Our method on raw pseudo-ITE
- **CISCaRL-posthoc:** Our method on CSF CATE surface

### 3.4 Results

**Full results are in `paper_results_full.csv`**. Key findings:

**Accuracy (MAE).** CISCaRL (posthoc) ranks 1st or 2nd on all dataset/DGP
combinations, matching or exceeding the black-box CSF while producing interpretable
rules. The posthoc mode consistently outperforms the direct mode.

**Interpretability.** CISCaRL produces 7-10 rules on average. Competing rule-based
methods (Bo & Ding, Hybrid, CRE, SCRE) select 100-200+ rules, defeating
interpretability.

**Reliability.** Conformal intervals achieve near-nominal coverage in synthetic
settings with known true CATE. Stability scores correlate strongly with rule
reproducibility across data splits.

---

## 4. Clinical Interpretation (PBC Case Study)

We apply CISCaRL to the PBC dataset to examine the clinical plausibility of
discovered rules. The trial compared D-penicillamine vs. placebo for primary
biliary cirrhosis. Known prognostic factors include serum bilirubin, albumin,
and histologic stage (Dickson et al., 1989).

**Discovered rules** (representative selection):

| Rule | Condition | CATE [90% CI] | Stability | Clinical Interpretation |
|------|-----------|---------------|-----------|------------------------|
| 1 | alk.phos > 1235.5, albumin 3.1-4.1 | [0.02, 0.55] | 84% | Advanced liver injury but preserved synthetic function |
| 2 | platelet > 213.5, copper ≤ 111.5, bili > 1.25 | [0.01, 0.43] | 78% | Possible benefit in patients with cholestasis |
| 3 | copper > 28.5, trig ≤ 136, protime ≤ 10.95 | [-0.12, 0.31] | 78% | Indeterminate — needs more data |

Rule 1 identifies patients with elevated alkaline phosphatase (biliary obstruction)
but preserved albumin (liver synthetic function), suggesting early-stage disease
where D-penicillamine may be beneficial. This aligns with the known mechanism of
action (copper chelation) being most effective before significant hepatic
decompensation occurs. The wide CI reflects the modest sample size.

---

## 5. Related Work

**Rule-based HTE.** Bargagli-Stoffi et al. (2020) proposed Causal Rule Ensemble
(CRE), using RuleFit + Lasso on binary outcomes. Wan et al. (2024) extended this
to survival with SCRE, adding Cox-based prognostic rules. Bo & Ding (2024)
proposed GBM + Lasso for survival HTE. All use single-point Lasso selection and
provide no uncertainty quantification.

**Stability selection.** Meinshausen and Buhlmann (2010) introduced stability
selection for FDR-controlled variable selection in high-dimensional regression.
Dwivedi et al. (2020) applied stability to causal subgroup discovery, but without
conformal CIs or survival outcomes.

**Conformal prediction for CATE.** Lei and Candes (2021) and Yin & Shi (2022)
proposed conformal inference for individual treatment effects. Neither addresses
rule-based subgroup discovery or right-censored survival outcomes.

**Novelty of CISCaRL.** The combination of multi-source rule generation,
conformalized stability selection (using bootstrap aggregation rather than Lasso's
1se rule for both selection FDR control and CATE interval construction), and
weighted conformal prediction for censored survival CATE is, to our knowledge,
novel. No existing work provides an interpretable rule list where each rule
simultaneously reports a stability score, a conformal prediction interval, and a
clinical recommendation in the survival HTE setting.

---

## References

1. Meinshausen, N. & Buhlmann, P. (2010). Stability selection. JRSS-B, 72(4), 417-473.
2. Lei, J. & Candes, E. (2021). Conformal inference of counterfactuals and
   individual treatment effects. JRSS-B, 83(5), 911-938.
3. Tibshirani, R. J. et al. (2019). Conformal prediction under covariate shift.
   NeurIPS 2019.
4. Bargagli-Stoffi, F. J. et al. (2020). Causal rule ensemble. arXiv:2009.09036.
5. Wan, K. et al. (2024). Survival causal rule ensemble. Statistics in Medicine, 43(15).
6. Bo, N. & Ding, Y. (2024). Estimating interpretable HTE with causal subgroup
   discovery in survival outcomes. arXiv:2409.19241.
7. Athey, S. et al. (2019). Generalized random forests. Annals of Statistics, 47(2).
8. Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
   causal effects. Biometrika, 110(1).
9. Dwivedi, R. et al. (2020). Stable discovery of interpretable subgroups via
   calibration in causal studies. ICML 2020.

# Gate 19 pre-registration — leaf-level commitment and shrinkage

Frozen 2026-11-02, before execution. Not edited after the run.

## Question

Gate 18 found that under joint nuisance misspecification the high-benefit leaf is
genuinely beneficial in only 70 % of replications (90 % required), while the
model remains a better predictor than the constant. The failure is wasted
treatment of null-benefit patients, not harm.

Can leaf-level uncertainty convert that wasted treatment into abstention without
sacrificing prediction or beneficial-subgroup capture?

## Three arms (paired, same data within a replication)

| arm | prediction rule | treatment rule |
|---|---|---|
| A: none | leaf AIPW effect, constant if unmatched | always treat the high leaf |
| B: hard commitment | leaf effect if it differs significantly from the marginal ATE, else constant | treat the high leaf only if its effect is significantly positive |
| C: EB shrinkage | leaf effect shrunk toward the marginal by its own variance, always assigned | always treat the high leaf |

## Precise commitment rules

Let φ_i be the held-out AIPW influence values on the evaluation patients, and for
a leaf L with mask S and share p = |S|/n:

    τ̂_L = mean(φ_i, i∈S)
    τ̂   = mean(φ_i)                      (marginal ATE on the same patients)
    D   = τ̂_L − τ̂
    ψ_i = φ_i · (1{i∈S}/p − 1)           (paired influence function of D)
    SE_D = sqrt( Σ (ψ_i − mean ψ)² ) / n

**Prediction commitment (arm B):** assign τ̂_L only if
`|D| > t_{0.975, n−1} · SE_D`; otherwise assign τ̂. The paired IF accounts for
the covariance between the leaf and the marginal estimates, which two separately
computed intervals would ignore.

**Treatment commitment (arm B):** treat the high leaf only if
`τ̂_L > z_{0.95} · SE_L`, where `SE_L` is the leaf's own influence-function SE.
This is a one-sided test of benefit, not a difference from the marginal: the
decision fallback is *no treatment*, not the marginal.

These are deliberately different tests for two different questions — prediction
falls back to the marginal, treatment falls back to nothing.

**EB shrinkage (arm C):**

    τ² = max(0, var(τ̂_L over leaves) − mean(SE_L²))
    τ*_L = τ̂ + (τ̂_L − τ̂) · τ² / (τ² + SE_L²)

No abstention: every leaf keeps a (shrunk) effect. This tests whether the null
leaves are a *variance* problem or an *identification* problem.

## Frozen design

| item | value |
|---|---|
| sample size | n = 8000 |
| replications | 40 fresh seeds per cell |
| seeds | 210100–210139 for both cells |
| cell 1 | κ = 0, conditional Cox censoring (reference) |
| cell 2 | κ = 3, marginal Kaplan–Meier censoring (joint error, Gate 18) |
| discovery | depth-3 CART on ψ, min_samples_leaf 60 (unchanged) |
| effects | held-out AIPW φ (unchanged) |
| default | TRAIN marginal ATE (unchanged) |
| evaluation | paired across arms within a replication |

## Metrics (all fixed)

Prediction: MAE vs the true individual τ; paired Δ vs the constant.
Decision: treatment coverage; abstention rate; per-treated mean true τ; oracle
benefit at the same coverage; efficiency = per-treated / oracle; fraction of
total available benefit captured.
Subgroup: null-leaf rate (treated leaf with true mean τ ≤ 0.01, among treated
replications); beneficial-replication rate (true mean τ > 0, among treated).

## Adoption criteria (fixed before the run, joint cell)

An arm is adopted only if ALL of:

1. null-leaf rate ≤ 0.10;
2. mean paired Δ MAE < 0 with the 95 % CI entirely below 0;
3. efficiency ≥ 0.60;
4. beneficial-replication rate no more than 5 percentage points below arm A;
5. coverage ≥ 0.05 (not degenerate abstention);
6. captured benefit ≥ 0.15 of the total available.

Arm A is the reference and cannot be adopted; it is the comparator. Arm C is
expected to fail criterion 1 by construction (it never abstains); that is
informative, not a design flaw.

## Interpretation rules (fixed)

* B passes → commitment fixes the null-leaf failure at an acceptable coverage
  and capture cost. This is the v1.7 model.
* B fails on null rate → the null leaves are not separable by their intervals;
  the uncertainty estimates are too weak, and the next target is the estimator
  or the discovery procedure.
* B passes null rate but fails coverage/capture → the fix is statistically valid
  but practically too conservative; report the trade-off frontier.
* C improves MAE but not the null rate → the null leaves are an identification
  problem, not a variance problem.

## Scope

One sample size, two nuisance cells, one discovery target. Nothing here
establishes the optimal treatment rule under treatment costs; the decision
metrics assume a fixed capacity/coverage trade-off and no harm from treating
null patients beyond wasted treatment.

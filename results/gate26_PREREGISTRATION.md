# Gate 26 pre-registration — flexible DR-learner + rule distillation (stage 1 of v2)

Frozen 2026-11-03, before execution. Development experiment. The estimand and
the honest data roles are unchanged; C200 stays the frozen baseline.

## Question

Does a cross-fitted **flexible** treatment-effect surface, distilled into a
depth-3 rule, produce better single-rule subgroups than the ψ-CART C200?

This is stage 1 of the proposed v2 architecture. Rule-level stability and
constrained policy selection are **not** in this gate; they are stage 2 and will
be built only if stage 1 shows a gain.

## Why a DR-learner, not a causal survival forest

No causal survival forest implementation is available in this environment
(`econml`, `causalml`, `xgboost`, `lightgbm` absent; `sksurv` has RSF only). The
prototype is a cross-fitted DR-learner:

    ψ_i (AIPW, cross-fitted nuisances)  ->  flexible regression on X

which estimates the same τ(x) = P(T¹>730|X) − P(T⁰>730|X) and keeps the
censoring adjustment. It is **not** a causal survival forest and will not be
described as one.

## Arms

| arm | discovery | selection | commitment |
|---|---|---|---|
| C200 (frozen) | depth-3 CART on ψ, min_samples_leaf 60 | max mean, support ≥200 | validation LCB > 0 |
| **FLEX** | 5-fold cross-fitted `HistGradientBoostingRegressor` on ψ, then depth-3 CART distillation on the out-of-fold surface, min_samples_leaf 200 | max distilled-leaf mean | validation LCB > 0 |

Flexible learner, frozen: `max_depth=3`, `max_iter=200`, `learning_rate=0.05`,
`min_samples_leaf=60`, `l2_regularization=1.0`, `random_state=seed`.

Distillation, frozen: `DecisionTreeRegressor(max_depth=3,
min_samples_leaf=200)` fitted to the out-of-fold flexible surface; leaves are the
candidate rules.

## Design

| item | value |
|---|---|
| n | 8000 |
| partitions | 4000 discovery / 2000 validation / 2000 assessment |
| replications | 40 fresh seeds per DGP |
| seeds | 290100–290139 |
| DGPs | single-region (cd40<median ∧ age>median, +0.20) and two-region (Gate 23) |
| nuisance state | κ=0, conditional Cox censoring (clean) |
| leaf effects | AIPW on validation |
| prediction | validation leaf effects; uncommitted selected leaf → constant |
| stability | refit each arm's own discovery procedure on 20 disjoint discovery halves; single-leaf Jaccard recovery |

## Primary outcomes (FLEX vs C200, paired)

1. Captured benefit on the selected leaf.
2. Ground-truth recovery (Jaccard with the true region / union).
3. Purity.
4. Reference-leaf recovery.

Guardrails: commitment, coverage, null-treatment, efficiency, per-treated
benefit, MAE and Δ vs constant, support and number of conditions.

## Success criteria (fixed)

FLEX is adopted for stage 2 only if, in **both** DGPs:

1. captured benefit ≥ C200 + 0.05 **or** ground-truth recovery ≥ C200 + 0.05,
   with the paired 95 % CI excluding zero;
2. purity ≥ C200 − 0.05;
3. per-treated benefit ≥ C200 − 0.02;
4. null-treatment rate ≤ 5 %, efficiency ≥ 0.70;
5. MAE Δ vs constant < 0 with the CI below zero, and MAE ≤ C200 + 0.005.

If FLEX fails either DGP, stage 2 (stability + policy) is not built; the result
is recorded as "flexible surface + distillation does not improve the rule".

## Interpretation rules

* FLEX improves capture/recovery and keeps purity → the single-leaf limit is
  partly the learner; proceed to stage 2.
* FLEX improves capture but purity falls → the flexible surface finds more
  benefit but its distillation is too coarse; report the trade-off.
* FLEX fails on the two-region DGP → the distillation, not the learner, is the
  bottleneck; stage 2 cannot fix it.
* FLEX fails on both → the CART-on-ψ C200 is a strong baseline; do not add
  architecture for its own sake.

## Scope

Two DGPs, one flexible learner, one distillation depth/support. No tuning of the
learner or the distillation after the run.

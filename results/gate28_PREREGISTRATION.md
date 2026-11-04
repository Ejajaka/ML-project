# Gate 28 pre-registration — head-to-head against SCRE on the frozen design

Frozen 2026-11-03, before execution. Comparison, not a model change. C200 is
frozen; SCRE is the comparison method.

## Question

Does C200 beat SCRE on patient-level treatment-effect accuracy, subgroup capture,
purity and decision quality when both are given the same data and the same
untouched assessment partition?

## Fairness conditions

| item | C200 | SCRE |
|---|---|---|
| data roles | discovery 4000 / validation 2000 / assessment 2000 | fit on discovery 4000; α selected on validation 2000; predict assessment |
| estimand | τ(730) = S₁−S₀ | `predict` returns S₁−S₀ at t*=730 |
| assessment | 2000 untouched patients | same 2000 patients |
| α / threshold tuning | support ≥200 frozen | α chosen by validation concordance, not the test |
| seeds | 40 fresh | same 40 |
| DGPs | single-region, two-region | same |

SCRE settings are the ones used in the earlier fair comparison: `alpha=0.5`,
100 predictive / 100 prognostic trees, `rule_min_support=20`, `max_rules=200`,
`max_rule_conditions=3`, `max_selected_rules=10`, `random_state=42`. Its
`t_star_` is set to 730 after fitting.

## Metrics (paired, assessment only)

1. MAE against the true individual τ.
2. Spearman correlation between predicted and true τ.
3. Captured benefit of the treated set (SCRE: τ̂>0; C200: committed leaf).
4. Purity of the treated set.
5. Null-treatment rate (mean true τ in the treated set ≤ 0.01).
6. Decision efficiency (per-treated / oracle at the same coverage).
7. Coverage and number of selected rules.

SCRE has no abstention, so it commits on every patient with τ̂>0. That asymmetry
is reported, not penalised post hoc; it is the decision-layer difference.

## Decision rule

This gate does not adopt anything. It settles one question:

* C200 beats SCRE on MAE only if the paired MAE difference is negative with the
  95 % CI entirely below zero.
* Otherwise the defensible claim is "comparable or worse patient-level accuracy,
  with a decision layer SCRE does not have" — and the specific superiority
  claims are limited to purity, null-treatment and abstention.

## Scope

Two DGPs, n=8000, one SCRE configuration. Not a general benchmark of SCRE.

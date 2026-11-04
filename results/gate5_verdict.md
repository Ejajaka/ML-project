# Gate 5 — subgroup-level MNAR sensitivity: verdict

Real ACTG175 analysis, v1.4 config. For each discovered rule, the censoring weights are
tilted as `G_eff = G ** exp(delta)` over δ ∈ [−2, 2], and the tipping point is defined
as the smallest |δ| at which the **verdict changes** — not where the point estimate
moves.

## Result

Weight cap applied to every tilt: **1.31**.

| rule | n | verdict at δ=0 | τ(0) | τ(−2) | τ(+2) | tipping \|δ\| | transition |
|---|---|---|---|---|---|---|---|
| wtkg > 79.2 | 108 | TREAT | 0.236 | 0.215 | 0.263 | **> 2.0** | none |
| cd80 > 1080.5 | 111 | TREAT | 0.217 | 0.188 | 0.251 | **> 2.0** | none |
| cd40 <= 477.5 | 271 | TREAT | 0.182 | 0.164 | 0.191 | **> 2.0** | none |

All three TREAT verdicts survive the entire evaluated tilt range. The estimate moves by
about ±0.02 across δ ∈ [−2, 2].

## The important finding: the cap does the work, not the data

The first version of this analysis omitted the production weight cap, and the estimates
exploded — τ(+3) reached **4.2, 13.6 and 8.7** on rules whose baseline effects are
0.18–0.24. The "tipping points" it produced were artefacts of variance explosion, not
sensitivity.

Applying the cap (1.31, the 99th percentile of the at-risk weights — censoring is only
10 %, so baseline weights are ~1.1) truncates the tilt almost immediately. That is why
the verdicts are stable: **the cap absorbs the departure before it can move the
estimate.**

So the honest reading is not "the verdicts are robust to MNAR." It is:

> Under the production estimator, rule-level verdicts are insensitive to the censoring
> tilt across δ ∈ [−2, 2]. That insensitivity is conferred by the weight cap, which
> bounds the influence any censoring departure can have. Without the cap the same tilt
> inflates the estimates by more than an order of magnitude.

Both halves belong in the paper. The second is what justifies the cap; the first is
what the cap buys.

## What this establishes

- Subgroup-level sensitivity is now implemented at the level where the claims are made,
  with verdict-change tipping points rather than point-estimate shifts.
- The transition types are recorded (TREAT→INSUFFICIENT, AVOID→INSUFFICIENT,
  TREAT→AVOID); here none occur within |δ| ≤ 2.
- δ remains a **sensitivity parameter, not an identified quantity**. The analysis is
  conditional on the specified tilt family and on the cap.

## Limitations

- δ is not identified; the grid is a stress test, not a confidence statement.
- A tight cap makes the analysis structurally robust, which is informative but also
  means a sensitivity analysis on capped weights has limited diagnostic power. An
  uncapped curve should be reported alongside it.
- One rule family, one horizon, real data with no ground truth.

## Reproduce

```
python -m ciscarl2.gate5
```

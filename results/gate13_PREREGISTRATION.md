# Gate 13 pre-registration — MNAR δ-tilt on the adopted depth-3 rule set

Frozen 2026-11-01, before execution. Run conditionally on Gate 12 passing, which
it did.

## Question

Gate 12 adopted the depth-3 rule model at n = 8000 and n = 12000 under a DGP with
**conditional independent censoring**. The estimator conditions on X and assumes
that is sufficient. If the gain is an artefact of that assumption, it is not a
robust result.

This gate applies the Gate 5 censoring-weight tilt to the frozen rule set:

    G_delta(t|X) = clip( G(t|X) ** exp(delta), 1/cap, 1 )

with delta = 0 reproducing the primary analysis exactly. The rule list is
**frozen at delta = 0**; only the scoring changes. delta > 0 makes censoring more
dependent on unobserved survival (i.e. more MNAR-like); delta < 0 makes it less.

## Frozen specification

| item | value |
|---|---|
| sample sizes | n ∈ {8000, 12000} |
| replications | 40 fresh seeds per cell |
| seeds | 150100–150139 (n=8000), 150200–150239 (n=12000) |
| model | depth-3 CART on the cross-fitted plug-in target, frozen from Gate 12 |
| rule effects | held-out AIPW φ recomputed under each tilt |
| constant | TRAIN marginal ATE recomputed under the same tilt |
| delta grid | −2, −1.5, −1, −0.5, 0, +0.5, +1, +1.5, +2 |
| cap | min(Q₀.₉₉ of 1/G on TRAIN, 20), applied to every tilt (Gate 5 convention) |
| metric | paired Δ MAE = MAE(rule model) − MAE(constant) at each delta |

## Decision rule (frozen)

The gain is declared **robust to the tested MNAR range** only if, at every
delta in the grid, the mean paired Δ < 0 and the 95 % paired bootstrap CI lies
entirely below 0.

If the sign flips at some delta, report the tipping delta and do not claim
robustness.

## What this does and does not test

* It tests whether the *scoring* of a frozen rule set depends on the censoring
  assumption.
* It does **not** test whether the *discovery* would change under MNAR (the rule
  list is held fixed), and it is not a proof of robustness to arbitrary MNAR; it
  is a bounded sensitivity analysis over delta ∈ [−2, 2].

## Reporting

Δ(delta), CI, fraction of reps better, MAE and constant MAE by delta for both
sample sizes. The delta at which any criterion first fails is reported even if
the rest of the grid passes.

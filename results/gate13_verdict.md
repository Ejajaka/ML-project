# Gate 13 verdict — the gain survives the MNAR δ-tilt

Pre-registration: `gate13_PREREGISTRATION.md`. Depth-3 rule list frozen at
δ = 0, scoring recomputed under G^exp(δ); 40 fresh seeds per sample size
(150100–150139, 150200–150239).

## n = 8000 (true marginal ATE = 0.041)

| δ | MAE | constant | Δ | 95 % CI | reps better | constant ATE | high-leaf effect |
|---|---|---|---|---|---|---|---|
| −2.0 | 0.0512 | 0.0694 | −0.0182 | [−0.0217, −0.0145] | 90 % | +0.041 | +0.146 |
| −1.0 | 0.0519 | 0.0703 | −0.0184 | [−0.0218, −0.0148] | 92 % | +0.043 | +0.154 |
| 0.0 | 0.0545 | 0.0728 | −0.0183 | [−0.0217, −0.0148] | 95 % | +0.048 | +0.176 |
| +1.0 | 0.0626 | 0.0769 | −0.0142 | [−0.0179, −0.0106] | 90 % | +0.056 | +0.204 |
| +2.0 | 0.0663 | 0.0777 | −0.0114 | [−0.0155, −0.0075] | 85 % | +0.058 | +0.206 |

## n = 12000 (true marginal ATE = 0.041)

| δ | MAE | constant | Δ | 95 % CI | reps better | constant ATE | high-leaf effect |
|---|---|---|---|---|---|---|---|
| −2.0 | 0.0450 | 0.0693 | −0.0243 | [−0.0280, −0.0205] | 92 % | +0.041 | +0.148 |
| −1.0 | 0.0453 | 0.0702 | −0.0249 | [−0.0286, −0.0211] | 92 % | +0.043 | +0.155 |
| 0.0 | 0.0479 | 0.0729 | −0.0250 | [−0.0285, −0.0213] | 95 % | +0.049 | +0.177 |
| +1.0 | 0.0552 | 0.0768 | −0.0216 | [−0.0251, −0.0179] | 95 % | +0.056 | +0.202 |
| +2.0 | 0.0577 | 0.0776 | −0.0199 | [−0.0237, −0.0159] | 92 % | +0.058 | +0.202 |

## Pre-registered decision

**PASS at both sample sizes**: mean Δ < 0 and the 95 % CI entirely below 0 at
every δ in [−2, 2]. The gain is robust to the tested censoring-weight tilt.

## Interpretation

* The tilt moves the **level** of everything: the estimated marginal ATE rises
  from +0.041 to +0.058 and the high-leaf effect from +0.146 to +0.206. It moves
  the **contrast** much less.
* The gain attenuates with positive δ (more MNAR-like censoring): −0.0183 →
  −0.0114 at n = 8000 (−38 %), −0.0250 → −0.0199 at n = 12000 (−20 %). It never
  approaches zero within the grid.
* The mechanism is visible in the table: the rule and the constant are scored
  with the same tilted weights, so a bias common to both largely cancels in the
  comparison. This is why a sensitivity analysis of the *contrast* can be more
  stable than the absolute estimates.

## Limits (do not overclaim)

* The rule list is **frozen at δ = 0**; discovery was not re-run under MNAR.
  This tests the scoring, not the discovery.
* The tilt is a weight perturbation on observed data, not a new MNAR DGP. It is
  a bounded sensitivity analysis over δ ∈ [−2, 2], not a proof of robustness to
  arbitrary MNAR.
* Combined with Gate 5 (per-rule verdicts stable over the same range), the
  weight cap is doing much of the stabilising work.

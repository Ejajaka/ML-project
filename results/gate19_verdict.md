# Gate 19 verdict — hard commitment removes null-benefit treatment at a measured cost

Pre-registration: `gate19_PREREGISTRATION.md`. n = 8000, 40 fresh seeds per cell,
depth-3 ψ discovery. Run on the corrected rule extractor
(`results/CORRECTION_rules_simplify.md`).

## Result

### Reference cell (κ=0, conditional censoring)

| arm | MAE | Δ vs constant | 95 % CI | reps better | abstention | coverage | per-treated | efficiency | captured | null leaves | beneficial reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: none | 0.0484 | −0.0241 | [−0.0292, −0.0186] | 92 % | 0 % | 0.108 | +0.158 | 0.80 | 0.37 | 10 % | 90 % |
| **B: hard commitment** | 0.0571 | −0.0155 | [−0.0203, −0.0105] | 78 % | 30 % | 0.101 | **+0.176** | **0.90** | 0.36 | **0 %** | **100 %** |
| C: EB shrinkage | 0.0587 | −0.0138 | [−0.0199, −0.0080] | 48 % | 0 % | 0.108 | +0.158 | 0.80 | 0.37 | 10 % | 90 % |

### Joint cell (κ=3, marginal KM)

| arm | MAE | Δ vs constant | 95 % CI | reps better | abstention | coverage | per-treated | efficiency | captured | null leaves | beneficial reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: none | 0.0493 | −0.0211 | [−0.0263, −0.0155] | 88 % | 0 % | 0.100 | +0.150 | 0.75 | 0.36 | 12 % | 88 % |
| **B: hard commitment** | 0.0556 | −0.0148 | [−0.0194, −0.0105] | 68 % | **45 %** | 0.088 | **+0.183** | **0.93** | 0.32 | **0 %** | **100 %** |
| C: EB shrinkage | 0.0577 | −0.0128 | [−0.0181, −0.0079] | 52 % | 0 % | 0.100 | +0.150 | 0.75 | 0.36 | 12 % | 88 % |

## Pre-registered checks (joint cell)

| criterion | arm B | arm C |
|---|---|---|
| null-leaf rate ≤ 10 % | **PASS (0 %)** | FAIL (12 %) |
| mean Δ < 0, CI below 0 | PASS | PASS |
| efficiency ≥ 0.60 | **PASS (0.93)** | PASS (0.75) |
| beneficial rate within 5 pp of A | **PASS (100 % vs 88 %)** | PASS (88 %) |
| coverage ≥ 0.05 | PASS (0.088) | PASS (0.100) |
| captured benefit ≥ 0.15 | PASS (0.32) | PASS (0.36) |

**Arm B is adopted. Arm C is not.**

## What commitment buys and costs

Buys:

* null-benefit treatment falls from 12 % to **0 %** of treated replications;
* per-treated benefit rises from +0.150 to **+0.183** (2.9× the treat-all
  marginal of +0.048);
* efficiency rises from 0.75 to **0.93** of the oracle at the same coverage;
* the beneficial-replication rate rises from 88 % to **100 %**.

Costs:

* abstains in **45 %** of replications (reference cell: 30 %);
* coverage falls from 0.100 to 0.088 (a small fall, because abstention removes
  small null leaves);
* captured benefit falls from 0.36 to **0.32** (−11 % relative);
* the MAE advantage falls from +21.1 % to **+14.8 %** (prediction falls back to
  the constant where the leaf is not committed).

So commitment converts wasted treatment into abstention and is **decision-positive
but prediction-negative**. That is the trade-off the gate was designed to expose.

## Why shrinkage failed

Arm C leaves the null-leaf rate unchanged by construction: it never abstains, so
it treats the same patients as A. It also *worsens* MAE slightly, because it
shrinks genuinely large leaf effects toward the marginal — the high leaf's true
effect is +0.15, so pulling it toward +0.048 adds bias. Shrinkage is the right
tool when leaf effects are small and noisy; here they are large and the problem
is identification, not variance.

## Interpretation

The null leaves *are* separable by their intervals: the one-sided test
`τ̂_L > 1.645·SE_L` removes all of them, at the cost of also abstaining on a
minority of genuinely beneficial leaves. The failure mode Gate 18 found is
therefore a **commitment problem**, not an estimation or discovery problem.

## Consequence — v1.7

The model gains a decision layer:

* discovery and prediction unchanged (depth-3 CART on ψ, AIPW leaf effects);
* **hard commitment** for treatment recommendations, using the leaf's own
  one-sided interval;
* paired-IF contrast against the marginal for the prediction fallback.

The predictive claim remains the v1.6 one (corrected numbers); the decision claim
is new and narrower: under the tested conditions, commitment eliminates
null-benefit treatment at a 45 % abstention rate and an 11 % loss of captured
benefit.

## Limits

* 45 % abstention is high; if a fixed-capacity policy is required, the
  coverage/capture frontier matters more than the null rate alone.
* One sample size, two nuisance cells, one commitment rule.
* The commitment test uses the held-out AIPW SE; it is not a sequential
  procedure and has no multiplicity correction across leaves.

# Gate 21 verdict — support-constrained selection wins

Pre-registration: `gate21_PREREGISTRATION.md`. n = 8000, 4000 discovery / 2000
validation / 2000 assessment, 40 fresh seeds (240100–240139). Only the high-leaf
selection rule changes.

## Joint cell (κ=3, marginal KM)

| arm | support | captured (selection) | purity | capture | Jaccard | stability | commit | coverage | efficiency | captured (committed) | null | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: max mean | 446 | 0.356 | 0.69 | 0.35 | 0.33 | 0.04 | 55 % | 0.082 | 0.83 | 0.288 | 5 % | 0.0601 |
| B: LCB | 509 | 0.426 | 0.81 | 0.42 | 0.39 | 0.05 | 62 % | 0.094 | 0.88 | 0.343 | 0 % | 0.0602 |
| **C: support ≥200** | **665** | **0.548** | **0.81** | **0.55** | **0.51** | 0.04 | **80 %** | **0.134** | 0.85 | **0.474** | **0 %** | 0.0602 |
| D: stability-weighted | 665 | 0.273 | 0.41 | 0.27 | 0.23 | 0.02 | 38 % | 0.077 | 0.73 | 0.203 | 13 % | 0.0606 |

## Reference cell (κ=0, conditional)

| arm | captured | purity | capture | Jaccard | commit | coverage | efficiency | captured (committed) | null | MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.229 | 0.63 | 0.23 | 0.22 | 42 % | 0.051 | 0.85 | 0.183 | 0 % | 0.0583 |
| B | 0.353 | 0.78 | 0.35 | 0.33 | 60 % | 0.085 | 0.86 | 0.300 | 0 % | 0.0586 |
| **C** | **0.516** | **0.82** | **0.51** | **0.46** | **80 %** | **0.142** | 0.84 | **0.458** | **0 %** | 0.0587 |
| D | 0.229 | 0.41 | 0.23 | 0.20 | 35 % | 0.049 | 0.78 | 0.171 | 14 % | 0.0588 |

## Paired contrasts (joint cell)

| contrast | captured benefit | purity | stability | MAE |
|---|---|---|---|---|
| B − A | +0.070 [+0.015, +0.131], Holm p = 0.017 | +0.118 | +0.004 | +0.000 |
| **C − A (primary)** | **+0.192 [+0.113, +0.278], p < 0.001** | **+0.122** | −0.001 | +0.000 |
| D − A | −0.083 [−0.207, +0.044], p = 0.195 | −0.286 | −0.019 | +0.001 |

## Pre-registered decision (joint cell)

| criterion | B | **C** | D |
|---|---|---|---|
| capture ≥ A+0.03 or stability ≥ A+0.10 | PASS | **PASS** | FAIL |
| purity ≥ A−0.05 | PASS | **PASS** | FAIL |
| stability ≥ A−0.05 | PASS | **PASS** | PASS |
| MAE Δ < 0 | PASS | **PASS** | PASS |
| MAE ≤ A+0.005 | PASS | **PASS** | FAIL |
| null rate ≤ 5 % | PASS | **PASS** | FAIL |
| efficiency ≥ 0.70 | PASS | **PASS** | PASS |

**C is adopted** as the v1.9 selection rule. B passes too, as secondary evidence;
D fails.

## What this establishes

1. **The winner's curse was a real cause of the low capture.** Requiring a
   discovery support of 200 patients raises captured benefit from 0.356 to 0.548
   (+54 % relative), purity from 0.69 to 0.81, capture from 0.35 to 0.55, and
   Jaccard from 0.33 to 0.51 — with no prediction cost (MAE 0.0601 → 0.0602).
2. **The downstream decision improves as well.** Commitment rate 55 % → 80 %,
   coverage 0.082 → 0.134, captured benefit under commitment 0.288 → 0.474
   (+65 %), null rate 5 % → 0 %, efficiency 0.85. The larger leaf is estimated
   more precisely on validation, so it commits more often and to better effect.
3. **The LCB rule also helps** (captured 0.426, purity 0.81) but less than the
   support constraint; it is significant after Holm adjustment (p = 0.017).
4. **The stability-weighted rule fails.** It selected leaves with purity 0.41 and
   a 13 % null rate; its selection frequency did not transfer to the independent
   evaluation subsamples.

## The stability outcome is uninformative in this gate

The pre-registered stability yardstick refits the **max-mean** rule on 20
disjoint discovery half-samples. That refit is itself winner's-cursed, so it
almost never matches the full-data leaf: stability is 0.02–0.05 for *every* arm.
It therefore cannot discriminate between selection rules and the stability
criterion passes trivially.

This is a **measurement limitation, not evidence that the rule lists are
stable**. The correct follow-up is to measure stability under each arm's *own*
selection rule (e.g. support-constrained refits for C) on disjoint subsamples.
That is the next experiment, not a reason to change this gate's decision.

## Limits

* One sample size, two nuisance cells, one support threshold (200). The
  threshold was not tuned; it was frozen before the run.
* Capture is 0.55, not 1.0: the model still finds a subset, not the region.
* The stability of the v1.9 rule list remains unmeasured by a valid yardstick.
* MAE is essentially unchanged across arms (~0.060), so selection improves
  *who* is identified, not the predictive fit.

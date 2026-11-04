# Gate 25 verdict — the two-leaf list is rejected on purity

Pre-registration: `gate25_PREREGISTRATION.md`. 40 fresh seeds (280100–280139),
two conditions. Models: A (max mean), C200 (single leaf), C400 (sensitivity),
LIST (two leaves). Both leaves come from one CART and are disjoint by
construction; actual overlap is measured (0.00) and reported.

## Clean two-region DGP

| model | leaves | support | captured benefit | purity | J_true | J_full | commit | coverage | per-treated | efficiency | captured (committed) | null | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 345 | 0.205 | 0.79 | 0.174 | 0.173 | 40 % | 0.063 | +0.175 | 0.88 | 0.154 | 0 % | 0.0690 |
| C200 | 1 | 745 | 0.414 | **0.86** | 0.337 | 0.274 | 75 % | 0.165 | +0.168 | 0.87 | 0.367 | 0 % | 0.0689 |
| C400 | 1 | 901 | 0.453 | 0.79 | 0.365 | 0.320 | 82 % | 0.189 | +0.166 | 0.86 | 0.414 | 0 % | 0.0689 |
| **LIST** | **2** | **1332** | **0.584** | **0.70** | **0.419** | **0.356** | **85 %** | **0.251** | +0.158 | 0.87 | **0.489** | **0 %** | 0.0689 |

Fallback to one leaf: 2 % of replications.

## Paired LIST − C200 (clean)

| metric | difference | 95 % CI | p |
|---|---|---|---|
| captured benefit | **+0.169** | [+0.123, +0.218] | <0.001 |
| ground-truth recovery | **+0.083** | [+0.041, +0.127] | <0.001 |
| purity | **−0.166** | [−0.208, −0.125] | <0.001 |
| per-treated benefit | −0.009 | [−0.016, −0.002] | 0.006 |
| captured under commitment | **+0.122** | [+0.069, +0.180] | <0.001 |
| MAE | 0.000 | — | 0.968 |

## Pre-registered checks (clean, LIST vs C200)

| criterion | result |
|---|---|
| capture +0.05 **or** J_true +0.05, CI excluding zero | **PASS** |
| purity ≥ C200 − 0.05 | **FAIL (0.70 vs 0.86, −0.166)** |
| per-treated ≥ C200 − 0.02 | PASS (0.158 vs 0.168) |
| null rate ≤ 5 % | PASS (0 %) |
| efficiency ≥ 0.70 | PASS (0.87) |
| MAE Δ < 0 | PASS |
| MAE ≤ C200 + 0.005 | PASS |

**The LIST is not adopted.** It fails the purity guardrail decisively.

## Stress condition (Gate 24 joint error)

| model | captured benefit | purity | J_true | commit | efficiency | captured (committed) | null |
|---|---|---|---|---|---|---|---|
| C200 | 0.421 | 0.84 | 0.340 | 78 % | 0.85 | 0.379 | 0 % |
| LIST | 0.581 | **0.67** | 0.412 | 85 % | 0.83 | 0.487 | 0 % |

The stress guardrail also fails: purity 0.67 vs C200 0.84.

## Interpretation

1. **The representation limit is real and the second leaf partly fixes it.**
   Capture rises from 0.414 to 0.584 and J_true from 0.337 to 0.419 under the
   clean DGP, with a 2 % fallback rate (the CART almost always has two eligible
   high-support leaves).
2. **The cost is precision.** Purity falls from 0.86 to 0.70. The second leaf is
   the highest-mean remaining high-support leaf; it covers part of the second
   region but includes many non-beneficiaries. The pre-registered tolerance
   (C200 − 0.05) is exceeded by a wide margin.
3. **Decision value rises, decision safety holds.** Captured benefit under
   commitment 0.367 → 0.489, null 0 %, efficiency unchanged at 0.87. Per-treated
   benefit falls only slightly (0.168 → 0.158).
4. **The trade-off is not worth it under the stated criteria.** A larger selected
   set captures more benefit trivially; the gate required the additional capture
   to preserve purity, and it does not.

## What this establishes

The single-leaf limit is **partly representational** — a second leaf does capture
the second region — but the natural two-leaf rule dilutes purity too much. The
single-leaf C200 remains the best defensible model under the pre-registered
purity/decision criteria.

This is the final architectural experiment. Further leaves or thresholds would
be post-hoc tuning of the same trade-off, which the pre-registration excludes.

## State

* v1.11/v1.12 model unchanged: C200 single leaf, depth-3 ψ CART, honest
  commitment.
* Two-leaf list documented as a rejected alternative with its full profile.
* Stop expanding the simulation matrix; move to the ACTG175 illustration and
  paper revision.

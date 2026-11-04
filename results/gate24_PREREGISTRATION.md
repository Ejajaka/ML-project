# Gate 24 pre-registration — joint nuisance stress test on the two-region DGP

Frozen 2026-11-02, before execution. **Pre-registered follow-up to Gate 23.**
Method frozen at v1.11: C200 primary, A comparator, C400 sensitivity only.

## Question

Does the C200 advantage persist when **both** the outcome and censoring nuisance
models are wrong, under the two-region treatment-effect structure?

This is a stress test, not a validity test: with a marginal-KM censoring model
the AIPW estimator's identification conditions are violated, so absolute
estimates may be biased. The question is whether the *relative* selection
advantage and the decision guardrails survive.

## DGP (truth unchanged from Gate 23)

| region | definition | effect |
|---|---|---|
| R1 | cd40 < median ∧ age > median | +0.20 in S(730) |
| R2 | karnof < median ∧ wtkg < median | +0.15 |
| overlap | max offset | +0.20 |
| outside | — | 0 |

**Outcome violation (κ=3, applied to the control arm inside R1):**

    control hazard multiplier for t <= 365:  e^{κ z} / cosh(κ z)
    control hazard multiplier for t >  365:  e^{-κ z} / cosh(κ z)
    z = 0.5 (age_z − cd40_z),  κ = 3

The multiplier integrates to 730 days exactly, so S0(730), S1(730) and the true
τ are unchanged. The control-arm Cox model is misspecified inside R1; the
treated arm and R2 are correct.

**Censoring:** marginal Kaplan–Meier (ignores X), calibrated to 10 % censored
before 730. This is the pre-registered marginal-KM condition.

## Frozen design

| item | value |
|---|---|
| n | 8000 |
| partitions | 4000 discovery / 2000 validation / 2000 assessment |
| replications | 40 fresh seeds |
| seeds | 270100–270139 |
| configs | A, C200 (primary), C400 (sensitivity) |
| discovery | depth-3 CART on ψ, min_samples_leaf 60 |
| selection | max mean; support ≥200 (C200); ≥400 (C400) |
| leaf effects | AIPW on validation (marginal G) |
| commitment | `τ̂_L > 1.645·SE_L` on validation; else no treatment |
| stability | refit each rule's own procedure on 20 disjoint discovery halves, support scaled |
| endpoints | same as Gate 23 |

## Primary contrast and success criteria (unchanged from Gate 23)

C200 vs A, paired within replication, 95 % bootstrap CIs. C200 generalises under
the stress test only if:

1. captured benefit > A with the CI excluding zero;
2. ground-truth recovery (Jaccard with R1 ∪ R2) > A with the CI excluding zero;
3. purity ≥ A − 0.05;
4. prediction guardrail: MAE Δ vs constant < 0 with the CI below zero;
5. decision guardrails: null rate ≤ 5 %, efficiency ≥ 0.70.

C400 is sensitivity only and cannot be adopted.

## Failure interpretation (fixed)

| result | interpretation |
|---|---|
| C200 still beats A on capture/recovery | the selection advantage is robust to joint nuisance error |
| capture improves but ground-truth recovery does not | the target is selecting a useful but different population under bias |
| both fail | the joint error breaks the discovery target, not just the estimates |
| decision guardrails fail but selection holds | selection survives, decision layer does not; report separately |

## Caveat

Marginal-KM censoring violates the AIPW identification condition. Absolute
capture, purity and efficiency are therefore stress-test quantities, not
unbiased estimates of what the method would achieve under correct censoring.
The relative C200-vs-A contrast is the primary inferential object.

## Scope

One DGP, one violation strength, one censoring misspecification, n = 8000.

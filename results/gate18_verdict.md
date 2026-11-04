# Gate 18 verdict — joint misspecification preserves prediction, not subgroup reliability

Pre-registration: `gate18_PREREGISTRATION.md`. n = 8000, 40 fresh seeds. The two
censoring variants are applied to the same simulated data within a replication.

## Full profile

| κ | censoring | ATE bias | Δ MAE gain | Jaccard | purity | capture | high-leaf true τ | effect error | beneficial reps | stability | efficiency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | conditional | −0.0012 | +30.5 % | 0.710 | 0.768 | 0.317 | +0.151 | +0.016 | 87.5 % | 0.178 | 0.766 |
| 0 | marginal | −0.0044 | +30.6 % | 0.689 | 0.687 | 0.364 | +0.134 | −0.015 | 75.0 % | 0.177 | 0.687 |
| 3 | conditional | −0.0013 | +30.1 % | 0.711 | 0.823 | 0.413 | +0.162 | −0.002 | 90.0 % | 0.217 | 0.830 |
| **3** | **marginal (D)** | **−0.0047** | **+31.0 %** | **0.709** | **0.659** | **0.373** | **+0.129** | **−0.017** | **70.0 %** | **0.198** | **0.660** |

`beneficial reps` = fraction of replications in which the high leaf's true mean τ
is strictly greater than zero.

## Pre-registered decision (cell D)

| check | result |
|---|---|
| mean Δ < 0 | PASS |
| 95 % CI entirely below 0 | PASS |
| Jaccard ≥ 0.107 | PASS |
| ≥ 70 % of reps better than constant | PASS (92 %) |
| \|ATE bias\| ≤ 0.015 | PASS (−0.0047) |
| purity ≥ 0.50 | PASS (0.659) |
| efficiency ≥ 0.50 | PASS (0.660) |
| high leaf beneficial in ≥ 90 % of reps | **FAIL (70 %)** |

**Not usable as a subgroup model under joint misspecification by the
pre-registered criteria.** The failure is specific and does not extend to
prediction.

## What the failure means — and what it does not

The high leaf is **never harmful**: `hi_true < 0` in **0 %** of replications in
every cell (minimum exactly 0.000). The failure is that in 30 % of replications
the leaf is **null** — it contains no truly benefiting patient, so the model
treats people who gain nothing.

| cell | high leaf harmful | null (τ ≤ 0) | near-null (τ < 0.05) |
|---|---|---|---|
| A: correct / conditional | 0 % | 12.5 % | 20 % |
| B: correct / marginal | 0 % | 25 % | 25 % |
| C: κ=3 / conditional | 0 % | 10 % | 12 % |
| **D: both wrong** | **0 %** | **30 %** | **30 %** |

So the joint error does not introduce harm; it introduces **wasted treatment**.
Relative to the correct-outcome cell C, the null-leaf rate triples (10 % → 30 %)
and purity falls 0.82 → 0.66.

Two smaller effects are consistent with a mildly wrong marginal censoring model:

* the ATE bias grows from −0.0013 to −0.0047 (still small, but 4× the reference);
* the high-leaf AIPW effect is biased **downward** by −0.017, whereas the
  conditional cells are unbiased (+0.016 and −0.002).

## What survives

* **Prediction:** +31.0 % MAE versus the constant, 92 % of replications, CI below
  zero. The joint misspecification does not break the predictive advantage.
* **Discovery:** Jaccard 0.709, unchanged.
* **Stability:** 0.198, comparable to the reference (0.178) and better than the
  correct-outcome cell C (0.217 is higher, but the marginal cells are 0.18–0.20).
* **Decision value:** per-treated benefit +0.129 against a treat-all marginal of
  +0.048 (2.7×), efficiency 0.66 of the oracle at the same coverage, 37.5 % of
  total available benefit captured while treating 11 % of patients.

## Interpretation

Double robustness protects the **average effect** even when both nuisances are
wrong at this censoring strength; it does not protect the **identity of the
subgroup**. The model still finds a set with high average benefit, but in three
replications in ten that set is empty of true beneficiaries.

That is a rule-quality failure invisible to MAE — precisely the distinction the
gate was designed to expose.

## Consequence

The v1.6 claim must be split:

* **Prediction:** the rule model beats the constant under joint nuisance
  misspecification (+31 %, 92 % of reps).
* **Subgroup identification:** the identified leaf fails the pre-registered
  beneficial-subset bar under joint misspecification (70 % vs the 90 % required),
  even though it is never harmful.

The next model improvement should target the null-leaf rate directly. The
obvious candidate is leaf-level commitment: assign a leaf's effect only if its
interval excludes zero, otherwise fall back to the constant. That uses the
uncertainty machinery CISCaRL already has and would convert wasted treatment
into abstention.

## Limits

One censoring strength (10 %), one censoring misspecification (marginal KM), one
outcome violation (κ=3), n = 8000. Stronger or event-dependent censoring may
degrade more than this.

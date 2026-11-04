# Gate 8 verdict — RMST does not fix discovery

Pre-registration: `gate8_PREREGISTRATION.md`. 30 replications, n = 2000,
identical DGP, folds, discovery, support (60), stability threshold (0.50) and
evaluation split as v1.4. Only the estimand changes. Each estimand is compared
to its own constant baseline, estimated on TRAIN.

## Result

| estimand | ψ sd | regional signal | SNR | rules | Jaccard | stability | MAE | constant | oracle | Δ vs constant | 95 % CI | reps better |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S(730) probability | 0.97 | 0.195 | **0.20** | 1.4 | 0.107 | 0.57 | 0.0790 | 0.0763 | 0.0454 | +0.0027 | [−0.0004, +0.0059] | 23 % |
| RMST(730), days | 441.1 | 83.6 | **0.19** | 2.2 | 0.176 | 0.65 | 35.77 | 32.28 | 21.82 | +3.495 | [+2.154, +4.932] | 20 % |

Δ = MAE(CISCaRL) − MAE(constant), paired within replication.
MAE is in native units per estimand, so rows are not comparable to each other.

## Pre-registered decision

| check | result |
|---|---|
| mean Δ < 0 | **FAIL** (+3.50 days) |
| 95 % CI excludes 0 | **FAIL** — it excludes 0 in the *wrong* direction |
| recovery ≥ 80 % of the day-730 value | PASS (Jaccard 0.176 vs 0.107) |
| ≥ 70 % of reps better | **FAIL** (20 %) |

**Do not adopt RMST.** It is not a wash: CISCaRL is significantly *worse* than the
RMST constant (+10.8 % MAE, CI entirely above zero).

## Why it failed — the mechanism

The pre-registration recorded this risk before running: with exponential event
times, RMST to a fixed horizon is a monotone transform of S(t\*):

    RMST(τ) = (1 − S(τ)) · τ / (−log S(τ))

Both estimands therefore carry the **same information** about the patient. The
individual pseudo-outcome variance is dominated by the variance of the observed
restricted survival time (sd ≈ 441 days), not by how the curve is summarised.
Standardised by the regional signal, the noise is unchanged:

* probability: 0.195 / 0.97 = 0.20
* RMST: 83.6 / 441.1 = 0.19

Integrating the curve did make discovery *slightly* better at ranking patients
(Jaccard 0.107 → 0.176), but per-rule effect assignment on an 84-day contrast
against a 21-day marginal is costlier in absolute days, so overall MAE got worse.

## What this rules out

The "estimand summarisation" family of fixes. Any functional of the same
survival curve — horizon probability, RMST, percentile, restricted mean loss —
is a transform of the same noisy observation. Changing it cannot change the
signal-to-noise ratio, which is what the trees are fighting.

The only remaining lever among the three original failure points is the first
one: the pseudo-outcome itself. Four discovery-side levers are now eliminated:

| lever | gate | outcome |
|---|---|---|
| split criterion | 4e | no change |
| rule granularity (min_support 60→200) | 7 | no MAE gain through ψ |
| ensemble structure | 4e–4g | already optimal |
| estimand (probability → RMST) | **8** | **worse** |

and one is confirmed to work when the target is clean:

| lever | gate | outcome |
|---|---|---|
| discovery target = true τ | 6 | Jaccard 0.98, stability 100 % |

So the target must be *de-noised*, not merely re-expressed.

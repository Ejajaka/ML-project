# Gate 4c — crossing hazards vs the actual SCRE implementation: verdict (reconciled)

Contrast = average treatment effect at **t\* = 730 days**, estimated once per
replication. Real `scre.py`, matched splits and seeds.

## Numerical reconciliation (previously inconsistent)

The earlier table did not reconcile: the marginal MAEs were computed over all
replications while the paired contrast used a subset. Two fixes:

1. **Sign convention made explicit.** `d = MAE(SCRE) − MAE(CISCaRL)`; positive
   favours CISCaRL.
2. **Same subset for both.** All marginals are now computed on the paired rep set,
   and the selected α is always included as a row.

The check column `marginal_check = MAE(SCRE) − MAE(CISCaRL)` now equals `d` exactly
in all three cells (−0.0189, +0.0100, +0.0167).

The earlier 8/40 missing replications at n = 589 came from the selected α falling
outside the admissible sweep; that no longer happens, and **0 % of selected α values
retained zero treatment-interaction terms** at any sample size.

## Tuning protocol (documented in full)

| item | value |
|---|---|
| α selection criterion | validation concordance (C-index) on a 25 % slice of the **training** data |
| evaluation data in selection | **never** |
| admissible α set | path values retaining 1–10 rule-interaction terms |
| sweep | up to 10 evenly spaced values across the admissible set |
| selected α | always reported, even if outside the sweep |
| protocol across sample sizes | identical |

Concordance is a **prognostic** criterion, not an HTE criterion. That is a boundary
on the conclusion, and the sweep is what bounds it. The minimum bias across the sweep
is a diagnostic selected from simulation results, not a deployable estimate.

## Pre-registered prediction — CONFIRMED

> If the bias is structural, it should persist across the α sweep.

| n | reps | share of admissible α with \|bias\|/τ ≥ 10 % | min \|bias\| over the sweep | as % of τ |
|---|---|---|---|---|
| 589 | 40 | **98 %** | 0.0353 | 60 % |
| 1000 | 30 | **100 %** | 0.0493 | 83 % |
| 2000 | 20 | **98 %** | 0.0476 | 80 % |

## Results

| n | true τ | CISCaRL bias | CISCaRL MAE | SCRE bias | SCRE MAE | d [95 % CI] |
|---|---|---|---|---|---|---|
| 589 | 0.0585 | −0.0111 (−19 %) | 0.0656 | **−0.0454 (−77 %)** | **0.0467** | −0.0189 [−0.0368, −0.0019] |
| 1000 | 0.0593 | +0.0051 (+9 %) | **0.0480** | **−0.0579 (−98 %)** | 0.0579 | +0.0100 [−0.0029, +0.0218] |
| 2000 | 0.0595 | +0.0036 (+6 %) | **0.0366** | **−0.0533 (−90 %)** | 0.0533 | +0.0167 [+0.0043, +0.0281] |

## What the evidence supports — and what it does not

**Empirical finding 1.** SCRE's estimate of the 730-day progression-free contrast is
attenuated toward zero by 77–98 % of the true effect across the three evaluated sample
sizes, with a minimum attenuation of 60–83 % of τ over the entire admissible penalty
path.

**Empirical finding 2.** CISCaRL's absolute bias declines over the same range: −19 %
(n = 589) → +9 % (n = 1000) → +6 % (n = 2000).

**Interpretation.** The observed pattern is consistent with the expected behaviour of a
doubly-robust estimator under the stated assumptions, and with attenuation from a
shared-basis Cox model that forces a constant log-hazard ratio.

**Limitation.** The simulation does not establish an asymptotic result: three sample
sizes with 20 replications at n = 2000 show *no clear reduction* in SCRE's bias; they do
not establish a limiting value. Likewise, "CISCaRL is consistent" requires the
consistency argument and its assumptions to be stated separately — the simulation shows
decreasing absolute bias, which is a different claim.

**Not supported:** any statement about a clinical false-null result in ACTG175. The DGP
is synthetic and the effect is small in absolute terms (τ ≈ 0.059), so a −90 % relative
bias is a −0.053 absolute error. Both numbers belong in the paper.

## Selected-α verification (all replications)

`α_sel` is chosen by `scre_select_alpha` under the same constraint used to build the
sweep (1–10 rule-interaction terms), so it lies inside the admissible set. Verified
across **all 90 replications** (40 + 30 + 20):

| check | result |
|---|---|
| α_sel in the admissible set | **100 %** at every sample size |
| terms retained at α_sel | 1–10 at every sample size |
| zero-term selections | **0 %** |

In the earlier run, 8 of 40 replications appeared to drop out because `α_sel` was not
one of the 10 evenly-spaced sweep points — a grid-sampling artefact, not a
model-selection outcome.

`α_sel` is evaluated by the identical routine (`_scre_tau_at`: same fitted Coxnet, same
design builder, same interpolation at t\*), with only α differing. Including it as an
extra row therefore introduces no mismatch in the tuning procedure.

## Numeric audit

Every quantity in this document was recomputed from
`results/gate4c_replications_n*.csv` by an independent script: true τ, both biases, both
MAEs, the paired `d`, both CI bounds, the relative biases, and the α claims. All match
to 5 × 10⁻⁴. `MAE(SCRE) − MAE(CISCaRL)` equals `d` exactly in every cell, which confirms
the two calculations use compatible data and definitions — it is an internal consistency
check, not an independent validation of the estimator.

## The MAE crossover

| n | lower MAE | d [95 % CI] | reading |
|---|---|---|---|
| 589 | SCRE (0.0467) | −0.0189 [−0.0368, −0.0019] | SCRE closer, significant |
| 1000 | CISCaRL (0.0480) | +0.0100 [−0.0029, +0.0218] | tie |
| 2000 | CISCaRL (0.0366) | +0.0167 [+0.0043, +0.0281] | CISCaRL closer, significant |

A bias–variance trade-off with a **crossover between n = 589 and n = 1000**. At the
smallest size SCRE's low variance hides its bias; above ~1000 the bias dominates.

The pre-registered rule was: *if SCRE has larger |bias| but lower MAE, report a
trade-off and do not claim an accuracy advantage.* That applies at n = 589. An
accuracy claim is available only at n = 2000, for this DGP only.

## Caveats

1. **The true effect is small in absolute terms** because the crossing structure
   partially cancels at 730 days by construction. Both the relative and absolute
   biases belong in the paper.
2. **SCRE is wrong here by construction.** The DGP violates PH; this says nothing
   about PH data, where Gate 4 showed SCRE winning.
3. **n = 2000 has 20 replications.** The paired CI excludes zero but the sample is
   small.
4. **The α protocol is ours.** The sweep bounds the tuning concern; it does not
   eliminate it.

## Reproduce

```
python -m ciscarl2.gate4c --reps 40 --n 589
python -m ciscarl2.gate4c --reps 30 --n 1000
python -m ciscarl2.gate4c --reps 20 --n 2000
```

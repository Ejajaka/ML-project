# What CISCaRL v1.3 output actually looks like

Produced by `python -m ciscarl2.demo_actg175` on the real ACTG175 data.

```
CISCaRL v1.3  |  ACTG175 arms 0 vs 1
====================================
Endpoint : progression-free (CD4 decline >=50, AIDS progression, or death)
Horizon  : 730 days
Treatment: 0 = zidovudine, 1 = zidovudine + didanosine
Discovery: train n=737   Evaluation: test n=317 (held out)

OVERALL (all test patients, no subgroup)
========================================
  tau(730) = +0.168   95% CI [+0.060, +0.276]   width 0.215   -> TREAT
  n = 317, SE = 0.055

RULE LIST  (discovered on train, evaluated on 317 test patients)
================================================================
  #  rule                                           n     tau              95% CI   stab  verdict
  ------------------------------------------------------------------------------------------------
  1  wtkg > 79.2                                  108  +0.236   [+0.058, +0.414]   0.65  TREAT
  2  cd80 > 1080.5                                111  +0.216   [+0.032, +0.400]   0.60  TREAT
  3  cd40 <= 477.5                                271  +0.181   [+0.061, +0.301]   0.55  TREAT

PRE-SPECIFIED SUBGROUPS  (full cohort, no discovery, no multiplicity)
=====================================================================
  subgroup                  n     tau              95% CI  half-width  verdict
  --------------------------------------------------------------------------------
  CD4 < 200                88  +0.168   [-0.082, +0.418]       0.250  INSUFFICIENT EVIDENCE
  CD4 >= 200              966  +0.167   [+0.106, +0.227]       0.061  TREAT
  prior ART (str2=1)      618  +0.188   [+0.113, +0.264]       0.076  TREAT
  ART-naive (str2=0)      436  +0.136   [+0.040, +0.232]       0.096  TREAT
  age > 35                477  +0.230   [+0.146, +0.313]       0.084  TREAT
  symptomatic             185  +0.148   [-0.008, +0.303]       0.155  INSUFFICIENT EVIDENCE

Note: endpoint is a composite (CD4 decline >=50 counted), follow-up appears
to end at treatment discontinuation, and intervals are ~93% not 95%.
```

## How to read it

**Overall.** Combination therapy raises 2-year progression-free probability by
about 17 percentage points. This statement is well supported: n = 317, half-width
0.11, the interval excludes zero comfortably. It is the strongest result the data
support.

**Rule list.** Read this section as **unreliable**, and the counters say why:

- `wtkg > 79.2` and `cd80 > 1080.5` are not established effect modifiers for
  antiretroviral therapy. Weight and CD8 count splitting the treatment effect is
  far more likely to be the tree fitting noise in ψ than a real biological signal.
  Gate 2c measured mean IoU 0.19–0.26 against the true effect-driving region, which
  is exactly the failure mode on display.
- `cd40 <= 477.5` selects 85 % of the test set. That is not a subgroup; it is the
  cohort with a threshold on it.
- All three carry `TREAT` with stability 0.55–0.65. Gate 2e found that at
  589-patient scale, TREAT precision was 67 % when harmful subgroups exist. These
  three verdicts should be read with that prior.

**Pre-specified subgroups.** Read this section as **usable**. It is the same
estimator with the discovery step removed, so there is no winner's curse and no
multiplicity:

- The benefit is consistent across every subgroup large enough to resolve
  (+0.14 to +0.23), with no subgroup showing a materially different effect.
- Two subgroups fall below the resolution floor: CD4 < 200 (n = 88) and
  symptomatic (n = 185). Their half-widths are 0.25 and 0.16 — wider than the
  effect, so `INSUFFICIENT EVIDENCE` is correct behaviour, not a failure.
- There is **no evidence of meaningful heterogeneity** in this dataset at this
  horizon. That is a real, reportable finding.

## One caveat about this particular run

This split (737 train / 317 test) is more favourable than the validated operating
regime. Gate 2d used outer 5-fold folds of 118 patients with ~32-patient rules and
observed 87 % abstention. Here the evaluation set is 317 and the rules hold
108–271 patients, so intervals are narrow enough to commit. **Rule size, not
sample size, drives the difference** — which is the same conclusion the power
calculation reached (≈ 460 patients per rule for 80 % power).

## The two-sentence version for a clinician

> In ACTG175, zidovudine + didanosine improves 2-year progression-free probability
> by about 17 percentage points, and the benefit is consistent across every
> subgroup large enough to evaluate. Subgroups smaller than roughly 200 patients
> cannot be assessed with this dataset.

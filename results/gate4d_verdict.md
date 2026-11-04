# Gate 4d — piecewise-constant DGP: verdict

**Pre-registered decision: RULE-SHAPED ADVANTAGE NOT SUPPORTED**, at both sample sizes.

Pre-registration: `results/gate4d_PREREGISTRATION.md` (with Amendment 1, made before any
results were examined: the clinical cut-points gave a 1 % region because ACTG175 enrolled
CD4 200–500, so the region was moved to cohort medians, prevalence 25 %).

## The decision rule, evaluated

| criterion | n = 589 | n = 2000 |
|---|---|---|
| Δ region MAE > 0, CI excludes zero | **PASS** +0.034 [+0.001, +0.067] | **FAIL** +0.015 [−0.006, +0.035] |
| sensitivity ≥ 0.60 | **FAIL** 0.11 [0.02, 0.20] | **FAIL** 0.28 [0.11, 0.46] |
| false-positive TREAT ≤ 0.10 | **PASS** 0.041 | **FAIL** 0.321 |

Two of three fail at n = 2000, and the decisive criterion — can the method *find* the
region — fails at both. **CISCaRL does not identify the true subgroup**, even when the
true effect surface is a threshold rule, which is its own model class.

## What the numbers show

| metric | CISCaRL 589 | SCRE 589 | CISCaRL 2000 | SCRE 2000 |
|---|---|---|---|---|
| region MAE | **0.149** | 0.184 | 0.137 | 0.152 |
| overall MAE | 0.124 | **0.064** | 0.082 | **0.058** |
| sensitivity | 0.11 | **0.66** | 0.28 | **0.85** |
| specificity | **0.96** | 0.40 | 0.68 | 0.54 |
| false-positive TREAT | **0.04** | 0.60 | 0.32 | 0.46 |
| commitment rate | 9 % | 100 % | 32 % | 100 % |
| directional precision | 75 % | 66 % | 88 % | 85 % |

**Three findings.**

**1. CISCaRL's region estimates are modestly better than SCRE's, and the gap closes
with n.** 0.149 vs 0.184 at n = 589 (CI excludes zero); 0.137 vs 0.152 at n = 2000
(CI includes zero). This is the first direct evidence for CISCaRL's estimation inside
its own model class — and it is small and not durable.

**2. SCRE can represent a step function after all.** Its design includes
rule×treatment terms, so a piecewise-constant truth is in its span. At n = 2000 it
matches CISCaRL in the region. The "home turf" hypothesis was wrong.

**3. The two methods fail in opposite directions.** This is the most interesting
result in the gate:

- **SCRE over-treats.** Sensitivity 0.66–0.85, but false-positive TREAT 0.46–0.60 —
  it recommends treatment to roughly half the patients who gain nothing.
- **CISCaRL under-treats.** Specificity 0.96, but sensitivity 0.11 — it abstains on
  ~90 % of patients and catches only about one in ten who would benefit.

CISCaRL's high specificity is **abstention, not discrimination**. A policy that commits
to almost nobody is trivially specific. The honest description is: it estimates the
region effect reasonably but does not act on it.

## Why overall MAE is so much worse than region MAE

CISCaRL assigns a rule's effect to *everyone* in that rule, including the 75 % null
region. A coarse partition cannot separate the region cleanly, so it puts non-zero
effect estimates on patients whose true effect is zero. That is the cost of the
piecewise-constant form when the rule does not align with the truth — 0.124 vs SCRE's
0.064 overall, despite being slightly better *inside* the region.

## Consequence for the project

This is the third pre-registered gate that has failed to establish an advantage for
CISCaRL's model class:

| gate | result |
|---|---|
| 4 (linear DGPs) | SCRE lower MAE in 3 of 4 cells |
| 4c (crossing hazards) | SCRE 77–98 % bias persisting across α, but lower MAE at n = 589 |
| **4d (piecewise-constant)** | **CISCaRL slightly better in-region; cannot identify the region; worse overall** |

Taken together, the evidence does not support the claim that a rule list is the right
representation for survival HTE at these sample sizes. What it *does* support is the
opposite-direction failure modes: SCRE over-treats, CISCaRL under-treats, and the
resolution problem is sample size, not architecture.

That is a legitimate paper — *"the bias–coverage trade-off in interpretable survival
subgroup analysis"* — but it is not the paper the project set out to write.

## Caveats

- 30 and 20 replications; the sensitivity CI at n = 589 is [0.02, 0.20], so the point
  estimate 0.11 is itself uncertain.
- Region defined at medians (prevalence 25 %), not the original clinical cut-points.
- SCRE is evaluated on its sign, since it has no abstention; CISCaRL on its verdicts.
  That asymmetry is inherent to the methods, not a choice.
- Commitment rate and interval coverage are separate; only commitment is reported here.

## Reproduce

```
python -m ciscarl2.gate4d --reps 30 --n 589
python -m ciscarl2.gate4d --reps 20 --n 2000
```

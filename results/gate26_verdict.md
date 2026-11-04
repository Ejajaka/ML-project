# Gate 26 verdict — flexible DR-learner + distillation does not beat C200

Pre-registration: `gate26_PREREGISTRATION.md`. 40 fresh seeds (290100–290139),
two DGPs, clean nuisances, honest 4000/2000/2000 partition.

## Result

### Single-region DGP

| arm | support | captured benefit | purity | J_true | J_full | commit | coverage | per-treated | efficiency | captured (committed) | null | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C200 | 692 | **0.548** | 0.83 | **0.500** | 0.302 | 68 % | 0.125 | +0.177 | 0.90 | **0.450** | 0 % | **0.0525** |
| FLEX | 510 | 0.454 | **0.90** | 0.427 | 0.318 | 80 % | 0.110 | +0.182 | 0.92 | 0.399 | 0 % | 0.0572 |

### Two-region DGP

| arm | support | captured benefit | purity | J_true | J_full | commit | coverage | per-treated | efficiency | captured (committed) | null | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C200 | 646 | **0.372** | 0.83 | **0.307** | 0.262 | 68 % | 0.129 | +0.168 | 0.85 | **0.305** | 0 % | 0.0690 |
| FLEX | 478 | 0.299 | **0.89** | 0.252 | 0.276 | 80 % | 0.102 | +0.174 | 0.87 | 0.256 | 0 % | 0.0698 |

## Paired FLEX − C200

| metric | single-region | two-region |
|---|---|---|
| captured benefit | **−0.094 [−0.178, −0.011]** | **−0.073 [−0.129, −0.022]** |
| ground-truth recovery | −0.073 [−0.150, +0.006] | **−0.055 [−0.101, −0.013]** |
| purity | **+0.068 [+0.012, +0.139]** | **+0.058 [+0.005, +0.114]** |
| reference recovery | +0.016 [−0.018, +0.051] | +0.014 [−0.019, +0.052] |
| MAE | +0.005 [+0.001, +0.009] | +0.001 [−0.003, +0.004] |

## Pre-registered checks

FLEX fails the primary criterion in both DGPs (no capture or J_true improvement
of +0.05). It passes the purity, per-treated, null, efficiency and MAE-relative
guardrails. **FLEX is not adopted; stage 2 (stability + policy) is not built.**

## Interpretation

The flexible DR-learner + distillation produces a **purer but narrower** rule:

* purity rises 0.83 → 0.89–0.90;
* capture and ground-truth recovery fall;
* the selected leaf is smaller (support ~480–510 vs ~650–690);
* commitment rises (80 % vs 68 %) because the smaller leaf is estimated more
  precisely on validation, but the committed captured benefit falls
  (0.450 → 0.399; 0.305 → 0.256);
* prediction is slightly worse or unchanged.

So the flexible surface does not find more of the truth here; the distillation
compresses it into a more conservative rectangular rule. The single-leaf ψ-CART
C200 remains the better model.

## Important confound (recorded, not fixed post hoc)

The pre-registered distillation used `min_samples_leaf = 200`, while C200's CART
uses `min_samples_leaf = 60` and then filters support ≥200. The FLEX arm is
therefore **more constrained than C200**, and part of its lower capture may come
from the distillation minimum rather than from the flexible learner. A fairer
test would match the distillation minimum to C200's 60 and apply the same
support filter. That test was not pre-registered and is **not** run here.

The negative result generalises to this configuration, not to every possible
distillation of a flexible survival HTE surface.

## What this means for v2

Stage 1 does not justify the stability + policy layer. Building stage 2 on a
surface that already captures less would optimize a weaker rule. The evidence
does not support replacing C200.

If the flexible-learner direction is pursued later, the next step is the fair
distillation test (`min_samples_leaf = 60`, support ≥200), and it should be
pre-registered separately.

## Decision

* C200 remains the frozen model.
* v2 stage 1 is recorded as a negative result.
* The simulation program stops here; move to the ACTG175 illustration and the
  paper.

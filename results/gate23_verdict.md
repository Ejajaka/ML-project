# Gate 23 verdict — v1.10 generalises to a two-region treatment effect

Pre-registration: `gate23_PREREGISTRATION.md`. n = 8000, 4000/2000/2000, 40
fresh seeds (260100–260139), censoring and nuisances fixed. Only the effect
structure changes: R1 (`cd40<median ∧ age>median`) +0.20, R2
(`karnof<median ∧ wtkg<median`) +0.15, overlap takes the larger offset,
outside 0.

## Result

| config | support | captured benefit | purity | region coverage | J_true | J_full | recovery freq | commit | coverage | efficiency | captured (committed) | null | MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: max mean | 248 | 0.148 | 0.77 | 0.13 | 0.127 | 0.160 | 3 % | 45 % | 0.040 | 0.87 | 0.098 | 0 % | 0.0695 |
| B: LCB | 311 | 0.194 | 0.83 | 0.17 | 0.166 | 0.205 | 4 % | 55 % | 0.057 | 0.88 | 0.141 | 0 % | 0.0697 |
| C100 | 433 | 0.246 | 0.85 | 0.22 | 0.201 | 0.197 | 4 % | 62 % | 0.084 | 0.87 | 0.189 | 0 % | 0.0698 |
| **C200 (v1.10)** | **677** | **0.364** | **0.82** | **0.33** | **0.295** | **0.275** | 5 % | **70 %** | **0.129** | 0.84 | **0.284** | **0 %** | 0.0695 |
| C400 (sensitivity) | 916 | 0.441 | 0.78 | 0.40 | 0.345 | 0.314 | 5 % | 75 % | 0.162 | 0.82 | 0.345 | 0 % | 0.0697 |
| D: stability-weighted | 566 | 0.210 | 0.67 | 0.21 | 0.179 | 0.156 | 3 % | 52 % | 0.084 | 0.67 | 0.149 | 0 % | 0.0699 |

## Primary contrast C200 vs A

| metric | difference | 95 % CI | p |
|---|---|---|---|
| captured benefit | **+0.216** | [+0.142, +0.291] | <0.001 |
| ground-truth recovery (J_true) | **+0.168** | [+0.112, +0.224] | <0.001 |
| reference-leaf recovery (J_full) | **+0.115** | [+0.078, +0.153] | <0.001 |
| purity | +0.046 | [−0.035, +0.132] | 0.281 |
| MAE | −0.000 | — | 0.844 |

## Pre-registered success checks

| criterion | result |
|---|---|
| captured benefit > A | **PASS** |
| ground-truth recovery > A | **PASS** |
| purity ≥ A − 0.05 | **PASS (0.82 vs 0.77)** |
| prediction guardrail | **PASS** |
| null rate ≤ 5 % | **PASS (0 %)** |
| efficiency ≥ 0.70 | **PASS (0.84)** |

**C200 generalises.** Under the fixed interpretation table this is the
"improves capture and recovery" case: the support constraint is not specific to
the single-region DGP.

## What the two-region DGP reveals

1. **The winner's curse is stronger here.** A selects a 248-patient leaf and
   captures only 0.148 of the benefit; C200 selects a 677-patient leaf and
   captures 0.364. The support constraint binds in most replications (the 2-rep
   smoke happened to select the same leaf and was misleading).
2. **The gain is larger in relative terms** (+0.216 capture, +0.168 J_true) but
   **absolute capture is lower** than in the single-region DGP (0.364 vs 0.623).
   That is the representation limit, not a selection failure: one leaf cannot
   cover two separated regions.
3. **Purity is maintained** (0.82) and the decision layer remains safe
   (70 % commit, 0 % null, 0.84 efficiency, 0.284 committed captured benefit).
4. **More support helps further** (C400: capture 0.441, J_true 0.345), but the
   primary rule remains 200 by freeze; 400 is sensitivity only.
5. **The stability-weighted rule still fails** (purity 0.67, efficiency 0.67).

## What this establishes

The v1.10 improvement travels to a treatment-effect structure the method was not
designed around. Combined with Gates 21–22, this supports the claim that the
support constraint addresses a real selection problem rather than fitting one
simulation.

## What it does not establish

* Full recovery: absolute capture 0.364 under two regions. The single-leaf
  representation is the limiting factor when the truth has multiple regions.
* A stable rule list: recovery frequency is 5 % at J > 0.8.
* Robustness beyond this second DGP, n = 8000, and conditional censoring. The
  pre-registered next robustness cell (marginal-KM censoring under the two-region
  DGP) has not been run.

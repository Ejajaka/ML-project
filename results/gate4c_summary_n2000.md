# Gate 4c - crossing hazards vs the actual SCRE implementation

Contrast = average treatment effect at t* = 730, estimated once per replication. **All MAEs are on the same paired rep subset**, so `MAE(SCRE) - MAE(CISCaRL)` reconciles exactly with d.

d = MAE(SCRE) - MAE(CISCaRL): **positive favours CISCaRL**.

| n | reps (of total) | zero-term reps | true tau | CISCaRL bias | CISCaRL MAE | SCRE bias | SCRE MAE | min abs bias over sweep | persist frac | d [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 2000.0 | 20.0 (20.0) | 0 % | 0.0595 | +0.0036 | 0.0366 | -0.0533 | 0.0533 | 0.0476 | 98 % | +0.0167 [+0.0043, +0.0281] |

persist frac = share of admissible alphas with |bias| / |tau| >= 10%. zero-term reps = share where the validation-selected alpha retained no treatment-interaction term.
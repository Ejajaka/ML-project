# Gate 4c - crossing hazards vs the actual SCRE implementation

Contrast = average treatment effect at t* = 730, estimated once per replication. **All MAEs are on the same paired rep subset**, so `MAE(SCRE) - MAE(CISCaRL)` reconciles exactly with d.

d = MAE(SCRE) - MAE(CISCaRL): **positive favours CISCaRL**.

| n | reps (of total) | zero-term reps | true tau | CISCaRL bias | CISCaRL MAE | SCRE bias | SCRE MAE | min abs bias over sweep | persist frac | d [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000.0 | 30.0 (30.0) | 0 % | 0.0593 | +0.0051 | 0.0480 | -0.0579 | 0.0579 | 0.0493 | 100 % | +0.0100 [-0.0029, +0.0218] |

persist frac = share of admissible alphas with |bias| / |tau| >= 10%. zero-term reps = share where the validation-selected alpha retained no treatment-interaction term.
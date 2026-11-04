# Gate 4c - crossing hazards vs the actual SCRE implementation

Contrast = average treatment effect at t* = 730, estimated once per replication. **All MAEs are on the same paired rep subset**, so `MAE(SCRE) - MAE(CISCaRL)` reconciles exactly with d.

d = MAE(SCRE) - MAE(CISCaRL): **positive favours CISCaRL**.

| n | reps (of total) | zero-term reps | true tau | CISCaRL bias | CISCaRL MAE | SCRE bias | SCRE MAE | min abs bias over sweep | persist frac | d [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 589.0 | 40.0 (40.0) | 0 % | 0.0585 | -0.0111 | 0.0656 | -0.0454 | 0.0467 | 0.0353 | 98 % | -0.0189 [-0.0368, -0.0019] |

persist frac = share of admissible alphas with |bias| / |tau| >= 10%. zero-term reps = share where the validation-selected alpha retained no treatment-interaction term.
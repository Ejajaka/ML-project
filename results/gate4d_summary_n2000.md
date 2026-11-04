# Gate 4d - piecewise-constant DGP

Pre-registered rule (`results/gate4d_PREREGISTRATION.md`): advantage supported only if Delta region MAE > 0 with CI excluding zero, sensitivity >= 0.60, and false-positive TREAT <= 0.10.

All rows are computed on the SAME held-out test patients from the same replication. Overall MAE is the mean over those patients, so the 25 % / 75 % region weighting is implicit and identical everywhere. The constant is the marginal ATE estimated on TRAIN and applied to every test patient.

| n | reps | region | tau in/out | const ATE | **overall MAE: const / CISCaRL / SCRE / oracle** | region MAE: const / CIS / SCRE / oracle |
|---|---|---|---|---|---|---|
| 2000.0 | 20.0 | 25 % | 0.194 / 0.000 | +0.050 | **0.0743 / 0.0820 / 0.0581 / 0.0444** | 0.1445 / 0.1374 / 0.1520 / 0.0660 |

Delta region MAE = MAE(SCRE) - MAE(CISCaRL) on the true region; positive favours CISCaRL.
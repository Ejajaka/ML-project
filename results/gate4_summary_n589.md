# Gate 4 - CISCaRL v1.4 vs SCRE (actual implementation)

Matched splits, horizon, and evaluation patients. Two questions separated.

## A. Estimation accuracy (patient-level tau)

| n | DGP | method | MAE | RMSE | bias | Spearman | sign acc | rules |
|---|---|---|---|---|---|---|---|---|
| 589 | all-beneficial | CISCaRL | 0.1302 | 0.1612 | +0.0157 | +0.32 | 79 % | 7.8 |
| 589 | all-beneficial | SCRE | 0.1074 | 0.1509 | -0.1054 | +0.38 | 72 % | 1.2 |

## B. Decision reliability

| n | DGP | base rate | CISCaRL coverage | CISCaRL precision | SCRE coverage | SCRE sign acc |
|---|---|---|---|---|---|---|
| 589 | all-beneficial | 96 % | 26.5 % | 98 % | 100 % | 72 % |

SCRE commits for every patient by construction (no interval). CISCaRL coverage is the share receiving TREAT or AVOID.

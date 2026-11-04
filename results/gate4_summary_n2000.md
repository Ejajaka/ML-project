# Gate 4 - CISCaRL v1.4 vs SCRE (actual implementation)

Matched splits, horizon, and evaluation patients. Two questions separated.

## A. Estimation accuracy (patient-level tau)

| n | DGP | method | MAE | RMSE | bias | Spearman | sign acc | rules |
|---|---|---|---|---|---|---|---|---|
| 2000 | all-beneficial | CISCaRL | 0.0978 | 0.1294 | +0.0148 | +0.26 | 93 % | 2.3 |
| 2000 | all-beneficial | SCRE | 0.1081 | 0.1344 | -0.1081 | +0.71 | 58 % | 1.4 |

## B. Decision reliability

| n | DGP | base rate | CISCaRL coverage | CISCaRL precision | SCRE coverage | SCRE sign acc |
|---|---|---|---|---|---|---|
| 2000 | all-beneficial | 96 % | 78.4 % | 96 % | 100 % | 58 % |

SCRE commits for every patient by construction (no interval). CISCaRL coverage is the share receiving TREAT or AVOID.

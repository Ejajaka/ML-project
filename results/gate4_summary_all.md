# Gate 4 - CISCaRL v1.4 vs SCRE (actual implementation)

Matched splits, horizon, and evaluation patients. Two questions separated.

## A. Estimation accuracy (patient-level tau)

| n | DGP | method | MAE | RMSE | bias | Spearman | sign acc | rules |
|---|---|---|---|---|---|---|---|---|
| 589 | all-beneficial | CISCaRL | 0.1302 | 0.1612 | +0.0157 | +0.32 | 79 % | 7.8 |
| 589 | all-beneficial | SCRE | 0.1074 | 0.1509 | -0.1054 | +0.38 | 72 % | 1.2 |
| 589 | sign-flip | CISCaRL | 0.1162 | 0.1412 | -0.0105 | +0.38 | 65 % | 7.7 |
| 589 | sign-flip | SCRE | 0.0397 | 0.0580 | +0.0073 | +0.87 | 88 % | 1.1 |
| 2000 | all-beneficial | CISCaRL | 0.0978 | 0.1294 | +0.0148 | +0.26 | 93 % | 2.3 |
| 2000 | all-beneficial | SCRE | 0.1081 | 0.1344 | -0.1081 | +0.71 | 58 % | 1.4 |
| 2000 | sign-flip | CISCaRL | 0.0864 | 0.1086 | -0.0173 | +0.47 | 58 % | 2.5 |
| 2000 | sign-flip | SCRE | 0.0249 | 0.0370 | +0.0026 | +0.95 | 93 % | 1.6 |

## B. Decision reliability

| n | DGP | base rate | CISCaRL coverage | CISCaRL precision | SCRE coverage | SCRE sign acc |
|---|---|---|---|---|---|---|
| 589 | all-beneficial | 96 % | 26.5 % | 98 % | 100 % | 72 % |
| 589 | sign-flip | 46 % | 12.8 % | 88 % | 100 % | 88 % |
| 2000 | all-beneficial | 96 % | 78.4 % | 96 % | 100 % | 58 % |
| 2000 | sign-flip | 46 % | 23.1 % | 50 % | 100 % | 93 % |

SCRE commits for every patient by construction (no interval). CISCaRL coverage is the share receiving TREAT or AVOID.

# Gate 4 - CISCaRL v1.4 vs SCRE (actual implementation)

Matched splits, horizon, and evaluation patients. Two questions separated.

## A. Estimation accuracy (patient-level tau)

| n | DGP | method | MAE | RMSE | bias | Spearman | sign acc | rules |
|---|---|---|---|---|---|---|---|---|
| 2000 | sign-flip | CISCaRL | 0.0864 | 0.1086 | -0.0173 | +0.47 | 58 % | 2.5 |
| 2000 | sign-flip | SCRE | 0.0249 | 0.0370 | +0.0026 | +0.95 | 93 % | 1.6 |

## B. Decision reliability

| n | DGP | base rate | CISCaRL coverage | CISCaRL precision | SCRE coverage | SCRE sign acc |
|---|---|---|---|---|---|---|
| 2000 | sign-flip | 46 % | 23.1 % | 50 % | 100 % | 93 % |

SCRE commits for every patient by construction (no interval). CISCaRL coverage is the share receiving TREAT or AVOID.

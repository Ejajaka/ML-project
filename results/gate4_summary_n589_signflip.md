# Gate 4 - CISCaRL v1.4 vs SCRE (actual implementation)

Matched splits, horizon, and evaluation patients. Two questions separated.

## A. Estimation accuracy (patient-level tau)

| n | DGP | method | MAE | RMSE | bias | Spearman | sign acc | rules |
|---|---|---|---|---|---|---|---|---|
| 589 | sign-flip | CISCaRL | 0.1162 | 0.1412 | -0.0105 | +0.38 | 65 % | 7.7 |
| 589 | sign-flip | SCRE | 0.0397 | 0.0580 | +0.0073 | +0.87 | 88 % | 1.1 |

## B. Decision reliability

| n | DGP | base rate | CISCaRL coverage | CISCaRL precision | SCRE coverage | SCRE sign acc |
|---|---|---|---|---|---|---|
| 589 | sign-flip | 46 % | 12.8 % | 88 % | 100 % | 88 % |

SCRE commits for every patient by construction (no interval). CISCaRL coverage is the share receiving TREAT or AVOID.

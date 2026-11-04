# Gate 2c - stability-threshold sweep

Discovery metrics were defined before the run. Coverage is conditional on a rule being discovered; null replications are counted as discovery failures, not as coverage zeros.

| cell | thr | null | rules | mean freq | top separation | mean separation | mean IoU | coverage (top) | coverage (all rules) | width |
|---|---|---|---|---|---|---|---|---|---|---|
| covariate | 0.40 | 3 % | 4.3 | 0.54 | 0.81 | 0.62 | 0.19 | 93.8 % | 96.3 % | 0.549 |
| covariate | 0.50 | 10 % | 3.0 | 0.61 | 0.79 | 0.68 | 0.20 | 93.3 % | 94.5 % | 0.545 |
| covariate | 0.60 | 22 % | 1.9 | 0.70 | 0.83 | 0.81 | 0.22 | 93.6 % | 94.2 % | 0.554 |
| covariate | 0.70 | 57 % | 1.4 | 0.80 | 0.97 | 0.97 | 0.25 | 95.3 % | 96.0 % | 0.558 |
| independent | 0.40 | 0 % | 4.2 | 0.56 | 1.13 | 0.68 | 0.21 | 93.0 % | 94.4 % | 0.347 |
| independent | 0.50 | 2 % | 2.9 | 0.63 | 1.15 | 0.84 | 0.24 | 92.9 % | 94.7 % | 0.347 |
| independent | 0.60 | 11 % | 1.9 | 0.70 | 1.24 | 0.98 | 0.26 | 92.1 % | 92.1 % | 0.345 |
| independent | 0.70 | 39 % | 1.3 | 0.78 | 1.31 | 1.22 | 0.29 | 90.2 % | 90.6 % | 0.350 |

## Paired differences vs the 0.60 baseline (same replications)

| cell | thr | d null (pp) | d separation | d IoU |
|---|---|---|---|---|
| independent | 0.40 | -11.0 +/- 3.1 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| independent | 0.50 | -9.0 +/- 2.9 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| independent | 0.70 | +28.0 +/- 4.5 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| covariate | 0.40 | -19.0 +/- 3.9 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| covariate | 0.50 | -12.0 +/- 3.3 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| covariate | 0.70 | +35.0 +/- 4.8 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
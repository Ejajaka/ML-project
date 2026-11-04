# Gate 2c - stability-threshold sweep

Discovery metrics were defined before the run. Coverage is conditional on a rule being discovered; null replications are counted as discovery failures, not as coverage zeros.

| cell | thr | null rate | rules | top freq | separation | IoU | precision | recall | coverage | width |
|---|---|---|---|---|---|---|---|---|---|---|
| covariate | 0.40 | 0 % | 4.2 | 0.67 | 0.63 | 0.19 | 0.46 | 0.23 | 98.0 % | 0.545 |
| covariate | 0.50 | 4 % | 3.1 | 0.68 | 0.62 | 0.19 | 0.46 | 0.23 | 97.9 % | 0.547 |
| covariate | 0.60 | 23 % | 1.9 | 0.72 | 0.58 | 0.19 | 0.45 | 0.23 | 97.4 % | 0.537 |
| covariate | 0.70 | 54 % | 1.3 | 0.78 | 0.68 | 0.20 | 0.47 | 0.24 | 95.7 % | 0.555 |
| independent | 0.40 | 0 % | 4.3 | 0.69 | 1.06 | 0.26 | 0.58 | 0.30 | 93.0 % | 0.351 |
| independent | 0.50 | 7 % | 2.8 | 0.70 | 1.09 | 0.26 | 0.59 | 0.30 | 93.5 % | 0.350 |
| independent | 0.60 | 25 % | 2.0 | 0.75 | 1.17 | 0.28 | 0.61 | 0.32 | 92.0 % | 0.348 |
| independent | 0.70 | 42 % | 1.4 | 0.79 | 1.17 | 0.28 | 0.61 | 0.32 | 89.7 % | 0.346 |

## Paired differences vs the 0.60 baseline (same replications)

| cell | thr | d null (pp) | d separation | d IoU |
|---|---|---|---|---|
| independent | 0.40 | -25.0 +/- 4.4 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| independent | 0.50 | -18.0 +/- 3.9 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| independent | 0.70 | +17.0 +/- 3.8 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| covariate | 0.40 | -23.0 +/- 4.2 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| covariate | 0.50 | -19.0 +/- 3.9 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
| covariate | 0.70 | +31.0 +/- 4.6 | +0.000 +/- 0.000 | +0.000 +/- 0.000 |
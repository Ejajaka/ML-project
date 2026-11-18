# Simulated operating range at the ACTG175 sample size

Frozen C200 pipeline, known single-region / two-region truth;
20 reps per (n, DGP); seeds 330100-330119.

| n | DGP | commit | purity (all) | purity (committed) | capture | per-treated | MAE |
|---|---|---|---|---|---|---|---|
| 2139 | single | 15.0% | 0.505 | 0.505 | 0.080 | +0.099 | 0.0963 |
| 2139 | two | 20.0% | 0.587 | 0.587 | 0.124 | +0.105 | 0.1001 |
| 4000 | single | 50.0% | 0.732 | 0.732 | 0.287 | +0.143 | 0.0807 |
| 4000 | two | 50.0% | 0.651 | 0.651 | 0.239 | +0.116 | 0.0871 |
| 8000 | single | 75.0% | 0.885 | 0.885 | 0.458 | +0.173 | 0.0612 |
| 8000 | two | 80.0% | 0.780 | 0.780 | 0.311 | +0.148 | 0.0712 |

# C200-always ablation (40 reps, n = 8,000)

C200-always = selected leaf always treated (no LCB/commitment layer).
C200-commit is excluded from this table. SCRE is the comparator.

| DGP | method | reps | MAE | purity | capture | per-treated | coverage | null |
|---|---|---|---|---|---|---|---|---|
| single | C200-always | 40 | 0.0560 | 0.834 | 0.627 | +0.163 | 0.187 | 0.0% |
| single | SCRE | 40 | 0.0519 | 0.433 | 0.947 | +0.084 | 0.544 | 0.0% |
| two | C200-always | 40 | 0.0713 | 0.847 | 0.374 | +0.163 | 0.157 | 0.0% |
| two | SCRE | 40 | 0.0685 | 0.585 | 0.802 | +0.098 | 0.560 | 0.0% |

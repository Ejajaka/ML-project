# Gate 28 verdict — C200 vs SCRE, same data and assessment

Pre-registration: `gate28_PREREGISTRATION.md`. 40 fresh seeds per DGP.
SCRE fitted on discovery 4000, α selected on validation 2000 by concordance,
predicting the untouched assessment 2000. C200 frozen. 100 % of SCRE fits
succeeded.

## Results

| DGP | method | MAE | Spearman | capture | purity | per-treated | efficiency | null | coverage | rules |
|---|---|---|---|---|---|---|---|---|---|---|
| single | C200 | 0.0562 | 0.598 | 0.591 | **0.86** | **+0.168** | 0.88 | 0 % | 0.175 | 1.0 |
| single | SCRE | **0.0519** | 0.596 | **0.947** | 0.43 | +0.084 | 0.95 | 0 % | 0.544 | 1.3 |
| two | C200 | 0.0715 | 0.441 | 0.334 | **0.87** | **+0.168** | 0.85 | 0 % | 0.140 | 1.0 |
| two | SCRE | **0.0685** | 0.492 | **0.802** | 0.58 | +0.098 | 0.80 | 0 % | 0.560 | 1.2 |

## Paired C200 − SCRE

| DGP | metric | difference | 95 % CI | p |
|---|---|---|---|---|
| single | MAE | **+0.0043** | [+0.0002, +0.0084] | 0.046 |
| single | Spearman | +0.0017 | [−0.0435, +0.0427] | 0.908 |
| single | capture | **−0.3565** | [−0.4597, −0.2625] | <0.001 |
| two | MAE | +0.0031 | [−0.0007, +0.0068] | 0.112 |
| two | Spearman | −0.0506 | [−0.1106, +0.0100] | 0.110 |
| two | capture | **−0.4677** | [−0.5400, −0.3932] | <0.001 |

## Verdict

**C200 does not beat SCRE on patient-level MAE.** In the single-region DGP SCRE
is significantly better (C200 worse by 0.0043, CI above zero); in the
two-region DGP SCRE is slightly better and the difference is not significant.
Ranking (Spearman) is tied in the single-region and slightly better for SCRE in
the two-region.

**C200 is a selective, high-purity policy, not a better global predictor.** It
treats 14–18 % of patients against SCRE's 54–56 %, and therefore captures much
less total benefit (0.33–0.59 vs 0.80–0.95). What it buys is precision:
purity 0.86–0.87 vs 0.43–0.58, and per-treated benefit +0.168 vs +0.084/+0.098.
Efficiency is mixed: SCRE is higher in the single-region (0.95 vs 0.88), C200 in
the two-region (0.85 vs 0.80).

## What this changes

* The claim "CISCaRL outperforms SCRE on HTE accuracy" is **false** and must be
  removed.
* The defensible claim is narrower: **comparable-to-slightly-worse patient-level
  accuracy, with much higher subgroup purity and an explicit abstention layer
  SCRE does not provide.**
* The structural gaps fixed earlier still stand: non-PH bias (Gate 4c), pooled
  Cox bias (Gate 1c), degenerate α selection, and the absence of abstention.
* Note: with validation-concordance α selection SCRE selects ~1.2 rules, so the
  comparator here is effectively a near-constant Cox model. Its MAE advantage
  may come from the linear predictor rather than from a usable rule list. The
  16-rule SCRE output is a different α and is not this comparator.

## Consequence

Do not position C200 as more accurate than SCRE. Position it as a
high-precision, abstention-capable subgroup method whose cost is coverage. If
the paper needs an accuracy claim, it cannot be against SCRE.

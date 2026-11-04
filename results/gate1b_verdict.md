# Gate 1b — weight stabilisation: verdict

200 replications per cell · 4 mechanisms × 3 intensities × 7 strategies · n = 600 ·
t\* = 730 days.

## The failure mode, confirmed

At 50 % covariate censoring the untruncated estimator reaches a **maximum weight of
1000** and RMSE **0.2942** against a bias of only +0.0254. Roughly **1–2 patients per
dataset** fall below the censoring model's training support (0.2 % of the cohort);
those patients carry the entire instability.

This is a tail phenomenon, not a systematic error. That distinction is why the Gate 1
summary looked contradictory — small mean bias, enormous RMSE.

## Capping fixes it — at 50 % covariate censoring

| strategy | bias | RMSE | coverage | max wt | ESS |
|---|---|---|---|---|---|
| none | +0.0254 | 0.2942 | 97.5 % | 1000 | 136 |
| cap50 | +0.0086 | 0.1317 | 97.5 % | 50 | 139 |
| cap20 | +0.0028 | 0.1172 | 97.5 % | 20 | 150 |
| cap10 | −0.0039 | 0.1043 | 97.0 % | 10 | 165 |
| q99 | −0.0096 | 0.0983 | 95.5 % | 17 | 175 |
| **q95** | −0.0203 | **0.0884** | 95.0 % | 6.6 | 191 |
| oracle | +0.0035 | 0.1341 | 97.0 % | 189 | 157 |

**RMSE falls by ~3×.** Note the oracle is *not* the best here: the extreme weights are
partly legitimate (genuinely low censoring probability), so capping trades a little
bias for a large variance reduction.

## Capping does **not** fix informative censoring

| mechanism @ 50 % | none | cap10 | q99 | oracle |
|---|---|---|---|---|
| 3 event-dependent | −0.0866 | −0.0847 | −0.0847 | n/a |
| 4 frailty | −0.0499 | −0.0499 | −0.0497 | −0.0449 |

Unchanged across every strategy, and coverage stays at 80.5–90.5 %. This is the clean
separation the gate was designed to produce:

- **Covariate-dependent censoring** — an estimation-variance problem; fixed by a
  better censoring model plus truncation.
- **Unmeasured informative censoring** — a bias problem; not addressable by
  conditioning on observed `X`. Requires the sensitivity analysis (Gate 3).

## Pre-registered selection

Rule fixed before the run: among usable strategies, minimise mean RMSE subject to
|bias| < 0.01 and coverage in [93 %, 97 %], pooling mechanisms 1–2 at 25 % and 50 %.

| strategy | max abs bias | min coverage | mean RMSE | eligible |
|---|---|---|---|---|
| cap10 | 0.0056 | 93.0 % | 0.0793 | yes |
| cap20 | 0.0057 | 93.0 % | 0.0828 | yes |
| cap50 | 0.0086 | 93.0 % | 0.0865 | yes |
| none | 0.0254 | 93.0 % | 0.1272 | no |
| q95 | 0.0203 | 93.5 % | 0.0733 | no (bias) |
| **q99** | 0.0096 | 93.5 % | **0.0766** | **yes — chosen** |

**Chosen: `q99`** — cap the weight at the 99th percentile of each fold's *training*
weights (no held-out information used to set the cap).

### Caveat on the margin

`q99` wins by 3 % RMSE over `cap10` but has **1.7× the bias**, and its max |bias|
(0.0096) sits within 4 % of the threshold. `cap10` is the safer choice on bias for
almost no RMSE cost. If the tolerance were tightened to |bias| < 0.005, **no strategy
qualifies** — a cap of 10 gives 0.0056.

Recommendation: adopt **q99 with a hard ceiling of 20** (i.e. `min(q99, 20)`), which
keeps `q99`'s variance behaviour while bounding the worst case. This is a minor
amendment to the frozen specification, which previously contained no truncation rule
at all.

## Non-regularity at high censoring

Gate 1 and Gate 1b use different seed sets for the same cell (covariate, 50 %):

| run | bias | RMSE |
|---|---|---|
| Gate 1 `aipw_cond` | **−0.0302** | 0.2591 |
| Gate 1b `none` | **+0.0254** | 0.2942 |

The bias estimate **flips sign between seed sets**. At 50 % covariate censoring the
estimator is heavy-tailed enough that 200 replications do not pin down its bias. Any
single-run bias number at that intensity should be reported with that caveat, and
capping removes the non-regularity (post-cap RE at 50 % is stable at ~0.10).

## Consequences for the frozen specification

1. **Add a weight-cap rule.** Previously absent. Frozen choice: `min(q99_fold, 20)`.
2. **Report the weight tail in every analysis** (max, p99, ESS, % capped) — at 50 %
   censoring these values, not the point estimate, determine whether the result is
   usable.
3. **At 730 days on ACTG175 (~10 % censoring)** capping will rarely bind (0.00–0.02 %
   of weights at 10 % censoring). It matters as insurance, not as a routine operation.
4. **Do not cite the 50 %-censoring bias figures as stable** until the cap is applied.

## Still outstanding

- **Non-PH DGP.** Every mechanism here is Cox, so the Cox comparator is correctly
  specified. The gate still cannot show a CISCaRL advantage, only its cost.
- Gate 2 (rule-level inference) and Gate 3 (delta sensitivity) are untouched.
- The ACTG175 discontinuation semantics remain unresolved.

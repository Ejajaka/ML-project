# Gate 2b — EIF-corrected subgroup intervals: verdict

100 replications per cell · n = 2000 · 50/50 discovery/evaluation · same seeds
as Gate 2 · analytic intervals only (no bootstrap noise).

## The EIF correction is negligible — and that is the finding

| folds | cell | rules | bias | **jackknife share of variance** | cover IF | cover EIF | cover EIF-t |
|---|---|---|---|---|---|---|---|
| 5 | independent 10 % | 168 | −0.0118 | **1 %** | 92.3 % | 92.3 % | 92.3 % |
| 5 | covariate 25 % | 165 | −0.0033 | **2 %** | 93.3 % | 93.9 % | 94.5 % |
| 3 | independent 10 % | 168 | −0.0126 | **1 %** | 92.3 % | 92.3 % | 92.3 % |
| 3 | covariate 25 % | 165 | −0.0046 | **2 %** | 92.7 % | 92.7 % | 93.3 % |

The fold-jackknife component of the influence function — the term that captures
nuisance-estimation variability — accounts for **1–2 % of total variance**. Adding it
moves coverage by 0.0–0.6 pp. The EIF fix was the right thing to try and it is not
the problem.

That is the expected behaviour: the AIPW is Neyman-orthogonal, and with nuisances fit
on n = 1000 by cross-fitting, their contribution to first order vanishes. There was
very little for the EIF to correct.

## The Gate 2 failure was under-powered, not real

| | rules | coverage | MC SE |
|---|---|---|---|
| Gate 2 (40 reps) | 103 | 88.3 % | 3.2 pp |
| **Gate 2b (100 reps)** | **333** | **93.1 %** | **1.4 pp** |

Gate 2's 88.3 % came from 103 rules. Tripling the rule count moves the estimate to
93.1 %, which is 1.4 SE below 95 % (p ≈ 0.17) rather than 2.1 SE (p ≈ 0.04). **The
earlier "coverage fails" verdict was substantially Monte Carlo noise and should be
withdrawn.**

Fold count does not matter either: 3-fold and 5-fold give 92.5 % and 93.1 % pooled
respectively. My hypothesis that Gate 2's 3-fold nuisances explained the gap was
wrong.

## Where coverage actually stands

Pooled over 333 rules:

| interval | coverage | vs 95 % |
|---|---|---|
| IF | 93.1 % ± 1.4 | 1.4 SE below |
| EIF | 93.1 % ± 1.4 | 1.4 SE below |
| **EIF-t** | **93.4 % ± 1.4** | **1.2 SE below** |

Against the pre-registered acceptance band from Gate 1 (**[93 %, 97 %]**), these pass —
but at the lower edge, and the honest description is **"approximately 93 % coverage"**,
not "95 % nominal".

## What the focused experiment settles

1. **Keep the plain IF, at 5-fold cross-fitting.** The EIF adds cost and 0.3 pp.
2. **Use the t-quantile.** It is free and recovers 0.3 pp, the largest single gain
   available here.
3. **Report 93 %, not 95 %.** The intervals are approximately calibrated. Claiming
   nominal 95 % is not supported; claiming ~93 % is.
4. **Stop chasing this.** The remaining ~1.5 pp gap is finite-sample behaviour of a
   heavy-tailed IPCW estimator at n_r ≈ 130. Closing it would need a different
   estimator (TMLE) or much larger subgroups, not a variance correction.

## Confirmed at higher precision: the empty-list problem

| cell | no-rule replications |
|---|---|
| independent 10 % | **21 / 100** |
| covariate 25 % | **16 / 100** |

16–21 % of datasets produce no rule at all. That is now measured on 100 replications
rather than 40, and it is the most serious remaining limitation of the method: the
output is frequently empty.

## Updated claim

> On ACTG175-scale data CISCaRL recovers a small, reproducible rule list in roughly
> 80 % of replications, with 85–92 % sign accuracy on committed verdicts. Its 95 %
> subgroup intervals achieve approximately 93 % coverage — close to nominal, at the
> lower edge of the pre-registered acceptable band.

That is now supported. It is not a claim of exact calibration, and it is not a claim
of superiority over SCRE.

## Consequence for the pipeline

The uncertainty layer is adequate to proceed. Gate 3 (delta sensitivity) can now be
built on intervals that are approximately calibrated, with 93 % coverage stated
rather than 95 %.

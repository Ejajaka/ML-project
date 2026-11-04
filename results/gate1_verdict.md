# Gate 1 — verdict

200 replications per cell · 4 mechanisms × 3 censoring intensities · n = 600 ·
t\* = 730 days · covariates resampled from ACTG175 · true τ ≈ 0.083.

## Pre-registered criterion

PASS if, under mechanisms 1–2 with `aipw_cond`, |bias| < 0.01 and coverage ∈ [93 %, 97 %].

| mechanism | censoring | bias | coverage | verdict |
|---|---|---|---|---|
| 1 independent | 10 % | −0.0014 | 93.5 % ± 1.7 | **PASS** |
| 1 independent | 25 % | +0.0016 | 94.5 % ± 1.6 | **PASS** |
| 1 independent | 50 % | +0.0057 | 93.0 % ± 1.8 | **PASS** (borderline) |
| 2 covariate | 10 % | +0.0001 | 96.0 % ± 1.4 | **PASS** |
| 2 covariate | 25 % | +0.0025 | 92.0 % ± 1.9 | **MARGINAL** — bias passes, coverage misses (92 % < 93 %) |
| 2 covariate | 50 % | −0.0302 | 94.5 % ± 1.6 | **FAIL** (bias) |

**Overall: partial pass.** Meets the criterion at 10 % censoring, misses coverage at
25 %, fails on bias at 50 %. Not an unconditional pass.

## Informative censoring (pre-declared hypothesis: bias grows with intensity)

| mechanism | 10 % | 25 % | 50 % | monotone? |
|---|---|---|---|---|
| 3 event-dependent | −0.0152 | −0.0354 | −0.0806 | **yes** (−18 % → −43 % → −97 % of τ) |
| 4 shared frailty | −0.0219 | −0.0264 | −0.0405 | **yes** (−26 % → −32 % → −49 %) |

Coverage falls to 88.5 % (event, 50 %). Both hypotheses confirmed.

## Three findings that change the plan

### 1. Weight truncation is not optional — and is not in the frozen spec

At 25 % covariate censoring the bias is +0.0025 but **RMSE is 0.148**; the median
absolute error is only 0.045. Excluding the worst 1 % of replications drops RMSE to
0.069. The cause is a handful of replications with a single weight of 500–700
(fitted out-of-sample `Ĝ` extrapolating below the true value). At 50 % the maximum
observed weight is 698.

The frozen specification has no truncation rule. A **Gate 1b** must choose the cap
on the bias–variance trade-off *in simulation*, then freeze it. Candidate: cap at the
99th percentile of `Ĝ`, or at a fixed weight, chosen by minimising RMSE subject to
|bias| < 0.01.

### 2. The AIPW estimator is *more* fragile than a Cox T-learner under informative censoring

At 50 % censoring:

| mechanism | AIPW bias | AIPW RMSE | Cox bias | Cox RMSE |
|---|---|---|---|---|
| 1 independent | +0.0057 | 0.1007 | +0.0019 | 0.0421 |
| 2 covariate | −0.0302 | 0.2591 | −0.0012 | 0.0457 |
| 3 event-dependent | −0.0806 | 0.1384 | −0.0022 | 0.0459 |
| 4 frailty | −0.0405 | 0.0904 | −0.0254 | 0.0416 |

This is the opposite of the hoped-for result and must be reported. **Important
caveat:** these DGPs are Cox, so the comparator is *correctly specified* — it is the
best case for Cox and the gate contains no scenario where Cox is wrong. The gate as
designed can only show the AIPW's cost, never its benefit.

### 3. The 730-day horizon is now evidence-supported, not forced

ACTG175 has ≈ 10 % censoring at 2 years and ≈ 50 % at 3 years. Gate 1 says the
estimator passes at the former and fails at the latter. That is evidence for the
horizon, conditional on the DGPs and censoring mechanisms tested — not a
mathematical constraint.

## Required follow-up (Gate 1b), before Gate 2

1. **Truncation sensitivity** — cap sweep, RMSE/bias trade-off, freeze a rule.
2. **A non-PH DGP** (crossing hazards) where the Cox T-learner is misspecified, so
   the comparison is not rigged in the comparator's favour.
3. **RSF outcome model** instead of Cox, to match the frozen ACTG175 specification.

## What Gate 1 does **not** establish

- Nothing about rule-level inference after selection (Gate 2).
- Nothing about the delta sensitivity analysis (Gate 3).
- Nothing about SCRE (Gate 4) — the comparator here is Cox, not SCRE.
- Nothing about ACTG175 itself; the discontinuation semantics remain unresolved.

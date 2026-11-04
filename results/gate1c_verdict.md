# Gate 1c — crossing hazards: verdict

200 replications per cell · 2 censoring mechanisms × 3 intensities · n = 600 ·
t\* = 730 days · hazards cross at 180 d (treated worse early, better late).

## The pooled Cox is structurally broken — as predicted

A pooled Cox with one baseline forces a constant log-hazard-ratio. Under crossing
hazards its bias is 16–45 % of the true effect, **and the sign flips with censoring
intensity**:

| censoring | pooled Cox bias | as % of τ |
|---|---|---|
| independent 10 % | +0.0271 | +42 % |
| independent 25 % | +0.0105 | +16 % |
| independent 50 % | −0.0197 | −31 % |
| covariate 10 % | +0.0289 | +45 % |
| covariate 25 % | +0.0140 | +22 % |
| covariate 50 % | −0.0185 | −29 % |

You cannot even sign the bias of a misspecified model without knowing the censoring
pattern. That is a strong, concrete argument against SCRE's shared-basis Cox structure
on non-PH data.

## The AIPW is unbiased here

| censoring | AIPW bias | T-learner bias |
|---|---|---|
| independent 10 / 25 / 50 % | +0.0009 / +0.0009 / −0.0072 | −0.0002 / −0.0013 / −0.0023 |
| covariate 10 / 25 / 50 % | +0.0034 / +0.0041 / −0.0073 | +0.0020 / +0.0028 / +0.0045 |

`aipw_rsf` and `aipw_cox` are indistinguishable — the RSF switch buys nothing when the
within-arm Cox model is already correct.

## …but the AIPW has the **worst RMSE in every cell**

| censoring | pooled Cox | Cox T-learner | AIPW |
|---|---|---|---|
| independent 10 % | **0.0296** | 0.0307 | 0.0353 |
| independent 25 % | **0.0192** | 0.0322 | 0.0480 |
| independent 50 % | **0.0305** | 0.0367 | 0.0928 |
| covariate 10 % | **0.0315** | 0.0351 | 0.0388 |
| covariate 25 % | **0.0219** | 0.0360 | 0.0576 |
| covariate 50 % | **0.0341** | 0.0453 | 0.1007 |

The biased pooled Cox still wins on RMSE, because its variance is ~5–10× smaller.
At 50 % censoring it beats the AIPW by a factor of three.

## What this means

**1. CISCaRL cannot beat a pooled Cox on RMSE at n = 600 — even when the Cox is
structurally wrong.** The bias is real and large, but it is swamped by the AIPW's
variance. The IPCW construction trades efficiency for robustness, and at this sample
size the efficiency loss is larger than the robustness gain.

**2. The estimator is not the differentiator; the rule layer is.** A plain Cox
T-learner matches the AIPW's bias everywhere for a fraction of the cost. Any advantage
CISCaRL has must come from what the T-learner cannot produce: a small stable rule
list, per-subgroup intervals, and an explicit "insufficient evidence" verdict.

**3. The honest comparative claim is now narrow and specific:**

> Under non-proportional hazards, a shared-basis Cox rule ensemble inherits a bias of
> 16–45 % whose sign depends on the censoring pattern. CISCaRL's estimator is unbiased
> in the same setting, at the cost of 1.2–3× the RMSE.

That is defensible and interesting. "CISCaRL has better HTE accuracy than SCRE" is not.

**4. Consequence for the project's objective.** The stated goal of outperforming SCRE
on HTE estimation is not achievable with this estimator at this sample size. The
winnable axes remain interpretability, calibrated uncertainty, abstention, and
stability — none of which SCRE offers.

## Gate 1-verify (companion result)

The v1.1 weight cap reproduces exactly through the outer cross-fitting pipeline:

| mechanism | censoring | Δbias | 2×MC | Δcoverage | verdict |
|---|---|---|---|---|---|
| covariate | 10 % | +0.0003 | 0.0114 | 0.0 pp | **PASS** |
| covariate | 25 % | +0.0007 | 0.0165 | 0.0 pp | **PASS** |
| covariate | 50 % | +0.0029 | 0.0310 | 0.0 pp | **PASS** |
| independent | 10 / 25 / 50 % | ≤0.0000 | — | 0.0 pp | **PASS** |

6/6 cells pass. Cap effect at covariate 50 %: RMSE 0.2949 → 0.1085, bias −0.0297 →
−0.0231.

Residual caveat: at covariate 50 % the capped bias is still −0.023 (≈28 % of τ) and
coverage 92 %. Capping fixed the variance, not the residual bias at heavy censoring.
This reinforces 730 days as the primary horizon.

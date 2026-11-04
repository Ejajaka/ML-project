# Gate 2 — rule-level inference, abstention, stability: verdict

40 replications per cell · n = 2000 · 50/50 discovery/evaluation split ·
τ ≈ 0.15–0.25 (ACTG175 scale) · t\* = 730 days.

## Headline: the interval machinery does **not** reach nominal coverage

Coverage is conditional on the rule being selected (the inferential target fixed in
the frozen spec).

| cell | rules/rep | no-rule reps | cover IF | cover mult | cover refit | width | abstain | TREAT | sign ok |
|---|---|---|---|---|---|---|---|---|---|
| independent 10 % | 1.2 | 12/40 | 89.6 % | 89.6 % | 87.5 % | 0.346 | 38 % | 62 % | 92 % |
| covariate 25 % | 1.4 | 6/40 | 87.3 % | 83.6 % | 87.3 % | 0.560 | 58 % | 42 % | 85 % |

Pooled: **91 of 103 rules covered = 88.3 % ± 3.2 pp** against a 95 % target — 2.1 SE
below nominal (p ≈ 0.04). The deficit is small but it is not Monte Carlo noise.

All three interval methods agree within a few points once the multiplier-bootstrap
indexing bug was fixed (it had been resampling the first `n_r` rows instead of the
rule's members). So this is not an implementation artefact of one method: **the
subgroup SE is underestimated by roughly 15–20 %** regardless of how it is
bootstrapped.

Bias is small (−0.016 in both cells, MC SE 0.014–0.026) and bias-correcting the
interval changes coverage by 0.0 pp. The problem is variance, not location.

## Two findings that matter as much as coverage

**1. The method often produces no rule at all.** 12/40 (30 %) and 6/40 (15 %) of
replications returned an empty rule list. For a method whose entire output *is* the
rule list, a 15–30 % null rate is a serious robustness limitation and must be
reported.

**2. Intervals are roughly twice the effect size.** Mean width 0.346–0.560 against
τ ≈ 0.21. Consequently 38–58 % of rules are labelled INSUFFICIENT EVIDENCE — the
method abstains more often than it commits. That is honest behaviour, but it means
the usable output on ACTG175-scale data will be thin.

## What does hold up

- **Stability:** selected rules have rediscovery frequency 0.69–0.71 under
  half-sample resampling of the discovery data. The rule list is not arbitrary.
- **Sign accuracy:** 85–92 % of committed verdicts have the correct sign.
- **Small lists:** 1.2–1.4 rules per replication, support ≈ 127 — interpretable by
  construction.
- **AVOID = 0 %**, which is correct here: every subgroup in this DGP has a positive
  effect, so abstaining or treating are the only defensible outputs.

## Caveats on this gate

- The stability measure refits **only the tree** on subsamples, holding the
  pseudo-outcome fixed. It is *tree* stability, not full-pipeline stability.
- 40 replications give ±4.5–5 pp on each cell's coverage and ±3.2 pp pooled. Enough
  to detect a 7 pp deficit, not enough to characterise the tail.
- Coverage is conditional on selection. Unconditional (procedure-level) coverage
  would be lower, since 15–30 % of runs produce no rule to cover.

## Consequence for the reframed claim

The claim "CISCaRL provides subgroup-level uncertainty" is **partially supported and
not yet established**. What the evidence supports today:

> On ACTG175-scale data the method recovers a small, reproducible rule list in
> 70–85 % of replications, with 85–92 % sign accuracy on committed verdicts. Its
> 95 % intervals cover at ≈ 88 %, so nominal coverage is not yet achieved and the
> uncertainty layer should be described as approximate.

Before the uncertainty claim can be made, one of these is required:

1. **An EIF-corrected interval** — add the nuisance-estimation terms to the
   influence function instead of treating them as fixed. This is the most likely fix,
   since all three bootstraps under-cover by a similar amount.
2. **Bias-corrected accelerated (BCa) bootstrap** with B ≥ 200 rather than B = 30–40.
3. **TMLE** instead of AIPW, which solves the estimating equation exactly and
   generally has better finite-sample coverage.

Option 1 is the cheapest and most likely to work; option 3 is the most defensible.

## Status

| Component | Result |
|---|---|
| Rule discovery produces a small list | yes, 1.2–1.4 rules, but empty 15–30 % of the time |
| Rule stability | 0.69–0.71 rediscovery |
| Subgroup point estimate | unbiased (−0.016) |
| 95 % interval coverage | **88.3 % — fails nominal** |
| Interval width | ~2× effect size |
| Abstention occurs when precision is low | yes, 38–58 % |
| Verdict sign accuracy | 85–92 % |
| Holm procedure | implemented; with 1–2 rules it rarely binds |

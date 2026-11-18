# ACTG175 full cohort: arm 0 vs arms 1-3

Contrast: ZDV monotherapy vs any alternative regimen (arms 1/2/3).
Propensity e = 0.751 (known by four-arm randomisation).
n = 2,139; events = 521; splits = 20.

- commit = 65.0% (13/20)
- abstain = 35.0%
- fallback = 0.0%
- mean validation LCB = +0.018
- mean validation effect = +0.163
- mean support = 442
- unique rules = 20/20
- mean Jaccard vs canonical = 0.504
- committed mean assessment = +0.192

## Canonical split (seed 2024)

- rule: `cd80 > 996.0`
- support: 437 (fallback: False)
- validation: +0.078 (SE 0.082, LCB -0.056)
- assessment: +0.147
- decision: INSUFFICIENT

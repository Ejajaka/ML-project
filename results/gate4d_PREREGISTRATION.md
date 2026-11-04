# Gate 4d — piecewise-constant DGP: pre-registration

**Written before the experiment is run.** Definitions here are frozen; the
post-run document may report results but not redefine these quantities.

## DGP

$$\tau(X) = 0.20 \cdot \mathbb 1\{\text{CD4} < 200 \wedge \text{age} > 40\}$$

implemented as a per-patient log-hazard offset, so the effect is **zero by
construction outside the region**. The realised mean τ inside and outside the region is
reported alongside every result; if it deviates from 0.20/0.00 the deviation is stated
rather than corrected.

Region prevalence is reported; it is expected to be ~10 % of the cohort.

| setting | value |
|---|---|
| n | 589 and 2000 |
| censoring | covariate-dependent, calibrated to 10 % at t\* |
| horizon | t\* = 730 days |
| split | 70/30 train/test, stratified, paired across methods |
| SCRE tuning | validation concordance on 25 % of training data (same protocol as Gate 4c) |
| CISCaRL | v1.4, stability ≥ 0.50, support ≥ 60, Jaccard 0.8, no rule cap |


## Amendment 1 — region definition (made before any results were examined)

A dry run used the clinical cut-points `CD4 < 200` and `age > 40` and reported a region
prevalence of **1 %**, against an assumed ~10 %. The cause is eligibility, not the
method: ACTG175 enrolled patients with CD4 between 200 and 500, so only 8.4 % have
CD4 < 200, and 1.8 % satisfy both conditions. A 1 % region gives ~3 patients per
test split, which cannot support a sensitivity estimate.

The region is therefore redefined **at cohort medians**:

$$R^* = \{\,\text{CD4} < 340 \ \wedge\ \text{age} > 34\,\}$$

which has prevalence **24.9 %** in the cohort — a genuine subgroup, a workable
denominator, and still a piecewise-constant effect surface. The effect stays
τ = 0.20 inside and 0.00 outside.

No outcome was inspected before making this change; only the prevalence. Everything
else in this document stands, including the three-part decision rule.

## Metrics — exact definitions

Let R\* = {i in the **test** set : CD4 < 200 ∧ age > 40}, and let
P = {i : the method issues a TREAT decision}.

**CISCaRL assignment:** a patient is assigned to the most specific matching rule; the
verdict of that rule determines P. Patients matching no rule receive the default
(overall) verdict. **This is the only aggregation procedure; there is no "any rule says
TREAT" alternative.**

**SCRE assignment:** SCRE has no abstention, so P = {i : τ̂_i > 0}.

| quantity | definition | denominator |
|---|---|---|
| **region MAE** | mean over i ∈ R\* of \|τ̂_i − τ_i\| | \|R\*\| |
| **Δ region MAE** | MAE_R\*(SCRE) − MAE_R\*(CISCaRL) — **positive favours CISCaRL** | — |
| overall MAE | mean over all test patients of \|τ̂_i − τ_i\| | n_test |
| **sensitivity** | \|P ∩ R\*\| / \|R\*\| | \|R\*\| |
| **specificity** | \|P̄ ∩ R̄\*\| / \|R̄\*\| | \|R̄\*\| |
| **false-positive TREAT** | \|P ∩ R̄\*\| / \|R̄\*\| | \|R̄\*\| |
| **commitment rate** | share of test patients receiving TREAT or AVOID | n_test |
| **directional precision** | among committed patients with \|τ_i\| > 0.01, share where sign(τ̂_i) = sign(τ_i) | that subset |

Directional precision is computed **separately for TREAT and AVOID commitments**
(correct = τ_i > 0 for TREAT, τ_i < 0 for AVOID). Patients with \|τ_i\| ≤ 0.01 are
excluded from that denominator and counted in a separate "near-zero" column.
Directional precision is *not* sensitivity and *not* membership in R\*.

## Pre-registered decision rule

A **rule-shaped advantage is supported only if all three hold**:

1. **Δ region MAE > 0** with a replication-level 95 % CI excluding zero;
2. **sensitivity ≥ 0.60**;
3. **false-positive TREAT ≤ 0.10**.

If any one fails, the advantage is **not supported**, and the specific failure is
reported (either "no estimation gain", "found the region too rarely", or "bought
sensitivity with broad TREAT assignment").

## Uncertainty reporting

Every metric is reported as a mean over replications with a **replication-level**
bootstrap 95 % CI. Sensitivity and false-positive rates are computed on the ~10 %
region, so their denominators are small; the CI width is reported and a threshold
result close to 0.60 is described as borderline rather than as a pass.

Commitment rate and interval coverage are reported as **separate** columns and are never
combined into a single "coverage" number.

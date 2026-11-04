# ACTG175 estimand — resolved from the source dictionary

Source: `speff2trial` CRAN documentation for `ACTG175` (rdrr.io/cran/speff2trial/man/ACTG175.html).

## What the dictionary says

| variable | documentation | consequence |
|---|---|---|
| `cens` | "indicator of observing the event in `days`" | **`cens = 1` is an observed event.** The coding used in the pipeline is correct. |
| `days` | "number of days until the first occurrence of: (i) a decline in CD4 T cell count of at least 50, (ii) an event indicating progression to AIDS, or (iii) death" | **The endpoint is a composite, not death.** |
| `offtrt` | "indicator of off-treatment before 96±5 weeks (0=no, 1=yes)" | A **post-randomization covariate**, not a censoring flag. |
| Details | "includes the following post-randomization covariates: CD4 and CD8 at 20±5 weeks and the indicator of whether or not the patient was taken off-treatment" | Confirms `offtrt`, `cd420`, `cd496`, `cd820`, `r` are post-randomization. |
| `arms` | 0 = zidovudine, 1 = zidovudine + didanosine, 2 = zidovudine + zalcitabine, 3 = didanosine | The arms 0 vs 1 cohort is correct and is a clean two-arm comparison. |

## Correction 1 — the outcome is not survival, it is progression-free time

Every table, figure and sentence in the write-up that says "3-year survival" or "mortality
benefit" is **wrong**. The estimand is:

> τ(730) = P(no CD4 decline ≥ 50 **and** no AIDS event **and** not dead by day 730),
> under treatment a vs b.

The clinically correct wording is **progression-free probability**, not survival. This is
not a cosmetic change: the composite is driven substantially by immunological decline,
which is measured at scheduled visits, so the outcome depends on the follow-up schedule
as well as on the patient.

## Correction 2 — the naive "ITT" reading is not available

The dictionary does **not** say follow-up stops at discontinuation, but the data say the
association is nearly deterministic:

| | n | censored before 730 d |
|---|---|---|
| `offtrt = 0` | 664 | **3** |
| `offtrt = 1` | 390 | **100** |

97 % of all censoring before 730 days occurs in patients who discontinued treatment. Two
readings are consistent with this:

1. Follow-up (specifically the CD4 component of the composite) ends when the patient
   leaves treatment, so censoring is **at discontinuation** — an on-treatment estimand,
   with informative censoring.
2. Discontinuation and loss to follow-up share a cause (clinical deterioration), so
   censoring is **informative but not mechanically at discontinuation**.

The data cannot distinguish these. What they exclude is the convenient third reading,
"censoring is unrelated to discontinuation".

**Consequence:** the estimand is an *observed-follow-up* contrast, not an
intention-to-treat effect. The paper must say so, and the Gate 3 sensitivity analysis is
not optional decoration — it is the only defensible way to report the result.

## Correction 3 — the composite endpoint weakens the simulation analogy

All gates simulated a single exponential/Cox event time with censoring. The real
endpoint is a **composite dominated by a lab-measured threshold crossing** (CD4 decline
≥ 50), which:

- is not a survival time in the usual sense;
- depends on the visit schedule (a patient measured more often is more likely to cross
  the threshold);
- has an event rate (284/1054 = 26.9 %) that a simple survival model matches only
  approximately.

The simulation results remain valid as *methodological* validation of the estimator and
the decision layer, but they are **not** a faithful model of the ACTG175 outcome
process. That limitation belongs in the paper.

## What is now settled

| question | status |
|---|---|
| Is `cens = 1` an event? | **Yes**, confirmed by the dictionary |
| Is the endpoint death? | **No** — composite: CD4 decline ≥ 50, AIDS, death |
| Is `offtrt` a censoring flag? | **No** — a post-randomization covariate |
| Are `cd420/cd496/cd820/r/offtrt` post-randomization? | **Yes**, correctly excluded from covariates |
| Is the estimand ITT? | **No** — observed-follow-up contrast |
| Is censoring plausibly informative? | **Yes** — 97 % of early censoring is in discontinuers |

## What remains open

1. Whether follow-up *mechanically* ends at discontinuation. The package documentation is
   silent; this would need the trial's protocol or an analysis of the raw CD4 visit data
   (`cd420`, `cd496` availability).
2. Whether a partial check is possible from `r` (the indicator for whether the 96-week CD4
   was observed): if `r = 0` patients are precisely the `offtrt = 1` patients, that would
   support reading 1.

## Follow-up check: censoring times support "censored at discontinuation"

Two pieces of evidence bear on reading 1.

**`r` is not a clean proxy for discontinuation.** Of `offtrt = 1` patients, 66 % (258/390)
have `r = 0`; of `offtrt = 0`, 21 % (142/664) do. So `r = 0` is enriched among
discontinuers but not exclusive to them, and it cannot settle the question.

**The censoring-time distribution is much more informative.**

| group | censored before 730 d | median time | range |
|---|---|---|---|
| `offtrt = 0` | 3 / 664 | 728 d | 709–729 |
| `offtrt = 1` | 100 / 390 | **415 d** | 125–730 |

Patients who never discontinued are censored only at the 730-day boundary (709–729 days,
i.e. administrative). Patients who discontinued are censored at a median of 415 days —
well inside the pre-672-day discontinuity window. That pattern is exactly what
"follow-up ends when treatment ends" predicts, and it is not what pure administrative
censoring predicts.

**Conclusion:** the most defensible reading is that the composite endpoint requires
on-treatment CD4 follow-up, so discontinuation censors the outcome. The estimand is
therefore **on-treatment**, censoring is **at discontinuation** for 97 % of early
censoring, and censoring is very likely **informative** because discontinuation reflects
clinical status.

## Required edits before any further reporting

- Replace "survival" with "progression-free probability" throughout.
- State the estimand as an observed-follow-up contrast, explicitly not ITT and not
  per-protocol.
- Report the censoring/discontinuation cross-tabulation as a limitation.
- Note that the composite endpoint makes the simulation analogy approximate.

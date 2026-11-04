# CISCaRL — consolidated claims after v1.3

**Status:** specification frozen at v1.3; estimation and uncertainty layers validated
under simulation; decision layer stopped at ACTG175 sample size; application-level
interpretation conditional on censoring assumptions.

This document supersedes `PAPER.md` and `PROJECT_STATUS.md` for the purposes of any
external writing. Those documents describe an earlier endpoint and an earlier set of
claims.

---

## 1. What CISCaRL is

CISCaRL produces a short list of interpretable threshold rules defining subgroups, and
for each subgroup an estimate of the treatment contrast in **progression-free
probability** at a fixed horizon, with an interval and an explicit
TREAT / AVOID / INSUFFICIENT EVIDENCE verdict.

It is **not** a clinical decision system, and it is **not** validated as one at the
sample size available.

---

## 2. The estimand (corrected from the source dictionary)

Source: `speff2trial` CRAN documentation for `ACTG175`.

| variable | meaning | consequence |
|---|---|---|
| `days` | time to first **CD4 decline ≥ 50**, **AIDS progression**, or **death** | the endpoint is a **composite**, not mortality |
| `cens` | indicator that the event was observed | `cens = 1` is an event — the pipeline coding is correct |
| `offtrt` | off-treatment before 96 ± 5 weeks | a **post-randomization covariate**, not a censoring flag |
| `r` | whether the 96-week CD4 was observed | does **not** cleanly identify discontinuation (66 % vs 21 %) |

**Corrected estimand:**

> τ(730) = contrast in the probability of remaining free of the composite endpoint
> (CD4 decline ≥ 50, AIDS progression, or death) through day 730, under the **observed
> follow-up process**.

This is an **observed-follow-up contrast**. It is not an intention-to-treat
treatment-policy estimand, and it is not a per-protocol estimand. Neither is identified
from these data without additional assumptions, because follow-up appears to terminate at
or shortly after discontinuation.

**Evidence for that (empirical, not from the dictionary):**

| group | censored before 730 d | median time | range |
|---|---|---|---|
| `offtrt = 0` | 3 / 664 | 728 d | 709–729 |
| `offtrt = 1` | 100 / 390 | **415 d** | 125–730 |

Non-discontinuers are censored only at the administrative boundary; discontinuers are
censored a median of 415 days out, inside the pre-672-day discontinuity window. 97 % of
early censoring is therefore discontinuation-related and plausibly **informative**.

**Terminology.** "Survival" is wrong throughout the earlier artifacts. The correct term
is **progression-free probability**, and this must be substituted everywhere.

---

## 3. What the simulation program validates — and what it does not

| Gate | Question | Verdict | Number |
|---|---|---|---|
| 1 | Estimator bias and coverage under four censoring mechanisms | **partial pass** | passes at 10 % censoring, fails at 50 % |
| 1b | Extreme-weight stabilisation | **pass** | cap = min(Q₀.₉₉, 20); RMSE 0.294 → 0.098 |
| 1-verify | Does the cap survive the outer pipeline? | **pass** | 6/6 cells, Δbias ≤ 0.003 |
| 1c | Is the Cox comparator fairly specified? | **pass** | pooled Cox biased 16–45 %, sign flips with censoring |
| 2 | Rule-level inference after selection | **withdrawn, under-powered** | 88.3 % on 103 rules |
| 2b | EIF-corrected intervals | **pass** | 93.1–93.4 % on 333 rules; EIF adds 0.3 pp |
| 2c | Stability threshold | **frozen at 0.50** | null rate 25 % → 2–10 % |
| 2d | Application-matched sample size | **informative** | 87 % abstention, CE width 5.2× effect |
| 2e | Decision layer | **STOP at n = 589** | power-limited, not threshold-fixable |
| 3 | Delta sensitivity under known MNAR | **revise** | outcome tilt inert (slope 0.003); weight tilt works |

**Scope limitation.** All of these simulate a single time-to-event outcome with a Cox or
piecewise hazard. The real endpoint is a **composite dominated by a lab-measured CD4
threshold crossing**, whose ascertainment depends on the visit schedule. The gates
therefore validate the *method*, not the *ACTG175 outcome process*.

---

## 4. The three layers, separated

### Estimation — validated

- Bias ≈ −0.003 to −0.012 in the tested DGPs; unbiased at the application-matched size.
- Consistent under a correct censoring model; **not** doubly robust (verified: a wrong
  censoring model at 50 % censoring gives +0.015 bias and 87.5 % coverage).
- Robust to outcome-model misspecification.

### Uncertainty — approximately calibrated

- Marginal coverage **93.1–93.4 %** (333 rules) against a 95 % nominal target, i.e.
  within the pre-registered [93, 97] band at its lower edge.
- Near-null false-commitment rate **7.5 %** (clustered CI [5.9, 8.9]) against 5 %
  nominal — modestly anti-conservative.
- Do **not** describe these as "95 % confidence intervals". Report ~93 %.

### Decision — stopped at n = 589

With a mixed-direction DGP (28.4 % of rules truly beneficial):

| | |
|---|---|
| TREAT precision | **66.7 %**, clustered 95 % CI [54.8, 79.3] |
| AVOID precision | 95 % |
| abstention | 90 % |
| base-rate ceiling at 5 % FPR | 88.8 % |

The 67 % is what a **valid test with ~38 % power** produces at that base rate. It is not
a calibration failure and it is not fixable by the commitment threshold: the sweep traces
one curve, with precision reaching 100 % only when 4 verdicts remain.

---

## 5. Power requirement (the quantitative limit)

Reference: mean SE 0.175 at mean support 32.3, from the application-matched simulation.
Support required per rule for a two-sided 5 % test:

| effect size | 50 % power | 80 % power | 90 % power |
|---|---|---|---|
| 0.09 | 471 | 963 | 1,289 |
| 0.13 | 226 | 461 | 618 |
| 0.20 | 95 | 195 | 261 |
| 0.30 | 42 | 87 | 116 |

ACTG175: training n = 589, whole cohort n = 1054.

At the observed effect of 0.128 and 80 % power, a rule needs **≈ 460 patients** — most of
the training set for a single rule. This is why the decision layer abstains on 90 % of
rules, and it is a property of the data, not of the method.

**Caveat:** these figures assume the evaluated effect size, censoring distribution,
treatment allocation and estimator. They are operating conditions, not universal
constants.

---

## 6. Frozen specification — v1.3

| Component | Value |
|---|---|
| Horizon | 730 days primary; 365 / 1096 descriptive |
| Propensity | known design value, e = 0.5 |
| Outcome nuisance | per-arm survival model, 5-fold cross-fitted |
| Censoring nuisance | per-arm conditional Cox on X, 5-fold cross-fitted |
| Weight cap | min(Q₀.₉₉ of fold-training weights, 20) |
| Subgroup estimator | AIPW; consistent under correct Ĝ; **not** doubly robust |
| Interval | IF + t-quantile, conditional on the selected rule list |
| Discovery | RF on the DR pseudo-outcome; stability ≥ 0.50; min support 60; Jaccard 0.8; **no rule-count cap** (v1.4) |
| Reported decision metric | **coverage at calibrated precision**, not rule count |
| Sensitivity | power tilt of Ĝ, $\hat G^{\exp(\delta)}$; δ = 0 reproduces baseline exactly |
| Multiplicity | Holm within the rule list |

**Rejected and documented as such:** the outcome-model logit tilt (leverage 0.003 per
unit δ — inert), and the 0.60 stability threshold (25 % null rate).

## Rule-list length is not a validated quality criterion

List length was inherited from the original brief and is not supported by the gates.
A "3-rule list" in the ACTG175 demo covered **93.4 %** of the cohort, with 194 of 317
patients matching more than one rule. Three rules did not mean three subgroups; they
meant the cohort cut into overlapping slices.

The metric that should be reported is **coverage at calibrated precision**: what share
of patients receive a committed verdict, and how precise those verdicts are against the
base-rate ceiling. Gate 2e traced that frontier (precision 62 % -> 100 % as coverage fell
from 2.2 % -> 0.2 %). A rule count is not a substitute for it.

The `max_rules = 5` cap is therefore removed in v1.4 (it was binding at ~4.7 rules per
fold, so removal costs almost nothing). Jaccard pruning is retained, because
redundancy -- not list length -- is the real threat to interpretability.

---

## 7. The claim we can defend

> CISCaRL provides interpretable subgroup estimation for a progression-free composite
> outcome, with uncertainty quantification and explicit abstention. In simulation studies
> it exhibits low estimation bias and approximately nominal marginal interval coverage
> (≈ 93 %) under the evaluated censoring mechanisms. At the ACTG175 sample size,
> directional subgroup decisions are power-limited — a rule needs roughly 460 patients
> for 80 % power at the observed effect — and application-level interpretation depends on
> assumptions about the observed follow-up and censoring process.

## Claims we cannot make

- ❌ "CISCaRL outperforms SCRE on HTE accuracy." Not tested fairly, and Gate 1c's
  RMSE table points the other way.
- ❌ "High-precision decision support." 67 % TREAT precision at n = 589.
- ❌ "95 % confidence intervals." ≈ 93 %.
- ❌ "Robust to MNAR." The sensitivity parameter is directional but imprecisely
  identified (sd(δ\*) 0.48–0.85).
- ❌ "Survival benefit." The endpoint is a composite.
- ❌ "Intention-to-treat effect." Follow-up appears to end at discontinuation.

---

## 8. Remaining work, in order

1. **Rewrite the paper sections** using the corrected endpoint, estimand and scope
   (items 1–7 of the write-up list). Every table carries the old terminology.
2. **Gate 4 against SCRE**, at both n = 589 and n = 2000, with a mixed-direction DGP,
   separating estimation accuracy from decision reliability. Only this can support any
   comparative statement.
3. **Subgroup-specific δ\*** — the sensitivity analysis is currently validated only for
   the marginal contrast.
4. **Stronger-MNAR run** to sharpen δ\* identification.
5. **A CD4-visit-aware simulation** if the composite endpoint is to be modelled faithfully
   rather than assumed.

---

## 9. Honest summary

The estimation and uncertainty layers are sound. The decision layer is limited by
statistical power, and the application is complicated by a composite endpoint whose
follow-up plausibly ends at discontinuation.

The project's original ambition — a rich list of actionable subgroup rules at ACTG175
scale — is not supported. What survives is narrower and better evidenced: a validated
estimator, approximately calibrated intervals, a documented failure mode in the
competitor's structural assumptions, and a quantified statement of when this approach
can and cannot support a decision.

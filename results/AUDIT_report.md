# Implementation and evidence audit — v1.7

Date: 2026-11-02. Scope: core rule utilities, statistical estimators, the
experimental pipeline, and a leakage check on the Gate 19 decision rule.
All source is in `ciscarl2/`; the audited files are listed at the end.

## 1. Core utilities

| check | result |
|---|---|
| `simplify` bound direction (`<=` min, `>` max) | **FIXED** (was inverted; see `CORRECTION_rules_simplify.md`) |
| `extract_leaf_rules`: simplified condition equals the stored mask | PASS (0 mismatches over 10 random trees) |
| `extract_leaf_rules`: leaves partition the data | PASS (sum of masks = n) |
| `assign`: most-specific rule wins, default elsewhere, no double assignment | PASS |
| purity / capture / Jaccard definitions | PASS (hand-checked) |

The `simplify` defect affected 22.7 % of leaves with a median 92 % coverage
inflation. It is fixed, and Gates 16–18 were re-run on the corrected code.

## 2. Statistical estimators

| check | result |
|---|---|
| AIPW bias at n = 2000, 10 reps | **+0.0031** (sd 0.0130) |
| arm means: E[φ₁] − S₁ / E[φ₀] − S₀ | +0.0015 / −0.0016 |
| paired IF contrast D and SE | exact match to the direct formula |
| cap and nuisance fitting use training rows only | code-path review: yes |
| marginal IF-CI coverage (10 reps) | 10/10 (not a coverage audit; Gate 2b's 93.1–93.4 % on 333 rules remains the reference) |

## 3. Pipeline audit

| check | result |
|---|---|
| Seed ranges non-overlapping across Gates 9–19 | PASS |
| Train/test separation in target construction | PASS |
| Constant baseline fitted on TRAIN only | PASS |
| Oracle uses true membership with held-out effects | PASS |
| Paired deltas computed within replication | PASS |
| Multiplicity across cells | **not controlled** — the four-part rule is per cell; fresh-seed replication mitigates but does not eliminate this |
| Gate 16–18 re-runs after the bug fix | reuse the pre-fix seeds — a re-analysis, not a fresh confirmation; state this explicitly |

## 4. The one substantive leakage: Gate 19's commitment is in-sample

Gate 19 estimates the high leaf's AIPW effect and SE on the **test** patients,
uses that estimate to decide whether to commit, and evaluates the decision on
the **same** patients. The decision and the assessment share data.

A split audit on the joint cell (20 reps, fresh seeds 220100–220119) compares:

* **full**: decide on all test patients, assess on all test patients;
* **split**: decide on a random half of test, assess on the other half.

| decision | commit rate | coverage | per-treated | efficiency | captured | null rate | beneficial |
|---|---|---|---|---|---|---|---|
| full (Gate 19) | 60 % | 0.084 | +0.169 | 0.85 | 0.315 | 0 % | 100 % |
| honest split | **40 %** | **0.065** | +0.164 | 0.83 | **0.248** | 0 % | 100 % |

Interpretation:

* The **quality** of the committed set is unchanged: null rate 0 %, beneficial
  rate 100 %, per-treated and efficiency essentially identical. The commitment
  rule is not overfitting the *identity* of the subgroup.
* The **quantity** is optimistic: the commit rate (60 % → 40 %), coverage
  (0.084 → 0.065) and captured benefit (0.315 → 0.248) are inflated by 25–30 %
  because the full test set is used both to decide and to assess.
* Part of the commit-rate drop is power (half the decision sample), so a proper
  validation split of the same size as the test would land between the two
  rows. The correct design is a separate validation set, not half the test.

**Consequence:** Gate 19's decision claim is qualitatively supported (commitment
removes null treatment and raises efficiency) but its quantitative coverage and
captured-benefit figures must be restated using an honest separation. The v1.7
claims should quote the split numbers or be re-run.

## 5. Latent issues found by review (not bugs)

1. **`assign` estimates a rule's effect on the full mask, not the assigned
   subset.** For a single CART the leaves are disjoint, so v1.7 is unaffected.
   For the RF + stability pipeline, overlapping rules could bias the effect.
   Fix before quoting any v1.4 numbers.
2. **High-leaf selection is winner's-cursed.** The leaf is chosen by the maximum
   training mean ψ, which favours small, noisy leaves. Observed high-leaf
   coverage is 6–11 % and capture only 0.23–0.25. Candidate fixes: select on a
   lower confidence bound, require a minimum support, or use a stability-based
   selection.
3. **Multiple testing.** Gates 16–19 test many cells without a family-wise
   correction. The fresh-seed replications are the mitigation; the paper must
   report the number of confirmatory tests.
4. **Stability remains the weakest dimension.** ψ split stability 0.16–0.20;
   the audit did not change this.
5. **Dual fallback.** The prediction fallback is the marginal ATE; the decision
   fallback is no treatment. Both are in the pre-registration but must be
   explicit in the paper.

## 6. Corrected baseline (what v1.7 actually achieves)

Using corrected, honest numbers:

| dimension | value |
|---|---|
| predictive gain vs constant | +35.7 % (n=8000), +46.8 % (n=12000), +29–31 % under nuisance violations |
| operating range | ψ ≥ 6000; plug-in ≥ 4000 and better at 4000–6000; crossover ≈8000 |
| subgroup purity / capture | 0.67–0.83 / **0.23–0.25** |
| split stability | 0.16–0.20 |
| decision (honest split) | commit 40 %, coverage 0.065, captured 0.248, null 0 %, efficiency 0.83 |
| estimation bias | ≤ 0.005 |

## 7. What the audit says to do next

The evidence points to two distinct problems, and neither is "next gate in the
list":

1. **Decision leakage (must fix first).** Re-establish the commitment claim with
   a validation set that is separate from the assessment set. This is a
   correctness fix, not an optimization. Expected outcome: commit rate 40–55 %,
   coverage 0.065–0.075, captured 0.25–0.28, null rate ~0–5 %.
2. **Low capture and low stability (the real scientific weaknesses).** The high
   leaf captures only ~24 % of the true region and its split stability is ~0.18.
   Both are consistent with the winner's-curse selection identified in §5.2.
   A targeted experiment should compare selection rules (max mean, lower
   confidence bound, minimum-support-constrained, stability-selected) on
   capture, purity and stability, with prediction and decision metrics as
   guardrails.

The audit does **not** support proceeding to a fixed-capacity frontier yet: the
decision metrics are not yet honest, and the subgroup is not yet stable enough
for a capacity policy to be meaningful.

## Audited files

`ciscarl2/rules.py`, `ciscarl2/estimators.py`, `ciscarl2/dgp.py`,
`ciscarl2/gate2.py` (`pseudo_outcome`), `ciscarl2/gate8.py` (`assign`),
`ciscarl2/gate11.py` (CART, high leaf, leaf stats, stability),
`ciscarl2/gate14.py` (misspecified DGP), `ciscarl2/gate15.py`–`gate19.py`
(metrics, commitment, summaries), `ciscarl2/audit.py`,
`ciscarl2/audit_commit.py`.

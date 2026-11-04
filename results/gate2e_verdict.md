# Gate 2e — decision layer: diagnosis and verdict

100 replications · n = 589 · sign-flip DGP · 2,385 rules · base rate 28.4 %
beneficial.

## Pre-registered criterion (and why it was wrong)

Fixed before the run: *salvageable if some threshold achieves TREAT precision ≥ 90 %
**and** ≥ 1 TREAT verdict in ≥ 50 % of replications.*

**Result: no threshold satisfies both.** k = 3.0 gives 83 % precision but only 10 % of
replications commit; k = 3.5 gives 100 % precision with 4 verdicts in 4 % of
replications.

But the criterion itself was badly specified, and I should own that. At a 28.4 % base
rate, **even an oracle test with perfect power and a nominal 5 % false-positive rate
caps out at**

$$\frac{0.284}{0.284 + 0.716 \times 0.05} = 88.8\% .$$

A 90 % precision requirement was therefore unreachable by construction. That is a
design error in my gate, not a discovery about the method.

## The real diagnosis: power, not calibration

The correct test is the **false-commitment rate among rules whose true effect is near
zero** — not coverage binned by |z|, which is circular (conditioning on |τ̂| > 1.96·SE
mechanically excludes zero from the interval).

| DGP | near-null subset | committed at 1.96 | nominal |
|---|---|---|---|
| all-beneficial | \|τ\| < 0.05 (n=92) | **6.5 %** [2.9, 10.7] | 5 % |
| sign-flip | \|τ\| < 0.05 (n=858) | **7.5 %** [5.9, 8.9] | 5 % |
| sign-flip | \|τ\| < 0.10 (n=1578) | **8.3 %** | 5 % |

So the intervals over-commit at ~7.5–8 %, about 1.5× nominal. **Modestly
anti-conservative, not broken.** My earlier "coverage collapses to 0 % at high |z|"
alarm was an artefact of the binning.

**The 67 % TREAT precision is what a valid test with ~38 % power produces at a 28 %
base rate.** Inverting the precision identity,

$$0.67 = \frac{0.284\pi}{0.284\pi + 0.716 \times 0.075} \;\Rightarrow\; \pi \approx 0.38 .$$

An implied power of 38 % at an SE of ~0.17 against true effects averaging 0.09. The
method is behaving as the statistics dictate; it is not malfunctioning.

## What the other diagnostics show

- **Support does not matter.** Coverage 92.7–95.0 % across support quartiles; precision
  is not concentrated in small rules.
- **Rule stability does not predict accuracy.** |error| is 0.140–0.147 whether stability
  is above or below the median. So the failure is *not* selection optimism from
  discovery — the outer cross-fit is doing its job.
- **The winner's curse is exactly as arithmetic predicts.** TREAT estimates average
  +0.395 against a true +0.016 — an inflation of +0.379, which is ≈ 2.4 SE, precisely
  the conditional mean of a normal truncated at 1.96.
- **Per-replication precision is bimodal**: median 100 %, IQR [0, 100]. Half the
  committing replications are perfectly right and half are perfectly wrong. With 48
  verdicts across 100 replications, the aggregate 67 % is an average over a coin flip.

## Why no threshold can fix this

The sweep traces a single trade-off curve, not a fixable defect:

| k | TREAT n | precision | commitment | reps with ≥1 |
|---|---|---|---|---|
| 1.96 | 53 | 62 % | 2.2 % | 36 % |
| 2.50 | 23 | 74 % | 1.0 % | 21 % |
| 3.00 | 12 | 83 % | 0.5 % | 10 % |
| 3.50 | 4 | 100 % | 0.2 % | 4 % |

Precision rises only because n falls. At k = 3.5 the method is precise by having almost
nothing to say. There is no interior optimum — because the problem is not where the
threshold sits.

## The binding constraint, quantified

For a rule to be resolvable at z ≈ 2 with a true effect of 0.09, its SE must be ≈ 0.045.
Since SE ∝ n_r^{-1/2}, and SE ≈ 0.17 at n_r ≈ 32:

$$n_r \approx 32 \times (0.17/0.045)^2 \approx 457 .$$

**Rules need roughly 450 patients.** With 589 training patients that means one rule
covering three-quarters of the cohort — which is not subgroup analysis.

## Verdict: STOP on the decision layer at n = 589

Not "revise the threshold" and not "fix the variance estimator". At ACTG175 scale the
effect-to-SE ratio is ~0.5, and no decision rule can deliver both precision and
coverage. The decision layer is **power-limited**, and power is set by sample size.

| component | status |
|---|---|
| estimation (bias) | sound |
| marginal interval coverage | ~94 % |
| near-null false-commitment | 7.5 % (≈1.5× nominal) — modestly anti-conservative |
| TREAT precision at n = 589 | 67 %, consistent with 38 % power |
| decision layer salvageable by threshold | **no** |
| salvageable by better variance estimate | **no** — the variance is approximately right |

## What this means for the paper

1. **The contribution cannot be "reliable subgroup decisions at ACTG175 scale."** It
   can be: a framework that is unbiased and approximately calibrated, whose decision
   layer is power-limited, with the sample-size requirement quantified (~450 per rule).
2. **Report the base-rate ceiling alongside precision.** Any precision claim without it
   is uninterpretable.
3. **The honest output at n = 589 is abstention.** 90 % abstention is not a bug; given
   a 0.5 effect-to-SE ratio it is the correct behaviour.
4. **A power calculation belongs in the methods section**, not the limitations: it tells
   the reader when the method can be used.

## Not established

- Whether any of this transfers to real ACTG175 data (the estimand question is still
  open, and that is now the dominant unresolved risk).
- Whether a materially better estimator (TMLE, or borrowing across rules) could reduce
  SE enough to change the conclusion. It would need to cut SE by ~4×, which is not
  plausible.

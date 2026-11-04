# Gate 2-funnel — attribution of empty rule lists

100 replications per cell · discovery only (no effect estimation) · same seeds as
Gate 2 / 2b.

## Result: every null is the stability threshold

| cell | reps | null rate | no candidates | stability filter | mean candidates | mean stable | best freq (mean) | best freq (p10) | median support |
|---|---|---|---|---|---|---|---|---|---|
| independent 10 % | 100 | **25 %** | **0 %** | **25 %** | 158.6 | 3.9 | 0.69 | **0.50** | 143 |
| covariate 25 % | 100 | **23 %** | **0 %** | **23 %** | 159.1 | 3.5 | 0.67 | **0.50** | 140 |

## What this excludes

Of the six candidate explanations:

| Hypothesis | Verdict |
|---|---|
| 1. Insufficient subgroup support | **excluded** — median support 143, well above the 60 floor |
| 2. Too few events in one arm | **excluded** — same censoring calibration as the successful reps |
| 3. Candidate rules failing eligibility | **excluded** — ~159 candidates every replication |
| 4. Discovery objective too weak | **partly** — see below |
| 5. Pruning removing all candidates | **excluded** — pruning runs after stability, and never sees an empty input |
| 6. Threshold too strict for the sample size | **this is the cause** |

**Zero replications ever failed at candidate generation.** The pipeline produces
~159 candidate rules, of which 3.5–3.9 survive on average. In the failing 23–25 %, no
single candidate reaches the 0.6 stability threshold: the 10th percentile of the *best*
candidate's frequency is exactly 0.50, i.e. a coin flip under half-sample resampling.

## Interpretation

This is not a bug and not a data problem. It is the expected behaviour of stability
selection applied to a **noisy discovery objective** (ψ has sd ≈ 0.9 against τ ≈ 0.2)
with a **half-sample** resampling scheme at n = 1000. Half-samples of 500 give the
ensemble enough noise that split thresholds move between replicates, so the Jaccard-0.8
match fails for roughly half the rules in the hardest quartile of datasets.

Two features of the current configuration make it stricter than necessary:

1. **Threshold 0.6 is a default, not a tuned value.** Meinshausen–Bühlmann discuss
   threshold choice but do not prescribe 0.6 for this regime.
2. **Half-sample subsampling** is the classical choice and is the most conservative;
   it maximises the difficulty of rediscovery.

## What follows

The fix is a **parameter choice**, not an algorithmic change: sweep the stability
threshold and measure what it costs in false rules. It is cheap — the existing Gate 2b
harness runs a cell in ~3 minutes, so a 4-point sweep is ~25 minutes.

The sweep should report, per threshold:

- null rate
- number of rules per replication
- **overlap of the selected rules' members with the true effect-driving regions**
  (a discovery-quality metric, not just a count)
- coverage of the resulting intervals

Choosing purely on null rate would drive the threshold to zero and admit noise rules;
choosing purely on precision would keep the 25 % null rate. The trade-off has to be
measured.

## Recommended ordering

| Step | Why |
|---|---|
| 1. Threshold sweep (Gate 2c) | cheap; changes which rules exist, so it must precede anything rule-dependent |
| 2. Freeze the threshold | one parameter, documented |
| 3. Gate 3 (delta sensitivity, ≥100 reps) | runs on a stable rule supply and does not need re-running later |

This differs slightly from the recommendation to run Gate 3 first. The reason is
narrow: Gate 3 validates the sensitivity **method**, which is orthogonal to rule
selection, but its effective sample size is the number of discovered rules. If the
threshold changes afterwards, Gate 3 would have to be repeated for a credible margin
of error — not because its conclusion would change, but because its precision would.
A 25-minute sweep is cheaper than that repetition.

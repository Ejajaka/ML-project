# Gate 2d — application-matched sample size: verdict

100 replications · **n = 589** · S0(t\*) = 0.735 · τ ≈ 0.13 · 10 % censoring at
730 d · outer 5-fold (discover on 4/5, evaluate on the held-out 1/5) · v1.2 config.

**DGP check:** τ = 0.118 (ACTG175 0.128), censored-before-t\* = 10.2 % (ACTG175
9.8 %), mean S0 = 0.766 (ACTG175 0.735). The matched simulation reproduces the
application's operating conditions.

## Side by side with the n = 2000 gates

| metric | n = 589 | n = 2000 |
|---|---|---|
| null rate (no rule in any fold) | **0 %** | 2–10 % |
| rules per replication | 23.3 | 2.9–3.0 |
| mean support (evaluation fold) | **32** | ~127 |
| mean stability | 0.66 | 0.61–0.63 |
| bias | −0.003 | −0.012 |
| RMSE | 0.186 | — |
| 95 % coverage | 94.8 % | 94.5–94.7 % |
| mean CI width | **0.732** | 0.35–0.55 |
| effect size | 0.141 | 0.21 |
| **width / effect** | **5.2×** | 1.7–2.6× |
| abstain | **87 %** | — |
| TREAT | 12 % | — |
| mean IoU | 0.23 | 0.19–0.26 |

Rules per replication rises from ~3 to ~23 because each of the five outer folds
yields its own list, and the rules are much smaller (support 32 rather than 127).

## The headline is not "87 % abstention"

| verdict | count | share |
|---|---|---|
| INSUFFICIENT | 2,029 | 87.1 % |
| TREAT | 288 | 12.4 % |
| AVOID | 13 | 0.6 % |

**TREAT precision = 100 % (288/288).** Every committed treat verdict had a truly
positive effect. Committed rules also had larger true effects than abstained ones
(0.159 vs 0.138, same mean support 33 vs 32). **91 % of replications produced at
least one committed verdict** (mean 2.9 per replication).

So at ACTG175 scale the method behaves as a **high-precision, low-recall detector**:
it abstains most of the time, and when it commits it is right in this simulation.

## Two honest caveats on that precision figure

1. **The DGP has no harmful subgroups.** Every τ is positive, so TREAT precision is
   near-tautological (TREAT requires lo > 0). The 13 AVOID verdicts cannot be
   evaluated at all. A DGP with negative-effect subgroups is required before any
   claim about AVOID or about two-sided precision.

2. **Sign accuracy of 77 % across all rules is the wrong metric here.** It is
   dominated by abstained rules whose point estimates are noise (74 % among
   abstained). Among committed verdicts, sign accuracy is 100 % by construction.
   The paper should report TREAT/AVOID precision, not overall sign accuracy.

## What actually changed at application scale

- **Interval width is 5.2× the effect** (0.73 against 0.14). This is the binding
  constraint: the estimator is unbiased (−0.003) but each rule carries a CI spanning
  roughly five times the quantity it is trying to resolve.
- **Rules are small** (median support 32 in the evaluation fold, quartiles 26–37).
  The 60-patient discovery floor is applied in a 471-patient discovery set, so a rule
  that clears it occupies only ~25–35 patients in a 118-patient evaluation fold.
- **Coverage holds (94.8 %)** only because the intervals are large. Calibration and
  informativeness are different things, and at this sample size the method buys the
  former with the latter.

## Answer to the question the gate was built for

> **Does CISCaRL produce useful subgroup findings at ACTG175 scale?**

**Partially, and in a specific sense.** It produces rules in every replication, and it
commits to a verdict in 91 % of replications, with 100 % precision on those
commitments in this simulation. But it abstains on 87 % of the rules it evaluates,
and the intervals are 5× the effect size. The output is a small number of
high-confidence statements, not a rich rule list.

That is a defensible product — "we tell you a little, and we are right" — but it is
not the "compact interpretable rule list" the original framing implied.

## Consequences

1. **Report precision/recall of verdicts, not sign accuracy.** They are the metrics
   that match how the output is used.
2. **State the operating envelope.** The method is validated at n ≈ 600–2000; at the
   low end it is a high-precision, low-recall detector.
3. **A negative-effect DGP is needed** to say anything about AVOID, and it is cheap
   (flip the sign of τ for one subgroup).
4. **Gate 4 must be run at both sample sizes**, or its conclusion will not transfer.

## Still blocking the real analysis

The `days` / `cens` / `offtrt` semantics remain unresolved. Nothing in this gate
changes that: at ACTG175 scale the method will abstain on most rules, so the
real-data output will consist of a handful of statements — and those statements are
only as meaningful as the estimand they target.

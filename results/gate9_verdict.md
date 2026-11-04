# Gate 9 verdict — the plug-in target fixes discovery but not prediction

Pre-registration: `gate9_PREREGISTRATION.md`. 30 replications, n = 2000. Only the
discovery target changes; effect estimation is the held-out AIPW φ in every arm.

## Pre-registered result

| target | target sd | corr with true τ | Jaccard | stability | rules | MAE | constant | Δ vs constant | 95 % CI | reps better |
|---|---|---|---|---|---|---|---|---|---|---|
| AIPW ψ (baseline) | 0.973 | 0.08 | 0.107 | 0.57 | 1.4 | 0.0787 | 0.0760 | +0.0027 | [−0.0004, +0.0060] | 23 % |
| Cox plug-in | 0.090 | 0.56 | **0.622** | **0.99** | 26.1 | 0.0807 | 0.0760 | +0.0047 | [−0.0024, +0.0122] | 43 % |
| RSF plug-in | 0.070 | 0.49 | **0.606** | **0.99** | 27.7 | 0.0853 | 0.0760 | +0.0094 | [+0.0027, +0.0164] | 40 % |

| check | Cox | RSF |
|---|---|---|
| mean Δ < 0 | FAIL | FAIL |
| CI entirely below 0 | FAIL | FAIL |
| Jaccard not worse than AIPW-ψ | PASS | PASS |
| ≥ 70 % reps better | FAIL | FAIL |

**Neither target is adopted.** The RSF arm is significantly worse than the
constant. The 2-replication smoke test that appeared to favour RSF did not
reproduce — recorded here as a reminder that 2 reps cannot support anything.

## What the target did fix

The plug-in target is a genuine discovery improvement:

* sd falls 0.97 → 0.070–0.090 (10–14× less noisy);
* correlation with the true individual τ rises 0.08 → 0.49–0.56;
* best-rule Jaccard rises 0.107 → 0.61–0.62;
* rule stability rises 0.57 → 0.99.

Gate 6's conclusion is confirmed: given a clean target, the rule machinery finds
the region.

## Why MAE still got worse — abstention, not discovery

Per-replication inspection of the assignment:

* the ψ pipeline produces 0–4 rules, covers ~26 % of test patients, and falls
  back to the constant when nothing passes the stability filter;
* the plug-in pipeline produces 20–40 rules, covers **100 %** of test patients,
  and assigns an AIPW effect to every one of them.

Illustrative replication (rep 11):

| | region MAE | null-region MAE | coverage |
|---|---|---|---|
| ψ | 0.167 | 0.028 | 0.00 |
| Cox plug-in | **0.042** | **0.125** | 1.00 |

The clean target partitions the whole covariate space. Each leaf's effect is an
AIPW estimate with SE ≈ 0.1, and 75 % of patients have true effect 0. So the
gain in the region is cancelled by variance injected into the null region.

## The two obvious remedies do not work (5-rep pilot)

| arm | rules | MAE | Δ vs constant |
|---|---|---|---|
| Cox plug-in, plain assignment | 24.8 | 0.0810 | +0.0067 |
| Cox plug-in, commit only if the rule's t-interval excludes the constant | 24.8 | 0.0739 | −0.0004 |
| Cox plug-in, empirical-Bayes shrinkage toward the constant | 24.8 | 0.0763 | +0.0020 |

Commitment removes the null-region damage but only returns to break-even.
Shrinkage does not help. The rule list itself is the problem.

## Where the loss actually is — the extraction layer

Diagnostic on the same 30 replications (pilot; seeds overlap Gate 9, so this
motivates Gate 10 and is not itself confirmatory):

| method | rules | MAE | Δ vs constant | 95 % CI | reps better |
|---|---|---|---|---|---|
| constant | 0 | 0.0760 | — | — | — |
| **smooth plug-in predictor τ̂(X), no rules** | — | **0.0593** | **−0.0167** | **[−0.0209, −0.0123]** | 87 % |
| **single-split CART on τ̂** | 2 | **0.0692** | −0.0068 | [−0.0136, +0.0012] | 77 % |
| CART depth 2 on τ̂ | 3.9 | 0.0779 | +0.0019 | [−0.0040, +0.0087] | 57 % |
| CART depth 3 on τ̂ | 6.9 | 0.0832 | +0.0073 | [−0.0003, +0.0153] | 43 % |
| RF + stability on τ̂ (Gate 9 arm) | 26.1 | 0.0807 | +0.0047 | [−0.0022, +0.0123] | 43 % |

Three things follow.

1. **The information is real and available**: the smooth plug-in predictor beats
   the constant by 22 %, with a CI well below zero. This is the first method in
   the project to beat the constant baseline decisively.
2. **The loss is in the representation, not the target**: the same target
   predicts well as a continuous function, and loses the advantage as soon as it
   is discretised into rules.
3. **Fewer rules is monotonically better** here: 2 rules > 4 > 7 > 26. This is
   the asymmetric MAE loss at work — assigning a benefit to a null patient costs
   ~0.11, while missing a region patient costs only the marginal ~0.05. Precision
   matters roughly twice as much as recall.

The single split is chosen on `age` in 24/30 replications, `cd40` in 5 and
`symptom` in 1, with a high-τ leaf covering 25 % of test patients and an AIPW
effect of +0.112 (true region effect 0.195, constant 0.050).

## Caveats

* The Cox outcome model is correctly specified in this DGP, so τ̂ is consistent.
  The gain could shrink under misspecification. The RSF plug-in as a *predictor*
  also beat the constant in the 5-rep pilot (0.0649), but it has not been tested
  at 30 reps.
* The 2-rule arm is **not yet significant** at 30 reps: CI [−0.0136, +0.0012].
  It needs more replications before any claim.
* These 30 replications are the Gate 9 seeds. A confirmatory run must use fresh
  seeds.
* Nothing here addresses n = 589, where Gate 6 showed even perfect discovery
  cannot beat the constant.

# Gate 2d-signflip — do the verdicts survive when harmful subgroups exist?

100 replications · n = 589 · outer 5-fold · v1.2 config · same ACTG175-matched
conditions as Gate 2d, but the treatment effect now **changes sign**:

$$\text{effect}(X) = 0.60 \times \text{cd40}_z$$

so treatment is beneficial for high CD4 and harmful for low CD4, with the crossover at
the median. 72 % of evaluated rules have a truly harmful effect.

## Why this gate exists

Gate 2d reported **TREAT precision 288/288 = 100 %** and this was read as evidence of a
"high-precision detector". That reading was wrong. TREAT requires the interval lower
bound to exceed zero; in a DGP where every true effect is positive, *every* TREAT
verdict is necessarily correct. The 100 % was a property of the DGP, not of CISCaRL.
This gate removes that artefact.

## Result

| metric | all-beneficial DGP | **sign-flip DGP** |
|---|---|---|
| rules evaluated | 2,330 | 2,385 |
| mean support | 32 | 32 |
| abstain | 87 % | 90 % |
| **TREAT verdicts** | 288 | 48 |
| **TREAT precision** | 100 % | **67 % ± 6.8** |
| **AVOID verdicts** | 13 | 195 |
| **AVOID precision** | not evaluable | **95 % ± 1.6** |
| 95 % coverage | 94.8 % | 93.6 % |
| mean CI width | 0.732 | 0.718 |
| true-beneficial rules | 100 % | 28 % |

## What the numbers mean

**1. One in three TREAT commitments is wrong.** 48 TREAT verdicts, 32 on truly
beneficial subgroups, 16 on truly harmful ones. Recommending a harmful treatment is the
costliest error this framework can make, and at ACTG175 scale it happens a third of the
time it commits.

**2. The cause is selection on noise, not a broken estimator.** The estimator remains
nearly unbiased (−0.006) and coverage is 93.6 %. But with a CI width of 0.72 against
effects of magnitude ~0.1–0.2, a rule only receives a TREAT verdict by landing in the
upper tail. Conditional on clearing that threshold, the estimate is biased upward — the
winner's curse. Selection makes the *verdict* far less reliable than the *coverage*
number suggests.

**3. AVOID is more reliable (95 %) but partly because it is the majority direction.**
With 72 % of rules truly harmful, always-AVOID would score 72 % precision. CISCaRL
achieves 95 %, so it adds real value — but a 1.3× lift, not a discovery.

**4. Against base rates, the verdicts are informative but modest.**

| verdict | precision | base rate | lift |
|---|---|---|---|
| TREAT | 67 % | 28 % | 2.4× |
| AVOID | 95 % | 72 % | 1.3× |

## Consequence: the "high-precision detector" framing is dead

The correct description at ACTG175 scale is:

> CISCaRL abstains on ~90 % of rules. When it commits, it is substantially better than
> chance (2.4× for TREAT, 1.3× for AVOID) but not reliable: at n = 589 with
> mixed-direction effects, one in three TREAT commitments lands on a truly harmful
> subgroup.

This is not a clinical decision tool at this sample size. It is an exploratory
screening device whose abstention mechanism works far better than its commitment
mechanism.

## What must change in the reporting

1. **Never report verdict precision without the base rate.** 100 % precision in an
   all-positive world is free; 67 % against a 28 % base rate is the real number.
2. **Report lift over base rate** as the headline decision metric.
3. **Report TREAT and AVOID separately** — they have very different reliability and the
   asymmetry is real.
4. **Report the winner's-curse gap**: unconditional coverage (93.6 %) versus conditional
   precision (67 %). The gap is the cost of committing.

## Implications for the project

- The framework's validated strength is **abstention**, not commitment. That is the
  opposite of the original framing ("compact list of actionable rules").
- Any paper claim must be **conditioned on the effect structure**: precision depends
  heavily on whether harmful subgroups exist and on their prevalence.
- **Gate 4 must use a mixed-direction DGP.** A comparison against SCRE on an
  all-beneficial DGP would flatter both methods.
- A higher commitment threshold (wider CI requirement than 1.96 SE) would trade
  abstention for precision; that is a cheap, testable fix and probably the single most
  valuable next experiment for the *decision* layer.

## Reproduce

```
python -m ciscarl2.gate2d --reps 100 --signflip
```

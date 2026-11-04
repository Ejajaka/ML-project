# Gate 16 verdict — ψ confirmed as the v1.6 discovery target

Pre-registration: `gate16_PREREGISTRATION.md`. 40 fresh seeds per cell, depth-3
CART, held-out AIPW effects, TRAIN constant.

## Result

| n | κ | target | MAE | constant | Δ | 95 % CI | reps better | gain |
|---|---|---|---|---|---|---|---|---|
| 8000 | 0 | **ψ** | **0.0452** | 0.0726 | **−0.0275** | [−0.0317, −0.0233] | **100 %** | **+37.8 %** |
| 8000 | 0 | plug-in | 0.0522 | 0.0726 | −0.0205 | [−0.0250, −0.0160] | 92 % | +28.2 % |
| 8000 | 3 | **ψ** | **0.0491** | 0.0731 | **−0.0239** | [−0.0288, −0.0187] | **95 %** | **+32.7 %** |
| 8000 | 3 | plug-in | 0.0760 | 0.0731 | +0.0030 | [−0.0005, +0.0064] | 38 % | −4.0 % |
| 12000 | 0 | **ψ** | **0.0372** | 0.0715 | **−0.0344** | [−0.0385, −0.0298] | **100 %** | **+48.0 %** |
| 12000 | 0 | plug-in | 0.0483 | 0.0715 | −0.0232 | [−0.0266, −0.0196] | 98 % | +32.5 % |
| 12000 | 3 | **ψ** | **0.0340** | 0.0729 | **−0.0390** | [−0.0429, −0.0348] | **100 %** | **+53.4 %** |
| 12000 | 3 | plug-in | 0.0682 | 0.0729 | −0.0048 | [−0.0094, −0.0006] | 57 % | +6.5 % |

ψ passed **all four pre-registered checks in all four cells**. The plug-in failed
the consistency check at κ=3 for both n (38 % and 57 %).

## Paired comparison on identical replications

| n | κ | ψ − plug-in MAE | 95 % CI | reps favouring ψ |
|---|---|---|---|---|
| 8000 | 0 | −0.0070 | [−0.0118, −0.0022] | 65 % |
| 8000 | 3 | −0.0269 | [−0.0326, −0.0212] | 92 % |
| 12000 | 0 | −0.0111 | [−0.0146, −0.0074] | 82 % |
| 12000 | 3 | −0.0342 | [−0.0392, −0.0289] | 95 % |

ψ is better at every sample size and nuisance state, including the fully correct
DGP. The Gate 15 advantage was not seed-specific.

## Diagnostics

| n | κ | target corr | Jaccard | purity | capture | stability |
|---|---|---|---|---|---|---|
| 8000 | 0 | 0.085 | 0.70 | 0.72 | 0.37 | 0.21 |
| 8000 | 3 | 0.086 | 0.65 | 0.73 | 0.30 | 0.20 |
| 12000 | 0 | 0.082 | 0.77 | 0.69 | 0.35 | 0.20 |
| 12000 | 3 | 0.085 | 0.84 | 0.73 | 0.43 | 0.25 |

The individual-level correlation is ~0.08 in every cell, yet Jaccard is 0.65–0.84.
This confirms that **correlation is the wrong diagnostic**: the tree uses leaf
means, and ψ is unbiased, so a leaf of several hundred patients estimates its
own effect precisely enough to be informative.

The marginal ATE is unbiased in every cell (0.046–0.049 vs 0.048), including
κ=3, as double robustness requires.

**Stability trade-off:** ψ's split stability is 0.20–0.25, against the plug-in's
0.61–0.68. The ψ rules are better predictors and more robust, but the rule list
is much less reproducible across half-samples. This must be stated, not hidden.

**Gate 15 anomaly:** the κ=3 gain exceeding the κ=0 gain did **not** replicate
consistently (n=8000: 32.7 % vs 37.8 %; n=12000: 53.4 % vs 48.0 %). It should be
treated as seed noise, not a mechanism. The event and censoring fractions are
~0.19 and ~0.10 in every cell, so there is no process-level explanation to give.

## Decision

**Adopt ψ as the v1.6 discovery target** at n ≥ 8000, with depth-3 CART
extraction and held-out AIPW effects.

Consequences:

* the method is **simpler** (no plug-in),
* **more accurate** at n ≥ 8000 (+38–48 % vs +28–33 %),
* **robust** to the Gate 14 outcome-model violation (+33–53 % vs −4 %/+6.5 %),
* at the cost of **partition stability** (0.20–0.25 vs 0.61–0.68).

The Gate 14 dependence clause is thereby removed **for discovery**, because ψ is
doubly robust — but only while the censoring model is correct. That condition is
untested and becomes the next gate.

## Limits

* Correct censoring model in every cell. ψ's double robustness needs *at least
  one* nuisance correct; joint misspecification is untested.
* One DGP family, one extractor, n ≥ 8000. The lower-n boundary for this
  extractor is not established (the n = 2000 failures used the RF + stability
  pipeline, not this one).

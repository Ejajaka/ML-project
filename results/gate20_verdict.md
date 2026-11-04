# Gate 20 verdict — honest commitment validated; validation size matters

Pre-registration: `gate20_PREREGISTRATION.md`. n = 8000, 4000 discovery / 2000
validation / 2000 assessment, 40 fresh seeds (230100–230139), depth-3 ψ
discovery. Nuisances fitted on discovery only; assessment never used for
decisions.

## Joint cell (κ=3, marginal KM)

| validation | arm | commit | coverage | per-treated | efficiency | captured | null | beneficial | MAE | Δ vs constant |
|---|---|---|---|---|---|---|---|---|---|---|
| full (2000) | A | 100 % | 0.113 | +0.127 | 0.68 | 0.304 | 17.5 % | 82.5 % | 0.0574 | −0.0129 [−0.0180, −0.0077] |
| **full (2000)** | **B** | **50 %** | **0.092** | **+0.161** | **0.88** | **0.258** | **0 %** | **100 %** | 0.0571 | −0.0131 [−0.0183, −0.0081] |
| half (1000) | A | 100 % | 0.113 | +0.127 | 0.68 | 0.304 | 17.5 % | 82.5 % | 0.0786 | +0.0083 |
| half (1000) | B | 42.5 % | 0.069 | +0.163 | 0.87 | 0.184 | 0 % | 100 % | 0.0787 | +0.0085 |

## Reference cell (κ=0, conditional)

| validation | arm | commit | coverage | per-treated | efficiency | captured | null | MAE | Δ |
|---|---|---|---|---|---|---|---|---|---|
| full | A | 100 % | 0.071 | +0.133 | 0.67 | 0.243 | 17.5 % | 0.0561 | −0.0157 |
| **full** | **B** | **40 %** | **0.053** | **+0.180** | **0.91** | **0.199** | **0 %** | 0.0559 | −0.0159 |
| half | A | 100 % | 0.071 | +0.133 | 0.67 | 0.243 | 17.5 % | 0.0751 | +0.0034 |
| half | B | 27.5 % | 0.031 | +0.186 | 0.93 | 0.116 | 0 % | 0.0754 | +0.0036 |

## Pre-registered checks (joint cell, V-full, arm B)

| criterion | result |
|---|---|
| safety: null rate ≤ 5 % | **PASS (0 %)** |
| utility: efficiency ≥ 0.70 | **PASS (0.88)** |
| utility: captured ≥ 0.20 | **PASS (0.258)** |
| captured ≥ 0.70 × A | **PASS (0.258 ≥ 0.213)** |
| prediction: Δ < 0, CI < 0 | **PASS (−0.0131 [−0.0183, −0.0081])** |

**Honest commitment adopted.**

## The power/honesty separation

The validation size is the decisive variable:

| | commitment | coverage | captured | null | efficiency | prediction |
|---|---|---|---|---|---|---|
| full 2000 | 50 % | 0.092 | 0.258 | 0 % | 0.88 | beats constant |
| half 1000 | 42.5 % | 0.069 | 0.184 | 0 % | 0.87 | **loses to constant** |

Two distinct effects:

1. **Quality is stable.** Null rate 0 % and efficiency 0.87–0.88 at both
   validation sizes. The commitment rule is not overfitting; it removes the
   null leaves reliably.
2. **Quantity and prediction need ≥2000 validation patients.** Halving the
   validation set costs ~25 % of coverage and ~29 % of captured benefit, and the
   leaf effects become too noisy for the prediction to beat the constant
   (Δ = +0.0083, CI entirely above zero).

So the audit's loss was **partly power** (V-full recovers coverage 0.069 → 0.092
and captured 0.184 → 0.258) and **partly honesty** (the full-sample Gate 19
captured 0.36, which does not reproduce under either honest validation size).

## Paired effects of commitment (V-full)

* null-treatment rate falls by **17.5 pp** (paired CI [0.075, 0.300]);
* captured benefit falls by **4.4–4.7 pp** (CI excludes zero) — a real but
  modest cost;
* MAE is unchanged (paired B−A ≈ 0), so commitment does not damage prediction;
* efficiency rises from 0.68 to 0.88, per-treated benefit from +0.127 to +0.161.

## What this establishes

The decision layer is now honestly validated: it eliminates null-benefit
treatment (17.5 % → 0 %) and raises decision efficiency to 0.88, at the cost of
~5 pp of captured benefit and 50 % abstention, with no prediction loss.

## What it does not establish

* **The winner's curse in leaf selection remains.** The high leaf is still chosen
  by maximum discovery mean ψ; in one diagnostic it selected a 64-patient extreme
  leaf (ψ = 0.434) over a 321-patient leaf (ψ = 0.335). This is why the
  no-commitment null rate is 17.5 % and capture is only 0.30.
* Capture is still 0.24–0.30 and split stability 0.16–0.17.
* One sample size, two nuisance cells, one partition ratio.

These are the next diagnosed problems; Gate 20 was not designed to fix them.

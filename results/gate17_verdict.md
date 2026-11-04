# Gate 17 verdict — the threshold is ~4000, and the best target depends on n

Pre-registration: `gate17_PREREGISTRATION.md`. 40 fresh seeds per cell, κ = 0,
depth-3 CART, held-out AIPW effects.

## Result

| n | target | MAE | constant | Δ | 95 % CI | reps better | gain |
|---|---|---|---|---|---|---|---|
| 2000 | **ψ** | 0.0903 | 0.0735 | **+0.0168** | [+0.0101, +0.0232] | 20 % | **−22.8 %** |
| 2000 | plug-in | 0.0867 | 0.0735 | +0.0132 | [+0.0053, +0.0210] | 32 % | −17.9 % |
| 2000 | **oracle** | **0.0440** | 0.0735 | **−0.0295** | [−0.0371, −0.0214] | 85 % | **+40.1 %** |
| 4000 | **ψ** | 0.0670 | 0.0729 | **−0.0059** | [−0.0111, −0.0005] | **70 %** | **+8.1 %** |
| 4000 | plug-in | 0.0608 | 0.0729 | −0.0121 | [−0.0184, −0.0058] | 72 % | +16.6 % |
| 4000 | oracle | 0.0295 | 0.0729 | −0.0434 | [−0.0490, −0.0376] | 100 % | +59.5 % |
| 6000 | **ψ** | 0.0561 | 0.0729 | **−0.0168** | [−0.0228, −0.0107] | **85 %** | **+23.0 %** |
| 6000 | plug-in | 0.0553 | 0.0729 | −0.0176 | [−0.0224, −0.0126] | 90 % | +24.2 % |
| 6000 | oracle | 0.0282 | 0.0729 | −0.0447 | [−0.0496, −0.0399] | 100 % | +61.3 % |

**Pre-registered decision (ψ):** FAIL at n=2000; PASS at n=4000 (marginally:
70 % exactly, CI upper bound −0.0005) and n=6000.

## Three findings

### 1. The old n ≥ 8000 threshold was partly an extractor artefact

The v1.6 configuration (ψ + depth-3 CART) beats the constant from n = 4000,
not n = 8000. The earlier threshold was measured with the RF + stability
pipeline, which produced 0–4 rules and no gain at n = 2000. Replacing the
extractor moved the operating range down by roughly one doubling.

The paper's sample-size section must therefore distinguish **two thresholds**:

* ~4000 for the v1.6 model (ψ + single CART);
* ~8000 for the old RF + stability pipeline.

### 2. The best target depends on n — a crossover at ~6000

| n | plug-in gain | ψ gain | better |
|---|---|---|---|
| 2000 | −17.9 % | −22.8 % | neither works |
| 4000 | **+16.6 %** | +8.1 % | plug-in |
| 6000 | +24.2 % | +23.0 % | tie |
| 8000 | +28.2 % | **+37.8 %** | ψ |
| 12000 | +32.5 % | **+48.0 %** | ψ |

At low n the plug-in's low variance is worth more than ψ's unbiasedness; at high
n the leaf-level averaging makes ψ precise *and* ψ is robust. The crossover is
around n = 6000. This is a coherent mechanism, not a nuisance:

* plug-in variance ≈ model-scale (sd 0.08), but biased under misspecification;
* ψ variance ≈ outcome-scale (sd 0.97), unbiased, and its leaf mean SE falls as
  n grows.

### 3. At n = 2000 the barrier is discovery, not information

The oracle — perfect region membership with the same held-out AIPW effects —
achieves **0.0440 vs a 0.0735 constant (+40.1 %)**. So the information is
available at n = 2000; the discovered rules are what fail. This is the
asymmetric-loss/impurity mechanism from Gate 10, not a lack of data.

It also rules out the simplest explanation ("the effect is too small to detect at
n = 2000"): with perfect membership the same estimator wins comfortably.

## Stability

ψ's split stability is 0.19–0.23 across n, against the plug-in's 0.56–0.72. The
ψ rule list is a better predictor and more robust, but less reproducible. At
n = 4000 the plug-in is also more accurate, so below ~6000 it dominates on both
counts.

## Consequence for v1.6

The adopted model is unchanged in form — depth-3 CART on ψ, held-out AIPW
effects — but its **operating range is n ≥ 4000 (marginal at 4000, solid from
6000)**, and the plug-in remains the better choice at 4000–6000 when the outcome
model is trusted. The claim should state both the range and the crossover.

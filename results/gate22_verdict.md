# Gate 22 verdict — the capture gain is reproducible and closer to the truth

Pre-registration: `gate22_PREREGISTRATION.md`. n = 8000, 4000 discovery /
2000 validation / 2000 assessment, 40 fresh seeds (250100–250139). Each
configuration refit with **its own** selection procedure on 20 disjoint
discovery half-samples; support thresholds scaled by the subsample fraction.

## Joint cell (κ=3, marginal KM)

| config | J_full mean | median | q25 | q75 | freq >0.8 | J_true mean | J(L_full, true) | capture |
|---|---|---|---|---|---|---|---|---|
| A: max mean | 0.199 | 0.14 | 0.07 | 0.28 | 4 % | 0.240 | 0.317 | 0.343 |
| B: LCB | 0.274 | 0.22 | 0.11 | 0.39 | 6 % | 0.268 | 0.416 | 0.460 |
| C100 | 0.231 | 0.16 | 0.08 | 0.32 | 5 % | 0.240 | 0.443 | 0.500 |
| **C200 (primary)** | **0.318** | 0.28 | 0.15 | 0.44 | 7 % | **0.334** | **0.522** | **0.623** |
| C400 (sensitivity) | 0.380 | 0.35 | 0.22 | 0.51 | 8 % | 0.387 | 0.537 | 0.660 |
| D: stability-weighted | 0.166 | 0.09 | 0.02 | 0.24 | 3 % | 0.157 | 0.328 | 0.370 |

## Reference cell (κ=0, conditional)

| config | J_full | J_true | capture |
|---|---|---|---|
| A | 0.253 | 0.252 | 0.380 |
| B | 0.272 | 0.273 | 0.400 |
| C100 | 0.264 | 0.252 | 0.530 |
| **C200** | **0.341** | **0.347** | **0.620** |
| C400 | 0.402 | 0.398 | 0.700 |
| D | 0.179 | 0.166 | 0.260 |

## Primary contrast C200 vs A (joint cell)

| metric | difference | 95 % CI | p |
|---|---|---|---|
| reference-leaf recovery (J_full) | **+0.119** | [+0.081, +0.159] | <0.001 |
| ground-truth recovery (J_true) | **+0.094** | [+0.076, +0.116] | <0.001 |
| capture of the true region | **+0.280** | — | — |

Both the reproducibility measure and the truth measure improve. Under the
pre-registered interpretation rule, this is the **"both up"** case: the capture
gain is accompanied by genuine reproducibility and by closer agreement with the
simulated truth.

## Threshold sensitivity (descriptive; 200 remains primary)

| threshold | capture | J_full | J_true |
|---|---|---|---|
| 100 | 0.500 | 0.231 | 0.240 |
| **200** | **0.623** | **0.318** | **0.334** |
| 400 | 0.660 | 0.380 | 0.387 |

The capture gain survives at 100 (0.500 vs A's 0.343) and is larger at 400. The
trend is monotone in support, consistent with the winner's-curse mechanism. Per
the freeze, 400 is reported as sensitivity only and the primary rule stays 200.

## What this establishes

1. **The Gate 21 gain is not an artefact of the invalid stability metric.** At
   the same seeds, C200's full-data leaf is closer to the true region
   (J = 0.522 vs 0.317) and captures more of it (0.623 vs 0.343).
2. **C200 is more reproducible than A** on both the reference-leaf measure
   (+0.119) and the ground-truth measure (+0.094), with CIs excluding zero.
3. **The LCB rule (B) also helps** (+0.075 J_full, +0.028 J_true) but less than
   the support constraint.
4. **The stability-weighted rule (D) fails again**: J_full 0.166, J_true 0.157,
   capture 0.370.

## The honest caveat

Absolute stability is still modest: mean reference-leaf Jaccard 0.32, median
0.28, and only 7 % of half-samples recover the full-data leaf at Jaccard > 0.8.
The distribution is wide (q25 0.15, q75 0.44). C200 is *better* than A, not
*stable* in an absolute sense. A rule list selected on this procedure would
still change materially across resamples.

Per the pre-registered frequency measure, the improvement is +2.6 percentage
points (7 % vs 4 %); per the Jaccard measure it is +0.119. Both are reported;
neither supports a claim of a stable rule list.

## Limits

* One DGP family, two nuisance cells, one sample size, one subsample fraction.
* Threshold 200 was not tuned here; the monotone trend at 400 is descriptive.
* Reproducibility of the *full-data leaf* is not the same as recovery of the
  true boundary; both are reported and both improve, but neither reaches 1.

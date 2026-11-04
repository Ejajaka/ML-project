# Gate 11 verdict — the rule model beats the marginal ATE at n ≥ 8000

Pre-registration: `gate11_PREREGISTRATION.md`. 40 fresh seeds per cell
(120100–120139, 120200–120239, 120300–120339). Depth-2 CART on the cross-fitted
Cox plug-in target, held-out AIPW leaf effects, TRAIN constant per cell.

## Result

| n | model | rules | MAE | constant | Δ | 95 % CI | reps better | gain |
|---|---|---|---|---|---|---|---|---|
| 4000 | **cart_d2 (primary)** | 4.0 | 0.0657 | 0.0724 | −0.0067 | [−0.0118, −0.0013] | 62 % | +9.2 % |
| 4000 | cart_d3 (descriptive) | 7.3 | 0.0644 | 0.0724 | −0.0080 | [−0.0140, −0.0017] | 62 % | +11.0 % |
| 8000 | **cart_d2 (primary)** | 4.0 | 0.0591 | 0.0735 | **−0.0144** | **[−0.0184, −0.0104]** | **82 %** | **+19.6 %** |
| 8000 | cart_d3 (descriptive) | 8.0 | 0.0515 | 0.0735 | −0.0219 | [−0.0267, −0.0169] | 88 % | +29.8 % |
| 12000 | **cart_d2 (primary)** | 4.0 | 0.0574 | 0.0723 | **−0.0149** | **[−0.0188, −0.0110]** | **90 %** | **+20.6 %** |
| 12000 | cart_d3 (descriptive) | 8.0 | 0.0476 | 0.0723 | −0.0247 | [−0.0283, −0.0212] | 100 % | +34.2 % |

## Pre-registered decision (primary = cart_d2)

| n | Δ < 0 | CI < 0 | ≥ 70 % reps | adopted |
|---|---|---|---|---|
| 4000 | PASS | PASS | **FAIL (62 %)** | **no** |
| 8000 | PASS | PASS | PASS (82 %) | **YES** |
| 12000 | PASS | PASS | PASS (90 %) | **YES** |

## Diagnostics (pre-specified, reported not used for selection)

| n | high-leaf prevalence | purity | capture | high-leaf effect | SE | effect/SE | best Jaccard | split stability |
|---|---|---|---|---|---|---|---|---|
| 4000 | 0.147 | 0.75 | 0.43 | +0.146 | 0.095 | 1.54 | 0.41 | 0.75 |
| 8000 | 0.171 | 0.69 | 0.44 | +0.139 | 0.059 | 2.36 | 0.38 | 0.85 |
| 12000 | 0.168 | 0.70 | 0.44 | +0.143 | 0.049 | 2.92 | 0.38 | 0.87 |

Reference: at n = 2000 (Gate 10), the best single split had purity 0.49, capture
0.54, effect +0.096 and SE 0.056, and did not beat the constant.

## Interpretation

1. **The estimated break-even purity under this DGP (≈0.50) is consistent with
   the results, but it is not sufficient on its own.** At n = 4000 purity is
   already 0.75, yet only 62 % of replications improve: the leaf effect is
   estimated with SE 0.095 against an effect of 0.146 (t ≈ 1.5). At n ≥ 8000 the
   SE falls to 0.059/0.049 and the same purity converts into a robust gain.

2. **Two statistical quantities have to clear, not one:** boundary accuracy
   (purity) and leaf-effect precision (SE). Purity improved between 2000 and
   4000 and then plateaued (~0.70); precision kept improving through 12000.
   The MAE gain tracks the latter.

3. **Approximation limitation remains visible.** Best-leaf Jaccard is only
   0.38–0.41 and capture 0.43–0.44, so the model isolates a *pure subset* of the
   region, not the region. The claim is therefore "beats the marginal ATE by
   concentrating on a high-purity subset", not "recovers the true subgroup".

4. **The descriptive depth-3 model is stronger at n ≥ 8000** (Jaccard 0.48–0.49,
   purity 0.93 at 8000/12000). Per the freeze it is not adopted here. It is a
   candidate for a separate confirmatory gate on fresh seeds.

## Scoped claim

Under the specified DGP (piecewise-constant regional effect +0.20 in S(730),
25 % prevalence, 10 % covariate-dependent censoring, correctly specified Cox
nuisance on a linear predictor), an interpretable depth-2 treatment-effect rule
model estimated with held-out AIPW effects achieves lower held-out MAE than the
TRAIN-only marginal ATE at n = 8000 and n = 12000, but not at n = 4000 (62 % of
replications improved, below the pre-registered 70 % consistency bar).

It does not establish: robustness to Cox misspecification; behaviour beyond
n = 12 000; recovery of the full region (capture ≈ 0.44); or anything about
n = 589.

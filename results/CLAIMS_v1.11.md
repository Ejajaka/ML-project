# CISCaRL — consolidated claims after v1.11

**Status:** v1.10 support-constrained selection confirmed on an **independent
two-region DGP** (Gate 23). Method unchanged; only the effect structure differs.

Supersedes `CLAIMS_v1.10.md`.

---

## 1. The model (unchanged v1.10)

| layer | specification |
|---|---|
| Estimand | progression-free probability at day 730 (composite) |
| Estimation | per-arm Cox nuisances, 5-fold cross-fitted; conditional censoring Cox; cap min(Q₀.₉₉, 20); AIPW |
| Discovery | depth-3 CART on ψ, min_samples_leaf 60, discovery partition |
| High-leaf selection | max discovery mean ψ among leaves with ≥200 discovery patients |
| Leaf effects | AIPW on validation |
| Prediction | validation leaf effects; discovery marginal ATE fallback |
| Decision | treat the high leaf only if `τ̂_L > 1.645·SE_L` on validation; else no treatment |

## 2. Independent DGP confirmation (Gate 23)

Two-region truth: `cd40<median ∧ age>median` (+0.20) and
`karnof<median ∧ wtkg<median` (+0.15), zero outside. n = 8000, 40 fresh seeds,
censoring and nuisances unchanged.

| config | captured benefit | purity | J_true | J_full | commit | coverage | efficiency | captured (committed) | null | MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| A: max mean | 0.148 | 0.77 | 0.127 | 0.160 | 45 % | 0.040 | 0.87 | 0.098 | 0 % | 0.0695 |
| **C200 (v1.10)** | **0.364** | **0.82** | **0.295** | **0.275** | **70 %** | **0.129** | 0.84 | **0.284** | **0 %** | 0.0695 |
| C400 (sensitivity) | 0.441 | 0.78 | 0.345 | 0.314 | 75 % | 0.162 | 0.82 | 0.345 | 0 % | 0.0697 |
| D: stability-weighted | 0.210 | 0.67 | 0.179 | 0.156 | 52 % | 0.084 | 0.67 | 0.149 | 0 % | 0.0699 |

**C200 vs A:** captured benefit +0.216 [+0.142, +0.291]; ground-truth recovery
+0.168 [+0.112, +0.224]; reference-leaf recovery +0.115 [+0.078, +0.153];
purity +0.046 (ns); MAE unchanged. All six pre-registered success checks pass.

Prediction MAE beats the constant by Δ = −0.0120 (~14.7 %).

## 3. The single- vs two-region comparison

| | single region (Gate 22) | two regions (Gate 23) |
|---|---|---|
| C200 captured benefit | 0.623 | 0.364 |
| C200 purity | 0.81 | 0.82 |
| C200 J_true | 0.334 | 0.295 |
| C200 reference recovery | 0.318 | 0.275 |
| C200 committed captured | 0.474 | 0.284 |
| A captured benefit | 0.343 | 0.148 |

The **relative** gain over A is larger under two regions (+0.216 vs +0.280? —
absolute difference +0.216 here versus +0.280 in Gate 22), but **absolute**
capture is lower because one leaf cannot cover two separated regions. That is a
representation limit, not a selection failure.

## 4. Claims we can defend

> Under two independent simulation DGPs (single- and two-region treatment
> effects, n = 8000, honest 4000/2000/2000 partition), a depth-3 rule model on
> the AIPW pseudo-outcome with support-constrained selection beats the max-mean
> rule on captured benefit, ground-truth recovery and reference-leaf recovery,
> with no prediction cost and a safe decision layer. In the two-region DGP,
> C200 captures 0.364 of the benefit at 0.82 purity, commits in 70 % of
> replications with 0 % null treatment and 0.84 efficiency, and its leaf is
> closer to the true union than the max-mean leaf (Jaccard 0.295 vs 0.127).

## 5. Claims we cannot make

- ❌ "Recovers multiple regions." One leaf cannot; absolute capture 0.364.
- ❌ "Stable rule list." Recovery frequency 5 % at J > 0.8; mean J_full 0.275.
- ❌ "Robust to censoring misspecification under the two-region DGP." Not run;
  the pre-registered marginal-KM robustness cell is still pending.
- ❌ "Validated beyond two DGPs, n = 8000, one partition ratio."
- ❌ "Support threshold 200 is optimal." 400 is descriptively better; 200 is
  frozen.
- ❌ "CISCaRL outperforms SCRE." Still untested fairly.
- ❌ "95 % confidence intervals." ≈ 93 %.

## 6. Next

1. **Marginal-KM robustness cell under the two-region DGP** (pre-registered
   follow-up to Gate 23): does the C200 gain survive joint outcome + censoring
   error on the independent DGP?
2. **Real-data ACTG175 illustration** under v1.11 (output format only).
3. Paper rewrite against v1.11 numbers.

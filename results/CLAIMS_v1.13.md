# CISCaRL — final consolidated claims (v1.13)

**Status:** final model = v1.12 (C200 single leaf + honest commitment). Gate 25
rejected the two-leaf extension on purity. The simulation program is closed.

Supersedes `CLAIMS_v1.11.md`. The model is unchanged from v1.11; this document
adds the Gate 24 stress result and the Gate 25 negative result, and fixes the
final paper scope.

---

## 1. The model

| layer | specification |
|---|---|
| Estimand | progression-free probability at day 730 (composite) |
| Estimation | per-arm Cox nuisances, 5-fold cross-fitted; conditional censoring Cox; cap min(Q₀.₉₉, 20); AIPW |
| Discovery | depth-3 CART on ψ, min_samples_leaf 60, discovery partition |
| High-leaf selection | max discovery mean ψ among leaves with ≥200 discovery patients |
| Leaf effects | AIPW on validation |
| Prediction | validation leaf effects; discovery marginal ATE fallback |
| Decision | treat the high leaf only if `τ̂_L > 1.645·SE_L` on validation; else no treatment |
| Roles | 4000 discovery / 2000 validation / 2000 assessment, disjoint |

## 2. Evidence across DGPs (n = 8000, 40 fresh seeds each)

| DGP | C200 capture | purity | J_true | commit | captured (committed) | null | MAE gain |
|---|---|---|---|---|---|---|---|
| single region, clean | 0.623 | 0.81 | 0.334 | 80 % | 0.474 | 0 % | ~14 % |
| two regions, clean | 0.414 | 0.86 | 0.337 | 75 % | 0.367 | 0 % | ~15 % |
| two regions, joint stress | 0.421 | 0.84 | 0.340 | 78 % | 0.379 | 0 % | ~15 % |

**C200 vs max-mean** (paired, CIs exclude zero in every DGP):

* single region: capture +0.280; J_true +0.094; reference recovery +0.119;
* two regions: capture +0.216; J_true +0.168; reference recovery +0.115;
* joint stress: capture +0.179; J_true +0.128; reference recovery +0.071;
  purity −0.055 (the one guardrail miss, by 0.5 pp; CI includes zero).

## 3. The rejected two-leaf extension (Gate 25)

| clean two-region | C200 | LIST |
|---|---|---|
| captured benefit | 0.414 | **0.584** |
| ground-truth recovery | 0.337 | **0.419** |
| purity | **0.86** | **0.70** |
| per-treated benefit | +0.168 | +0.158 |
| captured under commitment | 0.367 | **0.489** |
| efficiency | 0.87 | 0.87 |
| null | 0 % | 0 % |

Paired LIST − C200: capture +0.169 [+0.123, +0.218]; J_true +0.083
[+0.041, +0.127]; purity −0.166 [−0.208, −0.125]. The pre-registered purity bar
(C200 − 0.05) fails decisively, so the LIST is **not adopted**, despite the
higher capture and decision value.

## 4. Claims we can defend

> Under two simulation DGPs (single- and two-region treatment effects) at
> n = 8000 with disjoint discovery/validation/assessment partitions, a depth-3
> rule model on the AIPW pseudo-outcome with support-constrained selection
> (≥200 discovery patients) beats max-mean selection on captured benefit
> (+0.18 to +0.28), ground-truth recovery (+0.09 to +0.17) and reference-leaf
> recovery (+0.07 to +0.12), with no prediction cost. The selected leaf captures
> 41–62 % of the true regional benefit at 0.81–0.86 purity. The honest
> commitment layer treats the leaf only when its validation interval excludes
> zero, yielding 75–80 % commitment, 0 % null-benefit treatment, 0.82–0.87
> decision efficiency and ~15 % MAE gain over the constant. The gains survive a
> severe outcome-model violation and a marginal-KM censoring misspecification
> (capture and recovery hold; purity falls to 0.74–0.84, and the pre-registered
> 5 pp purity bar is missed by 0.5 pp under joint error).

## 5. Claims we cannot make

- ❌ "Recovers multiple regions." A two-leaf list captures more but dilutes
  purity to 0.70; the single leaf remains the defensible model.
- ❌ "Stable rule list." Mean reference recovery 0.22–0.32; only 5–7 % of
  half-samples recover the full-data leaf at J > 0.8.
- ❌ "Robust without qualification." The joint stress cell misses the purity
  bar by 0.5 pp; the two-leaf stress cell fails it outright.
- ❌ "The predictive gain is 35 %." It is ~14–15 % in the honest 50/25/25
  partition; 35 % requires a 70/30 discovery split.
- ❌ "Support threshold 200 is optimal." 400 is descriptively better in every
  cell; 200 was frozen.
- ❌ "Validated beyond two DGPs, n = 8000, one partition ratio."
- ❌ "CISCaRL outperforms SCRE." Still untested fairly.
- ❌ "95 % confidence intervals." ≈ 93 %.

## 6. Paper scope (final)

1. The marginal-ATE baseline problem.
2. AIPW estimation, uncertainty and abstention (validated in Gates 1–2).
3. The sample-size investigation: n = 2000 fails; n ≥ 6000 for ψ; crossover
   ≈8000; the plug-in is better at 4000–6000.
4. Support-constrained selection: capture/purity/recovery gains across two DGPs.
5. Honest commitment: 0 % null treatment, 0.82–0.87 efficiency.
6. Robustness: censoring tilt, outcome misspecification, joint stress; state the
   purity boundary.
7. Negative results: RMST, min_support, RF-stability selection, target blending,
   two-leaf list.
8. Scope: two DGPs, n = 8000, one partition ratio, not a universal claim.

## 7. What remains

* ACTG175 illustration under the final model (output format only, not evidence).
* Paper rewrite against v1.13 numbers.
* No further simulation gates unless a specific reviewer question requires one.

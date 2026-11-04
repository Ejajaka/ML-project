# CISCaRL — final claims after v1.14 (v2 stage 1 closed)

**Status:** C200 remains the model. The proposed flexible-learner + distillation
v2 did not beat it in stage 1 and is not pursued. Simulation program closed.

Supersedes `CLAIMS_v1.13.md`.

---

## 1. The model (unchanged v1.13)

| layer | specification |
|---|---|
| Estimand | P(T¹>730|X) − P(T⁰>730|X), composite progression-free endpoint |
| Estimation | per-arm Cox nuisances, 5-fold cross-fitted; conditional censoring Cox; cap min(Q₀.₉₉, 20); AIPW |
| Discovery | depth-3 CART on ψ, min_samples_leaf 60, discovery partition |
| Selection | max discovery mean ψ among leaves with ≥200 discovery patients |
| Leaf effects | AIPW on validation |
| Prediction | validation leaf effects; discovery marginal ATE fallback |
| Decision | TREAT only if `τ̂_L > 1.645·SE_L` on validation; else abstain |
| Roles | 4000 discovery / 2000 validation / 2000 assessment, disjoint |

## 2. Evidence (n = 8000, 40 fresh seeds per cell)

| DGP | C200 capture | purity | J_true | commit | captured (committed) | null | MAE gain |
|---|---|---|---|---|---|---|---|
| single region | 0.548–0.623 | 0.81–0.83 | 0.33–0.50 | 68–80 % | 0.45–0.47 | 0 % | ~15 % |
| two regions | 0.372–0.414 | 0.83–0.86 | 0.31–0.34 | 68–75 % | 0.31–0.37 | 0 % | ~15 % |
| two regions, joint stress | 0.421 | 0.84 | 0.340 | 78 % | 0.379 | 0 % | ~15 % |

## 3. v2 stage 1 — rejected (Gate 26)

Proposed stage 1: cross-fitted flexible DR-learner (`HistGradientBoosting` on ψ)
+ depth-3 distillation (`min_samples_leaf = 200`) + single-rule selection.

| | single: C200 → FLEX | two-region: C200 → FLEX |
|---|---|---|
| captured benefit | 0.548 → 0.454 (−0.094 [−0.178, −0.011]) | 0.372 → 0.299 (−0.073 [−0.129, −0.022]) |
| ground-truth recovery | 0.500 → 0.427 | 0.307 → 0.252 (−0.055) |
| purity | 0.83 → 0.90 | 0.83 → 0.89 |
| reference recovery | 0.302 → 0.318 | 0.262 → 0.276 |
| MAE | 0.0525 → 0.0572 | 0.0690 → 0.0698 |

The flexible surface + distillation produces a **purer but narrower** rule. It
does not improve capture or recovery, so stage 2 (rule stability + constrained
policy) is not built.

**Confound recorded:** the distillation used `min_samples_leaf = 200` while C200
uses 60 + a support filter. FLEX is therefore more constrained; part of its
capture loss may be the distillation minimum. A fair test (`min_samples_leaf =
60`) is a separate, unrun experiment and is **not** claimed here.

## 4. Claims we can defend

> Under two simulation DGPs at n = 8000 with disjoint discovery/validation/
> assessment partitions, a depth-3 CART on the AIPW pseudo-outcome with
> support-constrained selection and honest commitment beats max-mean selection
> on captured benefit, ground-truth recovery and reference-leaf recovery, with
> 0 % null-benefit treatment and ~15 % MAE gain over the constant. It captures
> 37–62 % of the true regional benefit at 0.81–0.86 purity, depending on the
> truth structure. A flexible DR-learner + distillation v2 did **not** improve
> this rule in stage 1.

## 5. Claims we cannot make

- ❌ "The flexible learner improves subgroup discovery." It produced a purer,
  narrower rule, not a better one, under the frozen distillation.
- ❌ "The two-leaf list is better." It captures more but fails the purity bar.
- ❌ "Stable rule list." Mean reference recovery 0.22–0.32.
- ❌ "Recovers multiple regions." Capture 0.37–0.41 under two regions.
- ❌ "Robust without qualification." The joint stress cell misses the purity
  bar by 0.5 pp.
- ❌ "Support threshold 200 is optimal." 400 is descriptively better.
- ❌ "Validated beyond two DGPs, n = 8000, one partition ratio."
- ❌ "CISCaRL outperforms SCRE." Still untested fairly.
- ❌ "95 % confidence intervals." ≈ 93 %.

## 6. Final paper scope

1. Marginal-ATE baseline problem.
2. AIPW estimation, uncertainty, abstention.
3. Sample-size investigation (n = 2000 fails; ψ from ~6000; crossover ~8000).
4. Support-constrained selection (Gates 21–23): capture/purity/recovery across
   two DGPs.
5. Honest commitment (Gates 19–20): 0 % null treatment, 0.82–0.87 efficiency.
6. Robustness: censoring tilt, outcome misspecification, joint stress; state the
   purity boundary.
7. Negative results: RMST, min_support, RF-stability, target blend, two-leaf
   list, flexible-learner v2 stage 1.
8. Scope: two DGPs, n = 8000, one partition ratio; not universal.

## 7. Remaining

* ACTG175 illustration under C200 (output format only).
* Paper rewrite against v1.14 numbers.
* No further simulation gates unless a reviewer raises a specific question; the
  fair distillation test would be a new pre-registration, not a rescue of v2.

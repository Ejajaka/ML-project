# CISCaRL — final claims after v1.15 (SCRE head-to-head added)

**Status:** C200 remains the model. Gate 28 supplies the formal SCRE comparison,
so the previously unanswered question is now answered — against C200 on
patient-level accuracy. Simulation program closed. ACTG175 illustration recorded
with the correct validation lower bound.

Supersedes `CLAIMS_v1.14.md`.

---

## 1. The model (unchanged)

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

## 3. Gate 28 — C200 vs SCRE on identical splits

SCRE fitted on discovery 4000, α selected on validation 2000 by concordance,
predicting the untouched assessment 2000; `t*` overridden to 730 to match the
estimand. 40 fresh seeds per DGP, 100 % of SCRE fits succeeded.

| DGP | method | MAE | capture | purity | per-treated | efficiency | coverage |
|---|---|---|---|---|---|---|---|
| single | C200 | 0.0562 | 0.591 | **0.86** | **+0.168** | 0.88 | 0.175 |
| single | SCRE | **0.0519** | **0.947** | 0.43 | +0.084 | **0.95** | 0.544 |
| two | C200 | 0.0715 | 0.334 | **0.87** | **+0.168** | **0.85** | 0.140 |
| two | SCRE | **0.0685** | **0.802** | 0.58 | +0.098 | 0.80 | 0.560 |

Paired C200 − SCRE: MAE +0.0043 [+0.0002, +0.0084], p = 0.046 (single);
+0.0031 [−0.0007, +0.0068], p = 0.112 (two-region). Capture −0.357 and −0.468,
both p < 0.001. Both methods 0 % null treatment.

**Reading.** SCRE predicts individual effects better and captures far more total
benefit because it treats 54–56 % of patients versus C200's 14–18 %. C200 is
purer (0.86–0.87 vs 0.43–0.58) and extracts roughly twice the per-treated benefit.
This is a precision/coverage trade, not a dominance result.

**Caveat on the comparator.** With validation-concordance α selection SCRE
returns ~1.2 rules, so the fitted object is close to a near-constant Cox model;
its MAE advantage may come from the linear predictor rather than from a usable
rule list. The 16-rule SCRE configuration is a different α and is not this
comparator.

## 4. ACTG175 illustration (arms 0/1, n = 1054, 527/263/264)

| quantity | value |
|---|---|
| overall τ(730) | **+0.143** [+0.025, +0.262] → TREAT |
| largest discovered leaf | 174 patients → support ≥200 **not met** |
| selected rule (fallback) | `cd40 ≤ 343.5 & age > 28.5 & homo ≤ 0.5` |
| discovery support | 62 / 527 |
| validation effect | +0.085, SE 0.189, one-sided LCB **−0.226** |
| decision | **INSUFFICIENT** (no commitment) |
| assessment effect (descriptive) | +0.540 [+0.178, +0.901], n = 30 |
| δ-tilt range | +0.447 to +0.623 |
| SCRE on the same assessment | mean τ̂ 0.041, 76.5 % predicted positive |

The assessment estimate rests on 30 patients and is not a validated subgroup
effect. C200 abstains, so its per-patient prediction collapses to the marginal
ATE and no rank correlation with SCRE is defined.

## 5. Claims we can defend

> Under two simulation DGPs at n = 8000 with disjoint discovery/validation/
> assessment partitions, a depth-3 CART on the AIPW pseudo-outcome with
> support-constrained selection and honest commitment beats max-mean selection
> on captured benefit, ground-truth recovery and reference-leaf recovery, with
> 0 % null-benefit treatment and ~15 % MAE gain over the constant. It captures
> 37–62 % of the true regional benefit at 0.81–0.86 purity. Against SCRE on the
> same splits it is 0.003–0.004 worse on patient-level MAE while being
> substantially purer (0.86 vs 0.43–0.58) and roughly twice as beneficial per
> treated patient.

## 6. Claims we cannot make

- ❌ "CISCaRL outperforms SCRE on HTE accuracy." Tested (Gate 28) and false:
  SCRE has lower MAE in both DGPs, significantly so in the single-region cell.
- ❌ "CISCaRL captures more benefit than SCRE." It captures 0.33–0.59 vs
  0.80–0.95 and treats a third as many patients.
- ❌ "The flexible learner improves subgroup discovery." Purer, narrower, not
  better, under the frozen distillation.
- ❌ "The two-leaf list is better." More capture, purity fails.
- ❌ "Stable rule list." Mean reference recovery 0.22–0.32.
- ❌ "Recovers multiple regions." Capture 0.37–0.41 under two regions.
- ❌ "Robust without qualification." The joint stress cell misses the purity bar
  by 0.5 pp.
- ❌ "Support threshold 200 is optimal." 400 is descriptively better.
- ❌ "Validated beyond two DGPs, n = 8000, one partition ratio."
- ❌ "Validated on ACTG175." C200 abstains there; the rule is an illustration.
- ❌ "95 % confidence intervals." ≈ 93 %.

## 7. Final paper scope

1. Marginal-ATE baseline problem.
2. AIPW estimation, uncertainty, abstention.
3. Sample-size investigation (n = 2000 fails; ψ from ~6000; crossover ~8000).
4. Support-constrained selection (Gates 21–23).
5. Honest commitment (Gates 19–20): 0 % null treatment, 0.82–0.87 efficiency.
6. Robustness: censoring tilt, outcome misspecification, joint stress; state the
   purity boundary.
7. Head-to-head with SCRE (Gate 28): accuracy loss, purity and per-treated gain.
8. Negative results: RMST, min_support, RF-stability, target blend, two-leaf
   list, flexible-learner v2 stage 1.
9. Scope: two DGPs, n = 8000, one partition ratio; not universal.

## 8. Remaining

* Nothing required for the notebook: `notebooks/CISCaRL_final.ipynb` executes
  end-to-end with 0 error cells and reproduces every number above.
* Paper: `paper/MANUSCRIPT.md` now carries the Gate 28 section.

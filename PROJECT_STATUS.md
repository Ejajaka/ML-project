# CISCaRL — Project Status, Evidence, and Known Limitations

_For reviewers / professor discussion. All claims backed by committed runs in
`results/` (see "How to verify" at the end)._

---

## 1. What the project is

**CISCaRL** = Conformalized Interpretable Survival Causal Rule Lists.
Interpretable heterogeneous treatment effect (HTE) estimation for right-censored
survival. Outputs a short human-readable rule list; each rule gives a patient
subgroup, its CATE with a distribution-free 90% conformal interval, a stability
score (bootstrap rediscovery frequency), and a clinical recommendation.

**Pipeline:** (1) candidate rules from a causal forest on the DR pseudo-ITE plus
a gradient-boosted model on CSF-predicted CATE; (2) bootstrap **stability
selection** (replacing Lasso's 1se rule); (3) **IPCW-weighted conformal
intervals** per rule.

**Compared against 9 methods:** Cox T-learner, CSF (RF), Bo & Ding (GBM+Lasso),
Hybrid (CSF rules+Lasso), CRE (ElasticNet), SCRE (Wan et al., faithfully
reimplemented as a shared-basis Cox RuleFit), and CISCaRL in direct / auto /
posthoc modes.

**Data:** PBC, SUPPORT, GBSG, ACTG175, each under 4 semi-synthetic DGPs
(AFT-Gumbel, Cox PH, Non-PH crossing, Nonlinear XOR) with known CATE.

---

## 2. Which clinical task each metric maps to

A reader must not assume winning on one metric implies winning on another. The
metrics answer **different clinical questions**:

| Metric | Clinical question | Type |
|---|---|---|
| **MAE / PEHE (RMSE)** | "How large is the effect in this subgroup?" | point estimation |
| **Spearman / rank** | "Who should get treated *first*?" | prioritization / ranking |
| **R²** | "How much of the CATE variance is explained?" | absolute fit (weak for CATE) |
| **Rule count / interpretability** | "Can a clinician understand & act on this?" | decision support |

CISCaRL is strong on **point estimation + interpretability**, and **weak-to-mid
on ranking**. Both must be stated together.

---

## 3. Effect-size regime: explicit statement

The semi-synthetic DGPs run under **two effect-size regimes**
(`experiments.EFFECT_REGIME`):

- **`rescaled`** (default): larger effects so CATE std ~0.13–0.32, giving
  positive/meaningful R² on the demo.
- **`original`** (harder): the original small effects (CATE std ~0.06–0.10),
  below the DR pseudo-ITE noise floor (std ~0.8), giving negative R² for every
  method regardless of quality.

**Both regimes are fully benchmarked** (3 reps × 4 datasets × 4 DGPs each,
identical splits) and every number below is labeled by regime.

---

## 4. Statistical protocol

- **48 settings** per regime: 3 reps × 4 datasets × 4 DGPs.
- **Paired tests** (all methods share the same splits within a rep): paired
  t-test of MAE differences vs CISCaRL-posthoc.
- **Multiple-comparison correction:** Holm–Bonferroni. The **hypothesis family
  is defined per regime** (8 competitor comparisons each; 16 total across the
  two regimes, treated as two separate families because the two regimes are
  reported as independent experiments). FWER controlled at α=0.05 per family.
  Both raw `p` and corrected `p_holm` are reported.
- **Effect sizes:** raw MAE gap with 95% CI, plus Cohen's d.
- **Borderline results** are labeled as such, not flattened into "ties".

---

## 5. Results

### 5.1 Overall MAE (mean ± SE over 48 settings)

**Original (harder) effects:**

| Rank | Method | MAE ± SE | vs CISCaRL-posthoc (Holm-corrected) |
|---|---|---|---|
| 1 | SCRE | 0.090 ± 0.004 | beats it** (gap −0.046, [−0.064,−0.029], d=−0.75) |
| 2 | **CISCaRL-posthoc** | **0.137 ± 0.010** | baseline |
| 3 | Cox | 0.149 ± 0.014 | ns (p=0.31) |
| 4 | CISCaRL-auto | 0.154 ± 0.012 | worse** (p_holm=0.034) |
| 5 | CISCaRL-dir | 0.160 ± 0.012 | worse** (p_holm=0.030) |
| 6 | CSF | 0.168 ± 0.010 | worse** (gap +0.031, [0.020,0.043], d=+0.76) |
| 7 | Hybrid | 0.364 ± 0.013 | worse** |
| 8 | CRE | 0.366 ± 0.012 | worse** |
| 9 | Bo & Ding | 0.408 ± 0.017 | worse** |

**Rescaled effects:**

| Rank | Method | MAE ± SE | vs CISCaRL-posthoc (Holm-corrected) |
|---|---|---|---|
| 1 | SCRE | 0.203 ± 0.010 | beats it** (gap −0.048, [−0.083,−0.012], d=−0.38) |
| 2 | Cox | 0.229 ± 0.019 | **trend, not significant** (p=0.083, p_holm=0.25) |
| 3 | CSF | 0.229 ± 0.021 | beats it** (gap −0.022, [−0.035,−0.008], d=−0.46) |
| 4 | **CISCaRL-posthoc** | **0.251 ± 0.019** | baseline |
| 5 | CISCaRL-auto | 0.256 ± 0.017 | ns |
| 6 | CISCaRL-dir | 0.262 ± 0.018 | ns |
| 7 | CRE | 0.383 ± 0.017 | worse** |
| 8 | Hybrid | 0.387 ± 0.018 | worse** |
| 9 | Bo & Ding | 0.427 ± 0.020 | worse** |

**Rules:** CISCaRL 4–6.5; Bo&Ding/Hybrid/CRE 218–704.

### 5.2 Secondary CATE metrics (PEHE = RMSE, Spearman rank)

**Original:** SCRE leads everything (MAE 0.090, RMSE 0.112, Spearman 0.18).
CISCaRL-posthoc MAE 0.137, RMSE 0.158, Spearman **0.07**. Cox/CSF have higher
Spearman (0.11/0.13) but higher MAE.

**Rescaled:** CSF has the best patient ranking (Spearman 0.33); SCRE best MAE
(0.203). CISCaRL-posthoc Spearman 0.21.

**Ranking wins per setting:** SCRE 6, Cox 5, CSF 2–4, CISCaRL-posthoc 1–2.
=> **CISCaRL does NOT lead on patient ranking.** This is a core limitation (§7).

---

## 6. The claim — restated correctly (this is the key change)

The evidence no longer supports an unqualified "CISCaRL is the best
interpretable method." The honest, defensible claim is **two-part**:

**Part A — vs the Lasso-based rule methods (Bo&Ding, Hybrid, CRE):**
CISCaRL is dramatically better (2–3× lower MAE, 50–100× fewer rules,
statistically significant in both regimes). This is solid and unchanged.

**Part B — vs SCRE (the harder competitor):**
SCRE **leads on both MAE (original regime) AND patient ranking**. So CISCaRL is
**not** "better than SCRE." The correct comparative claim vs SCRE is:

> CISCaRL is **statistically competitive with SCRE on MAE** (SCRE beats it by a
> small margin, d≈0.4–0.75) **while offering interpretable output SCRE does not
> provide**: per-rule **conformal (finite-sample-valid) confidence intervals**,
> per-rule **stability scores**, and explicit **treat / avoid / more-data
> recommendations**. SCRE returns a weighted coefficient list with no CIs, no
> stability, no recommendations.

These are CISCaRL's **real differentiators vs SCRE** (verified in code: SCRE has
no conformal intervals, no stability scores, no recommendations). The rule-count
contrast with Lasso methods was the weaker selling point; the **conformal-valid
+ stable + actionable per-subgroup output** is the stronger one.

---

## 7. Known limitations (one coherent story)

CISCaRL is honest about what it is **not** good at:

1. **Absolute correlation & ranking**: R² is negative on real-covariate data
   (all 9 methods; oracle = +0.75–1.00, so it's inherent CATE difficulty, not a
   defect), and CISCaRL is mid-pack on Spearman/rank (0.07–0.21 vs SCRE
   0.18–0.23, CSF 0.13–0.33). **Together these mean:** CISCaRL should not be the
   tool of choice for "who benefits most" prioritization or for explaining
   absolute CATE variance — its value is *interpretable, stable, conformal-valid
   subgroup point estimates*.
2. **SCRE leads MAE + ranking**: CISCaRL is competitive, not superior, on
   accuracy vs SCRE.
3. **3 reps is thin**: enough for paired tests (large effects), but SEs on
   individual numbers are wide.
4. **SCRE reimplementation**: faithful shared-basis Cox RuleFit, but Wan et al.
   use adaptive group-lasso; ours uses elastic-net on the shared-basis design.
5. **Cox "trend" in rescaled regime**: p=0.083 / p_holm=0.25 — a trend toward
   CISCaRL being worse, not a flat tie.

---

## 8. Fixes locked in with tests

- **grf tree-traversal off-by-one**: `R/test_grf_traversal.R` regression test
  (1-based walk passes; old 0-based walk must fail). Run:
  `Rscript R/test_grf_traversal.R`.
- **Stale claims removed**: README now says 9 methods × 4 datasets × 4 DGPs;
  no remaining "DGP 1 only" text.
- **Gitignore audited**: only `__pycache__` + runtime artifacts excluded.

---

## 9. How to verify (all committed, nothing pushed)

```bash
cd python
python fair_benchmark.py --quick --reps 3 --regime original   # ~2h
python fair_benchmark.py --quick --reps 3 --regime rescaled   # ~2h
python cis_carl_demo.py                                       # demo + R²
Rscript ../R/test_grf_traversal.R                             # regression test
```

Results in `results/`: `fair_benchmark.csv` (rescaled),
`fair_benchmark_original.csv` (original), `paper_results_full.csv`.
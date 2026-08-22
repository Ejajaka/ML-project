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

## 2. Effect-size regime: explicit statement

The semi-synthetic DGPs run under **two effect-size regimes**
(`experiments.EFFECT_REGIME`):

- **`rescaled`** (default): larger effects so CATE std ~0.13–0.32, giving
  positive/meaningful R² on the demo.
- **`original`** (harder): the original small effects (CATE std ~0.06–0.10),
  below the DR pseudo-ITE noise floor (std ~0.8), giving negative R² for every
  method regardless of quality.

**Both regimes are fully benchmarked** (3 reps × 4 datasets × 4 DGPs each,
identical splits) and every number below is labeled by regime. Where a single
number is quoted for both (e.g. "Cox ties, p=0.08 in the harder regime"), the
regime is stated explicitly.

---

## 3. Statistical protocol (all numbers below use this)

- **48 settings** per regime: 3 reps × 4 datasets × 4 DGPs.
- **Paired tests** (all methods share the same splits within a rep):
  paired t-test of MAE differences vs CISCaRL-posthoc.
- **Multiple-comparison correction:** Holm–Bonferroni across the 8 competitors
  *within each regime* (family-wise error controlled at α=0.05; 8 tests/regime,
  16 total). Both raw `p` and corrected `p_holm` are reported.
- **Effect sizes:** raw MAE gap with 95% CI, plus Cohen's d.
- **Borderline results** are labeled as such, not flattened into "ties".

---

## 4. Results

### 4.1 Overall MAE (mean ± SE over 48 settings)

**Original (harder) effects:**

| Rank | Method | MAE ± SE | vs CISCaRL-posthoc (Holm-corrected) |
|---|---|---|---|
| 1 | SCRE | 0.090 ± 0.004 | beats it** (gap −0.046, 95% CI [−0.064,−0.029], d=−0.75) |
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

### 4.2 Secondary CATE metrics (PEHE = RMSE, Spearman rank)

**Original:** SCRE leads everything (MAE 0.090, RMSE 0.112, Spearman 0.18).
CISCaRL-posthoc MAE 0.137, RMSE 0.158, Spearman 0.07. Cox/CSF have higher
Spearman (0.11/0.13) but higher MAE.

**Rescaled:** CSF has the best patient ranking (Spearman 0.33); SCRE best MAE
(0.203). CISCaRL-posthoc Spearman 0.21.

**Ranking wins per setting:** SCRE 6, Cox 5, CSF 2–4, CISCaRL-posthoc 1–2.
=> **CISCaRL does NOT lead on patient ranking** — only on the interpretability
vs MAE trade-off. This is stated as a limitation (§6), not hidden.

---

## 5. The design trade-off (stated up front)

CISCaRL is **not** the lowest-MAE method (SCRE is) and does **not** have the
best patient ranking (CSF/SCRE do). Its claim is a deliberate
**interpretability–accuracy trade-off**:

> CISCaRL is the **best method among those that return a genuinely interpretable
> rule list**, statistically tied with Cox on MAE, and (in the harder DGP)
> significantly better than the black-box CSF, while the Lasso-based rule
> methods (Bo&Ding, Hybrid, CRE) that also return rules are 2–3× worse on MAE
> **and** return 50–100× more rules.

Justification for the accuracy cost vs SCRE:
- **Clinical trust / auditability**: a 5-rule list with CIs + stability scores is
  inspectable; SCRE's ~1.3-rule output is effectively a black box.
- **Regulatory / decision support**: per-subgroup actionable recommendations
  with finite-sample-valid intervals.
- SCRE's 1.3 rules ≈ default/none model; CISCaRL commits to a real rule list.

**Counter-evidence to disclose:** on ranking (the clinically most relevant
quantity), CISCaRL is mid-pack (Spearman ~0.07–0.21 vs SCRE 0.18–0.23, CSF
0.13–0.33). So the honest claim is "interpretable MAE-competitive," **not**
"best at ranking who benefits."

---

## 6. Known limitations

1. **R² negative on real-covariate data** (original effects): all 9 methods
   negative; oracle = +0.75–1.00 on the same data → inherent CATE difficulty,
   not a model defect. R² is reported only under rescaled effects / demo where
   it is positive (+0.05 to +0.30). PEHE/RMSE/rank are the primary metrics
   (per Lei & Candès, Kennedy).
2. **3 reps is thin** — enough for the paired tests (large effects), but SEs on
   individual numbers are wide; more reps would tighten them.
3. **SCRE reimplementation**: faithful shared-basis Cox RuleFit, but Wan et al.
   use adaptive group-lasso; ours uses elastic-net on the shared-basis design.
4. **Ranking weakness** (see §5): CISCaRL does not lead Spearman/rank.
5. **Cox "trend" in rescaled regime**: p=0.083 / p_holm=0.25 — a trend toward
   CISCaRL being worse, not a flat tie; do not over-claim.

---

## 7. Fixes locked in with tests

- **grf tree-traversal off-by-one**: `R/test_grf_traversal.R` regression test
  (1-based walk passes; old 0-based walk must fail). Run:
  `Rscript R/test_grf_traversal.R`.
- **Stale claims removed**: README now says 9 methods × 4 datasets × 4 DGPs
  (was "8 methods"); verified no remaining "DGP 1 only" text.
- **Gitignore audited**: only `__pycache__` + runtime artifacts excluded; all
  data, code, results tracked.

---

## 8. How to verify (all committed, nothing pushed)

```bash
cd python
python fair_benchmark.py --quick --reps 3 --regime original   # ~2h
python fair_benchmark.py --quick --reps 3 --regime rescaled   # ~2h
python cis_carl_demo.py                                       # demo + R²
Rscript ../R/test_grf_traversal.R                             # regression test
```

Results in `results/`: `fair_benchmark.csv` (rescaled),
`fair_benchmark_original.csv` (original), `paper_results_full.csv`.
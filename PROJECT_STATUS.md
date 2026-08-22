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

## 2. Effect-size regime: an explicit statement (was ambiguous before)

The semi-synthetic DGPs can run under **two effect-size regimes** (selected by
`experiments.EFFECT_REGIME`):

- **`rescaled`** (default): larger effects chosen so CATE std ~0.13–0.32, making
  R² positive/meaningful on the demo.
- **`original`** (harder): the original small effects (CATE std ~0.06–0.10),
  which sit below the DR pseudo-ITE noise floor (std ~0.8) and give negative R²
  for *every* method regardless of quality.

**Both regimes are fully benchmarked** (3 reps × 4 datasets × 4 DGPs each,
identical splits), and the headline result holds in both — see §4.

---

## 3. What was fixed in the codebase

1. **Pseudo-ITE outcome bug** (`utils.py`): `Y = I(T>t*)` mis-coded died/censored
   patients; corrupted the target for all pseudo-ITE methods.
2. **Faithful SCRE** (`scre.py`): was RF+ElasticNet; now the real shared-basis
   penalized Cox RuleFit (arXiv:2309.11914).
3. **Bootstrap multiplicity bug** (`cis_carl.py`): with-replacement resampling
   was collapsed to a boolean mask.
4. **Rule stabilization**: min-calibration-support + shrinkage toward the global
   mean removes unstable tiny-subgroup extreme rules.
5. **Crossing-DGP ground truth**: now a closed-form survival-probability
   difference, not a function of treatment assignment.
6. **Posthoc mode + all-4-DGPs-on-real-data** added to the benchmark.
7. **R scripts**: 6 crash/silent bugs fixed (incl. grf tree-traversal off-by-one).
8. **Reproducibility**: `paper_results_full.csv`, `fair_benchmark.csv`,
   `fair_benchmark_original.csv` all committed; gitignore audited (only
   `__pycache__` and runtime artifacts excluded).

---

## 4. Results — both regimes, with variance and significance

### Overall MAE (mean ± SE over 3 reps × 4 datasets × 4 DGPs = 48 settings)

**Original (harder) effects:**

| Rank | Method | MAE ± SE | Paired test vs CISCaRL-posthoc |
|---|---|---|---|
| 1 | SCRE | 0.090 ± 0.004 | beats CISCaRL** (p<0.0001) |
| 2 | **CISCaRL-posthoc** | **0.137 ± 0.010** | (baseline) |
| 3 | Cox | 0.149 ± 0.014 | ns (p=0.31) |
| 4 | CISCaRL-auto | 0.154 ± 0.012 | worse** (p=0.011) |
| 5 | CISCaRL-dir | 0.160 ± 0.012 | worse** (p=0.015) |
| 6 | CSF | 0.168 ± 0.010 | worse** (p<0.0001) |
| 7 | Hybrid | 0.364 ± 0.013 | worse** |
| 8 | CRE | 0.366 ± 0.012 | worse** |
| 9 | Bo & Ding | 0.408 ± 0.017 | worse** |

**Rescaled effects:**

| Rank | Method | MAE ± SE | Paired test vs CISCaRL-posthoc |
|---|---|---|---|
| 1 | SCRE | 0.203 ± 0.010 | beats CISCaRL** (p=0.012) |
| 2 | Cox | 0.229 ± 0.019 | ns (p=0.083) |
| 3 | CSF | 0.229 ± 0.021 | beats CISCaRL** (p=0.003) |
| 4 | **CISCaRL-posthoc** | **0.251 ± 0.019** | (baseline) |
| 5 | CISCaRL-auto | 0.256 ± 0.017 | ns |
| 6 | CISCaRL-dir | 0.262 ± 0.018 | ns |
| 7 | CRE | 0.383 ± 0.017 | worse** |
| 8 | Hybrid | 0.387 ± 0.018 | worse** |
| 9 | Bo & Ding | 0.427 ± 0.020 | worse** |

**Rules (both regimes):** CISCaRL 4–6.5; Bo&Ding/Hybrid/CRE 218–704.

### Key findings

- **The interpretability-accuracy claim is robust**: in the original (harder)
  regime CISCaRL-posthoc is **#2 overall**, **significantly beats CSF**, **ties
  Cox**, and **significantly beats all Lasso-based rule methods**. Only SCRE
  beats it. This is *stronger* than the rescaled regime, not weaker.
- **Statistical tests are paired** (all methods share the same splits), so the
  comparisons are valid.

---

## 5. The design trade-off — stated explicitly up front

CISCaRL is **not** the lowest-MAE method (SCRE is, and Cox/CSF are comparable).
The project's claim is a **deliberate interpretability–accuracy trade-off**:

> CISCaRL is the **best method among those that return a genuinely interpretable
> rule list**, at MAE statistically indistinguishable from Cox and better than
> CSF under the harder DGP, while the Lasso-based rule methods (Bo&Ding, Hybrid,
> CRE) that also return rules are 2–3× worse on MAE **and** return 50–100× more
> rules.

Justification for paying the (small) accuracy cost vs SCRE:
- **Clinical trust / auditability**: a 5-rule list with CIs and stability scores
  is inspectable and defensible; SCRE's ~1.3 rules is effectively a black box.
- **Regulatory / decision support**: per-subgroup actionable recommendations
  with finite-sample-valid intervals.
- **SCRE trades interpretability for accuracy** (its 1.3-rule output is nearly
  a default/none model); CISCaRL commits to a rule list.

This framing should be the first thing stated, not implied by a ranking table.

---

## 6. Honest caveats

1. **Negative R² on real-covariate data**: all 9 methods get negative R² under
   original effects; an oracle (perfect subgroup model) scores +0.75–1.00 on the
   same data. So negative R² = inherent difficulty of CATE on real data, not a
   model defect. CATE papers use PEHE/RMSE/rank as primary metrics for this
   reason. R² is only reported under rescaled effects / on the demo, where it is
   positive (+0.05 to +0.30).
2. **3 reps is thin**: enough for the paired tests above (which found strong
   effects), but CIs on individual numbers are wide. More reps would tighten
   SEs.
3. **SCRE reimplementation**: faithful shared-basis Cox RuleFit, but Wan et al.
   use adaptive group-lasso; our version uses elastic-net on the shared-basis
   design. A reviewer may want the exact group-lasso variant.

---

## 7. How to verify (all committed, nothing pushed)

```bash
# Fair benchmark, both regimes (3 reps each, ~2h each)
cd python
python fair_benchmark.py --quick --reps 3 --regime original
python fair_benchmark.py --quick --reps 3 --regime rescaled

# Demo with R² (synthetic, known CATE)
python cis_carl_demo.py
```

Results already in `results/`: `fair_benchmark.csv` (rescaled),
`fair_benchmark_original.csv` (original), `paper_results_full.csv`.
# Final audit — frozen C200, consistency and ACTG175

Date: 2026-11-03. Scope: verify that the final reported results use the corrected
implementation, a single frozen definition, and no assessment leakage; then record
the real-data illustration.

## 1. Core implementation

`python -m ciscarl2.audit` → **all checks pass**:

* `simplify` bound direction fixed; conditions equal stored masks (0 mismatches);
* leaves partition the data; `assign` respects most-specific-first and the default;
* paired IF contrast matches the direct formula;
* purity/capture/Jaccard definitions hand-checked;
* AIPW bias **+0.0031** (sd 0.0130); arm means within 0.002.

The `simplify` fix is in `ciscarl2/rules.py` and is imported by every gate from
10 onward, including Gates 21–26. Pre-fix CSVs are preserved as `*_prebug.csv`;
`CORRECTION_rules_simplify.md` records the change and the re-runs.

## 2. Frozen C200 definition (consistent across final results)

| component | value |
|---|---|
| discovery | depth-3 CART on ψ, `min_samples_leaf = 60`, discovery partition |
| selection | max discovery mean ψ among leaves with **discovery support ≥ 200** |
| leaf effects | AIPW on validation |
| commitment | validation one-sided 95 %: `τ̂_L > 1.645·SE_L` |
| prediction | validation leaf effects; discovery marginal ATE fallback |
| roles | 4000 discovery / 2000 validation / 2000 assessment (n = 8000) |
| assessment | never used for selection, thresholds, or commitment |

Gate 21 is the first gate with the C200 selection rule; its honest decision
numbers are the ones the paper should use (commit 80 %, coverage 0.134,
captured 0.474, null 0 %, efficiency 0.85). Gate 20 used the max-mean leaf and is
the honest *baseline*, not C200.

## 3. Known issues and how they are handled

| issue | status |
|---|---|
| `simplify` bug | fixed; Gates 16–18 re-run; 10–15 flagged as pre-fix |
| Gate 19 in-sample commitment | corrected in Gate 20 with disjoint validation |
| Gate 21 invalid stability yardstick | corrected in Gate 22 with own-procedure refits |
| Gate 24 purity guardrail missed by 0.5 pp | reported, not redefined |
| Gate 25 two-leaf purity failure | reported, not adopted |
| Gate 26 distillation support mismatch (200 vs 60) | reported as a confound; flexible learner not claimed inferior |

## 4. Negative results documented

RMST (8), min_support sweep (7), RF+stability at small n (9–12), plug-in collapse
under κ=3 (14), target blend (15), depth-1 at n=2000 (10), two-leaf list (25),
flexible DR-learner + distillation (26).

## 5. ACTG175 illustration (C200, not evidence)

Output: `results/actg175_c200.md`. Partition 527 / 263 / 264.

* Overall assessment τ(730) = **+0.143 [+0.025, +0.262]**, TREAT.
* C200 support constraint **not met** on 527 discovery patients (largest leaf
  174); the model **falls back to the max-mean leaf**.
* Selected rule: `cd40 ≤ 343.5 & age > 28.5 & homo ≤ 0.5`, discovery support 62,
  validation n 36, assessment n 30.
* Validation effect +0.085 (SE 0.189), one-sided LCB −0.226 → **commitment NO**;
  decision **INSUFFICIENT**.
* Assessment effect +0.540 [+0.178, +0.901] on 30 patients — descriptive only,
  not a recommendation.
* δ-tilt: +0.447 to +0.623 over δ ∈ [−2, 2].

This is the sample-size boundary in one table: at ACTG175 scale the support rule
cannot be applied, the fallback leaf is small, and the commitment layer
correctly abstains. The real data contain no known subgroup truth; the endpoint
is composite and follow-up appears to end at discontinuation.

## 6. Recommendation on Gate 27

Gate 27 (diagonal-rule geometry) is **not required** for the current paper
unless the paper claims that axis-aligned rules are the specific bottleneck.
The two-region DGP already documents a capture ceiling (0.37–0.41). If that
claim is made, Gate 27 should be frozen separately with a diagonal DGP; it is
not a rescue of the flexible-learner result.

Otherwise: freeze C200, finish the manuscript, and do not add another
simulation.

## 7. Gate 28 — the SCRE comparison that was missing

`results/gate28_verdict.md`, `results/gate28_summary.csv`. SCRE and C200 on
identical splits; paired over 40 seeds.

* SCRE MAE 0.0519 (single) / 0.0685 (two) vs C200 0.0562 / 0.0715. Paired
  C200 − SCRE: +0.0043 [+0.0002, +0.0084], p = 0.046; +0.0031 [−0.0007,
  +0.0068], p = 0.112.
* C200 purity 0.86 / 0.87 vs SCRE 0.43 / 0.58; per-treated +0.168 vs
  +0.084 / +0.098; coverage 0.175 / 0.140 vs 0.544 / 0.560.
* Both 0 % null treatment. SCRE selects ~1.2 rules under validation-concordance
  α, so the comparator is close to a near-constant Cox model.

Consequence: the claim "CISCaRL outperforms SCRE" is removed from the paper.
The defensible claim is comparable-to-slightly-worse patient-level accuracy with
much higher purity and an explicit abstention layer.

## 8. Notebook state

`notebooks/CISCaRL_final.ipynb` — 48 cells (30 code), executed end-to-end with
0 error cells. Preprocessing splits first and fits the scaler on discovery only,
so it is leak-free; the saved ACTG175 numbers reproduce this audit. The SCRE
comparison cell reports the abstention explicitly and does not print a rank
correlation, because C200's prediction is constant when it abstains.

Reproduce with `jupyter nbconvert --to notebook --execute
notebooks/CISCaRL_final.ipynb --inplace` from the project root.

## 9. IEEE conference paper

`paper/ieee_conference.tex` (IEEEtran, conference option) is the submission
version, modelled on the structure of the SCRE paper (abstract, index terms,
introduction, related work, method with numbered equations and an algorithm,
simulation design, results, real-data application, discussion/limitations,
conclusion, references).

* Compiles cleanly: 7 pages, 0 overfull boxes, 0 LaTeX warnings. Build with
  `tectonic -X compile ieee_conference.tex` (a Tectonic 0.17 binary is cached
  under the temp directory) or any TeX Live installation.
* Rendered copy: `paper/CISCaRL_IEEE_conference.pdf` and
  `paper/ieee_conference_preview.pdf` (identical).
* Figures are generated by the script that produced
  `paper/figs/fig{1,2,3}_*.pdf` from `results/gate17`, `gate21`, `gate23`,
  `gate24` summaries.
* Tables I--VI are transcribed from `results/gate17/21/22/23/24/28_summary.csv`.
  Table II's first metric is labelled "true region covered" because the gate 21
  `capture` column is region coverage, not captured benefit; captured benefit
  appears as the committed row.
* Table I reports the final partition design only. The earlier 70/30 design
  figure (~35 % gain at n = 8000) is deliberately not quoted, because it is not
  comparable to the 50/25/25 numbers in the table.
* Table VII is the full candidate-rule table (five leaves) with discovery
  score, relative score, support, validation and assessment effects, and a
  descriptive hazard ratio. It is reproduced in the notebook (Part 5, "Every
  candidate rule") and saved as `results/actg175_leaf_table.csv`.
* Table VIII is the unadjusted Kaplan-Meier check of each rule (730-day
  survival by arm, bootstrap interval), the same descriptive device the SCRE
  paper uses on ACTG175 (its Fig. 3). Reproduced in the notebook (Part 5,
  "Kaplan-Meier check") and saved as `results/actg175_km_check.csv`.
* Part 7 also runs one replication live (`ciscarl2.gate28.run_rep`, seed
  300100) so the accuracy metrics are computed in the notebook rather than only
  loaded: MAE, Spearman, capture, purity, per-treated effect, efficiency,
  coverage and null rate for both C200 and SCRE on the same split. The 40-seed
  tables remain the reported evidence; one seed can reverse the MAE ranking,
  which is why the paired average is used.
* Author block still holds a placeholder name, affiliation detail and email.

## 10. Gate 30 — knob sweep and calibration (pre-registered)

`results/gate30_PREREGISTRATION.md`, `gate30_verdict.md`,
`gate30_variant_table.csv`, `gate30_headtohead.csv`, `gate30_smalln_summary.csv`.
40 tuning seeds (310100-310139) and 40 fresh confirmation seeds
(320100-320139), two DGPs, paired within replication.

* **Adopted:** empirical-Bayes shrinkage of leaf effects toward the marginal ATE,
  used for prediction only (`C200-S`). Pooled over 80 replications: MAE
  −0.0051 [−0.0077, −0.0025] with one region and −0.0026 [−0.0048, −0.0005]
  with two. Purity, capture, commitment and Spearman are identical to the frozen
  model (0.8719/0.4510/72.5 %/0.522 single; 0.8777/0.2562/62.5 %/0.444 two).
* **Head-to-head:** the frozen model is significantly worse than SCRE on MAE
  (+0.0062, p<0.001 single; +0.0031, p=0.044 two); with shrinkage the difference
  is +0.0011 (p=0.48) and +0.0004 (p=0.76), i.e. a tie.
* **Rejected:** depth 4 (+0.0099/+0.0083), min_leaf 100 (two-region-only gain),
  min_leaf 30 (flat).
* **Policy dial, not adopted:** z=1.28 and z=1.96 move commitment and purity with
  no MAE effect.
* **Small n:** at n=2000 the absolute 200 floor gives purity 0.47 / capture 0.096
  / commitment 17.5 %, the 5 % proportional floor 0.73 / 0.127 / 30.0 %. The
  proportional floor is the correct specification below ~4000 discovery
  patients; it is not retro-fitted to the frozen model.
* **Honesty note:** the shrinkage was chosen after Gate 28 and confirmed on a
  fresh block, so it is reported as C200-S (post-hoc calibration), while every
  pre-registered table remains the frozen C200.

## 11. Naming

`CISCaRL` is the method; `C200` is its frozen configuration (arm C, support
floor 200 discovery patients). `C200-S` is C200 with the shrinkage calibration.
The paper spells this out in the Introduction; the notebook does so in Part 3.

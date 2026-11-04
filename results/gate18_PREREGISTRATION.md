# Gate 18 pre-registration — joint nuisance misspecification, full model profile

Frozen 2026-11-02, after a 10-replication verification pass (recorded below) and
before the confirmatory run. Not edited after the run.

## Question

Can the v1.6 model (ψ discovery + depth-3 CART) remain *accurate as a model* when
both the outcome and censoring nuisance models are wrong?

MAE against the constant is necessary but not sufficient. The gate therefore
evaluates six dimensions: estimation, discovery, rule quality, stability,
robustness, and decision quality.

## Design — 2 × 2 nuisance factorial, n = 8000, 40 fresh seeds

| cell | outcome model | censoring model |
|---|---|---|
| A (reference) | correct | conditional Cox (correct) |
| B | correct | **marginal Kaplan–Meier** |
| C | **κ=3 violation** (Gate 14) | conditional Cox (correct) |
| D | **κ=3 violation** | **marginal Kaplan–Meier** |

* Seeds 200100–200139 for both κ blocks; the two censoring variants are applied
  to the **same simulated data** within a replication, so B vs A and D vs C are
  paired.
* The data remain MAR given X in every cell; the censoring misspecification is
  a *model* error (ignoring X), not a new MNAR data-generating process.
* Everything else frozen from Gate 16: depth-3 CART, min_samples_leaf 60,
  held-out AIPW effects, TRAIN marginal ATE, cap min(Q₀.₉₉, 20).

## Metrics (all fixed before the run)

**Estimation** — ATE estimate vs truth; bias.
**Discovery** — best-leaf Jaccard, purity, capture.
**Rule quality** — high-leaf AIPW effect vs the leaf's true mean τ; absolute
error; fraction of replications in which the high leaf is genuinely beneficial
(true mean τ > 0).
**Stability** — half-sample split stability (20 refits).
**Decision** — coverage; per-treated mean true τ; oracle benefit at the same
coverage (top-k true τ); efficiency = per-treated / oracle; fraction of total
available benefit captured; sensitivity/specificity for τ > 0.
**Predictive** — paired Δ MAE vs the TRAIN constant.

## Verification pass (10 reps, before freezing)

| κ | censoring | ATE bias | MAE gain | purity | capture | hi true effect | efficiency |
|---|---|---|---|---|---|---|---|
| 0 | conditional | −0.0035 | +26.2 % | 0.65 | 0.25 | +0.129 | 0.65 |
| 0 | marginal | −0.0077 | +26.7 % | 0.49 | 0.24 | +0.097 | 0.49 |
| 3 | conditional | −0.0037 | +27.7 % | 0.74 | 0.31 | +0.146 | 0.74 |
| 3 | marginal | −0.0081 | +31.9 % | 0.35 | 0.17 | +0.070 | 0.35 |

The cells isolate as intended. The marginal-KM error alone does not move the ATE
(consistent with double robustness while the outcome model is correct), and the
joint cell doubles the small residual bias. **MAE is insensitive to the joint
error**, while rule purity and decision efficiency fall by roughly half.

## Decision rule (fixed before the confirmatory run)

**Primary (unchanged):** in cell D, mean paired Δ MAE < 0, 95 % CI entirely
below 0, Jaccard ≥ 0.107, ≥ 70 % of replications better than the constant.

**Model-quality criteria (new, fixed, all must hold in cell D):**

1. |mean ATE bias| ≤ 0.015;
2. mean purity ≥ 0.50 (below that a rule is at or past the Gate 10 break-even);
3. efficiency ≥ 0.50 (the model captures at least half of the oracle benefit
   available at the same coverage);
4. the high leaf is genuinely beneficial (true mean τ > 0) in ≥ 90 % of
   replications.

Cell D is declared a **usable model under joint misspecification** only if the
primary and all four model-quality criteria hold. Cells A–C are reported in
full; no criterion is changed after seeing the confirmatory results.

## Interpretation rules (fixed)

* D passes everything → ψ discovery is robust to both nuisance models being
  wrong at this censoring strength; the v1.6 claim can say so.
* D keeps the MAE gain but fails purity/efficiency → the model remains a better
  *predictor* but no longer identifies a trustworthy subgroup. The claim must
  separate prediction from subgroup quality.
* D fails the MAE criteria → the joint error breaks the predictive advantage;
  report the boundary.
* Any cell where the high leaf is not truly beneficial in ≥90 % of reps is
  reported as a rule-quality failure even if MAE improves.

## Scope

One censoring strength (10 %), one censoring misspecification family (marginal
KM), one outcome violation (κ=3), one sample size. Stronger or event-dependent
censoring is a separate gate; do not generalise beyond these cells.

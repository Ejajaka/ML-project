# Gate 3 — delta sensitivity under known MNAR: verdict

200 replications per cell · 2 baseline censoring mechanisms × 4 MNAR strengths ·
n = 2000 · t\* = 730 days · censoring recalibrated to 25 % at every ψ.

**MNAR mechanism:** censoring hazard × exp(ψ) for patients whose latent T > t\*.
The tilt uses the latent T, so no function of observed covariates can recover the
correct weights.

## Check 1 — δ = 0 identity: PASS

`max |τ(0) − baseline AIPW| = 1.665e-16`. The adjusted estimator reproduces the
primary analysis to machine precision, so the sensitivity parameter is not entangled
with the baseline.

## Check 2 — recovery: the specified adjustment FAILS

The v1.1 adjustment tilts the **outcome model** for censored patients,
$\tilde p = \mathrm{expit}(\mathrm{logit}\,\hat p + \delta)$.

| cell | ψ | bias at δ=0 | slope of τ(δ) per unit δ | δ that "recovers" truth |
|---|---|---|---|---|
| covariate | 0.0 | −0.004 | **−0.003** | +0.31 |
| covariate | 0.5 | −0.011 | **−0.002** | +0.16 |
| covariate | 1.0 | −0.014 | **−0.001** | +0.10 |
| covariate | 1.5 | −0.019 | **−0.000** | +0.005 |
| independent | 0.0–1.5 | −0.002 … −0.004 | **−0.002 … −0.003** | noisy |

**The slope is zero.** One unit of δ moves the estimate by 0.003 on a treatment effect
of 0.2 — two orders of magnitude too little to correct a bias of 0.01–0.02. To correct
0.02 you would need δ ≈ −10, i.e. an odds ratio of 2×10⁻⁵. The "δ that recovers truth"
is therefore noise: it is +0.31 when there is *no* MNAR at all, and it moves in the
wrong direction as ψ grows.

**Diagnosis.** Under informative censoring the bias enters through the **weights**, not
through the imputed outcome. Tilting $\hat S_a$ for a censored minority that enters the
IPCW term with weight zero cannot move the estimator. The adjustment targets the wrong
quantity.

## The corrected parameterization works

Tilting the **censoring weights** instead —
$\hat G_{\text{eff}}(t^*\mid X) = \hat G(t^*\mid X)^{\exp(\delta)}$ — gives real leverage:

| cell | ψ | **slope (G tilt)** | **δ\* (G tilt)** | sd of δ\* | residual error | coverage at δ\* |
|---|---|---|---|---|---|---|
| covariate | 0.0 | +2.91 | **−0.022** | 0.49 | +0.0001 | 100 % |
| covariate | 0.5 | +2.51 | **+0.058** | 0.58 | −0.0007 | 99.5 % |
| covariate | 1.0 | +2.17 | **+0.146** | 0.48 | +0.0000 | 100 % |
| covariate | 1.5 | +2.14 | **+0.214** | 0.49 | +0.0000 | 100 % |
| independent | 0.0–1.5 | +0.26 | −0.10 … −0.14 | 0.78–0.85 | ≈0 | 100 % |

All three properties a sensitivity parameter must have:

1. **δ\* ≈ 0 when there is no MNAR** (−0.022 at ψ = 0). The specified μ-tilt fails this
   by a wide margin (+0.31).
2. **δ\* increases monotonically with the true departure** (−0.02 → +0.06 → +0.15 →
   +0.21). The implied calibration is δ\* ≈ 0.14 ψ, because the tilt applies to all
   patients while the true mechanism tilts only survivors.
3. **The estimate at δ\* recovers the truth** (residual error ≤ 0.0007 on τ ≈ 0.2),
   with intervals covering.

## The honest limitation: δ\* is imprecisely identified here

The standard deviation of δ\* across replications is **0.48–0.85** — wider than the
whole range of ψ tested. That is not a defect of the adjustment; it reflects the fact
that **the MNAR bias in this design is only 1–2 pp**, which is small relative to the
estimator's noise at 25 % censoring. The mean δ\* is monotone and interpretable, but an
individual δ\* is not well determined.

Consequence: the sensitivity analysis is useful as a **directional** device ("how much
tilt would be needed to overturn this verdict") and not as a precise estimate of the
departure. That is the correct reading of a sensitivity parameter, and it matches the
review's caution that δ is not an identified quantity.

A stronger MNAR configuration (higher ψ, or 50 % censoring) would sharpen the
identification, at the cost of a noisier AIPW. That trade-off has not been run yet.

## Consequence for the specification — v1.3

| Component | v1.2 | v1.3 |
|---|---|---|
| Sensitivity parameterization | logit tilt of $\hat S_a$ for censored patients | **power tilt of $\hat G$: $\hat G^{\exp(\delta)}$** |
| δ = 0 | reproduces baseline | reproduces baseline (verified, 1.7e-16) |
| Leverage | ~0.003 per unit δ — unusable | 2.1–2.9 per unit δ |
| Behaviour at ψ = 0 | δ\* = +0.31 (wrong) | δ\* = −0.02 (correct) |

The arm-specific outcome tilt is **rejected** and should be documented as a failed
attempt in the methods section — it is a useful negative result, because "tilt the
outcome model" is the intuitive first idea and it does not work.

## What Gate 3 establishes

- The baseline estimator's bias under this MNAR family is modest: −0.004 (MAR) to
  −0.019 (ψ = 1.5) on a τ of 0.2, i.e. up to ~10 % relative.
- Interval coverage at δ = 0 stays at 90–96.5 %, so the primary analysis is not
  catastrophically fragile to this particular departure.
- A weight tilt recovers the truth and has a monotone, interpretable calibration curve.
- The specified v1.1 adjustment does not work and must be replaced.

## Not established

- Recovery under strong MNAR with heavy censoring (the regime where the bias is large).
- Whether δ is stable across subgroups — δ\* is computed per rule but not yet compared.
- Anything about SCRE (Gate 4).

# Gate 14 pre-registration — outcome-model misspecification (revised before running)

Frozen 2026-11-01. **Revised after a pre-run DGP verification, before any
replication was executed.** The revision and its reason are recorded below.

## Why the first design was discarded

The first frozen design put a region-specific non-proportional hazard in the
**treated** arm. A pre-run check showed it does not test what it claims:

| design (20 seeds) | τ_in | τ_out | corr(target, τ) | ATE bias |
|---|---|---|---|---|
| no misspecification | 0.195 | 0 | 0.655 | −0.0008 |
| treated arm, κ=1 | 0.195 | 0 | 0.661 | −0.0008 |
| treated arm, κ=2 | 0.195 | 0 | 0.662 | −0.0008 |

The treated-arm violation leaves the target untouched. The region signal in
τ̂ = μ̂₁ − μ̂₀ is carried by the covariate main effects in the **control** arm,
and a zero-mean time-varying effect is what a Cox partial likelihood averages
away. A pass under that design would have been vacuous.

A second check applied the same violation to the control arm:

| control-arm violation | τ_in | τ_out | corr(target, τ) | ATE bias |
|---|---|---|---|---|
| κ=1 | 0.195 | 0 | 0.569 | −0.0009 |
| κ=3 | 0.195 | 0 | **0.018** | −0.0009 |
| κ=5 | 0.195 | 0 | −0.133 | −0.0009 |

κ = 3 destroys the discovery target while leaving the truth and the marginal
estimator untouched. That is the experiment.

## Exactly what is misspecified

For patient i with standardised age `a`, standardised CD40 `c`, region index
`zR = 0.5*(a − c)`, region `R = 1{cd40 < median AND age > median}`:

```
control arm:  h0(t) = lam0 exp(lp0) * m(t) * 1{R}
              m(t) = e^{κ zR} / cosh(κ zR)   for t <= 365
              m(t) = e^{-κ zR} / cosh(κ zR)  for t >  365
treated arm:  h1(t) = lam0 exp(lp0 − delta*R)   for all t
```

Because `e^x + e^{-x} = 2 cosh(x)`, the cumulative hazard over [0, 730] in the
control arm is unchanged by the multiplier:

    ∫0^730 h0 dt = lam0 exp(lp0) * 365 * (e^{κzR} + e^{-κzR}) / cosh(κzR)
                 = lam0 exp(lp0) * 730

So **S0(730), S1(730) and the true τ are identical to the correctly specified
DGP**, and τ is still exactly zero outside the region. Only the *path* of the
control-arm hazard inside the region is non-proportional.

| component | status |
|---|---|
| **control arm, inside region** | **misspecified** — non-PH with a large early/late log-hazard swing; a single PH Cox cannot represent it |
| treated arm | correctly specified (exponential, PH) |
| control arm, outside region | correctly specified (`m = 1`) |
| censoring | **correctly specified**: `C ~ Exp(lam_c0 exp(X@GAMMA))`, independent of T given X, fitted as a Cox on X |
| S(730) and the true τ | **unchanged by construction** |
| extractor, target construction, AIPW effects, cap, default | unchanged from Gate 12 |

A negative result is therefore attributable to the outcome model alone, not to
a changed estimand, not to a weaker signal, and not to censoring.

## Frozen specification

| item | value |
|---|---|
| sample size | n = 8000 |
| replications | 40 fresh seeds per arm |
| primary arm | **κ = 3** (control-arm violation), seeds 160100–160139 |
| descriptive | κ = 1 (mild), seeds 160300–160339; κ = 0 (control), seeds 160200–160239 |
| model | depth-3 CART on the cross-fitted Cox plug-in target |
| rule effects | held-out AIPW φ (doubly robust: correct censoring model) |
| default / comparator | TRAIN marginal ATE |
| censoring calibration | lam_c0 bisected so 10 % are censored before 730 under each κ |
| metric | paired held-out Δ MAE = MAE(rule model) − MAE(constant) |

κ = 0 is the control: same seeds machinery, same calibration, no violation. If
κ = 0 passes and κ = 3 fails, the misspecification is the cause.

## Decision rule (frozen)

The primary arm (κ = 3) is declared **robust** only if ALL of:

1. mean paired Δ < 0;
2. the 95 % paired bootstrap CI lies entirely below 0;
3. best-leaf Jaccard ≥ 0.107;
4. at least 70 % of replications have Δ < 0.

## Mandatory diagnostics

Target correlation with the true τ on TRAIN; target sd; regional target
contrast; high-leaf prevalence, purity, capture; leaf AIPW effect and SE;
best-leaf Jaccard; split stability; paired Δ MAE with CI; marginal ATE estimate
versus the true ATE.

## Interpretation

* κ = 0 pass + κ = 3 fail: **the v1.5 gain depends on the outcome model being
  approximately correct.** The κ = 1 result locates the tolerance.
* κ = 3 pass: the gain survives a target that is essentially uncorrelated with
  the truth, which would imply discovery is not using the plug-in signal — that
  would itself be suspicious and must be investigated before claiming
  robustness.
* κ = 0 fail: the new DGP machinery is at fault; no misspecification conclusion.

## Scope

One violation family, one covariate index, one sample size. Nothing here says
anything about joint outcome-plus-censoring misspecification (the censoring
model is deliberately correct) or about other DGP families.

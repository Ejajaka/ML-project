# Gate 8 pre-registration — RMST as the discovery estimand

Written before the experiment was executed (2026-10-31).

## Question

Gate 6 established that the discovery pipeline works when given the true
individual effect (RF on true tau: Jaccard 0.98, stability 100 %) and fails when
given the AIPW pseudo-outcome for S(730) (Jaccard 0.12). Gate 7 showed that
changing the rule granularity does not fix it. The remaining explanation is the
variance of psi itself: sd(psi) ~ 1.0 against a regional signal of 0.195.

This gate tests whether changing the *estimand* from a single-horizon survival
probability to RMST to day 730 reduces that variance enough to change discovery.

## Design (all components except the estimand held at v1.4)

| component | value |
|---|---|
| DGP | piecewise region `CD4 < median ∧ age > median`, effect +0.20 in S(730), 10 % covariate-dependent censoring |
| n | 2000 |
| replications | 30 |
| split | 70 / 30 stratified by A |
| discovery | RF depth 3, min_samples_leaf 60, 30 trees, stability threshold 0.50, Jaccard prune 0.8 |
| nuisance model | per-arm Cox, 5-fold cross-fitted |
| weight cap | min(Q0.99 of 1/G on fitting rows, 20), applied pointwise along the curve for RMST |
| evaluation | held-out 30 %; nuisances fitted on TRAIN only |
| constant baseline | marginal ATE estimated on TRAIN, each estimand gets its own constant |
| oracle | true region membership, region effects estimated on TEST from held-out phi |

## Estimands

* `prob_730`: tau = S_1(730|X) − S_0(730|X) (current production estimand)
* `rmst_730`: tau = ∫_0^730 S_1(t|X) dt − ∫_0^730 S_0(t|X) dt (days)

RMST is rebuilt end to end: per-arm Cox survival curves integrated on a
121-point grid, censoring model evaluated pointwise, IPCW pseudo-outcome
`Y = ∫_0^{min(T,730)} dt / G(t|X)`, and the true RMST computed analytically from
the exponential DGP rates (`(1−S)·730 / (−log S)`).

## Primary endpoint

Paired `delta = MAE(CISCaRL) − MAE(constant)` within replication, on the same
held-out patients, for the RMST estimand.

## Decision rule (fixed before running)

Adopt RMST as the discovery estimand only if all four hold:

1. mean paired delta < 0;
2. the 95 % paired bootstrap CI (5000 resamples) excludes zero;
3. best-rule Jaccard is not worse than 80 % of the day-730 value;
4. at least 70 % of replications have delta < 0.

## Reporting commitments

* Both arms are reported, including variance, signal-to-noise, Jaccard,
  stability, MAE, constant MAE and oracle MAE.
* MAE is in native units per estimand; the two arms are **not** compared to each
  other on MAE. Each is compared to its own constant.
* If RMST fails, that is reported as a loss and no further estimand changes are
  attempted in this gate.
* Note on the DGP: with exponential event times, RMST to a fixed horizon is a
  monotone transform of S(t*), so the two estimands are informationally
  equivalent up to noise propagation. Any RMST gain would therefore have to come
  from how noise propagates through the transform, not from new information.
  This is recorded now so that a null result is not over-interpreted later.

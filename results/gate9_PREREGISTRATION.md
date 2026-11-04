"""Gate 9 pre-registration - smooth plug-in target for discovery.

Written before the experiment was executed (2026-10-31).

## Question

Gates 4e, 6, 7 and 8 have eliminated every discovery-side lever that leaves the
pseudo-outcome unchanged. Gate 6 showed the pipeline recovers the region when
given the true individual effect (Jaccard 0.98, stability 100 %), and Gate 8
showed that re-expressing the same survival curve (RMST) does not change the
signal-to-noise ratio (0.20 vs 0.19) and does not help.

The remaining hypothesis is that the *variance* of the IPCW pseudo-outcome is
the bottleneck, not the rule search. This gate replaces the discovery target
with a cross-fitted plug-in contrast while leaving effect estimation untouched:

    discovery target   tau_hat(X) = mu_hat_1(X) - mu_hat_0(X)
    effect estimation  unchanged: held-out AIPW phi, rule effect = mean(dphi | rule)

The doubly robust effect-estimation component is therefore preserved; only rule
*generation* changes. Any gain cannot come from a different effect estimator.

## Design (frozen from v1.4, identical to Gate 8)

| component | value |
|---|---|
| DGP | region = CD4 < median(340) AND age > median(34), effect +0.20 in S(730), 10 % covariate censoring |
| n | 2000 |
| replications | 30 |
| split | 70 / 30 stratified by A, `random_state = rep` |
| discovery | RF depth 3, min_samples_leaf 60, 30 trees; stability >= 0.50; Jaccard prune 0.8; no rule cap |
| effect estimation | held-out AIPW phi, nuisances fitted on TRAIN only, cap = min(Q0.99, 20) |
| constant baseline | marginal ATE on TRAIN, applied to every test patient |
| oracle | true region membership, region effects estimated on TEST from held-out phi |
| bootstrap seed for stability | `random_state = rep + 1` in every arm (paired) |

## Arms

1. `aipw_psi` - current production target (baseline, not a candidate)
2. `cox_plugin` - cross-fitted per-arm Cox: `mu_hat_1 - mu_hat_0`
3. `rsf_plugin` - cross-fitted per-arm Random Survival Forest: `mu_hat_1 - mu_hat_0`

Arm 3 exists to test whether any gain depends on the Cox model being correctly
specified. RSF is *not* assumed to be automatically robust; it is a different
outcome-modelling strategy with its own failure modes.

## Primary endpoint

Paired `delta = MAE(CISCaRL) - MAE(constant)` within replication, on the same
held-out patients, same training-derived constant, same folds.

## Decision rule (fixed before running)

A target is adopted only if ALL of:

1. mean paired delta vs the constant < 0;
2. the 95 % paired bootstrap CI is entirely below 0;
3. mean best-rule Jaccard is not worse than the `aipw_psi` baseline
   (tolerance 0.01);
4. at least 70 % of replications have delta < 0.

Secondary (reported regardless of the decision, never used to adopt):
paired MAE difference against the `aipw_psi` baseline, and target fidelity on
TRAIN (correlation with true tau, sd, regional contrast).

## Secondary analyses (run only if a primary arm is adopted)

* repeat the Gate 7 support sweep {60, 100, 150, 200} with the winning target;
* repeat the Gate 5 MNAR censoring tilt delta in [-2, 2] on the winning rule set.

## Reporting commitments

* All three arms reported, including losses.
* No threshold, support, pruning or cap is changed in this gate.
* If neither plug-in target beats the constant, the conclusion is that the
  discovery layer is information-limited at n = 2000 even with a de-noised
  target, and that is recorded as a negative result.
"""

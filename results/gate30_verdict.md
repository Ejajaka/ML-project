# Gate 30 verdict — knob sweep for C200

Pre-registration: `gate30_PREREGISTRATION.md` (frozen before execution).
40 seeds for tuning (310100–310139), 40 fresh seeds for confirmation
(320100–320139), two DGPs, paired within replication.

## What was tested

| id | change |
|---|---|
| V0 | frozen C200 (depth 3, min_leaf 60, support ≥ 200, z = 1.645) |
| V2 | V0 + empirical-Bayes shrinkage of leaf effects toward the marginal ATE, prediction only |
| V3a | commitment z = 1.282 |
| V3b | commitment z = 1.960 |
| V4a | min_samples_leaf 30 |
| V4b | min_samples_leaf 100 |
| V5 | depth 4 |

## Tuning block (40 seeds)

| kind | variant | MAE | Δ vs V0 | 95 % CI | p |
|---|---|---|---|---|---|
| single | V2 | 0.0521 | −0.0054 | [−0.0086, −0.0021] | 0.003 |
| single | V3a | 0.0575 | −0.0000 | [−0.0003, +0.0002] | 0.77 |
| single | V3b | 0.0582 | +0.0007 | [+0.0000, +0.0017] | 0.039 |
| single | V4a | 0.0576 | +0.0000 | [−0.0020, +0.0021] | 0.98 |
| single | V4b | 0.0582 | +0.0007 | [−0.0007, +0.0022] | 0.39 |
| single | V5 | 0.0667 | +0.0092 | [+0.0070, +0.0116] | <0.001 |
| two | V2 | 0.0676 | −0.0025 | [−0.0054, +0.0005] | 0.11 |
| two | V3a | 0.0701 | −0.0000 | [−0.0001, +0.0000] | 0.25 |
| two | V3b | 0.0706 | +0.0005 | [−0.0000, +0.0012] | 0.071 |
| two | V4a | 0.0707 | +0.0005 | [−0.0007, +0.0019] | 0.43 |
| two | V4b | 0.0686 | −0.0015 | [−0.0028, −0.0003] | 0.011 |
| two | V5 | 0.0774 | +0.0073 | [+0.0051, +0.0096] | <0.001 |

## Confirmation block (fresh seeds)

| kind | variant | MAE | Δ vs V0 | 95 % CI | p |
|---|---|---|---|---|---|
| single | V2 | 0.0530 | −0.0047 | [−0.0089, −0.0006] | 0.022 |
| single | V4b | 0.0574 | −0.0003 | [−0.0026, +0.0017] | 0.80 |
| single | V5 | 0.0682 | +0.0105 | [+0.0076, +0.0134] | <0.001 |
| two | V2 | 0.0685 | −0.0027 | [−0.0057, +0.0003] | 0.078 |
| two | V4b | 0.0696 | −0.0016 | [−0.0033, −0.0000] | 0.045 |
| two | V5 | 0.0805 | +0.0093 | [+0.0070, +0.0117] | <0.001 |

Pooled over both blocks (80 paired replications):

* single, V2: −0.0051 [−0.0077, −0.0025], p < 0.001
* two, V2: −0.0026 [−0.0048, −0.0005], p = 0.014
* two, V4b: −0.0016 [−0.0026, −0.0006], p < 0.001

## V2 does not change the policy

On the confirmation block, V0 and V2 are identical in every policy quantity:

| kind | purity | capture | commitment | Spearman |
|---|---|---|---|---|
| single | 0.8719 / 0.8719 | 0.4510 / 0.4510 | 72.5 % / 72.5 % | 0.522 / 0.522 |
| two | 0.8777 / 0.8777 | 0.2562 / 0.2562 | 62.5 % / 62.5 % | 0.444 / 0.444 |

Shrinkage is a calibration of the reported per-patient effect, not a change in
which patients are treated or which rule is selected.

## Small-n grid (absolute 200 vs 5 % proportional floor)

| n | variant | purity | capture | commit | fallback | support |
|---|---|---|---|---|---|---|
| 1000 | abs200 | 0.614 | 0.105 | 27.5 % | 80.0 % | 115 |
| 1000 | prop5 | 0.624 | 0.102 | 27.5 % | 0 % | 87 |
| 2000 | abs200 | 0.468 | 0.096 | 17.5 % | 0 % | 373 |
| 2000 | prop5 | **0.728** | **0.127** | **30.0 %** | 0 % | 129 |
| 4000 | abs200 | 0.710 | 0.311 | 55.0 % | 0 % | 451 |
| 4000 | prop5 | 0.804 | 0.281 | 55.0 % | 0 % | 322 |

At n = 2000 the absolute floor is 20 % of discovery and behaves badly; the
proportional floor is purer, captures more, and commits more often. At n = 1000
neither commits much (validation is too small). At n = 4000 the two differ only
in the usual capture--purity trade.

## Verdict

**Adopted: V2 (shrinkage), for prediction only.** It improves MAE in the
single-region design in both blocks and pooled (−0.0051, p < 0.001), is
directionally consistent and pooled-significant with two regions (−0.0026,
p = 0.014), and changes no policy quantity. It is reported as `C200-S`.

**Rejected: depth 4 (V5).** Worse MAE in both blocks and both designs
(+0.008 to +0.011), and it loses capture.

**Secondary, not adopted: min_leaf 100 (V4b).** Consistent small gain in the
two-region design only (−0.0016 pooled), no gain with one region. Adopting it
would be a design-specific tune.

**Policy dial, reported not adopted: z.** z = 1.28 raises commitment from 72.5 %
to 82.5 % and lowers purity; z = 1.96 cuts commitment to 60–70 % and raises
purity. MAE is essentially unchanged.

**Small n: proportional floor is the fix.** It should replace the absolute 200
in any future specification applied to data below ~4000 discovery patients.

## Head-to-head consequence (80 paired replications)

Pooled over the two seed blocks, with SCRE fitted on the same splits each time:

| design | method | MAE | vs SCRE | 95 % CI | p | purity | capture | per-treated | coverage |
|---|---|---|---|---|---|---|---|---|---|
| single | C200 | 0.0576 | +0.0062 | [+0.0031, +0.0094] | <0.001 | 0.890 | 0.465 | +0.174 | 0.133 |
| single | **C200-S** | **0.0525** | **+0.0011** | [−0.0019, +0.0042] | 0.48 | 0.890 | 0.465 | +0.174 | 0.133 |
| single | SCRE | **0.0514** | — | — | — | 0.437 | 0.954 | +0.085 | 0.548 |
| two | C200 | 0.0707 | +0.0031 | [+0.0001, +0.0059] | 0.044 | 0.858 | 0.277 | +0.165 | 0.117 |
| two | **C200-S** | **0.0680** | **+0.0004** | [−0.0025, +0.0033] | 0.76 | 0.858 | 0.277 | +0.165 | 0.117 |
| two | SCRE | **0.0676** | — | — | — | 0.599 | 0.823 | +0.101 | 0.563 |

Files: `gate30_headtohead.csv`, `gate30_variant_table.csv`.

This changes the paper's sentence "0.003–0.004 worse on MAE" to "significantly
worse for the frozen model, statistically tied after shrinkage". The shrinkage
is reported as a post-hoc calibration with its own confirmation block, not as
part of the frozen C200.

## What to do next

1. Report C200-S alongside C200 in the paper, tables and notebook; keep the
   frozen configuration as the pre-registered object.
2. Do not adopt min_leaf 100: two-region-only gain, design-specific tune.
3. Future specification: proportional support floor for small samples.
4. Do not chase MAE further; the remaining gap is coverage, which is the point
   of the method.


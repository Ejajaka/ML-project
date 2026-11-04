# Gate 14 verdict — the gain depends on the outcome model, and here is where it breaks

Pre-registration: `gate14_PREREGISTRATION.md` (revised before running after a
pre-run DGP verification; both the revision and its reason are recorded there).
40 fresh seeds per arm, n = 8000.

## Result

| arm | κ | rules | MAE | constant | Δ | 95 % CI | reps better | gain |
|---|---|---|---|---|---|---|---|---|
| control (no violation) | 0 | 7.9 | 0.0502 | 0.0719 | −0.0217 | [−0.0261, −0.0173] | 92 % | +30.1 % |
| mild violation | 1 | 8.0 | 0.0543 | 0.0713 | −0.0170 | [−0.0221, −0.0117] | 85 % | +23.8 % |
| **severe violation (primary)** | **3** | 7.9 | 0.0765 | 0.0734 | **+0.0030** | **[−0.0001, +0.0060]** | **38 %** | **−4.1 %** |

**Pre-registered decision: NOT robust at κ = 3.** Mean Δ > 0, CI includes zero
(its upper bound is above), 38 % of replications improve.

## Diagnostics — the failure is in the target, exactly as designed

| arm | target corr with true τ | target contrast | true contrast | ATE estimate | true ATE |
|---|---|---|---|---|---|
| κ=0 | 0.650 | 0.115 | 0.195 | +0.046 | +0.049 |
| κ=1 | 0.544 | 0.061 | 0.195 | +0.046 | +0.048 |
| κ=3 | **0.093** | **0.007** | 0.195 | +0.050 | +0.048 |

| arm | purity | capture | high-leaf effect | SE | Jaccard | stability |
|---|---|---|---|---|---|---|
| κ=0 | 0.93 | 0.40 | +0.187 | 0.076 | 0.51 | 0.59 |
| κ=1 | 0.89 | 0.43 | +0.170 | 0.088 | 0.53 | 0.60 |
| κ=3 | **0.36** | **0.15** | +0.056 | 0.112 | 0.24 | 0.63 |

Three things are established.

1. **The violation worked.** Target correlation falls 0.650 → 0.544 → 0.093 and
   the target's regional contrast collapses 0.115 → 0.061 → 0.007 while the true
   contrast stays 0.195 in every arm (S(730) is unchanged by construction).
2. **The isolation worked.** The marginal ATE estimate is unbiased in all three
   arms (+0.046/+0.046/+0.050 vs +0.049/+0.048/+0.048). The correct censoring
   model keeps AIPW consistent, so the misspecification could only damage
   discovery — and only discovery was damaged.
3. **The extractor is not at fault.** The CART still produces ~8 rules with
   stability 0.63 under κ=3; the rules are simply built on a target that no
   longer contains the region. Purity 0.36 and capture 0.15, versus 0.93 and
   0.40 without the violation.

## The boundary

| violation strength | target corr | outcome |
|---|---|---|
| none | 0.650 | +30.1 %, 92 % of reps |
| mild (κ=1, hazard ≈1.8× early / 0.24× late at one region-unit) | 0.544 | +23.8 %, 85 % of reps |
| severe (κ=3, ≈2.0× early / 0.005× late) | 0.093 | −4.1 %, 38 % of reps |

Interpretation, scoped to this violation family and this DGP:

* The v1.5 gain **tolerates a substantial within-arm non-PH violation** — at
  κ=1 the target correlation falls by 16 % and the gain only falls from +30.1 %
  to +23.8 %.
* It **does not survive a violation that removes the region signal from the
  plug-in target** (κ=3, correlation 0.09).
* The practical statement: the method needs the outcome model to preserve the
  **ordering** of the treatment contrast across patients. Absolute accuracy of
  μ̂ is not required; the ranking is.

## What this changes

The v1.5 claim must carry a dependence clause: the gain holds **provided the
outcome nuisance approximately preserves the between-region ordering of τ**.
That is a real limitation, now measured rather than assumed.

It also identifies the next improvement target: a discovery target that does not
rest on one linear Cox fit per arm. Candidates are a flexible/ensemble plug-in
(noisier in Gate 9), or a target that combines the plug-in with the AIPW
pseudo-outcome so that it degrades gracefully rather than collapsing when the
nuisance is wrong. Neither is tested here.

## Limits

One violation family (control-arm non-PH in the region), one covariate index,
one sample size, one DGP. The censoring model is deliberately correct, so this
is **not** a joint-misspecification result. κ=2 was not run, so the exact
breakpoint between κ=1 and κ=3 is bounded, not located.

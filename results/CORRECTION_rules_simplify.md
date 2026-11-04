# Correction — `rules.simplify` bound-direction bug

Discovered 2026-11-02 during the Gate 19 pre-run smoke test. Fixed before Gate 19
was executed. This note records the defect, its scope, and every re-run.

## The defect

`ciscarl2/rules.py::simplify` collapsed repeated conditions on the same feature
to the wrong bound:

```python
le[f] = max(le[f], thr)   # for  x <= a AND x <= b  the binding bound is min(a,b)
gt[f] = min(gt[f], thr)   # for  x >  a AND x >  b  the binding bound is max(a,b)
```

Whenever a root-to-leaf path split **twice on the same feature**, the simplified
rule was a strict superset of the leaf. Example from a depth-3 tree: a leaf with
129 training patients evaluated to 403 patients under its simplified condition.

## Impact

* **22.7 % of leaves** (35/154 over 20 replications at n = 8000) had a condition
  that did not match their stored mask.
* Median coverage inflation among affected leaves: **+92 %**; maximum **+1247 %**.
* Scope: every gate that calls `extract_leaf_rules` — the single-CART extractor
  used from Gate 10 onward, and the RF + stability pipeline used in Gates 4d, 7
  and 9.
* Direction of the distortion: rules covered more patients than intended, so
  high-leaf purity, capture, coverage and the rule-effect estimates were all
  affected. The *selection* of the high leaf was unaffected (it used the correct
  stored training mask); the *test-set evaluation* of that leaf was inflated.

## The fix

`min` for `<=`, `max` for `>`, with a comment. Verified on three independent
trees: **0 mismatches**, and the leaf masks now partition the space exactly
(sum of mask sizes = n).

## Re-runs (same seeds, corrected code)

### Gate 16 — ψ vs plug-in

| n | κ | target | gain (pre-fix) | **gain (corrected)** | reps better |
|---|---|---|---|---|---|
| 8000 | 0 | ψ | +37.8 % | **+35.7 %** | 98 % |
| 8000 | 0 | plug-in | +28.2 % | **+32.5 %** | 98 % |
| 8000 | 3 | ψ | +32.7 % | **+30.9 %** | 92 % |
| 8000 | 3 | plug-in | −4.0 % | **−4.9 %** | 32 % |
| 12000 | 0 | ψ | +48.0 % | **+46.8 %** | 100 % |
| 12000 | 0 | plug-in | +32.5 % | **+39.2 %** | 100 % |
| 12000 | 3 | ψ | +53.4 % | **+51.9 %** | 100 % |
| 12000 | 3 | plug-in | +6.5 % | **+5.3 %** | 57 % |

ψ still passes all four checks in all four cells. Two corrections to the
interpretation:

1. **At n = 8000, κ = 0, ψ is no longer significantly better than the plug-in**:
   paired Δ = −0.0023 [−0.0069, +0.0026], 55 % of reps. At n = 12000 it is
   (−0.0054 [−0.0097, −0.0011]) and under κ = 3 it is decisively
   (−0.0261 and −0.0340).
2. **High leaves are much smaller than the inflated masks suggested.** Capture is
   0.06–0.29 rather than 0.30–0.43; the model's subgroup is a small, high-purity
   slice, not a near-complete region.

### Gate 17 — operating range

| n | ψ gain (pre-fix) | **ψ gain (corrected)** | plug-in gain | pass |
|---|---|---|---|---|
| 2000 | −22.8 % | **−26.4 %** | −29.6 % | neither |
| 4000 | +8.1 % | **+4.5 %**, CI [−0.009, +0.002] | +11.4 % | ψ fails; plug-in passes |
| 6000 | +23.0 % | **+21.7 %**, CI [−0.021, −0.010] | +26.3 % | both pass |

**Correction: the ψ operating range is n ≥ 6000, not 4000.** The plug-in works
from 4000 and is better at 4000–6000; the crossover is ≈8000, not ≈6000. The
oracle at n = 2000 still wins by 40.1 %, so the n = 2000 barrier remains
discovery, not information.

### Gate 18 — joint nuisance misspecification

| cell | gain (pre-fix) | **gain (corrected)** | purity | efficiency | beneficial reps |
|---|---|---|---|---|---|
| correct / conditional | +30.5 % | **+28.5 %** | 0.772 | 0.761 | 87.5 % |
| correct / marginal | +30.6 % | **+28.4 %** | 0.694 | 0.687 | 75.0 % |
| κ=3 / conditional | +30.1 % | **+28.8 %** | 0.833 | 0.825 | 90.0 % |
| **κ=3 / marginal** | **+31.0 %** | **+29.1 %** | **0.671** | **0.662** | **70.0 %** |

The Gate 18 conclusion is unchanged: prediction survives, the subgroup bar
(≥90 % beneficial) fails at 70 %, the leaf is never harmful.

## Not re-run

Gates 10–15 were executed pre-fix. Their **quantitative** numbers should not be
quoted without re-running; their **qualitative** conclusions (depth-1 fails at
n = 2000; RMST fails; the plug-in is better below ~6000; the blend fails) are
consistent with the corrected Gates 16–18, but the exact figures are stale.
Gate 4d/7 used the same utility through the RF pipeline and carry the same
caveat.

Every table in `CLAIMS_v1.7.md` and `paper/DRAFT.md` uses corrected numbers from
Gates 16–19 only.

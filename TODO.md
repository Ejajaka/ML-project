# TODO / Future Work

Tracked ideas that are not yet implemented or benchmarked. Each entry states the
motivation, the concrete deliverable, and what "done" looks like.

---

## 1. Horizon sensitivity analysis — rerun across several `t*`

**Motivation.** `t*` is a design choice, not a data constant. The estimand
`τ(x)` changes with the horizon, and under non-proportional hazards the ranking
of which subgroup benefits can flip between horizons. Reporting a single `t*`
(the current default is the median event time, `python/utils.py:14`) can
therefore hide horizon-dependent conclusions.

**Deliverable.**
- Run the full pipeline / fair benchmark at a small grid of horizons
  (e.g. `t*` at the 25th, 50th, and 75th percentile of observed event times).
- Report per-horizon rule lists, CATEs, intervals, and metric tables.
- State explicitly whether the headline claim (and rule membership) is stable
  across horizons.

**Done when.** A sensitivity table exists showing the same conclusion holds
(or not) across the horizon grid, and `PAPER.md` / `PROJECT_STATUS.md` state
which horizon the headline numbers use and why.

**Status:** not started.

---

## 2. A visual, elbow-style diagnostic for choosing `t*`

**Motivation.** kNN practitioners eyeball an elbow plot of error vs `k`. We have
no equivalent aid for `t*` — it is currently a fixed default. We need a
diagnostic figure that makes the trade-offs visible so `t*` can be justified
from data rather than picked arbitrarily.

**Candidate curves to plot against `t` (x-axis):**
1. **Fraction of patients with a known outcome** at `t` — `T > t` OR
   (`T ≤ t` AND `Δ = 1`). Monotone decreasing; low `t` keeps everyone, high `t`
   loses the censored.
2. **Label variance / balance** — for binary `Y`, `Var(Y) = p(1−p)` with
   `p = P(alive at t)`. Peaks near `p = 0.5`; a horizon where `Y` is nearly
   constant carries little signal.
3. **Treated-vs-control separation** — `|S₁(t) − S₀(t)|` across time (or a
   running log-rank / Mann–Whitney statistic). The horizon where the arms
   diverge most is where the effect is most detectable.
4. **IPCW weight inflation** — e.g. median or 95th-percentile `1/G(t)`. This
   flags when late `t` is propped up by a few heavily weighted survivors and
   intervals become untrustworthy.

**Proposed reading.** Choose `t*` near where curves 2–3 peak (or plateau) while
curves 1 and 4 are still acceptable — i.e. the "elbow" where added horizon buys
little signal but costs known-sample and weight stability. Because there are
several signals, a small multi-panel figure is clearer than a single number.

**Done when.** A function (e.g. `python/horizon_diagnostic.py`) produces this
multi-panel plot for a dataset, and the guide/library documents how to read it.

**Status:** not started. *Design sketch only — no code yet.*

---

## 3. Research: incorporate "when things happened" (event timing)

**Motivation.** The current method collapses survival to a binary label
`Y = I(T > t*)`. That discards timing: a death at month 2 and a death at month
11 both become `Y = 0`, and censoring before `t*` is dropped outright (with
IPCW correcting only the weighting of the known patients). Time-to-event
information is thrown away.

**Research questions.**
- Can we use the full survival curve / hazard instead of one binary horizon,
  while keeping the interpretable rule-list output?
- Options to investigate: restricted mean survival time (RMST) difference as
  the per-rule estimand; discrete-time hazard regression on rule indicators;
  pseudo-outcomes derived from the survival or cumulative-hazard scale rather
  than a single `t*`; IPCW variants that let censored patients contribute
  partial information instead of being excluded.
- Does using timing improve the known weaknesses — patient ranking (Spearman)
  and absolute CATE fit (R²) — which `PROJECT_STATUS.md §7` flags as the
  method's weak spots?
- How does the interpretability cost change (more outputs per rule, or a curve
  per rule instead of one number)?

**Deliverable.** A short literature + design note (which existing estimators
already do this: RMST-based HTE, survival pseudo-outcomes, discrete hazard
rule ensembles) plus a prototype on one dataset comparing rule stability and
ranking against the current binary-horizon version.

**Done when.** A design note exists in `deliverables/` with a recommendation,
and (if promising) a prototype is benchmarked in the fair harness.

**Status:** not started.

---

## Backlog / small items

- Update `deliverables/visual-guide/index.html`: soften "an early dropout is
  useless" — a dropout before `t*` has an *unknown* outcome and is excluded,
  but the later the censoring, the less information is lost (IPCW); deaths at
  any time ≤ `t*` are already used.
- Add a short "choosing `t*`" subsection to the visual guide pointing at the
  diagnostic in item 2.

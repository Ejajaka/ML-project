# Gate 20 pre-registration — honest commitment validation

Frozen 2026-11-02, before execution. This is a **correctness and validation**
gate, not an optimization. No model component is tuned.

## Question

Gate 19's commitment decision used the test patients to estimate the leaf effect
and to assess the decision. The audit (`results/AUDIT_report.md`) showed the
decision quality survives an honest split, but commitment rate, coverage and
captured benefit are inflated by 25–30 %.

This gate re-establishes the decision baseline with three disjoint data roles,
and separates two explanations for the lower commitment rate:

1. removal of in-sample optimism;
2. reduced power from a smaller decision sample.

## Data roles (n = 8000, 50 / 25 / 25)

| partition | size | permitted use |
|---|---|---|
| discovery | 4000 | fit the CART; select the high leaf; fit nuisance models |
| validation pool | 2000 | estimate leaf effects and SEs; apply the commitment rule |
| assessment | 2000 | evaluate the frozen leaf and the decision policy only |

The assessment partition never affects the commit decision, the leaf, any
threshold, or any model choice. Nuisance models for validation and assessment
are fitted on the discovery partition only.

**Power comparison:** within the fixed validation pool, two nested validation
sets are used, with the same discovery and assessment sets:

* **V-half**: a random 1000 patients from the validation pool;
* **V-full**: all 2000 validation patients.

If V-full recovers commitment/coverage relative to V-half, the audit's loss was
partly power. If it does not, the loss is the honesty correction.

## Arms (paired within replication, same discovery and assessment)

| arm | leaf effects | decision |
|---|---|---|
| A: no commitment | validation | always treat the high leaf |
| B: honest commitment | validation | treat the high leaf only if `τ̂_L > 1.645·SE_L`; otherwise **no treatment** |

The no-treatment fallback is explicit: a non-committed run contributes zero
coverage and zero captured benefit, not the marginal ATE. The constant remains
the *prediction* fallback and the MAE comparator.

## Frozen design

| item | value |
|---|---|
| n | 8000 |
| partitions | 4000 / 2000 / 2000, stratified by A |
| replications | 40 fresh seeds per cell |
| seeds | 230100–230139 |
| cells | κ=0 with conditional Cox censoring; κ=3 with marginal KM |
| discovery | depth-3 CART on ψ, min_samples_leaf 60 |
| commitment | one-sided 95 %, `τ̂_L > 1.645·SE_L` |
| assessment | 2000 held-out patients, same for every arm and validation size |

## Primary metrics (reported, not collapsed)

Commitment rate; coverage; captured benefit (against the total true τ in the
assessment set); null-treatment rate (committed runs whose high leaf has true
mean τ ≤ 0.01); decision efficiency (per-treated / oracle at the same k);
per-treated benefit.

Secondary guardrails: predictive MAE vs the TRAIN constant; purity; capture;
best-leaf Jaccard; split stability on the discovery set.

## Pass criteria (tied to safety and utility, not to forecast ranges)

The primary cell is κ=3 with marginal KM (joint misspecification). B is accepted
only if ALL of:

**Safety**
1. null-treatment rate ≤ 5 %;

**Utility**
2. decision efficiency ≥ 0.70;
3. captured benefit ≥ 0.20;

**Improvement over A (paired)**
4. null-treatment rate is lower than A, with a paired bootstrap 95 % CI on the
   difference excluding zero;
5. captured benefit is at least **0.70 × A's captured benefit** (the decision may
   not lose more than 30 % of the available captured benefit).

**Prediction guardrail**
6. MAE still beats the constant: mean Δ < 0 with the 95 % CI below zero.

Commitment rate, coverage, per-treated benefit and captured benefit are
**reported**. Per-treated benefit is *not* a paired improvement criterion: when
B abstains, its per-treated average is computed only over the committed runs, so
it is mechanically higher than A's. The paired utility quantity is captured
benefit, which credits abstention with zero. The audit forecasts (40–55 %,
0.065–0.075, 0.25–0.28) are not pass criteria.

## Interpretation rules (fixed)

* B passes everything → v1.8 adopts the honest commitment layer; the decision
  claim is restated with the honest assessment figures.
* B passes safety but fails utility → commitment is too conservative; report the
  coverage/capture frontier and do not adopt.
* B fails safety → the commitment rule is not reliable enough with an honest
  validation set; investigate the uncertainty estimates.
* V-full materially better than V-half → the audit's loss was substantially a
  power effect; report both and state the validation-size dependence.
* V-full no better than V-half → the loss is the honesty correction; the audit's
  quality conclusions stand and the quantity is corrected downward.

## Scope

One sample size, two nuisance cells, one commitment rule, one partition ratio.
The high-leaf selection remains winner's-cursed (Gate 20 does not fix it); that
is the next diagnosed problem after this gate.

# Gate 17 pre-registration — where does the v1.6 rule model start to work?

Frozen 2026-11-02, before execution. Not edited after the run.

## Question

The n ≥ 8000 operating point was established under the **v1.5 plug-in target and
the RF + stability pipeline**. Gate 16 replaced both: ψ as target, depth-3 CART
as extractor. The paper's central sample-size claim now rests on a threshold
measured under the old configuration.

This gate locates the lower boundary for the **v1.6** model:

* If ψ-CART works at n = 2000, the earlier failures were the extractor/target,
  not the data, and the sample-size claim must be rewritten.
* If it starts at n = 4000–6000, the threshold moves down but the qualitative
  story survives.
* If it still requires n = 8000, the threshold is a property of the data and the
  existing claim stands.

## Frozen specification

| item | value |
|---|---|
| sample sizes | n ∈ {2000, 4000, 6000} |
| nuisance state | κ = 0 (correctly specified outcome and censoring models) |
| replications | 40 fresh seeds per cell |
| seeds | 2000: 190100–190139; 4000: 190200–190239; 6000: 190300–190339 |
| primary | depth-3 CART on the AIPW pseudo-outcome ψ |
| comparator | constant; secondary paired comparator: plug-in on the same seeds |
| **ceiling diagnostic** | **oracle**: true region membership, region effects estimated from the held-out AIPW φ |
| rule effects | held-out AIPW φ |
| default | TRAIN marginal ATE |
| metric | paired held-out Δ MAE = MAE(model) − MAE(constant) |

The oracle row is not a candidate. It separates two failure modes: if the oracle
already loses to the constant at a given n, the barrier is **estimation noise**,
not discovery; if the oracle wins but ψ loses, the barrier is **discovery**.

## Decision rule (unchanged)

ψ is adopted at a sample size only if ALL of:

1. mean paired Δ < 0;
2. the 95 % paired bootstrap CI lies entirely below 0;
3. best-leaf Jaccard ≥ 0.107;
4. at least 70 % of replications have Δ < 0.

## Interpretation rules (fixed before the run)

* ψ passes at n=2000 → the old threshold was an artefact of the RF + stability
  extractor. State that explicitly and revise the paper's sample-size section.
* ψ fails at 2000 but passes at 4000 or 6000 → locate the boundary and report it
  as a range, not a point.
* ψ fails everywhere below 8000 → the Gate 11–12 threshold is confirmed for the
  new configuration too, which strengthens the sample-size argument.
* Wherever ψ fails, report the oracle row. If the oracle also fails, say the
  barrier is estimation; if the oracle wins, say it is discovery.

## Scope

One DGP family, one extractor, κ=0 only. Nothing here tests robustness; that is
already covered by Gates 13–16 and, for joint misspecification, by the next gate.

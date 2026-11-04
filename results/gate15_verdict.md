# Gate 15 verdict — the blend fails; the AIPW target alone is better and robust

Pre-registration: `gate15_PREREGISTRATION.md`. n = 8000, 40 fresh seeds per cell,
depth-3 CART, held-out AIPW effects, TRAIN constant.

## Pre-registered primary (κ=3, standardised λ=0.5 blend)

| κ | λ | MAE | constant | Δ | 95 % CI | reps better | gain |
|---|---|---|---|---|---|---|---|
| **3** | **0.5 (primary)** | 0.0749 | 0.0727 | **+0.0022** | **[−0.0025, +0.0069]** | **48 %** | **−3.0 %** |

**FAIL** on three of four checks. The blend does not rescue the collapsed
plug-in. The pre-run pilot predicted this: ψ's ordering signal is ~0.08 even
under a correct nuisance, so it cannot compensate for a target that has lost its
ordering.

## The unexpected result: the AIPW target alone

| κ | λ | target corr | MAE | Δ vs constant | 95 % CI | reps better | gain | Δ vs plug-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 (ψ) | 0.082 | 0.0526 | −0.0190 | [−0.0246, −0.0132] | 80 % | +26.6 % | **−0.0038** |
| 0 | 1 (plugin) | 0.646 | 0.0564 | −0.0153 | [−0.0208, −0.0090] | 85 % | +21.3 % | — |
| 1 | 0 (ψ) | 0.083 | 0.0501 | −0.0214 | [−0.0270, −0.0158] | 92 % | +29.9 % | **−0.0022** |
| 1 | 1 (plugin) | 0.551 | 0.0523 | −0.0192 | [−0.0249, −0.0132] | 82 % | +26.9 % | — |
| **3** | **0 (ψ)** | **0.087** | **0.0479** | **−0.0248** | **[−0.0297, −0.0194]** | **90 %** | **+34.1 %** | **−0.0260** |
| 3 | 1 (plugin) | 0.077 | 0.0740 | +0.0012 | [−0.0029, +0.0054] | 48 % | −1.7 % | — |

The ψ-only arm passes **all four pre-registered checks at κ=3** and beats the
plug-in at every κ, including the fully correct DGP. It is descriptive in this
gate (the primary was the blend), so it is not adopted here.

## Why ψ works at n = 8000 when it failed at n = 2000

The individual-level correlation is low (0.08) at both violations, but
correlation is the wrong diagnostic for a tree. What matters is the **leaf-level**
signal-to-noise:

* ψ is an unbiased estimate of τ at the patient level (AIPW with a correct
  censoring model), with sd ≈ 0.97;
* a depth-3 leaf at n = 8000 holds ~700 training patients, so its mean ψ has
  SE ≈ 0.97/√700 ≈ 0.037 against a true regional contrast of 0.195 — about 5σ;
* at n = 2000 the same leaf holds ~175 patients, SE ≈ 0.073, and the tree is
  choosing among thousands of noisy splits, which is where it failed.

ψ is also **doubly robust**: under κ=3 the outcome model is wrong, but the
censoring model is correct, so ψ stays unbiased. That is why it recovers the
region (purity 0.71, capture 0.36, Jaccard 0.71) while the plug-in collapses
(purity 0.34, capture 0.19, Jaccard 0.25).

## What this implies

1. The Gate 14 limitation — "the gain depends on the outcome nuisance preserving
   the ordering of τ" — **may be a property of the plug-in target, not of the
   method.** If ψ's advantage replicates, the method is robust to outcome-model
   misspecification because its discovery target is doubly robust.
2. The cross-fitted plug-in may have been a **small-n crutch**. It fixed
   discovery at n = 2000 by trading robustness for precision; at n = 8000 the
   AIPW target has enough leaf-level precision on its own and is more robust.
3. The v1.5 model may therefore simplify: **depth-3 CART on the AIPW
   pseudo-outcome**, not on the plug-in, at n ≥ 8000.

None of this is adopted yet. The ψ arm was descriptive, uses the single-CART
extractor (not the production RF + stability), and is one DGP. It needs a
confirmatory gate on fresh seeds before it changes the claims contract.

## Limits

* ψ at κ=3 uses a **correct censoring model**. Under joint misspecification it
  is not guaranteed to remain unbiased.
* The result is for the single depth-3 CART; the RF + stability pipeline on ψ is
  not tested at n = 8000.
* The ψ MAE gain at κ=3 (+34.1 %) exceeds its gain at κ=0 (+26.6 %), which is
  counter-intuitive and not yet explained. It may reflect the changed event
  process under κ=3 (more early control-arm events), and it should be checked in
  the confirmatory run rather than interpreted now.

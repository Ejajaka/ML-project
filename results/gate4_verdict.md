# Gate 4 — CISCaRL v1.4 vs the actual SCRE implementation

Matched simulated data, split, horizon (t\* = 730 d), and evaluation patients.
n = 589 (30 reps) and n = 2000 (20 reps) · all-beneficial and sign-flip DGPs ·
10 % censoring at t\*.

## A methodological correction that had to be made first

`scre.py` selects its penalty by walking the Coxnet path **from the most regularised
end** and stopping at the first model that retains ≥ 1 rule-interaction term. At that
end every other term is already zero. On this data it produced:

- 12 of 12 linear×treatment terms **zero**
- 1 rule×treatment term with coefficient **0.014**
- predicted τ ≈ **0.002** against a true τ of **0.129**
- sign accuracy 57 %

That is a degenerate fit, not SCRE. Reporting it would have flattered CISCaRL, so the
comparison below instead selects α by **validation concordance on a 25 % slice of the
training data**, which is the standard criterion and what the project notebook does.
Any future benchmark against `scre.py` must do the same or state the handicap.

## A. Estimation accuracy (patient-level τ)

| n | DGP | CISCaRL MAE | SCRE MAE | CISCaRL bias | SCRE bias | CISCaRL sign | SCRE sign |
|---|---|---|---|---|---|---|---|
| 589 | all-beneficial | 0.130 | **0.107** | +0.016 | −0.105 | **79 %** | 72 % |
| 589 | sign-flip | 0.116 | **0.040** | −0.010 | +0.007 | 65 % | **88 %** |
| 2000 | all-beneficial | **0.098** | 0.108 | +0.015 | −0.108 | **93 %** | 58 % |
| 2000 | sign-flip | 0.086 | **0.025** | −0.017 | +0.003 | 58 % | **93 %** |

**SCRE wins estimation in 3 of 4 settings — but the pattern is informative, not random.**

SCRE's bias splits cleanly by DGP:

- **Sign-flip, SCRE bias ≈ 0.00** (0.007 and 0.003). The true effect there is
  `0.60 × cd40_z`, i.e. exactly a linear covariate × treatment interaction. That is in
  SCRE's design matrix, so SCRE is almost correctly specified and reaches MAE 0.025.
- **All-beneficial, SCRE bias ≈ −0.11.** Effects are small (0.08–0.17) and all positive,
  so shrinkage attenuates the estimate toward zero, and roughly 40 % of predictions flip
  sign. CISCaRL has near-constant bias (+0.015) and therefore better sign accuracy.

**CISCaRL has near-zero bias in every setting** (+0.016, −0.010, +0.015, −0.017) but
higher MAE whenever the truth is a smooth linear surface, because a piecewise-constant
rule list cannot represent it.

## B. Decision reliability

| n | DGP | CISCaRL coverage | CISCaRL precision | SCRE coverage | SCRE sign accuracy |
|---|---|---|---|---|---|
| 589 | all-beneficial | 26.5 % | **98.4 %** | 100 % | 72.4 % |
| 589 | sign-flip | 12.8 % | 88.4 % | 100 % | 87.6 % |
| 2000 | all-beneficial | 78.4 % | **96.4 %** | 100 % | 57.9 % |
| 2000 | sign-flip | 23.1 % | 49.8 % | 100 % | **92.6 %** |

SCRE commits for every patient by construction — it has no interval, so it cannot
abstain. Two settings favour CISCaRL decisively (uniform small effects, where SCRE
attenuates), one favours SCRE decisively (linear effects, where SCRE is near-oracle),
and one is a tie on precision at very different coverage.

## The fairness problem with this gate

**Both DGPs are linear in the covariates.** The all-beneficial DGP uses
`τ = τ₀ + τ₁·age_z`; the sign-flip DGP uses `τ = 0.60·cd40_z`. Both therefore lie in
SCRE's model class (linear × treatment terms) and outside CISCaRL's (piecewise-constant
threshold rules).

So this gate is structurally tilted *toward SCRE*, and its result — "SCRE wins
estimation in 3 of 4 settings" — cannot be read as a general statement. The sign-flip
MAE of 0.025 is close to what an oracle would achieve, because the truth is nearly in
SCRE's span.

**A fair verdict needs a third DGP whose effect is genuinely piecewise-constant**, e.g.

$$\tau(x) = \begin{cases} 0.20 & \text{CD4} < 200 \wedge \text{age} > 40 \\ 0 & \text{otherwise}\end{cases}$$

which is in CISCaRL's model class and outside SCRE's. Without it, Gate 4 answers "which
model class matches a linear truth", which we already knew.

## What can be claimed now

1. **SCRE is a strong estimator when the treatment effect is linear in covariates.**
   Near-unbiased, MAE 0.025 at n = 2000. This should be stated plainly.
2. **SCRE attenuates small, uniform effects toward zero** (−0.11 bias) and consequently
   mis-signs ~40 % of patients. That is a genuine, reproducible weakness of the
   penalised shared-basis fit, and it is what CISCaRL's abstention compensates for.
3. **CISCaRL is nearly unbiased in every setting** but has higher MAE whenever the truth
   is smooth, because of its functional form.
4. **SCRE cannot abstain**, so "precision" is undefined for it — the honest comparison is
   coverage-versus-precision against always-commit.
5. **No universal winner.** Any claim of the form "CISCaRL beats SCRE" is false in this
   gate's sign-flip cells.

## What Gate 4 must add before the paper uses it

| item | why |
|---|---|
| **Piecewise-constant DGP** | the only setting where CISCaRL's model class matches the truth |
| **Non-PH DGP** (crossing hazards) | Gate 1c suggests SCRE's structure is biased here; untested against real SCRE |
| **Bootstrap intervals for SCRE** | makes the decision comparison like-for-like; without it SCRE cannot commit/abstain fairly |
| **Clustered uncertainty** | rules within a replication are dependent; current CIs are rule-level |

## Bottom line

Gate 4 does not support "CISCaRL outperforms SCRE." It supports a narrower and more
useful statement: **SCRE is the better estimator when the effect surface is linear, and
CISCaRL is the better *decision* device when effects are small and uniform — at the cost
of covering far fewer patients.** Whether either statement survives a DGP matched to
CISCaRL's model class is now the open question.

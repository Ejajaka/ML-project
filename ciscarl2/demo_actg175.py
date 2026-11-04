"""Demo: what CISCaRL v1.3 output actually looks like on ACTG175.

Discovery on the training partition, honest evaluation on the held-out test
partition, plus pre-specified subgroup estimates on the full cohort.

Run:  python -m ciscarl2.demo_actg175
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .estimators import (crossfit_nuisances, ipcw_outcome, phi_on_holdout,
                         truncate_weights)
from .rules import dedup, eval_rule, prune, rules_from_ensemble, stability_scores

from .paths import data_path

DATA = data_path()
FEATS = ["cd40", "cd80", "age", "wtkg", "karnof", "hemo",
         "homo", "race", "gender", "drugs", "str2", "symptom"]
TSTAR = 730.0
THR, MIN_SUPPORT, MAX_RULES = 0.50, 60, 99   # v1.4: no rule-count cap
SEED = 2024


def load():
    df = pd.read_csv(DATA)
    df = df[df.arms.isin([0, 1])].copy().reset_index(drop=True)
    X = df[FEATS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    return dict(df=df, X=(X - mu) / sd, mu=mu, sd=sd,
                A=(df.arms == 1).to_numpy(int),
                T=df.days.to_numpy(float), E=df.cens.to_numpy(int))


def verdict(lo, hi):
    if lo > 0:
        return "TREAT"
    if hi < 0:
        return "AVOID"
    return "INSUFFICIENT EVIDENCE"


def bar(title):
    print(f"\n{title}\n" + "=" * len(title))


def main():
    d = load()
    X, A, T, E = d["X"], d["A"], d["T"], d["E"]

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=A)

    bar("CISCaRL v1.3  |  ACTG175 arms 0 vs 1")
    print("Endpoint : progression-free (CD4 decline >=50, AIDS progression, or death)")
    print("Horizon  : 730 days")
    print("Treatment: 0 = zidovudine, 1 = zidovudine + didanosine")
    print(f"Discovery: train n={len(tr)}   Evaluation: test n={len(te)} (held out)")

    # ---------- overall contrast ----------
    p1, p0 = phi_on_holdout(X[tr], A[tr], T[tr], E[tr],
                            X[te], A[te], T[te], TSTAR, seed=SEED)
    dphi = p1 - p0
    tau = float(dphi.mean())
    se = float(np.sqrt(((dphi - tau) ** 2).sum()) / len(te))
    q = float(tdist.ppf(0.975, len(te) - 1))
    lo, hi = tau - q * se, tau + q * se

    bar("OVERALL (all test patients, no subgroup)")
    print(f"  tau(730) = {tau:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"width {hi-lo:.3f}   -> {verdict(lo, hi)}")
    print(f"  n = {len(te)}, SE = {se:.3f}")

    # ---------- discovery on train ----------
    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        X[tr], A[tr], T[tr], E[tr], TSTAR, seed=SEED, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    Y = ipcw_outcome(T[tr], TSTAR, GD)
    mu_a = np.where(A[tr] == 1, muD[:, 1], muD[:, 0])
    psi = (A[tr] - 0.5) / 0.25 * (Y - mu_a) + (muD[:, 1] - muD[:, 0])

    rf = RandomForestRegressor(n_estimators=30, max_depth=3,
                               min_samples_leaf=MIN_SUPPORT,
                               random_state=0, n_jobs=-1).fit(X[tr], psi)
    cands = dedup(rules_from_ensemble(rf, X[tr], MIN_SUPPORT, max_trees=30))
    rng = np.random.RandomState(SEED)
    subs = []
    for _ in range(20):
        ii = rng.choice(len(tr), len(tr) // 2, replace=False)
        t = RandomForestRegressor(n_estimators=15, max_depth=3,
                                  min_samples_leaf=max(10, MIN_SUPPORT // 2),
                                  random_state=0, n_jobs=-1).fit(X[tr][ii], psi[ii])
        subs.append(rules_from_ensemble(t, X[tr], 5, max_trees=15))
    freq = stability_scores(cands, subs, X[tr]) if cands else np.array([])
    sel = [dict(cond=c, mask=m, score=f)
           for (c, m), f in zip(cands, freq) if f >= THR]
    sel = prune(sel, X[tr], max_rules=MAX_RULES, overlap=0.8) if sel else []

    bar(f"RULE LIST  (discovered on train, evaluated on {len(te)} test patients)")
    if not sel:
        print("  no rule passed the stability threshold")
    else:
        print(f"  {'#':<3}{'rule':<44}{'n':>4}{'tau':>8}{'95% CI':>20}{'stab':>7}  verdict")
        print("  " + "-" * 96)
        for j, r in enumerate(sel, 1):
            m = eval_rule(r["cond"], X[te])
            nr = int(m.sum())
            if nr < 5:
                continue
            t_r = float(dphi[m].mean())
            s_r = float(np.sqrt(((dphi[m] - t_r) ** 2).sum()) / nr)
            qr = float(tdist.ppf(0.975, max(nr - 1, 1)))
            lo_r, hi_r = t_r - qr * s_r, t_r + qr * s_r
            mu_, sd_ = d["mu"], d["sd"]
            txt = " & ".join(
                f"{FEATS[f]} {op} {thr * sd_[f] + mu_[f]:.1f}"
                for f, op, thr in r["cond"])
            print(f"  {j:<3}{txt:<44}{nr:>4}{t_r:>+8.3f}"
                  f"   [{lo_r:+.3f}, {hi_r:+.3f}]{r['score']:>7.2f}  {verdict(lo_r, hi_r)}")

    # ---------- pre-specified subgroups ----------
    bar("PRE-SPECIFIED SUBGROUPS  (full cohort, no discovery, no multiplicity)")
    j = {f: i for i, f in enumerate(FEATS)}
    mu, sd = d["mu"], d["sd"]
    z = lambda f, v: (v - mu[j[f]]) / sd[j[f]]
    pre = {
        "CD4 < 200": X[:, j["cd40"]] < z("cd40", 200),
        "CD4 >= 200": X[:, j["cd40"]] >= z("cd40", 200),
        "prior ART (str2=1)": X[:, j["str2"]] > 0,
        "ART-naive (str2=0)": X[:, j["str2"]] <= 0,
        "age > 35": X[:, j["age"]] > z("age", 35),
        "symptomatic": X[:, j["symptom"]] > 0,
    }
    p1f, p0f = phi_on_holdout(X, A, T, E, X, A, T, TSTAR, seed=SEED)
    dphif = p1f - p0f
    print(f"  {'subgroup':<22}{'n':>5}{'tau':>8}{'95% CI':>20}{'half-width':>12}  verdict")
    print("  " + "-" * 80)
    for name, m in pre.items():
        nr = int(m.sum())
        if nr < 30:
            continue
        t_r = float(dphif[m].mean())
        s_r = float(np.sqrt(((dphif[m] - t_r) ** 2).sum()) / nr)
        qr = float(tdist.ppf(0.975, max(nr - 1, 1)))
        lo_r, hi_r = t_r - qr * s_r, t_r + qr * s_r
        print(f"  {name:<22}{nr:>5}{t_r:>+8.3f}   [{lo_r:+.3f}, {hi_r:+.3f}]"
              f"{qr*s_r:>12.3f}  {verdict(lo_r, hi_r)}")

    print("\nNote: endpoint is a composite (CD4 decline >=50 counted), follow-up appears")
    print("to end at treatment discontinuation, and intervals are ~93% not 95%.")


if __name__ == "__main__":
    main()

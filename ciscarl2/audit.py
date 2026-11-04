"""Implementation audit - core utilities and statistical estimators.

Run:  python -m ciscarl2.audit
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from . import dgp
from .estimators import aipw, crossfit_nuisances, truncate_weights
from .gate4d import EFFECT, TSTAR, _delta_for
from .gate11 import _cart
from .gate14 import _calibrate, _config, _region, _simulate
from .gate19 import _assign_effs, _contrast, _leaf_table
from .rules import eval_rule, extract_leaf_rules

FAILS = []


def _report(name, ok, detail=""):
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def check_rules(pool):
    match_ok = part_ok = True
    worst = 0
    for seed in range(10):
        rng = np.random.RandomState(seed)
        n = 3000
        X = pool[rng.choice(len(pool), n, replace=True)]
        y = rng.normal(size=n)
        t = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20,
                                  random_state=0).fit(X, y)
        leaves = extract_leaf_rules(t, X, 1)
        for c, m in leaves:
            if not (eval_rule(c, X) == m).all():
                match_ok = False
                worst = max(worst, int(abs(eval_rule(c, X).sum() - m.sum())))
        if sum(int(m.sum()) for c, m in leaves) != n:
            part_ok = False
    _report("extract_leaf_rules: condition == mask", match_ok, f"max diff {worst}")
    _report("extract_leaf_rules: leaves partition the data", part_ok)


def check_assign(pool):
    rng = np.random.RandomState(7)
    n = 3000
    X = pool[rng.choice(len(pool), n, replace=True)]
    target = rng.normal(size=n)
    phi = rng.normal(size=n)
    _, rules = _cart(target, X, 3)
    tbl = _leaf_table(rules, X, phi)
    effs = np.array([t["eff"] for t in tbl])
    default = -999.0
    pred = _assign_effs(tbl, X, effs, default)
    ok = True
    for i in range(n):
        hit = [(j, t) for j, t in enumerate(tbl) if t["mask"][i]]
        if not hit:
            ok &= pred[i] == default
        else:
            # most specific matching rule wins
            j = max(hit, key=lambda kv: len(kv[1]["cond"]))[0]
            ok &= np.isclose(pred[i], effs[j])
    _report("assign: most-specific rule wins, default elsewhere", ok)


def check_contrast():
    rng = np.random.RandomState(3)
    n = 2400
    phi = rng.normal(size=n)
    mask = rng.rand(n) < 0.2
    D, se = _contrast(phi, mask)
    D2 = phi[mask].mean() - phi.mean()
    p = mask.mean()
    psi = phi * (np.where(mask, 1.0 / p, 0.0) - 1.0)
    se2 = np.sqrt(((psi - psi.mean()) ** 2).sum()) / n
    _report("paired IF contrast D and SE",
            np.isclose(D, D2) and np.isclose(se, se2),
            f"D={D:.6f} vs {D2:.6f}")


def check_aipw(pool, mu, sd, i_cd4, i_age, cd4_cut, age_cut, lam0, lam_c0,
               reps=10, n=2000):
    cfg = (pool, mu, sd, i_cd4, i_age, cd4_cut, age_cut, lam0)
    biases, covers, s1err, s0err = [], [], [], []
    for rep in range(reps):
        rng = np.random.RandomState(9000 + rep)
        X = pool[rng.choice(len(pool), n, replace=True)]
        R = (X[:, i_cd4] < (cd4_cut - mu[i_cd4]) / sd[i_cd4]) & \
            (X[:, i_age] > (age_cut - mu[i_age]) / sd[i_age])
        lp0 = X @ dgp.BETA
        s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
        eff = np.where(R, _delta_for(EFFECT, s0), 0.0)
        A = rng.binomial(1, 0.5, n)
        sim = _simulate(X, A, lam_c0, lam0, rng, eff, R, 0.0, cfg)
        T, E = sim["Tobs"], sim["E"]
        tau_i = sim["S1"] - sim["S0"]
        muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
            X, A, T, E, TSTAR, seed=rep, return_folds=True)
        GD, _ = truncate_weights(GD, fid, fcaps)
        tau, se, phi1, phi0 = aipw(X, A, T, TSTAR, muD, GD)
        biases.append(tau - tau_i.mean())
        covers.append(abs(tau - tau_i.mean()) <= 1.96 * se)
        s1err.append(phi1.mean() - sim["S1"].mean())
        s0err.append(phi0.mean() - sim["S0"].mean())
    print(f"      AIPW bias {np.mean(biases):+.4f} (sd {np.std(biases):.4f}), "
          f"IF-CI coverage {100*np.mean(covers):.0f}%")
    print(f"      arm means: phi1 - S1 {np.mean(s1err):+.4f}, "
          f"phi0 - S0 {np.mean(s0err):+.4f}")
    _report("AIPW bias < 0.01", abs(np.mean(biases)) < 0.01)
    _report("arm means within 0.01", abs(np.mean(s1err)) < 0.01 and
            abs(np.mean(s0err)) < 0.01)


def check_metrics():
    region = np.zeros(100, bool)
    region[:25] = True
    leaf = np.zeros(100, bool)
    leaf[:20] = True          # 20 in leaf, all region
    purity = (leaf & region).sum() / leaf.sum()
    capture = (leaf & region).sum() / region.sum()
    jac = (leaf & region).sum() / (leaf | region).sum()
    _report("purity/capture/Jaccard definitions",
            np.isclose(purity, 1.0) and np.isclose(capture, 0.8) and
            np.isclose(jac, 20 / 25))


def main():
    cfg = _config()
    pool, mu, sd, i_cd4, i_age, cd4_cut, age_cut, lam0 = cfg
    lam_c0 = _calibrate(cfg, 0.0)
    print("=== CISCaRL implementation audit ===\n")
    check_rules(pool)
    check_assign(pool)
    check_contrast()
    check_metrics()
    check_aipw(pool, mu, sd, i_cd4, i_age, cd4_cut, age_cut, lam0, lam_c0)
    print(f"\n{len(FAILS)} failing check(s)" if FAILS else "\nall core checks passed")
    return FAILS


if __name__ == "__main__":
    main()

"""Gate 11 - does a depth-2 rule model beat the marginal ATE as n grows?

Pre-registration: `results/gate11_PREREGISTRATION.md`, frozen before this ran.

    n in {4000, 8000, 12000}, 40 fresh seeds per cell
    target   cross-fitted Cox plug-in tau_hat (TRAIN only)
    primary  CART depth 2, min_samples_leaf 60
    effects  held-out AIPW phi (TRAIN nuisances)
    default  TRAIN marginal ATE
    decide   mean delta < 0, 95 % CI entirely below 0, >= 70 % reps better

Purity, capture, leaf effect/SE, Jaccard and split stability are reported as
diagnostics; they are not used to modify the primary model.

Usage
-----
    python -m ciscarl2.gate11 --reps 2 --n 4000 --start 120900   # smoke, off-seed
    python -m ciscarl2.gate11 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from . import dgp
from .estimators import (aipw, crossfit_nuisances, phi_on_holdout,
                         truncate_weights)
from .gate2 import TAU, pseudo_outcome
from .gate4d import EFFECT, MIN_SUPPORT, TSTAR, _delta_for
from .gate8 import assign, best_jaccard
from .rules import eval_rule, extract_leaf_rules

from .paths import data_path

DATA = data_path()
CELLS = ((4000, 120100), (8000, 120200), (12000, 120300))
STAB_B = 20


def _cart(target, XD, depth):
    t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=MIN_SUPPORT,
                              random_state=0).fit(XD, target)
    return t, [dict(cond=c, mask=m, score=1.0)
               for c, m in extract_leaf_rules(t, XD, MIN_SUPPORT)]


def _hi_leaf(leaves, target, XD):
    """The rule whose TRAIN mean predicted effect is largest."""
    best, best_val = None, -np.inf
    for r in leaves:
        m = r["mask"]
        if m.sum() < MIN_SUPPORT:
            continue
        v = float(target[m].mean())
        if v > best_val:
            best, best_val = r, v
    return best


def _leaf_stats(cond, XE, regE, dphi):
    m = eval_rule(cond, XE)
    nr = int(m.sum())
    if nr == 0:
        return dict(share=0.0, purity=np.nan, capture=np.nan,
                    effect=np.nan, se=np.nan, jaccard=0.0)
    eff = float(dphi[m].mean())
    se = float(np.sqrt(((dphi[m] - eff) ** 2).sum()) / nr)
    return dict(
        share=float(m.mean()),
        purity=float((m & regE).sum() / nr),
        capture=float((m & regE).sum() / max(regE.sum(), 1)),
        effect=eff, se=se,
        jaccard=float((m & regE).sum() / max((m | regE).sum(), 1)))


def _split_stability(target, XD, XE, hi_cond, depth, seed, B=STAB_B):
    rng = np.random.RandomState(seed)
    ref = eval_rule(hi_cond, XE)
    agree = []
    for _ in range(B):
        idx = rng.choice(len(XD), len(XD) // 2, replace=False)
        t, leaves = _cart(target[idx], XD[idx], depth)
        best, best_val = None, -np.inf
        for r in leaves:
            mm = r["mask"]                    # mask over the subsample
            if mm.sum() < 5:
                continue
            v = float(target[idx][mm].mean())
            if v > best_val:
                best, best_val = r, v
        if best is None:
            agree.append(0.0)
            continue
        m1 = eval_rule(best["cond"], XE)
        u = (m1 | ref).sum()
        agree.append(float((m1 & ref).sum() / u) if u else 0.0)
    return float(np.mean(agree))


def run_rep(rep, n, start, cfg):
    (pool, mu_, sd_, i_cd4, i_age, cd4_cut, age_cut, lam0, lam_c0) = cfg
    seed = start + rep
    rng = np.random.RandomState(seed)
    X = pool[rng.choice(len(pool), n, replace=True)]
    reg = (X[:, i_cd4] < (cd4_cut - mu_[i_cd4]) / sd_[i_cd4]) & \
          (X[:, i_age] > (age_cut - mu_[i_age]) / sd_[i_age])
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    sim = dgp.generate(X, "covariate", lam_c0, lam0, rng, TSTAR,
                       effect_vec=np.where(reg, _delta_for(EFFECT, s0), 0.0))
    A, T, E = sim["A"], sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=seed, stratify=A)
    XD, XE = X[tr], X[te]
    regE, truth = reg[te], tau_i[te]

    muD, GD, _, fid, fcaps, _ = crossfit_nuisances(
        XD, A[tr], T[tr], E[tr], TSTAR, seed=seed, return_folds=True)
    GD, _ = truncate_weights(GD, fid, fcaps)
    _, _, q1, q0 = aipw(XD, A[tr], T[tr], TSTAR, muD, GD)
    const = float((q1 - q0).mean())
    p1, p0 = phi_on_holdout(XD, A[tr], T[tr], E[tr],
                            XE, A[te], T[te], TSTAR, seed=seed)
    dphi = p1 - p0
    tau_hat = muD[:, 1] - muD[:, 0]

    const_mae = float(np.abs(const - truth).mean())
    rows = [dict(rep=rep, n=n, model="constant", depth=0, rules=0,
                 mae=const_mae, const_mae=const_mae, delta=0.0)]

    for label, depth in (("cart_d2", 2), ("cart_d3", 3)):
        tree, leaves = _cart(tau_hat, XD, depth)
        hat = assign(leaves, XE, dphi, const)
        mae = float(np.abs(hat - truth).mean())
        hi = _hi_leaf(leaves, tau_hat, XD)
        st = _leaf_stats(hi["cond"], XE, regE, dphi) if hi else {}
        row = dict(rep=rep, n=n, model=label, depth=depth, rules=len(leaves),
                   mae=mae, const_mae=const_mae, delta=mae - const_mae,
                   region_share=100.0 * float(regE.mean()),
                   best_jaccard=best_jaccard(leaves, XE, regE),
                   hi_share=st.get("share", np.nan),
                   hi_purity=st.get("purity", np.nan),
                   hi_capture=st.get("capture", np.nan),
                   hi_effect=st.get("effect", np.nan),
                   hi_se=st.get("se", np.nan),
                   hi_jaccard=st.get("jaccard", np.nan),
                   stability=_split_stability(tau_hat, XD, XE, hi["cond"],
                                              depth, seed + 7) if hi else np.nan)
        rows.append(row)
    return rows


def _boot(d, seed=0, B=5000):
    rs = np.random.RandomState(seed)
    b = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(df):
    out = []
    for (n, model), g in df.groupby(["n", "model"]):
        d = g.delta.to_numpy()
        lo, hi = _boot(d)
        out.append(dict(
            n=n, model=model, reps=len(g), rules=g.rules.mean(),
            mae=g.mae.mean(), const_mae=g.const_mae.mean(), delta=d.mean(),
            lo=lo, hi=hi, frac_better=100.0 * float((d < 0).mean()),
            rel_gain=100.0 * (1.0 - g.mae.mean() / g.const_mae.mean()),
            best_jaccard=g.best_jaccard.mean(),
            hi_share=g.hi_share.mean(), hi_purity=g.hi_purity.mean(),
            hi_capture=g.hi_capture.mean(), hi_effect=g.hi_effect.mean(),
            hi_se=g.hi_se.mean(), hi_jaccard=g.hi_jaccard.mean(),
            stability=g.stability.mean()))
    return pd.DataFrame(out)


def decide(res):
    out = {}
    for n, g in res[res.model == "cart_d2"].groupby("n"):
        r = g.iloc[0]
        out[int(n)] = {
            "mean delta < 0": bool(r.delta < 0),
            "95% CI entirely below 0": bool(r.hi < 0),
            ">= 70% of reps better": bool(r.frac_better >= 70),
        }
    return out


def _config():
    pool = dgp.load_covariate_pool(DATA)
    raw = pd.read_csv(DATA)
    raw = raw[raw.arms.isin([0, 1])]
    Xr = raw[dgp.COVARIATES].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    cd4_cut, age_cut = float(raw["cd40"].median()), float(raw["age"].median())
    i_cd4 = dgp.COVARIATES.index("cd40")
    i_age = dgp.COVARIATES.index("age")
    lam0 = dgp.lambda0_for_target_survival(pool, target=0.735)
    lam_c0 = dgp.calibrate_lam_c("covariate", 0.10, pool, lam0, TSTAR,
                                 gen_kwargs={"tau": TAU})
    return (pool, mu, sd, i_cd4, i_age, cd4_cut, age_cut, lam0, lam_c0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n", type=int, nargs="+", default=None,
                    help="subset of sample sizes (default: all three cells)")
    ap.add_argument("--start", type=int, default=None,
                    help="override the frozen start seed (smoke test only)")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    cfg = _config()
    cells = [c for c in CELLS if args.n is None or c[0] in args.n]
    if args.start is not None:
        cells = [(n, args.start) for n, _ in cells]

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, n, start, cfg)
        for n, start in cells for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate11_replications.csv"), index=False)

    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate11_summary.csv"), index=False)
    checks = decide(res)

    print(f"\n=== Gate 11, {args.reps} reps per cell ===")
    print(f"{'n':>7}{'model':>9}{'rules':>7}{'MAE':>9}{'const':>9}{'delta':>10}"
          f"{'95% CI':>22}{'%bet':>6}{'gain':>7}{'purity':>8}{'capture':>9}"
          f"{'hi eff':>8}{'hi se':>7}{'jacc':>7}{'stab':>7}")
    for _, r in res.sort_values(["n", "model"]).iterrows():
        print(f"{r.n:>7.0f}{r.model:>9}{r.rules:>7.1f}{r.mae:>9.4f}"
              f"{r.const_mae:>9.4f}{r.delta:>+10.4f}"
              f"   [{r.lo:+.4f},{r.hi:+.4f}]{r.frac_better:>5.0f}%"
              f"{r.rel_gain:>+6.1f}%{r.hi_purity:>8.2f}{r.hi_capture:>9.2f}"
              f"{r.hi_effect:>+8.3f}{r.hi_se:>7.3f}{r.best_jaccard:>7.2f}"
              f"{r.stability:>7.2f}")

    print("\nHigh-leaf prevalence:", {
        int(n): round(float(g[g.model == "cart_d2"].hi_share.iloc[0]), 3)
        for n, g in res.groupby("n")})
    print("\nPre-registered decision checks (primary = cart_d2):")
    adopted = []
    for n, ch in checks.items():
        print(f"  n={n}:")
        for k, v in ch.items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if all(ch.values()):
            adopted.append(n)
    if adopted:
        print(f"\n-> ADOPT depth-2 rule model at n = {adopted}")
    else:
        print("\n-> No cell adopted; report the loss with its purity/capture "
              "diagnostics and do not change the model.")
    return df


if __name__ == "__main__":
    main()

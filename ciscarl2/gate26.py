"""Gate 26 - flexible DR-learner + rule distillation vs C200 (stage 1 of v2).

Pre-registration: `results/gate26_PREREGISTRATION.md`.

C200 : depth-3 CART on psi (min_leaf 60), max mean with support >=200.
FLEX : 5-fold HistGradientBoosting on psi, distilled to a depth-3 CART
       (min_leaf 200), max distilled-leaf mean.

Both use validation AIPW effects and the one-sided commitment rule.

Usage
-----
    python -m ciscarl2.gate26 --reps 2 --start 290900   # smoke, off-seed
    python -m ciscarl2.gate26 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

from . import dgp
from .estimators import aipw, crossfit_nuisances, phi_on_holdout
from .gate2 import pseudo_outcome
from .gate4d import TSTAR, _delta_for
from .gate11 import _cart
from .gate19 import _assign_effs, _leaf_table
from .gate20 import _capped_G
from .gate22 import _stats
from .gate23 import E1, E2, _config23, _regions
from .rules import eval_rule, extract_leaf_rules, jaccard

from .paths import data_path

DATA = data_path()
START = 290100
KINDS = ("single", "two")
N_EVAL = 20
MIN_LEAF_C200 = 60
MIN_LEAF_FLEX = 200
SUP = 200


def _regions_kind(X, cfg, kind):
    R1, R2 = _regions(X, cfg)
    if kind == "single":
        R2 = np.zeros(len(X), bool)
    return R1, R2


def _sim_kind(X, A, lam_c0, lam0, rng, e1, e2):
    lp0 = X @ dgp.BETA
    lp1 = lp0 - np.maximum(e1, e2)
    lam = lam0 * np.exp(np.where(A == 1, lp1, lp0))
    T = rng.exponential(1.0 / np.maximum(lam, 1e-12))
    C = rng.exponential(1.0 / np.maximum(lam_c0 * np.exp(X @ dgp.GAMMA), 1e-12))
    Tobs = np.minimum(T, C)
    E = (T <= C).astype(int)
    S0 = np.exp(-lam0 * np.exp(lp0) * TSTAR)
    S1 = np.exp(-lam0 * np.exp(lp1) * TSTAR)
    return dict(Tobs=Tobs, E=E, S0=S0, S1=S1)


def _calibrate_kind(cfg, kind, target=0.10, n_pilot=6000, seed=99):
    pool, mu, sd, cuts, idx, lam0 = cfg
    rng0 = np.random.RandomState(seed)
    Xp = pool[rng0.choice(len(pool), n_pilot, replace=True)]
    R1, R2 = _regions_kind(Xp, cfg, kind)
    lp0 = Xp @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    e1 = np.where(R1, _delta_for(E1, s0), 0.0)
    e2 = np.where(R2, _delta_for(E2, s0), 0.0)

    def frac(lc):
        rng = np.random.RandomState(seed)
        A = rng.binomial(1, 0.5, len(Xp))
        d = _sim_kind(Xp, A, lc, lam0, rng, e1, e2)
        return float(np.mean((d["Tobs"] <= TSTAR) & (d["E"] == 0)))

    lo, hi = 1e-7, 5e-2
    for _ in range(45):
        mid = float(np.sqrt(lo * hi))
        if frac(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _flex_fit(psi, X, seed):
    return HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.05,
        min_samples_leaf=60, l2_regularization=1.0,
        random_state=seed).fit(X, psi)


def _flex_oof(psi, X, seed):
    oof = np.zeros(len(X))
    for fit, hold in KFold(5, shuffle=True, random_state=seed).split(X):
        m = _flex_fit(psi[fit], X[fit], seed)
        oof[hold] = m.predict(X[hold])
    return oof


def _distill(tau, X, min_leaf=MIN_LEAF_FLEX, depth=3):
    t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf,
                              random_state=0).fit(X, tau)
    return [dict(cond=c, mask=m) for c, m in
            extract_leaf_rules(t, X, min_leaf)]


def run_rep(rep, kind, cfg, lam_c0):
    pool, mu_, sd_, cuts, idx, lam0 = cfg
    seed = START + rep
    rng = np.random.RandomState(seed)
    n = 8000
    X = pool[rng.choice(len(pool), n, replace=True)]
    R1, R2 = _regions_kind(X, cfg, kind)
    union = R1 | R2
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    e1 = np.where(R1, _delta_for(E1, s0), 0.0)
    e2 = np.where(R2, _delta_for(E2, s0), 0.0)
    A = rng.binomial(1, 0.5, n)
    sim = _sim_kind(X, A, lam_c0, lam0, rng, e1, e2)
    T, E = sim["Tobs"], sim["E"]
    tau_i = sim["S1"] - sim["S0"]

    ii = np.arange(n)
    disc, hold = train_test_split(ii, test_size=0.5, random_state=seed,
                                  stratify=A)
    val, asm = train_test_split(hold, test_size=0.5, random_state=seed + 1,
                                stratify=A[hold])
    XD, AD, TD, ED = X[disc], A[disc], T[disc], E[disc]
    XV, AV, TV, EV = X[val], A[val], T[val], E[val]
    XA, AA, TA, EA = X[asm], A[asm], T[asm], E[asm]
    uniA, truth = union[asm], tau_i[asm]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    G = _capped_G(muD, GD, Gmarg, fid, fcaps, "conditional", TD)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
    _, _, q1, q0 = aipw(XD, AD, TD, TSTAR, muD, G)
    const = float((q1 - q0).mean())
    const_mae = float(np.abs(const - truth).mean())

    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=seed)
    dphi_v = p1v - p0v
    p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, TSTAR, seed=seed)
    dphi_a = p1a - p0a

    # C200
    leaves_c = _cart(psi, XD, 3)[1]
    mc, sc, uc = _stats(psi, leaves_c)
    ok = uc >= SUP
    jc = int(np.nanargmax(np.where(ok, mc, -np.inf))) if ok.any() \
        else int(np.nanargmax(mc))
    # FLEX
    tau_oof = _flex_oof(psi, XD, seed)
    leaves_f = _distill(tau_oof, XD)
    mf, sf, uf = _stats(tau_oof, leaves_f)
    jf = int(np.nanargmax(np.where(uf >= SUP, mf, -np.inf))) \
        if (uf >= SUP).any() else int(np.nanargmax(mf))

    rng_eval = np.random.RandomState(seed + 41)
    evals = [rng_eval.choice(len(XD), len(XD) // 2, replace=False)
             for _ in range(N_EVAL)]

    rows = []
    for arm, leaves, j, target in (
            ("C200", leaves_c, jc, psi),
            ("FLEX", leaves_f, jf, tau_oof)):
        masks_a = [eval_rule(r["cond"], XA) for r in leaves]
        sel_mask = masks_a[j]
        k = int(sel_mask.sum())
        tbl_v = _leaf_table(leaves, XV, dphi_v)
        effs = np.array([t["eff"] for t in tbl_v], float)
        ses = np.array([t["se"] for t in tbl_v], float)
        effs = np.where(np.isfinite(effs), effs, const)
        mv = eval_rule(leaves[j]["cond"], XV)
        kv = int(mv.sum())
        hi_eff = float(dphi_v[mv].mean()) if kv else np.nan
        hi_se = (float(np.sqrt(((dphi_v[mv] - hi_eff) ** 2).sum()) / kv)
                 if kv else np.nan)
        committed = bool(np.isfinite(hi_eff) and np.isfinite(hi_se) and
                         hi_eff > norm.ppf(0.95) * hi_se)
        use = effs.copy()
        if not committed:
            use[j] = const
        tbl = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                    eff=use[t], se=ses[t]) for t in range(len(leaves))]
        pred = _assign_effs(tbl, XA, use, const)
        mae = float(np.abs(pred - truth).mean())

        # stability: refit the arm's own procedure on disjoint halves
        jfs = []
        for s_idx in evals:
            if arm == "C200":
                lb = _cart(psi[s_idx], XD[s_idx], 3)[1]
                mb, sb, ub = _stats(psi[s_idx], lb)
                need = max(int(round(SUP * len(s_idx) / 4000.0)), 5)
                okb = ub >= need
                jb = int(np.nanargmax(np.where(okb, mb, -np.inf))) if okb.any() \
                    else int(np.nanargmax(mb))
            else:
                # single flexible fit on the half-sample (in-sample surface)
                tau_b = _flex_fit(psi[s_idx], XD[s_idx], seed + 1000).predict(
                    XD[s_idx])
                lb = _distill(tau_b, XD[s_idx],
                              min_leaf=max(MIN_LEAF_FLEX // 2, 20))
                mb, sb, ub = _stats(tau_b, lb)
                need = max(int(round(SUP * len(s_idx) / 4000.0)), 5)
                okb = ub >= need
                jb = int(np.nanargmax(np.where(okb, mb, -np.inf))) if okb.any() \
                    else int(np.nanargmax(mb))
            jfs.append(jaccard(eval_rule(lb[jb]["cond"], XA), sel_mask))
        jfs = np.array(jfs)

        per = float(truth[sel_mask].mean()) if (committed and k) else np.nan
        oracle = (float(np.sort(truth)[::-1][:k].mean()) if (committed and k)
                  else np.nan)
        rows.append(dict(
            rep=rep, kind=kind, arm=arm, support=float(uc[j]) if arm == "C200"
            else float(uf[j]),
            n_cond=int(len(leaves[j]["cond"])),
            capture_sel=float(truth[sel_mask].sum() / truth.sum()),
            purity=float((sel_mask & uniA).sum() / k) if k else np.nan,
            j_true=jaccard(sel_mask, uniA),
            jf_mean=jfs.mean(), freq_full=100.0 * float((jfs > 0.8).mean()),
            committed=committed, coverage=float(sel_mask.mean()) if committed else 0.0,
            per_treated=per, efficiency=(per / oracle if (committed and oracle)
                                          else np.nan),
            captured_commit=(float(truth[sel_mask].sum() / truth.sum())
                             if committed else 0.0),
            null=bool(committed and np.isfinite(per) and per <= 0.01),
            beneficial=bool(committed and np.isfinite(per) and per > 0),
            mae=mae, const_mae=const_mae, delta=mae - const_mae))
    return rows


def _boot(d, seed=0, B=5000):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan, np.nan
    rs = np.random.RandomState(seed)
    bs = np.array([rs.choice(d, len(d), replace=True).mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2.0 * min(float((bs <= 0).mean()), float((bs >= 0).mean()))
    return float(lo), float(hi), min(p, 1.0)


def summarise(df):
    out = []
    for (kind, arm), g in df.groupby(["kind", "arm"]):
        tr = g[g.committed]
        out.append(dict(
            kind=kind, arm=arm, reps=len(g), support=g.support.mean(),
            n_cond=g.n_cond.mean(), capture=g.capture_sel.mean(),
            purity=g.purity.mean(), j_true=g.j_true.mean(),
            jf_mean=g.jf_mean.mean(), freq_full=g.freq_full.mean(),
            commit=100.0 * g.committed.mean(), coverage=g.coverage.mean(),
            per_treated=tr.per_treated.mean() if len(tr) else np.nan,
            efficiency=tr.efficiency.mean() if len(tr) else np.nan,
            captured_commit=g.captured_commit.mean(),
            null_rate=100.0 * tr.null.mean() if len(tr) else np.nan,
            mae=g.mae.mean(), delta=g.delta.mean()))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    global START
    if args.start is not None:
        START = args.start
    cfg = _config23()
    lam = {k: _calibrate_kind(cfg, k) for k in KINDS}
    for k in KINDS:
        print(f"  {k}: lam_c0={lam[k]:.3e}", flush=True)

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, k, cfg, lam[k])
        for k in KINDS for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate26_flex.csv"), index=False)
    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate26_summary.csv"), index=False)

    print(f"\n=== Gate 26, {args.reps} reps ===")
    print(f"{'kind':>7}{'arm':>6}{'support':>8}{'cond':>5}{'capture':>8}"
          f"{'purity':>7}{'J_true':>7}{'J_full':>7}{'commit':>7}{'cover':>7}"
          f"{'per-tr':>8}{'effic':>7}{'capC':>7}{'null%':>6}{'MAE':>8}")
    for _, r in res.sort_values(["kind", "arm"]).iterrows():
        print(f"{r.kind:>7}{r.arm:>6}{r.support:>8.0f}{r.n_cond:>5.1f}"
              f"{r.capture:>8.3f}{r.purity:>7.2f}{r.j_true:>7.3f}"
              f"{r.jf_mean:>7.3f}{r.commit:>6.0f}%{r.coverage:>7.3f}"
              f"{r.per_treated:>+8.3f}{r.efficiency:>7.2f}"
              f"{r.captured_commit:>7.3f}{r.null_rate:>6.0f}{r.mae:>8.4f}")

    print("\nPaired FLEX vs C200:")
    for kind in KINDS:
        g = df[df.kind == kind]
        a = g[g.arm == "C200"].set_index("rep")
        b = g[g.arm == "FLEX"].set_index("rep")
        common = sorted(set(a.index) & set(b.index))
        for metric in ("capture_sel", "j_true", "purity", "jf_mean",
                       "per_treated", "mae"):
            d = (b.loc[common, metric] - a.loc[common, metric]).to_numpy()
            lo, hi, p = _boot(d)
            print(f"  {kind:>6} {metric:>14}: {d.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] p={p:.3f}")

    print("\nPre-registered checks (FLEX vs C200):")
    all_ok = True
    for kind in KINDS:
        g = df[df.kind == kind]
        a = g[g.arm == "C200"].set_index("rep")
        b = g[g.arm == "FLEX"].set_index("rep")
        common = sorted(set(a.index) & set(b.index))
        dcap = (b.loc[common, "capture_sel"] - a.loc[common, "capture_sel"]).to_numpy()
        djt = (b.loc[common, "j_true"] - a.loc[common, "j_true"]).to_numpy()
        clo, chi, _ = _boot(dcap)
        jlo, jhi, _ = _boot(djt)
        c2 = res[(res.kind == kind) & (res.arm == "C200")].iloc[0]
        f2 = res[(res.kind == kind) & (res.arm == "FLEX")].iloc[0]
        checks = {
            "capture/J_true +0.05 with CI>0": bool(
                (dcap.mean() >= 0.05 and clo > 0) or
                (djt.mean() >= 0.05 and jlo > 0)),
            "purity >= C200-0.05": bool(f2.purity >= c2.purity - 0.05),
            "per-treated >= C200-0.02": bool(f2.per_treated >= c2.per_treated - 0.02),
            "null <= 5%": bool(f2.null_rate <= 5.0),
            "efficiency >= 0.70": bool(f2.efficiency >= 0.70),
            "MAE delta < 0": bool(f2.delta < 0),
            "MAE <= C200+0.005": bool(f2.mae <= c2.mae + 0.005),
        }
        print(f"  {kind}:")
        for k, v in checks.items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        all_ok &= all(checks.values())
    print(f"\n-> {'FLEX ADOPTED for stage 2' if all_ok else 'FLEX NOT adopted'}")
    return df


if __name__ == "__main__":
    main()

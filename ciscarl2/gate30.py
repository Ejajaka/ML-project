"""Gate 30 - knob sweep for C200 (pre-registration: gate30_PREREGISTRATION.md).

One nuisance fit per replication; every variant is then evaluated on the same
psi, validation phi and assessment phi. That makes the variants paired by
construction and the sweep cheap enough to run 40 seeds.

Usage
-----
    python -m ciscarl2.gate30 --mode sweep --reps 2 --start 310900   # smoke
    python -m ciscarl2.gate30 --mode sweep --reps 40 --jobs 10
    python -m ciscarl2.gate30 --mode smalln --reps 40 --jobs 10
    python -m ciscarl2.gate30 --mode confirm --reps 40 --start 320100
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from . import dgp
from .estimators import aipw, crossfit_nuisances, phi_on_holdout
from .gate2 import pseudo_outcome
from .gate4d import TSTAR, _delta_for
from .gate19 import _assign_effs, _leaf_table
from .gate20 import _capped_G
from .gate22 import _stats
from .gate23 import E1, E2, _boot, _config23, _regions
from .gate26 import _calibrate_kind, _sim_kind
from .rules import eval_rule, extract_leaf_rules

from .paths import data_path

DATA = data_path()
Z95 = float(norm.ppf(0.95))
Z90 = float(norm.ppf(0.90))
Z975 = float(norm.ppf(0.975))

# id -> dict(depth, leaf, floor, z, shrink)
VARIANTS = {
    "V0": dict(depth=3, leaf=60, floor=200, z=Z95, shrink=False),
    "V2": dict(depth=3, leaf=60, floor=200, z=Z95, shrink=True),
    "V3a": dict(depth=3, leaf=60, floor=200, z=Z90, shrink=False),
    "V3b": dict(depth=3, leaf=60, floor=200, z=Z975, shrink=False),
    "V4a": dict(depth=3, leaf=30, floor=200, z=Z95, shrink=False),
    "V4b": dict(depth=3, leaf=100, floor=200, z=Z95, shrink=False),
    "V5": dict(depth=4, leaf=60, floor=200, z=Z95, shrink=False),
}
SMALLN = {
    "abs200": dict(depth=3, leaf=60, floor=200, z=Z95, shrink=False),
    "prop5": dict(depth=3, leaf=60, floor=-1, z=Z95, shrink=False),  # -1 = 5 %
}
KINDS = ("single", "two")


def _regions_kind(X, cfg, kind):
    R1, R2 = _regions(X, cfg)
    if kind == "single":
        R2 = np.zeros(len(X), bool)
    return R1, R2


def _metrics(pred, treated, truth, region):
    treated = np.asarray(treated, bool)
    k = int(treated.sum())
    mae = float(np.abs(pred - truth).mean())
    sp = (float(spearmanr(pred, truth).correlation)
          if np.std(pred) > 0 and np.std(truth) > 0 else np.nan)
    cap = float(truth[treated].sum() / truth.sum()) if k else 0.0
    pur = float((treated & region).sum() / k) if k else np.nan
    per = float(truth[treated].mean()) if k else np.nan
    oracle = float(np.sort(truth)[::-1][:k].mean()) if k else np.nan
    eff = (per / oracle) if (k and oracle) else np.nan
    null = bool(k and per <= 0.01)
    return dict(mae=mae, spearman=sp, capture=cap, purity=pur,
                per_treated=per, efficiency=eff, null=null,
                coverage=float(treated.mean()))


def _fit_leaves(target, X, depth, leaf):
    t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf,
                              random_state=0).fit(X, target)
    return [dict(cond=c, mask=m, score=1.0)
            for c, m in extract_leaf_rules(t, X, leaf)]


def _shrinkage(effs, ses, ceiling=0.8):
    ok = np.isfinite(effs) & np.isfinite(ses)
    if ok.sum() < 2:
        return 0.0
    within = float(np.mean(ses[ok] ** 2))
    between = float(np.var(effs[ok], ddof=1))
    if within + between <= 0:
        return 0.0
    return float(np.clip(within / (within + between), 0.0, ceiling))


def run_rep(rep, kind, cfg, lam_c, variants, n, start):
    pool = cfg[0]
    lam0 = cfg[5]
    seed = start + rep
    rng = np.random.RandomState(seed)
    X = pool[rng.choice(len(pool), n, replace=True)]
    R1, R2 = _regions_kind(X, cfg, kind)
    union = R1 | R2
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    e1 = np.where(R1, _delta_for(E1, s0), 0.0)
    e2 = np.where(R2, _delta_for(E2, s0), 0.0)
    A = rng.binomial(1, 0.5, n)
    sim = _sim_kind(X, A, lam_c, lam0, rng, e1, e2)
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
    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=seed)
    dphi_v = p1v - p0v
    p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, TSTAR, seed=seed)
    dphi_a = p1a - p0a
    const = float(aipw(XD, AD, TD, TSTAR, muD, G)[0])

    rows = []
    for name, v in variants.items():
        leaves = _fit_leaves(psi, XD, v["depth"], v["leaf"])
        means, ses_d, sup = _stats(psi, leaves)
        floor = v["floor"]
        if floor < 0:
            floor = max(60, int(round(0.05 * len(XD))))
        ok = sup >= floor
        fallback = not bool(ok.any())
        j = (int(np.nanargmax(np.where(ok, means, -np.inf))) if ok.any()
             else int(np.nanargmax(means)))

        tbl_v = _leaf_table(leaves, XV, dphi_v)
        effs = np.array([t["eff"] for t in tbl_v], float)
        ses_v = np.array([t["se"] for t in tbl_v], float)
        effs_f = np.where(np.isfinite(effs), effs, const)

        shrink = _shrinkage(effs_f, ses_v) if v["shrink"] else 0.0
        use = (1.0 - shrink) * effs_f + shrink * const

        mv = eval_rule(leaves[j]["cond"], XV)
        kv = int(mv.sum())
        hi_eff = float(dphi_v[mv].mean()) if kv else np.nan
        hi_se = (float(np.sqrt(((dphi_v[mv] - hi_eff) ** 2).sum()) / kv)
                 if kv else np.nan)
        committed = bool(np.isfinite(hi_eff) and np.isfinite(hi_se)
                         and hi_eff > v["z"] * hi_se)

        use_j = use.copy()
        if not committed:
            use_j[j] = const

        masks_a = [eval_rule(r["cond"], XA) for r in leaves]
        tbl_a = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                      eff=use_j[t], se=ses_v[t]) for t in range(len(leaves))]
        pred = _assign_effs(tbl_a, XA, use_j, const)
        treated = masks_a[j] if committed else np.zeros(len(XA), bool)

        mc = _metrics(pred, treated, truth, uniA)
        rows.append(dict(rep=rep, kind=kind, variant=name, n=n,
                         floor=floor, support=int(sup[j]),
                         fallback=fallback, committed=committed,
                         shrink=shrink, z=v["z"], depth=v["depth"],
                         leaf=v["leaf"], **mc))
    return rows


def summarise(raw, baseline="V0"):
    out = []
    for kind in sorted(raw.kind.unique()):
        g = raw[raw.kind == kind]
        base = g[g.variant == baseline].set_index("rep")
        for name in sorted(g.variant.unique()):
            d = g[g.variant == name].set_index("rep")
            mae_d = (d.mae - base.mae).values
            lo, hi, p = _boot(mae_d)
            com = d[d.committed]
            com0 = base[base.committed]
            out.append(dict(
                kind=kind, variant=name, reps=len(d),
                mae=d.mae.mean(), mae_committed=com.mae.mean() if len(com) else np.nan,
                mae_committed_base=(com0.mae.mean() if len(com0) else np.nan),
                d_mae=float(np.mean(mae_d)), lo=lo, hi=hi, p=p,
                purity=d.purity.mean(), capture=d.capture.mean(),
                jaccard=np.nan, commit=100 * d.committed.mean(),
                coverage=d.coverage.mean(), per_treated=d.per_treated.mean(),
                efficiency=d.efficiency.mean(),
                null=100 * d.null.mean(),
                fallback=100 * d.fallback.mean(),
                support=d.support.mean()))
    return pd.DataFrame(out)


def _holm(df, col="p"):
    df = df.copy()
    order = df[col].sort_values().index
    m = len(order)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = min(1.0, (m - i) * df.loc[idx, col])
        running = max(running, val)
        adj[i] = running
    df["p_holm"] = np.nan
    df.loc[order, "p_holm"] = adj
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep",
                    choices=("sweep", "smalln", "confirm"))
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args(argv)

    cfg = _config23()
    if args.mode == "sweep":
        kinds, variants, n, start = KINDS, VARIANTS, 8000, 310100
        tag = "gate30_sweep"
    elif args.mode == "confirm":
        kinds, variants, n, start = KINDS, VARIANTS, 8000, 320100
        tag = "gate30_confirm"
    else:
        kinds, variants, n, start = ("single",), SMALLN, None, 310100
        tag = "gate30_smalln"
    if args.start is not None:
        start = args.start

    lam = {k: _calibrate_kind(cfg, k) for k in KINDS}

    from joblib import Parallel, delayed
    if args.mode == "smalln":
        jobs = [(rep, k, nn) for nn in (1000, 2000, 4000)
                for k in kinds for rep in range(args.reps)]
        out = Parallel(n_jobs=args.jobs)(
            delayed(run_rep)(rep, k, cfg, lam[k], variants, nn, start)
            for rep, k, nn in jobs)
    else:
        jobs = [(rep, k) for k in kinds for rep in range(args.reps)]
        out = Parallel(n_jobs=args.jobs)(
            delayed(run_rep)(rep, k, cfg, lam[k], variants, n, start)
            for rep, k in jobs)

    raw = pd.DataFrame([r for ch in out for r in ch])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw.to_csv(os.path.join(root, "results", f"{tag}_raw.csv"), index=False)

    if args.mode == "smalln":
        summ = (raw.groupby(["n", "variant"])
                .agg(reps=("rep", "size"), mae=("mae", "mean"),
                     purity=("purity", "mean"), capture=("capture", "mean"),
                     commit=("committed", lambda s: 100 * s.mean()),
                     coverage=("coverage", "mean"),
                     fallback=("fallback", lambda s: 100 * s.mean()),
                     support=("support", "mean"))
                .reset_index())
    else:
        summ = _holm(summarise(raw))
        summ = summ.sort_values(["kind", "d_mae"])
    summ.to_csv(os.path.join(root, "results", f"{tag}_summary.csv"), index=False)

    print(f"\n=== Gate 30 ({args.mode}) — {args.reps} reps ===")
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

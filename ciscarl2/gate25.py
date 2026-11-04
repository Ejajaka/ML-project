"""Gate 25 - two-leaf rule list on the two-region DGP.

Pre-registration: `results/gate25_PREREGISTRATION.md`.

Models: A (max mean), C200 (single leaf), C400 (sensitivity), LIST (two leaves).
Conditions: clean two-region DGP and Gate 24 joint stress.

The two leaves come from one CART and are disjoint by construction; actual
overlap is still measured and reported.

Usage
-----
    python -m ciscarl2.gate25 --reps 2 --start 280900   # smoke, off-seed
    python -m ciscarl2.gate25 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import aipw, crossfit_nuisances, phi_on_holdout
from .gate2 import pseudo_outcome
from .gate4d import TSTAR, _delta_for
from .gate11 import _cart
from .gate19 import _assign_effs, _leaf_table, _phi_marginal
from .gate20 import _capped_G
from .gate22 import _stats
from .gate23 import E1, E2, _calibrate_two, _config23, _regions, _sim_two
from .gate24 import _calibrate_joint, _sim_joint
from .rules import eval_rule, jaccard

from .paths import data_path

DATA = data_path()
START = 280100
MODELS = ("A", "C200", "C400", "LIST")
CONDITIONS = ("clean", "stress")
N_EVAL = 20
SUP = 200
JAC = 0.20


def _select_model(model, psi, leaves, n_ref):
    means, ses, sup = _stats(psi, leaves)
    scale = n_ref / 4000.0

    def thr(t):
        return max(int(round(t * scale)), 5)

    if model == "A":
        return [int(np.nanargmax(means))]
    if model in ("C200", "C400"):
        need = thr(SUP if model == "C200" else 400)
        ok = sup >= need
        return [int(np.nanargmax(np.where(ok, means, -np.inf)))] if ok.any() \
            else [int(np.nanargmax(means))]
    # LIST
    need = thr(SUP)
    ok = sup >= need
    if not ok.any():
        return [int(np.nanargmax(means))]
    first = int(np.nanargmax(np.where(ok, means, -np.inf)))
    m1 = leaves[first]["mask"]
    for j in sorted(range(len(leaves)), key=lambda j: -means[j]):
        if j == first or not ok[j]:
            continue
        m2 = leaves[j]["mask"]
        u = (m1 & m2).sum()
        if u / max((m1 | m2).sum(), 1) < JAC:
            return [first, j]
    return [first]


def _union_mask(sel, leaves, X, idx=None):
    m = np.zeros(len(X), bool)
    for j in sel:
        m |= eval_rule(leaves[j]["cond"], X)
    return m


def run_rep(rep, condition, cfg, lam_clean, lam_joint):
    pool, mu_, sd_, cuts, idx, lam0 = cfg
    seed = START + rep
    rng = np.random.RandomState(seed)
    n = 8000
    X = pool[rng.choice(len(pool), n, replace=True)]
    R1, R2 = _regions(X, cfg)
    union = R1 | R2
    lp0 = X @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    e1 = np.where(R1, _delta_for(E1, s0), 0.0)
    e2 = np.where(R2, _delta_for(E2, s0), 0.0)
    A = rng.binomial(1, 0.5, n)
    if condition == "clean":
        sim = _sim_two(X, A, lam_clean, lam0, rng, e1, e2)
    else:
        sim = _sim_joint(X, A, lam_joint, lam0, rng, e1, e2, R1, cfg)
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
    G = _capped_G(muD, GD, Gmarg, fid, fcaps,
                  "conditional" if condition == "clean" else "marginal", TD)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
    _, _, q1, q0 = aipw(XD, AD, TD, TSTAR, muD, G)
    const = float((q1 - q0).mean())
    const_mae = float(np.abs(const - truth).mean())
    leaves = _cart(psi, XD, 3)[1]
    means, ses_d, sup = _stats(psi, leaves)

    if condition == "clean":
        p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=seed)
        p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, TSTAR, seed=seed)
    else:
        p1v, p0v = _phi_marginal(XD, AD, TD, ED, XV, AV, TV)
        p1a, p0a = _phi_marginal(XD, AD, TD, ED, XA, AA, TA)
    dphi_v = p1v - p0v
    dphi_a = p1a - p0a
    tbl_v = _leaf_table(leaves, XV, dphi_v)
    effs_v = np.array([t["eff"] for t in tbl_v], float)
    ses_v = np.array([t["se"] for t in tbl_v], float)
    effs_v = np.where(np.isfinite(effs_v), effs_v, const)
    masks_a = [eval_rule(r["cond"], XA) for r in leaves]

    rng_eval = np.random.RandomState(seed + 41)
    evals = [rng_eval.choice(len(XD), len(XD) // 2, replace=False)
             for _ in range(N_EVAL)]

    rows = []
    for model in MODELS:
        sel = _select_model(model, psi, leaves, len(XD))
        sel_mask = _union_mask(sel, leaves, XA)
        k = int(sel_mask.sum())
        committed = []
        for j in sel:
            mv = eval_rule(leaves[j]["cond"], XV)
            kv = int(mv.sum())
            e = float(dphi_v[mv].mean()) if kv else np.nan
            s = (float(np.sqrt(((dphi_v[mv] - e) ** 2).sum()) / kv) if kv else np.nan)
            committed.append(bool(np.isfinite(e) and np.isfinite(s) and
                                  e > norm.ppf(0.95) * s))
        com_mask = np.zeros(len(XA), bool)
        for j, c in zip(sel, committed):
            if c:
                com_mask |= masks_a[j]
        # prediction: all leaves, uncommitted selected leaves -> const
        use = effs_v.copy()
        for j, c in zip(sel, committed):
            if not c:
                use[j] = const
        tbl = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                    eff=use[t], se=ses_v[t]) for t in range(len(leaves))]
        pred = _assign_effs(tbl, XA, use, const)
        mae = float(np.abs(pred - truth).mean())

        kc = int(com_mask.sum())
        per = float(truth[com_mask].mean()) if kc else np.nan
        oracle = float(np.sort(truth)[::-1][:kc].mean()) if kc else np.nan
        # stability: refit the same model on disjoint discovery halves
        jf = []
        for s_idx in evals:
            lb = _cart(psi[s_idx], XD[s_idx], 3)[1]
            sel_b = _select_model(model, psi[s_idx], lb, len(s_idx))
            mb = _union_mask(sel_b, lb, XA)
            jf.append(jaccard(mb, sel_mask))
        jf = np.array(jf)
        overlap = jaccard(masks_a[sel[0]], masks_a[sel[1]]) if len(sel) > 1 else 0.0
        both = ((masks_a[sel[0]] & masks_a[sel[1]]).sum() if len(sel) > 1 else 0)

        rows.append(dict(
            rep=rep, n=n, condition=condition, model=model,
            n_leaves=len(sel), fallback=bool(model == "LIST" and len(sel) == 1),
            support=float(sum(sup[j] for j in sel)),
            support1=float(sup[sel[0]]),
            support2=float(sup[sel[1]]) if len(sel) > 1 else np.nan,
            capture_sel=float(truth[sel_mask].sum() / truth.sum()),
            purity=float((sel_mask & uniA).sum() / k) if k else np.nan,
            capture_region=float((sel_mask & uniA).sum() / max(uniA.sum(), 1)),
            j_true=jaccard(sel_mask, uniA),
            jf_mean=jf.mean(), freq_full=100.0 * float((jf > 0.8).mean()),
            overlap=overlap, both=both,
            commit_any=bool(any(committed)),
            coverage=float(com_mask.mean()),
            per_treated=per, efficiency=(per / oracle if (kc and oracle) else np.nan),
            captured_commit=float(truth[com_mask].sum() / truth.sum()) if kc else 0.0,
            null=bool(kc and np.isfinite(per) and per <= 0.01),
            beneficial=bool(kc and np.isfinite(per) and per > 0),
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
    for (cond, model), g in df.groupby(["condition", "model"]):
        tr = g[g.commit_any]
        out.append(dict(
            condition=cond, model=model, reps=len(g),
            n_leaves=g.n_leaves.mean(), fallback=100.0 * g.fallback.mean(),
            support=g.support.mean(), support1=g.support1.mean(),
            overlap=g.overlap.mean(), both=g.both.mean(),
            capture=g.capture_sel.mean(), purity=g.purity.mean(),
            capture_region=g.capture_region.mean(), j_true=g.j_true.mean(),
            jf_mean=g.jf_mean.mean(), freq_full=g.freq_full.mean(),
            commit=100.0 * g.commit_any.mean(), coverage=g.coverage.mean(),
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
    lam_clean = _calibrate_two(cfg)
    lam_joint = _calibrate_joint(cfg)
    print(f"  lam_c0 clean={lam_clean:.3e} stress={lam_joint:.3e}", flush=True)

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, cond, cfg, lam_clean, lam_joint)
        for cond in CONDITIONS for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate25_rulelist.csv"), index=False)
    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate25_summary.csv"), index=False)

    print(f"\n=== Gate 25, {args.reps} reps ===")
    print(f"{'cond':>7}{'model':>6}{'nleaf':>6}{'fall%':>6}{'support':>8}"
          f"{'capture':>8}{'purity':>7}{'J_true':>7}{'J_full':>7}{'commit':>7}"
          f"{'cover':>7}{'per-tr':>8}{'effic':>7}{'capC':>7}{'null%':>6}{'MAE':>8}")
    for _, r in res.sort_values(["condition", "model"]).iterrows():
        print(f"{r.condition:>7}{r.model:>6}{r.n_leaves:>6.1f}"
              f"{r.fallback:>5.0f}%{r.support:>8.0f}{r.capture:>8.3f}"
              f"{r.purity:>7.2f}{r.j_true:>7.3f}{r.jf_mean:>7.3f}"
              f"{r.commit:>6.0f}%{r.coverage:>7.3f}{r.per_treated:>+8.3f}"
              f"{r.efficiency:>7.2f}{r.captured_commit:>7.3f}"
              f"{r.null_rate:>6.0f}{r.mae:>8.4f}")

    print("\nPaired LIST vs C200 (negative favourable for MAE):")
    for cond in CONDITIONS:
        g = df[df.condition == cond]
        a = g[g.model == "C200"].set_index("rep")
        b = g[g.model == "LIST"].set_index("rep")
        common = sorted(set(a.index) & set(b.index))
        for metric in ("capture_sel", "j_true", "purity", "per_treated",
                       "captured_commit", "mae"):
            d = (b.loc[common, metric] - a.loc[common, metric]).to_numpy()
            lo, hi, p = _boot(d)
            print(f"  {cond:>6} {metric:>16}: {d.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] p={p:.3f}")

    clean = res[res.condition == "clean"]
    c200 = clean[clean.model == "C200"].iloc[0]
    lst = clean[clean.model == "LIST"].iloc[0]
    g = df[df.condition == "clean"]
    a = g[g.model == "C200"].set_index("rep")
    b = g[g.model == "LIST"].set_index("rep")
    common = sorted(set(a.index) & set(b.index))
    dcap = (b.loc[common, "capture_sel"] - a.loc[common, "capture_sel"]).to_numpy()
    djt = (b.loc[common, "j_true"] - a.loc[common, "j_true"]).to_numpy()
    clo, chi, _ = _boot(dcap)
    jlo, jhi, _ = _boot(djt)
    checks = {
        "capture +0.05 with CI>0 OR J_true +0.05 with CI>0": bool(
            (dcap.mean() >= 0.05 and clo > 0) or (djt.mean() >= 0.05 and jlo > 0)),
        "purity >= C200-0.05": bool(lst.purity >= c200.purity - 0.05),
        "per-treated >= C200-0.02": bool(lst.per_treated >= c200.per_treated - 0.02),
        "null <= 5%": bool(lst.null_rate <= 5.0),
        "efficiency >= 0.70": bool(lst.efficiency >= 0.70),
        "MAE delta < 0": bool(lst.delta < 0),
        "MAE <= C200+0.005": bool(lst.mae <= c200.mae + 0.005),
    }
    print("\nPre-registered checks (clean, LIST vs C200):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    stress = res[(res.condition == "stress") & (res.model == "LIST")].iloc[0]
    sc200 = res[(res.condition == "stress") & (res.model == "C200")].iloc[0]
    stress_ok = (stress.purity >= sc200.purity - 0.05 and
                 stress.null_rate <= 5.0 and stress.efficiency >= 0.70)
    print(f"\nStress guardrail (LIST): purity {stress.purity:.2f} "
          f"(C200 {sc200.purity:.2f}), null {stress.null_rate:.0f}%, "
          f"efficiency {stress.efficiency:.2f} -> "
          f"{'PASS' if stress_ok else 'FAIL'}")
    print(f"\n-> {'LIST ADOPTED' if all(checks.values()) and stress_ok else 'LIST NOT adopted'}")
    return df


if __name__ == "__main__":
    main()

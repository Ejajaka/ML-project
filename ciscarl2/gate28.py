"""Gate 28 - head-to-head: frozen C200 vs SCRE on the same assessment patients.

Pre-registration: `results/gate28_PREREGISTRATION.md`.

Both methods see discovery 4000 / validation 2000 / assessment 2000.
SCRE is fitted on discovery, its alpha is selected on validation, and it
predicts the assessment. C200 is frozen.

Usage
-----
    python -m ciscarl2.gate28 --reps 2 --start 300900   # smoke, off-seed
    python -m ciscarl2.gate28 --reps 40
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import aipw, crossfit_nuisances, phi_on_holdout
from .gate2 import pseudo_outcome
from .gate4 import scre_select_alpha
from .gate4d import TSTAR, _delta_for
from .gate11 import _cart
from .gate19 import _assign_effs, _leaf_table
from .gate20 import _capped_G
from .gate22 import _stats
from .gate23 import E1, E2, _config23, _regions
from .gate26 import _calibrate_kind, _sim_kind
from .rules import eval_rule

try:
    from scre import SurvivalCausalRuleEnsemble              # noqa: E402
except Exception:                                            # noqa: BLE001
    SurvivalCausalRuleEnsemble = None

from .paths import data_path

DATA = data_path()
START = 300100
KINDS = ("single", "two")
SUP = 200


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


def run_rep(rep, kind, cfg, lam_c0, start=None):
    pool, mu_, sd_, cuts, idx, lam0 = cfg
    seed = (START if start is None else start) + rep
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

    # ---------------- C200 ----------------
    leaves = _cart(psi, XD, 3)[1]
    means, ses_d, sup = _stats(psi, leaves)
    ok = sup >= SUP
    j = int(np.nanargmax(np.where(ok, means, -np.inf))) if ok.any() \
        else int(np.nanargmax(means))
    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=seed)
    dphi_v = p1v - p0v
    p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, TSTAR, seed=seed)
    dphi_a = p1a - p0a
    tbl_v = _leaf_table(leaves, XV, dphi_v)
    effs = np.array([t["eff"] for t in tbl_v], float)
    ses = np.array([t["se"] for t in tbl_v], float)
    const = float(aipw(XD, AD, TD, TSTAR, muD, G)[0])
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
    masks_a = [eval_rule(r["cond"], XA) for r in leaves]
    tbl_a = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                  eff=use[t], se=ses[t]) for t in range(len(leaves))]
    pred_c = _assign_effs(tbl_a, XA, use, const)
    treated_c = masks_a[j] if committed else np.zeros(len(XA), bool)

    # ---------------- SCRE ----------------
    tau_s = np.full(len(XA), np.nan)
    n_rules_s = 0
    ok_s = True
    try:
        if SurvivalCausalRuleEnsemble is None:
            raise RuntimeError("SCRE not installed; set SCRE_PATH")
        s = SurvivalCausalRuleEnsemble(
            alpha=0.5, pred_n_estimators=100, progn_n_estimators=100,
            rule_min_support=20, max_rules=200, max_rule_conditions=3,
            max_selected_rules=10, random_state=42)
        with contextlib.redirect_stdout(io.StringIO()):
            s.fit(XD, psi, np.isfinite(psi),
                  feature_names=[f"x{i}" for i in range(XD.shape[1])],
                  time=TD, event=ED.astype(bool), treatment=AD)
            pick = scre_select_alpha(s, XV, AV, TV, EV.astype(bool), TSTAR)
        if pick is not None:
            s.alpha_ = s.alpha_final_ = pick[1]
            s.coef_ = pick[2]
        s.t_star_ = TSTAR
        with contextlib.redirect_stdout(io.StringIO()):
            tau_s = np.asarray(s.predict(XA), float)
        n_rules_s = len(getattr(s, "selected_rules_", []))
    except Exception:
        ok_s = False

    rows = []
    mc = _metrics(pred_c, treated_c, truth, uniA)
    rows.append(dict(rep=rep, kind=kind, method="C200", **mc,
                     committed=committed, n_rules=1, support=int(sup[j]),
                     ok=True))
    if ok_s and np.all(np.isfinite(tau_s)):
        ms = _metrics(tau_s, tau_s > 0, truth, uniA)
        rows.append(dict(rep=rep, kind=kind, method="SCRE", **ms,
                         committed=True, n_rules=n_rules_s, support=np.nan,
                         ok=True))
    else:
        rows.append(dict(rep=rep, kind=kind, method="SCRE", mae=np.nan,
                         spearman=np.nan, capture=np.nan, purity=np.nan,
                         per_treated=np.nan, efficiency=np.nan, null=np.nan,
                         coverage=np.nan, committed=True, n_rules=np.nan,
                         support=np.nan, ok=False))
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
    for (kind, method), g in df.groupby(["kind", "method"]):
        out.append(dict(
            kind=kind, method=method, reps=len(g),
            ok=100.0 * g.ok.mean(), mae=g.mae.mean(),
            spearman=g.spearman.mean(), capture=g.capture.mean(),
            purity=g.purity.mean(), per_treated=g.per_treated.mean(),
            efficiency=g.efficiency.mean(),
            null_rate=100.0 * g.null.mean(),
            coverage=g.coverage.mean(), rules=g.n_rules.mean(),
            commit=100.0 * g.committed.mean()))
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

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, k, cfg, lam[k])
        for k in KINDS for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate28_scre.csv"), index=False)
    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate28_summary.csv"), index=False)

    print(f"\n=== Gate 28, C200 vs SCRE, {args.reps} reps ===")
    print(f"{'kind':>7}{'method':>7}{'ok%':>5}{'MAE':>9}{'Spear':>7}"
          f"{'capture':>8}{'purity':>7}{'per-tr':>8}{'effic':>7}"
          f"{'null%':>6}{'cover':>7}{'rules':>6}{'commit%':>8}")
    for _, r in res.sort_values(["kind", "method"]).iterrows():
        print(f"{r.kind:>7}{r.method:>7}{r.ok:>5.0f}{r.mae:>9.4f}{r.spearman:>7.3f}"
              f"{r.capture:>8.3f}{r.purity:>7.2f}{r.per_treated:>+8.3f}"
              f"{r.efficiency:>7.2f}{r.null_rate:>6.0f}{r.coverage:>7.3f}"
              f"{r.rules:>6.1f}{r.commit:>7.0f}%")

    print("\nPaired C200 - SCRE:")
    for kind in KINDS:
        g = df[(df.kind == kind) & df.ok]
        a = g[g.method == "C200"].set_index("rep")
        b = g[g.method == "SCRE"].set_index("rep")
        common = sorted(set(a.index) & set(b.index))
        for metric in ("mae", "spearman", "capture"):
            d = (a.loc[common, metric] - b.loc[common, metric]).to_numpy()
            d = d[np.isfinite(d)]
            lo, hi, p = _boot(d)
            print(f"  {kind:>6} {metric:>12}: {d.mean():+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] p={p:.3f}")

    print("\nDecision: C200 beats SCRE on MAE only if the paired MAE difference "
          "is negative with the CI entirely below zero.")
    return df


if __name__ == "__main__":
    main()

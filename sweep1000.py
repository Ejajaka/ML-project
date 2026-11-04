"""Resumable 1000-seed evaluation: C300 and SCRE on identical splits.

Checkpoints every `--chunk` reps and resumes from the partial CSV, so a crash or
Colab disconnect does not lose progress.
"""
import os, sys, io, time, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.model_selection import train_test_split

from ciscarl2 import gate28
from ciscarl2.gate28 import (
    TSTAR, crossfit_nuisances, _capped_G, pseudo_outcome, aipw, _cart, _stats,
    phi_on_holdout, _leaf_table, _assign_effs, eval_rule, _metrics,
    _regions_kind, _sim_kind, _delta_for, E1, E2, dgp, scre_select_alpha,
    SurvivalCausalRuleEnsemble,
)

FLOORS = [(200, "C200"), (300, "C300"), (400, "C400")]


def run_rep(rep, kind, cfg, lam_c0, start):
    g = gate28
    pool, mu_, sd_, cuts, idx, lam0 = cfg
    seed = start + rep
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
    disc, hold = train_test_split(ii, test_size=0.5, random_state=seed, stratify=A)
    val, asm = train_test_split(hold, test_size=0.5, random_state=seed + 1, stratify=A[hold])
    XD, AD, TD, ED = X[disc], A[disc], T[disc], E[disc]
    XV, AV, TV, EV = X[val], A[val], T[val], E[val]
    XA, AA, TA, EA = X[asm], A[asm], T[asm], E[asm]
    uniA, truth = union[asm], tau_i[asm]

    muD, GD, Gmarg, fid, fcaps, _ = crossfit_nuisances(
        XD, AD, TD, ED, TSTAR, seed=seed, return_folds=True)
    G = _capped_G(muD, GD, Gmarg, fid, fcaps, "conditional", TD)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)

    leaves = _cart(psi, XD, 3)[1]
    means, ses_d, sup = _stats(psi, leaves)
    p1v, p0v = phi_on_holdout(XD, AD, TD, ED, XV, AV, TV, TSTAR, seed=seed)
    dphi_v = p1v - p0v
    p1a, p0a = phi_on_holdout(XD, AD, TD, ED, XA, AA, TA, TSTAR, seed=seed)
    dphi_a = p1a - p0a
    tbl_v = _leaf_table(leaves, XV, dphi_v)
    effs = np.array([t["eff"] for t in tbl_v], float)
    ses = np.array([t["se"] for t in tbl_v], float)
    const = float(aipw(XD, AD, TD, TSTAR, muD, G)[0])
    effs = np.where(np.isfinite(effs), effs, const)
    masks_a = [eval_rule(r["cond"], XA) for r in leaves]

    rows = []
    for floor, label in FLOORS:
        ok = sup >= floor
        j = int(np.nanargmax(np.where(ok, means, -np.inf))) if ok.any() else int(np.nanargmax(means))
        mv = eval_rule(leaves[j]["cond"], XV)
        kv = int(mv.sum())
        hi_eff = float(dphi_v[mv].mean()) if kv else np.nan
        hi_se = (float(np.sqrt(((dphi_v[mv] - hi_eff) ** 2).sum()) / kv) if kv else np.nan)
        committed = bool(np.isfinite(hi_eff) and np.isfinite(hi_se) and hi_eff > 1.645 * hi_se)
        use = effs.copy()
        if not committed:
            use[j] = const
        tbl_a = [dict(cond=leaves[t]["cond"], mask=masks_a[t], eff=use[t], se=ses[t])
                 for t in range(len(leaves))]
        pred_c = _assign_effs(tbl_a, XA, use, const)
        treated_c = masks_a[j] if committed else np.zeros(len(XA), bool)
        mc = _metrics(pred_c, treated_c, truth, uniA)
        rows.append(dict(rep=rep, kind=kind, method=label, **mc,
                         committed=committed, n_rules=1, support=int(sup[j]), ok=True))

    # ---- SCRE ----
    tau_s = np.full(len(XA), np.nan)
    n_rules_s = 0
    ok_s = True
    try:
        if SurvivalCausalRuleEnsemble is None:
            raise RuntimeError("SCRE unavailable")
        s = SurvivalCausalRuleEnsemble(
            alpha=0.5, pred_n_estimators=100, progn_n_estimators=100,
            rule_min_support=20, max_rules=200, max_rule_conditions=3,
            max_selected_rules=10, random_state=42)
        with g.contextlib.redirect_stdout(io.StringIO()):
            s.fit(XD, psi, np.isfinite(psi),
                  feature_names=[f"x{i}" for i in range(XD.shape[1])],
                  time=TD, event=ED.astype(bool), treatment=AD)
            pick = scre_select_alpha(s, XV, AV, TV, EV.astype(bool), TSTAR)
        if pick is not None:
            s.alpha_ = s.alpha_final_ = pick[1]
            s.coef_ = pick[2]
        s.t_star_ = TSTAR
        with g.contextlib.redirect_stdout(io.StringIO()):
            tau_s = np.asarray(s.predict(XA), float)
        n_rules_s = len(getattr(s, "selected_rules_", []))
    except Exception:
        ok_s = False

    if ok_s and np.all(np.isfinite(tau_s)):
        ms = _metrics(tau_s, tau_s > 0, truth, uniA)
        rows.append(dict(rep=rep, kind=kind, method="SCRE", **ms,
                         committed=True, n_rules=n_rules_s, support=np.nan, ok=True))
    else:
        rows.append(dict(rep=rep, kind=kind, method="SCRE", mae=np.nan, spearman=np.nan,
                         capture=np.nan, purity=np.nan, per_treated=np.nan,
                         efficiency=np.nan, null=np.nan, coverage=np.nan,
                         committed=True, n_rules=np.nan, support=np.nan, ok=False))
    return rows


def summarise(df):
    out = []
    for (kind, method), g in df.groupby(["kind", "method"]):
        n_all = len(g)
        p = g["purity"].dropna().to_numpy(float)
        n = len(p)
        if n:
            m = float(p.mean()); sd = float(p.std(ddof=1)) if n > 1 else np.nan
            half = float(tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            q = np.percentile(p, [10, 25, 50, 75, 90])
        else:
            m = sd = half = np.nan; q = np.array([np.nan] * 5)
        out.append(dict(
            kind=kind, method=method, n_all=n_all, n_committed=n,
            purity_mean=m, purity_sd=sd, purity_ci_lo=m - half, purity_ci_hi=m + half,
            purity_p10=q[0], purity_p25=q[1], purity_median=q[2],
            purity_p75=q[3], purity_p90=q[4],
            pct_ge_080_committed=float((p >= 0.80).mean() * 100) if n else np.nan,
            pct_ge_080_all=float((p >= 0.80).sum() / n_all * 100),
            coverage_mean=float(g["coverage"].mean()),
            capture_mean=float(g["capture"].mean()),
            per_treated_mean=float(g["per_treated"].mean()),
            commit_rate=float(g["committed"].mean() * 100),
            mae_mean=float(g["mae"].mean()),
        ))
    return pd.DataFrame(out).sort_values(["kind", "method"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--kind", default="single", choices=["single", "two"])
    ap.add_argument("--start", type=int, default=600100)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=25)
    args = ap.parse_args()

    cfg = gate28._config23()
    lam = gate28._calibrate_kind(cfg, args.kind)
    part = os.path.join(HERE, "results", f"sweep1000_{args.kind}_partial.csv")
    summ = os.path.join(HERE, "results", f"sweep1000_{args.kind}_summary.csv")

    done = set()
    if os.path.exists(part):
        old = pd.read_csv(part)
        done = set(old["rep"].unique().tolist())
    all_rows = pd.read_csv(part).to_dict("records") if os.path.exists(part) else []

    from joblib import Parallel, delayed
    t0 = time.time()
    todo = [r for r in range(args.reps) if r not in done]
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        out = Parallel(n_jobs=args.jobs)(
            delayed(run_rep)(rep, args.kind, cfg, lam, args.start) for rep in chunk)
        all_rows += [x for ch in out for x in ch if x]
        pd.DataFrame(all_rows).to_csv(part, index=False)
        df = summarise(pd.DataFrame(all_rows))
        df.to_csv(summ, index=False)
        print(f"[{time.time()-t0:6.0f}s] done {len(done)+i+len(chunk)}/{args.reps}", flush=True)

    print("\n=== FINAL ===")
    print(summarise(pd.DataFrame(all_rows)).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

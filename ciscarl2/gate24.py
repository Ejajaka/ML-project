"""Gate 24 - joint nuisance stress test on the two-region DGP.

Pre-registration: `results/gate24_PREREGISTRATION.md`.

Two-region truth (unchanged from Gate 23) + control-arm non-PH inside R1
(kappa=3, S(730) preserved) + marginal-KM censoring.

Configs: A (max mean), C200 (primary), C400 (sensitivity).

Usage
-----
    python -m ciscarl2.gate24 --reps 2 --start 270900   # smoke, off-seed
    python -m ciscarl2.gate24 --reps 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from . import dgp
from .estimators import aipw, crossfit_nuisances
from .gate2 import pseudo_outcome
from .gate4d import TSTAR, _delta_for
from .gate11 import _cart
from .gate19 import _assign_effs, _leaf_table, _phi_marginal
from .gate20 import _capped_G
from .gate22 import _pick, _stats
from .gate23 import E1, E2, _config23, _regions
from .rules import eval_rule, jaccard

from .paths import data_path

DATA = data_path()
START = 270100
CONFIGS = ("A", "C200", "C400")
THRESH = {"C200": 200, "C400": 400}
N_EVAL = 20
KAPPA = 3.0


def _sim_joint(X, A, lam_c0, lam0, rng, e1, e2, R1, cfg):
    _, _, _, _, idx, _ = cfg
    lp0 = X @ dgp.BETA
    lp1 = lp0 - np.maximum(e1, e2)
    z = 0.5 * (X[:, idx["age"]] - X[:, idx["cd40"]])
    a = KAPPA * z
    mf = np.where(R1, np.exp(a) / np.cosh(a), 1.0)
    ms = np.where(R1, np.exp(-a) / np.cosh(a), 1.0)
    base = lam0 * np.exp(np.where(A == 1, lp1, lp0))
    use = (A == 0)
    lf = base * np.where(use, mf, 1.0)
    ls = base * np.where(use, ms, 1.0)
    V = rng.exponential(1.0, len(X))
    Hc = lf * 365.0
    T = np.where(V <= Hc, V / np.maximum(lf, 1e-12),
                 365.0 + (V - Hc) / np.maximum(ls, 1e-12))
    C = rng.exponential(1.0 / np.maximum(lam_c0 * np.exp(X @ dgp.GAMMA), 1e-12))
    Tobs = np.minimum(T, C)
    E = (T <= C).astype(int)
    S0 = np.exp(-lam0 * np.exp(lp0) * (mf + ms) * 365.0)
    S1 = np.exp(-lam0 * np.exp(lp1) * 2.0 * 365.0)
    return dict(Tobs=Tobs, E=E, S0=S0, S1=S1)


def _calibrate_joint(cfg, target=0.10, n_pilot=6000, seed=99):
    pool, mu, sd, cuts, idx, lam0 = cfg
    rng0 = np.random.RandomState(seed)
    Xp = pool[rng0.choice(len(pool), n_pilot, replace=True)]
    R1, R2 = _regions(Xp, cfg)
    lp0 = Xp @ dgp.BETA
    s0 = np.exp(-lam0 * TSTAR * np.exp(lp0))
    e1 = np.where(R1, _delta_for(E1, s0), 0.0)
    e2 = np.where(R2, _delta_for(E2, s0), 0.0)

    def frac(lc):
        rng = np.random.RandomState(seed)
        A = rng.binomial(1, 0.5, len(Xp))
        d = _sim_joint(Xp, A, lc, lam0, rng, e1, e2, R1, cfg)
        return float(np.mean((d["Tobs"] <= TSTAR) & (d["E"] == 0)))

    lo, hi = 1e-7, 5e-2
    for _ in range(45):
        mid = float(np.sqrt(lo * hi))
        if frac(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def run_rep(rep, start, cfg, lam_c0):
    pool, mu_, sd_, cuts, idx, lam0 = cfg
    seed = start + rep
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
    sim = _sim_joint(X, A, lam_c0, lam0, rng, e1, e2, R1, cfg)
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
    G = _capped_G(muD, GD, Gmarg, fid, fcaps, "marginal", TD)
    psi = pseudo_outcome(AD, TD, TSTAR, muD, G)
    _, _, q1, q0 = aipw(XD, AD, TD, TSTAR, muD, G)
    const = float((q1 - q0).mean())
    const_mae = float(np.abs(const - truth).mean())

    leaves = _cart(psi, XD, 3)[1]
    means, ses, sup = _stats(psi, leaves)
    picks = {"A": _pick("A", means, ses, sup, len(XD))}
    for c in ("C200", "C400"):
        picks[c] = _pick(c, means, ses, sup, len(XD), THRESH[c])
    conds = {c: leaves[j]["cond"] for c, j in picks.items()}

    p1v, p0v = _phi_marginal(XD, AD, TD, ED, XV, AV, TV)
    dphi_v = p1v - p0v
    p1a, p0a = _phi_marginal(XD, AD, TD, ED, XA, AA, TA)
    dphi_a = p1a - p0a
    tbl_v = _leaf_table(leaves, XV, dphi_v)
    effs_v = np.array([t["eff"] for t in tbl_v], float)
    ses_v = np.array([t["se"] for t in tbl_v], float)
    effs_v = np.where(np.isfinite(effs_v), effs_v, const)
    masks_a = [eval_rule(r["cond"], XA) for r in leaves]

    rng_eval = np.random.RandomState(seed + 41)
    evals = [rng_eval.choice(len(XD), len(XD) // 2, replace=False)
             for _ in range(N_EVAL)]
    rec = {c: {"jf": [], "jt": []} for c in CONFIGS}
    for s_idx in evals:
        lb = _cart(psi[s_idx], XD[s_idx], 3)[1]
        mb, sb, ub = _stats(psi[s_idx], lb)
        for c in CONFIGS:
            j = _pick(c, mb, sb, ub, len(s_idx), THRESH.get(c))
            m = eval_rule(lb[j]["cond"], XA)
            rec[c]["jf"].append(jaccard(m, eval_rule(conds[c], XA)))
            rec[c]["jt"].append(jaccard(m, uniA))

    rows = []
    for c in CONFIGS:
        j = picks[c]
        mfull = masks_a[j]
        k = int(mfull.sum())
        jf = np.array(rec[c]["jf"])
        jt = np.array(rec[c]["jt"])
        mv = eval_rule(conds[c], XV)
        kv = int(mv.sum())
        hi_eff = float(dphi_v[mv].mean()) if kv else np.nan
        hi_se = (float(np.sqrt(((dphi_v[mv] - hi_eff) ** 2).sum()) / kv)
                 if kv else np.nan)
        committed = bool(np.isfinite(hi_eff) and np.isfinite(hi_se) and
                         hi_eff > norm.ppf(0.95) * hi_se)
        use = effs_v.copy()
        if not committed:
            use[j] = const
        tbl = [dict(cond=leaves[t]["cond"], mask=masks_a[t],
                    eff=use[t], se=ses_v[t]) for t in range(len(leaves))]
        pred = _assign_effs(tbl, XA, use, const)
        mae = float(np.abs(pred - truth).mean())

        per = float(truth[mfull].mean()) if (committed and k) else np.nan
        cov = float(mfull.mean()) if committed else 0.0
        cap = (float(truth[mfull].sum() / truth.sum())
               if (committed and truth.sum() > 0) else 0.0)
        oracle = (float(np.sort(truth)[::-1][:k].mean())
                  if (committed and k) else np.nan)
        rows.append(dict(
            rep=rep, n=n, config=c, support=int(sup[j]),
            capture_benefit=float(truth[mfull].sum() / truth.sum()),
            purity=float((mfull & uniA).sum() / k) if k else np.nan,
            capture_region=float((mfull & uniA).sum() / max(uniA.sum(), 1)),
            j_true=jaccard(mfull, uniA),
            jf_mean=jf.mean(), jf_median=np.median(jf),
            freq_full=100.0 * float((jf > 0.8).mean()),
            jt_mean=jt.mean(),
            committed=committed, coverage=cov, per_treated=per,
            efficiency=(per / oracle if (committed and oracle) else np.nan),
            captured_commit=cap,
            null=bool(committed and np.isfinite(per) and per <= 0.01),
            beneficial=bool(committed and np.isfinite(per) and per > 0),
            mae=mae, const_mae=const_mae, delta=mae - const_mae,
            ate_est=const, true_ate=float(truth.mean())))
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
    for c, g in df.groupby("config"):
        tr = g[g.committed]
        out.append(dict(
            config=c, reps=len(g), support=g.support.mean(),
            capture=g.capture_benefit.mean(), purity=g.purity.mean(),
            capture_region=g.capture_region.mean(), j_true=g.j_true.mean(),
            jf_mean=g.jf_mean.mean(), jf_median=g.jf_median.mean(),
            freq_full=g.freq_full.mean(), jt_mean=g.jt_mean.mean(),
            commit=100.0 * g.committed.mean(), coverage=g.coverage.mean(),
            per_treated=tr.per_treated.mean() if len(tr) else np.nan,
            efficiency=tr.efficiency.mean() if len(tr) else np.nan,
            captured_commit=g.captured_commit.mean(),
            null_rate=100.0 * tr.null.mean() if len(tr) else np.nan,
            beneficial_rate=100.0 * tr.beneficial.mean() if len(tr) else np.nan,
            mae=g.mae.mean(), delta=g.delta.mean(),
            ate_bias=(g.ate_est - g.true_ate).mean()))
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
    lam_c0 = _calibrate_joint(cfg)
    print(f"  lam_c0={lam_c0:.3e}", flush=True)

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=args.jobs)(
        delayed(run_rep)(rep, START, cfg, lam_c0) for rep in range(args.reps))
    df = pd.DataFrame([r for ch in out for r in ch if r])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(os.path.join(root, "results", "gate24_joint.csv"), index=False)
    res = summarise(df)
    res.to_csv(os.path.join(root, "results", "gate24_summary.csv"), index=False)

    print(f"\n=== Gate 24, joint stress test, {args.reps} reps ===")
    print(f"{'cfg':>5}{'support':>8}{'capture':>8}{'purity':>7}{'J_true':>7}"
          f"{'J_full':>7}{'freq%':>6}{'commit':>7}{'cover':>7}{'effic':>7}"
          f"{'capC':>7}{'null%':>6}{'MAE':>8}{'ATEbias':>9}")
    for _, r in res.iterrows():
        print(f"{r.config:>5}{r.support:>8.0f}{r.capture:>8.3f}{r.purity:>7.2f}"
              f"{r.j_true:>7.3f}{r.jf_mean:>7.3f}{r.freq_full:>5.0f}%"
              f"{r.commit:>6.0f}%{r.coverage:>7.3f}{r.efficiency:>7.2f}"
              f"{r.captured_commit:>7.3f}{r.null_rate:>6.0f}{r.mae:>8.4f}"
              f"{r.ate_bias:>+9.4f}")

    print("\nPaired contrasts vs A:")
    a = df[df.config == "A"].set_index("rep")
    for c in ("C200", "C400"):
        b = df[df.config == c].set_index("rep")
        common = sorted(set(a.index) & set(b.index))
        for metric in ("capture_benefit", "j_true", "jf_mean", "purity", "mae"):
            d = (b.loc[common, metric] - a.loc[common, metric]).to_numpy()
            lo, hi, p = _boot(d)
            print(f"  {c}-A {metric:>16}: {d.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] p={p:.3f}")

    c2 = res[res.config == "C200"].iloc[0]
    a0 = res[res.config == "A"].iloc[0]
    checks = {
        "capture > A": bool(c2.capture > a0.capture),
        "J_true > A": bool(c2.j_true > a0.j_true),
        "purity >= A-0.05": bool(c2.purity >= a0.purity - 0.05),
        "MAE delta < 0": bool(c2.delta < 0),
        "null <= 5%": bool(c2.null_rate <= 5.0),
        "efficiency >= 0.70": bool(c2.efficiency >= 0.70),
    }
    print("\nPre-registered checks (C200 vs A):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n-> {'C200 SURVIVES the joint stress test' if all(checks.values()) else 'C200 does not meet all criteria'}")
    print("Stress-test caveat: marginal-KM censoring violates an AIPW "
          "identification condition; absolute metrics are not unbiased.")
    return df


if __name__ == "__main__":
    main()

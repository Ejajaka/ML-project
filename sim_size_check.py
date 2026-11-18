"""Does the method commit when a real subgroup exists, at the ACTG175 sample size?

Runs the frozen C200 pipeline on simulated single-region and two-region DGPs at
n = 2,139 / 4,000 / 8,000, with a known truth. This separates "the method cannot
find subgroups" from "ACTG175 at n=1,054 is too small."

Run:  py -3.13 sim_size_check.py --reps 20 --ns 2139,4000,8000
Writes: results/sim_size_check.csv, .md, .json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ciscarl2.gate23 import _config23
from ciscarl2.gate26 import _calibrate_kind
from ciscarl2.gate30 import VARIANTS, run_rep

ROOT = Path(__file__).resolve().parent
START = 330100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ns", default="2139,4000,8000")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    cfg = _config23()
    lams = {k: _calibrate_kind(cfg, k) for k in ("single", "two")}
    variants = {"V0": VARIANTS["V0"]}

    from joblib import Parallel, delayed
    jobs = [(rep, kind, n) for n in ns for kind in ("single", "two")
            for rep in range(args.reps)]

    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_rep)(rep, kind, cfg, lams[kind], variants, n, START)
        for rep, kind, n in jobs)
    rows = [r for ch in out for r in ch if r]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "sim_size_check.csv", index=False)

    agg = []
    for (n, kind), g in df.groupby(["n", "kind"]):
        c = g[g.committed]
        agg.append({
            "n": int(n), "kind": kind, "reps": len(g),
            "commit": float(100 * g.committed.mean()),
            "fallback": float(100 * g.fallback.mean()),
            "support": float(g.support.mean()),
            "purity_all": float(g.purity.mean()),
            "purity_committed": float(c.purity.mean()) if len(c) else None,
            "capture": float(g.capture.mean()),
            "per_treated": float(c.per_treated.mean()) if len(c) else None,
            "mae": float(g.mae.mean()),
            "coverage": float(g.coverage.mean()),
            "null": float(100 * g["null"].mean()) if "null" in g else None,
        })
    summ = pd.DataFrame(agg).sort_values(["n", "kind"])
    summ.to_csv(ROOT / "results" / "sim_size_check_summary.csv", index=False)

    lines = [
        "# Simulated operating range at the ACTG175 sample size",
        "",
        "Frozen C200 pipeline, known single-region / two-region truth;",
        f"{args.reps} reps per (n, DGP); seeds {START}-{START + args.reps - 1}.",
        "",
        "| n | DGP | commit | purity (all) | purity (committed) | capture | per-treated | MAE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"| {int(r.n)} | {r.kind} | {r.commit:.1f}% | "
            f"{r.purity_all:.3f} | "
            f"{(f'{r.purity_committed:.3f}' if r.purity_committed is not None else '—')} | "
            f"{r.capture:.3f} | "
            f"{(f'{r.per_treated:+.3f}' if r.per_treated is not None else '—')} | "
            f"{r.mae:.4f} |")
    (ROOT / "results" / "sim_size_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    with open(ROOT / "results" / "sim_size_check.json", "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)

    print("\n=== simulated operating range ===")
    print(summ.round(3).to_string(index=False))
    print(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

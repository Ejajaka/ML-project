"""1,000-rep C200-always sweep (confidence layer removed), two-region DGP.

Uses the frozen pipeline with z = -inf so the selected leaf is always treated.
No SCRE (too slow at 1000 reps); this is the C200-always operating point.

Run:  py -3.13 always_sweep_1000.py --reps 1000 --jobs 8
Writes: results/always_sweep_1000_two.csv, _summary.csv, .md, .json
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
from ciscarl2.gate30 import run_rep

ROOT = Path(__file__).resolve().parent
START = 340100
VAR = {"V0A": dict(depth=3, leaf=60, floor=200, z=-1e9, shrink=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    cfg = _config23()
    lam = _calibrate_kind(cfg, "two")

    from joblib import Parallel, delayed
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(run_rep)(r, "two", cfg, lam, VAR, 8000, START)
        for r in range(args.reps))
    df = pd.DataFrame([x for ch in out for x in ch])
    df.to_csv(ROOT / "results" / "always_sweep_1000_two.csv", index=False)

    c = df[df.committed]
    summary = {
        "design": "two-region DGP",
        "method": "C200-always",
        "n": 8000,
        "reps": int(args.reps),
        "seeds": f"{START}-{START + args.reps - 1}",
        "commit": float(100 * df.committed.mean()),
        "fallback": float(100 * df.fallback.mean()),
        "support": float(df.support.mean()),
        "purity": float(df.purity.mean()),
        "capture": float(df.capture.mean()),
        "per_treated": float(c.per_treated.mean()) if len(c) else None,
        "coverage": float(df.coverage.mean()),
        "null_rate": float(100 * df["null"].mean()),
        "mae": float(df.mae.mean()),
    }
    pd.DataFrame([summary]).to_csv(
        ROOT / "results" / "always_sweep_1000_two_summary.csv", index=False)
    with open(ROOT / "results" / "always_sweep_1000_two.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md = [
        "# C200-always: 1,000-rep two-region sweep",
        "",
        f"n = 8,000; reps = {args.reps}; seeds {START}-{START + args.reps - 1}.",
        "The selected leaf is always treated (no confidence layer).",
        "",
        "| metric | value |",
        "|---|---|",
        f"| purity | {summary['purity']:.3f} |",
        f"| capture | {summary['capture']:.3f} |",
        f"| per-treated | {summary['per_treated']:+.3f} |",
        f"| coverage | {summary['coverage']:.3f} |",
        f"| null rate | {summary['null_rate']:.1f}% |",
        f"| MAE | {summary['mae']:.4f} |",
        f"| mean support | {summary['support']:.0f} |",
        f"| fallback | {summary['fallback']:.1f}% |",
    ]
    (ROOT / "results" / "always_sweep_1000_two.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

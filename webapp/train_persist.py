"""
Fit the CISCaRL live engine on ACTG175 and persist it.

Usage:
    py -3.13 webapp/train_persist.py
Writes: webapp/artifacts/engine.joblib  and  webapp/artifacts/summary.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import joblib
from engine import fit_engine, all_rules, predict_all, cis_matched_rule

ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)


def main():
    t0 = time.time()
    print("Fitting engine on ACTG175 (real RCT)...")
    engine = fit_engine()
    print(f"  fitted in {time.time()-t0:.0f}s")
    print(f"  n={engine['meta']['n']}, events={engine['meta']['n_events']}, "
          f"treated={engine['meta']['n_treated']}, t*={engine['t_star']:.0f}")

    rules = all_rules(engine)
    print("\nRules discovered per method:")
    for m, info in rules.items():
        print(f"  {m:18s}: {info['count']} rules"
              + (f" (from {info.get('n_candidates')} candidates)"
                 if info.get("n_candidates") else ""))

    demo = engine["demo_patient"]
    print("\nDemo patient predictions:")
    for m, v in predict_all(engine, demo).items():
        print(f"  {m:18s}: {v if v is None else round(v,4)}")
    mr = cis_matched_rule(engine, demo)
    print(f"  CISCaRL matched: {mr['condition_str']} -> {mr['recommendation']}")

    path = os.path.join(ART, "engine.joblib")
    joblib.dump(engine, path, compress=3)
    print(f"\nSaved {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    summary = {
        "meta": engine["meta"], "t_star": engine["t_star"],
        "features": engine["feature_names"],
        "rule_counts": {m: info["count"] for m, info in rules.items()},
        "demo_patient": demo,
    }
    with open(os.path.join(ART, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary.json")


if __name__ == "__main__":
    main()

"""Export meta + rules + metrics + demo prediction to static/data.json so the
GitHub Pages site works even when the API backend is unreachable."""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import joblib
from engine import predict_all, cis_matched_rule, all_rules
from metrics import load_metrics

ART = os.path.join(HERE, "artifacts", "engine.joblib")
OUT = os.path.join(HERE, "static", "data.json")


def main():
    e = joblib.load(ART)
    demo = e["demo_patient"]
    data = {
        "meta": {
            "feature_names": e["feature_names"], "ranges": e["ranges"],
            "demo_patient": demo, "t_star": e["t_star"],
            "dataset": "ACTG175 (Hammer et al. 1996) - real randomized HIV trial",
            "meta": e["meta"],
        },
        "rules": all_rules(e),
        "metrics": load_metrics("original"),
        "demo_prediction": {
            "predictions": predict_all(e, demo),
            "cis_matched_rule": cis_matched_rule(e, demo),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f)
    print(f"Wrote {OUT} ({os.path.getsize(OUT)/1e3:.0f} KB)")


if __name__ == "__main__":
    main()

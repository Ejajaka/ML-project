"""Load the committed benchmark results (original regime) into a clean
per-method accuracy table for the UI."""
import os
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

METHOD_MAP = {
    "Cox T-learner": "Cox T-learner",
    "CSF (RF)": "CSF (black-box)",
    "Bo & Ding": "Bo & Ding",
    "Hybrid": "Hybrid",
    "CRE": "CRE",
    "SCRE": "SCRE",
    "CISCaRL (posthoc)": "CISCaRL",
}
METRICS = ["MAE", "RMSE", "R2", "Spearman", "Acc", "Prec", "Rec", "F1", "AUC", "Rules"]
LOWER_BETTER = {"MAE", "RMSE", "Rules"}


def clean_nan(o):
    """Recursively replace NaN with None so responses are JSON-safe."""
    import math
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def load_metrics(regime: str = "rescaled"):
    path = os.path.join(RESULTS, f"fair_benchmark_{regime}.csv")
    if not os.path.exists(path):
        path = os.path.join(RESULTS, "fair_benchmark.csv")
    if not os.path.exists(path):
        return {"regime": regime, "available": False, "methods": [], "metrics": METRICS}
    df = pd.read_csv(path)
    df = df[df["Method"] != "Ablation"].copy()
    df["Method"] = df["Method"].map(lambda m: METHOD_MAP.get(m, m))
    df = df[df["Method"].isin(METHOD_MAP.values())]

    methods = []
    for m in METHOD_MAP.values():
        sub = df[df["Method"] == m]
        if len(sub) == 0:
            continue
        row = {"method": m, "n_settings": int(len(sub))}
        for met in METRICS:
            val = sub[met].mean() if met in sub else float("nan")
            row[met] = None if pd.isna(val) else float(val)
        methods.append(row)

    # rank per metric (1 = best)
    for met in METRICS:
        vals = [(r["method"], r[met]) for r in methods
                if r[met] is not None and not np.isnan(r[met])]
        vals.sort(key=lambda kv: kv[1], reverse=(met not in LOWER_BETTER))
        for rank, (meth, _) in enumerate(vals, 1):
            for r in methods:
                if r["method"] == meth:
                    r[f"rank_{met}"] = rank

    return {"regime": regime, "available": True, "metrics": METRICS,
            "lower_better": sorted(LOWER_BETTER), "methods": methods}


if __name__ == "__main__":
    import json
    print(json.dumps(load_metrics("original")["methods"], indent=2)[:1500])

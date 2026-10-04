"""FastAPI backend for the CISCaRL live demo.

Endpoints:
    GET  /api/meta      -> features, ranges, demo patient, t*, dataset meta
    GET  /api/rules     -> every method's discovered rules
    GET  /api/metrics   -> accuracy table (original regime)
    POST /api/predict   -> {patient:{...}} -> predictions + matched CISCaRL rule

Run locally:
    py -3.13 -m uvicorn webapp.server:app --reload --port 8000
(the frontend static/ is also served at / for local use)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import predict_all, cis_matched_rule, all_rules
from metrics import load_metrics, clean_nan

ART = os.path.join(HERE, "artifacts", "engine.joblib")
app = FastAPI(title="CISCaRL Live API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        if not os.path.exists(ART):
            raise HTTPException(503, "Model artifacts not built. Run webapp/train_persist.py")
        _engine = joblib.load(ART)
    return _engine


class Patient(BaseModel):
    patient: dict


@app.get("/api/meta")
def meta():
    e = get_engine()
    return {
        "feature_names": e["feature_names"],
        "ranges": e["ranges"],
        "demo_patient": e["demo_patient"],
        "t_star": e["t_star"],
        "dataset": "ACTG175 (Hammer et al. 1996) - real randomized HIV trial",
        "meta": e["meta"],
    }


@app.get("/api/rules")
def rules():
    e = get_engine()
    return all_rules(e)


@app.get("/api/metrics")
def metrics(regime: str = "rescaled"):
    out = load_metrics(regime)
    e = get_engine()
    # the exact metrics of THIS demo's configuration (matches the offline CSV row)
    out["canonical"] = {
        "setting": f"{e['canonical']['dataset']} / {e['canonical']['dgp']} / "
                   f"{e['canonical']['regime']} / seed {e['canonical']['seed']}",
        "methods": clean_nan(e["instance_metrics"]),
    }
    return out


@app.post("/api/predict")
def predict(body: Patient):
    e = get_engine()
    preds = predict_all(e, body.patient)
    matched = cis_matched_rule(e, body.patient)
    # out-of-range warnings vs ACTG175 ranges
    warns = []
    for c, v in body.patient.items():
        if c in e["ranges"]:
            r = e["ranges"][c]
            if v < r["min"] or v > r["max"]:
                warns.append(f"{c}={v} outside observed range [{r['min']:.3g}, {r['max']:.3g}]")
    return {"predictions": preds, "cis_matched_rule": matched,
            "t_star": e["t_star"], "warnings": warns}


@app.get("/api/health")
def health():
    return {"ok": True, "artifacts": os.path.exists(ART)}


# serve the static frontend at / (for local development)
STATIC = os.path.join(HERE, "static")
if os.path.isdir(STATIC):
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

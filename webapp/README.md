# CISCaRL Live — web demo

An interactive web app: enter a patient, run every comparison method, and read
the **rules** each method discovered. Backend = FastAPI; frontend = static
HTML/CSS/JS (Chart.js).

## What it shows
- **CISCaRL result** — the matched rule, its group effect, 90% conformal interval
  (with a CI bar), stability, support, and a treat/avoid recommendation.
- **Model predictions** — every method's predicted CATE at the horizon `t*`.
- **The rules** — every method's discovered rules (CISCaRL rich cards; competitors
  as weighted rule terms, top 25 by |coefficient|). Rules are the asset.
- **Accuracy** — the committed benchmark table (`original` regime) with per-metric
  ranking and a radar chart.

Models are fit once on the **real ACTG175** HIV randomized trial and served from
a persisted artifact. Accuracy is measured on the semi-synthetic ACTG175
benchmark (ground truth CATE), which is why it is a separate panel from the
per-patient predictions.

## Run locally

```bash
# 1) build the model artifact (once, ~8 min)
py -3.13 webapp/train_persist.py
py -3.13 webapp/export_static.py

# 2) serve
py -3.13 -m uvicorn webapp.server:app --port 8000
# open http://127.0.0.1:8000/
```

FastAPI also serves the frontend at `/`, so the same URL gives the UI and the API.

## Deploy (Option A)

- **Frontend → GitHub Pages** (branch `live`): the workflow
  `.github/workflows/pages-live.yml` publishes `webapp/static` at the site root
  and the visual guide at `/guide/`.
- **Backend → Hugging Face Spaces** (Docker): create a Space (SDK = Docker) from
  this repo; the root `Dockerfile` builds it. It listens on port 7860.
- **Connect them:** set the backend URL in `webapp/static/config.js`:
  ```js
  window.CISARL_API = "https://<user>-<space>.hf.space";
  ```
  If left empty, the page talks to same-origin `/api` (local) and falls back to
  the static `data.json` (so Pages still shows rules + accuracy without a backend).

## Files
| File | Purpose |
|------|---------|
| `engine.py` | fit all methods on ACTG175, extract readable rules, predict one patient |
| `train_persist.py` | build `artifacts/engine.joblib` |
| `export_static.py` | write `static/data.json` (offline fallback) |
| `metrics.py` | load benchmark CSV (original regime) into a metric table |
| `server.py` | FastAPI: `/api/meta`, `/api/rules`, `/api/metrics`, `/api/predict` |
| `static/` | the UI (`index.html`, `styles.css`, `app.js`, `config.js`, `data.json`) |

## Honest notes
- Predictions are **group-level** (the matched rule's subgroup mean), not
  individual. The UI labels this.
- Inputs are validated against ACTG175 ranges (out-of-range → warning).
- Public data only (no PHI); illustrative demo, not a clinical tool.

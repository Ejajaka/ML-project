---
title: CISCaRL Live
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Interpretable survival treatment-effect rules (backed by the real ACTG175 RCT)
---

# CISCaRL Live — backend API

FastAPI backend for the **CISCaRL Live** demo. It serves predictions for every
comparison method (fit once on the real **ACTG175** HIV randomized trial), the
discovered **rules** for each method, and the benchmark **accuracy** table.

Endpoints:
- `GET /api/meta` — features, ranges, demo patient, horizon t*
- `GET /api/rules` — every method's discovered rules
- `GET /api/metrics` — accuracy table (original regime)
- `POST /api/predict` — `{ "patient": { ... } }` → predictions + matched CISCaRL rule
- `GET /api/health`

The public frontend is at https://ejajaka.github.io/ML-project/ and calls this
API. Predictions are **group-level** effects (the matched rule's subgroup mean),
for illustration only — not a clinical tool. Public data only (no PHI).

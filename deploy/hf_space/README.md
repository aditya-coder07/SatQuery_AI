---
title: SatQuery AI API
emoji: 🛰️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: other
---

# SatQuery AI — API (CPU profile)

FastAPI backend of [SatQuery AI](https://github.com/aditya-coder07/SatQuery_AI)
running the `cpu` profile: land cover, change mask, optical–SAR fusion,
grounding, caption, change caption and change VQA specialists on CPU; the
3B vision-language model is shed (VQA questions receive the index-based
narrative). Weights are pulled from a private model repo at start.

Endpoints: `/health`, `/runs` (multipart: `query`, `images[]`), `/runs/stream`,
`/models`, `/benchmarks`. The web UI is deployed separately (Vercel) and
points here through `NEXT_PUBLIC_API_URL`.

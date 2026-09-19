# Free, all-cloud deployment: Vercel frontend + Modal serverless GPU

Nothing runs on a laptop or the lab cluster. The backend needs a CUDA GPU
(4.2 GB VRAM measured in serving) and ≈ 10 GB of weights; no cloud gives
that free *and always-on*, so the layout that is free and runs the real
models is a **scale-to-zero GPU**: the container exists only while a
request is being served.

```
browser ──HTTPS──► Vercel (Next.js, free Hobby)
                      │  NEXT_PUBLIC_API_URL
                      ▼
      https://<workspace>--satquery-api-api.modal.run   (Modal, T4, scale-to-zero)
                      │
                      ▼  Volumes: satquery-weights (10 GB), satquery-runs (db + previews)
```

| | Modal (this doc) | Hugging Face Spaces, CPU |
|---|---|---|
| cost | $30 credit **per month**, no card | free |
| models | real v3 stack on a T4 | no GPU → `SATQUERY_PROFILE=lite` only (no VLM answers) |
| availability | first request after idle waits ≈ 30–60 s for weights to load, then normal | sleeps after 48 h idle |
| capacity | ≈ 50 T4-hours/month, billed only while serving → thousands of MVP queries | unlimited, degraded |

`deploy/modal_app.py` is the deployment; it serves the same FastAPI app as
the Docker image with `configs/deploy.v3.yaml` resolved against the
weights Volume, and mirrors `docker/api.Dockerfile` (gpu-image) for the
Python stack.

## 1. Modal account and weights (once, ≈ 20 min, mostly upload time)

From a machine that has the weights (the cluster copy `~/satquery` is the
fastest uplink; the repo root on a laptop works too):

```bash
pip install modal
modal token new            # opens the browser: sign up (GitHub login), free plan, no card
```

```bash
modal volume create satquery-weights
modal volume put satquery-weights models/qwen25_vl_3b        models/qwen25_vl_3b
modal volume put satquery-weights checkpoints/v3             checkpoints/v3
modal volume put satquery-weights checkpoints/v2             checkpoints/v2
modal volume put satquery-weights checkpoints/change_caption checkpoints/change_caption
```

```bash
modal volume ls satquery-weights checkpoints/v3      # expect the eight v3 run directories
```

## 2. Deploy the API (≈ 5 min first time: image build; seconds afterwards)

From the repo root:

```bash
SATQUERY_CORS_ORIGINS=https://satquery-ai.vercel.app modal deploy deploy/modal_app.py
```

It prints the public URL, `https://<workspace>--satquery-api-api.modal.run`.
Check it (the first call pays the cold start):

```bash
curl https://<workspace>--satquery-api-api.modal.run/health
```

Knobs (environment variables at deploy time): `SATQUERY_MODAL_GPU`
(`T4` default; `A10G` ≈ 2× faster, ≈ 2× the per-second price),
`SATQUERY_MODAL_IDLE_S` (300: how long a warm container waits for the next
request before it is released; higher = fewer cold starts, more credit).

## 3. Frontend on Vercel (≈ 3 min)

1. vercel.com → *Add New → Project* → import `aditya-coder07/SatQuery_AI`.
2. **Root Directory `frontend`** (Next.js auto-detected; the Dockerfile is not used).
3. Environment variable: `NEXT_PUBLIC_API_URL = https://<workspace>--satquery-api-api.modal.run` (no trailing slash; inlined at build → change it, then *Redeploy*).
4. Project name `satquery-ai`, or redeploy the API with the origin Vercel assigned:
   `SATQUERY_CORS_ORIGINS=https://<name>.vercel.app modal deploy deploy/modal_app.py`.

CLI equivalent, from `frontend/` after `vercel login`:

```bash
vercel --prod -e NEXT_PUBLIC_API_URL=https://<workspace>--satquery-api-api.modal.run
```

## 4. Smoke test

Open `https://satquery-ai.vercel.app`: the header shows `CUDA:0 · n GB
free` once the API answers. *Ask a question* → attach `test.png` → "Is
there a road in this image?" → an answer with confidence and trace. The
first query after idle takes about a minute; the next ones a few seconds.

## What to expect

* **Cold start** ≈ 30–60 s after 5 min idle (weights load from the
  Volume). Raise `SATQUERY_MODAL_IDLE_S` for a demo session so it stays warm.
* **One GPU, in-process execution**: concurrent requests queue; a full
  Cartosat scene is ≈ 60 s. Right for an MVP, not for many simultaneous users.
* **No authentication** on the API: anyone with the URL can call it and
  spend the credit. Modal can require a proxy-auth token per endpoint
  (`modal.web_endpoint(requires_proxy_auth=True)`) when the MVP is shared
  beyond the team; the frontend would then send the token header.
* **Run records** (`satquery_runs.db`, previews) live on the `satquery-runs`
  Volume; SQLite there is fine for one container, which is what
  `max_inputs=4` on a single function gives.
* When the monthly credit is exhausted the endpoint returns errors until the
  next month or a card is added; the frontend on Vercel stays up.

## What does not work for free

| Option | Why |
|---|---|
| Render / Railway / Fly / Vercel functions for the API | CPU only, 0.5–2 GB RAM |
| Hugging Face Spaces (free) | CPU only; 4-bit loading needs CUDA; creating ZeroGPU Spaces needs PRO |
| Oracle always-free ARM VM | CPU only → lite profile only; a valid 24/7 fallback without VLM answers |
| Colab / Kaggle GPU + tunnel | demos only; sessions expire, URL rotates |
| AWS / GCP / Azure trial credits | real GPUs but one-off credits (90 days), then paid |

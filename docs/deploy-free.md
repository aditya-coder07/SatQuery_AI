# Free, all-cloud deployment (no card): Vercel frontend + Kaggle GPU backend

**What is free on Hugging Face without PRO, measured on the account
`DeepakShivhareEe` on 2026-09-19:** creating a `static` Space succeeds;
creating a `gradio` or `docker` Space returns `402 Payment Required`
("hosting Gradio and Docker Spaces on free cpu-basic requires a PRO
subscription"). A static Space serves files only, so the API cannot run on
Hugging Face for free. What HF still provides free, and what this layout
uses: **the private model repo that holds every trained checkpoint**
(`DeepakShivhareEe/satquery-cpu-weights`: the 7 v3/v2/v1 heads and the 4
v3 VLM adapters, 38 files).

The only free GPU that needs no card is a **Kaggle** session (T4, 30
h/week, ≤ 12 h per session). So:

```
browser ──► https://satquery-ai.vercel.app            (Vercel Hobby, free, always on)
                 │  API endpoint set at runtime: ?api=<url> or the header chip
                 ▼
     https://<random>.trycloudflare.com               (Cloudflare quick tunnel, free)
                 ▼
     Kaggle notebook: T4, full Phase 6 stack           (free; up while the notebook runs)
        code   ← GitHub main
        base   ← Qwen/Qwen2.5-VL-3B-Instruct (HF Hub)
        weights← DeepakShivhareEe/satquery-cpu-weights (private HF repo, HF_TOKEN secret)
```

This runs the **complete v3 system** — 4-bit Qwen2.5-VL-3B with the VQA /
grounding (arm E) / caption / change-caption adapters plus the v3 heads —
not the CPU-degraded set. The trade: the backend is up only while you have
a Kaggle session running, and its URL changes per session, which is why
the frontend accepts the endpoint at runtime.

## 1. Frontend on Vercel (once, ≈ 3 min)

1. vercel.com → *Add New → Project* → import `aditya-coder07/SatQuery_AI`.
2. Project name **`satquery-ai`**; **Root Directory `frontend`**.
3. No environment variable needed: the API endpoint is set in the browser
   (below). Optionally `NEXT_PUBLIC_API_URL` as a default for a fixed backend.
4. Deploy → `https://satquery-ai.vercel.app`.

## 2. Backend on Kaggle (each session, ≈ 10 min to come up)

1. kaggle.com → *Create → New Notebook* → *File → Import Notebook* →
   upload `deploy/kaggle/satquery_api.ipynb` (or paste its one cell).
2. Notebook settings: **Accelerator GPU T4 x2**, **Internet On**.
3. *Add-ons → Secrets*: `HF_TOKEN` = a Hugging Face **read** token
   (huggingface.co/settings/tokens) — the checkpoints are in a private repo.
4. Run the cell. It clones `main`, installs the pinned stack, downloads the
   base model (7 GB, from the Hub) and the checkpoints (1.9 GB), runs
   `verify_deploy.py` (must print `8/8 loaded`), starts the API and a
   Cloudflare quick tunnel, then prints:
   ```
   API:      https://<random>.trycloudflare.com
   frontend: https://satquery-ai.vercel.app/query?api=https://<random>.trycloudflare.com
   ```
5. Open the printed frontend link. The `?api=` value is saved in that
   browser; afterwards `https://satquery-ai.vercel.app` alone works until
   the next session. The header chip (`CUDA:0 · n GB FREE`) shows which
   endpoint is in use; click it to change or reset it. When the backend is
   down the chip reads `API OFFLINE`.

Keep the cell running; stop it (or let Kaggle's 12 h limit end it) to
close the session. Restart = run the cell again, share the new link.

CORS: the API allows `https://satquery-ai.vercel.app` (and localhost) by
default (`SATQUERY_CORS_ORIGINS` in `deploy/kaggle/satquery_kaggle.py`);
set the env var in the notebook if the Vercel name differs.

## 3. Smoke test

From the frontend link: attach `test.png` (or the LEVIR pair) → "Is there a
road in this image?" → an answer with confidence and trace from the real
VLM; "What changed between these two dates?" → change mask + caption.

## Expectations

* First request after start: models already loaded by the verify step, so
  seconds. Grounding at 1024 px ≈ 5–8 s on a T4, VQA ≈ 2–4 s.
* One GPU, in-process: concurrent users queue.
* Quick-tunnel URLs are random per start and carry no authentication:
  share the link only with the people who should use the session.
* Kaggle quota: 30 GPU-hours per week, 12 h per session.

## When a card or a subscription becomes acceptable

Everything below is already built and needs only the account step:

| Layout | Files | Cost | Notes |
|---|---|---|---|
| Hugging Face **ZeroGPU** or CPU Space | `deploy/hf_space/` (+ ZeroGPU: GPU map) | PRO $9/month | always on, full models on ZeroGPU |
| **Modal** serverless T4 | `deploy/modal_app.py` (image built, weights on the Volume) | card on file, $30/month free credit | scale-to-zero, ≈ 50 T4-hours/month free |
| Own GPU box / cluster | `docker-compose.v3.yml` or `scripts/serve_local.py` | — | as verified 8/8 on compute01 |

CPU-only hosts (any tier) use `SATQUERY_PROFILE=cpu` with
`configs/deploy.cpu.yaml`: every specialist runs, the 3B VLM is shed
(7/7 load-verified; rehearsal inputs answer in 0.5–7 s).

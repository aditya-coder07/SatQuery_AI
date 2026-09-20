# Free, all-cloud deployment (no card): Vercel frontend + a free CPU (Lightning AI) or GPU session (Kaggle)

**What is free on Hugging Face without PRO, measured on the account
`DeepakShivhareEe` on 2026-09-19:** creating a `static` Space succeeds;
creating a `gradio` or `docker` Space returns `402 Payment Required`
("hosting Gradio and Docker Spaces on free cpu-basic requires a PRO
subscription"). A static Space serves files only, so the API cannot run on
Hugging Face for free. What HF still provides free, and what this layout
uses: **the private model repo that holds every trained checkpoint**
(`DeepakShivhareEe/satquery-cpu-weights`: the 7 v3/v2/v1 heads and the 4
v3 VLM adapters, 38 files).

Two free hosts need no card: a **Lightning AI CPU Studio** (always available, slow — option A) and a **Kaggle GPU session** (fast, up while you run it — option B). Both serve the same complete v3 stack and the same Vercel frontend:

```
browser ──► https://satquery-ai-self.vercel.app            (Vercel Hobby, free, always on)
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

## The whole system runs on a CPU (slowly)

Since 2026-09-19 the VLM loader (`satquery/tools/rs_vqa.py`) takes a bf16
CPU path when there is no CUDA device (the 4-bit path needs bitsandbytes,
which is CUDA-only). The same adapters attach unchanged, so **all 9 tools
run on a CPU with ≥ 12 GB RAM** from the ordinary `configs/deploy.v3.yaml`
map — the `cpu` profile (VLM shed) is no longer required, only faster.
Measured on a laptop CPU (8 threads, other work running), one request at a
time:

| Path | Time on CPU | Same on a T4 |
|---|---|---|
| VQA (`rs_vqa`, short answer) | 25 s | 2–4 s |
| VQA, long answer (≈ 80 tokens) | ≈ 4.5 min | ≈ 6 s |
| Grounding, arm E at 1024 px | ≈ 4.7 min | 5–8 s |
| VLM caption + land cover | ≈ 4.6 min | ≈ 5 s |
| VLM change caption (pair) + change mask | ≈ 9.7 min | ≈ 8 s |
| change mask + change VQA | 1.3 s | < 1 s |
| optical–SAR fusion | 4 s | < 1 s |

Knobs for a CPU host: `SATQUERY_GROUNDING_MIN_PIXELS=262144` makes
grounding ≈ 4× faster at some accuracy cost (the deployed number was
measured at 1048576); `SATQUERY_VLM_CPU_DTYPE=float32` is faster on CPUs
without bf16 arithmetic but needs ≈ 14 GB for the base model;
`SATQUERY_THREADS` caps torch threads. `SATQUERY_CAPTION_BEAMS` sets the
caption beam width (default 1 = greedy, the measured choice:
`docs/research/model_improvement_report.md` E3).

## Option A — always-on free CPU: Lightning AI Studio

Lightning AI's free tier gives a persistent **CPU Studio (4 cores, 16 GB
RAM, disk)** with a **public URL per exposed port**, plus monthly credits
that can switch the same Studio to a T4 for a demo. Sign-up needs phone
verification, not a card. Free Studios auto-sleep after inactivity and
wake on the next request (≈ 1 min).

1. lightning.ai → New Studio (CPU) → open the terminal.
2. ```bash
   git clone https://github.com/aditya-coder07/SatQuery_AI.git && cd SatQuery_AI
   ```
   ```bash
   export HF_TOKEN=hf_...   # a Hugging Face READ token (the checkpoints are in a private repo)
   ```
   ```bash
   bash deploy/lightning/setup.sh
   ```
   (installs the pinned stack, downloads the base model and checkpoints,
   prints `8/8 loaded`; ≈ 10 min.)
3. ```bash
   bash deploy/lightning/serve.sh
   ```
   then in the Studio's **Ports** panel expose **8000** publicly → a URL
   like `https://8000-<studio-id>.cloudspaces.litng.ai`.
4. Open `https://satquery-ai-self.vercel.app/query?api=<that URL>` once; the
   browser remembers it.

Switching the Studio to a GPU (credits) and rerunning `serve.sh` serves
the 4-bit path at GPU speed with no other change.

Dependency note (2026-09-19): Lightning Studios ship `pandas 2.1.4` and
`matplotlib 3.8.2` built against the numpy 1.x ABI. `requirements.txt`
therefore pins `numpy==1.26.4` (with `rasterio==1.4.3`, the last rasterio
on that ABI) so `setup.sh` installs cleanly beside them; a numpy 2 pin
produced `numpy.dtype size changed, may indicate binary incompatibility`.
The checkpoints, written under numpy 2, load on 1.26 through the
`numpy._core` shim in `training/common/checkpointing.py`. Verified in a
Python 3.12 CPU image built from the pinned file with those Studio
packages present: imports OK, `pip check` clean, 8/8 tools, VQA and
change-detection requests answered, test suite green.

## Frontend on Vercel (both options; once, ≈ 3 min)

1. vercel.com → *Add New → Project* → import `aditya-coder07/SatQuery_AI`.
2. Project name (Vercel assigned **`satquery-ai-self`**; the deployed site is `https://satquery-ai-self.vercel.app`); **Root Directory `frontend`**.
3. No environment variable needed: the API endpoint is set in the browser
   (below). Optionally `NEXT_PUBLIC_API_URL` as a default for a fixed backend.
4. Deploy → `https://satquery-ai-self.vercel.app`.

## Option B — free GPU sessions: Kaggle (each session, ≈ 10 min to come up)

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
   frontend: https://satquery-ai-self.vercel.app/query?api=https://<random>.trycloudflare.com
   ```
5. Open the printed frontend link. The `?api=` value is saved in that
   browser; afterwards `https://satquery-ai-self.vercel.app` alone works until
   the next session. The header chip (`CUDA:0 · n GB FREE`) shows which
   endpoint is in use; click it to change or reset it. When the backend is
   down the chip reads `API OFFLINE`.

Keep the cell running; stop it (or let Kaggle's 12 h limit end it) to
close the session. Restart = run the cell again, share the new link.

CORS: the API allows `https://satquery-ai-self.vercel.app` (and localhost) by
default (`SATQUERY_CORS_ORIGINS` in `deploy/kaggle/satquery_kaggle.py`);
set the env var in the notebook if the Vercel name differs.

## Smoke test

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

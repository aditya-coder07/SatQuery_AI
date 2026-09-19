# Free, all-cloud deployment: Hugging Face Space (API) + Vercel (frontend)

No laptop, no cluster, no card. The API runs on a free Hugging Face Docker
Space (2 vCPU, 16 GB RAM, no GPU) under the **`cpu` profile**; the web UI
runs on Vercel's free Hobby plan.

```
browser ──HTTPS──► https://satquery-ai.vercel.app        (Vercel, Next.js)
                        │  NEXT_PUBLIC_API_URL
                        ▼
      https://<user>-satquery-api.hf.space               (HF Docker Space, port 7860)
                        │  at start: snapshot_download(<user>/satquery-cpu-weights)
                        ▼
                 /app/checkpoints  (1.4 GB: v3 heads + v2/v1 specialists)
```

## What runs, and what does not

| Tool | On the Space | Model |
|---|---|---|
| landcover | yes | v3 SSL4EO head (`landcover_full/best.pt`) — official test micro mAP 0.885 |
| change_mask | yes | v3 (`change_mask/best.pt`) — LEVIR-CD F1 0.904 |
| optsar_fusion | yes | v3 (`optsar_fusion/best.pt`) |
| grounding | yes | **v2 specialist** (`grounding_pre`, 32 M params) — the arm-E VLM adapter needs the 3B base |
| caption | yes | **v2 specialist** (`caption_pre`, BLEU-4 0.174; the VLM adapter scores 0.256 on GPU) |
| change_caption | yes | v1 specialist (`change_caption`) |
| change_vqa | yes | v2 (`change_vqa/best.pt`) |
| index_engine | yes | deterministic |
| rs_vqa | **shed** | the 3B VLM cannot run in useful time on a CPU; a VQA question is answered with an explicit "this profile cannot answer" abstain plus the index narrative where one exists |

Measured on a laptop CPU with the GPU hidden: every rehearsal input in
`data/demo_bundle` answers in 0.3–6 s; the ingest gates (footprint
overlap, cloud cover) reject as on GPU. `scripts/verify_deploy.py --map
configs/deploy.cpu.yaml` → 7/7 loaded on CPU.

Files: `configs/profiles/cpu.yaml` (profile), `configs/deploy.cpu.yaml`
(which checkpoint each tool loads), `deploy/hf_space/{Dockerfile,start.sh,README.md}`
(the Space).

## 1 + 2. Weights and Space in one command (≈ 15 min, mostly upload)

The CPU weight set is staged at `C:\Users\dk231\Desktop\SatQuery_AI\hf_stage\checkpoints`
(1.39 GB, 25 files; on the cluster the same set is `~/satquery/.hf_stage/checkpoints`).
It goes to a **private** model repo because the grounding specialist was
trained on DIOR-RSVG (CC-BY-NC).

```bash
pip install -U "huggingface_hub[cli]"
```
```bash
hf auth login
```
(paste a **write** token from huggingface.co/settings/tokens)
```bash
python deploy/hf_space/publish.py --weights "C:/Users/dk231/Desktop/SatQuery_AI/hf_stage/checkpoints"
```

`publish.py` creates `<user>/satquery-cpu-weights` (private), uploads the
weights, creates the Docker Space `<user>/satquery-api` from
`deploy/hf_space/`, sets the `HF_TOKEN` secret and the
`SATQUERY_WEIGHTS_REPO` / `SATQUERY_CORS_ORIGINS` / `SATQUERY_GIT_REF`
variables, and restarts it. It prints the API URL. Re-running updates in
place (`--skip-weights` to leave the weights alone, `--cors` to change the
allowed origin, `--read-token` to give the Space a read-only token).

The Space then builds (clones `main`, installs the CPU torch stack, ≈ 10
min), downloads the weights (≈ 1 min) and serves on 7860. Check:
`https://<user>-satquery-api.hf.space/health` → `{"status":"ok"}`.

Manual equivalent, if preferred: New Space → Docker → Blank → CPU basic;
upload `Dockerfile`, `start.sh`, `README.md`; Settings → secret `HF_TOKEN`
(read token), variables `SATQUERY_WEIGHTS_REPO`, `SATQUERY_CORS_ORIGINS`.

## 3. Frontend on Vercel (≈ 3 min)

1. vercel.com → *Add New → Project* → import `aditya-coder07/SatQuery_AI`.
2. Project name **`satquery-ai`**; **Root Directory `frontend`**.
3. Environment variable `NEXT_PUBLIC_API_URL` = `https://<user>-satquery-api.hf.space` (no trailing slash; inlined at build → change it, then *Redeploy*).
4. Deploy. If Vercel assigns a different name, set that origin in the
   Space's `SATQUERY_CORS_ORIGINS` variable (the Space restarts by itself).

## 4. Smoke test

Open `https://satquery-ai.vercel.app` → *Ask a question* → attach two
LEVIR tiles or `test.png` → "What changed between these two dates?" /
"Describe the land cover." A VQA question ("Is there a road?") returns the
honest abstain of the cpu profile.

## Expectations

* **Free Space sleeps after 48 h without traffic**; the first request then
  takes ≈ 2 min (container start + weight download). Any visit wakes it.
* CPU: single requests 1–6 s on 256-px tiles; a full Cartosat scene is
  tiled at 256 px and will take minutes — the UI streams progress.
* **No VLM answers** on this tier. The GPU layouts (`docker-compose.v3.yml`,
  `deploy/modal_app.py`) are unchanged and take over when a GPU is available:
  Modal needs a card on file (free $30/month credit), HF ZeroGPU needs PRO.
* The API has no authentication. Anyone with the Space URL can call it;
  fine for an MVP, add a gate before sharing widely.

## Why not …

| Option | Why not |
|---|---|
| HF ZeroGPU Space | creating one needs a PRO account ($9/month) |
| Modal / GCP / AWS free credits | require a payment method |
| Render / Railway / Fly free | CPU with 0.5–2 GB RAM: too small even for the specialists |
| Kaggle / Colab GPU | real T4 for free, but sessions expire and need a tunnel: demo-only |

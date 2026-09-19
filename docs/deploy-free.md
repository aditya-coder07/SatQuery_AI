# Free deployment: Vercel frontend + your own GPU behind a tunnel

The backend needs a CUDA GPU (≈ 4–6 GB VRAM in serving, measured 4.2 GB
peak on an RTX 4050) and ≈ 10 GB of weights (`models/qwen25_vl_3b` 7.1 GB,
`checkpoints/{v2,v3,change_caption}` ≈ 2.5 GB). No free hosting tier gives
that permanently, so the free layout is:

```
browser ──HTTPS──► Vercel (Next.js, free Hobby)
                      │  NEXT_PUBLIC_API_URL
                      ▼
            https://<tunnel-host>  ──► localhost:8000 on the GPU machine
```

Everything below is copy-paste; each step is verified except the two that
only the account owner can perform (tunnel start, Vercel login).

## 1. Backend on the GPU machine (laptop or compute01)

```bash
# once: weights in place (models/, checkpoints/v2, checkpoints/v3, checkpoints/change_caption)
python scripts/verify_deploy.py --map configs/deploy.v3.yaml      # must print 8/8 loaded
```

Start the API with CORS limited to the Vercel origin you will create in
step 3 (pick the project name now; `satquery-ai` → `https://satquery-ai.vercel.app`):

```bash
SATQUERY_CORS_ORIGINS=https://satquery-ai.vercel.app,http://localhost:3000 python scripts/serve_local.py --map configs/deploy.v3.yaml --host 127.0.0.1 --port 8000
```

PowerShell equivalent:

```powershell
$env:SATQUERY_CORS_ORIGINS = "https://satquery-ai.vercel.app,http://localhost:3000"; python scripts/serve_local.py --map configs/deploy.v3.yaml --host 127.0.0.1 --port 8000
```

`--host 127.0.0.1` is deliberate: the API is reachable only through the
tunnel, never directly on the LAN.

## 2. Tunnel (free, HTTPS)

Preferred: **Cloudflare quick tunnel** — no account, no interstitial page,
works for `<img>` previews and streaming.

```powershell
winget install --id Cloudflare.cloudflared
```

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints `https://<random>.trycloudflare.com`. The hostname changes every
time the tunnel restarts, so keep the process running; a permanent
hostname needs a free Cloudflare account plus a domain (named tunnel:
`cloudflared tunnel login` → `cloudflared tunnel create satquery` → route
`api.<your-domain>`), and Cloudflare Access (free ≤ 50 users) can then gate
it with a login, since the API has no authentication of its own.

Alternative already installed on the dev laptop: `ngrok http 8000`. Its
free plan shows a warning page on browser requests, which breaks image
previews unless every request carries `ngrok-skip-browser-warning`; use
Cloudflare unless you have an ngrok static domain.

Check from outside:

```bash
curl https://<tunnel-host>/health
```

## 3. Frontend on Vercel

1. vercel.com → *Add New → Project* → import `aditya-coder07/SatQuery_AI`.
2. **Root Directory: `frontend`** (framework Next.js is auto-detected;
   build `next build`, output default). The Dockerfile is not used.
3. Environment variable (all environments):
   `NEXT_PUBLIC_API_URL = https://<tunnel-host>` — no trailing slash.
   It is inlined at build time, so change it → *Redeploy*.
4. Deploy. Project name must match the origin you put in
   `SATQUERY_CORS_ORIGINS` in step 1 (or restart the API with the name
   Vercel assigned).

CLI equivalent from `frontend/` (after `vercel login`):

```bash
vercel --prod -e NEXT_PUBLIC_API_URL=https://<tunnel-host>
```

## 4. Smoke test

Open `https://<project>.vercel.app` → the header shows the GPU telemetry
(`CUDA:0 · n GB free`) when the API is reachable. *Ask a question* →
attach `test.png` → "Is there a road in this image?" → answer with
confidence and the trace.

## Limits of the free layout

* The service is up only while the GPU machine and the tunnel are up.
* Quick-tunnel hostnames rotate on restart → one Vercel redeploy each time.
* One GPU, in-process execution: requests queue; a full Cartosat scene is
  ≈ 60 s. Fine for a demo/MVP, not for concurrent users.
* Anyone with the URL can call the API. Put Cloudflare Access (free) in
  front before sharing it beyond the team.

## What does not work for free

| Option | Why |
|---|---|
| Render / Railway / Fly / Vercel functions for the API | CPU only, 0.5–2 GB RAM |
| Hugging Face Spaces (free) | CPU only; 4-bit loading needs CUDA; ZeroGPU creation needs PRO |
| Oracle always-free ARM VM | CPU only → `SATQUERY_PROFILE=lite` only (index engine + small heads, no VLM answers) — a valid 24/7 fallback |
| Colab / Kaggle GPU + tunnel | works for demos; sessions expire (12 h / 30 h per week) and the URL changes |

"""SatQuery API on a free Kaggle GPU session, reachable from the internet.

Runs the full Phase 6 stack (configs/deploy.v3.yaml: 4-bit Qwen2.5-VL-3B with
the task adapters, plus the v3 heads) on the session's T4, and exposes it
through a Cloudflare quick tunnel. Nothing is stored on a laptop or a
cluster: code from GitHub, the base model from the Hugging Face Hub, the
trained checkpoints from a private HF model repo.

Kaggle notebook settings: Accelerator GPU T4 x2 (or P100), Internet ON,
Secret `HF_TOKEN` (a read token for the private weights repo) added under
Add-ons -> Secrets. Then one cell:

    !git clone --depth 1 https://github.com/aditya-coder07/SatQuery_AI.git /kaggle/working/satquery
    %run /kaggle/working/satquery/deploy/kaggle/satquery_kaggle.py

Prints the public API URL and a ready-made frontend link
(`https://satquery-ai.vercel.app/query?api=<url>`). The session ends after
Kaggle's limit (12 h) or when the notebook is stopped; the URL changes on
every start, which is why the frontend accepts `?api=` at runtime.

Environment overrides: SATQUERY_WEIGHTS_REPO (default
DeepakShivhareEe/satquery-cpu-weights - it holds the GPU set too),
SATQUERY_CORS_ORIGINS, SATQUERY_FRONTEND (the Vercel origin used in the link).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(os.environ.get("SATQUERY_REPO", "/kaggle/working/satquery"))
WEIGHTS_REPO = os.environ.get("SATQUERY_WEIGHTS_REPO", "DeepakShivhareEe/satquery-cpu-weights")
BASE_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
FRONTEND = os.environ.get("SATQUERY_FRONTEND", "https://satquery-ai.vercel.app")
CORS = os.environ.get("SATQUERY_CORS_ORIGINS", f"{FRONTEND},http://localhost:3000")
PORT = 8000
PINS = ["transformers==5.15.1", "peft==0.20.0", "bitsandbytes==0.50.2", "accelerate==1.14.0", "timm==1.0.29",
        "huggingface_hub>=0.30"]


def sh(cmd: str, **kw) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def hf_token() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001 - outside Kaggle or no secret: fall back to the env
        return os.environ.get("HF_TOKEN")


def main() -> int:
    if not REPO.exists():
        sh(f"git clone --depth 1 https://github.com/aditya-coder07/SatQuery_AI.git {REPO}")
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    print("== python packages", flush=True)
    sh(f"{sys.executable} -m pip install -q -r requirements.txt " + " ".join(f'"{p}"' for p in PINS))
    import torch

    assert torch.cuda.is_available(), "no CUDA device: set the notebook Accelerator to GPU"
    print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)

    print("== weights", flush=True)
    from huggingface_hub import snapshot_download

    token = hf_token()
    if token is None:
        print("no HF_TOKEN secret: the private checkpoints cannot be downloaded", file=sys.stderr)
        return 1
    snapshot_download(BASE_REPO, local_dir=str(REPO / "models" / "qwen25_vl_3b"), token=token,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    snapshot_download(WEIGHTS_REPO, repo_type="model", local_dir=str(REPO), token=token,
                      allow_patterns=["checkpoints/**"])
    n = sum(1 for _ in (REPO / "checkpoints").rglob("*") if _.is_file())
    print(f"checkpoints: {n} files", flush=True)

    print("== load check", flush=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SATQUERY_CORS_ORIGINS": CORS}
    sh(f"{sys.executable} scripts/verify_deploy.py --map configs/deploy.v3.yaml", env=env)

    print("== api", flush=True)
    api = subprocess.Popen([sys.executable, "scripts/serve_local.py", "--map", "configs/deploy.v3.yaml",
                            "--host", "127.0.0.1", "--port", str(PORT)], env=env,
                           stdout=open(REPO / "api.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    else:
        print(open(REPO / "api.log").read()[-3000:], file=sys.stderr)
        return 1
    print("api healthy", flush=True)

    print("== tunnel", flush=True)
    cf = Path("/kaggle/working/cloudflared")
    if not cf.exists():
        sh(f"curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o {cf} && chmod +x {cf}")
    tun = subprocess.Popen([str(cf), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    deadline = time.time() + 90
    while time.time() < deadline and url is None:
        line = tun.stdout.readline()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line or "")
        if m:
            url = m.group(0)
    if url is None:
        print("tunnel did not report a URL", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"API:      {url}")
    print(f"health:   {url}/health")
    print(f"frontend: {FRONTEND}/query?api={url}")
    print("=" * 72 + "\n", flush=True)
    print("Session stays up while this cell runs (Kaggle limit 12 h). Stop the cell to end it.", flush=True)
    try:
        while True:
            time.sleep(300)
            alive = api.poll() is None and tun.poll() is None
            print(time.strftime("%H:%M:%S"), "api" if api.poll() is None else "API DIED",
                  "tunnel" if tun.poll() is None else "TUNNEL DIED", flush=True)
            if not alive:
                return 1
    except KeyboardInterrupt:
        pass
    finally:
        tun.terminate()
        api.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

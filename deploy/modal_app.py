"""SatQuery API on Modal (serverless GPU, scale-to-zero).

The same FastAPI app the Docker image runs (`satquery.api.main:app`), with
the Phase 6 deployment map (`configs/deploy.v3.yaml`) resolved against a
Modal Volume that holds the weights. The image mirrors docker/api.Dockerfile
(gpu-image): requirements.txt plus the pinned CUDA stack.

One-time setup (from the repo root):

    pip install modal
    modal token new                                   # opens the browser; free account, no card
    modal volume create satquery-weights
    modal volume put satquery-weights models/qwen25_vl_3b            models/qwen25_vl_3b
    modal volume put satquery-weights checkpoints/v3                 checkpoints/v3
    modal volume put satquery-weights checkpoints/v2                 checkpoints/v2
    modal volume put satquery-weights checkpoints/change_caption     checkpoints/change_caption

Deploy / update:

    modal deploy deploy/modal_app.py
    # prints https://<workspace>--satquery-api-api.modal.run  -> NEXT_PUBLIC_API_URL on Vercel

Set the browser origin that may call it (repeat after Vercel assigns the name):

    modal deploy deploy/modal_app.py   # after editing CORS_ORIGINS below, or
    SATQUERY_CORS_ORIGINS=https://satquery-ai-self.vercel.app modal deploy deploy/modal_app.py

Cost model: billed per second only while a request is being served plus the
SCALEDOWN_WINDOW after it; idle costs nothing. Free tier: $30/month.
Cold start (first request after idle): image is cached, weights load from
the Volume, ≈ 30-60 s; warm requests behave like the local server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
GPU = os.environ.get("SATQUERY_MODAL_GPU", "T4")          # T4 is the cheapest that fits (4-6 GB used); "A10G" is ~2x faster
SCALEDOWN_WINDOW = int(os.environ.get("SATQUERY_MODAL_IDLE_S", "300"))
CORS_ORIGINS = os.environ.get("SATQUERY_CORS_ORIGINS", "https://satquery-ai-self.vercel.app,https://satquery-ai.vercel.app,http://localhost:3000")

weights = modal.Volume.from_name("satquery-weights", create_if_missing=True)
runs = modal.Volume.from_name("satquery-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libexpat1", "libgl1", "gcc", "g++")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .pip_install(
        "torch==2.13.0+cu126", "torchvision==0.28.0+cu126",
        "transformers==5.15.1", "peft==0.20.0", "bitsandbytes==0.50.2",
        "accelerate==1.14.0", "timm==1.0.29",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "SATQUERY_PROFILE": "full",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SATQUERY_CORS_ORIGINS": CORS_ORIGINS})
    .add_local_dir(str(REPO), remote_path="/app",
                   ignore=["data", "checkpoints", "checkpoints_backup", "models", "artifacts", ".git",
                           "frontend/node_modules", "frontend/.next", "notebooks", "**/__pycache__", "*.db", "trace*.json"])
)

app = modal.App("satquery-api")


def _export_map(map_path: Path, weights_root: Path, app_root: Path) -> None:
    """Same resolution as scripts/serve_local.py, split across two roots:
    configs/* live in the image, everything else on the weights Volume."""
    import yaml

    spec = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    pairs: dict[str, str] = {}
    for tool in (spec.get("tools") or {}).values():
        for block in ("env", "fallback_env"):
            pairs.update({k: str(v) for k, v in (tool.get(block) or {}).items()})
    pairs.update({k: str(v) for k, v in (spec.get("registries") or {}).items()})
    missing = []
    for k, v in pairs.items():
        if v.strip().isdigit():
            os.environ[k] = v.strip()
            continue
        root = app_root if v.startswith("configs/") else weights_root
        p = root / v
        if not p.exists():
            missing.append(f"{k}={p}")
        os.environ[k] = str(p)
    if missing:
        raise RuntimeError("deployment map incomplete on the Volume:\n  " + "\n  ".join(missing))


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/weights": weights, "/runs": runs},
    memory=16384,
    timeout=600,
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=4)   # one GPU, in-process execution: requests queue behind each other
@modal.asgi_app()
def api():
    os.chdir("/app")
    sys.path.insert(0, "/app")
    _export_map(Path("/app/configs/deploy.v3.yaml"), Path("/weights"), Path("/app"))
    # The run store (satquery_runs.db) and per-run artifacts are cwd-relative;
    # point both at the runs Volume so they outlive the container.
    Path("/runs/artifacts").mkdir(parents=True, exist_ok=True)
    for name, target in (("artifacts", "/runs/artifacts"), ("satquery_runs.db", "/runs/satquery_runs.db")):
        link = Path("/app") / name
        if not link.exists() and not link.is_symlink():
            if name.endswith(".db"):
                Path(target).touch()
            link.symlink_to(target)
    from satquery.api.main import app as fastapi_app

    return fastapi_app

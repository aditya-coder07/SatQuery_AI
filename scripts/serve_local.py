"""Serve the API on this machine with a deployment map, no Docker.

Reads `configs/deploy.<tag>.yaml` (the same file `verify_deploy.py` checks),
exports every tool's `env` / `fallback_env` / `registries` entry with the
relative paths resolved against the repo root, then starts uvicorn. Missing
paths are reported and refused, so the process never starts on a partial map.

Usage::

    python scripts/serve_local.py --map configs/deploy.v3.yaml [--port 8000] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# `python scripts/serve_local.py` puts scripts/ first on sys.path, not the
# repo root. `satquery` is installed, but the tools import the model builders
# from `training/`, which is not - Docker gets it from WORKDIR /app.
# Running a script puts scripts/ on sys.path, not the repo root; the tools
# import training.* at load time (the image has /app as its working dir).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def export(map_path: Path) -> list[str]:
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
        p = (ROOT / v).resolve()
        if not p.exists():
            missing.append(f"{k}={p}")
        os.environ[k] = str(p)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", type=Path, default=ROOT / "configs" / "deploy.v3.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    missing = export(args.map)
    if missing:
        print("refusing to start, missing:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("SATQUERY_PROFILE", "full")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for k in sorted(os.environ):
        if k.startswith("SATQUERY_"):
            print(f"{k}={os.environ[k]}")
    import uvicorn

    uvicorn.run("satquery.api.main:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

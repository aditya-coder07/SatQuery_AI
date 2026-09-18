"""Machine-readable experiment registry.

One JSONL file, one record per experiment event, appended and never rewritten.
The registry exists so that "which checkpoint produced which number, on which
data, at which commit" is answerable from one file rather than from a memory
of the campaign. Every serious run appends a record at start (status
`running`) and again at the end (status `done` / `failed`), keyed by the same
`experiment_id`; readers take the latest record per id.

Fields follow docs/research/reproducibility.md. Anything not known at write
time is written as null rather than omitted, so a missing value is visible in
a diff instead of silently absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("artifacts/experiment_registry/registry.jsonl")

FIELDS = (
    "experiment_id", "timestamp", "git_commit", "dataset_versions",
    "dataset_manifest_hash", "model", "architecture", "hyperparameters",
    "training_steps", "seed", "hardware", "duration_s", "checkpoint",
    "checkpoint_sha256", "validation_metrics", "test_metrics", "status",
    "notes",
)


def git_commit(repo: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
            text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "host": platform.node(), "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_total_gb"] = round(total / 2**30, 1)
            info["gpu_free_at_start_gb"] = round(free / 2**30, 1)
    except Exception:  # torch optional on CPU-only hosts
        pass
    return info


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """Hash of (relative name, file hash) pairs, sorted: stable across hosts."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def record(path: Path | None = None, **fields: Any) -> dict[str, Any]:
    """Append one record. Unknown fields are kept; known ones default to null."""
    path = Path(path or os.environ.get("SATQUERY_REGISTRY", DEFAULT_PATH))
    rec: dict[str, Any] = {k: None for k in FIELDS}
    rec.update(fields)
    # Smoke tests write their checkpoints under /tmp; they are not experiments
    # and must not appear in the lineage.
    ckpt = str(rec.get("checkpoint") or "").replace("\\", "/")
    if ckpt.startswith("/tmp/") or "/AppData/Local/Temp/" in ckpt:
        return rec
    path.parent.mkdir(parents=True, exist_ok=True)
    rec.setdefault("experiment_id", new_id("exp"))
    rec["timestamp"] = rec.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if rec.get("git_commit") is None:
        rec["git_commit"] = git_commit()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return rec


def load(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """One record per experiment_id: the records for an id are merged in
    file order, a later non-null field overriding an earlier one. A trainer
    writes `running` (hyperparameters, dataset versions, manifest hash) and
    later `done` (metrics, checkpoint sha256) - both halves must survive."""
    path = Path(path or os.environ.get("SATQUERY_REGISTRY", DEFAULT_PATH))
    merged: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return merged
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cur = merged.setdefault(rec["experiment_id"], {})
                for k, v in rec.items():
                    if v is not None or k not in cur:
                        cur[k] = v
    return merged

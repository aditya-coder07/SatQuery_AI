"""Reusable checkpoint / resume infrastructure.

Generalised from the Phase 0 step 6 proof (`training/checkpoint_resume_test.py`),
which demonstrated that a run can be killed and resumed without losing state.
Every training script in this repo must use this module rather than rolling its
own, because the free-tier failure mode the plan warns about - Kaggle killing a
10-hour run at hour 11 - is only survivable if resume actually works.

What is restored, and why each matters:

* model + optimizer + LR scheduler state - the obvious part
* **RNG state** (python, numpy, torch, cuda) - without this, resuming replays a
  different data order and different dropout masks, so the resumed run is not
  a continuation of the original one
* the step counter and the dataloader position

Checkpoints are written atomically (temp file + replace) so a process killed
*during* a save cannot leave a truncated file that breaks the next resume.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

CKPT_PATTERN = "ckpt_step_*.pt"
_STEP_RE = re.compile(r"ckpt_step_(\d+)\.pt$")


def set_seed(seed: int) -> None:
    """Seed every generator a training run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch requires a ByteTensor on CPU; a checkpoint moved between machines
    # can arrive as something else.
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(s, dtype=torch.uint8) for s in state["cuda"]]
            )
        except (RuntimeError, ValueError):
            # Resuming on a machine with a different GPU count is legitimate;
            # losing CUDA RNG continuity is a warning, not a failure.
            print(
                "WARNING: could not restore CUDA RNG state (different GPU "
                "count?); continuing with a freshly seeded CUDA generator"
            )


def find_latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Highest-step checkpoint in `ckpt_dir`, or None."""
    matches = glob.glob(str(Path(ckpt_dir) / CKPT_PATTERN))
    if not matches:
        return None

    def step_of(path: str) -> int:
        m = _STEP_RE.search(path.replace("\\", "/"))
        return int(m.group(1)) if m else -1

    return Path(max(matches, key=step_of))


@dataclass
class TrainingState:
    """Everything needed to continue a run exactly where it stopped."""

    step: int = 0
    epoch: int = 0
    best_metric: float | None = None
    metrics_history: list[dict] = field(default_factory=list)


def save_checkpoint(
    ckpt_dir: str | Path,
    step: int,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    state: TrainingState | None = None,
    extra: dict | None = None,
    keep_last: int = 3,
    is_peft: bool = False,
) -> Path:
    """Write a checkpoint atomically and prune old ones.

    `is_peft` saves only the adapter weights rather than the full base model,
    which is the whole point of QLoRA: a 3B base plus a ~40 MB adapter, not a
    6 GB checkpoint every save interval.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / f"ckpt_step_{step}.pt"
    tmp = ckpt_dir / f".ckpt_step_{step}.pt.tmp"

    if is_peft:
        # Adapter weights are saved by peft into their own directory; the .pt
        # holds only the training state that peft does not manage.
        adapter_dir = ckpt_dir / f"adapter_step_{step}"
        model.save_pretrained(str(adapter_dir))
        model_state = {"adapter_dir": str(adapter_dir)}
    else:
        model_state = {"model_state_dict": model.state_dict()}

    payload: dict[str, Any] = {
        "step": step,
        "is_peft": is_peft,
        **model_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "training_state": vars(state) if state else vars(TrainingState(step=step)),
        "rng_state": capture_rng_state(),
        "extra": extra or {},
    }

    # Atomic: a kill during torch.save leaves only the .tmp behind, and the
    # previous good checkpoint is still the latest.
    torch.save(payload, tmp)
    os.replace(tmp, target)

    _prune_old_checkpoints(ckpt_dir, keep_last)
    return target


def _prune_old_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    """Keep only the newest `keep_last` checkpoints; free tiers have small disks."""
    if keep_last <= 0:
        return
    matches = sorted(
        glob.glob(str(ckpt_dir / CKPT_PATTERN)),
        key=lambda p: int(_STEP_RE.search(p.replace("\\", "/")).group(1)),  # type: ignore[union-attr]
    )
    for stale in matches[:-keep_last]:
        stale_path = Path(stale)
        step = int(_STEP_RE.search(stale.replace("\\", "/")).group(1))  # type: ignore[union-attr]
        stale_path.unlink(missing_ok=True)
        adapter = ckpt_dir / f"adapter_step_{step}"
        if adapter.is_dir():
            shutil.rmtree(adapter, ignore_errors=True)


def load_checkpoint(
    path: str | Path,
    model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    map_location: str = "cpu",
) -> tuple[TrainingState, dict]:
    """Restore a checkpoint, returning (training state, extra).

    NOTE ON `weights_only=False`: the checkpoint deliberately stores non-tensor
    objects (RNG states, python tuples), which the safe loader rejects. This is
    only ever used on checkpoints this project wrote to a local directory.
    Never point it at a checkpoint downloaded from an untrusted source -
    torch.load with weights_only=False executes pickle and is a code-execution
    risk. Third-party weights must go through scripts/fetch_models.py, which
    verifies a checksum and loads via safetensors.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        if payload.get("is_peft"):
            adapter_dir = payload.get("adapter_dir")
            if adapter_dir and Path(adapter_dir).is_dir():
                model.load_adapter(adapter_dir, adapter_name="default")
            else:
                raise FileNotFoundError(
                    f"checkpoint {path} references adapter dir {adapter_dir!r}, "
                    "which is missing - the checkpoint directory was moved or "
                    "partially deleted"
                )
        else:
            model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    if payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])

    ts_dict = payload.get("training_state") or {}
    state = TrainingState(
        step=ts_dict.get("step", payload.get("step", 0)),
        epoch=ts_dict.get("epoch", 0),
        best_metric=ts_dict.get("best_metric"),
        metrics_history=ts_dict.get("metrics_history", []),
    )
    return state, payload.get("extra", {})


def maybe_resume(
    ckpt_dir: str | Path,
    model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    enabled: bool = True,
) -> tuple[TrainingState, dict]:
    """Resume from the newest checkpoint if one exists, else start fresh."""
    if not enabled:
        return TrainingState(), {}
    latest = find_latest_checkpoint(ckpt_dir)
    if latest is None:
        print(f"No checkpoint in {ckpt_dir}; starting from step 0.")
        return TrainingState(), {}
    print(f"Resuming from {latest}")
    state, extra = load_checkpoint(latest, model, optimizer, scheduler)
    print(f"RESUMED AT STEP {state.step}", flush=True)
    return state, extra


def write_run_metadata(ckpt_dir: str | Path, metadata: dict) -> Path:
    """Record run provenance next to the checkpoints.

    Task 1.7/1.10 need the weights hash and config in the trace; writing it at
    train time means the served model can always be tied back to its run.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path

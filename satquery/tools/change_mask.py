"""`change_mask_v1` backed by the trained detector (plan task 2.4).

Produces a georeferenced binary change raster from a bitemporal pair, plus the
changed area in km2 so the answer carries a number rather than only a picture.

Like `rs_vqa_v1`, this activates only when a checkpoint is configured
(`SATQUERY_CHANGE_MASK`); otherwise the stub stays so CI and GPU-less
machines keep a green suite instead of half-loading a model.

The mask is written in the *reference image's* CRS and transform, so it opens
aligned in QGIS. Emitting an ungeoreferenced PNG would make the headline
deliverable of a change-detection system unusable in a GIS.

Confidence is `mean(|p - 0.5|) * 2` over the per-pixel change probabilities:
0 when every pixel sits at the undecided midpoint, 1 when every pixel is
saturated. That is **sharpness** - how decisive the detector was - and it is
reported under `confidence_method="sharpness"` for exactly that reason. It is
NOT a probability of correctness: a detector can be uniformly saturated and
uniformly wrong, and this number would read 1.0. It therefore feeds the
three-part confidence combiner as one weak signal and is excluded from
calibration, which is only defined on a probability. Until Phase 3 this was
labelled `softmax_temp_scaled`, which was wrong twice over - nothing was
temperature-scaled, and the value was never a softmax probability.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import Artifact, ToolPayload, ToolResult
from satquery.tools.base import ToolProtocol
from satquery.tools.provenance import record

TOOL_NAME = "change_mask"
TOOL_VERSION = "1.1.1"  # 1.1.0: dihedral TTA (2026-09-21); 1.1.1: m2 below 1 km2
ENV_CHECKPOINT = "SATQUERY_CHANGE_MASK"

# The detector was trained on 256px RGB tiles; inference tiles at the same
# size so the model sees the scale it learned.
TILE = 256
DEFAULT_THRESHOLD = 0.5
# 8-fold dihedral test-time augmentation (2026-09-21). Measured on the
# official LEVIR-CD test with the deployed champion and loader: F1 0.9038 ->
# 0.9101 at threshold 0.5 (0.9139 / IoU 0.8414 at the val-selected 0.75);
# two retrains of the champion did not beat it, so the gain is inference-
# side. Eight passes of a 24M-parameter net: well under a second on a GPU,
# a few seconds on a CPU host. SATQUERY_CHANGE_MASK_TTA=0 disables it.
ENV_TTA = "SATQUERY_CHANGE_MASK_TTA"


def _area_text(km2: float) -> str:
    # 0.007 km2 printed as "0.01 km2" overstated a 7,000 m2 change by 40%.
    if km2 >= 1.0:
        return f"{km2:.2f} km2"
    return f"about {round(km2 * 1_000_000, -2):,.0f} m2"


def tta_enabled() -> bool:
    return os.getenv(ENV_TTA, "1").strip().lower() not in ("0", "false", "no", "off")


def dihedral_probability(torch, model, a, b):
    """Mean sigmoid over the 8 dihedral transforms, both dates transformed
    identically and the output undone - the evaluator's `dihedral_tta`."""
    acc = None
    for k in range(4):
        for flip in (False, True):
            ta, tb = torch.rot90(a, k, (2, 3)), torch.rot90(b, k, (2, 3))
            if flip:
                ta, tb = torch.flip(ta, (3,)), torch.flip(tb, (3,))
            pr = torch.sigmoid(model(ta, tb))
            if flip:
                pr = torch.flip(pr, (3,))
            pr = torch.rot90(pr, -k, (2, 3))
            acc = pr if acc is None else acc + pr
    return acc / 8


class ChangeMaskPayload(ToolPayload):
    data: dict[str, Any]


def is_available() -> tuple[bool, str]:
    path = os.getenv(ENV_CHECKPOINT)
    if not path:
        return False, f"{ENV_CHECKPOINT} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


class _Handle:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, checkpoint: Path):
        import torch

        from training.common.checkpointing import (
            find_latest_checkpoint,
            load_checkpoint,
            safe_torch_load,
        )
        from training.train_change_mask import build_model

        latest = find_latest_checkpoint(checkpoint) or checkpoint
        payload = safe_torch_load(latest)
        extra = payload.get("extra") or {}
        dim = extra.get("dim", 16)
        # Which architecture wrote these weights. A checkpoint from before
        # Phase 5 has no `arch` field, so the default is v1 and every
        # existing checkpoint rebuilds exactly as it always did. Guessing
        # instead would load v1 weights into a v2 graph and fail on a key
        # mismatch that says nothing about the cause.
        model = build_model(dim, arch=extra.get("arch", "v1"))
        load_checkpoint(latest, model, map_location="cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.path = str(latest)
        # The bytes that are now in memory, hashed once per process, so
        # `Trace.weights_hashes` names the weights that produced the answer
        # rather than being empty. See satquery/tools/provenance.py.
        record("change_mask_v1", latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


def _read_rgb(meta, size: int) -> np.ndarray:
    """Read a 3-band RGB view, scaled to 0-1 by percentile stretch."""
    order = [b for b in ("RED", "GREEN", "BLUE") if b in meta.bands]
    with rasterio.open(meta.path) as src:
        indices = (
            [meta.bands.index(b) + 1 for b in order] if len(order) == 3
            else list(range(1, min(3, src.count) + 1))
        )
        while len(indices) < 3:
            indices.append(indices[-1])
        channels = []
        for idx in indices:
            arr = src.read(
                idx, out_shape=(size, size),
                resampling=Resampling.bilinear, masked=True,
            )
            arr = np.ma.filled(arr.astype("float32"), np.nan)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                lo, hi = np.percentile(finite, [2, 98])
                arr = np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else arr * 0
            channels.append(np.nan_to_num(arr))
    # float32 explicitly: np.percentile promotes to float64, and torch
    # conv2d rejects a DoubleTensor against float32 weights.
    return np.stack(channels).astype("float32")


class ChangeMaskTool(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        warnings: list[str] = []

        if len(manifest.images) != 2:
            raise ValueError("change detection requires two images")

        t1, t2 = manifest.images
        handle = _Handle.get(Path(os.environ[ENV_CHECKPOINT]))
        torch = handle.torch
        # The matrix names the parameter `change_threshold`; `threshold` is
        # kept for callers that passed it directly. Before 2026-09-21 only
        # the latter was read, so the matrix default never reached here.
        threshold = float(params.get("change_threshold", params.get("threshold", DEFAULT_THRESHOLD)))
        tta = tta_enabled()

        a = _read_rgb(t1, TILE)[None]
        b = _read_rgb(t2, TILE)[None]
        with torch.no_grad():
            ta = torch.from_numpy(a).to(handle.device)
            tb = torch.from_numpy(b).to(handle.device)
            if tta:
                probability = dihedral_probability(torch, handle.model, ta, tb)[0, 0].cpu().numpy()
            else:
                probability = torch.sigmoid(handle.model(ta, tb))[0, 0].cpu().numpy()

        mask = (probability >= threshold).astype("uint8")
        changed_fraction = float(mask.mean())
        area_km2 = changed_fraction * t1.width * t1.height * (t1.gsd_m ** 2) / 1e6

        artifacts: list[Artifact] = []
        out_dir = Path(params.get("output_dir", "artifacts")) / manifest.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "change_mask.tif"

        # Written with the reference image's CRS and transform, scaled for the
        # tile size, so the mask opens aligned in QGIS.
        with rasterio.open(t1.path) as src:
            transform = src.transform * src.transform.scale(
                src.width / TILE, src.height / TILE
            )
            profile = {
                "driver": "COG", "dtype": "uint8", "count": 1,
                "height": TILE, "width": TILE, "crs": src.crs,
                "transform": transform, "nodata": 255,
            }
        with rasterio.open(target, "w", **profile) as dst:
            dst.write(mask, 1)
        artifacts.append(Artifact(
            key="change_mask", kind="cog", path=target, crs=t1.crs,
            description="Binary change mask (1 = changed)",
        ))

        if changed_fraction > 0.5:
            warnings.append(
                f"{changed_fraction:.0%} of the scene flagged as changed; "
                "check co-registration and radiometric consistency"
            )

        payload = ChangeMaskPayload(data={
            "changed_fraction": round(changed_fraction, 6),
            "changed_area_km2": round(area_km2, 4),
            "threshold": threshold,
            "tta": "dihedral8" if tta else "none",
            "mean_change_probability": round(float(probability.mean()), 6),
            "answer": (
                f"About {changed_fraction:.1%} of the scene changed "
                f"({_area_text(area_km2)}), detected at threshold {threshold}."
            ),
        })

        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION, payload=payload,
            artifacts=artifacts,
            confidence=float(np.abs(probability - 0.5).mean() * 2),
            confidence_method="sharpness",
            model_card=f"tiny siamese change detector ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]

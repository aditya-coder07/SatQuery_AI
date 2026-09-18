"""`optsar_fusion_v1` in triad mode (plan task 2.3).

The PS requires extracting *complementary* information from a co-registered
optical + SAR pair. Per docs/01 section 5.2, a fused number alone does not
demonstrate complementarity, so every query runs three passes - optical-only,
SAR-only, fused - and reports a per-query complementarity score:

    gain      = fused - max(optical, sar)
    agreement = how often the two modalities pick the same classes
    attribution = which modality drove each class

MEASURED CAVEAT, reported rather than buried: on WHU-OPT-SAR scene-level
multi-label classification the gain is about -0.006 - fusion does not help.
Both modalities can independently answer "is there water somewhere in this
tile", leaving nothing for fusion to add. Complementarity is inherently
spatial (SAR seeing through cloud, separating water from shadow), so
demonstrating it needs a per-pixel segmentation head, which is task 2.9/3.x
work. The triad machinery here is correct and reports the honest number;
the number is currently ~zero.

Confidence is the mean fused probability over the classes the tool actually
asserted - those at or above `PRESENCE_THRESHOLD` - reported as
`confidence_method="mean_asserted_probability"`. Unlike the change mask's
sharpness this genuinely is a probability: it is the head's own estimate of
its precision over the positive predictions it made. It is still not a
calibratable P(correct) for this answer, for two reasons worth keeping
straight:

* it is conditioned on `p >= PRESENCE_THRESHOLD`, so it describes only the
  asserted subset and says nothing about classes the tool stayed silent on;
* it is an aggregate. A fitted calibration is a *nonlinear* map, so applying
  it to a mean of probabilities is not the same as calibrating each class and
  then averaging. Calibrating this head means transforming `p_fused`
  per class before this line, not transforming the number it produces.

Until Phase 3 this was labelled `softmax_temp_scaled`; nothing was ever
temperature-scaled.
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
from satquery.contracts.tool_result import ToolPayload, ToolResult
from satquery.tools.base import ToolProtocol
from satquery.tools.provenance import record

TOOL_NAME = "optsar_fusion"
TOOL_VERSION = "1.0.0-triad"
ENV_CHECKPOINT = "SATQUERY_FUSION"
PATCH = 120
PRESENCE_THRESHOLD = 0.5

OPTICAL_MODALITIES = {"OPTICAL", "MSI", "PAN"}


class FusionPayload(ToolPayload):
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

        from training.common.checkpointing import find_latest_checkpoint, load_checkpoint
        from training.train_optsar_fusion import build_model

        latest = find_latest_checkpoint(checkpoint) or checkpoint
        from training.common.checkpointing import safe_torch_load

        extra = (safe_torch_load(latest).get("extra") or {})
        # Which architecture wrote these weights. A checkpoint from before
        # Phase 5 has no `arch` field, so the default is v1 and every
        # existing checkpoint rebuilds exactly as it always did. Guessing
        # instead would load v1 weights into a v2 graph and fail on a key
        # mismatch that says nothing about the cause.
        self.arch = extra.get("arch", "v1")
        model = build_model(extra.get("dim", 32), arch=self.arch)
        load_checkpoint(latest, model, map_location="cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.path = str(latest)
        # The bytes that are now in memory, hashed once per process, so
        # `Trace.weights_hashes` names the weights that produced the answer
        # rather than being empty. See satquery/tools/provenance.py.
        record("optsar_fusion_v1", latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


def _read(meta, n_bands: int) -> np.ndarray:
    """Read and standardise `n_bands` channels at the trained patch size."""
    with rasterio.open(meta.path) as src:
        count = min(src.count, n_bands)
        arr = src.read(
            list(range(1, count + 1)), out_shape=(count, PATCH, PATCH),
            resampling=Resampling.bilinear, masked=True,
        ).astype("float32")
    arr = np.ma.filled(arr, np.nan)
    out = []
    for band in arr:
        finite = band[np.isfinite(band)]
        if finite.size:
            mean, std = float(finite.mean()), float(finite.std()) or 1.0
            band = (band - mean) / std
        out.append(np.nan_to_num(band))
    stacked = np.stack(out)
    while stacked.shape[0] < n_bands:
        stacked = np.concatenate([stacked, stacked[-1:]])
    return stacked.astype("float32")


def _read_v3(meta, n_bands: int, size: int) -> np.ndarray:
    """Bands scaled to [0, 1] by their 2nd-98th percentile, then the fixed
    normalisation the v3 trainer used on 8-bit tiles ((x - 0.5) / 0.25). A
    robust scale rather than /255 so 16-bit Sentinel inputs land in the same
    range the model saw."""
    with rasterio.open(meta.path) as src:
        count = min(src.count, n_bands)
        arr = src.read(
            list(range(1, count + 1)), out_shape=(count, size, size),
            resampling=Resampling.bilinear, masked=True,
        ).astype("float32")
    arr = np.ma.filled(arr, np.nan)
    out = []
    for band in arr:
        finite = band[np.isfinite(band)]
        lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
        band = (band - lo) / max(hi - lo, 1e-6)
        out.append((np.clip(np.nan_to_num(band), 0, 1) - 0.5) / 0.25)
    stacked = np.stack(out)
    while stacked.shape[0] < n_bands:
        stacked = np.concatenate([stacked, stacked[-1:]])
    return stacked.astype("float32")


class OptSARFusionTool(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()

        optical = next(
            (i for i in manifest.images if i.modality in OPTICAL_MODALITIES), None
        )
        sar = next((i for i in manifest.images if i.modality == "SAR"), None)
        if optical is None or sar is None:
            raise ValueError(
                "cross-modal fusion needs one optical and one SAR image; got "
                f"{[i.modality for i in manifest.images]}"
            )

        handle = _Handle.get(Path(os.environ[ENV_CHECKPOINT]))
        torch = handle.torch
        if handle.arch == "v3":
            return self._run_v3(handle, optical, sar, started)

        o = _read(optical, 4)[None]
        s = _read(sar, 1)[None]
        with torch.no_grad():
            lo, ls, lf = handle.model(
                torch.from_numpy(o).to(handle.device),
                torch.from_numpy(s).to(handle.device),
            )
            p_opt = torch.sigmoid(lo)[0].cpu().numpy()
            p_sar = torch.sigmoid(ls)[0].cpu().numpy()
            p_fused = torch.sigmoid(lf)[0].cpu().numpy()

        from training.prepare.whu_opt_sar import CLASSES

        def present(p):
            return {CLASSES[i] for i, v in enumerate(p) if v >= PRESENCE_THRESHOLD}

        set_o, set_s, set_f = present(p_opt), present(p_sar), present(p_fused)
        union = set_o | set_s
        # Low agreement means the modalities disagree, which is exactly when
        # fusion had something to resolve.
        agreement = len(set_o & set_s) / len(union) if union else 1.0

        attribution = {}
        for i, name in enumerate(CLASSES):
            if p_fused[i] < PRESENCE_THRESHOLD:
                continue
            attribution[name] = (
                "optical" if p_opt[i] > p_sar[i] else
                "sar" if p_sar[i] > p_opt[i] else "equal"
            )

        # Per-query proxy for the offline gain: how much more confident the
        # fused head is than the better single modality, averaged over the
        # classes it actually asserts.
        asserted = [i for i in range(len(CLASSES)) if p_fused[i] >= PRESENCE_THRESHOLD]
        gain = (
            float(np.mean([p_fused[i] - max(p_opt[i], p_sar[i]) for i in asserted]))
            if asserted else 0.0
        )

        classes_text = ", ".join(sorted(set_f)) or "no class above threshold"
        answer = (
            f"Fused optical+SAR analysis identifies: {classes_text}. "
            f"The two modalities agreed on {agreement:.0%} of detected classes."
        )

        payload = FusionPayload(data={
            "answer": answer,
            "mode": "triad",
            "optical_only": {CLASSES[i]: round(float(p_opt[i]), 4) for i in range(len(CLASSES))},
            "sar_only": {CLASSES[i]: round(float(p_sar[i]), 4) for i in range(len(CLASSES))},
            "fused": {CLASSES[i]: round(float(p_fused[i]), 4) for i in range(len(CLASSES))},
            "complementarity": {
                "gain": round(gain, 6),
                "modality_agreement": round(agreement, 4),
                "attribution": attribution,
                "note": (
                    "Offline gain on WHU-OPT-SAR scene-level classification is "
                    "about -0.006: fusion does not currently beat optical alone. "
                    "Scene-level presence is too coarse a task for "
                    "complementarity to appear; a per-pixel head is required."
                ),
            },
        })

        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION, payload=payload, artifacts=[],
            confidence=float(np.mean([p_fused[i] for i in asserted])) if asserted else 0.0,
            confidence_method=(
                "mean_asserted_probability" if asserted else "no_assertion"
            ),
            model_card=f"optical-SAR triad ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=(
                [] if asserted else ["no class exceeded the presence threshold"]
            ),
        )

    def _run_v3(self, handle, optical, sar, started: float) -> ToolResult:
        """Per-pixel triad (Phase 6): class maps from optical, SAR and the
        gated fusion, with the complementarity read off the pixels.

        The v3 model is the one whose offline gain is positive on aligned,
        scene-disjoint WHU-OPT-SAR (+0.021 mIoU, per-tile CI excluding zero;
        docs/assets/phase6/optsar_fusion). Its output is a class map, so the
        payload reports class *fractions* per arm rather than presence
        probabilities, plus the fraction of pixels where fusion changed the
        optical decision and which modality it sided with.
        """
        from training.prepare.whu_opt_sar import CLASSES  # background + 7

        names = CLASSES[1:]  # the v3 head predicts the 7 labelled classes
        torch = handle.torch
        size = 256
        o = _read_v3(optical, 4, size)[None]
        s = _read_v3(sar, 1, size)[None]
        with torch.no_grad():
            ob, sb = torch.from_numpy(o).to(handle.device), torch.from_numpy(s).to(handle.device)
            lo, ls, lf = handle.model(ob, sb)
            _, _, lf_no_sar = handle.model(ob, sb, drop="sar")
            probs = {k: torch.softmax(v[0].float(), dim=0).cpu().numpy()
                     for k, v in (("optical", lo), ("sar", ls), ("fused", lf), ("fused_no_sar", lf_no_sar))}
        maps = {k: v.argmax(0) for k, v in probs.items()}

        def fractions(m):
            return {names[c]: round(float((m == c).mean()), 4) for c in range(len(names))}

        changed = maps["fused"] != maps["optical"]
        sided_sar = changed & (maps["fused"] == maps["sar"])
        agreement = float((maps["optical"] == maps["sar"]).mean())
        # Where fusion overrode optical, did SAR get the vote?
        attribution = {}
        for c, name in enumerate(names):
            sel = changed & (maps["fused"] == c)
            if sel.any():
                attribution[name] = "sar" if (maps["sar"][sel] == c).mean() > 0.5 else "fused_feature"
        fused_conf = float(probs["fused"].max(0).mean())
        dominant = sorted(fractions(maps["fused"]).items(), key=lambda kv: -kv[1])[:3]
        answer = (
            "Fused optical+SAR land-cover map: " + ", ".join(f"{k} {v:.0%}" for k, v in dominant) +
            f". Optical and SAR agree on {agreement:.0%} of pixels; fusion revised "
            f"{float(changed.mean()):.1%} of the optical decisions, siding with SAR on "
            f"{float(sided_sar.sum()) / max(1, int(changed.sum())):.0%} of those."
        )
        payload = FusionPayload(data={
            "answer": answer,
            "mode": "triad_segmentation_v3",
            "optical_only": fractions(maps["optical"]),
            "sar_only": fractions(maps["sar"]),
            "fused": fractions(maps["fused"]),
            "fused_without_sar": fractions(maps["fused_no_sar"]),
            "complementarity": {
                "pixels_revised_by_fusion": round(float(changed.mean()), 4),
                "revisions_siding_with_sar": round(float(sided_sar.sum()) / max(1, int(changed.sum())), 4),
                "modality_agreement": round(agreement, 4),
                "attribution": attribution,
                "note": ("Offline, scene-disjoint WHU-OPT-SAR with aligned labels: optical 0.447, "
                         "SAR 0.388, fused 0.468 mIoU; per-tile gain +0.009, 95% CI [+0.006, +0.012]. "
                         "See docs/assets/phase6/optsar_fusion."),
            },
        })
        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION, payload=payload, artifacts=[],
            confidence=fused_conf, confidence_method="mean_asserted_probability",
            model_card=f"optical-SAR segmentation triad v3 ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000), warnings=[],
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]

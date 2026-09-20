"""`grounding_v1` backed by the trained referring grounder (plan task 2.7).

Task 2.7 trained the grounder and reported mIoU 0.1405 / Acc@0.5 0.0762 on
DIOR-RSVG, but no tool wired it into the pipeline - the registry kept the
stub, so the model was unreachable from a query. This is that wiring.

Opt-in via `SATQUERY_GROUNDING`, the same pattern the other learned tools use.

## Read the metric before trusting a box

Published DIOR-RSVG results reach roughly 70-80% Acc@0.5. **This model reaches
7.6%**, and the cause is architectural rather than mysterious: it
global-average-pools the visual feature map before regressing the box, which
discards exactly the spatial information localisation depends on, so it can
only learn an "average" box. Task 2.7 recorded that, and wiring the model in
does not change it.

The tool therefore reports the box **with the model's own weak confidence**,
and the three-component combiner and the abstention policy are what stop a
near-random box being presented as a finding. It is wired because "the
pipeline cannot reach a model we trained" is a worse state than "the pipeline
reaches a model whose limitations are measured and recorded" - not because
the box is good.

Boxes are emitted in pixel coordinates. `satquery/report/evidence_pack.py`
projects them to GeoJSON using the image transform, which is where task 2.7's
"boxes exported as GeoJSON" is satisfied.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import ToolPayload, ToolResult
from satquery.tools.base import ToolProtocol
from satquery.tools.provenance import record
from satquery.tools.sidecars import readable_json, readable_safetensors
from satquery.tools.imaging import selected_image, to_rgb_preview

TOOL_NAME = "grounding"
TOOL_VERSION = "1.1.0"
ENV_CHECKPOINT = "SATQUERY_GROUNDING"
# Phase 6: a LoRA adapter for the shared Qwen2.5-VL base (the VQA tool's
# SATQUERY_VQA_BASE). When set, it takes precedence over the CNN checkpoint.
# Trained by training/train_grounding_vlm.py on the official DIOR-RSVG train
# split; scored by evaluation/grounding_official_eval.py.
ENV_VLM_ADAPTER = "SATQUERY_GROUNDING_ADAPTER"
VLM_ADAPTER_NAME = "grounding"
# Pixel budget the adapter was trained and benchmarked with. Arm E (Phase 6)
# upscales inputs to 1024x1024 (`min_pixels` 1048576); serving it at the
# processor default would put the model 3 points below its measured
# number. Applied once, to the shared processor, before the first call.
ENV_VLM_MIN_PIXELS = "SATQUERY_GROUNDING_MIN_PIXELS"
ENV_VLM_MAX_PIXELS = "SATQUERY_GROUNDING_MAX_PIXELS"


def pixel_budget() -> tuple[int | None, int | None]:
    def _int(name: str) -> int | None:
        v = os.getenv(name)
        return int(v) if v and v.strip() else None
    return _int(ENV_VLM_MIN_PIXELS), _int(ENV_VLM_MAX_PIXELS)


class GroundingPayload(ToolPayload):
    data: dict[str, Any]


def vlm_configured() -> bool:
    from satquery.tools import rs_vqa

    return bool(os.getenv(ENV_VLM_ADAPTER)) and bool(os.getenv(rs_vqa.ENV_BASE))


def is_available() -> tuple[bool, str]:
    if vlm_configured():
        from satquery.tools import rs_vqa

        adapter = Path(os.environ[ENV_VLM_ADAPTER])
        if not adapter.exists():
            return False, f"grounding adapter not found: {adapter}"
        ok, reason = readable_safetensors(adapter)
        if not ok:
            return False, reason
        if not Path(os.environ[rs_vqa.ENV_BASE]).exists():
            return False, f"base model not found: {os.environ[rs_vqa.ENV_BASE]}"
        for module in ("torch", "peft", "transformers"):
            try:
                __import__(module)
            except ImportError:
                return False, f"{module} is not installed"
        return True, "ready (VLM adapter)"
    path = os.getenv(ENV_CHECKPOINT)
    if not path:
        return False, f"{ENV_CHECKPOINT} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    # Checkpoint contents before environment - see caption.py.
    # Readable, not merely present - see caption.py. Restoring this
    # checkpoint from a shadow copy returned vocab.json as 1,106 bytes of NUL,
    # and this check answered "ready" until it was taught to parse the file.
    ok, reason = readable_json(Path(path) / "vocab.json", expect=dict)
    if not ok:
        return False, reason
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


def _phrase_from(params: dict) -> tuple[str, list[str]]:
    """The referring expression the grounder is asked to locate.

    `_phrase` is the object the router extracted from the query, with its
    spatial qualifier, phrased the way the adapter's training expressions
    read ("the airplane on the right"). Before 2026-09-20 the whole sentence
    was used, so "Find the airport in this image." became the prompt
    "Locate the Find the airport in this image in the image ..." - out of
    the adapter's distribution and, for anything but a bare noun phrase,
    a worse box than the benchmark number promises. The raw query remains
    the fallback when nothing could be extracted.
    """
    warnings: list[str] = []
    phrase = str(params.get("_phrase") or "").strip()
    if not phrase:
        phrase = str(params.get("_query") or "").strip()
        if phrase:
            warnings.append("no referring expression extracted from the query; used the sentence as typed")
    if not phrase:
        phrase = "the main object"
        warnings.append("no referring expression supplied; used a generic one")
    return phrase, warnings


class _Handle:
    _instance: "_Handle | None" = None
    _lock = threading.Lock()

    def __init__(self, checkpoint: Path):
        import torch

        from training.common.checkpointing import (
            find_latest_checkpoint,
            load_checkpoint,
            safe_torch_load,
        )
        from training.train_grounding import build_model

        self.vocab: dict[str, int] = json.loads(
            (checkpoint / "vocab.json").read_text(encoding="utf-8")
        )
        latest = find_latest_checkpoint(checkpoint) or checkpoint
        payload = safe_torch_load(latest)
        extra = payload.get("extra") or {}
        dim = extra.get("dim", 128)
        # Which architecture wrote these weights. A checkpoint from before
        # Phase 5 has no `arch` field, so the default is v1 and every
        # existing checkpoint rebuilds exactly as it always did. Guessing
        # instead would load v1 weights into a v2 graph and fail on a key
        # mismatch that says nothing about the cause.
        # `pretrained` selects a different backbone, so it must be known
        # before the graph is built - exactly like `arch`. Two checkpoints
        # were written before it was recorded; for those it is inferred from
        # the weights: only the pretrained variant has a `proj.` projection
        # after its 2048-wide ResNet-50 (the from-scratch trunk is Identity
        # there). The same pattern `landcover` uses to infer `has_gsd`.
        state = payload.get("model_state_dict", {})
        pretrained = extra.get("pretrained")
        if pretrained is None:
            pretrained = any(k.startswith("proj.") for k in state)
        model = build_model(vocab_size=len(self.vocab), dim=dim,
                            arch=extra.get("arch", "v1"), pretrained=pretrained)
        load_checkpoint(latest, model, map_location="cpu")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.path = str(latest)
        # The bytes that are now in memory, hashed once per process, so
        # `Trace.weights_hashes` names the weights that produced the answer
        # rather than being empty. See satquery/tools/provenance.py.
        record("grounding_v1", latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


class GroundingTool(ToolProtocol):
    name = TOOL_NAME
    version = TOOL_VERSION

    def run(self, manifest: InputManifest, params: dict) -> ToolResult:
        if vlm_configured():
            return self._run_vlm(manifest, params)
        started = time.perf_counter()
        handle = _Handle.get(Path(os.environ[ENV_CHECKPOINT]))
        torch = handle.torch

        from training.train_grounding import IMAGE_SIZE, encode_text

        phrase, warnings = _phrase_from(params)

        meta = selected_image(manifest, params)
        image, _ = to_rgb_preview(meta, max_edge=IMAGE_SIZE)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0

        tokens = encode_text(phrase, handle.vocab)
        with torch.no_grad():
            box = handle.model(
                torch.from_numpy(array).unsqueeze(0).to(handle.device),
                torch.from_numpy(np.asarray(tokens)).unsqueeze(0).to(handle.device),
            )[0].tolist()

        # The head emits normalised (cx, cy, w, h) through a sigmoid, so the
        # box is inside the frame by construction. Converted to pixel corners
        # against the ACTUAL image size, not the 224px model input, or every
        # exported box would be wrong on any scene that is not square.
        cx, cy, w, h = box
        width, height = meta.width, meta.height
        x0 = max(0.0, (cx - w / 2) * width)
        y0 = max(0.0, (cy - h / 2) * height)
        x1 = min(float(width), (cx + w / 2) * width)
        y1 = min(float(height), (cy + h / 2) * height)

        # There is no objectness head, so there is no learned score to report.
        # Fabricating one would be worse than reporting the measured ceiling:
        # this is the model's Acc@0.5 on its own test split, which is what a
        # box from it is actually worth.
        confidence = 0.0762

        return ToolResult(
            tool=TOOL_NAME,
            version=TOOL_VERSION,
            payload=GroundingPayload(
                data={
                    "phrase": phrase,
                    "bounding_boxes": [
                        {
                            "x0": round(x0, 2), "y0": round(y0, 2),
                            "x1": round(x1, 2), "y1": round(y1, 2),
                            "label": phrase,
                        }
                    ],
                    "normalised_cxcywh": [round(v, 6) for v in box],
                    "image_size": [width, height],
                }
            ),
            artifacts=[],
            confidence=confidence,
            # Not a per-prediction score: a fixed dataset-level accuracy. Named
            # so nobody mistakes it for the model's own certainty.
            confidence_method="threshold_rule",
            model_card=f"referring grounder ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings + [
                "grounding Acc@0.5 is 0.0762 on DIOR-RSVG against ~70-80% "
                "published; the head pools away spatial detail before "
                "regressing the box, so this localisation is weak by design"
            ],
        )

    def _run_vlm(self, manifest: InputManifest, params: dict) -> ToolResult:
        """Grounding through the shared Qwen2.5-VL base + the grounding adapter.

        The box is parsed from the model's JSON reply in the frame the model
        saw (the processor's 28-multiple resize) and mapped back to the
        source image's pixel frame - the same two-step the evaluator uses, so
        the deployed box and the benchmarked box are the same box.
        """
        from satquery.tools import rs_vqa
        from training.common.vlm_grounding import (
            build_chat, parse_box, resized_hw, scale_box,
        )

        started = time.perf_counter()
        phrase, warnings = _phrase_from(params)

        base = Path(os.environ[rs_vqa.ENV_BASE])
        adapter = Path(os.environ[ENV_VLM_ADAPTER])
        handle = rs_vqa._ModelHandle.get(base, Path(os.environ[rs_vqa.ENV_ADAPTER]))
        handle.ensure_adapter(VLM_ADAPTER_NAME, adapter)
        torch = handle.torch
        min_px, max_px = pixel_budget()
        if (min_px or max_px) and getattr(handle, "_grounding_pixel_budget", None) != (min_px, max_px):
            from training.common.vlm_grounding import set_pixel_budget

            set_pixel_budget(handle.processor, min_px, max_px)
            handle._grounding_pixel_budget = (min_px, max_px)

        meta = selected_image(manifest, params)
        image, _ = to_rgb_preview(meta, max_edge=1024)
        chat = build_chat(image, phrase, None)
        text = handle.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        batch = handle.processor(text=[text], images=[image], return_tensors="pt").to(handle.model.device)
        with torch.no_grad(), handle.using(VLM_ADAPTER_NAME) as model:
            generated = model.generate(
                **batch, max_new_tokens=48, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )
        reply = handle.processor.decode(
            generated.sequences[0][batch["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # Mean token probability over the reply, as the VQA tool reports;
        # calibrated per tool by configs/calibration*.json when a fit exists.
        confidence = rs_vqa._mean_token_probability(torch, generated.scores)

        box = parse_box(reply)
        width, height = meta.width, meta.height
        if box is None:
            return ToolResult(
                tool=TOOL_NAME, version=TOOL_VERSION,
                payload=GroundingPayload(data={"phrase": phrase, "bounding_boxes": [],
                                               "reply": reply, "image_size": [width, height]}),
                artifacts=[], confidence=0.0, confidence_method="no_assertion",
                model_card=f"{base.name} + grounding adapter ({adapter.name})",
                runtime_ms=int((time.perf_counter() - started) * 1000),
                warnings=warnings + ["model reply contained no box; nothing asserted"],
            )
        rh, rw = resized_hw(batch["image_grid_thw"][0])
        # model frame -> preview frame -> source pixel frame
        x0, y0, x1, y1 = scale_box(box, image.width / rw, image.height / rh)
        sx, sy = width / image.width, height / image.height
        x0, y0, x1, y1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(width), x1), min(float(height), y1)
        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION,
            payload=GroundingPayload(data={
                "phrase": phrase,
                "bounding_boxes": [{"x0": round(x0, 2), "y0": round(y0, 2),
                                    "x1": round(x1, 2), "y1": round(y1, 2), "label": phrase}],
                "normalised_cxcywh": [round(v, 6) for v in (
                    (x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
                    (x1 - x0) / width, (y1 - y0) / height)],
                "image_size": [width, height],
                "reply": reply,
            }),
            artifacts=[], confidence=confidence, confidence_method="logprob",
            model_card=f"{base.name} + grounding adapter ({adapter.name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]

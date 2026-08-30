"""`caption_v1` backed by the trained scene captioner (plan task 2.8).

Task 2.8 trained the captioner and reported BLEU-4 0.2446 on RSICD, but no
tool ever wired it into the pipeline - the registry kept the stub, so the
model was unreachable from a query. This is that wiring.

Activation is explicit and opt-in via `SATQUERY_CAPTION`, matching
`rs_vqa_v1` and `change_mask_v1`: without it the stub stays, so CI and
GPU-less machines keep a green suite rather than half-loading a model.

## The confidence this reports, and why it is not a probability

Mean per-token probability over a greedy decode, reported as
`confidence_method="logprob"`. That is a *fluency* signal, not P(correct) -
the model can be certain of every token in a caption describing the wrong
scene, which is exactly what task 2.8 measured: fluent, plausible
remote-sensing prose with only 13.4% unique captions. It is therefore
excluded from `CALIBRATABLE_CONFIDENCE_METHODS` along with everything else,
and it feeds the confidence combiner as one weak signal.

**The caption is the half of the answer the entailment gate exists for.**
Anything the physics can measure is described deterministically by
`synth/narrative.py` and checked against the indices; this model handles only
genuinely open-ended description, and task 3.5's gate removes any sentence it
produces that contradicts a measured index.
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
from satquery.tools.imaging import to_rgb_preview

TOOL_NAME = "caption"
TOOL_VERSION = "1.0.0"
ENV_CHECKPOINT = "SATQUERY_CAPTION"

def image_size() -> int:
    """The size the weights were fitted at, read from the training module.

    Looked up lazily, NOT imported at module scope. `training.train_caption`
    imports torch at import time, and `stubs.py` builds the registry when it
    is imported - so a module-level import here made the entire package
    unimportable on any machine without torch, which is every CI runner.
    Restating the number instead would let the tool drift from the weights,
    and a size mismatch degrades quality silently: the model still runs, it
    just sees the wrong scale.
    """
    from training.train_caption import IMAGE_SIZE

    return IMAGE_SIZE


class CaptionPayload(ToolPayload):
    data: dict[str, Any]


def is_available() -> tuple[bool, str]:
    path = os.getenv(ENV_CHECKPOINT)
    if not path:
        return False, f"{ENV_CHECKPOINT} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    # Checkpoint contents first, environment second: a missing vocab.json is a
    # mistake in the path the operator just supplied, and naming it is more
    # useful than "torch is not installed" when both are true.
    if not (Path(path) / "vocab.json").exists():
        # The vocabulary is built from the training captions and saved beside
        # the weights. Without it the token ids decode to nothing, which would
        # surface as an empty caption rather than an obvious failure.
        return False, f"vocab.json not found beside {path}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


class _Handle:
    """Lazily loaded captioner, shared process-wide."""

    _instance: "_Handle | None" = None
    _lock = threading.Lock()

    def __init__(self, checkpoint: Path):
        import torch

        from training.common.checkpointing import (
            find_latest_checkpoint,
            load_checkpoint,
            safe_torch_load,
        )
        from training.train_caption import build_model

        self.vocab: dict[str, int] = json.loads(
            (checkpoint / "vocab.json").read_text(encoding="utf-8")
        )
        self.inverse = {i: w for w, i in self.vocab.items()}

        latest = find_latest_checkpoint(checkpoint) or checkpoint
        payload = safe_torch_load(latest)
        dim = (payload.get("extra") or {}).get("dim", 192)
        model = build_model(vocab_size=len(self.vocab), dim=dim)
        load_checkpoint(latest, model, map_location="cpu")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.path = str(latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


def _image_array(meta, size: int) -> np.ndarray:
    """RGB array in the layout the captioner trained on.

    Routed through `to_rgb_preview`, the same function the VQA tool and the
    browser preview endpoint use, so what the model sees is what a user is
    shown - band selection and stretch included.
    """
    image, _ = to_rgb_preview(meta, max_edge=size)
    image = image.resize((size, size))
    return np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0


class CaptionTool(ToolProtocol):
    name = TOOL_NAME
    version = TOOL_VERSION

    def run(self, manifest: InputManifest, params: dict) -> ToolResult:
        started = time.perf_counter()
        checkpoint = Path(os.environ[ENV_CHECKPOINT])
        handle = _Handle.get(checkpoint)
        torch = handle.torch

        from training.train_change_caption import EOS, PAD

        array = _image_array(manifest.images[0], image_size())
        batch = torch.from_numpy(array).unsqueeze(0).to(handle.device)

        with torch.no_grad():
            tokens = handle.model.generate(batch)[0].tolist()

        words: list[str] = []
        for token in tokens:
            if token in (EOS, PAD):
                break
            word = handle.inverse.get(int(token))
            if word and not word.startswith("<"):
                words.append(word)
        caption = " ".join(words).strip()

        warnings: list[str] = []
        if not caption:
            # Never return an empty string: the executor turns an empty answer
            # into a named abstention, and "the captioner emitted nothing" is
            # more useful to a reader than silence.
            caption = "No caption could be generated for this image."
            warnings.append("captioner produced no tokens")

        confidence = _mean_token_probability(torch, handle, batch, tokens)

        return ToolResult(
            tool=TOOL_NAME,
            version=TOOL_VERSION,
            payload=CaptionPayload(
                data={
                    "caption": caption,
                    "n_tokens": len(words),
                    "vocab_size": len(handle.vocab),
                }
            ),
            artifacts=[],
            confidence=confidence,
            # Fluency, not correctness - see the module docstring.
            confidence_method="logprob",
            model_card=f"scene captioner ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]


def _mean_token_probability(torch, handle, batch, tokens) -> float:
    """Mean probability of the tokens the greedy decode chose.

    Re-runs the decoder teacher-forced on its own output rather than asking
    `generate` for scores, because `generate` returns only argmax ids. The
    values are identical - the same model on the same prefix - and this keeps
    the training script's generate() untouched.
    """
    if not tokens:
        return 0.0
    with torch.no_grad():
        hidden = handle.model.vision(batch).flatten(1).unsqueeze(0).contiguous()
        from training.train_change_caption import BOS

        token = torch.full((1, 1), BOS, dtype=torch.long, device=handle.device)
        probs = []
        for expected in tokens:
            out, hidden = handle.model.gru(handle.model.embed(token), hidden)
            step = torch.softmax(handle.model.out(out[:, -1]).float(), dim=-1)
            probs.append(float(step[0, int(expected)]))
            token = torch.full(
                (1, 1), int(expected), dtype=torch.long, device=handle.device
            )
    return round(sum(probs) / len(probs), 6) if probs else 0.0

"""Qwen2.5-VL grounding: prompt format, coordinate frames, output parsing.

Shared by the trainer and the evaluator so the two cannot drift - the
Phase 5 caption evaluators died on a `generate()` the trainer never
exercised, and the fix for that class of bug is one module, not two copies.

Coordinate frame
----------------
Qwen2.5-VL was pretrained to emit boxes in **absolute pixels of the image
the processor actually fed the model**, which is the input resized so both
sides are multiples of 28 (`smart_resize`). For an 800x800 DIOR image that
is 812x812. Targets are therefore scaled by (resized / original) before they
are rendered into the answer, and predictions are scaled back before IoU is
computed against the original-frame ground truth. The resized size is read
from `image_grid_thw` (patch grid x 14) rather than recomputed, so a change
of processor defaults cannot silently move the frame.

Answer format
-------------
The pretraining format, kept verbatim so the zero-shot model and the
fine-tuned one speak the same language::

    ```json
    [{"bbox_2d": [x1, y1, x2, y2], "label": "golffield"}]
    ```

`label` is the DIOR category, not the phrase: it is a free auxiliary
classification signal and it is short.
"""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_PROMPT = (
    "You are a remote-sensing image analyst. Locate objects precisely in the "
    "imagery you are shown."
)
PROMPT_TEMPLATE = (
    "Locate the {phrase} in the image and output its bounding box "
    "coordinates in JSON format."
)
PATCH = 14
_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def user_prompt(phrase: str) -> str:
    return PROMPT_TEMPLATE.format(phrase=phrase.strip().rstrip("."))


def render_answer(box_xyxy: list[float], label: str) -> str:
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    body = json.dumps([{"bbox_2d": [x1, y1, x2, y2], "label": label}])
    return f"```json\n{body}\n```"


def resized_hw(image_grid_thw) -> tuple[int, int]:
    """(height, width) in pixels of the image the model actually saw."""
    _, gh, gw = (int(v) for v in image_grid_thw)
    return gh * PATCH, gw * PATCH


def scale_box(box_xyxy, sx: float, sy: float) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def parse_box(text: str) -> list[float] | None:
    """First [x1,y1,x2,y2] in the reply, or None. Tolerates missing fences."""
    m = _BOX_RE.search(text)
    if not m:
        return None
    x1, y1, x2, y2 = (float(v) for v in m.groups())
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def iou_xyxy(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_chat(image, phrase: str, answer: str | None) -> list[dict[str, Any]]:
    chat: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt(phrase)},
        ]},
    ]
    if answer is not None:
        chat.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return chat


def assistant_start(input_ids, marker_ids: list[int]) -> int:
    """Index just past the LAST `<|im_start|>assistant\\n` in `input_ids`.

    Found by token search rather than by re-tokenising the prompt with the
    image, which halves the processor work per example and cannot disagree
    with the sequence actually being trained on.
    """
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    n = len(marker_ids)
    for i in range(len(ids) - n, -1, -1):
        if ids[i:i + n] == marker_ids:
            return i + n
    raise ValueError("assistant marker not found in input_ids")


def set_pixel_budget(processor, min_pixels: int | None = None, max_pixels: int | None = None) -> None:
    """Override the processor's resize budget. `min_pixels` above an image's
    native size UPSCALES it (more visual tokens per object - the lever for
    small objects); `max_pixels` caps it. Both keys are written in the two
    places transformers versions read them from."""
    ip = processor.image_processor
    size = getattr(ip, "size", None)
    if size is None:
        size = ip.size = {}
    if max_pixels:
        size["longest_edge"] = max_pixels
        ip.max_pixels = max_pixels
    if min_pixels:
        size["shortest_edge"] = min_pixels
        ip.min_pixels = min_pixels


def smart_resized_hw(processor, width: int, height: int) -> tuple[int, int]:
    """What the processor will resize (width, height) to, without running it."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size
    size = getattr(ip, "size", None) or {}
    min_px = size.get("shortest_edge", getattr(ip, "min_pixels", 56 * 56))
    max_px = size.get("longest_edge", getattr(ip, "max_pixels", 28 * 28 * 1280))
    h, w = smart_resize(height, width, factor=factor, min_pixels=min_px, max_pixels=max_px)
    return h, w


_MARKER_CACHE: dict[int, list[int]] = {}


def marker_ids(tokenizer) -> list[int]:
    key = id(tokenizer)
    if key not in _MARKER_CACHE:
        _MARKER_CACHE[key] = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    return _MARKER_CACHE[key]


def predict_batch(model, processor, images, phrases, max_new_tokens: int = 48):
    """Greedy boxes for a batch, returned in ORIGINAL image pixels.

    Left-padded so every sequence ends at the generation point; the reply is
    parsed and rescaled from the frame the model saw to the frame the
    ground truth lives in.
    """
    import torch

    tok = processor.tokenizer
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        texts = [
            processor.apply_chat_template(build_chat(img, ph, None), tokenize=False,
                                          add_generation_prompt=True)
            for img, ph in zip(images, phrases)
        ]
        batch = processor(text=texts, images=list(images), return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev
    batch = batch.to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, pad_token_id=tok.pad_token_id)
    new_tokens = out[:, batch["input_ids"].shape[1]:]
    replies = tok.batch_decode(new_tokens, skip_special_tokens=True)
    grids = batch["image_grid_thw"]
    preds, parsed = [], []
    for img, reply, grid in zip(images, replies, grids):
        box = parse_box(reply)
        parsed.append(box is not None)
        if box is None:
            preds.append(None)
            continue
        rh, rw = resized_hw(grid)
        preds.append(scale_box(box, img.width / rw, img.height / rh))
    return preds, replies, parsed


def size_bucket(box_xyxy, width: int, height: int) -> str:
    """COCO-style area buckets on the original frame, relative to the image."""
    area = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
    frac = area / float(width * height)
    if frac < 0.01:
        return "small(<1%)"
    if frac < 0.10:
        return "medium(1-10%)"
    return "large(>10%)"

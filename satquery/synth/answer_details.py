"""Structured answer: the finished trace laid out as labelled sections.

Every item is read from a tool's own output, the ingest record or the
confidence block; nothing here is generated, so the sections can only repeat
what the run measured. The prose answer (`trace.answer`) is unchanged.
"""

from __future__ import annotations

from typing import Any

TOOL_NAMES = {
    "rs_vqa_v1": "visual question answering (Qwen2.5-VL-3B + LoRA)",
    "caption_v1": "scene captioner (Qwen2.5-VL-3B + LoRA)",
    "grounding_v1": "visual grounding (Qwen2.5-VL-3B + LoRA)",
    "landcover_v1": "land-cover classifier (BigEarthNet, 19 classes)",
    "optsar_fusion_v1": "optical-SAR fusion segmenter",
    "change_mask_v1": "change detector",
    "change_caption_v1": "change captioner (Qwen2.5-VL-3B + LoRA)",
    "change_vqa_v1": "change question answering",
    "index_engine_v1": "spectral index engine",
}

INTENT_LABELS = {
    "ask": "a question about the image",
    "describe": "describe the scene",
    "locate": "find something in the image",
    "classify": "classify the land cover",
    "change_describe": "describe what changed",
    "change_ask": "a question about the change",
    "change_map": "map where it changed",
    "fuse": "combine optical and radar",
}

INDEX_LABELS = {
    "ndvi": "vegetation (NDVI)",
    "ndwi": "water (NDWI)",
    "mndwi": "water (MNDWI)",
    "ndbi": "built-up (NDBI)",
    "builtup_proxy": "likely built-up",
}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _area_km2(km2: float) -> str:
    if km2 >= 1.0:
        return f"{km2:.2f} km²"
    return f"{round(km2 * 1_000_000, -2):,.0f} m²"


def _position(cx: float, cy: float) -> str:
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    if row == "middle" and col == "centre":
        return "centre"
    if row == "middle":
        return f"middle {col}"
    return f"{row} {col}" if col != "centre" else f"{row} centre"


def _sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = text.replace(" .", ".")
    return text if text[-1] in ".!?" else text + "."


def _outputs(trace: dict, tool: str) -> dict | None:
    for step in trace.get("execution") or []:
        if step.get("tool") == tool:
            return step.get("outputs") or {}
    return None


def _findings(trace: dict) -> list[str]:
    items: list[str] = []

    vqa = _outputs(trace, "rs_vqa_v1")
    if vqa and vqa.get("answer"):
        u = (trace.get("routing") or {}).get("understanding") or {}
        answer = vqa["answer"].strip().rstrip(".")
        if u.get("quantity") == "count" and answer.isdigit() and u.get("object_filter"):
            items.append(f"Counted: {answer} {u['object_filter']}")
        else:
            items.append(f"Answer: {_sentence(vqa['answer'])}")

    cap = _outputs(trace, "caption_v1")
    if cap and (cap.get("caption") or cap.get("answer")):
        items.append(f"Scene description: {_sentence(cap.get('caption') or cap.get('answer'))}")

    lc = _outputs(trace, "landcover_v1")
    if lc is not None:
        asserted = lc.get("asserted") or []
        for a in asserted:
            items.append(f"Land cover: {a['class']} (probability {a['probability']:.2f})")
        if not asserted:
            items.append("Land cover: no class reached the classifier's decision threshold")
        undecided = len(lc.get("abstained") or [])
        if undecided:
            items.append(f"Land cover: undecided on {undecided} class(es) near the threshold")
        if lc.get("n_denied"):
            items.append(f"Land cover: {lc['n_denied']} other classes ruled out")

    gr = _outputs(trace, "grounding_v1")
    if gr is not None:
        boxes = gr.get("normalised_cxcywh") or []
        if boxes and not isinstance(boxes[0], (list, tuple)):
            boxes = [boxes]
        phrase = gr.get("phrase") or "the object"
        if not boxes:
            items.append(f"Located: no region matching “{phrase}”")
        for i, (cx, cy, w, h) in enumerate(boxes, 1):
            label = f"Region {i}" if len(boxes) > 1 else "Located"
            items.append(
                f"{label}: “{phrase}” in the {_position(cx, cy)} of the image, "
                f"covering about {_pct(w * h)} of it"
            )

    cm = _outputs(trace, "change_mask_v1")
    if cm and cm.get("changed_fraction") is not None:
        line = f"Changed area: {_pct(cm['changed_fraction'])} of the scene"
        if cm.get("changed_area_km2") is not None:
            line += f" ({_area_km2(cm['changed_area_km2'])})"
        items.append(line)
        detail = f"Change threshold {cm.get('threshold')}"
        if cm.get("tta") and cm["tta"] != "none":
            detail += ", 8-fold test-time augmentation"
        items.append(detail)

    cc = _outputs(trace, "change_caption_v1")
    if cc and (cc.get("caption") or cc.get("answer")):
        items.append(f"What changed: {_sentence(cc.get('caption') or cc.get('answer'))}")

    cv = _outputs(trace, "change_vqa_v1")
    if cv and cv.get("answer"):
        items.append(f"Change answer: {_sentence(cv['answer'])}")

    fu = _outputs(trace, "optsar_fusion_v1")
    if fu and fu.get("answer"):
        items.append(f"Fusion: {_sentence(fu['answer'])}")

    ie = _outputs(trace, "index_engine_v1")
    if ie:
        for key, entry in (ie.get("indices") or {}).items():
            frac = (entry or {}).get("fraction_above_threshold")
            if key in INDEX_LABELS and frac is not None:
                items.append(f"Index: {INDEX_LABELS[key]} above threshold on {_pct(frac)} of the scene")
    return items


def _scene(trace: dict) -> list[str]:
    items: list[str] = []
    images = (trace.get("ingest") or {}).get("images") or []
    for img in images:
        bands = ", ".join(img.get("bands") or []) or "unknown bands"
        line = f"{img.get('role', 'image')}: {img.get('container_format') or 'image'}, {bands}"
        if img.get("gsd_m"):
            line += f", {img['gsd_m']} m per pixel"
        items.append(line)
    geo = next((i for i in images if i.get("georeferenced") and i.get("lonlat_bounds")), None)
    if geo:
        w, s, e, n = geo["lonlat_bounds"]
        lat, lon = (s + n) / 2, (w + e) / 2
        items.append(
            f"Centre: {abs(lat):.4f}° {'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.4f}° {'E' if lon >= 0 else 'W'}"
        )
    elif images:
        items.append("Not georeferenced: no coordinates or ground areas can be given")
    return items


def _method(trace: dict) -> list[str]:
    items: list[str] = []
    routing = trace.get("routing") or {}
    u = routing.get("understanding") or {}
    if u.get("intent"):
        line = f"Understood as: {INTENT_LABELS.get(u['intent'], u['intent'].replace('_', ' '))}"
        if u.get("resolved_query") and u.get("resolved_query") != u.get("query"):
            line += f" (“{u['resolved_query']}”)"
        items.append(line)
    tools = [s.get("tool") for s in trace.get("execution") or []]
    named = [TOOL_NAMES.get(t, t) for t in tools if t]
    if named:
        items.append("Models used: " + "; ".join(named))
    conf = trace.get("confidence") or {}
    if conf.get("final") is not None:
        items.append(f"Confidence: {conf.get('band', '')} ({conf['final']:.2f})")
    return items


def _limits(trace: dict) -> list[str]:
    items: list[str] = []
    conf = trace.get("confidence") or {}
    method = (conf.get("calibration") or {}).get("method", "")
    if method.startswith("uncalibrated"):
        items.append("The confidence score is a ranking signal, not a calibrated probability")
    for step in trace.get("execution") or []:
        if step.get("confidence_method") == "stub":
            items.append(f"{TOOL_NAMES.get(step['tool'], step['tool'])} is not loaded; its output is a placeholder")
    lc = _outputs(trace, "landcover_v1")
    if lc and lc.get("asserted"):
        images = (trace.get("ingest") or {}).get("images") or []
        gsd = next((i.get("gsd_m") for i in images if i.get("gsd_m")), None)
        if (lc.get("bands_present") or 0) < 4 or gsd is None or gsd < 5:
            items.append(
                "The land-cover classifier was trained on 10 m Sentinel-2 imagery; "
                "on this image (" + (f"{gsd} m" if gsd else "unknown resolution")
                + f", {lc.get('bands_present', '?')} bands) its classes are indicative only"
            )
    gate = (trace.get("verification") or {}).get("entailment_gate") or {}
    if gate.get("flagged"):
        items.append(f"{gate['flagged']} sentence(s) contradicted the measurements and were flagged")
    return items


def answer_details(trace: dict[str, Any]) -> dict | None:
    """Sections for a completed, non-abstained run; None otherwise."""
    if not trace or trace.get("abstained") or not trace.get("answer"):
        return None
    sections = [
        {"title": "Findings", "items": _findings(trace)},
        {"title": "Imagery", "items": _scene(trace)},
        {"title": "How this was answered", "items": _method(trace)},
        {"title": "Limits", "items": _limits(trace)},
    ]
    return {"sections": [s for s in sections if s["items"]]}

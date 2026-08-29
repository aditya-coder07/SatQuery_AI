"""Deterministic narrative synthesis.

Turns structured tool output into a human-readable sentence. Every number in
the text comes from the deterministic index engine, so the narrative cannot
assert something the physics does not support - which is the property the
entailment gate (task 3.5) will later check automatically.

This is deliberately template-based rather than generative. A generated
summary of already-structured numbers adds a hallucination risk for no
benefit; the learned models are for tasks that genuinely need language
understanding.
"""

from __future__ import annotations

# Fraction of the scene above an index threshold, below which a class is not
# worth mentioning at all.
MENTION_THRESHOLD = 0.05

INDEX_MEANING = {
    "ndvi": "vegetation",
    "ndwi": "water",
    "mndwi": "water",
    "ndbi": "built-up land",
    "builtup_proxy": "likely built-up land",
}


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.0f}%"


def describe_indices(index_payload: dict) -> str:
    """One sentence describing scene composition from index statistics."""
    indices = index_payload.get("indices", {})
    parts: list[str] = []

    # MNDWI supersedes NDWI when both exist: it is the better water index.
    water_key = "mndwi" if "mndwi" in indices else "ndwi"
    ordered = ["ndvi", water_key, "ndbi", "builtup_proxy"]

    seen_meanings: set[str] = set()
    for key in ordered:
        entry = indices.get(key)
        if not entry:
            continue
        meaning = INDEX_MEANING.get(key)
        if meaning is None or meaning in seen_meanings:
            continue
        fraction = entry.get("fraction_above_threshold")
        if fraction is None or fraction < MENTION_THRESHOLD:
            continue
        seen_meanings.add(meaning)
        parts.append(f"{_percent(fraction)} {meaning}")

    if not parts:
        return "No dominant land-cover class exceeded its detection threshold."

    if len(parts) == 1:
        return f"Index thresholds indicate {parts[0]} coverage."

    body = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    # These fractions come from independent per-index thresholds, so they
    # overlap and will not sum to 100%. Saying "the scene is X% A and Y% B"
    # would imply a partition that the indices do not provide, so the
    # wording states the overlap explicitly instead.
    return (
        f"Index thresholds indicate {body}. These classes are measured "
        "independently and may overlap."
    )


def describe_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    readable = [str(label).replace("_", " ") for label in labels]
    if len(readable) == 1:
        return f"Detected land-cover class: {readable[0]}."
    return "Detected land-cover classes: " + ", ".join(readable) + "."


def synthesise_answer(
    task: str, step_outputs: list[dict], index_payload: dict
) -> str:
    """Build an answer for tasks whose tools return structure, not prose.

    Returns an empty string when there is nothing defensible to say, so the
    caller can fall back rather than emitting a fabricated sentence.
    """
    labels: list[str] = []
    boxes: list = []
    for out in step_outputs:
        labels.extend(out.get("labels", []) or [])
        boxes.extend(out.get("bounding_boxes", []) or [])

    if task == "SINGLE_LANDCOVER":
        pieces = [describe_labels(labels), describe_indices(index_payload)]
        return " ".join(p for p in pieces if p)

    if task == "SINGLE_GROUND":
        if boxes:
            noun = "region" if len(boxes) == 1 else "regions"
            return f"Localised {len(boxes)} matching {noun}."
        return "No matching region was localised."

    if task in ("SINGLE_CAPTION", "TEMPORAL_CHANGE_DESC"):
        return describe_indices(index_payload)

    if task == "TEMPORAL_CHANGE_MAP":
        return "Produced a change mask; see the exported raster artifact."

    # VQA-style tasks are answered by the model itself, not synthesised here.
    return ""

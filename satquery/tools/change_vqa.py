"""`change_vqa_v1` deterministic template path (task 2.6).

The plan is explicit that the template path comes *first*, before the CDVQA
head. The reason is sound: "how much did the water area change" has an exact
arithmetic answer from two index rasters, and asking a neural model to
estimate a number the physics already knows exactly is strictly worse - it
adds error and removes auditability.

So this answers area-delta questions by measuring, and defers anything it
cannot measure rather than guessing. `confidence_method` is "deterministic"
because the arithmetic has no uncertainty; the uncertainty lives in the
thresholds, which are reported separately.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import ToolPayload, ToolResult
from satquery.ingest.reader import read_canonical_band
from satquery.tools.base import ToolProtocol
from satquery.verify.indices import mndwi, ndvi, ndwi
from satquery.verify.thresholding import adaptive_threshold, apply_threshold
from satquery.verify.verifier import SUBJECT_TERMS

TOOL_NAME = "change_vqa"
TOOL_VERSION = "1.0.0-template"

FIXED_PRIORS = {"vegetation": 0.3, "water": 0.0}

# Question shapes this path can answer exactly.
_QUANTITY_RE = re.compile(
    r"\b(how much|how many|by what|what (?:is|was) the (?:net )?change|"
    r"quantify|percentage|percent|area)\b", re.IGNORECASE
)
_DIRECTION_RE = re.compile(
    r"\b(increase|decrease|grow|shrink|expand|reduce|more|less|gain|loss|lost)\b",
    re.IGNORECASE,
)


class ChangeVQAPayload(ToolPayload):
    data: dict[str, Any]


def subject_of(question: str) -> str | None:
    lowered = question.lower()
    for subject, terms in SUBJECT_TERMS.items():
        if any(t in lowered for t in terms):
            return subject
    return None


def _index_for(subject: str, meta) -> tuple[np.ndarray, str] | None:
    """Compute the index that measures `subject` for one image."""
    bands = set(meta.bands)
    if subject == "vegetation" and {"RED", "NIR"} <= bands:
        return ndvi(read_canonical_band(meta, "RED"),
                    read_canonical_band(meta, "NIR")), "ndvi"
    if subject == "water":
        if {"GREEN", "SWIR1"} <= bands:
            return mndwi(read_canonical_band(meta, "GREEN"),
                         read_canonical_band(meta, "SWIR1")), "mndwi"
        if {"GREEN", "NIR"} <= bands:
            return ndwi(read_canonical_band(meta, "GREEN"),
                        read_canonical_band(meta, "NIR")), "ndwi"
    return None


def measure_change(subject: str, t1, t2) -> dict | None:
    """Fraction of each scene above the index threshold, and the delta.

    The threshold is derived from t1 and applied to both dates. Re-deriving it
    per date would let a threshold shift masquerade as real change, which is
    the classic way naive change detection invents results.
    """
    a = _index_for(subject, t1)
    b = _index_for(subject, t2)
    if a is None or b is None:
        return None

    arr1, index_name = a
    arr2, _ = b

    threshold = adaptive_threshold(arr1, fixed_prior=FIXED_PRIORS.get(subject, 0.0))
    m1 = apply_threshold(arr1, threshold)
    m2 = apply_threshold(arr2, threshold)

    f1 = float(m1.sum()) / max(m1.size, 1)
    f2 = float(m2.sum()) / max(m2.size, 1)
    gsd = t1.gsd_m or 0.0
    area_km2 = (f2 - f1) * m1.size * gsd * gsd / 1e6

    return {
        "index": index_name,
        "threshold": round(threshold.value, 6),
        "threshold_method": threshold.method,
        "fraction_t1": round(f1, 6),
        "fraction_t2": round(f2, 6),
        "delta_fraction": round(f2 - f1, 6),
        "relative_change": round((f2 - f1) / f1, 6) if f1 > 0 else None,
        "delta_area_km2": round(area_km2, 4),
    }


def phrase(subject: str, m: dict) -> str:
    delta = m["delta_fraction"]
    direction = "increased" if delta > 0 else "decreased" if delta < 0 else "did not change"
    if delta == 0:
        return f"The {subject} extent did not change measurably between the two dates."
    rel = (
        f" ({abs(m['relative_change']):.0%} relative)"
        if m["relative_change"] is not None else ""
    )
    return (
        f"The {subject} extent {direction} from {m['fraction_t1']:.1%} to "
        f"{m['fraction_t2']:.1%} of the scene{rel}, a change of about "
        f"{abs(m['delta_area_km2']):.2f} km2, measured by {m['index'].upper()}."
    )


class ChangeVQATemplate(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        warnings: list[str] = []
        question = str(params.get("_query") or "")

        if len(manifest.images) != 2:
            return self._defer(
                started, "change questions need two images; only one was supplied"
            )

        t1, t2 = manifest.images
        subject = subject_of(question)
        answerable = bool(_QUANTITY_RE.search(question) or _DIRECTION_RE.search(question))

        if subject is None or not answerable:
            return self._defer(
                started,
                "this question is not an area-change measurement; the "
                "deterministic path defers to the learned change-VQA head "
                "(task 2.6, not yet trained)",
            )

        measured = measure_change(subject, t1, t2)
        if measured is None:
            return self._defer(
                started,
                f"cannot measure {subject} change: the required bands are not "
                f"present in both images ({t1.bands} / {t2.bands})",
            )

        if measured["threshold_method"] == "fixed_prior":
            warnings.append(
                f"{measured['index']} threshold fell back to a fixed prior; "
                "the change estimate is less reliable"
            )

        payload = ChangeVQAPayload(
            data={
                "answer": phrase(subject, measured),
                "question": question,
                "subject": subject,
                "measurement": measured,
                "path": "deterministic_template",
            }
        )
        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION, payload=payload, artifacts=[],
            confidence=1.0 if measured["threshold_method"] != "fixed_prior" else 0.6,
            confidence_method="deterministic",
            model_card="change_vqa_v1 template path (closed-form index delta)",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def _defer(self, started: float, reason: str) -> ToolResult:
        """Say what cannot be measured rather than guessing at it."""
        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION,
            payload=ChangeVQAPayload(data={"answer": "", "deferred": True,
                                           "reason": reason, "path": "deferred"}),
            artifacts=[], confidence=0.0, confidence_method="deterministic",
            model_card="change_vqa_v1 template path (deferred)",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=[reason],
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]

"""Layer 0 ingest pipeline: files on disk -> validated `InputManifest`.

This is the entry point the controller calls. It infers the input
configuration, reads every image, runs the checks, co-registers pairs, and
reports which spectral indices are computable so the planner can route around
missing bands instead of failing.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from satquery.contracts.input_manifest import (
    ImageMeta,
    IngestMode,
    InputManifest,
    TilingReport,
)

from .checks import blocking_failures, run_checks
from .coreg import coregister
from .modality import index_availability
from .reader import read_image

OPTICAL_MODALITIES = {"OPTICAL", "MSI", "PAN"}

# Scenes larger than this are candidates for the tile pyramid (task 2.10).
# Phase 1 records the need without implementing retrieval.
TILING_TRIGGER_PX = 4096
TILE_SIZE_PX = 1024


def infer_config(images: list[ImageMeta]) -> str:
    """SINGLE / CROSSMODAL_PAIR / BITEMPORAL_PAIR from the images themselves."""
    if len(images) == 1:
        return "SINGLE"
    if len(images) != 2:
        raise ValueError(f"expected 1 or 2 images, got {len(images)}")

    mods = [img.modality for img in images]
    has_sar = "SAR" in mods
    has_optical = any(m in OPTICAL_MODALITIES for m in mods)
    if has_sar and has_optical:
        return "CROSSMODAL_PAIR"
    return "BITEMPORAL_PAIR"


def assign_roles(images: list[ImageMeta], config: str) -> list[ImageMeta]:
    """Set each image's role to match the inferred configuration."""
    if config == "SINGLE":
        return [images[0].model_copy(update={"role": "single"})]

    if config == "CROSSMODAL_PAIR":
        out = []
        for img in images:
            role = "sar" if img.modality == "SAR" else "optical"
            out.append(img.model_copy(update={"role": role}))
        # Keep optical first for a stable reference image in co-registration.
        out.sort(key=lambda i: i.role != "optical")
        return out

    # BITEMPORAL_PAIR: order by acquisition date when both are known,
    # otherwise trust the caller's input order.
    ordered = list(images)
    if all(i.acquisition_dt is not None for i in ordered):
        ordered.sort(key=lambda i: i.acquisition_dt)
    return [
        ordered[0].model_copy(update={"role": "t1"}),
        ordered[1].model_copy(update={"role": "t2"}),
    ]


def build_tiling_report(images: list[ImageMeta]) -> TilingReport:
    """Report whether the scene is large enough to need the tile pyramid."""
    largest = max((img.width * img.height) for img in images)
    biggest_dim = max(max(img.width, img.height) for img in images)
    if biggest_dim < TILING_TRIGGER_PX:
        return TilingReport(applied=False)
    tiles = -(-largest // (TILE_SIZE_PX * TILE_SIZE_PX))  # ceiling division
    return TilingReport(
        applied=False,
        level1_tiles=int(tiles),
        retrieved_tiles=None,
        retrieval_reason=(
            f"scene exceeds {TILING_TRIGGER_PX}px; coarse-to-fine retrieval "
            "is Phase 2 (task 2.10) - full scene processed for now"
        ),
    )


def merged_index_availability(images: list[ImageMeta]) -> dict[str, bool]:
    """Union band presence across images, then derive index availability."""
    n = len(images[0].band_presence)
    merged = [any(img.band_presence[i] for img in images) for i in range(n)]
    modalities = [img.modality for img in images]
    pols: list[str] = []
    for img in images:
        for p in img.polarisations or []:
            if p not in pols:
                pols.append(p)
    return index_availability(merged, modalities, pols)


def ingest(
    paths: list[str | Path],
    mode: IngestMode = IngestMode.OPERATIONAL,
    benchmark: str | None = None,
    run_id: str | None = None,
) -> InputManifest:
    """Build an `InputManifest` from one or two raster paths."""
    if not paths:
        raise ValueError("at least one image path is required")
    if len(paths) > 2:
        raise ValueError(f"at most two images are supported, got {len(paths)}")

    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

    images = [read_image(p) for p in paths]
    config = infer_config(images)
    images = assign_roles(images, config)

    checks = run_checks(images, config)
    failures = blocking_failures(checks)

    coreg = None
    if len(images) == 2 and not failures:
        # Co-registration reads pixels; skip it when the inputs are already
        # known-bad, so a broken file surfaces as a check failure rather than
        # an exception from deep inside the correlator.
        coreg = coregister(images[0], images[1])

    return InputManifest(
        run_id=run_id,
        ingest_mode=mode,
        benchmark=benchmark,
        images=images,
        config=config,
        checks=checks,
        coreg=coreg,
        tiling=build_tiling_report(images),
        artifacts={},
        blocking_failures=failures,
        index_availability=merged_index_availability(images),
    )

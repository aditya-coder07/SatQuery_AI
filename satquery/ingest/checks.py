"""Input validation checks.

Each check returns a `CheckResult`. FAIL statuses become blocking failures and
force the planner toward CLARIFY_OR_ABSTAIN rather than producing an answer
the physics cannot support (docs/02). WARN statuses are surfaced in the trace
and feed the input-quality confidence component.
"""

from __future__ import annotations

from satquery.contracts.input_manifest import CheckResult, ImageMeta

# Thresholds. Kept here rather than inline so configs/thresholds.yaml can
# override them later without hunting through logic.
NODATA_WARN_PCT = 20.0
NODATA_FAIL_PCT = 80.0
GSD_RATIO_WARN = 4.0
GSD_RATIO_FAIL = 20.0
MIN_DIMENSION_PX = 32


def _pass(name: str, message: str, value=None) -> CheckResult:
    return CheckResult(name=name, status="PASS", value=value, message=message)


def _warn(name: str, message: str, value=None, threshold=None) -> CheckResult:
    return CheckResult(
        name=name, status="WARN", value=value, threshold=threshold, message=message
    )


def _fail(name: str, message: str, value=None, threshold=None) -> CheckResult:
    return CheckResult(
        name=name, status="FAIL", value=value, threshold=threshold, message=message
    )


def check_crs_present(img: ImageMeta) -> CheckResult:
    if img.crs and img.crs != "UNKNOWN":
        return _pass("crs_present", f"{img.role}: CRS {img.crs}", img.crs)
    return _fail(
        "crs_present",
        f"{img.role}: no CRS - cannot georeference outputs or co-register",
        img.crs,
    )


def check_nodata(img: ImageMeta) -> CheckResult:
    v = img.nodata_pct
    if v >= NODATA_FAIL_PCT:
        return _fail(
            "nodata_fraction",
            f"{img.role}: {v:.1f}% nodata - image is mostly empty",
            v,
            NODATA_FAIL_PCT,
        )
    if v >= NODATA_WARN_PCT:
        return _warn(
            "nodata_fraction",
            f"{img.role}: {v:.1f}% nodata - answers may cover only part of the scene",
            v,
            NODATA_WARN_PCT,
        )
    return _pass("nodata_fraction", f"{img.role}: {v:.1f}% nodata", v)


def check_dimensions(img: ImageMeta) -> CheckResult:
    smallest = min(img.width, img.height)
    if smallest < MIN_DIMENSION_PX:
        return _fail(
            "min_dimension",
            f"{img.role}: {img.width}x{img.height} is too small to analyse",
            smallest,
            MIN_DIMENSION_PX,
        )
    return _pass("min_dimension", f"{img.role}: {img.width}x{img.height}", smallest)


def check_crs_match(a: ImageMeta, b: ImageMeta) -> CheckResult:
    if a.crs == b.crs:
        return _pass("crs_match", f"both images in {a.crs}", a.crs)
    return _warn(
        "crs_match",
        f"CRS mismatch ({a.crs} vs {b.crs}) - reprojection required before co-registration",
        f"{a.crs}|{b.crs}",
    )


def check_gsd_ratio(a: ImageMeta, b: ImageMeta) -> CheckResult:
    lo, hi = sorted((a.gsd_m, b.gsd_m))
    if lo <= 0:
        return _fail("gsd_ratio", "non-positive GSD reported", lo)
    ratio = hi / lo
    if ratio >= GSD_RATIO_FAIL:
        return _fail(
            "gsd_ratio",
            f"GSD ratio {ratio:.1f}x is too large to compare meaningfully",
            round(ratio, 3),
            GSD_RATIO_FAIL,
        )
    if ratio >= GSD_RATIO_WARN:
        return _warn(
            "gsd_ratio",
            f"GSD ratio {ratio:.1f}x - the coarser image limits achievable detail",
            round(ratio, 3),
            GSD_RATIO_WARN,
        )
    return _pass("gsd_ratio", f"GSD ratio {ratio:.2f}x", round(ratio, 3))


def check_crossmodal_pairing(images: list[ImageMeta]) -> CheckResult:
    mods = {img.modality for img in images}
    has_sar = "SAR" in mods
    has_optical = bool(mods & {"OPTICAL", "MSI", "PAN"})
    if has_sar and has_optical:
        return _pass("crossmodal_pairing", "one optical and one SAR image present")
    return _fail(
        "crossmodal_pairing",
        f"cross-modal analysis needs one optical and one SAR image; got {sorted(mods)}",
        ",".join(sorted(mods)),
    )


def check_temporal_order(a: ImageMeta, b: ImageMeta) -> CheckResult:
    if a.acquisition_dt is None or b.acquisition_dt is None:
        return _warn(
            "temporal_order",
            "acquisition dates missing - t1/t2 order taken from input order",
        )
    if a.acquisition_dt <= b.acquisition_dt:
        return _pass(
            "temporal_order",
            f"t1 {a.acquisition_dt.date()} precedes t2 {b.acquisition_dt.date()}",
        )
    return _warn(
        "temporal_order",
        f"t1 {a.acquisition_dt.date()} is AFTER t2 {b.acquisition_dt.date()} - "
        "change direction will be reported reversed unless inputs are swapped",
    )


def run_checks(images: list[ImageMeta], config: str) -> list[CheckResult]:
    """Run every check applicable to this input configuration."""
    results: list[CheckResult] = []
    for img in images:
        results.append(check_crs_present(img))
        results.append(check_nodata(img))
        results.append(check_dimensions(img))

    if len(images) == 2:
        a, b = images
        results.append(check_crs_match(a, b))
        results.append(check_gsd_ratio(a, b))
        if config == "CROSSMODAL_PAIR":
            results.append(check_crossmodal_pairing(images))
        elif config == "BITEMPORAL_PAIR":
            results.append(check_temporal_order(a, b))

    return results


def blocking_failures(checks: list[CheckResult]) -> list[str]:
    """Names of checks that must stop the pipeline producing a real answer."""
    return [c.name for c in checks if c.status == "FAIL"]

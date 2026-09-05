"""Raster reader: turns a file on disk into a validated `ImageMeta`.

Everything here is metadata-driven with statistical fallbacks. No sensor is
assumed, because the evaluation set (Cartosat/RISAT) differs from the training
set (Sentinel-1/2) in band layout, GSD and dtype.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from satquery.contracts.input_manifest import ImageMeta

from .modality import detect_polarisations, harmonise_bands, infer_modality
from .product import resolve as resolve_product

# Metadata keys that commonly carry an acquisition timestamp.
_DATE_KEYS = (
    "ACQUISITION_DATE",
    "ACQUISITION_DATETIME",
    "DATE_ACQUIRED",
    "IMAGING_DATE",
    "TIFFTAG_DATETIME",
    "DATETIME",
    "START_TIME",
)

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d",
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y",
)

# How many pixels to sample for statistics. Full reads are wasteful on the
# 8000x8000 scenes docs/04 expects, and a sample is sufficient for dtype
# range, nodata fraction and the backscatter heuristic.
_SAMPLE_MAX = 512


def parse_acquisition_dt(tags: dict) -> datetime | None:
    """Best-effort acquisition timestamp from vendor metadata."""
    for key in _DATE_KEYS:
        for tag_key, raw in tags.items():
            if tag_key.upper() != key:
                continue
            value = str(raw).strip()
            for fmt in _DATE_FORMATS:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
    return None


def estimate_effective_bits(sample: np.ndarray, dtype: str) -> int:
    """Actual used bit depth, which often differs from the container dtype.

    12-bit sensor data is routinely delivered in a uint16 container. Knowing
    the real depth matters for normalisation - scaling by 65535 when the data
    only spans 0-4095 crushes contrast.
    """
    if dtype.startswith("float"):
        return 32 if dtype == "float32" else 64
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return int(np.dtype(dtype).itemsize * 8)
    peak = float(finite.max())
    if peak <= 0:
        return int(np.dtype(dtype).itemsize * 8)
    bits = int(np.ceil(np.log2(peak + 1)))
    return max(1, min(bits, int(np.dtype(dtype).itemsize * 8)))


def _read_sample(src) -> np.ndarray:
    """Read a decimated sample of band 1 for statistics."""
    out_h = min(src.height, _SAMPLE_MAX)
    out_w = min(src.width, _SAMPLE_MAX)
    arr = src.read(1, out_shape=(out_h, out_w), masked=True)
    return np.ma.filled(arr.astype("float64"), np.nan)


def _gsd_metres(src) -> float:
    """Ground sample distance in metres, converting from degrees if needed.

    For an ungeoreferenced raster - a plain PNG or JPEG - GDAL hands back the
    identity transform, so this returns 1.0. That is a placeholder, not a
    measurement, and `ImageMeta.georeferenced` is what tells the trace and the
    tools apart; see the note on that field.
    """
    x_res = abs(src.transform.a)
    if src.crs is not None and src.crs.is_geographic:
        # Approximate: 1 degree of latitude ~= 111320 m. Good enough for a
        # manifest field used for ordering-of-magnitude decisions; projected
        # products (the normal case) are exact.
        return float(x_res * 111_320.0)
    return float(x_res)


def _footprint(
    src,
) -> tuple[tuple[float, float, float, float] | None, tuple[float, float] | None]:
    """WGS84 bounds and centre of a raster, or `(None, None)`.

    Returns nothing rather than a guess in every case where the file does not
    actually say where it is:

    * no CRS - a PNG or JPEG cannot carry one;
    * the identity transform - GDAL hands this back for an ungeoreferenced
      raster, and a GeoTIFF written without a geotransform gets it too. The
      bounds would come out as pixel indices dressed as coordinates, which is
      worse than silence because they look plausible;
    * a projection that will not transform - a broken or exotic CRS must
      degrade the answer, not fail the ingest.

    `densify_pts=21` matches `report/evidence_pack.raster_footprint`: a
    reprojected rectangle has curved edges, and sampling the sides keeps the
    envelope from cutting the corners off the real footprint.
    """
    if src.crs is None or src.transform.is_identity:
        return None, None
    try:
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds, densify_pts=21
        )
    except Exception:  # noqa: BLE001 - a bad CRS must not fail the ingest
        return None, None

    if not all(map(math.isfinite, (west, south, east, north))):
        return None, None

    bounds = (float(west), float(south), float(east), float(north))
    # (latitude, longitude), in that order - the order a reader says them in.
    centroid = ((south + north) / 2.0, (west + east) / 2.0)
    return bounds, (float(centroid[0]), float(centroid[1]))


def read_canonical_band(meta: ImageMeta, band: str) -> np.ndarray:
    """Read one canonical band (e.g. "RED") from an image as float64.

    Raises KeyError if the band is not present - callers must check
    `index_availability` first rather than relying on an exception.
    """
    if band not in meta.bands:
        raise KeyError(
            f"band {band!r} not present in {meta.path.name}; has {meta.bands}"
        )
    idx = meta.bands.index(band) + 1  # rasterio bands are 1-indexed
    with rasterio.open(meta.path) as src:
        arr = src.read(idx, masked=True)
    return np.ma.filled(arr.astype("float64"), np.nan)


def read_image(
    path: str | Path,
    role: Literal["single", "optical", "sar", "t1", "t2"] = "single",
) -> ImageMeta:
    """Open a raster and build its `ImageMeta`. Raises on unreadable files."""
    # Vendor products ship one file per band (Cartosat MX: BAND1..4.tif;
    # EOS-04: scene_<POL>/imagery_<POL>.tif). resolve_product() unifies those
    # into a single openable path via a VRT, and hands back the vendor
    # metadata, which carries the band/polarisation identities and the radar
    # frequency that the raster headers do not.
    original = Path(path)
    path, layout = resolve_product(original)

    with rasterio.open(path) as src:
        tags = dict(src.tags())
        descriptions = list(src.descriptions)

        # Vendor band/polarisation names beat the raster header, which for
        # these products is empty.
        if layout.band_names and len(layout.band_names) == src.count:
            descriptions = list(layout.band_names)

        # Surface vendor metadata as tags so modality inference can see the
        # satellite, sensor and polarisations it would otherwise miss.
        for key in ("satellite", "sensor", "radar_band", "imaging_mode"):
            value = layout.metadata.get(key)
            if value:
                tags[key.upper()] = str(value)
        if layout.metadata.get("polarisations"):
            tags["POLARISATION"] = " ".join(layout.metadata["polarisations"])
        sample = _read_sample(src)

        modality, evidence = infer_modality(
            band_count=src.count,
            dtype=src.dtypes[0],
            tags=tags,
            band_descriptions=descriptions,
            sample=sample,
        )

        # Carry vendor-level limitations into the evidence dict so the
        # checks can name the real reason a product is unusable, rather than
        # only reporting the downstream symptom (a missing CRS).
        for key in ("requires_geocoding", "unsupported_reason", "processing_level",
                    "radar_frequency_ghz", "radar_band", "n_beams"):
            if layout.metadata.get(key) is not None:
                evidence[key] = layout.metadata[key]

        bands, band_presence = harmonise_bands(descriptions, modality)
        pols = detect_polarisations(tags, descriptions)

        finite = sample[np.isfinite(sample)]
        nodata_pct = float(100.0 * (1.0 - finite.size / sample.size)) if sample.size else 0.0

        bounds_wgs84, centroid_wgs84 = _footprint(src)

        sensor_guess = (
            tags.get("SATELLITE")
            or tags.get("SENSOR")
            or tags.get("MISSION")
            or tags.get("PLATFORM")
        )

        return ImageMeta(
            role=role,
            path=path,
            modality=modality,
            modality_evidence=evidence,
            crs=str(src.crs) if src.crs else "UNKNOWN",
            gsd_m=_gsd_metres(src),
            width=src.width,
            height=src.height,
            bands=bands,
            band_presence=band_presence,
            dtype=src.dtypes[0],
            effective_bits=estimate_effective_bits(sample, src.dtypes[0]),
            acquisition_dt=parse_acquisition_dt(tags),
            nodata_pct=round(nodata_pct, 4),
            cloud_pct=None,  # requires a cloud mask; Phase 2 work
            sensor_guess=str(sensor_guess) if sensor_guess else None,
            polarisations=pols or None,
            # Equivalent number of looks, from the vendor's RangeLooks x
            # AzimuthLooks. Confirmed present in real EOS-04 metadata
            # (verification item 5), so it is no longer a placeholder.
            look_count_est=layout.metadata.get("equivalent_looks"),
            container_format=src.driver,
            georeferenced=src.crs is not None,
            bounds_wgs84=bounds_wgs84,
            centroid_wgs84=centroid_wgs84,
            crs_is_projected=bool(src.crs is not None and src.crs.is_projected),
        )

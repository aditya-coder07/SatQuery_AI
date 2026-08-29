"""Shared synthetic raster fixtures.

Real Cartosat/RISAT products are not in the repo (see docs/verification.md),
so tests build synthetic rasters with known properties. Every fixture states
what it is meant to represent so a failure is interpretable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The repo root holds `training/` and `scripts/`, which are operational code
# rather than installed runtime packages. Putting the root on sys.path lets the
# tests import them without publishing a generically-named `scripts` package in
# the installed distribution, where it could collide with another project's.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

# NOTE: deliberately no shared module-level RNG. A single generator whose
# state advances across fixtures makes raster content depend on how many
# other tests ran first, which silently breaks the golden-trace comparisons
# when the suite is run in a different order or subset. Every fixture seeds
# its own generator instead.


def write_raster(
    path,
    array: np.ndarray,
    *,
    band_names: list[str] | None = None,
    crs: str = "EPSG:32643",
    gsd: float = 10.0,
    origin: tuple[float, float] = (500000.0, 2000000.0),
    tags: dict | None = None,
    dtype: str | None = None,
):
    """Write a (bands, h, w) array to a GeoTIFF with the given metadata."""
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    count, height, width = array.shape
    dtype = dtype or array.dtype.name
    transform = from_origin(origin[0], origin[1], gsd, gsd)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array.astype(dtype))
        if band_names:
            dst.descriptions = tuple(band_names)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def _structured_scene(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A scene with real spatial structure (edges), not pure noise.

    Co-registration needs actual features to lock onto; white noise has no
    stable correlation peak.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w), dtype="float64")
    # A few bright rectangles act as field/building boundaries.
    for _ in range(6):
        y0, x0 = rng.integers(0, h - h // 4), rng.integers(0, w - w // 4)
        dy, dx = rng.integers(h // 8, h // 4), rng.integers(w // 8, w // 4)
        base[y0 : y0 + dy, x0 : x0 + dx] += rng.uniform(0.4, 1.0)
    base += rng.normal(0, 0.02, size=(h, w))
    return base


@pytest.fixture
def msi_4band(tmp_path):
    """4-band VNIR optical, the assumed Cartosat-2S MX layout (no SWIR)."""
    h = w = 128
    scene = _structured_scene(h, w, seed=1)
    bands = np.stack(
        [
            (scene * 800 + 200),   # BLUE
            (scene * 900 + 250),   # GREEN
            (scene * 700 + 180),   # RED
            (scene * 2200 + 400),  # NIR - vegetation bright
        ]
    ).astype("uint16")
    path = tmp_path / "msi_4band.tif"
    return write_raster(
        path,
        bands,
        band_names=["BLUE", "GREEN", "RED", "NIR"],
        gsd=1.6,
        tags={"SATELLITE": "CARTOSAT-2S", "SENSOR": "MX"},
    )


@pytest.fixture
def msi_6band(tmp_path):
    """6-band optical including SWIR - enables MNDWI and NDBI."""
    h = w = 128
    scene = _structured_scene(h, w, seed=2)
    bands = np.stack(
        [
            scene * 800 + 200,
            scene * 900 + 250,
            scene * 700 + 180,
            scene * 2200 + 400,
            scene * 1100 + 300,  # SWIR1
            scene * 800 + 220,   # SWIR2
        ]
    ).astype("uint16")
    path = tmp_path / "msi_6band.tif"
    return write_raster(
        path,
        bands,
        band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        gsd=10.0,
        tags={"SATELLITE": "SENTINEL-2", "ACQUISITION_DATE": "2026-03-01"},
    )


@pytest.fixture
def sar_dualpol(tmp_path):
    """2-band C-band SAR, VV+VH, float32 - EOS-04/Sentinel-1 shaped."""
    h = w = 128
    scene = _structured_scene(h, w, seed=3)
    # Gamma-distributed speckle on top of structure: right-skewed, non-negative.
    speckle = np.random.default_rng(303).gamma(shape=2.0, scale=0.5, size=(h, w))
    vv = np.clip((scene + 0.2) * speckle, 1e-4, None)
    vh = np.clip((scene + 0.1) * speckle * 0.4, 1e-4, None)
    path = tmp_path / "sar_dualpol.tif"
    return write_raster(
        path,
        np.stack([vv, vh]).astype("float32"),
        band_names=["VV", "VH"],
        gsd=10.0,
        tags={"SATELLITE": "EOS-04", "SENSOR": "SAR", "POLARISATION": "VV VH"},
        dtype="float32",
    )


@pytest.fixture
def pan_1band(tmp_path):
    """Single-band panchromatic optical."""
    h = w = 128
    scene = _structured_scene(h, w, seed=4)
    path = tmp_path / "pan.tif"
    return write_raster(
        path,
        (scene * 3000 + 500).astype("uint16"),
        band_names=["PAN"],
        gsd=0.65,
        tags={"SATELLITE": "CARTOSAT-2S", "SENSOR": "PAN"},
    )


@pytest.fixture
def msi_6band_t2(tmp_path):
    """A second, later acquisition of the same area - bitemporal partner."""
    h = w = 128
    scene = _structured_scene(h, w, seed=2)
    # Simulate change: one region loses vegetation.
    scene[20:60, 20:60] *= 0.3
    bands = np.stack(
        [
            scene * 800 + 200,
            scene * 900 + 250,
            scene * 700 + 180,
            scene * 2200 + 400,
            scene * 1100 + 300,
            scene * 800 + 220,
        ]
    ).astype("uint16")
    path = tmp_path / "msi_6band_t2.tif"
    return write_raster(
        path,
        bands,
        band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        gsd=10.0,
        tags={"SATELLITE": "SENTINEL-2", "ACQUISITION_DATE": "2026-06-01"},
    )


@pytest.fixture
def tiny_raster(tmp_path):
    """Below the minimum dimension - must trigger a blocking check failure."""
    path = tmp_path / "tiny.tif"
    data = np.random.default_rng(404).random((1, 8, 8)).astype("float32")
    return write_raster(path, data, dtype="float32")


@pytest.fixture
def no_crs_raster(tmp_path):
    """Missing CRS - must trigger a blocking check failure."""
    path = tmp_path / "nocrs.tif"
    array = (np.random.default_rng(505).random((3, 64, 64)) * 1000).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint16"
    ) as dst:
        dst.write(array)
    return path


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Regenerate golden trace files instead of comparing against them.",
    )

"""Texture measures: GLCM statistics and speckle coefficient of variation.

Texture is the main SWIR-free discriminator for built-up surfaces, and CoV is
the standard SAR speckle diagnostic. Both are windowed, so they return a
raster of the same shape as the input rather than a single number.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops

GLCM_PROPERTIES = ("contrast", "homogeneity", "energy", "correlation")


def coefficient_of_variation(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """Windowed std/mean - the classic SAR speckle measure.

    For fully developed speckle CoV approaches a known constant set by the
    number of looks, so departures from it indicate real texture rather than
    noise. Smooth surfaces (water) give low CoV; urban areas give high CoV.
    """
    a = np.asarray(arr, dtype="float64")
    valid = np.isfinite(a)
    filled = np.where(valid, a, 0.0)

    # Windowed moments computed only over valid pixels.
    ones = valid.astype("float64")
    count = ndimage.uniform_filter(ones, size=window) * window * window
    total = ndimage.uniform_filter(filled, size=window) * window * window
    total_sq = ndimage.uniform_filter(filled**2, size=window) * window * window

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        mean_sq = np.where(count > 0, total_sq / count, np.nan)
        var = np.clip(mean_sq - mean**2, 0.0, None)
        cov = np.where(mean > 0, np.sqrt(var) / mean, np.nan)

    return np.where(valid, cov, np.nan)


def _quantise(arr: np.ndarray, levels: int) -> np.ndarray:
    """Rescale finite values to 0..levels-1 integers for GLCM."""
    a = np.asarray(arr, dtype="float64")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype="uint8")
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype="uint8")
    scaled = (a - lo) / (hi - lo) * (levels - 1)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.clip(np.rint(scaled), 0, levels - 1).astype("uint8")


def glcm_features(
    arr: np.ndarray,
    *,
    levels: int = 32,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> dict[str, float]:
    """Scene-level GLCM statistics, averaged over angles (rotation invariant).

    Returns one number per property. Windowed GLCM over a whole scene is
    prohibitively slow, and the verifier compares distributions rather than
    per-pixel values, so scene-level statistics are the right granularity.
    """
    q = _quantise(arr, levels)
    glcm = graycomatrix(
        q, distances=list(distances), angles=list(angles), levels=levels,
        symmetric=True, normed=True,
    )
    out: dict[str, float] = {}
    for prop in GLCM_PROPERTIES:
        out[prop] = float(np.mean(graycoprops(glcm, prop)))
    return out


def local_variance(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """Windowed variance - a cheap texture-roughness raster.

    Used by the SWIR-free built-up proxy, where a full GLCM per pixel would be
    far too slow.
    """
    a = np.asarray(arr, dtype="float64")
    valid = np.isfinite(a)
    filled = np.where(valid, a, 0.0)
    ones = valid.astype("float64")

    count = ndimage.uniform_filter(ones, size=window) * window * window
    total = ndimage.uniform_filter(filled, size=window) * window * window
    total_sq = ndimage.uniform_filter(filled**2, size=window) * window * window

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        mean_sq = np.where(count > 0, total_sq / count, np.nan)
        var = np.clip(mean_sq - mean**2, 0.0, None)

    return np.where(valid, var, np.nan)

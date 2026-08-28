"""Deterministic remote-sensing indices.

These are the physics half of the system: closed-form, no learned parameters,
no failure modes beyond bad input. Everything here is a pure array function so
it can be unit tested against hand-computed answers, which is what makes it
usable as an independent check on neural outputs (docs/01).

Convention: all functions take and return float arrays, propagate NaN for
undefined pixels (zero denominators, nodata), and never raise on bad data.
"""

from __future__ import annotations

import numpy as np

# Indices and the canonical bands they require. The planner reads this to
# decide whether an index is computable before it builds a plan.
INDEX_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "ndvi": ("RED", "NIR"),
    "ndwi": ("GREEN", "NIR"),
    "mndwi": ("GREEN", "SWIR1"),
    "ndbi": ("SWIR1", "NIR"),
}

# Physically meaningful range for every normalised difference index.
INDEX_RANGE = (-1.0, 1.0)


def _as_float(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype="float64")


def normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), with NaN where the denominator vanishes.

    This is the shared kernel of NDVI/NDWI/MNDWI/NDBI. Keeping it in one place
    means the zero-denominator and NaN-propagation behaviour is identical
    across every index, which matters when they are compared to each other.
    """
    a = _as_float(a)
    b = _as_float(b)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0, np.nan, (a - b) / denom)
    # Preserve NaN inputs rather than letting them become spurious values.
    out = np.where(np.isnan(a) | np.isnan(b), np.nan, out)
    return out


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index. High = vegetation."""
    return normalised_difference(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI. High = open water."""
    return normalised_difference(green, nir)


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified NDWI (Xu). Better water/built-up separation than NDWI.

    Requires SWIR1, so it is unavailable on 4-band VNIR products such as
    Cartosat-2S MX - see `swir_free_water_fallback`.
    """
    return normalised_difference(green, swir1)


def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Built-up Index. Requires SWIR1."""
    return normalised_difference(swir1, nir)


def sigma0_db(
    amplitude: np.ndarray, calibration_constant: float = 0.0, *, is_intensity: bool = False
) -> np.ndarray:
    """SAR backscatter coefficient in decibels.

    `amplitude` is the DN as delivered. For amplitude products sigma0 goes as
    the square of DN; for intensity products it is linear. The calibration
    constant is sensor-specific and additive in dB - left at 0 it yields
    relative backscatter, which is what the adaptive thresholds actually use
    (docs/03: keep all sigma0 thresholds adaptive, since the RISAT sensor and
    hence its absolute calibration is not yet confirmed).
    """
    a = _as_float(amplitude)
    power = a if is_intensity else a**2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(power > 0, 10.0 * np.log10(power), np.nan)
    return out + calibration_constant


def polarisation_ratio_db(cross: np.ndarray, co: np.ndarray) -> np.ndarray:
    """VH/VV (or HV/HH) ratio in dB.

    Volume scatterers (vegetation) depolarise and so raise the cross-pol
    return; smooth surfaces (water, roads) do not. This is the main
    SWIR-free discriminator available from SAR.
    """
    cross = _as_float(cross)
    co = _as_float(co)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((co > 0) & (cross > 0), cross / co, np.nan)
        return np.where(np.isnan(ratio), np.nan, 10.0 * np.log10(ratio))


def swir_free_water_fallback(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Water index for products with no SWIR band.

    MNDWI is preferred but needs SWIR1. NDWI uses only GREEN and NIR and is
    the documented fallback; it separates water less cleanly from built-up
    surfaces, which the caller must record as a confidence penalty.
    """
    return ndwi(green, nir)


def swir_free_builtup_proxy(
    red: np.ndarray,
    nir: np.ndarray,
    *,
    sigma0_vv: np.ndarray | None = None,
    texture: np.ndarray | None = None,
) -> np.ndarray:
    """Built-up likelihood without SWIR, in [0, 1].

    NDBI is impossible without SWIR1. The documented substitute (docs/02) is
    to combine evidence that built-up surfaces are: not vegetated (low NDVI),
    bright in SAR (corner reflection off buildings), and texturally rough.
    Each available term contributes equally; the caller records which terms
    were present so the trace explains the substitution.

    This is a *proxy*, not NDBI, and is deliberately named so.
    """
    terms: list[np.ndarray] = []

    # Low NDVI is necessary but not sufficient for built-up.
    veg = ndvi(red, nir)
    terms.append(np.clip((0.2 - veg) / 0.4, 0.0, 1.0))

    if sigma0_vv is not None:
        s = _as_float(sigma0_vv)
        finite = s[np.isfinite(s)]
        if finite.size:
            lo, hi = np.percentile(finite, [10, 90])
            if hi > lo:
                terms.append(np.clip((s - lo) / (hi - lo), 0.0, 1.0))

    if texture is not None:
        t = _as_float(texture)
        finite = t[np.isfinite(t)]
        if finite.size:
            lo, hi = np.percentile(finite, [10, 90])
            if hi > lo:
                terms.append(np.clip((t - lo) / (hi - lo), 0.0, 1.0))

    stacked = np.stack(terms)
    return np.nanmean(stacked, axis=0)


def index_stats(arr: np.ndarray) -> dict[str, float]:
    """Summary statistics used by the verifier and reported in the trace."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "valid_fraction": 0.0,
        }
    p10, p50, p90 = np.percentile(finite, [10, 50, 90])
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "valid_fraction": float(finite.size / arr.size),
    }

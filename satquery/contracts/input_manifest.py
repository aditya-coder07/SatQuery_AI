from enum import Enum
from typing import Literal
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

class IngestMode(str, Enum):
    OPERATIONAL = "operational"
    BENCHMARK = "benchmark"

class ImageMeta(BaseModel):
    role: Literal["single", "optical", "sar", "t1", "t2"]
    path: Path
    modality: Literal["OPTICAL", "MSI", "PAN", "SAR"]
    modality_evidence: dict
    crs: str
    gsd_m: float
    width: int
    height: int
    bands: list[str]
    band_presence: list[bool]
    dtype: str
    effective_bits: int
    acquisition_dt: datetime | None
    nodata_pct: float
    cloud_pct: float | None
    sensor_guess: str | None
    polarisations: list[str] | None
    look_count_est: float | None
    # GDAL driver of the container the pixels came out of, and whether
    # that container actually carried georeferencing. A PNG or JPEG
    # cannot carry a CRS at all, which is a different fact from a
    # GeoTIFF that was written without one - the first is a format
    # limitation to disclose, the second is a defective product to
    # reject. Defaulted so existing constructions stay valid.
    container_format: str | None = None
    georeferenced: bool = True
    # Footprint in WGS84 (west, south, east, north) and its centre
    # (latitude, longitude), transformed from the container's own CRS and
    # geotransform at ingest. `None` when the file carries no georeferencing,
    # when its transform is the identity, or when the projection could not be
    # transformed - the answer path says nothing about location rather than
    # guessing in any of those cases.
    #
    # Stored rather than derived on demand so the answer path can name a
    # location without reopening the raster. `report/evidence_pack` and the
    # tile endpoint still derive their own - they need the native-CRS
    # polygon and the Web Mercator envelope respectively, not this centre -
    # so this is an addition rather than a deduplication of those.
    bounds_wgs84: tuple[float, float, float, float] | None = None
    centroid_wgs84: tuple[float, float] | None = None
    # Whether the CRS measures in metres. `gsd_m` is exact for a projected
    # CRS and an approximation for a geographic one (see `_gsd_metres`, which
    # converts degrees at a flat 111320 m and is therefore wrong by the
    # cosine of the latitude in x). Ground extent is only reported for the
    # exact case; the approximation is fine for ordering-of-magnitude routing
    # decisions but not for a number shown to a reader as a distance.
    crs_is_projected: bool = False

    @property
    def ground_extent_m(self) -> tuple[float, float] | None:
        """Scene width and height on the ground, in metres.

        Only defined where `gsd_m` is a real measurement in metres.
        """
        if not self.georeferenced or not self.crs_is_projected:
            return None
        return (self.width * self.gsd_m, self.height * self.gsd_m)

class CheckResult(BaseModel):
    name: str
    status: Literal["PASS", "WARN", "FAIL"]
    value: float | str | bool | None = None
    threshold: float | str | None = None
    message: str

class CoregReport(BaseModel):
    method: Literal["phase_correlation", "gradient_phase_correlation", "mutual_information"]
    shift_px: tuple[float, float]
    shift_m: tuple[float, float]
    residual_px: float
    applied_correction: bool

class TilingReport(BaseModel):
    applied: bool
    level1_tiles: int | None = None
    retrieved_tiles: int | None = None
    retrieval_reason: str | None = None

class InputManifest(BaseModel):
    run_id: str
    ingest_mode: IngestMode
    benchmark: str | None = None
    images: list[ImageMeta]
    config: Literal["SINGLE", "CROSSMODAL_PAIR", "BITEMPORAL_PAIR"]
    checks: list[CheckResult]
    coreg: CoregReport | None = None
    tiling: TilingReport | None = None
    artifacts: dict[str, Path]
    blocking_failures: list[str]
    index_availability: dict[str, bool]

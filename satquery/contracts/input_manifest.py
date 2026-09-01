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

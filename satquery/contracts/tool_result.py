from typing import Literal
from pathlib import Path
from pydantic import BaseModel

class Artifact(BaseModel):
    key: str
    kind: Literal["geojson", "geotiff", "cog", "png", "json"]
    path: Path
    crs: str | None = None
    description: str | None = None

class ToolPayload(BaseModel):
    pass

class ToolResult(BaseModel):
    tool: str
    version: str
    payload: ToolPayload
    artifacts: list[Artifact]
    confidence: float
    confidence_method: Literal["logprob", "softmax_temp_scaled", "threshold_rule", "deterministic"]
    model_card: str
    runtime_ms: int
    warnings: list[str]

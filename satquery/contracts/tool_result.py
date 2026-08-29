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
    # What the `confidence` number actually IS, so a consumer can tell whether
    # it is a probability of correctness or something else entirely. Only
    # `logprob` and `mean_asserted_probability` are probabilities at all, and
    # neither is P(correct) - see CALIBRATABLE_CONFIDENCE_METHODS in
    # satquery/controller/calibration.py for what each one measures.
    #
    # `softmax_temp_scaled` was removed: it was claimed by two tools that
    # never temperature-scaled anything, and keeping a name in the contract
    # that nothing computes invites the same mistake again.
    confidence_method: Literal[
        "logprob",
        "sharpness",
        "mean_asserted_probability",
        "threshold_rule",
        "deterministic",
    ]
    model_card: str
    runtime_ms: int
    warnings: list[str]

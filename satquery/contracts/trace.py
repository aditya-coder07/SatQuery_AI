from typing import Literal
from pydantic import BaseModel
from .plan import TaskID, RationaleTag

class IngestTrace(BaseModel):
    mode: str
    config: str
    images: list[dict]
    index_availability: dict[str, bool]
    checks: list[dict]
    tiling: dict

class ClassifierTrace(BaseModel):
    name: str
    top1: float
    margin: float

class RoutingTrace(BaseModel):
    legal_tasks: list[TaskID]
    selected_task: TaskID
    classifier: ClassifierTrace
    llm_tiebreak_invoked: bool
    capability_matrix_version: str

class StepExecutionTrace(BaseModel):
    step: str
    tool: str
    version: str
    params: dict
    rationale_tag: RationaleTag
    outputs: dict
    confidence: float
    confidence_method: str
    runtime_ms: int

class EntailmentGateTrace(BaseModel):
    sentences: int
    retained: int
    flagged: int

class VerificationTrace(BaseModel):
    physics_agreement: dict[str, float]
    built_up_path: str
    complementarity: dict
    conflicts: list[str]
    entailment_gate: EntailmentGateTrace

class ConfidenceComponentsTrace(BaseModel):
    model: float
    agreement: float
    input_quality: float

class ConfidenceCalibrationTrace(BaseModel):
    method: str
    T: float
    ece_after: float

class ConfidenceTrace(BaseModel):
    final: float
    band: Literal["HIGH", "MEDIUM", "LOW"]
    components: ConfidenceComponentsTrace
    calibration: ConfidenceCalibrationTrace

class Trace(BaseModel):
    run_id: str
    timestamp_utc: str
    code_version: str
    query: str
    ingest: IngestTrace
    routing: RoutingTrace
    execution: list[StepExecutionTrace]
    verification: VerificationTrace
    confidence: ConfidenceTrace
    answer: str
    artifacts: list[str]
    abstained: bool
    abstain_reason: str | None = None
    weights_hashes: dict[str, str]

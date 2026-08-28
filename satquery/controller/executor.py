from datetime import datetime, timezone
from satquery.contracts.plan import Plan
from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.trace import (
    Trace, IngestTrace, RoutingTrace, ClassifierTrace,
    StepExecutionTrace, VerificationTrace, EntailmentGateTrace,
    ConfidenceTrace, ConfidenceComponentsTrace, ConfidenceCalibrationTrace
)
from satquery.tools.stubs import REGISTRY

class Executor:
    def execute(self, plan: Plan, manifest: InputManifest, query: str) -> Trace:
        execution_traces = []
        artifacts = []
        final_answer = ""
        
        for step in plan.steps:
            tool = REGISTRY[step.tool]
            result = tool.run(manifest, step.params)
            
            if "answer" in result.payload.data:
                final_answer = result.payload.data["answer"]
                
            execution_traces.append(
                StepExecutionTrace(
                    step=step.step_id,
                    tool=step.tool,
                    version=result.version,
                    params=step.params,
                    rationale_tag=step.rationale_tag,
                    outputs=result.payload.data,
                    confidence=result.confidence,
                    confidence_method=result.confidence_method,
                    runtime_ms=result.runtime_ms
                )
            )
            for art in result.artifacts:
                artifacts.append(art.key)
                
        # Mock IngestTrace from manifest
        ingest = IngestTrace(
            mode=manifest.ingest_mode.value,
            config=manifest.config,
            images=[{"path": str(img.path)} for img in manifest.images],
            index_availability=manifest.index_availability,
            checks=[{"name": c.name, "status": c.status} for c in manifest.checks],
            tiling={"applied": False}
        )
        
        # Mock RoutingTrace
        routing = RoutingTrace(
            legal_tasks=plan.legal_tasks,
            selected_task=plan.tasks[0],
            classifier=ClassifierTrace(name="stub", top1=0.99, margin=0.5),
            llm_tiebreak_invoked=False,
            capability_matrix_version=plan.matrix_version
        )
        
        # Mock VerificationTrace
        verification = VerificationTrace(
            physics_agreement={},
            built_up_path="",
            complementarity={},
            conflicts=[],
            entailment_gate=EntailmentGateTrace(sentences=1, retained=1, flagged=0)
        )
        
        # Mock ConfidenceTrace
        confidence = ConfidenceTrace(
            final=0.9,
            band="HIGH",
            components=ConfidenceComponentsTrace(model=0.9, agreement=0.9, input_quality=0.9),
            calibration=ConfidenceCalibrationTrace(method="stub", T=1.0, ece_after=0.05)
        )
        
        return Trace(
            run_id=plan.run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            code_version="0.1.0",
            query=query,
            ingest=ingest,
            routing=routing,
            execution=execution_traces,
            verification=verification,
            confidence=confidence,
            answer=final_answer,
            artifacts=artifacts,
            abstained=False,
            weights_hashes={}
        )

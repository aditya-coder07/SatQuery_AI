"""Executor: runs a validated plan and builds a real trace.

Phase 0 filled the trace with placeholders. This version derives every field
from something that actually happened: ingest from the manifest, routing from
the classifier's own probabilities, verification from the deterministic index
engine, and confidence from the three-component combiner.

Where a Phase 1 component genuinely does not exist yet (the NLI entailment
gate is task 3.5), the trace says so explicitly rather than reporting a
fabricated number.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.plan import Plan
from satquery.contracts.trace import (
    ClassifierTrace,
    ConfidenceTrace,
    EntailmentGateTrace,
    IngestTrace,
    RoutingTrace,
    StepExecutionTrace,
    Trace,
    VerificationTrace,
)
from satquery.controller.confidence import compute_confidence
from satquery.controller.intent import CLASSIFIER_NAME, IntentPrediction
from satquery.synth.narrative import synthesise_answer
from satquery.tools.stubs import REGISTRY

CODE_VERSION = "0.2.0-phase1"

# Answer-bearing payload keys, in priority order.
_ANSWER_KEYS = ("answer", "caption", "description", "summary")


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so the trace always serialises to valid JSON.

    Index statistics over an all-nodata band legitimately produce NaN. NaN is
    not valid JSON and would break the SSE stream, so it becomes None here -
    a missing value, which is what it means.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def physics_agreement_from_indices(payload: dict) -> tuple[dict[str, float], list[str]]:
    """Turn index-engine output into per-claim agreement scores and conflicts.

    Phase 1 scope: agreement is high when an index produced a confident,
    genuinely bimodal split (the threshold is trustworthy) and low when the
    engine had to fall back to a fixed prior. Task 2.9 extends this to
    per-claim entailment against the generated text.
    """
    agreements: dict[str, float] = {}
    conflicts: list[str] = []

    for report in payload.get("thresholds", []):
        name = report["index"]
        if report["method"] == "fixed_prior":
            agreements[name] = 0.4
            conflicts.append(
                f"{name}: no bimodal split found, threshold is a fixed prior "
                "rather than data-derived"
            )
        elif report.get("bimodal"):
            agreements[name] = 1.0
        else:
            agreements[name] = 0.7

    return agreements, conflicts


def built_up_path(payload: dict) -> str:
    """Which built-up derivation was used - NDBI or the SWIR-free proxy."""
    indices = payload.get("indices", {})
    if "ndbi" in indices:
        return "ndbi"
    if "builtup_proxy" in indices:
        return "swir_free_proxy"
    return "not_computed"


class Executor:
    """Runs plan steps and assembles the trace."""

    def execute(
        self,
        plan: Plan,
        manifest: InputManifest,
        query: str,
        prediction: IntentPrediction | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> Trace:
        """Run the plan. `on_event(name, data)` fires as each stage completes,
        which is what lets the API stream the trace live rather than posting it
        all at the end."""

        def emit(name: str, data: dict) -> None:
            if on_event is not None:
                on_event(name, _json_safe(data))

        ingest_trace = IngestTrace(
            mode=manifest.ingest_mode.value,
            config=manifest.config,
            images=[
                {
                    "role": img.role,
                    "path": str(img.path),
                    "modality": img.modality,
                    "modality_reason": img.modality_evidence.get("reason"),
                    "crs": img.crs,
                    "gsd_m": img.gsd_m,
                    "bands": img.bands,
                    "nodata_pct": img.nodata_pct,
                    "sensor_guess": img.sensor_guess,
                    "polarisations": img.polarisations,
                }
                for img in manifest.images
            ],
            index_availability=manifest.index_availability,
            checks=[
                {"name": c.name, "status": c.status, "message": c.message}
                for c in manifest.checks
            ],
            tiling=manifest.tiling.model_dump() if manifest.tiling else {"applied": False},
        )

        emit("ingest", ingest_trace.model_dump())

        execution_traces: list[StepExecutionTrace] = []
        artifacts: list[str] = []
        final_answer = ""
        model_confidence = 1.0
        index_payload: dict = {}
        warnings: list[str] = []

        if prediction is not None:
            classifier = ClassifierTrace(
                name=CLASSIFIER_NAME,
                top1=prediction.top1,
                margin=prediction.margin,
            )
        else:
            # Abstention path: routing was forced by input checks, not by the
            # classifier, so report that rather than inventing a score.
            classifier = ClassifierTrace(name="not_invoked", top1=0.0, margin=0.0)

        routing = RoutingTrace(
            legal_tasks=plan.legal_tasks,
            selected_task=plan.tasks[0],
            classifier=classifier,
            llm_tiebreak_invoked=False,  # Tier-2 tiebreak is Phase 3
            capability_matrix_version=plan.matrix_version,
        )

        emit("routing", routing.model_dump())

        for step in plan.steps:
            tool = REGISTRY[step.tool]
            # The user's question is injected at execution time under a
            # reserved key rather than being placed in the plan. The
            # capability matrix governs *tunable parameters*; the query is
            # input data. Keeping it out of step.params means the plan that
            # gets validated for legality stays exactly what the matrix
            # permits, and the query is already recorded verbatim in the
            # trace's own `query` field.
            runtime_params = {**step.params, "_query": query}
            try:
                result = tool.run(manifest, runtime_params)
            except Exception as exc:  # noqa: BLE001 - degradation, not a crash
                if step.on_failure == "abort":
                    raise
                warnings.append(f"{step.tool} failed ({exc}); continuing degraded")
                continue

            data = result.payload.data
            if step.tool == "index_engine_v1":
                index_payload = data
            else:
                # Only learned tools contribute to the model confidence
                # component; the index engine is deterministic by construction.
                model_confidence = min(model_confidence, result.confidence)

            for key in _ANSWER_KEYS:
                if key in data:
                    final_answer = str(data[key])
                    break

            execution_traces.append(
                StepExecutionTrace(
                    step=step.step_id,
                    tool=step.tool,
                    version=result.version,
                    params=step.params,  # plan params only; _query is not one
                    rationale_tag=step.rationale_tag,
                    outputs=_json_safe(data),
                    confidence=result.confidence,
                    confidence_method=result.confidence_method,
                    runtime_ms=result.runtime_ms,
                )
            )
            warnings.extend(result.warnings)
            artifacts.extend(a.key for a in result.artifacts)
            emit("step", execution_traces[-1].model_dump())

        agreements, conflicts = physics_agreement_from_indices(index_payload)
        for sub in index_payload.get("substitutions", []):
            conflicts.append(f"substitution: {sub}")

        verification = VerificationTrace(
            physics_agreement=agreements,
            built_up_path=built_up_path(index_payload),
            complementarity={},  # optical-SAR complementarity is task 2.3
            conflicts=conflicts,
            entailment_gate=EntailmentGateTrace(
                # The NLI gate is task 3.5. Reporting zeros makes it obvious
                # that no sentences were gated, rather than implying they
                # passed a check that does not exist yet.
                sentences=0,
                retained=0,
                flagged=0,
            ),
        )

        if not final_answer:
            # Tools that return structure rather than prose (land cover,
            # grounding) still owe the user a sentence. It is synthesised
            # deterministically from the numbers already computed, so it
            # cannot assert anything the index engine did not measure.
            final_answer = synthesise_answer(
                plan.tasks[0],
                [t.outputs for t in execution_traces],
                index_payload,
            )

        emit("verification", verification.model_dump())

        confidence: ConfidenceTrace = compute_confidence(
            model_confidence=model_confidence,
            manifest=manifest,
            agreements=agreements,
        )

        emit("confidence", confidence.model_dump())

        abstained = plan.tasks[0] == "CLARIFY_OR_ABSTAIN"
        abstain_reason = None
        if abstained:
            if manifest.blocking_failures:
                failed = ", ".join(manifest.blocking_failures)
                abstain_reason = (
                    f"input validation failed ({failed}) - resolve the input "
                    "before a reliable answer is possible"
                )
            else:
                abstain_reason = (
                    "the query could not be mapped to a supported task with "
                    "sufficient confidence - please rephrase or be more specific"
                )
            final_answer = final_answer or abstain_reason

        return Trace(
            run_id=plan.run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            code_version=CODE_VERSION,
            query=query,
            ingest=ingest_trace,
            routing=routing,
            execution=execution_traces,
            verification=verification,
            confidence=confidence,
            answer=final_answer,
            artifacts=artifacts,
            abstained=abstained,
            abstain_reason=abstain_reason,
            weights_hashes={},  # populated when real checkpoints load (1.7/1.10)
        )

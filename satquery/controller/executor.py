"""Executor: runs a validated plan and builds a real trace.

Phase 0 filled the trace with placeholders. This version derives every field
from something that actually happened: ingest from the manifest, routing from
the classifier's own probabilities, verification from the deterministic index
engine, and confidence from the three-component combiner.

Where a component genuinely does not exist yet the trace says so explicitly
rather than reporting a fabricated number - `complementarity` is still `{}`
for that reason.
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
    FlaggedSentenceTrace,
    IngestTrace,
    RoutingTrace,
    StepExecutionTrace,
    Trace,
    VerificationTrace,
)
from satquery.controller.abstention import AbstentionPolicy, decide
from satquery.controller.calibration import CALIBRATABLE_CONFIDENCE_METHODS
from satquery.controller.confidence import compute_confidence
from satquery.controller.intent import CLASSIFIER_NAME, IntentPrediction
from satquery.synth.narrative import synthesise_answer
from satquery.verify.entailment import run_gate
from satquery.verify.verifier import verify as verify_claims
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

    def __init__(
        self,
        verifier_enabled: bool = True,
        abstention_policy: AbstentionPolicy | None = None,
    ):
        # The off arm of the verifier ablation (task 3.7). Disabling it skips
        # the entailment gate entirely rather than running it and ignoring the
        # result, so the ablation measures the gate's real cost as well as its
        # effect.
        self.verifier_enabled = verifier_enabled
        # Loaded once per executor rather than per query: the thresholds are
        # configuration, and re-reading the file mid-run would let a demo
        # change behaviour between two queries in the same session.
        self.abstention_policy = abstention_policy or AbstentionPolicy.load()

    def execute(
        self,
        plan: Plan,
        manifest: InputManifest,
        query: str,
        prediction: IntentPrediction | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        config_excluded: str | None = None,
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
        # The `confidence_method` of the tool that set the running minimum.
        # Calibration is only defined on a probability of correctness, so the
        # method - not merely "a learned tool ran" - decides whether the
        # fitted transform may be applied. See CALIBRATABLE_CONFIDENCE_METHODS.
        confidence_method: str | None = None
        index_payload: dict = {}
        warnings: list[str] = []
        # Set when a step with on_failure="abort" raised. The run stops and
        # becomes a named abstention rather than a traceback (task 3.13).
        tool_failure: str | None = None
        # Set when no tool could answer because the profile shed them all
        # (task 3.10). Distinct from tool_failure: nothing broke.
        profile_degraded: str | None = None

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
            config_excluded_task=config_excluded,
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
                    # Task 3.13: an aborting tool used to re-raise, which put
                    # a Python traceback in front of whoever called the API.
                    # A crash is not a graceful degradation, and "zero stack
                    # traces surfaced to the user" is the requirement. The
                    # run stops here and becomes an abstention that names the
                    # tool and the error; the traceback belongs in the logs,
                    # not the answer.
                    tool_failure = (
                        f"{step.tool} failed and the plan cannot continue "
                        f"without it ({type(exc).__name__}: {exc})"
                    )
                    warnings.append(tool_failure)
                    break
                warnings.append(f"{step.tool} failed ({exc}); continuing degraded")
                continue

            data = result.payload.data
            if step.tool == "index_engine_v1":
                index_payload = data
            else:
                # Only learned tools contribute to the model confidence
                # component; the index engine is deterministic by construction.
                if result.confidence <= model_confidence:
                    model_confidence = result.confidence
                    confidence_method = result.confidence_method

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

        if not final_answer:
            # Tools that return structure rather than prose (land cover,
            # grounding) still owe the user a sentence. It is synthesised
            # deterministically from the numbers already computed, so it
            # cannot assert anything the index engine did not measure.
            final_answer = synthesise_answer(
                plan.tasks[0],
                [t.outputs for t in execution_traces],
                index_payload,
                artifacts=artifacts,
            )

        # `model_confidence == 0.0` means a learned tool ran and explicitly
        # reported that it could not measure anything - change_vqa_v1's
        # documented "cannot measure" path. That is a deliberate decline, not
        # a fault, so it is left to the low-confidence rule, which produces
        # the right message. Overriding it with "the tool failed, retrying is
        # reasonable" would send the user to repeat something that already
        # worked as designed.
        tool_declined = model_confidence <= 0.0

        if not final_answer.strip() and not tool_failure and not tool_declined:
            # Nothing produced a sentence and nothing raised. This is the
            # degraded-profile case (task 3.10): the learned tool that would
            # have answered was shed by the VRAM budget. An empty answer is a
            # silent failure, so it becomes a named one.
            learned_ran = any(
                t.tool != "index_engine_v1" for t in execution_traces
            )
            if learned_ran:
                # A learned tool ran and returned nothing usable. That is a
                # tool problem, not a profile problem, and saying "run the
                # full profile" would send the user somewhere that will do
                # exactly the same thing.
                tool_failure = (
                    f"the tool for {plan.tasks[0]} ran but produced no "
                    f"answer for these inputs"
                )
            else:
                profile_degraded = (
                    f"{plan.tasks[0]} needs a learned tool, and none was "
                    f"available under this profile's resource budget; the "
                    f"task has no deterministic fallback"
                )

        agreements, conflicts = physics_agreement_from_indices(index_payload)
        for sub in index_payload.get("substitutions", []):
            conflicts.append(f"substitution: {sub}")

        # Task 2.9: check what the answer actually *claims* against measured
        # indices. Threshold quality says whether the instrument is
        # trustworthy; this says whether the statement is true.
        claim_report = verify_claims(final_answer, index_payload)
        agreements.update(claim_report["agreements"])
        conflicts.extend(claim_report["conflicts"])
        if claim_report["built_up_path"]:
            built_up = claim_report["built_up_path"]
        else:
            built_up = built_up_path(index_payload)

        # Task 3.5: gate every sentence of the answer against the payload.
        # This runs AFTER verify_claims because it reuses those verdicts, and
        # it can rewrite `final_answer` - a sentence that contradicts the
        # measured indices is removed rather than shown. The original text is
        # kept verbatim in the trace, so nothing is hidden.
        gate = run_gate(final_answer, index_payload, enabled=self.verifier_enabled)
        final_answer = gate.answer
        conflicts.extend(
            f"entailment gate removed: {v.reason}"
            for v in gate.verdicts
            if v.status == "flagged"
        )

        verification = VerificationTrace(
            physics_agreement=agreements,
            built_up_path=built_up,
            complementarity={},  # optical-SAR complementarity is task 2.3
            conflicts=conflicts,
            entailment_gate=EntailmentGateTrace(
                sentences=gate.sentences,
                retained=gate.retained,
                flagged=gate.flagged,
                unverifiable=gate.unverifiable,
                backend=gate.backend,
                action=gate.action,
                flagged_detail=[
                    FlaggedSentenceTrace(
                        sentence=v.sentence, reason=v.reason,
                        backend=v.backend, score=v.score,
                    )
                    for v in gate.verdicts
                    if v.status == "flagged"
                ],
            ),
        )

        emit("verification", verification.model_dump())

        confidence: ConfidenceTrace = compute_confidence(
            model_confidence=model_confidence,
            manifest=manifest,
            agreements=agreements,
            head=(
                plan.tasks[0]
                if confidence_method in CALIBRATABLE_CONFIDENCE_METHODS
                else None
            ),
        )

        emit("confidence", confidence.model_dump())

        # Task 3.6: abstention is a policy over the whole run, not just a
        # routing outcome. A confidently-routed plan whose answer the physics
        # contradicts must also be able to decline.
        decision = decide(
            policy=self.abstention_policy,
            routed_to_abstain=plan.tasks[0] == "CLARIFY_OR_ABSTAIN",
            blocking_failures=list(manifest.blocking_failures),
            final_confidence=confidence.final,
            components=confidence.components.model_dump(),
            failing_checks=[
                c.name for c in manifest.checks if c.status in ("FAIL", "WARN")
            ],
            conflicts=conflicts,
            gate_sentences=gate.sentences,
            gate_flagged=gate.flagged,
            tool_failure=tool_failure,
            profile_degraded=profile_degraded,
        )
        abstained = decision.abstained
        abstain_reason = decision.reason
        if abstained and decision.resolving_input:
            # The resolving input is part of the message the user sees, not
            # only trace metadata: an abstention nobody can act on is just a
            # refusal.
            abstain_reason = f"{decision.reason} - {decision.resolving_input}"
        if abstained:
            final_answer = abstain_reason or final_answer
        elif config_excluded:
            # The user asked for something these inputs cannot support. The
            # plan is legal and the answer is real, but it is not the answer
            # that was asked for, and saying so is the difference between a
            # helpful fallback and a silent substitution. This is a prefix
            # rather than an abstention because the fallback answer is often
            # still useful - abstaining on every mismatch would trade a large
            # amount of coverage for a small amount of precision.
            final_answer = (
                f"Note: this input configuration ({manifest.config}) cannot "
                f"support {config_excluded}, which is what the question asks "
                f"for. Answering with {plan.tasks[0]} instead. "
            ) + final_answer

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
            abstain_trigger=decision.trigger,
            abstain_resolving_input=decision.resolving_input,
            abstain_limiting_component=decision.limiting_component,
            weights_hashes={},  # populated when real checkpoints load (1.7/1.10)
        )

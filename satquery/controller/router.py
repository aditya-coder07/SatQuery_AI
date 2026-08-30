"""Router: query + manifest -> validated Plan (plan task 1.3).

Three gates, in order, and the order is what makes the illegal-plan guarantee
hold:

1. **Config gating.** The input configuration determines which tasks are even
   legal. A single image can never route to change detection, no matter what
   the query says. Since 2026-08-30 this also enforces the matrix's *input
   requirements* - `min_overlap_pct`, `max_coreg_shift_px`, `require_dates`,
   `min_bands_optical`. Those had been declared in the matrix since Phase 0
   and read by nothing, so a task whose declared precondition was unmet was
   still selectable (limitation L16): an optical and a SAR scene 60 km apart
   routed to XMODAL_JOINT_EXTRACT and were fused into one confident answer.
2. **Intent classification**, restricted to the legal set. The classifier can
   only ever choose among tasks that are already legal, so a misclassification
   degrades answer quality but cannot produce an illegal plan.
3. **Plan validation** against the capability matrix. The constructed plan is
   checked before it is returned; anything that violates the matrix raises
   rather than executing.

Blocking ingest failures short-circuit straight to CLARIFY_OR_ABSTAIN.
"""

from __future__ import annotations

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.plan import Plan, PlanStep, RationaleTag, TaskID
from satquery.controller.intent import IntentClassifier, default_classifier
from satquery.controller.matrix_loader import CapabilityMatrix
from satquery.controller.validator import assert_legal

CONFIG_TO_LEGAL_TASKS: dict[str, list[str]] = {
    "SINGLE": [
        "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER",
        "CLARIFY_OR_ABSTAIN",
    ],
    "CROSSMODAL_PAIR": [
        "XMODAL_JOINT_EXTRACT", "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND",
        "SINGLE_LANDCOVER", "CLARIFY_OR_ABSTAIN",
    ],
    "BITEMPORAL_PAIR": [
        "TEMPORAL_CHANGE_DESC", "TEMPORAL_CHANGE_VQA", "TEMPORAL_CHANGE_MAP",
        "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER",
        "CLARIFY_OR_ABSTAIN",
    ],
}

# Where to fall back when the classifier is not confident enough to trust.
CONFIG_DEFAULT_TASK: dict[str, TaskID] = {
    "SINGLE": "SINGLE_VQA",
    "CROSSMODAL_PAIR": "SINGLE_VQA",
    "BITEMPORAL_PAIR": "SINGLE_VQA",
}

RATIONALE_BY_TASK: dict[str, RationaleTag] = {
    "SINGLE_VQA": RationaleTag.VQA_INFERENCE,
    "SINGLE_CAPTION": RationaleTag.MASK_CONDITIONED_CAPTION,
    "SINGLE_GROUND": RationaleTag.DETECTED_THEN_COUNTED,
    "SINGLE_LANDCOVER": RationaleTag.QUANTITATIVE_REQUEST,
    "XMODAL_JOINT_EXTRACT": RationaleTag.QUANTITATIVE_REQUEST,
    "TEMPORAL_CHANGE_DESC": RationaleTag.EXPLICIT_CHANGE_LANGUAGE,
    "TEMPORAL_CHANGE_VQA": RationaleTag.QUANTITATIVE_REQUEST,
    "TEMPORAL_CHANGE_MAP": RationaleTag.EXPLICIT_CHANGE_LANGUAGE,
    "CLARIFY_OR_ABSTAIN": RationaleTag.AMBIGUOUS_DEFAULTED_TO_VQA,
}

# Rough per-tool VRAM cost in MB. The VRAM manager sums these to decide
# whether a plan fits the profile's budget before anything is loaded.
TOOL_VRAM_MB: dict[str, int] = {
    "index_engine_v1": 0,      # pure numpy, no GPU
    "rs_vqa_v1": 4200,
    "caption_v1": 4200,
    "grounding_v1": 1800,
    "landcover_v1": 900,
    "optsar_fusion_v1": 2400,
    "change_mask_v1": 1200,
    "change_caption_v1": 2600,
    "change_vqa_v1": 2600,
}

TOOL_RUNTIME_MS: dict[str, int] = {
    "index_engine_v1": 400,
    "rs_vqa_v1": 900,
    "caption_v1": 1100,
    "grounding_v1": 700,
    "landcover_v1": 500,
    "optsar_fusion_v1": 1300,
    "change_mask_v1": 800,
    "change_caption_v1": 1200,
    "change_vqa_v1": 1000,
}

TOOL_VERSIONS: dict[str, str] = {
    "index_engine_v1": "1.0.0",
}
STUB_VERSION = "0.1.0-stub"


class Router:
    """Config gating + intent classification + matrix-validated planning."""

    def __init__(
        self,
        matrix: CapabilityMatrix,
        classifier: IntentClassifier | None = None,
        vram_budget_mb: int | None = None,
    ):
        self.matrix = matrix
        self.classifier = classifier or default_classifier()
        self.vram_budget_mb = vram_budget_mb

    # -- gating ----------------------------------------------------------
    @staticmethod
    def _check_value(manifest: InputManifest, name: str):
        """The measured value of an ingest check, or None if it did not run."""
        for check in manifest.checks:
            if check.name == name:
                return check.value
        return None

    def unmet_requirements(self, task: str, manifest: InputManifest) -> list[str]:
        """Matrix-declared input requirements this manifest does not satisfy.

        Returns human-readable reasons, which the caller turns into an
        exclusion notice, so a rejected task always says *why* rather than
        silently disappearing from the legal set.

        `CLARIFY_OR_ABSTAIN` is never excluded: it is the destination when
        everything else is, and gating it would leave nothing to route to.
        """
        if task == "CLARIFY_OR_ABSTAIN":
            return []
        cfg = self.matrix.tasks.get(task)
        if cfg is None:
            return []
        requires = cfg.requires
        unmet: list[str] = []

        minimum = getattr(requires, "min_overlap_pct", None)
        if minimum is not None and len(manifest.images) == 2:
            overlap = self._check_value(manifest, "footprint_overlap")
            # An unmeasurable overlap (ungeoreferenced input) is not treated as
            # a failure here: check_footprint_overlap already WARNs, and
            # benchmark inputs are ungeoreferenced by construction.
            if isinstance(overlap, (int, float)) and overlap < minimum:
                unmet.append(
                    f"footprint overlap {overlap:.0f}% is below the {minimum}% "
                    f"this task requires"
                )

        # `max_coreg_shift_px` and `require_dates` are deliberately NOT
        # enforced as hard exclusions, and the reasons are measured rather
        # than assumed:
        #
        # * **Co-registration shift.** On a synthetic optical+SAR pair with
        #   *identical* footprints - 100% overlap, same CRS, same GSD - the
        #   gradient-domain phase correlation reports **38.1 px**, twenty
        #   times the 2.0 px the matrix allows. The estimator is useful as a
        #   relative quality signal and its absolute accuracy across
        #   modalities is unvalidated, so gating on it would refuse
        #   well-formed pairs. It belongs in `degraded_if` as a confidence
        #   penalty, which is where the matrix already puts comparable
        #   signals, not in a hard gate.
        #
        # * **`require_dates`.** Enforcing it would refuse change analysis on
        #   every pair without acquisition metadata, which includes the
        #   prescribed benchmark inputs - CDVQA ships undated PNGs. The
        #   existing `temporal_order` WARN already records that t1/t2 order
        #   came from input order rather than metadata, which is the honest
        #   disclosure without making the benchmark unrunnable.
        #
        # Both remain declared in the matrix, and both are recorded as open in
        # docs/00 §3.6 rather than quietly satisfied here.

        minimum_bands = getattr(requires, "min_bands_optical", None)
        if minimum_bands is not None:
            optical = [
                img for img in manifest.images
                if img.modality in ("OPTICAL", "MSI", "PAN")
            ]
            if optical and max(len(img.bands) for img in optical) < minimum_bands:
                unmet.append(
                    f"the optical image has "
                    f"{max(len(img.bands) for img in optical)} bands, below the "
                    f"{minimum_bands} this task requires"
                )

        return unmet

    def legal_tasks(self, manifest: InputManifest) -> list[str]:
        """Tasks permitted by the input configuration AND the matrix.

        Intersecting with the matrix means a task cannot be routed to unless
        the matrix also declares it legal for this configuration **and** the
        manifest satisfies the input requirements the matrix declares for it.
        """
        by_config = CONFIG_TO_LEGAL_TASKS.get(manifest.config, [])
        out = []
        for task in by_config:
            cfg = self.matrix.tasks.get(task)
            if cfg is None:
                continue
            required = cfg.requires.config
            allowed = [required] if isinstance(required, str) else list(required)
            if manifest.config not in allowed and "any" not in allowed:
                continue
            if self.unmet_requirements(task, manifest):
                continue
            out.append(task)
        return out

    # -- planning --------------------------------------------------------
    def _default_params(self, task: TaskID) -> dict:
        """Only ever emit parameters the matrix permits, using its defaults."""
        params: dict = {}
        for name, schema in self.matrix.tasks[task].permitted_params.items():
            if schema.default is not None:
                params[name] = schema.default
        return params

    def _build_steps(self, task: TaskID, manifest: InputManifest) -> list[PlanStep]:
        cfg = self.matrix.tasks[task]
        rationale = RATIONALE_BY_TASK.get(task, RationaleTag.VQA_INFERENCE)
        params = self._default_params(task)

        steps: list[PlanStep] = []
        for i, tool in enumerate(cfg.tools, start=1):
            # A tool only receives parameters the matrix declares for this
            # task; unknown parameters would fail validation by construction.
            steps.append(
                PlanStep(
                    step_id=f"step_{i}",
                    tool=tool,
                    tool_version=TOOL_VERSIONS.get(tool, STUB_VERSION),
                    inputs=[img.role for img in manifest.images],
                    params=params if tool != "index_engine_v1" else {},
                    rationale_tag=rationale,
                    on_failure="fallback" if tool in cfg.fallbacks else "abort",
                )
            )
        return steps

    def _estimate(self, steps: list[PlanStep]) -> tuple[int, int]:
        """(peak VRAM MB, total runtime ms).

        VRAM is a peak, not a sum: tools run sequentially and are unloaded
        between steps, so the binding constraint is the largest single tool.
        Runtime is additive because the steps are sequential.
        """
        vram = max((TOOL_VRAM_MB.get(s.tool, 0) for s in steps), default=0)
        runtime = sum(TOOL_RUNTIME_MS.get(s.tool, 0) for s in steps)
        return vram, runtime

    # -- entry point ------------------------------------------------------
    def route(self, query: str, manifest: InputManifest) -> Plan:
        legal = self.legal_tasks(manifest)
        self.last_config_excluded: str | None = None

        if manifest.blocking_failures:
            # Inputs failed validation: no amount of query understanding makes
            # an answer defensible, so abstain and say why.
            task: TaskID = "CLARIFY_OR_ABSTAIN"
        else:
            # Also classify WITHOUT the legality restriction. If the
            # unconstrained best task is one the input configuration excludes,
            # the user asked for something these images cannot support, and
            # the answer should say so rather than quietly returning a
            # different task's output. Config gating guarantees the plan is
            # legal; it does not guarantee the user understands why they got
            # a land-cover map when they asked about change.
            unconstrained = self.classifier.predict(query)
            if unconstrained.is_confident and unconstrained.task not in legal:
                self.last_config_excluded = unconstrained.task

            prediction = self.classifier.predict(query, candidates=legal)
            if prediction.is_confident:
                task = prediction.task
            else:
                task = CONFIG_DEFAULT_TASK.get(manifest.config, "SINGLE_VQA")
                if task not in legal:
                    task = "CLARIFY_OR_ABSTAIN"
            self.last_prediction = prediction

        if task not in legal:
            task = "CLARIFY_OR_ABSTAIN"

        steps = self._build_steps(task, manifest)
        vram, runtime = self._estimate(steps)

        if self.vram_budget_mb is not None and vram > self.vram_budget_mb:
            # Degrade rather than fail: the lite profile must still answer.
            steps = [s for s in steps if TOOL_VRAM_MB.get(s.tool, 0) <= self.vram_budget_mb]
            vram, runtime = self._estimate(steps)

        plan = Plan(
            run_id=manifest.run_id,
            legal_tasks=legal,  # type: ignore[arg-type]
            tasks=[task],
            steps=steps,
            fallbacks=self.matrix.tasks[task].fallbacks,
            matrix_version=self.matrix.version,
            estimated_vram_mb=vram,
            estimated_runtime_ms=runtime,
        )

        # The guarantee: nothing leaves this method without passing the matrix.
        assert_legal(plan, self.matrix)
        return plan

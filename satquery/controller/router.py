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

from dataclasses import dataclass

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.plan import Plan, PlanStep, RationaleTag, TaskID
from satquery.controller.intent import (  # noqa: F401 - CONFIG_TO_LEGAL_TASKS re-exported
    CONFIG_TO_LEGAL_TASKS,
    IntentClassifier,
    IntentPrediction,
    default_classifier,
)
from satquery.controller.matrix_loader import CapabilityMatrix
from satquery.controller.understanding import (
    QueryUnderstanding,
    is_bare_comparison,
    is_open_scene_question,
    is_overall_change_amount,
    resolve_follow_up,
    understand,
)
from satquery.controller.validator import assert_legal


@dataclass(frozen=True)
class RouteDecision:
    """Everything one routing decision produced, carried rather than stashed.

    `prediction` is None when the input checks forced the abstention, because
    the classifier was not consulted at all. `config_excluded` names the task
    the query actually asked for when the input configuration cannot support
    it - the plan is still legal, and the answer says so.
    """

    plan: Plan
    prediction: IntentPrediction | None
    config_excluded: str | None
    # What the query was understood to ask (2026-09-20): the resolved
    # sentence, the object, the classes, the spatial scope ... Carried in the
    # routing trace and handed to the tools as runtime parameters. None only
    # on the blocking-failure path, where nothing was interpreted.
    understanding: QueryUnderstanding | None = None

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
        shed_tools: tuple[str, ...] | list[str] = (),
    ):
        self.matrix = matrix
        self.classifier = classifier or default_classifier()
        self.vram_budget_mb = vram_budget_mb
        self.shed_tools = frozenset(shed_tools)

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

    def _captioner_loaded(self) -> bool:
        from satquery.tools.stubs import REGISTRY, CaptionStub

        if self.shed_tools and "caption_v1" in self.shed_tools:
            return False
        return not isinstance(REGISTRY.get("caption_v1"), CaptionStub)

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
    def _default_params(self, task: TaskID, classes: list[str] | None = None) -> dict:
        """Only ever emit parameters the matrix permits, using its defaults.

        `classes` - the land-cover classes the query named - replaces the
        default when the task permits a `classes` parameter and every named
        class is in its `enum_subset`. A class the matrix does not know is
        dropped rather than passed through; the validator would reject the
        plan otherwise, and the query cannot widen the vocabulary.
        """
        params: dict = {}
        for name, schema in self.matrix.tasks[task].permitted_params.items():
            if schema.default is not None:
                params[name] = schema.default
            if name == "classes" and classes:
                allowed = list(getattr(schema, "enum_subset", None) or [])
                chosen = [c for c in classes if c in allowed]
                if chosen:
                    params[name] = chosen
        return params

    def _build_steps(
        self, task: TaskID, manifest: InputManifest, classes: list[str] | None = None
    ) -> list[PlanStep]:
        cfg = self.matrix.tasks[task]
        rationale = RATIONALE_BY_TASK.get(task, RationaleTag.VQA_INFERENCE)
        params = self._default_params(task, classes)

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
    def route(self, query: str, manifest: InputManifest, history: list[dict] | None = None) -> Plan:
        """The plan alone. Equivalent to `decide(...).plan`.

        Kept because most callers - the adversarial harness, the ablations,
        the routing tests - want only the plan. It also refreshes the
        `last_*` attributes below for callers that read them, which is safe
        in a single-threaded caller and unsafe across concurrent ones; that
        is exactly why the controller uses `decide` instead.
        """
        return self.decide(query, manifest, history=history).plan

    def decide(
        self, query: str, manifest: InputManifest, history: list[dict] | None = None
    ) -> RouteDecision:
        """Plan plus everything the executor needs to explain it.

        The prediction and the config-excluded task are **returned**, not
        stored on `self`. They used to be attributes that `route()` set and
        the controller read immediately afterwards, which is correct only
        while one request at a time is in flight. The API builds a single
        `Controller` - so a single `Router` - and serves `/runs/stream` from
        one thread per request, so two concurrent runs could interleave the
        write and the read: run A would attach run B's "this input
        configuration cannot support TEMPORAL_CHANGE_MAP" notice to its own
        answer, or take B's classifier scores into its trace. Nothing in the
        trace would show it happened.
        """
        legal = self.legal_tasks(manifest)
        config_excluded: str | None = None
        prediction: IntentPrediction | None = None

        # A follow-up ("Where exactly?", "only buildings") is rewritten into
        # a standalone sentence BEFORE anything is classified, so it goes
        # through the same gates as a fresh query. `history` is the previous
        # turns of this conversation as the API received them; without one
        # the query is used as typed.
        resolved = resolve_follow_up(query, history, manifest.config)
        text = resolved[0] or query

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
            unconstrained = self.classifier.predict(text)
            if unconstrained.is_confident and unconstrained.task not in legal:
                config_excluded = unconstrained.task

            # The constrained prediction sees the input configuration as a
            # token (intent.CONFIG_TOKENS): "are there any new structures?"
            # is a change question on a pair and a presence question on one
            # image, and only the token tells the model which it is facing.
            prediction = self.classifier.predict(text, candidates=legal, config=manifest.config)
            # A low-confidence pick is normally discarded in favour of the
            # configuration default, because a weak guess at *which* capability
            # to use still beats refusing. CLARIFY_OR_ABSTAIN is the one class
            # where that reasoning inverts: the fallback ANSWERS, so overriding
            # a weak abstain pick converts "I could not understand this" into a
            # confident-looking answer to a query with no content in it.
            #
            # An empty query, "   " and "hmm" carry no features at all, so they
            # land on the class prior - CLARIFY_OR_ABSTAIN top-1 at 0.353 with a
            # 0.042 margin, below the generic confidence bar. Whether they
            # abstained was therefore decided by where a linear model's prior
            # happened to sit, and it moved every time the template bank
            # changed size. Honouring the abstain pick regardless of its
            # confidence takes that off the knife edge in the safe direction.
            if prediction.is_confident or prediction.task == "CLARIFY_OR_ABSTAIN":
                task = prediction.task
            else:
                task = CONFIG_DEFAULT_TASK.get(manifest.config, "SINGLE_VQA")
                if task not in legal:
                    task = "CLARIFY_OR_ABSTAIN"
            if (
                task == "CLARIFY_OR_ABSTAIN"
                and manifest.config == "BITEMPORAL_PAIR"
                and is_bare_comparison(text)
            ):
                task = "TEMPORAL_CHANGE_DESC"
            if (
                task == "TEMPORAL_CHANGE_VQA"
                and "TEMPORAL_CHANGE_MAP" in legal
                and is_overall_change_amount(text)
            ):
                # "How much area changed?" names no subject, so the change-VQA
                # path (per-subject index deltas) declines it; the change mask
                # measures exactly this total.
                task = "TEMPORAL_CHANGE_MAP"
            if (
                task in ("SINGLE_VQA", "CLARIFY_OR_ABSTAIN")
                and "SINGLE_CAPTION" in legal
                and is_open_scene_question(text)
                and self._captioner_loaded()
            ):
                # Only with the real captioner: when it is a stub the VQA
                # adapter is still the better of the two answers.
                task = "SINGLE_CAPTION"

        if task not in legal:
            task = "CLARIFY_OR_ABSTAIN"

        understanding = understand(
            query, manifest.config, task, [img.role for img in manifest.images],
            history=history, resolved=resolved,
        )
        steps = self._build_steps(task, manifest, classes=understanding.classes)
        if self.shed_tools:
            # Profile-level shedding (the cpu profile drops the 3B VLM). Same
            # degrade-not-fail contract as the budget below.
            steps = [s for s in steps if s.tool not in self.shed_tools]
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

        # Refreshed for the callers that still read them. Both are written on
        # every call, including this abstention path, so a stale value from an
        # earlier query can never be read back as if it belonged to this one.
        self.last_prediction = prediction
        self.last_config_excluded = config_excluded

        return RouteDecision(
            plan=plan,
            # Routing was decided by the input checks rather than by the
            # classifier, so there is no prediction to report. The executor
            # renders this as `classifier="not_invoked"` instead of inventing
            # a score - which is why it is None here rather than a zeroed
            # prediction object.
            prediction=None if manifest.blocking_failures else prediction,
            config_excluded=config_excluded,
            understanding=None if manifest.blocking_failures else understanding,
        )

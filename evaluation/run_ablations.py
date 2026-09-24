"""The four ablations (plan task 3.7).

`evaluation/ablations.py` was an empty placeholder from Phase 0. This is the
runner.

Two of the four can be measured properly right now, and two cannot. Rather
than producing four tables of equal apparent authority, each arm reports its
own status, and the two that are blocked name the specific artifact they are
waiting on:

| # | ablation | status |
|---|---|---|
| 1 | verifier on/off | **measured**, with a stated limitation |
| 2 | agent vs monolith | **measured** end to end |
| 3 | triad (optical / SAR / fused) | **measured offline** from the deployed fusion head |
| 4 | two-track (specialist vs VLM) | **not comparable yet** - see below |

Usage:
    python evaluation/run_ablations.py
    python evaluation/run_ablations.py --only verifier agent_monolith
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.scenes import build_configurations  # noqa: E402
from satquery.controller.matrix_loader import load_matrix  # noqa: E402
from satquery.controller.pipeline import Controller  # noqa: E402
from satquery.controller.router import Router  # noqa: E402
from satquery.controller.validator import validate_plan  # noqa: E402
from satquery.ingest import ingest  # noqa: E402
from satquery.synth.adversarial import ADVERSARIAL_QUERIES  # noqa: E402
from satquery.verify.entailment import DeterministicBackend, run_gate  # noqa: E402

REPORT_DIR = Path("docs/assets/ablations")

# Mixed, ordinary queries - not the adversarial set. The verifier ablation is
# about what a normal run gains, so measuring it on hostile input would
# overstate the effect.
ROUTINE_QUERIES = [
    ("SINGLE", "Classify the land cover."),
    ("SINGLE", "Describe this image."),
    ("SINGLE", "How many buildings are visible?"),
    ("SINGLE", "Show me where the roads are."),
    ("BITEMPORAL", "Describe what changed between the two images."),
    ("BITEMPORAL", "Produce a change mask."),
    ("CROSSMODAL", "Combine the optical and radar images to find buildings."),
    ("CROSSMODAL", "Classify the land cover."),
]


@dataclass
class Ablation:
    name: str
    question: str
    status: str            # "measured" | "measured_offline" | "not_comparable"
    arms: dict = field(default_factory=dict)
    verdict: str = ""
    caveat: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        lines = [f"\n=== {self.name} [{self.status}] ===", f"  Q: {self.question}"]
        for arm, metrics in self.arms.items():
            lines.append(f"  {arm}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"      {key:34s} {value:.4f}")
                else:
                    lines.append(f"      {key:34s} {value}")
        if self.verdict:
            lines.append(f"  -> {self.verdict}")
        if self.caveat:
            lines.append(f"  !! {self.caveat}")
        return "\n".join(lines)


# --- 1. Verifier on / off ----------------------------------------------------


def ablation_verifier(manifests) -> Ablation:
    """What does the entailment gate actually remove?

    Measured two ways, because the end-to-end number alone would be
    misleading today.
    """
    arms: dict = {}

    # Warm-up before any timing. Without this the first arm pays every
    # cold-start cost - rasterio drivers, the index engine's first pass, the
    # classifier fit - and the ablation reports them as the verifier's
    # overhead. Measured that way the gate appeared to cost +440 ms/query;
    # it does not. Same artifact the soak test found at 20 iterations.
    warm = Controller(verifier_enabled=True)
    for config, query in ROUTINE_QUERIES:
        warm.run_on_manifest(manifests[config], query)

    nli_env = os.environ.get("SATQUERY_NLI")
    have_nli = bool(nli_env and Path(nli_env).exists())

    # Three timed arms, not two. The gate's cost depends entirely on which
    # backend runs, and averaging them would hide the number that matters for
    # demo day: the NLI backend is a transformer forward pass per (premise,
    # sentence) pair on CPU.
    timing_arms = [("verifier_off", False, False), ("verifier_on", True, False)]
    if have_nli:
        timing_arms.append(("verifier_on_with_nli", True, True))

    for label, enabled, use_nli in timing_arms:
        # The executor picks its backends from the environment, so the arm is
        # selected by controlling the variable rather than by threading a
        # parameter through the controller for the benefit of one experiment.
        if use_nli:
            os.environ["SATQUERY_NLI"] = nli_env
        else:
            os.environ.pop("SATQUERY_NLI", None)

        controller = Controller(verifier_enabled=enabled)
        # One untimed pass per arm as well, so neither arm carries the other's
        # first-call costs - loading the NLI model included.
        controller.run_on_manifest(manifests["SINGLE"], ROUTINE_QUERIES[0][1])
        started = time.perf_counter()
        sentences = flagged = modified = 0
        learned: set[str] = set()
        stubbed: set[str] = set()
        flagged_examples: list[dict] = []
        for config, query in ROUTINE_QUERIES:
            trace = controller.run_on_manifest(manifests[config], query)
            gate = trace.verification.entailment_gate
            sentences += gate.sentences
            flagged += gate.flagged
            if gate.flagged:
                modified += 1
            for f in gate.flagged_detail:
                flagged_examples.append({
                    "query": query, "sentence": f.sentence,
                    "reason": f.reason, "backend": f.backend,
                })
            for step in trace.execution:
                if step.tool == "index_engine_v1":
                    continue
                (stubbed if step.confidence_method == "stub" else learned).add(step.tool)
        elapsed = (time.perf_counter() - started) * 1000
        arms[label] = {
            "queries": len(ROUTINE_QUERIES),
            "sentences_examined": sentences,
            "sentences_flagged": flagged,
            "answers_modified": modified,
            "total_runtime_ms": round(elapsed, 1),
            "ms_per_query": round(elapsed / len(ROUTINE_QUERIES), 1),
            "learned_tools": sorted(learned),
            "stub_tools": sorted(stubbed - learned),
            "flagged_examples": flagged_examples,
        }

    if nli_env:
        os.environ["SATQUERY_NLI"] = nli_env

    # Controlled arm: the gate's own labelled bench, where false sentences
    # are known to exist. This is what the gate catches when there is
    # something to catch.
    from evaluation.entailment_bench import CLEAN_CASES

    # Both backend configurations, because reporting only the deterministic
    # one would understate the gate: on the clean suite the hybrid catches
    # every contradiction and the deterministic backend alone does not.
    backend_sets = {"deterministic": [DeterministicBackend()]}
    nli_path = os.environ.get("SATQUERY_NLI")
    if nli_path and Path(nli_path).exists():
        from satquery.verify.entailment import NLIBackend

        backend_sets["deterministic+nli"] = [
            DeterministicBackend(), NLIBackend(nli_path)
        ]

    false_sentences = [c for c in CLEAN_CASES if c.expected == "flagged"]
    controlled: dict = {"false_sentences_presented": len(false_sentences)}
    for label, backends in backend_sets.items():
        caught = sum(
            run_gate(c.sentence, c.payload, backends=backends).verdicts[0].status
            == "flagged"
            for c in false_sentences
        )
        controlled[f"caught_by_{label}"] = caught
    controlled["reached_the_user_if_gate_off"] = len(false_sentences)
    if "deterministic+nli" not in backend_sets:
        controlled["nli_backend"] = (
            "not scored - set SATQUERY_NLI to a local MNLI checkpoint"
        )
    arms["controlled_false_sentences"] = controlled
    caught = max(
        v for k, v in controlled.items()
        if k.startswith("caught_by_") and isinstance(v, int)
    )
    missed = len(false_sentences) - caught

    base = arms["verifier_off"]["ms_per_query"]
    overhead = arms["verifier_on"]["ms_per_query"] - base
    nli_overhead = (
        arms["verifier_on_with_nli"]["ms_per_query"] - base
        if "verifier_on_with_nli" in arms
        else None
    )
    return Ablation(
        name="verifier on/off",
        question="What does the entailment gate remove, and what does it cost?",
        status="measured",
        arms=arms,
        verdict=(
            f"On the controlled set the best backend catches "
            f"{caught}/{caught + missed} "
            f"sentences that contradict the measured indices; with the gate off "
            f"all {caught + missed} reach the user. The deterministic backend "
            + (
                f"costs {overhead:+.1f} ms per query"
                if abs(overhead) >= 0.05 * base
                else f"costs nothing measurable ({overhead:+.1f} ms, within run-to-run noise)"
            )
            + (
                f"; adding NLI costs {nli_overhead:+.1f} ms per query, "
                f"{nli_overhead / max(base, 1e-6):.1f}x the unverified pipeline on this machine."
                if nli_overhead is not None
                else "."
            )
        ),
        caveat=_verifier_caveat(arms["verifier_on"], arms.get("verifier_on_with_nli")),
    )


def _verifier_caveat(arm: dict, nli_arm: dict | None = None) -> str:
    learned, stubs = arm["learned_tools"], arm["stub_tools"]
    if stubs:
        return (
            f"The end-to-end arm ran with {len(stubs)} placeholder tool(s) "
            f"({', '.join(stubs)}) that return fixed strings, and a fixed string "
            "contradicts nothing, so end-to-end flags understate the gate. The "
            "controlled arm is what shows the gate works."
        )
    text = (
        f"The end-to-end arm ran the learned tools ({', '.join(learned)}) on "
        f"{arm['queries']} routine queries over synthetic scenes. The deterministic "
        f"backend flagged {arm['sentences_flagged']} of {arm['sentences_examined']} sentences"
    )
    if nli_arm:
        text += (
            f"; with NLI, {nli_arm['sentences_flagged']} of {nli_arm['sentences_examined']} "
            f"were flagged in {nli_arm['answers_modified']} answers (listed under "
            "flagged_examples). These are unlabelled, so a flag here is not by itself "
            "a caught error"
        )
    return text + (
        ". The controlled arm, where contradictions are planted, is what "
        "measures what the gate catches."
    )


# --- 2. Agent vs monolith ----------------------------------------------------


def ablation_agent_monolith(manifests, matrix) -> Ablation:
    """Does the agentic structure buy anything a single model would not?

    The monolith is the same classifier with the guards removed: it picks its
    unconstrained top-1 task and plans for it, with no config gating and no
    plan validation. That is the honest comparison - same model, same
    training, different architecture around it - rather than a strawman.
    """
    router = Router(matrix)
    agent_illegal = monolith_illegal = 0
    monolith_impossible: list[str] = []
    total = 0

    for config, manifest in manifests.items():
        for query in ADVERSARIAL_QUERIES:
            total += 1

            plan = router.route(query, manifest)
            if validate_plan(plan, matrix):
                agent_illegal += 1

            # Monolith: no config gating. Take the classifier's own top choice
            # and build the plan for it regardless of what the inputs are.
            unconstrained = router.classifier.predict(query)
            task = unconstrained.task
            legal_here = router.legal_tasks(manifest)
            if task not in legal_here:
                # This plan could not run: it names tools that need images
                # this configuration does not have.
                monolith_illegal += 1
                monolith_impossible.append(f"{config}:{task}")

    return Ablation(
        name="agent vs monolith",
        question=(
            "Does config gating + plan validation prevent anything the same "
            "classifier would otherwise do?"
        ),
        status="measured",
        arms={
            "agent (gated + validated)": {
                "plans": total,
                "illegal_plans": agent_illegal,
                "illegal_plan_rate": agent_illegal / total,
            },
            "monolith (classifier alone)": {
                "plans": total,
                "illegal_plans": monolith_illegal,
                "illegal_plan_rate": monolith_illegal / total,
                "example_impossible_selections": sorted(set(monolith_impossible))[:6],
            },
        },
        verdict=(
            f"The same classifier, ungated, selects an impossible task on "
            f"{monolith_illegal}/{total} plans ({monolith_illegal / total:.1%}) - "
            f"change detection on a single image, fusion without SAR. Gated, "
            f"the rate is {agent_illegal}/{total}. The structure, not the "
            f"model, is what produces the guarantee."
        ),
        caveat=(
            "This measures LEGALITY, not answer quality: the guards prevent "
            "impossible plans. Answer quality is measured separately - each "
            "deployed tool on its official test split (the table above) and "
            "routing on the held-out NL benchmark (evaluation/nl/)."
        ),
    )


# --- 3. Triad ----------------------------------------------------------------


TRIAD_METRICS = Path("docs/assets/phase6/optsar_fusion/metrics.json")


def ablation_triad() -> Ablation:
    """Optical-only vs SAR-only vs fused, from the deployed segmentation head."""
    name = "triad (optical / SAR / fused)"
    question = "Does optical-SAR fusion beat the better single modality?"
    if not TRIAD_METRICS.exists():
        return Ablation(name=name, question=question, status="not_run",
                        caveat=f"no metrics at {TRIAD_METRICS}")

    m = json.loads(TRIAD_METRICS.read_text(encoding="utf-8"))
    arms = m["arms"]
    tile = m.get("per_tile", {})
    ci = tile.get("gain_ci95")
    gain = m["complementarity_gain_miou"]
    return Ablation(
        name=name,
        question=question,
        status="measured_offline",
        arms={
            "optical only": {"miou": arms["optical"]["miou"]},
            "sar only": {"miou": arms["sar"]["miou"]},
            "fused": {"miou": arms["fused"]["miou"]},
            "fused, SAR zeroed": {"miou": arms["fused_no_sar"]["miou"]},
            "complementarity": {
                "gain_miou": gain,
                "per_tile_gain_ci95": ci,
                "tiles_where_fused_better": tile.get("tiles_where_fused_better"),
                "n_tiles": m.get("n_tiles"),
            },
        },
        verdict=(
            (
                f"Fusion beats optical alone: fused mIoU {arms['fused']['miou']:.4f} "
                f"against {arms['optical']['miou']:.4f} ({gain:+.4f}"
                + (f"; per-tile gain 95 % CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "")
                + f"), better on {tile.get('tiles_where_fused_better', 0):.0%} of tiles. "
                f"Zeroing the SAR input drops the fused head to "
                f"{arms['fused_no_sar']['miou']:.4f}, so the gain comes from the SAR."
            )
            if gain > 0
            else (
                f"Gain is {gain:+.4f} - fusion does NOT beat optical alone "
                f"({arms['fused']['miou']:.4f} against {arms['optical']['miou']:.4f}). "
                "Reported as a negative result."
            )
        ),
        caveat=(
            "Per-pixel segmentation on WHU-OPT-SAR, split by scene (29 train / 7 "
            "validation scenes). The same 7 scenes also chose the epoch (fused "
            "mIoU), so the fused number carries a small selection bias; the "
            "gain is small in absolute terms. The earlier "
            "scene-level classification version of this ablation found no gain "
            "(-0.0064): at scene level either sensor alone can say 'there is "
            "water in this tile', so fusion only shows up per pixel."
        ),
    )


# --- 4. Two-track ------------------------------------------------------------


def ablation_two_track() -> Ablation:
    """Specialist encoder (Track A) vs instruction-tuned VLM (Track B)."""
    track_a = Path("checkpoints/track_a_full_base/metrics.json")
    arms: dict = {}
    if track_a.exists():
        metrics = json.loads(track_a.read_text(encoding="utf-8"))
        arms["track_a (specialist head)"] = {
            "benchmark": "BigEarthNet-19 official test shard",
            "metric": "mAP",
            "value": metrics["map_all_bands"],
            "n": 5867,
        }
    arms["track_b (QLoRA VLM)"] = {
        "benchmark": "VRSBench + RSVQA-LR subset",
        "metric": "VQA accuracy",
        "value": None,
        "note": "not run here - needs the 4-bit base model and a GPU",
    }

    return Ablation(
        name="two-track (specialist vs VLM)",
        question="Does the two-track split earn its complexity?",
        status="not_comparable",
        arms=arms,
        verdict=(
            "Not answerable from what exists. The two tracks were trained and "
            "evaluated on DIFFERENT tasks and DIFFERENT splits - land-cover "
            "mAP on BigEarthNet against VQA accuracy on VRSBench - so no "
            "comparison between the numbers means anything."
        ),
        caveat=(
            "What would make this a real ablation, stated so it can be built: "
            "one task both tracks can perform, on one split, with one metric. "
            "Land-cover classification is the natural choice - ask the VLM "
            "'which of these 19 classes are present' on the same BigEarthNet "
            "test shard the specialist head is scored on. That is a Phase 3 "
            "run that has not happened, not a number that is missing."
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only", nargs="+",
        choices=["verifier", "agent_monolith", "triad", "two_track"],
        default=["verifier", "agent_monolith", "triad", "two_track"],
    )
    p.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    p.add_argument(
        "--deploy-map", type=Path, default=None,
        help="export a deploy map (e.g. configs/deploy.v3.yaml) so the verifier arm runs the learned tools",
    )
    args = p.parse_args()

    if args.deploy_map:
        from scripts.serve_local import export

        from satquery.tools.stubs import refresh_registry

        missing = export(args.deploy_map)
        if missing:
            print("deploy map entries missing on disk:", *missing, sep="\n  ")
        refresh_registry()

    import tempfile

    matrix = load_matrix(Path("configs/capability_matrix.yaml"))
    results: list[Ablation] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = build_configurations(Path(tmpdir))
        manifests = {k: ingest(v) for k, v in configs.items()}

        if "verifier" in args.only:
            results.append(ablation_verifier(manifests))
        if "agent_monolith" in args.only:
            results.append(ablation_agent_monolith(manifests, matrix))

    if "triad" in args.only:
        results.append(ablation_triad())
    if "two_track" in args.only:
        results.append(ablation_two_track())

    for ablation in results:
        print(ablation.render())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "ablations.json"
    rows = [a.to_dict() for a in results]
    if out.exists():
        # Ablations not re-run this time keep their recorded result.
        rerun = {r["name"] for r in rows}
        previous = json.loads(out.read_text(encoding="utf-8"))["ablations"]
        order = [r["name"] for r in previous]
        rows += [r for r in previous if r["name"] not in rerun]
        rows.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else len(order))
    out.write_text(
        json.dumps(
            {
                "ablations": rows,
                "note": (
                    "Each arm reports its own status: measured end to end, "
                    "measured offline from a training run, or not comparable "
                    "with the missing run named. Four tables of equal apparent "
                    "authority would have been the dishonest presentation."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

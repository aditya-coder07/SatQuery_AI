"""Query understanding (2026-09-20): the NL -> IR layer.

Three things are pinned here:

1. **Extraction** is deterministic and never invents: the referring phrase,
   the land-cover classes, the spatial scope, the quantity and the image
   index come out of the sentence or are None.
2. **Follow-ups** are resolved against the previous turn into a standalone
   sentence that routes like a fresh query, and the trace records it.
3. **The benchmark stays held out**: no expansion of the template bank may
   equal a query in `evaluation/nl/*.jsonl`, and the router must clear a
   floor on both splits so a bank edit that regresses natural language is
   caught in CI, not in a demo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evaluation.nl_understanding_eval import evaluate, manifests
from satquery.controller.intent import IntentClassifier, with_config
from satquery.controller.matrix_loader import load_matrix
from satquery.controller.router import Router
from satquery.controller.understanding import (
    extract_classes,
    extract_image_index,
    extract_object,
    extract_quantity,
    extract_spatial,
    referring_expression,
    resolve_follow_up,
)
from satquery.synth.query_bank import generate

NL_DIR = Path("evaluation/nl")


@pytest.fixture(scope="module")
def router():
    return Router(load_matrix("configs/capability_matrix.yaml"))


@pytest.fixture(scope="module")
def by_config(tmp_path_factory):
    return manifests(tmp_path_factory.mktemp("nl"))


class TestExtraction:
    @pytest.mark.parametrize("query,expected", [
        ("Find the airport in this image.", "airport"),
        ("Where is the road?", "road"),
        ("Show me where the roads are.", "roads"),
        ("How many buildings are visible?", "buildings"),
        ("Is there a stadium", "stadium"),
        ("Could you tell me where the football pitch is?", "football pitch"),
        ("Has the built-up area increased, decreased, or remained unchanged?", "built-up area"),
        ("Perform referring expression comprehension for 'the plane at the top'.", "plane"),
        ("Give me a mask of the new buildings.", "buildings"),
        ("Describe this image.", None),
        ("What is the weather forecast for this location", None),
        ("hmm", None),
        ("", None),
    ])
    def test_object(self, query, expected):
        assert extract_object(query) == expected

    def test_referring_expression_keeps_the_spatial_qualifier(self):
        q = "Where is the airplane on the right side of the runway"
        obj, spatial = extract_object(q), extract_spatial(q)
        assert obj == "airplane"
        assert spatial.startswith("right")
        assert referring_expression(q, obj, spatial).startswith("the airplane ")
        assert "right" in referring_expression(q, obj, spatial)

    def test_referring_expression_is_none_without_an_object(self):
        assert referring_expression("Describe this image.", None, None) is None

    @pytest.mark.parametrize("query,expected", [
        ("Can you identify the objects near the top-right?", "top-right"),
        ("Tell me what is in the top left corner", "top left corner"),
        ("What is the large structure at the bottom right?", "bottom right"),
        ("Is there a river?", None),
    ])
    def test_spatial(self, query, expected):
        assert extract_spatial(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("Break the scene down into water, vegetation and built-up areas.", ["built_up", "water", "vegetation"]),
        ("Map the bare soil.", ["bare_soil"]),
        ("How much vegetation is present?", ["vegetation"]),
        ("Describe this image.", None),
    ])
    def test_classes(self, query, expected):
        assert extract_classes(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("How many aircraft are parked on the apron?", "count"),
        ("what percent of this is water", "fraction"),
        ("How many hectares of forest were lost?", "area"),
        ("By how much did the water body grow?", "area"),
        ("Is the water area bigger than the built-up area?", "comparison"),
        ("Is there more water in the second image?", "comparison"),
        ("Is there a bridge?", None),
    ])
    def test_quantity(self, query, expected):
        assert extract_quantity(query) == expected

    @pytest.mark.parametrize("query,config,expected", [
        ("Is there a bridge in the second image?", "BITEMPORAL_PAIR", 1),
        ("Describe the first image.", "BITEMPORAL_PAIR", 0),
        ("Is the second picture more built up than the first?", "BITEMPORAL_PAIR", None),
        ("Is there a road in the optical image?", "CROSSMODAL_PAIR", 0),
        ("Describe the radar image.", "CROSSMODAL_PAIR", 1),
        ("Is there a bridge in the second image?", "SINGLE", None),
    ])
    def test_image_index(self, query, config, expected):
        assert extract_image_index(query, config) == expected


class TestFollowUps:
    def test_where_exactly_after_a_change_description_maps_the_change(self):
        history = [{"query": "What changed in these images?", "task": "TEMPORAL_CHANGE_DESC",
                    "answer": "Several buildings appear to have changed."}]
        resolved, note = resolve_follow_up("Where exactly?", history, "BITEMPORAL_PAIR")
        assert "where" in resolved.lower() and "change" in resolved.lower()
        assert note and "TEMPORAL_CHANGE_DESC" in note

    def test_only_narrows_the_object(self):
        history = [{"query": "What changed?", "task": "TEMPORAL_CHANGE_DESC", "answer": "..."}]
        resolved, _ = resolve_follow_up("Only show buildings.", history, "BITEMPORAL_PAIR")
        assert "buildings" in resolved and "changed" in resolved

    def test_pronoun_is_replaced_by_the_previous_object(self):
        history = [{"query": "Where is the airport?", "task": "SINGLE_GROUND", "answer": "box"}]
        resolved, note = resolve_follow_up("Is it near the coast?", history, "SINGLE")
        assert resolved == "Is the airport near the coast?"
        assert note

    def test_a_fresh_sentence_is_never_rewritten(self):
        history = [{"query": "Where is the airport?", "task": "SINGLE_GROUND", "answer": "box"}]
        q = "How many buildings are visible in this image?"
        assert resolve_follow_up(q, history, "SINGLE") == (q, None)

    def test_no_history_means_no_resolution(self):
        assert resolve_follow_up("Where exactly?", None, "SINGLE") == ("Where exactly?", None)
        assert resolve_follow_up("Where exactly?", [], "SINGLE") == ("Where exactly?", None)

    def test_the_router_records_the_resolution_in_the_understanding(self, router, by_config):
        history = [{"query": "How many ships are there?", "task": "SINGLE_VQA", "answer": "4"}]
        decision = router.decide("Where are they?", by_config["SINGLE"], history=history)
        u = decision.understanding
        assert decision.plan.tasks[0] == "SINGLE_GROUND"
        assert u.follow_up and u.query == "Where are they?"
        assert u.resolved_query == "Where are the ships?"
        assert u.referring_expression == "the ships"


class TestRouterIntegration:
    def test_implicit_change_on_a_pair_is_a_change_task(self, router, by_config):
        for q in ("did anything get built?", "are there any new structures?", "what got demolished"):
            task = router.decide(q, by_config["BITEMPORAL_PAIR"]).plan.tasks[0]
            assert task.startswith("TEMPORAL_CHANGE"), (q, task)

    def test_the_same_words_on_one_image_are_a_presence_question(self, router, by_config):
        task = router.decide("are there any new structures?", by_config["SINGLE"]).plan.tasks[0]
        assert task == "SINGLE_VQA"

    def test_named_classes_reach_the_plan(self, router, by_config):
        plan = router.decide("Map the bare soil.", by_config["SINGLE"]).plan
        assert plan.tasks[0] == "SINGLE_LANDCOVER"
        assert plan.steps[-1].params["classes"] == ["bare_soil"]

    def test_unknown_classes_never_reach_the_plan(self, router, by_config):
        plan = router.decide("Classify the land cover.", by_config["SINGLE"]).plan
        assert plan.steps[-1].params["classes"] == ["built_up", "water"]  # matrix default

    def test_understanding_is_none_on_the_blocking_path(self, router, tmp_path):
        from evaluation.scenes import build_tiny_raster
        from satquery.ingest import ingest

        manifest = ingest([build_tiny_raster(tmp_path / "tiny.tif")])
        assert manifest.blocking_failures
        decision = router.decide("Where is the road?", manifest)
        assert decision.understanding is None

    def test_config_token_is_a_prefix(self):
        assert with_config("x", "BITEMPORAL_PAIR") == "[bitemporal] x"
        assert with_config("x", None) == "x"

    def test_config_exclusion_still_detected_without_a_token(self, router, by_config):
        decision = router.decide("Produce a change mask for these images.", by_config["SINGLE"])
        assert decision.config_excluded == "TEMPORAL_CHANGE_MAP"

    BARE_COMPARISONS = (
        "compare the scenes", "Compare them.", "compare", "show me the difference",
        "Can you compare these two images please?", "how do they differ?",
        "spot the differences", "what's the difference between them",
    )

    @pytest.mark.parametrize("query", BARE_COMPARISONS)
    def test_bare_comparison_on_a_pair_describes_the_change(self, router, by_config, query):
        # run_875f94d9fa16 (2026-09-24): "compare the scenes" on a pair abstained.
        decision = router.decide(query, by_config["BITEMPORAL_PAIR"])
        assert decision.plan.tasks[0] == "TEMPORAL_CHANGE_DESC"
        assert decision.understanding.intent == "change_describe"

    # "spot the differences" on one image is routed to grounding by the
    # classifier (unchanged by the pair rule), so it is not asserted here.
    @pytest.mark.parametrize("query", [q for q in BARE_COMPARISONS if q != "spot the differences"])
    def test_bare_comparison_on_one_image_still_abstains(self, router, by_config, query):
        assert router.decide(query, by_config["SINGLE"]).plan.tasks[0] == "CLARIFY_OR_ABSTAIN"

    @pytest.mark.parametrize("query", ["thx", "hmm", "ok", "hello", "which one is better?", "nice"])
    def test_filler_on_a_pair_still_abstains(self, router, by_config, query):
        assert router.decide(query, by_config["BITEMPORAL_PAIR"]).plan.tasks[0] == "CLARIFY_OR_ABSTAIN"

    @pytest.mark.parametrize("query,expected", [
        ("compare the scenes", True),
        ("Show me the differences, please.", True),
        ("compare the vegetation in both", False),  # has a subject: the classifier's job
        ("which one is better", False),
        ("thanks", False),
        ("", False),
    ])
    def test_is_bare_comparison(self, query, expected):
        from satquery.controller.understanding import is_bare_comparison
        assert is_bare_comparison(query) is expected


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class TestBenchmarkIntegrity:
    @pytest.fixture(scope="class")
    def eval_rows(self):
        rows = []
        for name in ("queries.jsonl", "queries_test.jsonl", "queries_final.jsonl"):
            rows += [json.loads(l) for l in (NL_DIR / name).read_text(encoding="utf-8").splitlines() if l.strip()]
        return rows

    def test_no_bank_expansion_equals_a_benchmark_query(self, eval_rows):
        bank = {_norm(e.text) for e in generate()}
        bank |= {_norm(e.text) for e in generate(decorate=False)}
        leaked = [r["query"] for r in eval_rows if _norm(r["query"]) in bank]
        assert not leaked, f"benchmark queries present in the template bank: {leaked}"

    def test_benchmark_ids_are_unique(self, eval_rows):
        ids = [r["id"] for r in eval_rows]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("name,floor", [("queries.jsonl", 0.97), ("queries_test.jsonl", 0.93)])
    def test_routing_floor(self, router, by_config, name, floor):
        rows = [json.loads(l) for l in (NL_DIR / name).read_text(encoding="utf-8").splitlines() if l.strip()]
        report = evaluate(router, rows, by_config)
        acc = report["summary"]["routing"]["accuracy"]
        assert acc >= floor, f"{name}: routing {acc} below the floor {floor}; misses: " + \
            "; ".join(f"{f['id']} {f['query']!r} -> {f['task']}" for f in report["failures"] if f["field"] == "routing")

    def test_extraction_floor(self, router, by_config):
        rows = [json.loads(l) for l in (NL_DIR / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        summary = evaluate(router, rows, by_config)["summary"]
        # Floors, not exact numbers: the bank is sampled, and a template
        # edit moves single queries. `image` and `spatial` have n < 10.
        for field, floor in (("object", 0.95), ("classes", 0.95), ("quantity", 0.95),
                             ("temporal", 0.95), ("spatial", 0.85), ("image", 0.8)):
            assert summary[field]["accuracy"] >= floor, (field, summary[field])

    def test_the_classifier_fits_quickly(self):
        import time

        t = time.perf_counter()
        IntentClassifier()
        assert time.perf_counter() - t < 20.0

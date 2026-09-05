"""Answer composition tests.

The executor used to let the last answer-bearing tool in the plan overwrite
`final_answer` outright, and synthesised prose only when *no* tool had
produced any. For SINGLE_CAPTION that meant `index_engine_v1` and
`landcover_v1` both ran, both produced numbers, and both were discarded
because `caption_v1` - last in the plan - emitted one short sentence. These
tests pin the composition that replaced it, and, more importantly, pin the
places it must NOT reach.
"""

from __future__ import annotations

from satquery.synth.narrative import (
    ENRICHABLE_TASKS,
    compose_answer,
    describe_georeferencing,
    describe_location,
)

# 82% water, 18% vegetation - the shape the index engine emits.
INDEX_PAYLOAD = {
    "indices": {
        "ndvi": {"fraction_above_threshold": 0.18},
        "mndwi": {"fraction_above_threshold": 0.82},
    }
}


class TestEnrichment:
    def test_a_caption_keeps_the_measured_description(self):
        answer = compose_answer(
            "SINGLE_CAPTION", "a river runs through the scene", [], INDEX_PAYLOAD
        )
        assert answer.startswith("a river runs through the scene.")
        assert "82% water" in answer
        assert "18% vegetation" in answer

    def test_an_unpunctuated_caption_is_terminated_before_joining(self):
        """The captioner emits `" ".join(words)` with no full stop, which
        concatenated into "...through the scene Index thresholds indicate"."""
        answer = compose_answer(
            "SINGLE_CAPTION", "a river runs through the scene", [], INDEX_PAYLOAD
        )
        assert "scene Index" not in answer

    def test_an_already_punctuated_caption_gains_no_second_full_stop(self):
        answer = compose_answer(
            "SINGLE_CAPTION", "A river runs through the scene.", [], INDEX_PAYLOAD
        )
        assert ".." not in answer

    def test_landcover_classes_reach_a_caption_answer(self):
        """`landcover_v1` is in the SINGLE_CAPTION plan; its labels were
        computed and then dropped from the prose."""
        answer = compose_answer(
            "SINGLE_CAPTION",
            "a coastal scene",
            [{"labels": ["Marine waters"]}],
            INDEX_PAYLOAD,
        )
        assert "Marine waters" in answer

    def test_a_missing_tool_answer_still_synthesises(self):
        answer = compose_answer("SINGLE_CAPTION", "", [], INDEX_PAYLOAD)
        assert "82% water" in answer
        assert not answer.startswith(".")


class TestScoping:
    """The guard that keeps composition out of tasks it would damage."""

    def test_a_vqa_answer_is_left_alone(self):
        """"Did vegetation increase?" wants a direction, not scene
        statistics. `synthesise_answer` returns "" for VQA tasks, and that
        is the signal to leave the model's own answer standing alone."""
        answer = compose_answer(
            "TEMPORAL_CHANGE_VQA", "Vegetation increased.", [], INDEX_PAYLOAD
        )
        assert answer == "Vegetation increased."

    def test_change_map_does_not_append_a_contradicting_sentence(self):
        """TEMPORAL_CHANGE_MAP's synthesised string explains why a tool did
        NOT run. Appending it to a real tool answer would contradict it, so
        that task stays replace-only."""
        answer = compose_answer(
            "TEMPORAL_CHANGE_MAP",
            "The reservoir shrank along its western shore.",
            [],
            INDEX_PAYLOAD,
            artifacts=["change_mask"],
        )
        assert answer == "The reservoir shrank along its western shore."
        assert "did not run in this profile" not in answer

    def test_xmodal_stays_replace_only(self):
        answer = compose_answer(
            "XMODAL_JOINT_EXTRACT", "Fused optical and SAR bands.", [], INDEX_PAYLOAD
        )
        assert answer == "Fused optical and SAR bands."

    def test_the_enrichable_set_is_the_additive_tasks_only(self):
        assert ENRICHABLE_TASKS == {
            "SINGLE_CAPTION",
            "SINGLE_LANDCOVER",
            "TEMPORAL_CHANGE_DESC",
        }


class TestGeoreferencingDisclosure:
    def test_an_ungeoreferenced_file_says_so(self):
        """A PNG cannot carry a CRS, so there is no latitude, longitude or
        area in metres to report - `gsd_m` is GDAL's identity placeholder,
        not a measurement. Silence would read as a system that declined to
        mention where the scene is."""
        note = describe_georeferencing(False, "PNG")
        assert "PNG" in note
        assert "no georeferencing" in note

    def test_a_georeferenced_file_adds_nothing(self):
        assert describe_georeferencing(True, "GTiff") == ""

    def test_the_disclosure_reaches_a_composed_answer(self):
        answer = compose_answer(
            "SINGLE_CAPTION",
            "a coastal scene",
            [],
            INDEX_PAYLOAD,
            georeferenced=False,
            container_format="PNG",
        )
        assert "no georeferencing" in answer

    def test_no_coordinates_are_invented_for_a_georeferenced_file(self):
        """Composition adds measurement, never geography. Nothing in the
        answer path knows where the scene is."""
        answer = compose_answer(
            "SINGLE_CAPTION", "a coastal scene", [], INDEX_PAYLOAD
        )
        for word in ("latitude", "longitude", "°", "EPSG"):
            assert word not in answer


class TestLocation:
    """The location sentence. Every number in it is transformed from the
    file's own CRS and geotransform at ingest; nothing is inferred."""

    def test_a_located_scene_reports_its_centre_and_extent(self):
        sentence = describe_location(True, "GTiff", (18.0829, 75.0060), (1280.0, 1280.0))
        assert "18.0829° N" in sentence
        assert "75.0060° E" in sentence
        assert "1.3 km by 1.3 km" in sentence

    def test_southern_and_western_hemispheres_get_the_right_letters(self):
        """A bare minus sign in front of a coordinate is easy to misread,
        and easy to lose entirely when the text is reflowed."""
        sentence = describe_location(True, "GTiff", (-33.86, -70.66), None)
        assert "33.8600° S" in sentence
        assert "70.6600° W" in sentence
        assert "-" not in sentence

    def test_a_geographic_crs_gets_a_centre_but_no_extent(self):
        """`extent_m` is None for a geographic CRS because `gsd_m` is only
        approximate there. The centre is exact either way."""
        sentence = describe_location(True, "GTiff", (18.08, 75.01), None)
        assert "centred at" in sentence
        assert "on the ground" not in sentence

    def test_a_georeferenced_file_that_will_not_transform_says_nothing(self):
        """The alternative is a coordinate we do not have."""
        assert describe_location(True, "GTiff", None, None) == ""

    def test_an_ungeoreferenced_file_still_gets_the_disclosure(self):
        sentence = describe_location(False, "PNG", None, None)
        assert "no georeferencing" in sentence

    def test_a_sub_kilometre_scene_is_reported_in_metres(self):
        sentence = describe_location(True, "GTiff", (18.08, 75.01), (640.0, 320.0))
        assert "640 m by 320 m" in sentence

    def test_the_location_reaches_a_composed_answer(self):
        answer = compose_answer(
            "SINGLE_CAPTION",
            "a coastal scene",
            [],
            INDEX_PAYLOAD,
            centroid=(18.0829, 75.0060),
            extent_m=(1280.0, 1280.0),
        )
        assert "a coastal scene." in answer
        assert "82% water" in answer
        assert "18.0829° N" in answer

    def test_no_location_is_claimed_without_a_footprint(self):
        """compose_answer defaults carry no centroid, and must not invent
        one to fill the slot."""
        answer = compose_answer("SINGLE_CAPTION", "a coastal scene", [], INDEX_PAYLOAD)
        assert "centred at" not in answer
        for word in ("latitude", "longitude", "°"):
            assert word not in answer

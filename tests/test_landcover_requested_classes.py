"""`landcover_v1` answers about the classes the query named (2026-09-20).

Pure functions, so this runs without torch - the CI environment.
"""

from __future__ import annotations


class TestRequestedClasses:
    """`_classes` (2026-09-20): a query that names classes gets an answer
    about those classes first, in the head's own labels."""

    def test_summary_names_present_and_absent_labels(self):
        from satquery.tools.landcover import requested_summary

        asserted = [{"class": "Urban fabric", "probability": 0.9}]
        denied = ["Inland waters", "Marine waters", "Coastal wetlands", "Inland wetlands", "Beaches, dunes, sands"]
        abstained = [{"class": "Arable land", "probability": 0.5}]
        text = requested_summary(["built_up", "water", "bare_soil"], asserted, denied, abstained)
        assert text.startswith("Asked-for classes - ")
        assert "built up: present as Urban fabric" in text
        assert "water: none of its land-cover labels asserted" in text
        assert "bare soil: none of its land-cover labels asserted" in text

    def test_no_requested_classes_means_no_summary(self):
        from satquery.tools.landcover import requested_summary

        assert requested_summary(None, [], [], []) is None
        assert requested_summary([], [], [], []) is None

    def test_undecided_labels_are_reported_as_such(self):
        from satquery.tools.landcover import requested_summary

        text = requested_summary(["vegetation"], [], [], [{"class": "Arable land", "probability": 0.6}])
        assert "vegetation: undecided" in text

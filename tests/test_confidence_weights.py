"""Weighted confidence combiner and stress response (plan task 3.4)."""

from __future__ import annotations

import math

import pytest

from evaluation.confidence_stress import STRESSORS
from satquery.controller.confidence import (
    DEFAULT_WEIGHTS,
    geometric_mean,
    load_weights,
)


class TestWeightedGeometricMean:
    def test_equal_weights_match_the_unweighted_mean(self):
        assert geometric_mean(0.5, 0.8, 1.0) == pytest.approx(
            (0.5 * 0.8 * 1.0) ** (1 / 3)
        )

    def test_a_heavier_weight_pulls_toward_that_component(self):
        light = geometric_mean(0.2, 0.9, 0.9, weights=(1, 1, 1))
        heavy = geometric_mean(0.2, 0.9, 0.9, weights=(5, 1, 1))
        assert heavy < light

    def test_any_zero_component_collapses_the_score(self):
        """The reason for a geometric rather than arithmetic mean."""
        assert geometric_mean(0.99, 0.99, 0.0) == 0.0
        assert geometric_mean(0.99, 0.99, 0.0, weights=(9, 9, 1)) == 0.0

    def test_all_ones_is_one(self):
        assert geometric_mean(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_mismatched_weight_count_is_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            geometric_mean(0.5, 0.5, weights=(1.0,))

    def test_zero_total_weight_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            geometric_mean(0.5, 0.5, weights=(0.0, 0.0))

    def test_matches_the_closed_form(self):
        values, weights = (0.4, 0.7, 0.9), (2.0, 3.0, 5.0)
        expected = math.exp(
            sum(w * math.log(v) for v, w in zip(values, weights, strict=True))
            / sum(weights)
        )
        assert geometric_mean(*values, weights=weights) == pytest.approx(expected)


class TestWeightLoading:
    def test_shipped_weights_are_equal(self):
        """They must stay equal until there is data to fit them on.

        Fitting needs (components -> was the answer correct) pairs, and no
        learned tool reports a probability of correctness yet - the same gap
        recorded under tasks 3.3 and 3.6. A fitted weight would be fitted to
        nothing. If this fails, the fit must be documented.
        """
        assert load_weights() == DEFAULT_WEIGHTS

    def test_missing_file_falls_back_to_equal(self, tmp_path):
        assert load_weights(tmp_path / "absent.yaml") == DEFAULT_WEIGHTS

    def test_malformed_file_falls_back_to_equal(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("confidence: [not, a, mapping]", encoding="utf-8")
        assert load_weights(path) == DEFAULT_WEIGHTS

    def test_negative_weights_are_rejected(self, tmp_path):
        path = tmp_path / "neg.yaml"
        path.write_text(
            "confidence:\n  weights:\n    model: -1.0\n", encoding="utf-8"
        )
        assert load_weights(path) == DEFAULT_WEIGHTS

    def test_valid_weights_are_honoured(self, tmp_path):
        path = tmp_path / "w.yaml"
        path.write_text(
            "confidence:\n  weights:\n    model: 2.0\n    agreement: 1.0\n"
            "    input_quality: 3.0\n",
            encoding="utf-8",
        )
        assert load_weights(path) == (2.0, 1.0, 3.0)


class TestStressorDefinitions:
    def test_every_stressor_declares_what_it_targets(self):
        for stressor in STRESSORS:
            assert stressor.targets in {"model", "agreement", "input_quality"}
            assert stressor.description

    def test_expected_collateral_carries_a_reason(self):
        """A declared exception must say why, or it is just an excuse."""
        for stressor in STRESSORS:
            for component, why in stressor.expected_collateral.items():
                assert component in {"model", "agreement", "input_quality"}
                assert len(why) > 60, f"{stressor.name}/{component} needs a reason"

    def test_the_suite_covers_more_than_one_target(self):
        assert len({s.targets for s in STRESSORS}) >= 2

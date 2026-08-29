"""The three learned tools wired in for tasks 2.5, 2.7 and 2.8.

Those tasks say `caption_v1`, `grounding_v1` and `change_caption_v1` should be
REAL tools. The models were trained and their metrics reported, but no tool
module existed - the registry kept the stub, so the pipeline could not reach
any of them. These tests cover the wiring.

The real-model tests skip without a checkpoint, matching every other learned
tool. The fallback tests do not skip: "the registry degrades to a stub when
nothing is configured" is what keeps CI green and must always hold.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from satquery.tools import caption as caption_mod
from satquery.tools import change_caption as change_caption_mod
from satquery.tools import grounding as grounding_mod
from satquery.tools.stubs import REGISTRY


def has_ckpt(path: str) -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return Path(path).exists() and (Path(path) / "vocab.json").exists()


class TestAvailabilityGates:
    @pytest.mark.parametrize(
        "module,env",
        [
            (caption_mod, "SATQUERY_CAPTION"),
            (grounding_mod, "SATQUERY_GROUNDING"),
            (change_caption_mod, "SATQUERY_CHANGE_CAPTION"),
        ],
        ids=["caption", "grounding", "change_caption"],
    )
    def test_unset_env_reports_why(self, module, env, monkeypatch):
        monkeypatch.delenv(env, raising=False)
        available, reason = module.is_available()
        assert available is False
        assert env in reason

    @pytest.mark.parametrize(
        "module,env",
        [
            (caption_mod, "SATQUERY_CAPTION"),
            (grounding_mod, "SATQUERY_GROUNDING"),
            (change_caption_mod, "SATQUERY_CHANGE_CAPTION"),
        ],
        ids=["caption", "grounding", "change_caption"],
    )
    def test_missing_checkpoint_reports_the_path(self, module, env, monkeypatch, tmp_path):
        monkeypatch.setenv(env, str(tmp_path / "absent"))
        available, reason = module.is_available()
        assert available is False
        assert "not found" in reason

    def test_a_checkpoint_without_a_vocab_is_refused(self, monkeypatch, tmp_path):
        """Without vocab.json the ids decode to nothing and the caption is
        silently empty, which is a worse failure than refusing to load."""
        (tmp_path / "ckpt").mkdir()
        monkeypatch.setenv("SATQUERY_CAPTION", str(tmp_path / "ckpt"))
        available, reason = caption_mod.is_available()
        assert available is False
        assert "vocab.json" in reason


class TestRegistryFallback:
    @pytest.mark.parametrize(
        "name", ["caption_v1", "grounding_v1", "change_caption_v1"]
    )
    def test_registry_holds_a_working_tool_either_way(self, name):
        tool = REGISTRY[name]
        assert hasattr(tool, "run") and hasattr(tool, "run_batch")

    def test_no_env_means_stubs_so_ci_stays_green(self, monkeypatch):
        for env in (
            "SATQUERY_CAPTION", "SATQUERY_GROUNDING", "SATQUERY_CHANGE_CAPTION"
        ):
            monkeypatch.delenv(env, raising=False)
        from satquery.tools.stubs import (
            _caption_tool, _change_caption_tool, _grounding_tool,
        )
        for factory in (_caption_tool, _grounding_tool, _change_caption_tool):
            assert type(factory()).__module__.endswith("stubs")


@pytest.mark.skipif(
    not has_ckpt("checkpoints/caption"), reason="no caption checkpoint"
)
class TestCaptionTool:
    def test_it_produces_a_non_empty_caption(self, monkeypatch, msi_6band):
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_CAPTION", "checkpoints/caption")
        result = caption_mod.CaptionTool().run(ingest([msi_6band]), {})
        assert result.payload.data["caption"].strip()
        assert result.confidence_method == "logprob"
        assert 0.0 <= result.confidence <= 1.0


@pytest.mark.skipif(
    not has_ckpt("checkpoints/grounding"), reason="no grounding checkpoint"
)
class TestGroundingTool:
    def test_the_box_is_inside_the_image(self, monkeypatch, msi_6band):
        """The head sigmoids centre and size, so this must hold by
        construction - and the pixel conversion must use the ACTUAL image
        size, not the 224px model input, or every box on a non-square scene
        is wrong."""
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_GROUNDING", "checkpoints/grounding")
        manifest = ingest([msi_6band])
        result = grounding_mod.GroundingTool().run(
            manifest, {"_query": "show me the roads"}
        )
        box = result.payload.data["bounding_boxes"][0]
        width, height = manifest.images[0].width, manifest.images[0].height
        assert 0 <= box["x0"] <= box["x1"] <= width
        assert 0 <= box["y0"] <= box["y1"] <= height

    def test_it_warns_about_its_own_weakness(self, monkeypatch, msi_6band):
        """Acc@0.5 is 0.0762. A box from this model must not be presented
        as a finding without that travelling with it."""
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_GROUNDING", "checkpoints/grounding")
        result = grounding_mod.GroundingTool().run(
            ingest([msi_6band]), {"_query": "roads"}
        )
        assert any("0.0762" in w for w in result.warnings)


@pytest.mark.skipif(
    not has_ckpt("checkpoints/change_caption"),
    reason="no change_caption checkpoint",
)
class TestChangeCaptionTool:
    def test_a_missing_mask_is_warned_about_not_hidden(
        self, monkeypatch, msi_6band, msi_6band_t2
    ):
        """It is a MASK-conditioned model. Feeding zeros runs it outside its
        training distribution, so that has to be stated."""
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_CHANGE_CAPTION", "checkpoints/change_caption")
        monkeypatch.delenv("SATQUERY_CHANGE_MASK", raising=False)
        result = change_caption_mod.ChangeCaptionTool().run(
            ingest([msi_6band, msi_6band_t2]), {}
        )
        assert result.payload.data["mask_conditioned"] is False
        assert any("zero mask" in w for w in result.warnings)

    def test_a_single_image_is_rejected(self, monkeypatch, msi_6band):
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_CHANGE_CAPTION", "checkpoints/change_caption")
        with pytest.raises(ValueError, match="bi-temporal"):
            change_caption_mod.ChangeCaptionTool().run(ingest([msi_6band]), {})

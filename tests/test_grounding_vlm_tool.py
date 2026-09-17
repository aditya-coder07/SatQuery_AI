"""The VLM grounding path of `satquery/tools/grounding.py`, with a fake model.

Pins the coordinate chain (model frame -> preview -> source pixels) and the
no-box abstention, without loading Qwen.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from satquery.tools import grounding as g  # noqa: E402
from satquery.tools import rs_vqa  # noqa: E402


class _Batch(dict):
    def to(self, device):
        return self


class _Processor:
    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=False):
        return "prompt"

    def __call__(self, text, images, return_tensors="pt"):
        img = images[0]
        # Pretend the processor resized to the nearest multiple of 28.
        gh, gw = round(img.height / 28) * 2, round(img.width / 28) * 2
        return _Batch(input_ids=torch.zeros(1, 5, dtype=torch.long),
                      image_grid_thw=torch.tensor([[1, gh, gw]]))

    def decode(self, ids, skip_special_tokens=True):
        return self.reply


class _Generated:
    def __init__(self):
        self.sequences = torch.zeros(1, 9, dtype=torch.long)
        self.scores = [torch.tensor([[0.0, 3.0]]) for _ in range(4)]


class _Model:
    device = "cpu"

    def generate(self, **kw):
        return _Generated()


class _Handle:
    DEFAULT_ADAPTER = "rs_vqa"

    def __init__(self, reply):
        self.processor = _Processor()
        self.processor.reply = reply
        self.model = _Model()
        self.torch = torch
        self.loaded = {}

    def ensure_adapter(self, name, path):
        self.loaded[name] = str(path)

    @contextmanager
    def using(self, name):
        yield self.model


def _manifest(width, height, tmp_path):
    from satquery.contracts.input_manifest import ImageMeta, IngestMode, InputManifest

    return InputManifest(
        run_id="run_ground", ingest_mode=IngestMode.OPERATIONAL, config="SINGLE", checks=[],
        index_availability={}, artifacts={}, blocking_failures=[],
        images=[ImageMeta(
            role="single", path=tmp_path / "scene.tif", modality="OPTICAL", modality_evidence={},
            crs="EPSG:32643", gsd_m=1.0, width=width, height=height, bands=["RED", "GREEN", "BLUE"],
            band_presence=[True, True, True], dtype="uint8", effective_bits=8, acquisition_dt=None,
            nodata_pct=0.0, cloud_pct=0.0, sensor_guess=None, polarisations=None, look_count_est=None,
        )],
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    monkeypatch.setenv(g.ENV_VLM_ADAPTER, str(adapter))
    monkeypatch.setenv(rs_vqa.ENV_BASE, str(tmp_path))
    monkeypatch.setenv(rs_vqa.ENV_ADAPTER, str(adapter))
    return adapter


def _run(monkeypatch, reply, width, height, tmp_path):
    handle = _Handle(reply)
    monkeypatch.setattr(rs_vqa._ModelHandle, "get", classmethod(lambda cls, b, a: handle))
    from PIL import Image

    def preview(meta, max_edge=1024):  # aspect-preserving, like to_rgb_preview
        scale = min(1.0, max_edge / max(width, height))
        return Image.new("RGB", (round(width * scale), round(height * scale))), {}

    monkeypatch.setattr(g, "to_rgb_preview", preview)
    result = g.GroundingTool().run(_manifest(width, height, tmp_path), {"_query": "the red ship"})
    return result, handle


def test_box_is_mapped_from_model_frame_to_source_pixels(monkeypatch, env, tmp_path):
    # Source 800x800 -> preview 800x800 -> model frame 812x812 (29 patches x 28).
    reply = json.dumps([{"bbox_2d": [406, 203, 812, 609], "label": "ship"}])
    result, handle = _run(monkeypatch, reply, 800, 800, tmp_path)
    box = result.payload.data["bounding_boxes"][0]
    assert box["x0"] == pytest.approx(400, abs=0.5) and box["y0"] == pytest.approx(200, abs=0.5)
    assert box["x1"] == pytest.approx(800, abs=0.5) and box["y1"] == pytest.approx(600, abs=0.5)
    assert result.confidence_method == "logprob" and 0.9 < result.confidence <= 1.0
    assert handle.loaded == {g.VLM_ADAPTER_NAME: str(env)}


def test_large_source_is_scaled_back_from_the_preview(monkeypatch, env, tmp_path):
    # Source 2048x1024 is previewed at max_edge 1024 -> 1024x512 -> frame 1036x504.
    reply = json.dumps([{"bbox_2d": [0, 0, 518, 252], "label": "ship"}])
    result, _ = _run(monkeypatch, reply, 2048, 1024, tmp_path)
    box = result.payload.data["bounding_boxes"][0]
    assert box["x1"] == pytest.approx(1024, abs=1) and box["y1"] == pytest.approx(512, abs=1)


def test_no_box_in_reply_asserts_nothing(monkeypatch, env, tmp_path):
    result, _ = _run(monkeypatch, "I cannot find it.", 800, 800, tmp_path)
    assert result.payload.data["bounding_boxes"] == []
    assert result.confidence == 0.0 and result.confidence_method == "no_assertion"


def test_availability_prefers_vlm_when_configured(env, monkeypatch):
    ok, reason = g.is_available()
    assert "VLM" in reason or "not installed" in reason or "safetensors" in reason.lower()
    monkeypatch.delenv(g.ENV_VLM_ADAPTER)
    monkeypatch.delenv(g.ENV_CHECKPOINT, raising=False)
    assert g.is_available() == (False, f"{g.ENV_CHECKPOINT} is not set")


def test_pixel_budget_env_is_read(monkeypatch):
    from satquery.tools import grounding as g

    monkeypatch.delenv(g.ENV_VLM_MIN_PIXELS, raising=False)
    monkeypatch.delenv(g.ENV_VLM_MAX_PIXELS, raising=False)
    assert g.pixel_budget() == (None, None)
    monkeypatch.setenv(g.ENV_VLM_MIN_PIXELS, "1048576")
    assert g.pixel_budget() == (1048576, None)
    monkeypatch.setenv(g.ENV_VLM_MAX_PIXELS, " ")
    assert g.pixel_budget() == (1048576, None)

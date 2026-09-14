"""The VLM adapter paths of the caption and change-caption tools, with a
fake shared handle: the trained prompt is used verbatim, both images reach
the model for change captioning, the adapter is attached under its own
name, and an empty reply is never returned as an empty caption."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from satquery.tools import caption as cap  # noqa: E402
from satquery.tools import change_caption as cc  # noqa: E402
from satquery.tools import rs_vqa, vlm_text  # noqa: E402


class _Batch(dict):
    def to(self, device):
        return self


class _Processor:
    def __init__(self):
        self.seen = []

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=False):
        self.seen.append(chat)
        return "prompt"

    def __call__(self, text, images, return_tensors="pt"):
        self.n_images = len(images)
        return _Batch(input_ids=torch.zeros(1, 5, dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return self.reply


class _Generated:
    sequences = torch.zeros(1, 9, dtype=torch.long)
    scores = [torch.tensor([[0.0, 3.0]]) for _ in range(4)]


class _Model:
    device = "cpu"

    def generate(self, **kw):
        return _Generated()


class _Handle:
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


def _manifest(n_images, tmp_path):
    from satquery.contracts.input_manifest import ImageMeta, IngestMode, InputManifest

    metas = [ImageMeta(
        role="t1" if i == 0 else "t2", path=tmp_path / f"scene{i}.tif", modality="OPTICAL", modality_evidence={},
        crs="EPSG:32643", gsd_m=1.0, width=256, height=256, bands=["RED", "GREEN", "BLUE"],
        band_presence=[True, True, True], dtype="uint8", effective_bits=8, acquisition_dt=None,
        nodata_pct=0.0, cloud_pct=0.0, sensor_guess=None, polarisations=None, look_count_est=None,
    ) for i in range(n_images)]
    return InputManifest(run_id="run_cap", ingest_mode=IngestMode.OPERATIONAL,
                         config="SINGLE" if n_images == 1 else "CROSSMODAL_PAIR", checks=[],
                         index_availability={}, artifacts={}, blocking_failures=[], images=metas)


@pytest.fixture
def env(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    monkeypatch.setenv(cap.ENV_VLM_ADAPTER, str(adapter))
    monkeypatch.setenv(cc.ENV_VLM_ADAPTER, str(adapter))
    monkeypatch.setenv(rs_vqa.ENV_BASE, str(tmp_path))
    monkeypatch.setenv(rs_vqa.ENV_ADAPTER, str(adapter))
    from PIL import Image

    for mod in (cap, cc):
        monkeypatch.setattr(mod, "to_rgb_preview", lambda meta, max_edge=1024: (Image.new("RGB", (256, 256)), {}))
    return adapter


def _stub(monkeypatch, reply):
    handle = _Handle(reply)
    monkeypatch.setattr(rs_vqa._ModelHandle, "get", classmethod(lambda cls, b, a: handle))
    return handle


def test_caption_uses_trained_prompt_and_adapter(monkeypatch, env, tmp_path):
    handle = _stub(monkeypatch, "A large airport with green grass beside it.")
    result = cap.CaptionTool().run(_manifest(1, tmp_path), {})
    assert result.payload.data["caption"].startswith("A large airport")
    assert result.confidence_method == "logprob" and 0.9 < result.confidence <= 1.0
    assert handle.loaded == {cap.VLM_ADAPTER_NAME: str(env)}
    chat = handle.processor.seen[0]
    assert chat[1]["content"][-1]["text"] == vlm_text.CAPTION_QUESTION
    assert handle.processor.n_images == 1 and result.version == "1.1.0"


def test_change_caption_sends_both_images_no_mask(monkeypatch, env, tmp_path):
    handle = _stub(monkeypatch, "a road has been built in the middle of the area .")
    result = cc.ChangeCaptionTool().run(_manifest(2, tmp_path), {})
    assert handle.processor.n_images == 2
    assert result.payload.data["mask_source"].startswith("none")
    assert handle.processor.seen[0][1]["content"][-1]["text"] == vlm_text.CHANGE_QUESTION
    assert handle.loaded == {cc.VLM_ADAPTER_NAME: str(env)}


def test_empty_reply_is_named_not_blank(monkeypatch, env, tmp_path):
    _stub(monkeypatch, "   ")
    result = cap.CaptionTool().run(_manifest(1, tmp_path), {})
    assert result.payload.data["caption"] and "no tokens" in result.warnings[0]


def test_availability_reports_vlm_path(env):
    ok, reason = cap.is_available()
    assert ok or "safetensors" in reason.lower() or "not installed" in reason

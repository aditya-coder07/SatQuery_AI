"""The landcover tool accepts a v3 weights FILE (`best.pt`) with
band_stats.json beside it, and loads the SSL4EO trunk through the deployed
`_Handle` path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from satquery.tools import landcover as lc  # noqa: E402
from training.v3.landcover import build_landcover_v3  # noqa: E402


def test_v3_file_checkpoint_is_available_and_loads(tmp_path, monkeypatch):
    model = build_landcover_v3(pretrained=False)
    ck = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 3, "extra": {"arch": "v3", "pretrained": False}}, ck)
    (tmp_path / "band_stats.json").write_text(json.dumps({"mean": [0.1] * 12, "std": [0.05] * 12}), encoding="utf-8")
    monkeypatch.setenv(lc.ENV_CHECKPOINT, str(ck))
    ok, reason = lc.is_available()
    assert ok, reason
    mean, std, scale = lc.load_band_stats(ck)
    assert mean.shape == (12,) and scale == 10000.0
    handle = lc._Handle(ck)
    assert handle.path == str(ck) and handle.has_gsd is False
    x = torch.zeros(1, 12, 120, 120, device=handle.device)
    with torch.no_grad():
        out = handle.model(x, torch.ones(1, 12, device=handle.device))
    assert out.shape == (1, 19)


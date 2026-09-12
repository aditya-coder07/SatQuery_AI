"""The v3 (per-pixel triad) path of the fusion tool, with a stand-in model."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from satquery.contracts.input_manifest import ImageMeta, IngestMode, InputManifest  # noqa: E402
from satquery.tools import optsar_fusion as f  # noqa: E402


class _TriadV3(torch.nn.Module):
    """Optical says class 0 everywhere; SAR says class 3 (water) on the left
    half; fused sides with SAR on that half."""

    def forward(self, o, s, drop=None):
        b, _, h, w = o.shape
        lo = torch.zeros(b, 7, h, w)
        lo[:, 0] = 5.0
        ls = torch.zeros(b, 7, h, w)
        ls[:, 3, :, : w // 2] = 5.0
        ls[:, 0, :, w // 2:] = 5.0
        lf = ls.clone() if drop != "sar" else lo.clone()
        return lo, ls, lf


def _tif(path, bands, size=32):
    data = (np.random.default_rng(0).random((bands, size, size)) * 255).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands, dtype="uint8") as dst:
        dst.write(data)
    return path


def _meta(path, modality, role, bands):
    return ImageMeta(role=role, path=path, modality=modality, modality_evidence={}, crs="EPSG:32643",
                     gsd_m=5.0, width=32, height=32, bands=bands, band_presence=[True] * len(bands),
                     dtype="uint8", effective_bits=8, acquisition_dt=None, nodata_pct=0.0, cloud_pct=0.0,
                     sensor_guess=None, polarisations=None, look_count_est=None)


def test_v3_path_reports_fractions_and_complementarity(tmp_path, monkeypatch):
    opt = _tif(tmp_path / "opt.tif", 4)
    sar = _tif(tmp_path / "sar.tif", 1)
    manifest = InputManifest(
        run_id="r", ingest_mode=IngestMode.OPERATIONAL, config="CROSSMODAL_PAIR", checks=[],
        index_availability={}, artifacts={}, blocking_failures=[],
        images=[_meta(opt, "OPTICAL", "optical", ["RED", "GREEN", "BLUE", "NIR"]),
                _meta(sar, "SAR", "sar", ["VV"])],
    )

    class Handle:
        arch = "v3"
        model = _TriadV3().eval()
        device = "cpu"
        path = "best.pt"
        torch = torch

    monkeypatch.setenv(f.ENV_CHECKPOINT, str(tmp_path))
    monkeypatch.setattr(f._Handle, "get", classmethod(lambda cls, p: Handle()))
    result = f.OptSARFusionTool().run(manifest, {})
    d = result.payload.data
    assert d["mode"] == "triad_segmentation_v3"
    assert d["optical_only"]["farmland"] == 1.0
    assert d["sar_only"]["water"] == pytest.approx(0.5)
    assert d["fused"]["water"] == pytest.approx(0.5)
    assert d["fused_without_sar"]["farmland"] == 1.0
    comp = d["complementarity"]
    assert comp["pixels_revised_by_fusion"] == pytest.approx(0.5)
    assert comp["revisions_siding_with_sar"] == 1.0
    assert comp["attribution"] == {"water": "sar"}
    assert comp["modality_agreement"] == pytest.approx(0.5)
    assert result.confidence_method == "mean_asserted_probability" and result.confidence > 0.9
    assert "siding with SAR on 100%" in d["answer"]

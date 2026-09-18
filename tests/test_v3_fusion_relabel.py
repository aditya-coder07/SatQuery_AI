"""WHU-OPT-SAR re-cut: the 1-based crop, the NDWI alignment test, and the v3 triad contract."""

import json

import numpy as np
import pytest

from training.prepare import whu_opt_sar_relabel as rl
from training.prepare import whu_opt_sar_resplit as rs


def test_one_based_crop_matches_tile_grid():
    # A synthetic scene whose value encodes its (row, col) tile of origin.
    scene = np.zeros((3704, 5556), dtype="uint8")
    for r in range(1, 7):
        for c in range(1, 10):
            scene[(r - 1) * 512:r * 512, (c - 1) * 512:c * 512] = 10 * ((r * 9 + c) % 8)
    crop = scene[(3 - 1) * 512:3 * 512, (5 - 1) * 512:5 * 512]
    assert crop.shape == (512, 512)
    assert np.unique(crop // rl.LABEL_STRIDE).tolist() == [(3 * 9 + 5) % 8]
    # The old 0-based cut of the same suffix lands one tile down-right.
    old = scene[3 * 512:4 * 512, 5 * 512:6 * 512]
    assert np.unique(old // rl.LABEL_STRIDE).tolist() == [(4 * 9 + 6) % 8]


def test_ndwi_gap_is_large_only_when_aligned(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    h = w = 64
    opt = np.full((4, h, w), 100, dtype="uint8")
    opt[1, :32] = 200  # green high on the top half -> NDWI high there
    opt[3, :32] = 20
    path = tmp_path / "t.tif"
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=4, dtype="uint8") as dst:
        dst.write(opt)
    aligned = np.zeros((h, w), dtype="uint8")
    aligned[:32] = rl.WATER
    misaligned = np.zeros((h, w), dtype="uint8")
    misaligned[:, :32] = rl.WATER  # left half instead of top half
    assert rl.ndwi_gap(path, aligned) > 0.5
    assert abs(rl.ndwi_gap(path, misaligned)) < 0.05


def test_scene_resplit_is_disjoint(tmp_path):
    rows = [{"id": f"S{s:02d}_{r:02d}_{c:02d}", "optical": "o", "sar": "s", "label": "l"}
            for s in range(10) for r in range(1, 3) for c in range(1, 3)]
    index = {"classes": ["a"], "splits": {"train": rows[:30], "validation": rows[30:]}}
    src, out = tmp_path / "index.json", tmp_path / "out.json"
    src.write_text(json.dumps(index))
    import sys

    argv = sys.argv
    sys.argv = ["x", "--index", str(src), "--out", str(out), "--val-fraction", "0.3"]
    try:
        assert rs.main() == 0
    finally:
        sys.argv = argv
    new = json.loads(out.read_text())
    tr = {rs.scene_of(r["id"]) for r in new["splits"]["train"]}
    va = {rs.scene_of(r["id"]) for r in new["splits"]["validation"]}
    assert tr and va and not (tr & va)
    assert len(va) == 3


def test_fusion_v3_triad_shapes_and_dropout():
    torch = pytest.importorskip("torch")
    from training.v3.optsar_fusion import build_optsar_fusion_v3

    m = build_optsar_fusion_v3(dim=16, weights=None).eval()
    o, s = torch.rand(2, 4, 64, 64), torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        lo, ls, lf = m(o, s)
        lo2, ls2, lf_nosar = m(o, s, drop="sar")
    assert lo.shape == ls.shape == lf.shape == (2, 7, 64, 64)
    # Single-modality heads are unaffected by dropping SAR in the fused stream.
    assert torch.allclose(lo, lo2) and torch.allclose(ls, ls2)


def test_landcover_v3_scratch_forward_accepts_mask_and_gsd():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from training.v3.landcover import build_landcover_v3

    m = build_landcover_v3(pretrained=False).eval()
    x = torch.rand(2, 12, 120, 120)
    mask = torch.ones(2, 12)
    mask[:, 4:] = 0
    with torch.no_grad():
        out = m(x, mask, torch.full((2,), 10.0))
    assert out.shape == (2, 19)

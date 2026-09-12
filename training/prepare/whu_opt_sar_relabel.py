"""Re-cut the WHU-OPT-SAR label tiles so they align with the imagery tiles.

The bug
-------
`whu_opt_sar.py` cropped each label tile from the full-scene mask at
`row*512, col*512` with the tile's `_RR_CC` suffix taken as 0-based 512-px
grid indices. They are **1-based**: the mirror's indices run 1..6 x 1..9 per
scene (the 512-px tiles that fit inside 3704 x 5556, edge remainder
dropped). So every label tile was cut one tile down and one tile right of
the imagery it was paired with. The audit caught it because water-labelled
pixels had the same NDWI as everything else (docs/research/dataset_audit.md,
finding F2). Six crop hypotheses were tested against the NDWI water gap:
0-based +0.007, 1-based +0.234, 1-based on a 1.2057x downscaled scene
+0.017, row/col swapped +0.032, transposed −0.001.

Consequence: every `optsar_fusion` number to date (v1 −0.006, v2 −0.030,
v3 −0.002 "complementarity gain") was measured against labels that do not
describe the pixels. Tile-level class *presence* still correlated with the
neighbouring true tile, which is why the tile-mAP numbers looked plausible.

The fix
-------
Crop `[(r-1)*512:r*512, (c-1)*512:c*512]` from the full-scene label, no
resampling. Written to
`prepared/lbl_v2/`; the old `prepared/lbl/` is left in place so the Phase 5
artefacts stay reproducible. A new index (`index_v2.json`, and the
scene-disjoint `index_v2_scene_split.json`) points at the re-cut labels.

Verification is built in: for tiles with both water and non-water pixels,
the mean NDWI gap (water minus other) is printed for the old and the new
labels. Aligned labels give a large positive gap.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

TILE = 512
GRID_COLS, GRID_ROWS = 9, 6
LABEL_STRIDE = 10
WATER = 4  # class value in the 0..7 scheme (background, farmland, city, village, water, ...)
_TILE_RE = re.compile(r"^(.+)_(\d+)_(\d+)$")


def load_full_labels(zip_path: Path) -> dict[str, np.ndarray]:
    import rasterio

    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".tif") or "(" in name:  # skip "(1)" duplicates
                continue
            with z.open(name) as fh:
                data = fh.read()
            with rasterio.open(io.BytesIO(data)) as src:
                out[Path(name).stem] = src.read(1)
    return out


def resize_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = (np.arange(h) * arr.shape[0] / h).astype(int)
    xs = (np.arange(w) * arr.shape[1] / w).astype(int)
    return arr[ys][:, xs]


def ndwi_gap(opt_path: Path, lbl: np.ndarray) -> float | None:
    import rasterio

    with rasterio.open(opt_path) as src:
        o = src.read().astype("float32")
    g, nir = o[1], o[3]
    n = (g - nir) / (g + nir + 1e-6)
    w = lbl == WATER
    if w.sum() < 500 or (~w).sum() < 500:
        return None
    return float(n[w].mean() - n[~w].mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/whu_opt_sar"))
    p.add_argument("--labels-zip", type=Path, default=None)
    p.add_argument("--check", type=int, default=200, help="tiles to verify with the NDWI test")
    args = p.parse_args()
    import rasterio

    zip_path = args.labels_zip or (args.data / "lbl_full.zip")
    full = load_full_labels(zip_path)
    print(f"full-scene labels: {len(full)} scenes, e.g. {next(iter(full.values())).shape}")
    resized = full  # 1-based crop on the native grid; no resampling (see module docstring)

    opt_dir, out_dir = args.data / "prepared" / "opt", args.data / "prepared" / "lbl_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    written, missing, old_gaps, new_gaps = 0, [], [], []
    tiles = sorted(opt_dir.glob("*.tif"))
    for i, tile in enumerate(tiles):
        m = _TILE_RE.match(tile.stem)
        scene, r, c = m.group(1), int(m.group(2)), int(m.group(3))
        if scene not in resized:
            missing.append(scene)
            continue
        crop = resized[scene][(r - 1) * TILE:r * TILE, (c - 1) * TILE:c * TILE]
        assert crop.shape == (TILE, TILE), (tile.stem, crop.shape)
        classes = (crop // LABEL_STRIDE).astype("uint8")
        with rasterio.open(out_dir / tile.name, "w", driver="GTiff", height=TILE, width=TILE,
                           count=1, dtype="uint8") as dst:
            dst.write(classes, 1)
        written += 1
        if i % max(1, len(tiles) // args.check) == 0:
            g_new = ndwi_gap(tile, classes)
            old = args.data / "prepared" / "lbl" / tile.name
            if g_new is not None and old.exists():
                with rasterio.open(old) as src:
                    g_old = ndwi_gap(tile, src.read(1))
                if g_old is not None:
                    old_gaps.append(g_old)
                    new_gaps.append(g_new)
    print(f"wrote {written} re-cut label tiles to {out_dir}; scenes without full label: {sorted(set(missing))}")
    print(f"NDWI water-minus-other gap over {len(new_gaps)} tiles: OLD labels {np.mean(old_gaps):+.3f}, "
          f"NEW labels {np.mean(new_gaps):+.3f}")

    for src_name, dst_name in (("index.json", "index_v2.json"),
                               ("index_scene_split.json", "index_v2_scene_split.json")):
        src = args.data / src_name
        if not src.exists():
            continue
        index = json.loads(src.read_text(encoding="utf-8"))
        for rows in index["splits"].values():
            for row in rows:
                row["label"] = row["label"].replace("\\", "/").replace("/prepared/lbl/", "/prepared/lbl_v2/")
        index["label_version"] = "lbl_v2 (re-cut 2026-09-12; see training/prepare/whu_opt_sar_relabel.py)"
        (args.data / dst_name).write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"wrote {args.data / dst_name}")
    report = {"written": written, "ndwi_gap_old": float(np.mean(old_gaps)) if old_gaps else None,
              "ndwi_gap_new": float(np.mean(new_gaps)) if new_gaps else None, "n_checked": len(new_gaps),
              "grid": [GRID_ROWS, GRID_COLS], "indexing": "1-based, native resolution"}
    (args.data / "relabel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

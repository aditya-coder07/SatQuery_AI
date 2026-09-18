"""Scene-disjoint re-split of the prepared WHU-OPT-SAR tiles.

`training/prepare/whu_opt_sar.py` splits tiles at random and says so:
"NOT geographic ... validation scores here are optimistic". The audit
(docs/research/dataset_audit.md, finding F1) measured how optimistic: all 36
source scenes appear on both sides. Neighbouring 512-px crops of one
5,500x3,700 scene share season, sensor pass and land-use, so a tile-random
validation set measures memorisation of the scene as much as generalisation.
The fusion question ("does SAR add anything the optical stream lacks?") is
exactly the kind of small effect that leaks.

This script writes a second index next to the original - it does not touch
`index.json` - with whole scenes held out. Tile ids are `SCENE_ROW_COL`, so
the scene is recoverable without re-reading any raster.

Usage::

    python training/prepare/whu_opt_sar_resplit.py --index data/whu_opt_sar/index.json \
        --out data/whu_opt_sar/index_scene_split.json --val-fraction 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def scene_of(tile_id: str) -> str:
    return tile_id.rsplit("_", 2)[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    index = json.loads(args.index.read_text(encoding="utf-8"))
    rows = [r for split in index["splits"].values() for r in split]
    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_scene[scene_of(r["id"])].append(r)
    scenes = sorted(by_scene)
    random.Random(args.seed).shuffle(scenes)
    n_val = max(1, int(round(len(scenes) * args.val_fraction)))
    val_scenes, train_scenes = set(scenes[:n_val]), set(scenes[n_val:])
    out = dict(index)
    out["splits"] = {
        "train": [r for s in sorted(train_scenes) for r in by_scene[s]],
        "validation": [r for s in sorted(val_scenes) for r in by_scene[s]],
    }
    out["split_method"] = (
        f"SCENE-DISJOINT: {len(train_scenes)} scenes train / {len(val_scenes)} scenes validation, "
        f"seed {args.seed}; derived from {args.index.name} without re-tiling"
    )
    out["scenes"] = {"train": sorted(train_scenes), "validation": sorted(val_scenes)}
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"train {len(out['splits']['train'])} tiles / {len(train_scenes)} scenes; "
          f"validation {len(out['splits']['validation'])} tiles / {len(val_scenes)} scenes")
    assert not (train_scenes & val_scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

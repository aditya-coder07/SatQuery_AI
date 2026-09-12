"""Prepare DIOR-RSVG from the authors' release, with the OFFICIAL splits.

Why this exists
---------------
The `data/dior_rsvg` parquet mirror this project trained on in Phase 5 holds
only the official **test** split (7,500 expressions). `train_grounding.py`
found no `train-*` shard, fell back to "all rows", and cut them 85/15 by
image - so every grounding number published so far was trained on 85% of
the official test set and scored on the other 15%. That is not leakage
inside our own split (it is grouped by image) but it makes the number
incomparable to every published DIOR-RSVG result, and it means the model
card's "all ~38k expressions" was wrong: `n_train` was 6,359.

This script builds the dataset the way the RSVG paper does
(ZhanYang-nwpu/RSVG-pytorch `data_loader.py`): the sorted list of XML files
is flattened object-by-object, and `train.txt` / `val.txt` / `test.txt`
hold indices into that flat list. 26,991 / 3,829 / 7,500 = 38,320 over
17,402 images.

Licence: CC-BY-NC-4.0 (research only). Recorded in docs/research/licensing.md.

Output (unified instruction format, docs/research/reproducibility.md §6)::

    data/dior_rsvg_official/
      JPEGImages/*.jpg           extracted once
      manifests/{train,val,test}.jsonl
      manifests/stats.json       counts, overlap, validator report, hashes

Each row::

    {"id": "dior_rsvg:00001:0", "image": "JPEGImages/00001.jpg",
     "modality": "rgb", "task": "<GROUNDING>", "question": "...",
     "target": {"bbox_xyxy": [x1, y1, x2, y2], "category": "golffield",
                "width": 800, "height": 800},
     "split": "train", "source": "DIOR-RSVG", "license": "CC-BY-NC-4.0",
     "synthetic": false}

Usage::

    python training/prepare/dior_rsvg_official.py \
        --raw data/raw/dior_rsvg_official --out data/dior_rsvg_official
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")
LICENSE = "CC-BY-NC-4.0"
SOURCE = "DIOR-RSVG"


def extract(raw: Path, out: Path) -> tuple[Path, Path]:
    ann_dir, img_dir = out / "Annotations", out / "JPEGImages"
    if not ann_dir.is_dir():
        with zipfile.ZipFile(raw / "Annotations.zip") as z:
            z.extractall(out)
    if not img_dir.is_dir() or not any(img_dir.iterdir()):
        with zipfile.ZipFile(raw / "JPEGImages.zip") as z:
            z.extractall(out)
    return ann_dir, img_dir


def parse_objects(ann_dir: Path) -> list[dict]:
    """Flatten every XML in sorted order, object by object - the paper's index."""
    flat: list[dict] = []
    for xml_path in sorted(ann_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")
        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
        for k, obj in enumerate(root.findall("object")):
            box = obj.find("bndbox")
            flat.append({
                "id": f"dior_rsvg:{xml_path.stem}:{k}",
                "image": f"JPEGImages/{filename}",
                "modality": "rgb",
                "task": "<GROUNDING>",
                "question": (obj.findtext("description") or "").strip(),
                "target": {
                    "bbox_xyxy": [int(box.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax")],
                    "category": (obj.findtext("name") or "").strip(),
                    "width": width, "height": height,
                },
                "source": SOURCE, "license": LICENSE, "synthetic": False,
            })
    return flat


def read_index(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def validate(rows: list[dict], img_dir: Path) -> dict:
    """Reject rows that cannot be trained on; report, never silently drop."""
    bad: list[dict] = []
    for r in rows:
        x1, y1, x2, y2 = r["target"]["bbox_xyxy"]
        w, h = r["target"]["width"], r["target"]["height"]
        reasons = []
        if not r["question"]:
            reasons.append("empty_phrase")
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            reasons.append("box_out_of_bounds_or_degenerate")
        if not (img_dir.parent / r["image"]).is_file():
            reasons.append("missing_image")
        if reasons:
            bad.append({"id": r["id"], "reasons": reasons})
    return {"n": len(rows), "n_bad": len(bad), "bad": bad[:200]}


def sha256_text(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ann_dir, img_dir = extract(args.raw, args.out)
    flat = parse_objects(ann_dir)
    print(f"flattened objects: {len(flat)} from {len(list(ann_dir.glob('*.xml')))} xml files")

    indices = {s: read_index(args.raw / f"{s}.txt") for s in SPLITS}
    all_idx = [i for s in SPLITS for i in indices[s]]
    if len(set(all_idx)) != len(all_idx):
        print("ERROR: split index files overlap", file=sys.stderr)
        return 1
    if max(all_idx) >= len(flat):
        print(f"ERROR: index {max(all_idx)} beyond {len(flat)} objects", file=sys.stderr)
        return 1

    man_dir = args.out / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"source": SOURCE, "license": LICENSE, "n_images": len(list(ann_dir.glob("*.xml"))),
                   "n_objects": len(flat), "splits": {}, "hashes": {}}
    images_by_split: dict[str, set[str]] = {}
    for s in SPLITS:
        rows = [dict(flat[i], split=s) for i in indices[s]]
        report = validate(rows, img_dir)
        lines = [json.dumps(r, sort_keys=True) for r in rows]
        (man_dir / f"{s}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        images_by_split[s] = {r["image"] for r in rows}
        cats = Counter(r["target"]["category"] for r in rows)
        stats["splits"][s] = {
            "n": len(rows), "n_images": len(images_by_split[s]),
            "validator": {"n_bad": report["n_bad"], "bad": report["bad"]},
            "categories": dict(sorted(cats.items())),
        }
        stats["hashes"][f"{s}.jsonl"] = sha256_text(lines)
        print(f"{s}: {len(rows)} expressions over {len(images_by_split[s])} images, "
              f"{report['n_bad']} rejected by validator")

    # The official protocol indexes objects, not images, so one image can
    # carry a train expression and a test expression. Measure it; do not
    # "fix" it - changing the split would break comparability with every
    # published number.
    stats["image_overlap"] = {
        "train_test": len(images_by_split["train"] & images_by_split["test"]),
        "train_val": len(images_by_split["train"] & images_by_split["val"]),
        "val_test": len(images_by_split["val"] & images_by_split["test"]),
        "note": "official object-level split; image overlap is a property of the benchmark",
    }
    print(f"image overlap train/test: {stats['image_overlap']['train_test']}")
    (man_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

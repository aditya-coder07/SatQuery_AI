"""Index WHU-OPT-SAR for the Stage A2 resolution bridge (task 2.2).

WHU-OPT-SAR is ~5 m co-registered optical + SAR with land-cover labels. It
sits between Sentinel-2 (10 m, where Track A was trained) and Cartosat-2E
(1.6 m, the evaluation sensor), which is exactly the gap the cross-sensor test
identified as the dominant transfer problem: vegetation agreement collapsed
from +0.476 at 10 m to -0.135 at native 1.6 m, while band adaptation cost far
less. Stage A2 was a plan assumption; that measurement made it the priority.

Layouts differ between mirrors, so discovery is by directory name rather than
a hardcoded tree, and the script reports what it found instead of failing
silently on an unexpected structure.

Usage:
    python training/prepare/whu_opt_sar.py --src data/whu_opt_sar/extracted \
        --out data/whu_opt_sar/index.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# WHU-OPT-SAR's published 7-class land-cover nomenclature (background = 0).
CLASSES = [
    "background", "farmland", "city", "village", "water", "forest", "road",
    "others",
]

OPTICAL_HINTS = ("optical", "opt", "rgb", "image")
SAR_HINTS = ("sar",)
LABEL_HINTS = ("lbl", "label", "mask", "gt", "annotation")

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg"}


def _classify_dir(path: Path) -> str | None:
    name = path.name.lower()
    # Labels first: a directory called "sar_label" is a label directory.
    if any(h in name for h in LABEL_HINTS):
        return "label"
    if any(h in name for h in SAR_HINTS):
        return "sar"
    if any(h in name for h in OPTICAL_HINTS):
        return "optical"
    return None


def discover(src: Path) -> dict[str, dict[str, Path]]:
    """Map stem -> {optical, sar, label} across whatever layout is present."""
    found: dict[str, dict[str, Path]] = {}
    for directory in sorted(p for p in src.rglob("*") if p.is_dir()):
        kind = _classify_dir(directory)
        if kind is None:
            continue
        for image in directory.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                found.setdefault(image.stem, {})[kind] = image
    return found


def build_index(src: Path, val_fraction: float, seed: int) -> dict:
    found = discover(src)
    complete = {
        stem: parts for stem, parts in found.items()
        if "optical" in parts and "label" in parts
    }

    # A geographic split is not possible without tile coordinates, so a
    # deterministic random split is used and labelled as such. WHU-OPT-SAR
    # tiles come from a small number of large scenes, so neighbouring tiles
    # are correlated and this split is optimistic - stated rather than hidden.
    rng = random.Random(seed)
    stems = sorted(complete)
    rng.shuffle(stems)
    cut = int(len(stems) * val_fraction)
    splits = {"validation": stems[:cut], "train": stems[cut:]}

    return {
        "classes": CLASSES,
        "split_method": (
            "deterministic random by tile; NOT geographic. WHU-OPT-SAR tiles "
            "derive from a few large scenes, so adjacent tiles are correlated "
            "and validation scores here are optimistic."
        ),
        "splits": {
            name: [
                {
                    "id": stem,
                    "optical": str(complete[stem]["optical"]),
                    "sar": str(complete[stem]["sar"]) if "sar" in complete[stem] else None,
                    "label": str(complete[stem]["label"]),
                }
                for stem in members
            ]
            for name, members in splits.items()
        },
        "n_with_sar": sum(1 for s in complete.values() if "sar" in s),
        "n_incomplete": len(found) - len(complete),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.src.is_dir():
        print(f"source not found: {args.src}", file=sys.stderr)
        return 1

    index = build_index(args.src, args.val_fraction, args.seed)
    counts = {k: len(v) for k, v in index["splits"].items()}

    if not any(counts.values()):
        print("No optical/label pairs found. Directories seen:", file=sys.stderr)
        for d in sorted({p.parent.name for p in args.src.rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES})[:20]:
            print(f"    {d}", file=sys.stderr)
        print("Extend OPTICAL_HINTS / SAR_HINTS / LABEL_HINTS to match.",
              file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index), encoding="utf-8")
    print(f"Wrote {args.out}")
    for name, n in counts.items():
        print(f"  {name:<11} {n:>6} tiles")
    print(f"  with paired SAR : {index['n_with_sar']}")
    if index["n_incomplete"]:
        print(f"  incomplete (skipped): {index['n_incomplete']}")
    print(f"  split: {index['split_method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Index Landsat-SCD (CC BY 4.0) as the licensed semantic-change benchmark.

Why
---
SECOND, behind `change_vqa_v1`'s semantic head, states no licence at all, so
no number measured on it can be published without a caveat and no weights
trained on it can be released (`docs/research/licensing.md`). Landsat-SCD
(Yuan et al. 2022, figshare 10.6084/m9.figshare.19946135, **CC BY 4.0**) is
the replacement: 30 m Landsat pairs over Tumushuke, 1990-2020, four land
cover classes (farmland, desert, building, water) and a single label map per
pair whose value is the **semantic change type** - 0 = no change, 1..9 = one
ordered land-cover transition each. The figshare release does not ship the
code -> (from, to) table, so this project keeps the dataset's own 10-way
label as the target and reports change-type mIoU, binary-change IoU/F1 and
SeK on it. That is Category B against papers that decode per-date maps,
and is stated as such.

Layout of the release: `A/`, `B/`, `label/` with 8,468 pairs = 2,385
originals + 869 originals x 7 augmented copies (`CropResize0-2`,
`Zhedang1-2` = occlusion, `rotate90/180`). A copy carries the same scene as
its original, so **the split is by original**: originals are cut 3:1:1
(the paper's ratio) deterministically, augmented copies follow their
original and are used for training only.

Output: `data/landsat_scd/index.json` in the `change_mask` index shape
(`splits -> [{id, a, b, label}]`) plus `manifests/stats.json`.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

N_CLASSES = 10  # 0 = no change, 1..9 = change types as released
AUG_RE = re.compile(r"(CropResize\d+|Zhedang\d+|rotate\d+)$")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=Path("data/landsat_scd/Landsat-SCD_dataset"))
    p.add_argument("--out", type=Path, default=Path("data/landsat_scd/index.json"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    names = sorted(q.stem for q in (args.root / "label").glob("*.png"))
    originals = [n for n in names if not AUG_RE.search(n)]
    copies = collections.defaultdict(list)
    for n in names:
        m = AUG_RE.search(n)
        if m:
            copies[n[:m.start()]].append(n)
    rng = random.Random(args.seed)
    shuffled = originals[:]
    rng.shuffle(shuffled)
    n_val = n_test = len(shuffled) // 5
    split_of = {}
    for i, n in enumerate(shuffled):
        split_of[n] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    rel = lambda sub, n: f"data/landsat_scd/Landsat-SCD_dataset/{sub}/{n}.png"  # noqa: E731
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for n in originals:
        s = split_of[n]
        splits[s].append({"id": n, "a": rel("A", n), "b": rel("B", n), "label": rel("label", n), "augmented": False})
        if s == "train":
            for c in copies.get(n, []):
                splits[s].append({"id": c, "a": rel("A", c), "b": rel("B", c), "label": rel("label", c),
                                  "augmented": True, "original": n})
    orphans = [c for base, cs in copies.items() if base not in split_of for c in cs]
    index = {
        "dataset": "Landsat-SCD", "license": "CC BY 4.0",
        "source": "https://doi.org/10.6084/m9.figshare.19946135.v1",
        "classes": ["no_change"] + [f"change_type_{k}" for k in range(1, N_CLASSES)],
        "split_method": (f"originals cut 3:1:1 by seed {args.seed}; augmented copies follow their "
                         f"original and are train-only; {len(orphans)} copies without an original dropped"),
        "splits": splits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index), encoding="utf-8")
    man = args.out.parent / "manifests"
    man.mkdir(exist_ok=True)
    stats = {"n_originals": len(originals), "n_copies": sum(len(v) for v in copies.values()),
             "n_orphan_copies": len(orphans),
             "splits": {k: {"n": len(v), "n_original": sum(1 for r in v if not r["augmented"])}
                        for k, v in splits.items()},
             "index_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    (man / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

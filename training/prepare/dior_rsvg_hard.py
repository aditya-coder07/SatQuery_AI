"""Hard-negative subset of the DIOR-RSVG official TRAIN split for grounding.

The official-test failure analysis (`docs/research/failure_analysis.md`)
shows `wrong_object` as the least-improved class after fine-tuning: the
model finds *an* object of the right category but not the one the phrase
picks out. The training rows that teach exactly that are the expressions
whose image holds two or more distinct objects of the same category, so
the phrase has to be read, not just the category. This preparer writes
those rows to `manifests/train_hard.jsonl` (a subset of `train.jsonl`,
same format, never anything from val/test) with `metadata.n_same_category`
so a trainer can oversample them (`--train-weight`).

Distinct objects are counted by box (IoU < 0.5 between boxes of the same
category on the same image); several expressions for one object count
once.

Usage::

    python training/prepare/dior_rsvg_hard.py --root data/dior_rsvg_official
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def distinct_boxes(boxes: list[list[float]]) -> int:
    kept: list[list[float]] = []
    for b in boxes:
        if all(iou(b, k) < 0.5 for k in kept):
            kept.append(b)
    return len(kept)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=Path("data/dior_rsvg_official"))
    p.add_argument("--split", default="train", choices=["train"], help="hard negatives come from TRAIN only")
    args = p.parse_args()
    src = args.root / "manifests" / f"{args.split}.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_img_cat: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for r in rows:
        cat = r["target"].get("category") or r["target"].get("label") or ""
        by_img_cat[(r["image"], cat)].append(r["target"]["bbox_xyxy"])
    n_distinct = {k: distinct_boxes(v) for k, v in by_img_cat.items()}
    hard = []
    for r in rows:
        cat = r["target"].get("category") or r["target"].get("label") or ""
        n = n_distinct[(r["image"], cat)]
        if n >= 2:
            h = dict(r)
            h["metadata"] = {**(r.get("metadata") or {}), "n_same_category": n, "hard_negative": True}
            hard.append(h)
    out = args.root / "manifests" / f"{args.split}_hard.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for h in hard:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    stats_path = args.root / "manifests" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats.setdefault("hard_negatives", {})[args.split] = {
        "n_rows": len(hard), "n_images": len({h["image"] for h in hard}), "of_rows": len(rows),
        "rule": "expression whose image has >=2 distinct (IoU<0.5) objects of the same category",
        "sha256": digest, "file": out.name}
    stats.setdefault("hashes", {})[out.name] = digest
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats["hard_negatives"][args.split], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

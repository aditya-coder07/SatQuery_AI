"""Prepare VRSBench (Li et al. 2024, CC-BY-4.0) as unified manifests for
captioning, VQA and grounding.

Source: `xiang709/VRSBench` on the Hub - 29,614 512-px crops of DOTA-v2
(`P####_####.png`) and DIOR (`#####_####.png`) images with GPT-4V-drafted,
human-verified captions, referring expressions and QA pairs. Train rows
come from the per-image `Annotations_train/*.json` (precise boxes from
`obj_corner`, normalised OBB corners → HBB), not from the chat-format
`VRSBench_train.json` whose boxes are rounded to 1/100 of the image. The
three `VRSBench_EVAL_*.json` files are the official evaluation split.

Leakage handled here, because the DIOR half of VRSBench is cut from the
same images DIOR-RSVG uses:

* every train row whose DIOR source image is in the DIOR-RSVG official
  **val or test** split is quarantined (`manifests/quarantine_train.jsonl`,
  never sampled) - 1,320 test + 724 val source images, ≈ 11.8k rows;
* every eval row records `metadata.dior_source_in_rsvg_train`, so a model
  trained on DIOR-RSVG train can be scored on the DOTA-only / clean subset
  and the whole split side by side.

Output (`--out data/vrsbench`)::

    images/{train,val}/*.png
    manifests/train.jsonl                  CAPTION + VQA + GROUNDING rows
    manifests/train_{caption,vqa,grounding}.jsonl   the same rows, one task per file
    manifests/val_caption.jsonl, val_vqa.jsonl, val_grounding.jsonl
    manifests/quarantine_train.jsonl
    manifests/stats.json

Usage::

    python training/prepare/vrsbench.py --raw data/raw/vrsbench --out data/vrsbench \
        --dior-rsvg data/dior_rsvg_official
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

LICENSE = "CC-BY-4.0"
SOURCE = "VRSBench (xiang709/VRSBench); crops of DOTA-v2 and DIOR"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def dior_source(image_name: str) -> str | None:
    """'00002_0000.png' -> '00002' (DIOR id); DOTA crops ('P0003_0002.png') -> None."""
    stem = image_name.split(".")[0]
    return None if stem.startswith("P") else stem.split("_")[0]


def rsvg_image_ids(manifest: Path) -> set[str]:
    ids = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ids.add(re.sub(r"\D", "", Path(r["image"]).stem))
    return ids


def corner_to_hbb(corner: list[float], w: int, h: int) -> list[float]:
    xs, ys = corner[0::2], corner[1::2]
    return [min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]


def unzip(zip_path: Path, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-x", "__MACOSX/*", "-d", str(dest.parent)], check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", type=Path, default=Path("data/raw/vrsbench"))
    p.add_argument("--out", type=Path, default=Path("data/vrsbench"))
    p.add_argument("--dior-rsvg", type=Path, default=Path("data/dior_rsvg_official"))
    args = p.parse_args()
    from PIL import Image

    out = args.out
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    for split in ("train", "val"):
        unzip(args.raw / f"Images_{split}.zip", out / "images" / f"Images_{split}")
        unzip(args.raw / f"Annotations_{split}.zip", args.raw / f"Annotations_{split}")
    rsvg_val = rsvg_image_ids(args.dior_rsvg / "manifests" / "val.jsonl")
    rsvg_test = rsvg_image_ids(args.dior_rsvg / "manifests" / "test.jsonl")
    rsvg_train = rsvg_image_ids(args.dior_rsvg / "manifests" / "train.jsonl")
    blocked = rsvg_val | rsvg_test

    sizes: dict[str, tuple[int, int]] = {}

    def size_of(split: str, name: str) -> tuple[int, int]:
        key = f"{split}/{name}"
        if key not in sizes:
            with Image.open(out / "images" / f"Images_{split}" / name) as im:
                sizes[key] = im.size
        return sizes[key]

    def base_row(split: str, name: str, task: str, rid: str) -> dict:
        src = dior_source(name)
        return {"id": rid, "image": f"images/Images_{split}/{name}", "modality": "optical", "task": task,
                "split": split, "source": SOURCE, "license": LICENSE, "synthetic": False,
                "metadata": {"origin": "DIOR" if src else "DOTA-v2", "dior_source": src,
                             "annotation": "GPT-4V draft, human-verified",
                             "dior_source_in_rsvg_train": bool(src and src in rsvg_train)}}

    # --- train: per-image JSONs ------------------------------------------
    train_rows, quarantine = [], []
    counts = Counter()
    for js in sorted((args.raw / "Annotations_train").glob("*.json")):
        name = js.stem + ".png"
        if not (out / "images" / "Images_train" / name).exists():
            counts["missing_image"] += 1
            continue
        d = json.loads(js.read_text(encoding="utf-8"))
        src = dior_source(name)
        sink = quarantine if (src and src in blocked) else train_rows
        w, h = size_of("train", name)
        rows = []
        if d.get("caption"):
            r = base_row("train", name, "<CAPTION>", f"vrs_train_{js.stem}_cap")
            r.update({"question": "Describe the image in detail.", "target": [d["caption"].strip()]})
            rows.append(r)
        for o in d.get("objects", []):
            sent = (o.get("referring_sentence") or "").strip()
            if not sent or not o.get("obj_corner"):
                counts["object_without_sentence_or_box"] += 1
                continue
            box = corner_to_hbb(o["obj_corner"], w, h)
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                counts["degenerate_box"] += 1
                continue
            r = base_row("train", name, "<GROUNDING>", f"vrs_train_{js.stem}_ref{o.get('obj_id')}")
            r.update({"question": sent.rstrip("."), "target": {"bbox_xyxy": [round(v, 2) for v in box], "label": o.get("obj_cls"),
                                                             "category": o.get("obj_cls")}})
            r["metadata"].update({"obj_cls": o.get("obj_cls"), "is_unique": o.get("is_unique"), "obj_size": o.get("obj_size")})
            rows.append(r)
        for q in d.get("qa_pairs", []):
            if not q.get("question") or q.get("answer") in (None, ""):
                counts["qa_incomplete"] += 1
                continue
            r = base_row("train", name, "<VQA>", f"vrs_train_{js.stem}_q{q.get('ques_id')}")
            r.update({"question": q["question"].strip(), "target": str(q["answer"]).strip()})
            r["metadata"]["type"] = q.get("type")
            rows.append(r)
        sink.extend(rows)
    # --- eval: official files -------------------------------------------
    evals = {}
    for task, fname, tag in (("caption", "VRSBench_EVAL_Cap.json", "<CAPTION>"), ("vqa", "VRSBench_EVAL_vqa.json", "<VQA>"),
                             ("grounding", "VRSBench_EVAL_referring.json", "<GROUNDING>")):
        rows = []
        for e in json.loads((args.raw / fname).read_text(encoding="utf-8")):
            name = e["image_id"]
            if not (out / "images" / "Images_val" / name).exists():
                counts[f"missing_eval_image_{task}"] += 1
                continue
            r = base_row("val", name, tag, f"vrs_val_{task}_{Path(name).stem}_{e.get('question_id')}")
            if task == "caption":
                r.update({"question": "Describe the image in detail.", "target": [e["ground_truth"].strip()]})
            elif task == "vqa":
                r.update({"question": e["question"].strip(), "target": str(e["ground_truth"]).strip()})
                r["metadata"]["type"] = e.get("type")
            else:
                w, h = size_of("val", name)
                box = corner_to_hbb(e["obj_corner"], w, h)
                r.update({"question": e["question"].strip().rstrip("."),
                          "target": {"bbox_xyxy": [round(v, 2) for v in box], "label": e.get("obj_cls"), "category": e.get("obj_cls")}})
                r["metadata"].update({"obj_cls": e.get("obj_cls"), "is_unique": e.get("unique"), "size_group": e.get("size_group")})
            rows.append(r)
        evals[task] = rows

    def dump(name: str, rows: list[dict]) -> str:
        path = out / "manifests" / name
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return sha256_file(path)

    hashes = {"train.jsonl": dump("train.jsonl", train_rows), "quarantine_train.jsonl": dump("quarantine_train.jsonl", quarantine)}
    for task, tag in (("caption", "<CAPTION>"), ("vqa", "<VQA>"), ("grounding", "<GROUNDING>")):
        hashes[f"train_{task}.jsonl"] = dump(f"train_{task}.jsonl", [r for r in train_rows if r["task"] == tag])
    for task, rows in evals.items():
        hashes[f"val_{task}.jsonl"] = dump(f"val_{task}.jsonl", rows)
    by_task = Counter(r["task"] for r in train_rows)
    stats = {
        "source": SOURCE, "license": LICENSE, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train": {"n": len(train_rows), "by_task": dict(by_task), "n_images": len({r["image"] for r in train_rows}),
                  "n_dior_images": len({r["image"] for r in train_rows if r["metadata"]["origin"] == "DIOR"})},
        "quarantine_train": {"n": len(quarantine), "by_task": dict(Counter(r["task"] for r in quarantine)),
                             "n_images": len({r["image"] for r in quarantine}),
                             "reason": "DIOR source image is in the DIOR-RSVG official val or test split"},
        "val": {task: {"n": len(rows), "n_images": len({r["image"] for r in rows}),
                       "n_dior_source_in_rsvg_train": sum(r["metadata"]["dior_source_in_rsvg_train"] for r in rows)}
                for task, rows in evals.items()},
        "skipped": dict(counts), "hashes": hashes,
        "leakage": {"dior_rsvg_val_test_source_images_blocked": len(blocked),
                    "note": "VRSBench val rows on DIOR-RSVG train images are flagged, not removed: report the clean subset alongside"},
    }
    (out / "manifests" / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "hashes"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

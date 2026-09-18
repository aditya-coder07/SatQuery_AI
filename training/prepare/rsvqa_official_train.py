"""Unified-format manifests for the OFFICIAL RSVQA-LR train and val splits.

Why
---
Every Track B adapter so far was trained on 1,793 questions from a 2,000-row
HuggingFace redistribution of the RSVQA-LR **validation** split. The official
**train** split - 57,223 active questions over 572 images, zero image overlap
with the 100 test images - has been on disk under `data/rsvqa_lr_official`
since Phase 4 and never used for training. Every published RSVQA-LR number
in the 90-93% range trains on it. This is the single largest lever on VQA,
and it is the dataset's own protocol.

Val here is the official validation split, derived as "active questions
whose image is in neither the train nor the test image list" (the Zenodo
record ships explicit lists only for train and test). It is the selection
set; the test split is never read by any trainer.

Output: `data/rsvqa_lr_official/manifests/{train,val}.jsonl` + `stats.json`.
Row shape follows docs/research/reproducibility.md §6, task tag `<VQA>`,
`metadata.type` carrying the dataset's own question type.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

LICENSE = "CC-BY-4.0"
SOURCE = "RSVQA-LR (Zenodo 6344334)"


def load(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return next(iter(d.values())) if isinstance(d, dict) else d


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/rsvqa_lr_official"))
    args = p.parse_args()

    train_q = [q for q in load(args.data / "LR_split_train_questions.json") if q["active"]]
    test_q = [q for q in load(args.data / "LR_split_test_questions.json") if q["active"]]
    all_q = [q for q in load(args.data / "all_questions.json") if q["active"]]
    answers = {a["question_id"]: a["answer"] for a in load(args.data / "all_answers.json") if a["active"]}
    train_imgs = {q["img_id"] for q in train_q}
    test_imgs = {q["img_id"] for q in test_q}
    assert not (train_imgs & test_imgs), "train/test image overlap"
    val_q = [q for q in all_q if q["img_id"] not in train_imgs and q["img_id"] not in test_imgs]

    def rows(qs: list[dict], split: str) -> list[dict]:
        out = []
        for q in qs:
            if q["id"] not in answers:
                continue
            out.append({
                "id": f"rsvqa_lr:{q['id']}", "image": f"Images_LR/{q['img_id']}.tif",
                "modality": "rgb", "task": "<VQA>", "question": q["question"],
                "target": str(answers[q["id"]]), "split": split,
                "metadata": {"type": q["type"], "img_id": q["img_id"]},
                "source": SOURCE, "license": LICENSE, "synthetic": False,
            })
        return out

    man = args.data / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    stats = {"source": SOURCE, "license": LICENSE, "splits": {}, "hashes": {},
             "image_overlap": {"train_test": 0, "train_val": len(train_imgs & {q["img_id"] for q in val_q}),
                               "val_test": len(test_imgs & {q["img_id"] for q in val_q})}}
    for split, qs in (("train", train_q), ("val", val_q)):
        rs = rows(qs, split)
        lines = [json.dumps(r, sort_keys=True) for r in rs]
        (man / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        h = hashlib.sha256()
        for line in lines:
            h.update(line.encode()); h.update(b"\n")
        stats["hashes"][f"{split}.jsonl"] = h.hexdigest()
        stats["splits"][split] = {"n": len(rs), "n_images": len({r["metadata"]["img_id"] for r in rs}),
                                  "types": dict(collections.Counter(r["metadata"]["type"] for r in rs)),
                                  "missing_images": sum(1 for r in rs if not (args.data / r["image"]).is_file())}
        print(split, stats["splits"][split])
    print("overlap", stats["image_overlap"])
    (man / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

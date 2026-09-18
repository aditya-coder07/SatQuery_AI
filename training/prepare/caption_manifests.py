"""Unified-format manifests for RSICD (scene captions) and LEVIR-CC (change
captions), for the Qwen2.5-VL SFT trainer.

RSICD: the parquet mirror's official train / valid / test splits, images
extracted once to `data/rsicd/images/`. Train rows are one per (image,
reference) pair - five per image, the multi-caption supervision every
published captioner uses - and val/test rows are one per image carrying all
five references in `target` so corpus metrics score against all of them.

LEVIR-CC: from `data/levir_mci/index.json` (official train / val / test),
two images per row (`images: [a, b]`), five references, and `changeflag`
kept in `metadata` so the changed / unchanged halves can be reported.

Rows::

    {"id", "image" | "images", "modality": "rgb", "task": "<CAPTION>" |
     "<CHANGE_CAPTION>", "question": <fixed instruction>, "target": str
     (train) | [str, ...] (eval), "split", "source", "license", ...}
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

CAPTION_PROMPT = "Describe this remote-sensing image in one sentence."
CHANGE_PROMPT = ("These two images show the same area at two different times. "
                 "Describe what has changed between them in one sentence. "
                 "If nothing has changed, say so.")


def write(rows: list[dict], path: Path) -> str:
    lines = [json.dumps(r, sort_keys=True) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode()); h.update(b"\n")
    return h.hexdigest()


def rsicd(root: Path) -> dict:
    import pyarrow.parquet as pq

    img_dir = root / "images"
    img_dir.mkdir(exist_ok=True)
    man = root / "manifests"
    man.mkdir(exist_ok=True)
    stats = {"source": "RSICD", "license": "unstated (research use by convention)", "splits": {}, "hashes": {}}
    for split, name in (("train", "train"), ("val", "valid"), ("test", "test")):
        files = sorted(glob.glob(str(root / "**" / f"{name}-*.parquet"), recursive=True))
        rows = []
        for f in files:
            t = pq.read_table(f).to_pydict()
            for fn, caps, img in zip(t["filename"], t["captions"], t["image"]):
                stem = Path(fn).name
                out = img_dir / stem
                if not out.exists():
                    out.write_bytes(img["bytes"])
                caps = [str(c).strip() for c in caps if str(c).strip()]
                base = {"image": f"images/{stem}", "modality": "rgb", "task": "<CAPTION>",
                        "question": CAPTION_PROMPT, "split": split, "source": "RSICD",
                        "license": "unstated", "synthetic": False}
                if split == "train":
                    for k, c in enumerate(caps):
                        rows.append({"id": f"rsicd:{stem}:{k}", "target": c, **base})
                else:
                    rows.append({"id": f"rsicd:{stem}", "target": caps, **base})
        stats["hashes"][f"{split}.jsonl"] = write(rows, man / f"{split}.jsonl")
        stats["splits"][split] = {"n": len(rows), "n_images": len({r["image"] for r in rows})}
        print("rsicd", split, stats["splits"][split])
    (man / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def levir_cc(root: Path) -> dict:
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    man = root / "manifests"
    man.mkdir(exist_ok=True)
    stats = {"source": "LEVIR-CC (LEVIR-MCI)", "license": "academic only (LEVIR-CD terms)", "splits": {}, "hashes": {}}
    for split, rows_in in index["splits"].items():
        rows = []
        for r in rows_in:
            caps = [str(c).strip() for c in (r.get("captions") or [r["caption"]]) if str(c).strip()]
            # index.json paths are repo-relative (data/levir_mci/...); the
            # manifest wants them relative to the dataset root.
            def rel(path: str) -> str:
                parts = path.replace("\\", "/").split("/")
                return "/".join(parts[parts.index(root.name) + 1:]) if root.name in parts else "/".join(parts)

            base = {"images": [rel(r["a"]), rel(r["b"])], "modality": "rgb",
                    "task": "<CHANGE_CAPTION>", "question": CHANGE_PROMPT, "split": split,
                    "source": "LEVIR-CC", "license": "academic-only", "synthetic": False,
                    "metadata": {"changeflag": int(r.get("changeflag", 1))}}
            if split == "train":
                for k, c in enumerate(caps):
                    rows.append({"id": f"levircc:{r['id']}:{k}", "target": c, **base})
            else:
                rows.append({"id": f"levircc:{r['id']}", "target": caps, **base})
        stats["hashes"][f"{split}.jsonl"] = write(rows, man / f"{split}.jsonl")
        stats["splits"][split] = {"n": len(rows), "n_pairs": len(rows_in),
                                  "n_changed": sum(1 for r in rows_in if int(r.get("changeflag", 1)))}
        print("levir_cc", split, stats["splits"][split])
    (man / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rsicd", type=Path, default=Path("data/rsicd"))
    p.add_argument("--levir-mci", type=Path, default=Path("data/levir_mci"))
    p.add_argument("--only", choices=["rsicd", "levir_cc"], default=None)
    args = p.parse_args()
    if args.only in (None, "rsicd"):
        rsicd(args.rsicd)
    if args.only in (None, "levir_cc"):
        levir_cc(args.levir_mci)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

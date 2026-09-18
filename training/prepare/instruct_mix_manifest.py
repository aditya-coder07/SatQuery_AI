"""Convert `data/instruct_mix/{instruct,val}.jsonl` to the unified manifest format.

The instruction mix carries what the official RSVQA-LR train split does not:
SAR and optical tile questions over WHU-OPT-SAR (answers derived from label
masks) and the **refusal** examples that give the deployed `rs_vqa` tool its
abstention behaviour. A VQA adapter trained on RSVQA-LR alone would answer
every question; mixing this in keeps the refusal skill. Rows are tagged
`metadata.kind` (vqa / refusal) and `metadata.source` so the evaluator can
report them separately, and the `whu_opt_sar` rows inherit the scene-leak
caveat from the dataset audit (F1) - they are training signal, not a
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def convert(src: Path, dst: Path, split: str) -> dict:
    rows = []
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "id": f"instruct_mix:{split}:{i}", "image": r["image"].replace("\\", "/"),
            "modality": r.get("modality", "rgb"), "task": "<VQA>", "question": r["question"],
            "target": str(r["answer"]), "split": split,
            "metadata": {"type": r.get("kind", "vqa"), "source": r.get("source", "unknown")},
            "source": "instruct_mix", "license": "mixed (see licensing.md)",
            "synthetic": r.get("source") == "synthetic_refusal",
        })
    lines = [json.dumps(r, sort_keys=True) for r in rows]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    h = hashlib.sha256()
    for l in lines:
        h.update(l.encode()); h.update(b"\n")
    kinds = {}
    for r in rows:
        k = f"{r['metadata']['source']}/{r['metadata']['type']}"
        kinds[k] = kinds.get(k, 0) + 1
    return {"n": len(rows), "kinds": kinds, "hash": h.hexdigest()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/instruct_mix"))
    args = p.parse_args()
    man = args.data / "manifests"
    man.mkdir(exist_ok=True)
    stats = {"source": "instruct_mix", "splits": {}, "hashes": {}}
    for split, name in (("train", "instruct.jsonl"), ("val", "val.jsonl")):
        if (args.data / name).exists():
            st = convert(args.data / name, man / f"{split}.jsonl", split)
            stats["hashes"][f"{split}.jsonl"] = st.pop("hash")
            stats["splits"][split] = st
            print(split, st)
    # The mix's rsvqa_lr rows are the HF redistribution of the OFFICIAL
    # VALIDATION split - the selection set for every Phase 6 VQA run. A
    # training manifest without them keeps selection uncontaminated; the
    # official train split supplies the RSVQA signal instead.
    src = man / "train.jsonl"
    if src.exists():
        keep = [l for l in src.read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["metadata"]["source"] != "rsvqa_lr"]
        (man / "train_no_rsvqa.jsonl").write_text("\n".join(keep) + "\n", encoding="utf-8")
        h = hashlib.sha256()
        for l in keep:
            h.update(l.encode()); h.update(b"\n")
        stats["hashes"]["train_no_rsvqa.jsonl"] = h.hexdigest()
        stats["splits"]["train_no_rsvqa"] = {"n": len(keep)}
        print("train_no_rsvqa", len(keep))
    (man / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

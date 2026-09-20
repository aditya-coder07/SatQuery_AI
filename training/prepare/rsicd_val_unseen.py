"""RSICD: a val selection set whose references are NOT memorisable from train.

Why (measured 2026-09-20 on the parquet release, `scripts` in
docs/research/model_improvement_report.md §E4):

    split   refs verbatim in train   images with all 5 refs in train   distinct refs
    val     32.3%                    18.4%                             59.5%
    test    11.3%                     0.9%                             84.8%

The official val rewards a captioner for reproducing training sentences
- a third of its reference captions are training captions - while the
test does not. That is why `caption_vlm` scored corpus BLEU-4 0.40-0.42 on
val and 0.256 on test, and why "val still rising" at the end of the
1-epoch schedule is partly memorisation, not learning. Selecting a
checkpoint on this val picks the step that memorised most.

This writes `manifests/val_unseen.jsonl`: the val images none of whose
references appears verbatim (case-, punctuation-insensitive) in any train
reference. Images and references are untouched; nothing is removed from
the official val, which stays the reporting split for comparability. It
is a *selection* split, and the train split it is measured against is the
one the checkpoint is trained on.

Usage (from the dataset root that holds manifests/{train,val}.jsonl)::

    python training/prepare/rsicd_val_unseen.py --manifests data/rsicd/manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def norm(caption: str) -> str:
    return re.sub(r"[^a-z ]", "", str(caption).lower()).strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--out-name", default="val_unseen.jsonl")
    args = p.parse_args()

    train = [json.loads(l) for l in (args.manifests / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val = [json.loads(l) for l in (args.manifests / "val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {norm(r["target"]) for r in train if isinstance(r["target"], str)}
    seen |= {norm(c) for r in train if isinstance(r["target"], list) for c in r["target"]}

    kept, n_refs, n_seen = [], 0, 0
    for r in val:
        refs = r["target"] if isinstance(r["target"], list) else [r["target"]]
        hits = [norm(c) in seen for c in refs]
        n_refs += len(refs)
        n_seen += sum(hits)
        if not any(hits):
            kept.append(r)
    out = args.manifests / args.out_name
    lines = [json.dumps(r, sort_keys=True) for r in kept]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode()); h.update(b"\n")
    stats_path = args.manifests / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats.setdefault("hashes", {})[args.out_name] = h.hexdigest()
    stats.setdefault("splits", {})["val_unseen"] = {
        "n": len(kept), "n_images": len(kept), "of_val": len(val),
        "val_refs_verbatim_in_train": round(n_seen / max(1, n_refs), 4),
        "rule": "val images with no reference verbatim in any train reference",
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"val {len(val)} images -> {len(kept)} with no reference seen in train "
          f"({n_seen / max(1, n_refs):.1%} of val references are verbatim train captions); wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

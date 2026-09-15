"""Re-score a VRSBench grounding predictions file on subsets.

`vlm_task_eval.py` writes every reply to `<out>.predictions.jsonl`; this
reads them back, recomputes IoU from the reply and the manifest's target
(the model frame is the processor's smart-resize of the image, recomputed
from the image size), and reports Acc@0.5 on:

* all rows;
* the **clean** subset - rows whose DIOR source image is NOT in the
  DIOR-RSVG official train split (`metadata.dior_source_in_rsvg_train`
  false), the fair number for an adapter trained on DIOR-RSVG train;
* DOTA-only and DIOR-only halves.

Usage::

    python evaluation/vrsbench_subsets.py --manifest data/vrsbench/manifests/val_grounding.jsonl \
        --predictions artifacts/benchmark_reports/vrsbench_val_grounding.predictions.jsonl \
        --base models/qwen25_vl_3b --out artifacts/benchmark_reports/vrsbench_val_grounding_subsets.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import vlm_grounding as vg  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.base))
    root = args.manifest.parent.parent
    rows = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    sizes: dict[str, tuple[int, int]] = {}
    hits: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        r = rows[q["id"]]
        w, h = r["target"].get("width"), r["target"].get("height")
        if w is None:
            if r["image"] not in sizes:
                with Image.open(root / r["image"]) as im:
                    sizes[r["image"]] = im.size
            w, h = sizes[r["image"]]
        rh, rw = vg.smart_resized_hw(processor, w, h)
        box = vg.parse_box(q["pred"])
        iou = vg.iou_xyxy(vg.scale_box(box, w / rw, h / rh), r["target"]["bbox_xyxy"]) if box else 0.0
        hit = iou >= 0.5
        meta = r.get("metadata") or {}
        arm = q["arm"]
        hits[arm]["all"].append(hit)
        hits[arm]["clean(not_in_rsvg_train)" if not meta.get("dior_source_in_rsvg_train") else "dior_source_in_rsvg_train"].append(hit)
        hits[arm]["dota" if meta.get("origin") == "DOTA-v2" else "dior"].append(hit)
    rng = np.random.default_rng(0)
    report = {"manifest": str(args.manifest), "predictions": str(args.predictions), "arms": {}}
    for arm, subsets in hits.items():
        report["arms"][arm] = {}
        for name, hs in subsets.items():
            a = np.array(hs, dtype=float)
            boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)]
            report["arms"][arm][name] = {"n": len(a), "acc@0.5": float(a.mean()),
                                         "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for arm, subs in report["arms"].items():
        print(arm, {k: (v["n"], round(v["acc@0.5"], 4)) for k, v in subs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

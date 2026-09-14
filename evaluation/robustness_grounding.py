"""Corruption robustness of a grounding adapter on a fixed subsample of the
DIOR-RSVG official test split.

Conditions on the input image (the phrase is untouched): clean; JPEG
quality 30; gaussian blur sigma 1.0 and 2.0; brightness x0.8 / x1.2;
gaussian noise sigma 8/255; horizontal flip (the box is flipped with it,
so Acc@0.5 should hold if the adapter is not relying on absolute
position); a 25% downscale then upscale (resolution loss - the failure
mode arm A's small-object result predicts).

Reports Acc@0.5 / mIoU per condition and the delta to clean. Read-only.

Usage::

    python evaluation/robustness_grounding.py --base models/qwen25_vl_3b \
        --adapter checkpoints/v3/grounding_vlm_r16/adapter_best --quant 4bit \
        --manifest data/dior_rsvg_official/manifests/test.jsonl --limit 1000 \
        --out artifacts/benchmark_reports/dior_rsvg_robustness_lora_r16.json
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.vlm_task_eval import load_arm  # noqa: E402
from training.common import registry, vlm_grounding as vg  # noqa: E402

CONDITIONS = ["clean", "jpeg_q30", "blur_1.0", "blur_2.0", "bright_0.8", "bright_1.2", "noise_8", "hflip", "downscale_0.25"]


def corrupt(image, box, name: str, rng: random.Random):
    from PIL import Image, ImageEnhance, ImageFilter

    w, h = image.size
    if name == "clean":
        return image, box
    if name == "jpeg_q30":
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=30)
        buf.seek(0)
        return Image.open(buf).convert("RGB"), box
    if name.startswith("blur_"):
        return image.filter(ImageFilter.GaussianBlur(float(name.split("_")[1]))), box
    if name.startswith("bright_"):
        return ImageEnhance.Brightness(image).enhance(float(name.split("_")[1])), box
    if name == "noise_8":
        a = np.asarray(image, dtype="float32")
        a = a + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 8, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8")), box
    if name == "hflip":
        x0, y0, x1, y1 = box
        return image.transpose(Image.FLIP_LEFT_RIGHT), [w - x1, y0, w - x0, y1]
    if name == "downscale_0.25":
        small = image.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR), box
    raise ValueError(name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    import torch
    from PIL import Image

    root = args.manifest.parent.parent
    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]
    model, processor = load_arm(args.base, args.adapter, args.quant, torch)
    results = {}
    t0 = time.time()
    for cond in CONDITIONS:
        hits, ious = [], []
        for s in range(0, len(rows), args.batch):
            chunk = rows[s: s + args.batch]
            images, phrases, boxes = [], [], []
            for r in chunk:
                im = Image.open(root / r["image"]).convert("RGB")
                im, box = corrupt(im, list(r["target"]["bbox_xyxy"]), cond, rng)
                images.append(im)
                phrases.append(r["question"])
                boxes.append(box)
            preds, _, _ = vg.predict_batch(model, processor, images, phrases)
            for pred, gt in zip(preds, boxes):
                iou = vg.iou_xyxy(pred, gt) if pred is not None else 0.0
                ious.append(iou)
                hits.append(iou >= 0.5)
        results[cond] = {"acc@0.5": float(np.mean(hits)), "miou": float(np.mean(ious)), "n": len(rows)}
        print(f"{cond:16s} acc@0.5 {results[cond]['acc@0.5']:.4f} miou {results[cond]['miou']:.4f}", flush=True)
    clean = results["clean"]["acc@0.5"]
    for r in results.values():
        r["delta_acc"] = r["acc@0.5"] - clean
    report = {"adapter": args.adapter, "quant": args.quant, "manifest": str(args.manifest), "n": len(rows), "seed": args.seed,
              "conditions": results, "seconds": time.time() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    registry.record(experiment_id=registry.new_id("robustness_grounding"), model="grounding_vlm", status="done",
                    checkpoint=args.adapter, test_metrics={"clean": clean, "worst": min(results, key=lambda c: results[c]["acc@0.5"]),
                                                             "worst_acc": min(r["acc@0.5"] for r in results.values()), "n": len(rows)},
                    notes=f"corruption suite, {args.quant}, DIOR-RSVG test subsample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

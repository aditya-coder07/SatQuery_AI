"""Score grounding arms on the OFFICIAL DIOR-RSVG test split.

Reports, for each arm (base model zero-shot, or base + adapter):

* Acc@0.25 / Acc@0.5 / Acc@0.7 and mIoU, with bootstrap 95% CIs;
* the same on the **image-disjoint subset** - test expressions whose image
  never appears in the official train split (1,896 of 6,102 images). The
  official protocol splits by object, so 69% of test images are also train
  images with a different phrase; the disjoint subset is the stricter number;
* per-category and per-size-bucket Acc@0.5;
* parse-failure rate (reply had no box);
* a failure taxonomy over the misses: wrong-object (IoU < 0.1 with the
  ground truth but >= 0.5 with another annotated object of the same image),
  localisation-drift (0.1 <= IoU < 0.5), scale-error (centre inside the box
  but IoU < 0.5), no-box;
* a paired McNemar test between any two arms on Acc@0.5 hits.

Read-only: writes only `--out` and a per-question predictions JSONL beside
it. Never touches checkpoints, configs, or docs.

Usage::

    python evaluation/grounding_official_eval.py --base models/qwen25_vl_3b \
        --data data/dior_rsvg_official \
        --arms zero_shot=BASE lora=checkpoints/v3/grounding_vlm/adapter_best \
        --out artifacts/benchmark_reports/dior_rsvg_official.json
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import registry  # noqa: E402
from training.common.vlm_grounding import iou_xyxy, predict_batch, set_pixel_budget, size_bucket  # noqa: E402

PIXEL_BUDGET = [None, None]  # (min_pixels, max_pixels) applied to every loaded processor


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return [0.0, 0.0]
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)]


def mcnemar(a_hits: list[bool], b_hits: list[bool]) -> dict:
    b = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    c = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    return {"a_only": b, "b_only": c, "chi2_cc": round(chi2, 3),
            "significant_at_0.05": chi2 > 3.841}


def summarize(rows: list[dict], preds: list, parsed: list[bool], train_images: set[str],
              objects_by_image: dict[str, list[list[int]]]) -> dict:
    ious, hits5 = [], []
    per_cat: dict[str, list[bool]] = collections.defaultdict(list)
    per_size: dict[str, list[bool]] = collections.defaultdict(list)
    disjoint: list[bool] = []
    disjoint_iou: list[float] = []
    taxonomy = collections.Counter()
    for r, p, ok in zip(rows, preds, parsed):
        gt = r["target"]["bbox_xyxy"]
        iou = iou_xyxy(p, gt) if p else 0.0
        ious.append(iou)
        hit = iou >= 0.5
        hits5.append(hit)
        per_cat[r["target"]["category"]].append(hit)
        per_size[size_bucket(gt, r["target"]["width"], r["target"]["height"])].append(hit)
        if r["image"] not in train_images:
            disjoint.append(hit)
            disjoint_iou.append(iou)
        if hit:
            continue
        if not ok or p is None:
            taxonomy["no_box"] += 1
        elif iou < 0.1 and any(iou_xyxy(p, o) >= 0.5 for o in objects_by_image.get(r["image"], []) if o != gt):
            taxonomy["wrong_object"] += 1
        elif iou < 0.1:
            taxonomy["wrong_box"] += 1
        else:
            cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
            inside = gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]
            taxonomy["scale_error" if inside else "localisation_drift"] += 1
    n = len(rows)
    out = {
        "n": n,
        "acc@0.25": round(sum(i >= 0.25 for i in ious) / n, 4),
        "acc@0.5": round(sum(hits5) / n, 4),
        "acc@0.5_ci95": bootstrap_ci([float(h) for h in hits5]),
        "acc@0.7": round(sum(i >= 0.7 for i in ious) / n, 4),
        "miou": round(sum(ious) / n, 4),
        "miou_ci95": bootstrap_ci(ious),
        "parse_rate": round(sum(parsed) / n, 4),
        "image_disjoint_subset": {
            "n": len(disjoint),
            "acc@0.5": round(sum(disjoint) / max(1, len(disjoint)), 4),
            "acc@0.5_ci95": bootstrap_ci([float(h) for h in disjoint]),
            "miou": round(sum(disjoint_iou) / max(1, len(disjoint_iou)), 4),
        },
        "per_category_acc@0.5": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_cat.items())},
        "per_size_acc@0.5": {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)}
                             for k, v in sorted(per_size.items())},
        "failure_taxonomy": dict(taxonomy),
        "iou_histogram": {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": sum(1 for i in ious if b / 10 <= i < (b + 1) / 10)
                          for b in range(10)},
    }
    return out


def load_arm(base: Path, adapter: str, torch):
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    processor = AutoProcessor.from_pretrained(str(base), local_files_only=True)
    if PIXEL_BUDGET[0] or PIXEL_BUDGET[1]:
        set_pixel_budget(processor, PIXEL_BUDGET[0], PIXEL_BUDGET[1])
    model = AutoVLM.from_pretrained(str(base), dtype=torch.bfloat16, device_map={"": 0},
                                    local_files_only=True, trust_remote_code=False)
    if adapter != "BASE":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, processor


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True, help="name=BASE or name=/path/to/adapter")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="debug only; a limited run is NOT a benchmark")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--min-pixels", type=int, default=None, help="processor min_pixels (upscale), as trained")
    p.add_argument("--max-pixels", type=int, default=None, help="processor max_pixels, as trained")
    args = p.parse_args()
    PIXEL_BUDGET[0], PIXEL_BUDGET[1] = args.min_pixels, args.max_pixels

    import torch

    man = args.data / "manifests"
    rows = [json.loads(l) for l in (man / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    train_rows = [json.loads(l) for l in (man / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    train_images = {r["image"] for r in train_rows}
    objects_by_image: dict[str, list[list[int]]] = collections.defaultdict(list)
    for r in rows + train_rows:
        objects_by_image[r["image"]].append(r["target"]["bbox_xyxy"])
    if args.limit:
        rows = rows[:args.limit]
    stats = json.loads((man / "stats.json").read_text())

    report = {"benchmark": "DIOR-RSVG official test split", "n": len(rows),
              "limited_debug_run": bool(args.limit), "manifest_hashes": stats["hashes"],
              "image_overlap_train_test": stats["image_overlap"]["train_test"], "arms": {}}
    hits_by_arm: dict[str, list[bool]] = {}
    pred_path = args.out.with_suffix(".predictions.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred_fh = pred_path.open("w", encoding="utf-8")

    from PIL import Image

    for spec in args.arms:
        name, adapter = spec.split("=", 1)
        model, processor = load_arm(args.base, adapter, torch)
        preds, parsed, t0 = [], [], time.time()
        for s in range(0, len(rows), args.batch):
            chunk = rows[s:s + args.batch]
            images = [Image.open(args.data / r["image"]).convert("RGB") for r in chunk]
            b_preds, replies, b_parsed = predict_batch(model, processor, images, [r["question"] for r in chunk])
            preds.extend(b_preds)
            parsed.extend(b_parsed)
            for r, pr, rep in zip(chunk, b_preds, replies):
                pred_fh.write(json.dumps({"arm": name, "id": r["id"], "pred": pr, "reply": rep,
                                          "gt": r["target"]["bbox_xyxy"]}) + "\n")
            if (s // args.batch) % 20 == 0:
                print(f"[{name}] {s + len(chunk)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        el = time.time() - t0
        summary = summarize(rows, preds, parsed, train_images, objects_by_image)
        summary["timing"] = {"generate_s": round(el, 1), "s_per_question": round(el / len(rows), 4),
                             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        summary["adapter"] = adapter
        report["arms"][name] = summary
        hits_by_arm[name] = [iou_xyxy(pr, r["target"]["bbox_xyxy"]) >= 0.5 if pr else False
                             for r, pr in zip(rows, preds)]
        print(f"[{name}] acc@0.5 {summary['acc@0.5']} ci {summary['acc@0.5_ci95']} miou {summary['miou']} "
              f"disjoint {summary['image_disjoint_subset']['acc@0.5']} parse {summary['parse_rate']}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    names = list(hits_by_arm)
    report["paired_tests"] = {f"{a}_vs_{b}": mcnemar(hits_by_arm[a], hits_by_arm[b])
                              for i, a in enumerate(names) for b in names[i + 1:]}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pred_fh.close()
    if not args.limit:
        for name, s in report["arms"].items():
            registry.record(experiment_id=registry.new_id(f"eval_grounding_{name}"),
                            model=str(args.base), checkpoint=s["adapter"], status="done",
                            test_metrics={"benchmark": report["benchmark"], **{k: s[k] for k in
                                          ("acc@0.5", "acc@0.5_ci95", "miou", "image_disjoint_subset")}},
                            dataset_manifest_hash=stats["hashes"]["test.jsonl"], hardware=registry.hardware())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

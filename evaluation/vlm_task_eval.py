"""Score a Qwen2.5-VL adapter (or the bare base) on a unified-format manifest.

The generation-based metrics are the same ones `train_vlm_sft.py` selects
on, applied to a **test** manifest the trainer never reads: exact match by
type for `<VQA>`, corpus BLEU-1..4 / ROUGE-L / CIDEr-D (all references) for
`<CAPTION>` and `<CHANGE_CAPTION>` with changed / unchanged halves, and
Acc@0.5 / mIoU for `<GROUNDING>` (use `grounding_official_eval.py` for the
full grounding analysis). Bootstrap 95% CIs on the headline metric, per-row
predictions beside the report, registry record on completion.

Read-only with respect to checkpoints, configs and docs.

Usage::

    python evaluation/vlm_task_eval.py --base models/qwen25_vl_3b \
        --manifest data/rsicd/manifests/test.jsonl \
        --arms base=BASE caption_lora=checkpoints/v3/caption_vlm/adapter_best \
        --out artifacts/benchmark_reports/rsicd_test_vlm.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics.caption_corpus import score_corpus  # noqa: E402
from evaluation.metrics.vqa import normalise_answer  # noqa: E402
from training.common import registry  # noqa: E402
from training.common import vlm_grounding as vg  # noqa: E402
from training.train_vlm_sft import PUBLISHED_TYPES, generate_batch, load_manifest  # noqa: E402


def bootstrap(fn, items, n_boot=1000, seed=0):
    rng = random.Random(seed)
    n = len(items)
    vals = sorted(fn([items[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return [round(vals[int(0.025 * n_boot)], 4), round(vals[int(0.975 * n_boot) - 1], 4)]


def load_arm(base: Path, adapter: str, quant: str, torch):
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    processor = AutoProcessor.from_pretrained(str(base), local_files_only=True)
    kwargs = dict(device_map={"": 0}, local_files_only=True, trust_remote_code=False)
    if quant == "4bit":
        # The deployed tools load the base in NF4; scoring in the same
        # precision measures the model that actually ships.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoVLM.from_pretrained(str(base), **kwargs)
    if adapter != "BASE":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if quant != "4bit":
            model = model.merge_and_unload()
    model.eval()
    return model, processor


def score(rows: list[dict], replies: list[str], grids) -> dict:
    task = rows[0]["task"]
    if task in ("<CAPTION>", "<CHANGE_CAPTION>"):
        items = [(rep, r["target"] if isinstance(r["target"], list) else [str(r["target"])],
                  (r.get("metadata") or {}).get("changeflag")) for r, rep in zip(rows, replies)]
        res = {"n": len(items), "corpus": score_corpus([i[0] for i in items], [i[1] for i in items])}
        res["bleu4_ci95"] = bootstrap(lambda it: score_corpus([i[0] for i in it], [i[1] for i in it])["bleu4"], items, 300)
        res["cider_d_ci95"] = bootstrap(lambda it: score_corpus([i[0] for i in it], [i[1] for i in it])["cider_d"], items, 300)
        if task == "<CHANGE_CAPTION>":
            for name, flag in (("changed", 1), ("unchanged", 0)):
                sub = [i for i in items if i[2] == flag]
                if sub:
                    res[name] = score_corpus([i[0] for i in sub], [i[1] for i in sub])
        res["unique_fraction"] = round(len(set(i[0] for i in items)) / len(items), 4)
        res["headline"] = res["corpus"]["bleu4"]
        return res
    if task == "<GROUNDING>":
        hits, ious = [], []
        for r, rep, grid in zip(rows, replies, grids):
            box = vg.parse_box(rep)
            iou = 0.0
            if box:
                rh, rw = vg.resized_hw(grid)
                w, h = r["target"].get("width"), r["target"].get("height")
                if w is None or h is None:  # VRSBench rows carry no size; read the image once
                    from PIL import Image

                    with Image.open(r["_images"][0]) as im:
                        w, h = im.size
                iou = vg.iou_xyxy(vg.scale_box(box, w / rw, h / rh), r["target"]["bbox_xyxy"])
            hits.append(iou >= 0.5)
            ious.append(iou)
        return {"n": len(rows), "acc@0.5": sum(hits) / len(hits), "miou": sum(ious) / len(ious),
                "acc@0.5_ci95": bootstrap(lambda it: sum(it) / len(it), hits), "headline": sum(hits) / len(hits)}
    # VQA-like: exact match
    items = [(normalise_answer(rep) == normalise_answer(str(r["target"])),
              (r.get("metadata") or {}).get("type", "all")) for r, rep in zip(rows, replies)]
    by_type: dict[str, list[bool]] = {}
    for hit, t in items:
        by_type.setdefault(t, []).append(hit)
    pub = [h for h, t in items if t in PUBLISHED_TYPES]
    res = {"n": len(items), "acc_all": sum(h for h, _ in items) / len(items),
           "by_type": {t: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for t, v in sorted(by_type.items())}}
    if pub:
        res["acc_published"] = sum(pub) / len(pub)
        res["acc_published_ci95"] = bootstrap(lambda it: sum(it) / len(it), pub)
    res["headline"] = res.get("acc_published", res["acc_all"])
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True, help="name=BASE | name=/path/to/adapter")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--limit", type=int, default=None, help="debug only; not a benchmark")
    args = p.parse_args()

    import torch
    from PIL import Image

    rows = load_manifest(args.manifest, args.limit, 0)
    stats_path = args.manifest.parent / "stats.json"
    manifest_hash = json.loads(stats_path.read_text()).get("hashes", {}).get(args.manifest.name) if stats_path.exists() else None
    report = {"manifest": str(args.manifest), "manifest_hash": manifest_hash, "n": len(rows), "quant": args.quant,
              "limited_debug_run": bool(args.limit), "task": rows[0]["task"], "arms": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred_fh = args.out.with_suffix(".predictions.jsonl").open("w", encoding="utf-8")
    for spec in args.arms:
        name, adapter = spec.split("=", 1)
        model, processor = load_arm(args.base, adapter, args.quant, torch)
        replies, grids, t0 = [], [], time.time()
        for s in range(0, len(rows), args.batch):
            chunk = rows[s:s + args.batch]
            image_lists = [[Image.open(q).convert("RGB") for q in r["_images"]] for r in chunk]
            reps, g = generate_batch(model, processor, image_lists, chunk, args.max_new_tokens)
            replies.extend(reps)
            gi = 0
            for imgs in image_lists:
                grids.append(g[gi])
                gi += len(imgs)
            for r, rep in zip(chunk, reps):
                pred_fh.write(json.dumps({"arm": name, "id": r["id"], "pred": rep, "target": r["target"]}) + "\n")
            if (s // args.batch) % 25 == 0:
                print(f"[{name}] {s + len(chunk)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        res = score(rows, replies, grids)
        res["adapter"] = adapter
        res["timing"] = {"generate_s": round(time.time() - t0, 1), "s_per_item": round((time.time() - t0) / len(rows), 4),
                         "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        report["arms"][name] = res
        print(f"[{name}] headline {res['headline']:.4f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pred_fh.close()
    if not args.limit:
        for name, res in report["arms"].items():
            registry.record(experiment_id=registry.new_id(f"eval_{report['task'].strip('<>').lower()}_{name}"),
                            model=str(args.base), checkpoint=res["adapter"], status="done",
                            test_metrics={"manifest": str(args.manifest), "headline": res["headline"],
                                          **{k: v for k, v in res.items() if k not in ("adapter", "timing")}},
                            dataset_manifest_hash=manifest_hash, hardware=registry.hardware())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

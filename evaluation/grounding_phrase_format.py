"""Does the grounder care HOW the phrase reaches it? (2026-09-20)

The served grounding path used to hand the adapter the user's whole
sentence as the referring phrase, so "Find the airport in this image."
became the prompt "Locate the Find the airport in this image in the image
...". The adapter was trained on DIOR-RSVG expressions - bare noun phrases.
This experiment measures what that cost and what the extractor in
`satquery/controller/understanding.py` recovers, on the official DIOR-RSVG
test expressions, with the deployed adapter in the deployed precision.

Three conditions over the SAME expressions and images:

* ``bare``      - the expression as annotated (the benchmark condition);
* ``sentence``  - the expression wrapped the way a user types it
                  ("Find <expr> in this image." / "Where is <expr>?" /
                  "Can you locate <expr>?"), passed through unchanged - the
                  pre-fix served path;
* ``extracted`` - the same sentences run through `extract_object` +
                  `referring_expression` first - the post-fix served path.

Acc@0.5 per condition with bootstrap CIs and McNemar between conditions.
A deterministic subsample (`--n`, seed 0) of the 7,500 test expressions:
this is a controlled comparison of *prompt formats*, not a benchmark
number, and the report says so.

Usage::

    python evaluation/grounding_phrase_format.py --base models/qwen25_vl_3b \
        --adapter checkpoints/v3/grounding_vlm_hires/adapter_best \
        --parquet-dir data/dior_rsvg/data --n 200 --min-pixels 1048576 \
        --out artifacts/benchmark_reports/grounding_phrase_format.json
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.grounding_official_eval import bootstrap_ci, mcnemar  # noqa: E402
from evaluation.vlm_task_eval import PIXEL_BUDGET, load_arm  # noqa: E402
from satquery.controller.understanding import extract_object, extract_spatial, referring_expression  # noqa: E402
from training.common.vlm_grounding import iou_xyxy, predict_batch  # noqa: E402

WRAPPERS = [
    "Find {expr} in this image.",
    "Where is {expr}?",
    "Can you locate {expr} in this picture?",
    "Show me {expr}.",
    "Point out {expr} for me.",
]


def user_sentence(expr: str, i: int) -> str:
    """Wrap an annotated expression the way a person would type it. The
    article is the annotator's ("a green basketball court") - kept, since a
    user writes "find the ..." and "find a ..." interchangeably."""
    e = expr.strip().rstrip(".")
    e = e[0].lower() + e[1:] if e else e
    return WRAPPERS[i % len(WRAPPERS)].format(expr=e)


def extracted_phrase(sentence: str) -> str:
    obj = extract_object(sentence)
    phrase = referring_expression(sentence, obj, extract_spatial(sentence))
    return phrase or sentence


def load_rows(parquet_dir: Path, n: int, seed: int) -> list[dict]:
    import pandas as pd

    frames = [pd.read_parquet(p) for p in sorted(parquet_dir.glob("test-*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(df)), n))
    rows = []
    for i in idx:
        r = df.iloc[i]
        box = [int(v) for v in str(r["bbox"]).strip("[]").split()]
        rows.append({"id": f"{r['image_id']}:{r['question_id']}", "expr": str(r["question"]),
                     "bbox": box, "image_bytes": r["image"]["bytes"]})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    PIXEL_BUDGET[0], PIXEL_BUDGET[1] = args.min_pixels, args.max_pixels

    import torch
    from PIL import Image

    rows = load_rows(args.parquet_dir, args.n, args.seed)
    sentences = [user_sentence(r["expr"], i) for i, r in enumerate(rows)]
    conditions = {
        "bare": [r["expr"] for r in rows],
        "sentence": sentences,
        "extracted": [extracted_phrase(s) for s in sentences],
    }
    model, processor = load_arm(args.base, args.adapter, args.quant, torch)
    report = {"experiment": "grounding phrase format", "n": len(rows), "seed": args.seed,
              "subsample_not_a_benchmark": True, "adapter": args.adapter, "quant": args.quant,
              "pixel_budget": [args.min_pixels, args.max_pixels], "conditions": {},
              "examples": [{"expr": r["expr"], "sentence": s, "extracted": e}
                           for r, s, e in list(zip(rows, sentences, conditions["extracted"]))[:12]]}
    hits: dict[str, list[bool]] = {}
    pred_fh = args.out.with_suffix(".predictions.jsonl").open("w", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, phrases in conditions.items():
        t0 = time.time()
        ious: list[float] = []
        parsed_n = 0
        for s in range(0, len(rows), args.batch):
            chunk = rows[s:s + args.batch]
            images = [Image.open(io.BytesIO(r["image_bytes"])).convert("RGB") for r in chunk]
            preds, replies, parsed = predict_batch(model, processor, images, phrases[s:s + args.batch])
            for r, pr, rep, ok, ph in zip(chunk, preds, replies, parsed, phrases[s:s + args.batch]):
                iou = iou_xyxy(pr, r["bbox"]) if pr else 0.0
                ious.append(iou)
                parsed_n += int(ok)
                pred_fh.write(json.dumps({"condition": name, "id": r["id"], "phrase": ph, "pred": pr,
                                          "gt": r["bbox"], "iou": round(iou, 4), "reply": rep}) + "\n")
            if (s // args.batch) % 10 == 0:
                print(f"[{name}] {s + len(chunk)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        h = [i >= 0.5 for i in ious]
        hits[name] = h
        report["conditions"][name] = {
            "acc@0.5": round(sum(h) / len(h), 4), "acc@0.5_ci95": bootstrap_ci([float(x) for x in h]),
            "miou": round(sum(ious) / len(ious), 4), "parse_rate": round(parsed_n / len(rows), 4),
            "s_per_item": round((time.time() - t0) / len(rows), 3),
        }
        print(f"[{name}] acc@0.5 {report['conditions'][name]['acc@0.5']} miou {report['conditions'][name]['miou']}", flush=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = list(hits)
    report["paired_tests"] = {f"{a}_vs_{b}": mcnemar(hits[a], hits[b])
                              for i, a in enumerate(names) for b in names[i + 1:]}
    report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pred_fh.close()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

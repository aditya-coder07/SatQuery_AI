"""Caption decoding ablation: the same adapter, different `generate()` settings.

The deployed captioner (`caption_vlm/adapter_best`, RSICD test corpus BLEU-4
0.256 / CIDEr-D 0.793) decodes greedily. 40% of its test captions are
duplicates of another test caption (`unique_fraction` 0.60) - the generic
"many buildings are in a dense residential area" answer - which is the
symptom decoding can move without touching the weights. This script scores
decoding settings under the deployed precision (4-bit) on:

* a **selection** split: the official RSICD val (`--val-n` for a seeded
  subsample; the trainer selected its checkpoint on a seeded random 300 and
  reported corpus BLEU-4 0.42 there against 0.256 on the test split, a gap
  this run measures on the full val as a by-product);
* the **report** split: the complete official RSICD test.

The setting is chosen on val and reported on test; the report keeps both.
Corpus BLEU-1..4, ROUGE-L, CIDEr-D, METEOR-exact and the unique fraction,
from `evaluation/metrics/caption_corpus.py`, the same scorer as every other
caption number in this repository.

Usage::

    python evaluation/caption_decoding.py --base models/qwen25_vl_3b \
        --adapter checkpoints/v3/caption_vlm/adapter_best \
        --parquet-dir data/rsicd/data --split val --val-n 400 \
        --out artifacts/benchmark_reports/rsicd_decoding_val.json
    python evaluation/caption_decoding.py ... --split test --conditions greedy beam3 \
        --out artifacts/benchmark_reports/rsicd_decoding_test.json
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

from evaluation.metrics.caption_corpus import score_corpus  # noqa: E402
from evaluation.vlm_task_eval import bootstrap, load_arm  # noqa: E402
from training.prepare.caption_manifests import CAPTION_PROMPT  # noqa: E402
from training.train_vlm_sft import VQA_SYSTEM_PROMPT  # noqa: E402

CONDITIONS: dict[str, dict] = {
    "greedy": {"num_beams": 1},
    "beam3": {"num_beams": 3},
    "beam3_norepeat3": {"num_beams": 3, "no_repeat_ngram_size": 3},
    "beam5": {"num_beams": 5},
    "greedy_norepeat3": {"num_beams": 1, "no_repeat_ngram_size": 3},
}


# Function words and hedges: never counted as content. Everything else in a
# caption is a claim about the scene (an object, a colour, a count, a
# relation), and a claim that none of the five references makes is the
# caption-level hallucination proxy used here (CHAIR-style, at word level,
# without a detector: the references are the only ground truth RSICD has).
_STOP = set("""a an the and or of in on at to with by from near next beside between around along
over under above below into onto is are was were be been being there this that these those it its
some many several few lots lot number two three four five six seven eight nine ten one
image scene picture photo view area areas region regions part parts side sides which who what
very quite also as while where when than then very more most other another each both all any
can could may might""".split())


def unsupported_rate(hyps: list[str], refs: list[list[str]]) -> dict:
    """Fraction of content words in the generated captions that appear in
    none of the image's references (micro over words; macro over
    captions). A word missing from all five references is either an object
    the annotators did not see or a synonym they did not use; both count,
    so the rate is an upper bound on hallucination and is only meaningful
    as a *difference between conditions* on the same images."""
    import re

    tot = bad = 0
    per_caption = []
    for h, rs in zip(hyps, refs):
        ref_words = set(w for r in rs for w in re.findall(r"[a-z]+", r.lower()))
        words = [w for w in re.findall(r"[a-z]+", h.lower()) if w not in _STOP and len(w) > 2]
        if not words:
            continue
        miss = [w for w in words if w not in ref_words and w.rstrip("s") not in ref_words and w + "s" not in ref_words]
        tot += len(words)
        bad += len(miss)
        per_caption.append(len(miss) / len(words))
    return {"micro": round(bad / max(1, tot), 4),
            "macro": round(sum(per_caption) / max(1, len(per_caption)), 4),
            "captions_with_any": round(sum(1 for x in per_caption if x > 0) / max(1, len(per_caption)), 4)}


def load_rows(parquet_dir: Path, split: str, n: int | None, seed: int) -> list[dict]:
    import pandas as pd

    name = {"val": "valid", "test": "test"}[split]
    df = pd.read_parquet(next(parquet_dir.glob(f"{name}-*.parquet")))
    idx = list(range(len(df)))
    if n is not None and n < len(df):
        idx = sorted(random.Random(seed).sample(idx, n))
    rows = []
    for i in idx:
        r = df.iloc[i]
        rows.append({"id": str(r["filename"]), "refs": [str(c) for c in r["captions"]],
                     "image_bytes": r["image"]["bytes"]})
    return rows


def generate(model, processor, images, kwargs: dict, max_new_tokens: int) -> list[str]:
    import torch

    tok = processor.tokenizer
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        chats = [[{"role": "system", "content": [{"type": "text", "text": VQA_SYSTEM_PROMPT}]},
                  {"role": "user", "content": [{"type": "image", "image": im},
                                               {"type": "text", "text": CAPTION_PROMPT}]}] for im in images]
        texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        batch = processor(text=texts, images=list(images), return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev
    batch = batch.to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, **kwargs)
    return [t.strip() for t in tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--val-n", type=int, default=400, help="val subsample size (selection split only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="debug only")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    from PIL import Image

    rows = load_rows(args.parquet_dir, args.split, args.val_n if args.split == "val" else None, args.seed)
    if args.limit:
        rows = rows[:args.limit]
    model, processor = load_arm(args.base, args.adapter, args.quant, torch)
    report = {"experiment": "caption decoding", "split": args.split, "n": len(rows), "seed": args.seed,
              "adapter": args.adapter, "quant": args.quant, "max_new_tokens": args.max_new_tokens,
              "prompt": CAPTION_PROMPT, "limited_debug_run": bool(args.limit), "conditions": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred_fh = args.out.with_suffix(".predictions.jsonl").open("w", encoding="utf-8")
    refs = [r["refs"] for r in rows]
    for name in args.conditions:
        kwargs = CONDITIONS[name]
        hyps: list[str] = []
        t0 = time.time()
        for s in range(0, len(rows), args.batch):
            chunk = rows[s:s + args.batch]
            images = [Image.open(io.BytesIO(r["image_bytes"])).convert("RGB") for r in chunk]
            reps = generate(model, processor, images, kwargs, args.max_new_tokens)
            hyps.extend(reps)
            for r, rep in zip(chunk, reps):
                pred_fh.write(json.dumps({"condition": name, "id": r["id"], "pred": rep}) + "\n")
            if (s // args.batch) % 20 == 0:
                print(f"[{name}] {s + len(chunk)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        items = list(zip(hyps, refs))
        res = {"generate_kwargs": kwargs, "corpus": score_corpus(hyps, refs),
               "bleu4_ci95": bootstrap(lambda it: score_corpus([i[0] for i in it], [i[1] for i in it])["bleu4"], items, 300),
               "cider_d_ci95": bootstrap(lambda it: score_corpus([i[0] for i in it], [i[1] for i in it])["cider_d"], items, 300),
               "unique_fraction": round(len(set(hyps)) / len(hyps), 4),
               "unsupported_content_words": unsupported_rate(hyps, refs),
               "mean_words": round(sum(len(h.split()) for h in hyps) / len(hyps), 2),
               "s_per_item": round((time.time() - t0) / len(rows), 3)}
        report["conditions"][name] = res
        c = res["corpus"]
        print(f"[{name}] bleu4 {c['bleu4']:.4f} cider {c['cider_d']:.4f} rougeL {c['rouge_l']:.4f} "
              f"unique {res['unique_fraction']} words {res['mean_words']} "
              f"unsupported {res['unsupported_content_words']['micro']}", flush=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pred_fh.close()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Re-score a caption-decoding run on the val images whose references are
not memorisable from train (the `val_unseen` rule of
training/prepare/rsicd_val_unseen.py), from its predictions file.

RSICD's val references are verbatim train captions 3x as often as the
test's (report §E4), so a decoding setting chosen on the full val sample
can be the one that reproduces training sentences best. This reads the
`.predictions.jsonl` beside a `caption_decoding.py` report, keeps the
images with no reference seen in train, and scores every condition on
that subset with the same corpus scorer.

Usage::

    python evaluation/caption_rescore_unseen.py --report artifacts/benchmark_reports/rsicd_decoding_val.json \
        --parquet-dir data/rsicd/data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.caption_decoding import unsupported_rate  # noqa: E402
from evaluation.metrics.caption_corpus import score_corpus  # noqa: E402
from training.prepare.rsicd_val_unseen import norm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--parquet-dir", type=Path, required=True)
    args = p.parse_args()

    import pandas as pd

    report = json.loads(args.report.read_text(encoding="utf-8"))
    split = {"val": "valid", "test": "test"}[report["split"]]
    train = pd.read_parquet(next(args.parquet_dir.glob("train-*.parquet")))
    seen = {norm(c) for caps in train["captions"] for c in caps}
    df = pd.read_parquet(next(args.parquet_dir.glob(f"{split}-*.parquet")))
    refs_by_id = {str(r["filename"]): [str(c) for c in r["captions"]] for _, r in df.iterrows()}
    unseen = {k for k, refs in refs_by_id.items() if not any(norm(c) in seen for c in refs)}

    preds: dict[str, dict[str, str]] = defaultdict(dict)
    for line in args.report.with_suffix(".predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["condition"]][r["id"]] = r["pred"]
    out = {"report": str(args.report), "rule": "images with no reference verbatim in train", "conditions": {}}
    for cond, by_id in preds.items():
        ids = [i for i in by_id if i in unseen]
        hyps = [by_id[i] for i in ids]
        refs = [refs_by_id[i] for i in ids]
        c = score_corpus(hyps, refs)
        out["conditions"][cond] = {"n": len(ids), "corpus": c,
                                   "unique_fraction": round(len(set(hyps)) / max(1, len(hyps)), 4),
                                   "unsupported_content_words": unsupported_rate(hyps, refs)}
        print(f"[{cond}] unseen n={len(ids)} bleu4 {c['bleu4']:.4f} cider {c['cider_d']:.4f} "
              f"rougeL {c['rouge_l']:.4f} unique {out['conditions'][cond]['unique_fraction']}")
    report["unseen_subset"] = out
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("updated", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

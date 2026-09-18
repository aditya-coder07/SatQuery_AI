"""Independent LEVIR-CD test evaluation of a change-mask checkpoint.

"Independent" of the trainer in every way that matters: the weights are
loaded through the deployed tool's own loader (`satquery.tools.change_mask
._Handle`, the code path that serves users), the tiles are read here with
PIL and the pooled change-class F1 / IoU / precision / recall are recomputed
from scratch, so a number agreeing with `checkpoints/.../metrics.json` was
reached by two implementations that share only the data.

Beyond the trainer's number it reports what a champion claim needs:

* a per-tile bootstrap 95% CI of the pooled metrics (tiles resampled,
  pixels pooled),
* the same metrics at thresholds 0.3-0.7 (0.5 is the headline; the sweep
  shows the number is not a threshold artefact),
* per-tile F1 quantiles and the count of empty-label tiles scored,
* the sha256 of the checkpoint and of the split index, the git commit, and
  the number of test tiles, all written into the report and the registry,
* with `--val-split val`, the threshold that maximises F1 on the official
  validation split and the test metrics at that threshold - a legitimate
  operating point (chosen without looking at test), reported beside the
  0.5 headline, never in its place.

Usage::

    python evaluation/change_mask_official_eval.py --checkpoint checkpoints/v3/change_mask/best.pt \
        --index data/levircd/index.json --out artifacts/benchmark_reports/levircd_test_independent_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import registry  # noqa: E402
from training.common.paths import index_path  # noqa: E402


def counts(prob: np.ndarray, truth: np.ndarray, thr: float) -> np.ndarray:
    p, t = prob >= thr, truth >= 0.5
    return np.array([(p & t).sum(), (p & ~t).sum(), (~p & t).sum()], dtype="float64")  # tp, fp, fn


def prf(c: np.ndarray) -> dict:
    tp, fp, fn = c
    return {"f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path("data/levircd/index.json"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-split", default=None, help="also pick the F1-maximising threshold on this split")
    args = p.parse_args()

    import torch
    from PIL import Image

    from satquery.tools.change_mask import _Handle

    index = json.loads(args.index.read_text(encoding="utf-8"))["splits"]
    rows = index[args.split]
    if args.limit:
        rows = rows[: args.limit]
    handle = _Handle(args.checkpoint)
    model, device = handle.model, handle.device
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    def score(rows):
        per_tile = {t: np.zeros((len(rows), 3)) for t in thresholds}
        with torch.no_grad():
            for s in range(0, len(rows), args.batch):
                chunk = rows[s: s + args.batch]
                a = np.stack([np.asarray(Image.open(index_path(r["a"])).convert("RGB"), dtype="float32") / 255.0 for r in chunk])
                b = np.stack([np.asarray(Image.open(index_path(r["b"])).convert("RGB"), dtype="float32") / 255.0 for r in chunk])
                m = np.stack([(np.asarray(Image.open(index_path(r["label"])).convert("L")) > 127).astype("float32") for r in chunk])
                logits = model(torch.from_numpy(a.transpose(0, 3, 1, 2)).to(device),
                               torch.from_numpy(b.transpose(0, 3, 1, 2)).to(device))
                prob = torch.sigmoid(logits)[:, 0].cpu().numpy()
                for j in range(len(chunk)):
                    for t in thresholds:
                        per_tile[t][s + j] = counts(prob[j], m[j], t)
        return per_tile

    t0 = time.time()
    per_tile = score(rows)
    pooled = {str(t): prf(per_tile[t].sum(0)) for t in thresholds}
    val_choice = None
    if args.val_split:
        v = score(index[args.val_split][: args.limit] if args.limit else index[args.val_split])
        vf1 = {t: prf(v[t].sum(0))["f1"] for t in thresholds}
        best_t = max(vf1, key=vf1.get)
        val_choice = {"val_split": args.val_split, "n_val": len(index[args.val_split]), "val_f1_by_threshold": {str(t): vf1[t] for t in thresholds},
                      "threshold": best_t, "test_at_val_threshold": pooled[str(best_t)]}
    rng = np.random.default_rng(0)
    boots = {"f1": [], "iou": []}
    c5 = per_tile[0.5]
    for _ in range(args.bootstrap):
        pick = rng.integers(0, len(rows), len(rows))
        r = prf(c5[pick].sum(0))
        boots["f1"].append(r["f1"])
        boots["iou"].append(r["iou"])
    tile_f1 = np.array([prf(c)["f1"] for c in c5])
    has_change = (c5[:, 0] + c5[:, 2]) > 0
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": registry.sha256_file(args.checkpoint),
        "loader": "satquery.tools.change_mask._Handle (deployed path)", "index": str(args.index),
        "index_sha256": registry.sha256_file(args.index), "split": args.split, "n_tiles": len(rows),
        "n_tiles_with_change": int(has_change.sum()), "git_commit": registry.git_commit(),
        "headline_threshold": 0.5, "pooled": pooled, "val_selected_threshold": val_choice,
        "ci95_f1": [float(np.percentile(boots["f1"], 2.5)), float(np.percentile(boots["f1"], 97.5))],
        "ci95_iou": [float(np.percentile(boots["iou"], 2.5)), float(np.percentile(boots["iou"], 97.5))],
        "per_tile_f1_quantiles_changed_tiles": {q: float(np.percentile(tile_f1[has_change], int(q))) for q in ("10", "25", "50", "75", "90")},
        "seconds": time.time() - t0, "device": device,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    exp = registry.new_id("eval_levircd_independent")
    registry.record(experiment_id=exp, model="change_mask", status="done", checkpoint=str(args.checkpoint),
                    checkpoint_sha256=report["checkpoint_sha256"], dataset_manifest_hash=report["index_sha256"],
                    test_metrics={"f1": pooled["0.5"]["f1"], "iou": pooled["0.5"]["iou"], "precision": pooled["0.5"]["precision"],
                                  "recall": pooled["0.5"]["recall"], "n": len(rows)},
                    notes="independent re-evaluation through the deployed loader; per-tile bootstrap CI")
    print(json.dumps({k: v for k, v in report.items() if k not in ("pooled", "val_selected_threshold")}, indent=1))
    if val_choice:
        print("val-selected threshold", val_choice["threshold"], json.dumps(val_choice["test_at_val_threshold"]))
    print(json.dumps(pooled["0.5"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

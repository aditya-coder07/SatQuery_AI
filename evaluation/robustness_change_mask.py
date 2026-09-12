"""Corruption robustness of a change-mask checkpoint on the LEVIR-CD test split.

Conditions (applied identically to both dates unless stated):

* clean;
* gaussian noise sigma 0.02 / 0.05 (on the 0-1 scale);
* gaussian blur sigma 1.0;
* brightness -20% / +20%; contrast x0.7; gamma 1.5;
* **date_swap**: (b, a) instead of (a, b) - the model is symmetric by
  construction, so F1 must match `clean` to numerical precision;
* **asymmetric_brightness**: +20% on date B only - the illumination change
  a real revisit shows and the one a difference-based detector confuses
  with change.

Reports change-class F1 / IoU / precision / recall per condition and the
drop relative to clean. Read-only; writes `--out` only.

Usage::

    python evaluation/robustness_change_mask.py --index data/levircd/index.json \
        --ckpt checkpoints/v3/change_mask/best.pt --out artifacts/benchmark_reports/levircd_robustness_v3.json
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
from training.train_change_mask import LevirCD, build_model, f1_iou  # noqa: E402


def gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return np.stack([gaussian_filter(c, sigma) for c in x]).astype("float32")


def corrupt(a: np.ndarray, b: np.ndarray, name: str, rng: np.random.Generator):
    if name == "clean":
        return a, b
    if name.startswith("noise_"):
        s = float(name.split("_")[1])
        return (np.clip(a + rng.normal(0, s, a.shape), 0, 1).astype("float32"),
                np.clip(b + rng.normal(0, s, b.shape), 0, 1).astype("float32"))
    if name == "blur_1.0":
        return gaussian_blur(a, 1.0), gaussian_blur(b, 1.0)
    if name == "brightness_-0.2":
        return np.clip(a - 0.2, 0, 1), np.clip(b - 0.2, 0, 1)
    if name == "brightness_+0.2":
        return np.clip(a + 0.2, 0, 1), np.clip(b + 0.2, 0, 1)
    if name == "contrast_0.7":
        return (np.clip((a - 0.5) * 0.7 + 0.5, 0, 1).astype("float32"),
                np.clip((b - 0.5) * 0.7 + 0.5, 0, 1).astype("float32"))
    if name == "gamma_1.5":
        return np.power(a, 1.5).astype("float32"), np.power(b, 1.5).astype("float32")
    if name == "date_swap":
        return b, a
    if name == "asymmetric_brightness":
        return a, np.clip(b + 0.2, 0, 1).astype("float32")
    raise ValueError(name)


CONDITIONS = ["clean", "noise_0.02", "noise_0.05", "blur_1.0", "brightness_-0.2", "brightness_+0.2",
              "contrast_0.7", "gamma_1.5", "date_swap", "asymmetric_brightness"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--conditions", nargs="*", default=CONDITIONS)
    args = p.parse_args()

    import torch
    from training.common.checkpointing import load_checkpoint, safe_torch_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extra = safe_torch_load(args.ckpt).get("extra") or {}
    model = build_model(extra.get("dim", 16), arch=extra.get("arch", "v1"))
    load_checkpoint(args.ckpt, model, map_location="cpu")
    model = model.to(device).eval()
    rows = json.loads(args.index.read_text(encoding="utf-8"))["splits"]["test"]
    if args.limit:
        rows = rows[: args.limit]
    ds = LevirCD(rows)
    tiles = [ds[i] for i in range(len(ds))]
    report = {"checkpoint": str(args.ckpt), "arch": extra.get("arch", "v1"), "n_tiles": len(tiles),
              "limited_debug_run": bool(args.limit), "conditions": {}}
    for cond in args.conditions:
        rng = np.random.default_rng(0)
        preds, truths, t0 = [], [], time.time()
        with torch.no_grad():
            for s in range(0, len(tiles), args.batch_size):
                chunk = tiles[s:s + args.batch_size]
                a = np.stack([c[0] for c in chunk]); b = np.stack([c[1] for c in chunk]); m = np.stack([c[2] for c in chunk])
                a, b = corrupt(a, b, cond, rng)
                logits = model(torch.from_numpy(np.ascontiguousarray(a)).to(device),
                               torch.from_numpy(np.ascontiguousarray(b)).to(device))
                preds.append(torch.sigmoid(logits.float()).cpu().numpy())
                truths.append(m)
        m = f1_iou(np.concatenate(preds), np.concatenate(truths))
        m["seconds"] = round(time.time() - t0, 1)
        report["conditions"][cond] = m
        clean_f1 = report["conditions"].get("clean", m)["f1"]
        m["f1_drop_vs_clean"] = round(clean_f1 - m["f1"], 4)
        print(f"{cond:<24} F1 {m['f1']:.4f}  IoU {m['iou']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
              f"drop {m['f1_drop_vs_clean']:+.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.limit:
        registry.record(experiment_id=registry.new_id("robustness_change_mask"), checkpoint=str(args.ckpt),
                        status="done", test_metrics={c: {k: v["f1"]} for c, v in report["conditions"].items() for k in ("f1",)},
                        notes="LEVIR-CD test corruption suite", hardware=registry.hardware())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

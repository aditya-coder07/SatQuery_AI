"""Corruption robustness of a land-cover checkpoint on BigEarthNet's official
test split (full layout, `data/ben_v1_full`; a fixed random subsample).

Conditions, applied to the normalised 12-band cube:

* clean;
* gaussian noise sigma 0.1 / 0.3 (in normalised units, i.e. ~0.1-0.3 of a
  band's std);
* brightness x0.8 / x1.2 on reflectance before normalisation;
* rot90 / hflip - dihedral augmentation was trained with, so mAP must
  match clean closely;
* drop_B01 … drop_B12: one band zeroed with its mask bit cleared (the
  tool's missing-band path) - shows which bands the head leans on;
* cartosat_4band: only B02 B03 B04 B08 present (the deployment sensor);
* cloud_patch: a 40x40 block set to a bright constant (a small cloud).

Reports micro / macro mAP per condition and the delta to clean. Read-only.

Usage::

    python evaluation/robustness_landcover.py --data data/ben_v1_full \
        --ckpt checkpoints/v3/landcover_full/best.pt --limit 20000 \
        --out artifacts/benchmark_reports/ben_robustness_landcover_v3.json
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
from training.track_a_full import BAND_NAMES_12, CARTOSAT_IDX_12  # noqa: E402
from training.train_landcover_v3 import metrics  # noqa: E402
from training.v3.ben_memmap import MemmapBigEarthNet  # noqa: E402

CONDITIONS = ["clean", "noise_0.1", "noise_0.3", "bright_0.8", "bright_1.2", "rot90", "hflip",
              "cartosat_4band", "cloud_patch"] + [f"drop_{b}" for b in BAND_NAMES_12]


def corrupt(x: np.ndarray, name: str, stats, rng: np.random.Generator):
    """x is the normalised cube (B,12,H,W). Returns (x', mask)."""
    mask = np.ones((x.shape[0], x.shape[1]), dtype="float32")
    mean, std = stats
    if name == "clean":
        return x, mask
    if name.startswith("noise_"):
        return x + rng.normal(0, float(name.split("_")[1]), x.shape).astype("float32"), mask
    if name.startswith("bright_"):
        f = float(name.split("_")[1])
        raw = x * std[None, :, None, None] + mean[None, :, None, None]
        return ((raw * f - mean[None, :, None, None]) / std[None, :, None, None]).astype("float32"), mask
    if name == "rot90":
        return np.ascontiguousarray(np.rot90(x, 1, (2, 3))), mask
    if name == "hflip":
        return np.ascontiguousarray(x[:, :, :, ::-1]), mask
    if name == "cartosat_4band":
        mask[:] = 0.0
        mask[:, CARTOSAT_IDX_12] = 1.0
        return x, mask
    if name == "cloud_patch":
        y = x.copy()
        raw_bright = (1.0 - mean) / std  # reflectance 1.0 (saturated) in normalised units
        y[:, :, 40:80, 40:80] = raw_bright[None, :, None, None]
        return y, mask
    if name.startswith("drop_"):
        i = BAND_NAMES_12.index(name[5:])
        mask[:, i] = 0.0
        return x, mask
    raise ValueError(name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    import torch

    from evaluation.splits.multires import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, latest, _ = load_model(args.ckpt, torch, device, 64)
    blob = json.loads((latest.parent / "band_stats.json").read_text(encoding="utf-8"))
    stats = (np.asarray(blob["mean"], dtype="float32"), np.asarray(blob["std"], dtype="float32"))
    n_all = np.load(args.data / f"{args.split}_labels19.npy", mmap_mode="r").shape[0]
    picks = np.random.default_rng(args.seed).choice(n_all, size=min(args.limit, n_all), replace=False)
    ds = MemmapBigEarthNet(args.data, args.split, stats, subset=picks)
    rng = np.random.default_rng(args.seed)
    results = {}
    t0 = time.time()
    for cond in CONDITIONS:
        scores, targets = [], []
        with torch.no_grad():
            for s in range(0, len(ds), args.batch):
                x, y = ds.batch(np.arange(s, min(len(ds), s + args.batch)))
                x, m = corrupt(x, cond, stats, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = model(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))
                scores.append(torch.sigmoid(out.float()).cpu().numpy())
                targets.append(y)
        r = metrics(np.concatenate(scores), np.concatenate(targets))
        results[cond] = {"map_micro": r["map_micro"], "map_macro": r["map_macro"], "f1_micro@0.5": r["f1_micro@0.5"]}
        print(f"{cond:16s} micro {r['map_micro']:.4f} macro {r['map_macro']:.4f}", flush=True)
    clean = results["clean"]
    for cond, r in results.items():
        r["delta_micro"] = r["map_micro"] - clean["map_micro"]
    report = {"checkpoint": str(args.ckpt), "checkpoint_sha256": registry.sha256_file(args.ckpt), "split": args.split,
              "n": len(ds), "seed": args.seed, "conditions": results, "seconds": time.time() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    registry.record(experiment_id=registry.new_id("robustness_landcover"), model="landcover_v3", status="done",
                    checkpoint=str(args.ckpt), checkpoint_sha256=report["checkpoint_sha256"],
                    test_metrics={"clean_micro": clean["map_micro"], "worst": min(results, key=lambda c: results[c]["map_micro"]),
                                  "worst_micro": min(r["map_micro"] for r in results.values()), "n": len(ds)},
                    notes="corruption suite on the official test subsample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

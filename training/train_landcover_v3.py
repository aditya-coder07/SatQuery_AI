"""Fine-tune the v3 land-cover encoder on the prepared BigEarthNet shards.

Reads the same HDF5 shards and band statistics as `track_a_full.py`, keeps
its evaluation (mAP over 19 classes, all 12 bands and the Cartosat 4-band
subset, retention = 4-band / 12-band), and adds micro mAP, per-class AP,
and a fixed-threshold macro/micro F1 so the number can be placed against
papers (BigEarthNet results are quoted as micro mAP in most of them).

Training: band dropout (p=0.3, at least 2 bands kept), light geometric
augmentation (dihedral group), BCE, AdamW with a lower learning rate on the
pretrained trunk than on the head, cosine schedule, bf16. Selection by test
mAP is NOT done: BigEarthNet's test shard is scored once at the end; the
checkpoint is the final epoch (there is no separate validation shard in the
prepared subset, and using test for selection would be leakage).

Usage::

    python training/train_landcover_v3.py --data data/ben_full \
        --ckpt-dir checkpoints/v3/landcover --epochs 30 --batch-size 128
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import registry  # noqa: E402
from training.track_a_encoder import average_precision, band_dropout_mask  # noqa: E402
from training.track_a_full import CARTOSAT_IDX_12, ShardedBigEarthNet, compute_stats  # noqa: E402
from training.v3.landcover import N_CLASSES, build_landcover_v3  # noqa: E402


def metrics(scores: np.ndarray, targets: np.ndarray) -> dict:
    per_class = [average_precision(scores[:, c], targets[:, c]) for c in range(N_CLASSES)]
    valid = [v for v in per_class if not np.isnan(v)]
    # Micro AP: every (sample, class) decision pooled into one ranking.
    micro = average_precision(scores.ravel(), targets.ravel())
    pred = scores >= 0.5
    tp = (pred & (targets > 0.5)).sum(0)
    fp = (pred & (targets <= 0.5)).sum(0)
    fn = (~pred & (targets > 0.5)).sum(0)
    f1c = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    return {"map_macro": float(np.mean(valid)), "map_micro": float(micro),
            "f1_macro@0.5": float(np.nanmean(f1c)),
            "f1_micro@0.5": float(2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())),
            "per_class_ap": [None if np.isnan(v) else float(v) for v in per_class]}


def evaluate(model, ds, torch, batch_size, device, keep=None) -> dict:
    model.eval()
    scores, targets = [], []
    n = len(ds)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            idx = np.arange(s, min(n, s + batch_size))
            x, y = ds.batch(idx)
            mask = np.zeros((x.shape[0], x.shape[1]), dtype="float32")
            if keep is None:
                mask[:] = 1.0
            else:
                mask[:, keep] = 1.0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device))
            scores.append(torch.sigmoid(out.float()).cpu().numpy())
            targets.append(y)
    model.train()
    return metrics(np.concatenate(scores), np.concatenate(targets))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="head; the trunk uses --lr * --trunk-lr-scale")
    p.add_argument("--trunk-lr-scale", type=float, default=0.1)
    p.add_argument("--band-dropout", type=float, default=0.3)
    p.add_argument("--no-pretrained", action="store_true", help="ablation: same trunk from scratch")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    import torch.nn as nn

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_paths = sorted(Path(x) for x in glob.glob(str(args.data / "*train*.hdf5")))
    test_paths = sorted(Path(x) for x in glob.glob(str(args.data / "*test*.hdf5")))
    stats = compute_stats(ShardedBigEarthNet(train_paths))
    train_ds = ShardedBigEarthNet(train_paths, stats, preload=True)
    test_ds = ShardedBigEarthNet(test_paths, stats, preload=True)
    n_train = min(len(train_ds), args.limit_train or len(train_ds))
    print(f"train {n_train} | test {len(test_ds)}")

    model = build_landcover_v3(pretrained=not args.no_pretrained).to(device)
    head_params = list(model.head.parameters())
    head_ids = {id(q) for q in head_params}
    trunk_params = [q for q in model.parameters() if id(q) not in head_ids]
    opt = torch.optim.AdamW([{"params": trunk_params, "lr": args.lr * args.trunk_lr_scale},
                             {"params": head_params, "lr": args.lr}], weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = n_train // args.batch_size
    total = steps_per_epoch * args.epochs
    crit = nn.BCEWithLogitsLoss()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_id = registry.new_id("landcover_v3")
    hw = registry.hardware()
    registry.record(experiment_id=exp_id, model="landcover_v3", status="running", seed=args.seed, hardware=hw,
                    architecture="SSL4EO-S12 MoCo ResNet-50 (12-band) + linear head",
                    hyperparameters={k: str(v) for k, v in vars(args).items()}, checkpoint=str(args.ckpt_dir),
                    dataset_versions={"train_shards": [q.name for q in train_paths], "n_train": n_train})
    (args.ckpt_dir / "run_metadata.json").write_text(json.dumps({
        "experiment_id": exp_id, "task": "landcover_v3", "n_train": n_train, "n_test": len(test_ds),
        "pretrained": not args.no_pretrained, "args": {k: str(v) for k, v in vars(args).items()},
        "hardware": hw}, indent=2), encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    step, t0 = 0, time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_ds))[:n_train]
        run = 0.0
        for s in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = np.sort(order[s:s + args.batch_size])
            x, y = train_ds.batch(idx)
            k = int(rng.integers(4))
            x = np.rot90(x, k, (2, 3))
            if rng.random() < 0.5:
                x = x[:, :, :, ::-1]
            x = np.ascontiguousarray(x)
            mask = band_dropout_mask(x.shape[0], x.shape[1], args.band_dropout, rng).astype("float32")
            frac = step / max(1, total)
            warm = min(1.0, (step + 1) / max(1, steps_per_epoch))
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * warm * 0.5 * (1 + np.cos(np.pi * frac))
            xb, yb, mb = (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
                          torch.from_numpy(mask).to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(xb, mb)
            loss = crit(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
        history.append({"epoch": epoch + 1, "loss": run / max(1, steps_per_epoch)})
        print(f"epoch {epoch + 1}/{args.epochs} loss {run / max(1, steps_per_epoch):.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                    "extra": {"arch": "v3", "pretrained": not args.no_pretrained}}, args.ckpt_dir / "last.pt")

    full = evaluate(model, test_ds, torch, args.batch_size, device)
    cart = evaluate(model, test_ds, torch, args.batch_size, device, keep=CARTOSAT_IDX_12)
    result = {"map_all_bands": full["map_macro"], "map_micro_all_bands": full["map_micro"],
              "f1_macro@0.5": full["f1_macro@0.5"], "f1_micro@0.5": full["f1_micro@0.5"],
              "per_class_ap": full["per_class_ap"],
              "map_cartosat_4band": cart["map_macro"], "map_micro_cartosat_4band": cart["map_micro"],
              "retention": cart["map_macro"] / full["map_macro"] if full["map_macro"] else None,
              "n_test": len(test_ds), "n_train": n_train, "history": history,
              "gpu_hours": (time.time() - t0) / 3600, "pretrained": not args.no_pretrained}
    (args.ckpt_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs,
                "extra": {"arch": "v3", "pretrained": not args.no_pretrained}}, args.ckpt_dir / "final.pt")
    (args.ckpt_dir / "band_stats.json").write_text(json.dumps({"mean": stats[0].tolist(), "std": stats[1].tolist()}),
                                                    encoding="utf-8")
    registry.record(experiment_id=exp_id, status="done", duration_s=time.time() - t0, training_steps=step,
                    test_metrics={k: result[k] for k in ("map_all_bands", "map_micro_all_bands", "map_cartosat_4band",
                                                         "retention", "f1_micro@0.5")},
                    checkpoint=str(args.ckpt_dir / "final.pt"),
                    checkpoint_sha256=registry.sha256_file(args.ckpt_dir / "final.pt"), hardware=hw)
    print(json.dumps({k: v for k, v in result.items() if k not in ("history", "per_class_ap")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

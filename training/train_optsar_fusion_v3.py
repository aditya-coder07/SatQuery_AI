"""Train and score the v3 optical-SAR segmentation triad on WHU-OPT-SAR.

Reports, per arm (optical / SAR / fused, plus fused-with-SAR-dropped and
fused-with-optical-dropped):

* mIoU over the 7 labelled classes, per-class IoU, pixel accuracy;
* tile-level multi-label mAP (the Phase 5 metric, kept for continuity);
* **complementarity gain** = fused − max(optical, SAR), in both metrics;
* gain on the **difficult subset**: tiles in the bottom quartile of
  optical-only tile mIoU - the "when does SAR help?" number;
* a paired bootstrap CI on the per-tile gain (fused − optical).

Uses the scene-disjoint index by default (`index_scene_split.json`); pass
`--index data/whu_opt_sar/index.json` to reproduce the leaky split for
comparison, and the report says which it was.

Usage::

    python training/train_optsar_fusion_v3.py \
        --index data/whu_opt_sar/index_scene_split.json \
        --ckpt-dir checkpoints/v3/optsar_fusion --epochs 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import registry  # noqa: E402
from training.common.paths import index_path  # noqa: E402
from training.track_a_encoder import average_precision  # noqa: E402
from training.v3.optsar_fusion import N_CLASSES, build_optsar_fusion_v3  # noqa: E402

SIZE = 256
ARMS = ("optical", "sar", "fused", "fused_no_sar", "fused_no_optical")


def read_tile(path: str, count: int) -> np.ndarray:
    import rasterio

    with rasterio.open(index_path(path)) as src:
        arr = src.read(out_shape=(src.count, SIZE, SIZE)).astype("float32")
    arr = arr[:count] if arr.shape[0] >= count else np.concatenate([arr] + [arr[-1:]] * (count - arr.shape[0]))
    return (arr / 255.0 - 0.5) / 0.25


def read_label(path: str) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(index_path(path)) as src:
        lbl = src.read(1, out_shape=(SIZE, SIZE), resampling=Resampling.nearest)
    return lbl.astype("int64") - 1  # 1..7 -> 0..6; 0 (unlabeled) -> -1 (ignored)


class WHUSeg:
    def __init__(self, rows, augment=False, seed=0):
        self.rows = [r for r in rows if r.get("sar")]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        o, s, y = read_tile(r["optical"], 4), read_tile(r["sar"], 1), read_label(r["label"])
        if self.augment:
            k = int(self.rng.integers(4))
            o, s, y = np.rot90(o, k, (1, 2)), np.rot90(s, k, (1, 2)), np.rot90(y, k)
            if self.rng.random() < 0.5:
                o, s, y = o[:, :, ::-1], s[:, :, ::-1], y[:, ::-1]
            o, s, y = np.ascontiguousarray(o), np.ascontiguousarray(s), np.ascontiguousarray(y)
        return o, s, y


def confusion(pred: np.ndarray, truth: np.ndarray, n: int) -> np.ndarray:
    m = truth >= 0
    return np.bincount(n * truth[m] + pred[m], minlength=n * n).reshape(n, n)


def miou_from(cm: np.ndarray) -> tuple[float, list[float]]:
    inter = np.diag(cm).astype("float64")
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), [None if np.isnan(v) else float(v) for v in iou]


def evaluate(model, ds, torch, batch_size, device) -> dict:
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    cms = {a: np.zeros((N_CLASSES, N_CLASSES), dtype="int64") for a in ARMS}
    tile_miou = {a: [] for a in ARMS}
    scores = {a: [] for a in ARMS}
    presence = []
    with torch.no_grad():
        for o, s, y in loader:
            o, s = o.to(device), s.to(device)
            yn = y.numpy()
            outs = {}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                lo, ls, lf = model(o, s)
                _, _, lf_ns = model(o, s, drop="sar")
                _, _, lf_no = model(o, s, drop="optical")
            outs = {"optical": lo, "sar": ls, "fused": lf, "fused_no_sar": lf_ns, "fused_no_optical": lf_no}
            for b in range(yn.shape[0]):
                t = yn[b]
                presence.append(np.array([(t == c).any() for c in range(N_CLASSES)], dtype="float32"))
            for a, logits in outs.items():
                prob = torch.softmax(logits.float(), dim=1)
                pred = prob.argmax(1).cpu().numpy()
                scores[a].extend(prob.mean(dim=(2, 3)).cpu().numpy())
                for b in range(yn.shape[0]):
                    cm = confusion(pred[b], yn[b], N_CLASSES)
                    cms[a] += cm
                    tile_miou[a].append(miou_from(cm)[0] if (yn[b] >= 0).any() else np.nan)
    model.train()
    y = np.stack(presence)
    out: dict = {"n_tiles": len(ds), "arms": {}}
    for a in ARMS:
        m, per = miou_from(cms[a])
        sc = np.stack(scores[a])
        aps = [average_precision(sc[:, c], y[:, c]) for c in range(N_CLASSES)]
        aps = [v for v in aps if not np.isnan(v)]
        out["arms"][a] = {"miou": m, "per_class_iou": per,
                          "pixel_acc": float(np.diag(cms[a]).sum() / max(1, cms[a].sum())),
                          "tile_map": float(np.mean(aps)) if aps else None}
    best_single = max(out["arms"]["optical"]["miou"], out["arms"]["sar"]["miou"])
    out["complementarity_gain_miou"] = out["arms"]["fused"]["miou"] - best_single
    out["complementarity_gain_tile_map"] = out["arms"]["fused"]["tile_map"] - max(
        out["arms"]["optical"]["tile_map"], out["arms"]["sar"]["tile_map"])
    # Difficult subset and paired bootstrap on per-tile gain.
    to, tf = np.array(tile_miou["optical"]), np.array(tile_miou["fused"])
    ok = ~np.isnan(to) & ~np.isnan(tf)
    to, tf = to[ok], tf[ok]
    gain = tf - to
    rng = np.random.default_rng(0)
    boots = np.sort([gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(2000)])
    q = np.quantile(to, 0.25)
    hard = to <= q
    out["per_tile"] = {
        "n": int(ok.sum()), "mean_gain_fused_minus_optical": float(gain.mean()),
        "gain_ci95": [float(boots[50]), float(boots[1949])],
        "tiles_where_fused_better": float((gain > 0).mean()),
        "difficult_subset": {"n": int(hard.sum()), "optical_miou": float(to[hard].mean()),
                             "fused_miou": float(tf[hard].mean()), "gain": float(gain[hard].mean())},
        "easy_subset": {"n": int((~hard).sum()), "optical_miou": float(to[~hard].mean()),
                        "fused_miou": float(tf[~hard].mean()), "gain": float(gain[~hard].mean())},
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--index", type=Path, default=Path("data/whu_opt_sar/index_scene_split.json"))
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--modality-dropout", type=float, default=0.15,
                   help="per-batch probability of dropping SAR (and, half as often, optical) in the fused stream")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-only", type=Path, default=None, help="score this checkpoint and exit")
    args = p.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = json.loads(args.index.read_text(encoding="utf-8"))
    train_ds = WHUSeg(index["splits"]["train"], augment=True, seed=args.seed)
    val_ds = WHUSeg(index["splits"]["validation"])
    print(f"split: {index.get('split_method')}\ntrain {len(train_ds)} | val {len(val_ds)}")

    model = build_optsar_fusion_v3(dim=args.dim, weights=None if args.eval_only else "IMAGENET1K_V2").to(device)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device, weights_only=False)["model_state_dict"])
        rep = evaluate(model, val_ds, torch, args.batch_size, device)
        rep["split_method"] = index.get("split_method")
        (args.ckpt_dir / "metrics.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1))
        return 0

    exp_id = registry.new_id("optsar_v3")
    hw = registry.hardware()
    registry.record(experiment_id=exp_id, model="optsar_fusion_v3", architecture="dual ImageNet-R50 + gated FPN triad",
                    hyperparameters=vars(args) | {"size": SIZE}, seed=args.seed, hardware=hw, status="running",
                    checkpoint=str(args.ckpt_dir), dataset_versions={"index": str(args.index),
                                                                     "split_method": index.get("split_method")})
    (args.ckpt_dir / "run_metadata.json").write_text(json.dumps({
        "experiment_id": exp_id, "task": "optsar_fusion_v3_segmentation", "split_method": index.get("split_method"),
        "n_train": len(train_ds), "n_val": len(val_ds), "args": {k: str(v) for k, v in vars(args).items()},
        "hardware": hw}, indent=2), encoding="utf-8")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    total = args.epochs * len(loader)
    step, best, history, t0 = 0, {"miou": -1}, [], time.time()
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        for o, s, y in loader:
            lr = args.lr * min(1.0, (step + 1) / len(loader)) * 0.5 * (1 + np.cos(np.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            o, s, y = o.to(device, non_blocking=True), s.to(device, non_blocking=True), y.to(device, non_blocking=True)
            r = rng.random()
            drop = "sar" if r < args.modality_dropout else ("optical" if r < 1.5 * args.modality_dropout else None)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                lo, ls, lf = model(o, s, drop=drop)
            loss = ce(lo.float(), y) + ce(ls.float(), y) + ce(lf.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
        rep = evaluate(model, val_ds, torch, args.batch_size, device)
        arms = {a: round(rep["arms"][a]["miou"], 4) for a in ARMS}
        history.append({"epoch": epoch + 1, "loss": run / len(loader), **arms,
                        "gain": rep["complementarity_gain_miou"]})
        print(f"epoch {epoch + 1}/{args.epochs} loss {run / len(loader):.4f} val {arms} "
              f"gain {rep['complementarity_gain_miou']:+.4f} ({time.time() - t0:.0f}s)", flush=True)
        if rep["arms"]["fused"]["miou"] > best["miou"]:
            best = {"miou": rep["arms"]["fused"]["miou"], "epoch": epoch + 1}
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "extra": {"arch": "v3", "dim": args.dim}}, args.ckpt_dir / "best.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                    "extra": {"arch": "v3", "dim": args.dim}}, args.ckpt_dir / "last.pt")

    model.load_state_dict(torch.load(args.ckpt_dir / "best.pt", map_location=device, weights_only=False)["model_state_dict"])
    rep = evaluate(model, val_ds, torch, args.batch_size, device)
    rep.update({"split_method": index.get("split_method"), "best_epoch": best["epoch"], "history": history,
                "gpu_hours": (time.time() - t0) / 3600, "selected": "best.pt (val fused mIoU)"})
    (args.ckpt_dir / "metrics.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    registry.record(experiment_id=exp_id, status="done", duration_s=time.time() - t0, training_steps=step,
                    validation_metrics={a: rep["arms"][a]["miou"] for a in ARMS} | {
                        "gain_miou": rep["complementarity_gain_miou"], "per_tile": rep["per_tile"]},
                    checkpoint=str(args.ckpt_dir / "best.pt"),
                    checkpoint_sha256=registry.sha256_file(args.ckpt_dir / "best.pt"), hardware=hw)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("arms", "history")}, indent=1))
    print({a: rep["arms"][a]["miou"] for a in ARMS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

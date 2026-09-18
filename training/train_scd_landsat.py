"""Semantic change detection on Landsat-SCD (licensed replacement for SECOND).

Model: one shared ImageNet ResNet-50 encoder over both dates (the same trunk
as `training/v3/change_mask.py`), an FPN decoder over the per-scale
concatenation `[f_a, f_b, |f_a - f_b|]` - a change *type* is an ordered
transition, so the decoder must see which date is which, unlike the binary
detector - and a 10-way head over the dataset's own labels (0 no change,
1..9 change types).

Metrics (test split, originals only):

* OA and mIoU over all 10 classes; mIoU over the 9 change types;
* binary change IoU / F1 (any change type vs none);
* **SeK** (Yang et al. 2021, the SECOND metric): Cohen's kappa over the
  confusion matrix with the no-change/no-change cell zeroed, times
  exp(IoU_change − 1); and Score = 0.3·mIoU + 0.7·SeK;
* bootstrap 95% CI on mIoU over test pairs.

Loss: cross-entropy with median-frequency class weights (no-change is ~81%
of pixels). Augmentation: dihedral, applied identically to both dates and the
label; date order is NOT swapped because the label is an ordered transition.

Usage::

    python training/train_scd_landsat.py --index data/landsat_scd/index.json \
        --ckpt-dir checkpoints/v3/scd_landsat --epochs 40
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
from training.prepare.landsat_scd import N_CLASSES  # noqa: E402

SIZE = 416


def build_scd_model(dim: int = 96, n_classes: int = N_CLASSES, weights: str | None = "IMAGENET1K_V2"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from training.v2.architectures import build_pretrained_backbone

    backbone = build_pretrained_backbone(cin=3, weights=weights)

    def conv_bn(cin, cout, k=3):
        return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class SCDNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = backbone
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            widths = backbone.widths
            self.lateral = nn.ModuleList([conv_bn(3 * w, dim, 1) for w in widths])
            self.smooth = nn.ModuleList([conv_bn(dim, dim) for _ in widths])
            self.head = nn.Sequential(conv_bn(dim, dim), nn.Conv2d(dim, n_classes, 1))

        def forward(self, a, b):
            fa = self.encoder((a - self.mean) / self.std)
            fb = self.encoder((b - self.mean) / self.std)
            feats = [lat(torch.cat([x, y, torch.abs(x - y)], 1)) for lat, x, y in zip(self.lateral, fa, fb)]
            h = self.smooth[-1](feats[-1])
            for i in range(len(feats) - 2, -1, -1):
                h = F.interpolate(h, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
                h = self.smooth[i](h + feats[i])
            return F.interpolate(self.head(h), size=a.shape[-2:], mode="bilinear", align_corners=False)

    return SCDNet()


class LandsatSCD:
    def __init__(self, rows, augment=False, seed=0):
        self.rows, self.augment = rows, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        r = self.rows[i]
        a = np.asarray(Image.open(index_path(r["a"])).convert("RGB"), dtype="float32") / 255.0
        b = np.asarray(Image.open(index_path(r["b"])).convert("RGB"), dtype="float32") / 255.0
        y = np.asarray(Image.open(index_path(r["label"])).convert("L"), dtype="int64")
        if self.augment:
            k = int(self.rng.integers(4))
            a, b, y = np.rot90(a, k), np.rot90(b, k), np.rot90(y, k)
            if self.rng.random() < 0.5:
                a, b, y = a[:, ::-1], b[:, ::-1], y[:, ::-1]
            a, b, y = np.ascontiguousarray(a), np.ascontiguousarray(b), np.ascontiguousarray(y)
        return a.transpose(2, 0, 1), b.transpose(2, 0, 1), y


def scd_metrics(cm: np.ndarray) -> dict:
    n = cm.shape[0]
    inter = np.diag(cm).astype("float64")
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    # Binary change: class 0 vs everything else.
    tp = cm[1:, 1:].sum()
    fp = cm[0, 1:].sum()
    fn = cm[1:, 0].sum()
    iou_change = tp / max(1, tp + fp + fn)
    f1_change = 2 * tp / max(1, 2 * tp + fp + fn)
    # SeK: kappa on the matrix with the (0, 0) cell removed, scaled by change IoU.
    q = cm.astype("float64").copy()
    q[0, 0] = 0
    total = q.sum()
    po = np.trace(q) / max(1e-9, total)
    pe = (q.sum(0) * q.sum(1)).sum() / max(1e-9, total**2)
    kappa = (po - pe) / max(1e-9, 1 - pe)
    sek = kappa * float(np.exp(iou_change - 1))
    miou_all = float(np.nanmean(iou))
    return {"oa": float(inter.sum() / max(1, cm.sum())), "miou_all": miou_all,
            "miou_change_types": float(np.nanmean(iou[1:])), "iou_change_binary": float(iou_change),
            "f1_change_binary": float(f1_change), "sek": float(sek), "kappa_change": float(kappa),
            "score": 0.3 * miou_all + 0.7 * float(sek),
            "per_class_iou": [None if np.isnan(v) else float(v) for v in iou]}


def evaluate(model, ds, torch, batch_size, device, per_pair=False):
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype="int64")
    pair_cms = []
    with torch.no_grad():
        for a, b, y in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(a.to(device), b.to(device))
            pred = logits.argmax(1).cpu().numpy()
            yn = y.numpy()
            for i in range(yn.shape[0]):
                c = np.bincount(N_CLASSES * yn[i].ravel() + pred[i].ravel(), minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
                cm += c
                if per_pair:
                    pair_cms.append(c)
    model.train()
    out = scd_metrics(cm)
    if per_pair:
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(500):
            idx = rng.integers(0, len(pair_cms), len(pair_cms))
            boots.append(scd_metrics(sum(pair_cms[j] for j in idx))["miou_all"])
        boots.sort()
        out["miou_all_ci95"] = [float(boots[12]), float(boots[487])]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--index", type=Path, default=Path("data/landsat_scd/index.json"))
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = json.loads(args.index.read_text(encoding="utf-8"))
    train_rows = index["splits"]["train"][: args.limit_train] if args.limit_train else index["splits"]["train"]
    val_rows = [r for r in index["splits"]["val"] if not r.get("augmented")]
    test_rows = [r for r in index["splits"]["test"] if not r.get("augmented")]
    train_ds, val_ds, test_ds = (LandsatSCD(train_rows, True, args.seed), LandsatSCD(val_rows), LandsatSCD(test_rows))
    print(f"split: {index.get('split_method')}\ntrain {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    # Median-frequency class weights from a sample of the training labels.
    counts = np.zeros(N_CLASSES, dtype="float64")
    for i in range(0, len(train_ds), max(1, len(train_ds) // 200)):
        counts += np.bincount(train_ds[i][2].ravel(), minlength=N_CLASSES)
    freq = counts / counts.sum()
    present = freq > 0
    weights = np.ones(N_CLASSES, dtype="float32")
    weights[present] = np.median(freq[present]) / freq[present]
    weights = np.clip(weights, 0.1, 10.0)
    print("class weights", np.round(weights, 2).tolist())

    model = build_scd_model(dim=args.dim).to(device)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_id = registry.new_id("scd_landsat")
    hw = registry.hardware()
    registry.record(experiment_id=exp_id, model="scd_landsat_v3", status="running", seed=args.seed, hardware=hw,
                    architecture="siamese ImageNet-R50 + FPN over [fa, fb, |fa-fb|], 10-way change-type head",
                    hyperparameters={k: str(v) for k, v in vars(args).items()}, checkpoint=str(args.ckpt_dir),
                    dataset_versions={"index": str(args.index), "split_method": index.get("split_method")})
    (args.ckpt_dir / "run_metadata.json").write_text(json.dumps({
        "experiment_id": exp_id, "task": "semantic_change_landsat_scd", "n_train": len(train_ds),
        "n_val": len(val_ds), "n_test": len(test_ds), "split_method": index.get("split_method"),
        "license": index.get("license"), "args": {k: str(v) for k, v in vars(args).items()}, "hardware": hw},
        indent=2), encoding="utf-8")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    total = args.epochs * len(loader)
    step, best, history, t0 = 0, {"score": -1.0}, [], time.time()
    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        for a, b, y in loader:
            lr = args.lr * min(1.0, (step + 1) / len(loader)) * 0.5 * (1 + np.cos(np.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            a, b, y = a.to(device, non_blocking=True), b.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(a, b)
            loss = ce(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
        vm = evaluate(model, val_ds, torch, args.batch_size, device)
        history.append({"epoch": epoch + 1, "loss": run / len(loader), **{k: vm[k] for k in ("miou_all", "sek", "score", "f1_change_binary")}})
        print(f"epoch {epoch + 1}/{args.epochs} loss {run / len(loader):.4f} val mIoU {vm['miou_all']:.4f} "
              f"SeK {vm['sek']:.4f} score {vm['score']:.4f} F1chg {vm['f1_change_binary']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vm["score"] > best["score"]:
            best = {"score": vm["score"], "epoch": epoch + 1}
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "extra": {"arch": "scd_v3", "dim": args.dim, "n_classes": N_CLASSES}}, args.ckpt_dir / "best.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                    "extra": {"arch": "scd_v3", "dim": args.dim, "n_classes": N_CLASSES}}, args.ckpt_dir / "last.pt")

    model.load_state_dict(torch.load(args.ckpt_dir / "best.pt", map_location=device, weights_only=False)["model_state_dict"])
    test = evaluate(model, test_ds, torch, args.batch_size, device, per_pair=True)
    result = {"test": test, "best_val": best, "history": history, "gpu_hours": (time.time() - t0) / 3600,
              "split_method": index.get("split_method"), "license": index.get("license"),
              "selected": "best.pt (val score = 0.3 mIoU + 0.7 SeK)"}
    (args.ckpt_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    registry.record(experiment_id=exp_id, status="done", duration_s=time.time() - t0, training_steps=step,
                    test_metrics={k: test[k] for k in ("oa", "miou_all", "miou_change_types", "iou_change_binary",
                                                       "f1_change_binary", "sek", "score", "miou_all_ci95")},
                    validation_metrics=best, checkpoint=str(args.ckpt_dir / "best.pt"),
                    checkpoint_sha256=registry.sha256_file(args.ckpt_dir / "best.pt"), hardware=hw)
    print(json.dumps({k: v for k, v in test.items() if k != "per_class_iou"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

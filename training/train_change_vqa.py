"""Semantic change head for `change_vqa_v1` (plan task 2.6, the learned half).

`change_vqa_v1` shipped with only its deterministic index path, which cannot
fire on CDVQA because that benchmark's imagery is RGB and every classical
index needs NIR or SWIR. The measured consequence was 0.0000 on the PS's
prescribed change-VQA benchmark.

`satquery/verify/semantic_change.py` then established that **all eight CDVQA
question types are exact arithmetic over a pair of semantic change maps** -
0.9975 from ground-truth maps on the full test split. So the learned problem
is not question answering at all. It is one segmentation task: given two dates
of RGB, predict each date's class map over SECOND's six change classes plus
"unchanged". Everything after that is counting pixels.

Architecture, following `train_change_mask.py`'s reasoning: a **shared**
encoder sees both dates, because two independent encoders can drift so that
identical input produces different features, which appears as change that is
not there. Each date's decoder then sees its own features **concatenated with
the absolute difference**, so "did this pixel change" and "what is it now" are
decided from the same evidence.

## The split, and the leak it would be easy to walk into

CDVQA's splits partition SECOND's 2,968 labelled pairs: **1,600 train, 400
val, 968 test**. Every CDVQA image id resolves in SECOND, which is what makes
the full-coverage benchmark manifest possible - and it means **SECOND ships
the labels for the 968 test pairs too**. Training on all of SECOND would leak
the benchmark completely and produce a beautiful, worthless number.

This script therefore takes its ids from CDVQA's own `Train_images.json` and
`Val_images.json` and never reads the test ids. `--split-check` prints the
overlap so the property is verified rather than trusted.

Usage:
    python training/train_change_vqa.py --second data/second \
        --annotations data/cdvqa --ckpt-dir checkpoints/change_vqa --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satquery.verify.semantic_change import CLASSES, decode_label  # noqa: E402
from training.common.checkpointing import (  # noqa: E402
    maybe_resume,
    save_checkpoint,
    set_seed,
    write_run_metadata,
)

N_CLASSES = len(CLASSES)
CROP = 256


def build_model(dim: int = 32):
    import torch
    import torch.nn as nn

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class SemanticChangeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = block(3, dim)
            self.enc2 = block(dim, dim * 2, stride=2)
            self.enc3 = block(dim * 2, dim * 4, stride=2)
            self.enc4 = block(dim * 4, dim * 4)

            def decoder():
                return nn.Sequential(
                    # Own features and the difference, side by side.
                    block(dim * 8, dim * 4),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    block(dim * 4, dim * 2),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    block(dim * 2, dim),
                    nn.Conv2d(dim, N_CLASSES, 1),
                )

            # Two decoders, one per date. They must not be shared: the same
            # pixel is "buildings" at one date and "trees" at the other, and a
            # shared decoder would have to resolve that from features alone.
            self.dec1 = decoder()
            self.dec2 = decoder()

        def encode(self, x):
            return self.enc4(self.enc3(self.enc2(self.enc1(x))))

        def forward(self, a, b):
            fa, fb = self.encode(a), self.encode(b)
            diff = torch.abs(fa - fb)
            return (
                self.dec1(torch.cat([fa, diff], dim=1)),
                self.dec2(torch.cat([fb, diff], dim=1)),
            )

    return SemanticChangeNet()


def split_ids(annotations: Path, split: str) -> list[str]:
    """Image ids for one CDVQA split, from the official image index."""
    path = annotations / f"{split}_images.json"
    records = json.loads(path.read_text(encoding="utf-8"))["images"]
    return sorted({r["file_name"] for r in records})


class SecondPairs:
    """CDVQA-split image pairs with their SECOND semantic labels."""

    def __init__(self, second: Path, ids: list[str], crop: int | None = CROP,
                 seed: int = 0):
        self.second = second
        self.ids = ids
        self.crop = crop
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        from PIL import Image

        name = self.ids[i]
        a = np.asarray(Image.open(self.second / "im1" / name).convert("RGB"))
        b = np.asarray(Image.open(self.second / "im2" / name).convert("RGB"))
        l1 = decode_label(
            np.asarray(Image.open(self.second / "label1" / name).convert("RGB"))
        )
        l2 = decode_label(
            np.asarray(Image.open(self.second / "label2" / name).convert("RGB"))
        )

        if self.crop and a.shape[0] > self.crop:
            top = int(self.rng.integers(0, a.shape[0] - self.crop + 1))
            left = int(self.rng.integers(0, a.shape[1] - self.crop + 1))
            sl = (slice(top, top + self.crop), slice(left, left + self.crop))
            a, b, l1, l2 = a[sl], b[sl], l1[sl], l2[sl]

        return (
            a.transpose(2, 0, 1).astype("float32") / 255.0,
            b.transpose(2, 0, 1).astype("float32") / 255.0,
            l1.astype("int64"),
            l2.astype("int64"),
        )


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        a, b, l1, l2 = zip(*(dataset[int(i)] for i in idx))
        yield np.stack(a), np.stack(b), np.stack(l1), np.stack(l2)


def class_weights(dataset, n: int = 64) -> np.ndarray:
    """Median-frequency balancing (Eigen & Fergus) over the sampled classes.

    "unchanged" is about 80% of pixels; unweighted cross-entropy converges to
    predicting it everywhere, which scores well per-pixel and answers every
    CDVQA question wrong.

    Plain inverse frequency does not work here, and the failure is not subtle:
    playgrounds are 0.14% of pixels and absent altogether from a small sample,
    so its weight goes to the frequency floor's reciprocal, dominates the
    normalising mean, and drives **every other class to zero**. A smoke run
    showed exactly that - `playgrounds=7.00` and all six others at `0.00`.
    Dividing the median frequency by each class's own frequency is bounded by
    construction, and classes absent from the sample get weight 1 rather than
    the largest weight in the vector.
    """
    counts = np.zeros(N_CLASSES, dtype="float64")
    for i in range(min(n, len(dataset))):
        _, _, l1, l2 = dataset[i]
        for labels in (l1, l2):
            counts += np.bincount(labels.ravel(), minlength=N_CLASSES)

    frequency = counts / max(counts.sum(), 1)
    present = frequency > 0
    weights = np.ones(N_CLASSES, dtype="float64")
    if present.any():
        median = float(np.median(frequency[present]))
        weights[present] = median / frequency[present]
    # A rare class still must not outweigh the rest by orders of magnitude:
    # the gradient it contributes is noisy, and an unbounded weight turns that
    # noise into the dominant training signal.
    return np.clip(weights, 0.1, 10.0).astype("float32")


def segmentation_metrics(confusion: np.ndarray) -> dict:
    """Per-class IoU and the two aggregates worth reporting."""
    tp = np.diag(confusion).astype("float64")
    fp = confusion.sum(0) - tp
    fn = confusion.sum(1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = denom > 0
    return {
        "pixel_accuracy": float(tp.sum() / max(confusion.sum(), 1)),
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        # The change classes are what CDVQA asks about; "unchanged" dominates
        # the pixel count and would flatter the mean.
        "miou_change_classes": float(iou[1:][present[1:]].mean())
        if present[1:].any()
        else 0.0,
        "iou_per_class": {name: float(iou[i]) for i, name in enumerate(CLASSES)},
    }


def evaluate(model, dataset, torch, batch_size, device) -> dict:
    model.eval()
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype="int64")
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for a, b, l1, l2 in batches(dataset, batch_size, rng, shuffle=False):
            p1, p2 = model(
                torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
            )
            for logits, truth in ((p1, l1), (p2, l2)):
                pred = logits.argmax(1).cpu().numpy().ravel()
                flat = truth.ravel()
                np.add.at(confusion, (flat, pred), 1)
    return segmentation_metrics(confusion)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--second", type=Path, default=Path("data/second"))
    p.add_argument("--annotations", type=Path, default=Path("data/cdvqa"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/change_vqa"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--crop", type=int, default=CROP)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--split-check",
        action="store_true",
        help="print the train/val/test id overlap and exit",
    )
    args = p.parse_args()

    train_ids = split_ids(args.annotations, "Train")
    val_ids = split_ids(args.annotations, "Val")
    test_ids = split_ids(args.annotations, "Test")

    if args.split_check:
        print(f"train {len(train_ids)}  val {len(val_ids)}  test {len(test_ids)}")
        print(f"train n val  : {len(set(train_ids) & set(val_ids))}")
        print(f"train n test : {len(set(train_ids) & set(test_ids))}")
        print(f"val   n test : {len(set(val_ids) & set(test_ids))}")
        print(f"union        : {len(set(train_ids) | set(val_ids) | set(test_ids))}")
        return 0

    leaked = set(train_ids) & set(test_ids)
    if leaked:
        raise SystemExit(f"{len(leaked)} train ids are in the test split; refusing to train")

    import torch
    import torch.nn as nn

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.limit_train:
        train_ids = train_ids[: args.limit_train]

    train_ds = SecondPairs(args.second, train_ids, crop=args.crop, seed=args.seed)
    # Validation is scored on whole 512px scenes, because that is the input
    # the benchmark actually presents.
    val_ds = SecondPairs(args.second, val_ids, crop=None)
    print(f"train {len(train_ds)} pairs | val {len(val_ds)} pairs")

    model = build_model(args.dim).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"parameters: {n_params/1e6:.3f}M")

    weights = class_weights(train_ds)
    print("class weights: " + ", ".join(
        f"{name}={w:.2f}" for name, w in zip(CLASSES, weights)
    ))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    state, _ = maybe_resume(args.ckpt_dir, model, optimizer, enabled=args.resume)

    write_run_metadata(args.ckpt_dir, {
        "task": "semantic_change_second",
        "n_train_pairs": len(train_ds),
        "n_val_pairs": len(val_ds),
        "split_source": "CDVQA Train/Val image ids; test ids never read",
        "epochs": args.epochs, "lr": args.lr, "dim": args.dim, "crop": args.crop,
        "n_params": n_params,
        "class_weights": {n: float(w) for n, w in zip(CLASSES, weights)},
    })

    rng = np.random.default_rng(args.seed)
    step = state.step
    started = time.time()
    best = -1.0

    for epoch in range(state.epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        for a, b, l1, l2 in batches(train_ds, args.batch_size, rng):
            ab = torch.from_numpy(a).to(device)
            bb = torch.from_numpy(b).to(device)
            t1 = torch.from_numpy(l1).to(device)
            t2 = torch.from_numpy(l2).to(device)

            p1, p2 = model(ab, bb)
            loss = criterion(p1, t1) + criterion(p2, t2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * a.shape[0]
            seen += a.shape[0]
            step += 1
            if step % args.save_every == 0:
                state.step, state.epoch = step, epoch
                save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)

        message = (f"epoch {epoch+1}/{args.epochs}  loss {running/max(seen,1):.4f}  "
                   f"({time.time()-started:.0f}s)")

        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            metrics = evaluate(model, val_ds, torch, max(1, args.batch_size // 2), device)
            message += (f"  val mIoU {metrics['miou']:.4f}  "
                        f"change-class mIoU {metrics['miou_change_classes']:.4f}")
            if metrics["miou_change_classes"] > best:
                best = metrics["miou_change_classes"]
                # Selection is on the change classes, so the best checkpoint
                # is kept separately from the rolling step checkpoints, which
                # `save_checkpoint` prunes.
                args.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "dim": args.dim,
                     "classes": list(CLASSES), "step": step, "epoch": epoch + 1},
                    args.ckpt_dir / "best.pt",
                )
                (args.ckpt_dir / "metrics.json").write_text(
                    json.dumps({**metrics, "epoch": epoch + 1, "step": step}, indent=2),
                    encoding="utf-8",
                )
        print(message, flush=True)

    state.step, state.epoch = step, args.epochs
    save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)
    print(f"\nbest val change-class mIoU: {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

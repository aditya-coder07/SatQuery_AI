"""Track A full: 12-band BigEarthNet with GSD conditioning (plan task 2.1).

Scales up Track A v0 (task 1.10), which trained on 7,180 patches - 1.5% of
BigEarthNet - and reached mAP 0.42 against a published ~0.65-0.85. Three
changes, each named in the plan's 2.1 row:

* **All 12 Sentinel-2 bands**, not the 10-band subset. The shards carry the
  60 m atmospheric bands too, and band dropout means extra bands cost nothing
  at inference on a sensor that lacks them.
* **Band-presence masking**, already the architecture's basis, now exercised
  across a much wider band set.
* **GSD conditioning** - the encoder is told the ground sample distance of
  its input. The cross-sensor test showed resolution, not bands, is the
  dominant transfer gap (vegetation agreement 0.455 at 10 m against 0.161 at
  1.6 m), so a model that cannot represent its own input scale has no way to
  compensate for it.

Data comes from the HDF5 shards, which embed `images` (N,12,120,120) and
`labels19` (N,19) directly - no metadata join, and no per-file open, which is
what made the v0 loader the bottleneck rather than the GPU.

Usage:
    python training/track_a_full.py --data data/ben_full --ckpt-dir checkpoints/track_a_full \
        --epochs 3 --ablation
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

from training.common.checkpointing import (  # noqa: E402
    TrainingState, maybe_resume, save_checkpoint, set_seed, write_run_metadata,
)
from training.track_a_encoder import (  # noqa: E402
    average_precision, band_dropout_mask, mean_average_precision,
)

# BigEarthNet 12-band order as stored in these shards.
BAND_NAMES_12 = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]
CANONICAL_12 = {
    "B02": "BLUE", "B03": "GREEN", "B04": "RED", "B08": "NIR",
    "B11": "SWIR1", "B12": "SWIR2",
}
# Cartosat-2E MX: 4-band VNIR, no SWIR (verification item 6).
CARTOSAT_BANDS_12 = ["B02", "B03", "B04", "B08"]
CARTOSAT_IDX_12 = [BAND_NAMES_12.index(b) for b in CARTOSAT_BANDS_12]

N_CLASSES = 19
REFLECTANCE_SCALE = 10000.0
BEN_GSD_M = 10.0


class ShardedBigEarthNet:
    """Reads directly from HDF5 shards, keeping files open across items."""

    def __init__(self, paths: list[Path], stats=None):
        import h5py

        self.files = [h5py.File(p, "r") for p in paths]
        self.offsets, total = [], 0
        for f in self.files:
            self.offsets.append(total)
            total += f["images"].shape[0]
        self.total = total
        self.stats = stats

    def __len__(self) -> int:
        return self.total

    def _locate(self, i: int) -> tuple:
        shard = 0
        for s, offset in enumerate(self.offsets):
            if i >= offset:
                shard = s
        return self.files[shard], i - self.offsets[shard]

    def __getitem__(self, i: int):
        f, local = self._locate(int(i))
        image = np.asarray(f["images"][local], dtype="float32") / REFLECTANCE_SCALE
        if self.stats is not None:
            mean, std = self.stats
            image = (image - mean[:, None, None]) / std[:, None, None]
        target = np.asarray(f["labels19"][local], dtype="float32")
        return image, target

    def close(self) -> None:
        for f in self.files:
            f.close()


def compute_stats(dataset, sample: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dataset), size=min(sample, len(dataset)), replace=False)
    acc = [dataset[int(i)][0].reshape(12, -1) for i in picks]
    stacked = np.concatenate(acc, axis=1)
    mean, std = stacked.mean(axis=1), stacked.std(axis=1)
    std[std < 1e-6] = 1.0
    return mean.astype("float32"), std.astype("float32")


def build_model(n_bands: int = 12, dim: int = 96, gsd_conditioning: bool = True):
    import torch
    import torch.nn as nn

    class BandAgnosticGSD(nn.Module):
        """Band-agnostic encoder with optional GSD conditioning."""

        def __init__(self) -> None:
            super().__init__()
            self.gsd_conditioning = gsd_conditioning
            self.stem = nn.Sequential(
                nn.Conv2d(1, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            )
            self.band_embed = nn.Parameter(torch.zeros(n_bands, dim))

            # GSD enters as a FiLM-style scale/shift on the pooled band
            # features. A model with no representation of its own input scale
            # cannot compensate for a resolution change; this gives it one,
            # and log-scaling keeps 1.6 m and 10 m a modest distance apart
            # rather than an order of magnitude.
            if gsd_conditioning:
                self.gsd_mlp = nn.Sequential(
                    nn.Linear(1, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim * 2)
                )

            self.trunk = nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim * 2), nn.ReLU(inplace=True),
                nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim * 4), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(dim * 4, N_CLASSES)

        def forward(self, x, mask, gsd=None):
            b, c, h, w = x.shape
            x = x * mask[:, :, None, None]
            z = self.stem(x.reshape(b * c, 1, h, w)).reshape(b, c, -1, h, w)
            z = z + self.band_embed[None, :, :, None, None]

            m = mask[:, :, None, None, None]
            z = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

            if self.gsd_conditioning and gsd is not None:
                params = self.gsd_mlp(torch.log(gsd.clamp(min=1e-3)).unsqueeze(-1))
                scale, shift = params.chunk(2, dim=-1)
                z = z * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

            return self.head(self.trunk(z).flatten(1))

    return BandAgnosticGSD()


# Resolutions simulated by downsampling. BigEarthNet is uniformly 10 m, so
# without this the GSD input is constant during training and the conditioning
# has nothing to learn from - it would be present in the architecture and
# inert in practice. Downsampling by k and upsampling back produces a patch
# whose *effective* resolution is 10*k m at unchanged tensor shape, which is
# what the Cartosat path will actually present.
MULTIRES_FACTORS = (1, 2, 3, 4)


def degrade_resolution(images: np.ndarray, factor: int) -> np.ndarray:
    """Blur to a coarser effective GSD, keeping the array shape."""
    if factor <= 1:
        return images
    n, c, h, w = images.shape
    small_h, small_w = max(1, h // factor), max(1, w // factor)
    # Block-average down, then repeat back up. Cheap, and it removes exactly
    # the high-frequency detail a coarser sensor would not have resolved.
    trimmed = images[:, :, : small_h * factor, : small_w * factor]
    down = trimmed.reshape(n, c, small_h, factor, small_w, factor).mean(axis=(3, 5))
    up = np.repeat(np.repeat(down, factor, axis=2), factor, axis=3)
    out = np.zeros_like(images)
    out[:, :, : up.shape[2], : up.shape[3]] = up[:, :, :h, :w]
    return out


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = np.sort(order[start : start + size])  # sorted reads are far faster in HDF5
        xs, ys = zip(*(dataset[int(i)] for i in idx))
        yield np.stack(xs), np.stack(ys)


def evaluate(model, dataset, torch, batch_size, device, keep=None, gsd=BEN_GSD_M):
    model.eval()
    rng = np.random.default_rng(0)
    scores, targets = [], []
    with torch.no_grad():
        for x, y in batches(dataset, batch_size, rng, shuffle=False):
            mask = np.zeros((x.shape[0], x.shape[1]), dtype="float32")
            if keep is None:
                mask[:] = 1.0
            else:
                mask[:, keep] = 1.0
            out = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(mask).to(device),
                torch.full((x.shape[0],), gsd, device=device),
            )
            scores.append(torch.sigmoid(out).cpu().numpy())
            targets.append(y)
    return mean_average_precision(np.concatenate(scores), np.concatenate(targets))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/track_a_full"))
    p.add_argument("--epochs", type=int, default=3)
    # MEMORY: the per-band stem reshapes to (batch * bands, 1, H, W), so
    # activation memory scales with batch x bands, not batch alone. At 12
    # bands a batch of 128 becomes 1,536 images through the stem - about
    # 8.5 GiB at dim=96, which OOMs a 6 GB card. Keep batch * bands * dim
    # within roughly 25,000 on 6 GB.
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--band-dropout", type=float, default=0.3)
    p.add_argument("--no-gsd", action="store_true", help="disable GSD conditioning")
    p.add_argument(
        "--multires", action="store_true",
        help="augment with simulated coarser resolutions so GSD conditioning "
             "has a varying signal to learn from",
    )
    p.add_argument("--limit-eval", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ablation", action="store_true")
    args = p.parse_args()

    import torch
    import torch.nn as nn

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_paths = sorted(Path(x) for x in glob.glob(str(args.data / "*train*.hdf5")))
    test_paths = sorted(Path(x) for x in glob.glob(str(args.data / "*test*.hdf5")))
    if not train_paths:
        print(f"no train shards in {args.data}", file=sys.stderr)
        return 1
    print(f"train shards: {[p.name for p in train_paths]}")
    print(f"test shards : {[p.name for p in test_paths]}")

    raw_train = ShardedBigEarthNet(train_paths)
    print("computing band statistics...")
    stats = compute_stats(raw_train)
    raw_train.close()

    train_ds = ShardedBigEarthNet(train_paths, stats)
    test_ds = ShardedBigEarthNet(test_paths, stats) if test_paths else None
    print(f"train {len(train_ds)} patches" + (f" | test {len(test_ds)}" if test_ds else ""))

    model = build_model(dim=args.dim, gsd_conditioning=not args.no_gsd).to(device)
    print(f"parameters: {sum(q.numel() for q in model.parameters())/1e6:.2f}M")

    budget = args.batch_size * len(BAND_NAMES_12) * args.dim
    print(f"stem load: batch {args.batch_size} x 12 bands x dim {args.dim} = {budget}")
    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gb < 8 and budget > 30000:
            print(
                f"WARNING: {total_gb:.1f} GiB GPU with stem load {budget}; "
                "reduce --batch-size or --dim if this OOMs"
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    state, _ = maybe_resume(args.ckpt_dir, model, optimizer, enabled=args.resume)

    # The band statistics are part of the model: inference MUST apply the same
    # transform, and recomputing them needs the training shards, which a
    # deployment does not have. They were not saved, and the consequence was
    # measured - satquery/tools/landcover.py standardised per image instead
    # and the head asserted class 0 on every patch at 0.9 confidence, wrong
    # every time. Saved beside the weights so the tool can refuse to run
    # without them.
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    (args.ckpt_dir / "band_stats.json").write_text(
        json.dumps({
            "bands": BAND_NAMES_12,
            "reflectance_scale": REFLECTANCE_SCALE,
            "mean": [float(v) for v in stats[0]],
            "std": [float(v) for v in stats[1]],
            "provenance": (
                f"compute_stats(seed=0, sample=2000) over "
                f"{[p.name for p in train_paths]}"
            ),
        }, indent=2),
        encoding="utf-8",
    )

    write_run_metadata(args.ckpt_dir, {
        "task": "track_a_full", "bands": BAND_NAMES_12,
        "n_train": len(train_ds), "epochs": args.epochs, "lr": args.lr,
        "dim": args.dim, "band_dropout": args.band_dropout,
        "gsd_conditioning": not args.no_gsd, "multires_augmentation": args.multires,
        "seed": args.seed,
    })

    rng = np.random.default_rng(args.seed)
    step = state.step
    started = time.time()

    for epoch in range(state.epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in batches(train_ds, args.batch_size, rng):
            mask = band_dropout_mask(x.shape[0], x.shape[1], args.band_dropout, rng)

            gsd_value = BEN_GSD_M
            if args.multires:
                factor = int(rng.choice(MULTIRES_FACTORS))
                x = degrade_resolution(x, factor)
                gsd_value = BEN_GSD_M * factor

            out = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(mask).to(device),
                torch.full((x.shape[0],), gsd_value, device=device),
            )
            loss = criterion(out, torch.from_numpy(y).to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
            step += 1
            if step % args.save_every == 0:
                state.step, state.epoch = step, epoch
                save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)
                print(f"  step {step}  loss {running/max(seen,1):.4f}", flush=True)

        print(f"epoch {epoch+1}/{args.epochs}  loss {running/max(seen,1):.4f}  "
              f"({time.time()-started:.0f}s)", flush=True)

    state.step, state.epoch = step, args.epochs
    save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)

    results = {}
    if test_ds:
        if args.limit_eval:
            test_ds.total = min(test_ds.total, args.limit_eval)
        full, per_class = evaluate(model, test_ds, torch, args.batch_size, device)
        print(f"\nofficial test mAP, all 12 bands : {full:.4f}")
        results["map_all_bands"] = full

        if args.ablation:
            cart, _ = evaluate(
                model, test_ds, torch, args.batch_size, device, keep=CARTOSAT_IDX_12
            )
            print(f"  Cartosat 4-band VNIR         : {cart:.4f} "
                  f"({cart/full*100:.1f}% retained)")
            results["map_cartosat_4band"] = cart
            results["retention"] = cart / full if full else float("nan")

        results["per_class_ap"] = {
            f"class_{i}": (None if np.isnan(v) else round(v, 6))
            for i, v in enumerate(per_class)
        }
        (args.ckpt_dir / "metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(f"\nMetrics -> {args.ckpt_dir / 'metrics.json'}")

    train_ds.close()
    if test_ds:
        test_ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

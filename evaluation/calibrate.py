"""Fit and report per-head calibration (plan task 3.3).

Produces logits from the trained heads, hands them to
`evaluation/calibration.py`, and writes three things:

* `docs/assets/calibration/report.json` - every head, fitted and rejected alike
* `docs/assets/calibration/*.svg` - reliability diagrams before and after
* `configs/calibration.json` - the registry the runtime actually reads

Only heads whose fit is **accepted** reach the registry. A rejected head stays
uncalibrated at runtime and keeps reporting the `ece_after = -1.0` sentinel,
which is the correct outcome: shipping a temperature fitted on 14 points
would be worse than shipping none, and silently doing so is exactly the class
of mistake this project has already had to correct twice.

Usage:
    python evaluation/calibrate.py --heads landcover intent change_mask \
        --ben-data data/ben_full --levir-index data/levircd/index.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.calibration import (  # noqa: E402
    CalibrationCurve,
    CalibrationReport,
    Bin,
    calibrate_head,
    reliability_svg,
    write_report,
)

REGISTRY_PATH = Path("configs/calibration.json")

# The report and the diagrams are report deliverables and belong in version
# control; `artifacts/` is gitignored runtime output, so they would vanish
# from a fresh clone if they were written there.
REPORT_DIR = Path("docs/assets/calibration")

# The cached logits are large and reproducible, so they stay in the ignored
# tree. Nothing but a re-fit reads them.
CACHE_DIR = Path("artifacts/calibration/logits")

# Registry keys are the task IDs the executor plans, so a lookup at answer
# time is `registry["heads"][plan.tasks[0]]` with no translation table.
HEAD_TASK_ID = {
    "landcover": "SINGLE_LANDCOVER",
    "change_mask": "TEMPORAL_CHANGE_MAP",
    "intent": "_router_intent",  # not a task; the Tier-1 router's own head
}

# Per-head fitting setup. Every multi-label head is fitted with BOTH methods:
# temperature is what task 3.3 names, affine is the two-parameter fallback,
# and running both means the choice between them is made on held-out ECE
# rather than on which one was tried first.
HEAD_SPEC: dict[str, dict] = {
    "landcover": {
        "mode": "multilabel",
        "dataset": "BigEarthNet-19",
        "methods": ["temperature", "affine"],
    },
    "change_mask": {
        "mode": "multilabel",
        "dataset": "LEVIR-CD",
        "methods": ["temperature", "affine"],
    },
    "intent": {
        # Affine scaling is a binary/multi-label construction; the multiclass
        # analogue is vector scaling, which is not what 3.3 asks for.
        "mode": "multiclass",
        "dataset": "CLEAN_HOLDOUT",
        "methods": ["temperature"],
    },
}


# --- Logit producers --------------------------------------------------------


def landcover_logits(data_dir: Path, checkpoint: Path, dim: int, batch_size: int,
                     split: str = "test", limit: int | None = 20000, seed: int = 1):
    """Land-cover head logits.

    Two layouts: the Phase 1-6 HDF5 shards (`*test*.hdf5`; only a test
    shard exists, which is why the v1/v2 fits were on test) and the complete
    official split written by `training/prepare/bigearthnet_v1_full.py`,
    where the fit is on the official **validation** split (`--ben-split
    val`, a `limit`-item random subsample at `seed`, never test). Band
    statistics are read from `band_stats.json` beside the weights when it
    exists, so the head sees exactly the normalisation it was trained with.
    """
    import torch

    from evaluation.splits.multires import load_model
    from training.track_a_full import BEN_GSD_M

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, latest, has_gsd = load_model(checkpoint, torch, device, dim)
    stats_file = (latest.parent if latest.is_file() else latest) / "band_stats.json"
    stats = None
    if stats_file.exists():
        blob = json.loads(stats_file.read_text(encoding="utf-8"))
        stats = (np.asarray(blob["mean"], dtype="float32"), np.asarray(blob["std"], dtype="float32"))

    if (data_dir / f"{split}_images.u16.npy").exists():
        from training.v3.ben_memmap import MemmapBigEarthNet

        if stats is None:
            from training.track_a_full import compute_stats

            stats = compute_stats(MemmapBigEarthNet(data_dir, "train"))
        n_all = np.load(data_dir / f"{split}_labels19.npy", mmap_mode="r").shape[0]
        picks = None
        if limit and limit < n_all:
            picks = np.random.default_rng(seed).choice(n_all, size=limit, replace=False)
        dataset = MemmapBigEarthNet(data_dir, split, stats, subset=picks)
        note = (f"BigEarthNet-S2 v1.0 official {split} split, {len(dataset)} of {n_all} patches "
                f"(random subsample, seed {seed}), all 12 bands at native 10 m, checkpoint {latest.name}")

        def batches(ds, bs, rng, shuffle=False):
            for s in range(0, len(ds), bs):
                yield ds.batch(np.arange(s, min(len(ds), s + bs)))
    else:
        from training.track_a_full import ShardedBigEarthNet, batches, compute_stats

        test_paths = sorted(Path(x) for x in glob.glob(str(data_dir / f"*{split}*.hdf5")))
        if not test_paths:
            raise SystemExit(f"no {split} shards in {data_dir}")
        train_paths = sorted(Path(x) for x in glob.glob(str(data_dir / "*train*.hdf5")))
        if stats is None:
            # Normalisation statistics must come from the same place training
            # took them, or the logits are produced under inputs the model
            # never saw.
            raw = ShardedBigEarthNet(train_paths or test_paths)
            stats = compute_stats(raw)
            raw.close()
        dataset = ShardedBigEarthNet(test_paths, stats)
        note = (f"BigEarthNet official {split} shard ({', '.join(p.name for p in test_paths)}), "
                f"all 12 bands at native 10 m, checkpoint {latest.name} "
                f"(gsd_conditioning={has_gsd}). Uniformly 10 m, so this split measures "
                f"the head at its training resolution; cross-resolution behaviour is the "
                f"multi-resolution split.")

    logits, labels = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for x, y in batches(dataset, batch_size, rng, shuffle=False):
            mask = np.ones((x.shape[0], x.shape[1]), dtype="float32")
            out = model(
                torch.from_numpy(np.ascontiguousarray(x)).to(device),
                torch.from_numpy(mask).to(device),
                torch.full((x.shape[0],), BEN_GSD_M, device=device),
            )
            logits.append(out.float().cpu().numpy())
            labels.append(y)
    dataset.close()
    return np.concatenate(logits), np.concatenate(labels), note


def produce_logits(head: str, args):
    """Dispatch to the right logit producer for `head`."""
    if head == "landcover":
        return landcover_logits(
            args.ben_data, args.track_a_ckpt, args.dim, args.batch_size,
            split=args.ben_split, limit=args.ben_limit,
        )
    if head == "intent":
        return intent_logits()
    return change_mask_logits(
        args.levir_index, args.change_ckpt, args.change_dim,
        args.batch_size, args.pixels_per_image, args.limit_change,
        split=args.levir_split,
    )


# --- Driver -----------------------------------------------------------------


def _curve_from_dict(d: dict) -> CalibrationCurve:
    bins = [Bin(**b) for b in d["bins"]]
    return CalibrationCurve(**{k: v for k, v in d.items() if k != "bins"}, bins=bins)


def emit_diagrams(report: CalibrationReport, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stage in ("before", "after"):
        curve = _curve_from_dict(getattr(report, stage))
        title = f"{report.head} - {stage} ({report.method})"
        path = out_dir / f"{report.head}_{report.method}_{stage}.svg"
        path.write_text(reliability_svg(curve, title), encoding="utf-8")
        written.append(str(path))
    return written


def build_registry(reports: list[CalibrationReport]) -> dict:
    """Registry from reports, one entry per head.

    A head may be fitted with more than one method. Only the accepted one
    with the lowest held-out ECE ships; the others stay in report.json as the
    evidence for that choice.
    """
    heads, rejected = {}, {}
    for r in reports:
        entry = {
            "method": r.method,
            "T": r.fit.get("T", r.fit.get("T_equivalent", 1.0)),
            "a": r.fit.get("a"),
            "b": r.fit.get("b"),
            "ece_before": r.before["ece"],
            "ece_after": r.after["ece"],
            "n_fit": r.n_fit,
            "n_eval": r.n_eval,
            "dataset": r.dataset,
            "split_note": r.split_note,
        }
        key = HEAD_TASK_ID.get(r.head, r.head)
        if r.accepted:
            incumbent = heads.get(key)
            if incumbent is None or entry["ece_after"] < incumbent["ece_after"]:
                heads[key] = entry
        elif key not in heads:
            # Keep the rejection that is most informative: the first one, or
            # any later one that at least improved ECE more.
            incumbent = rejected.get(key)
            if incumbent is None or entry["ece_after"] < incumbent["ece_after"]:
                rejected[key] = {**entry, "rejection_reason": r.rejection_reason}
    # A head that ended up calibrated by one method is not "rejected" because
    # another method failed on it.
    rejected = {k: v for k, v in rejected.items() if k not in heads}
    return {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "heads": heads,
        "rejected": rejected,
        "note": (
            "Heads under `rejected` are deliberately left uncalibrated at "
            "runtime and keep reporting ece_after = -1.0. A fitted parameter "
            "that did not survive its own held-out check is worse than none."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", nargs="+", default=["landcover", "intent"],
                   choices=sorted(HEAD_TASK_ID))
    p.add_argument("--ben-data", type=Path, default=Path("data/ben_full"))
    p.add_argument("--ben-split", default="test", choices=["test", "val"],
                   help="full layout only: fit on the official val split (Phase 6)")
    p.add_argument("--ben-limit", type=int, default=20000, help="full layout: val subsample size")
    p.add_argument("--track-a-ckpt", type=Path,
                   default=Path("checkpoints/track_a_full_base"))
    p.add_argument("--levir-index", type=Path, default=Path("data/levircd/index.json"))
    p.add_argument("--change-ckpt", type=Path, default=Path("checkpoints/change_mask"))
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--change-dim", type=int, default=16)
    p.add_argument("--levir-split", default="test", choices=["test", "val"],
                   help="which LEVIR-CD split the change head is fitted on (Phase 6: val)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pixels-per-image", type=int, default=1024)
    p.add_argument("--limit-change", type=int)
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    p.add_argument("--no-write-registry", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                   help="cache raw logits here so a re-fit needs no GPU")
    p.add_argument("--refresh-cache", action="store_true",
                   help="recompute logits even if a cache file exists")
    args = p.parse_args()

    reports: list[CalibrationReport] = []

    for head in args.heads:
        print(f"\n=== {head} ===", flush=True)
        spec = HEAD_SPEC[head]
        # Producing logits needs torch, a GPU and the datasets on disk;
        # re-fitting them needs neither. Caching keeps a re-fit cheap enough
        # that trying a second method is never a reason not to.
        cached = args.cache_dir / f"{head}.npz" if args.cache_dir else None
        if cached is not None and cached.exists() and not args.refresh_cache:
            blob = np.load(cached, allow_pickle=False)
            logits, labels, note = blob["logits"], blob["labels"], str(blob["note"])
            print(f"loaded cached logits from {cached}", flush=True)
        else:
            logits, labels, note = produce_logits(head, args)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cached, logits=logits, labels=labels, note=np.array(note)
                )
                print(f"cached logits to {cached}", flush=True)

        print(f"logits {logits.shape}  labels {labels.shape}", flush=True)

        for method in spec["methods"]:
            report = calibrate_head(
                logits, labels, head=head, mode=spec["mode"],
                dataset=spec["dataset"], split_note=note, method=method,
                n_bins=args.bins, seed=args.seed,
            )
            print("  " + report.summary(), flush=True)
            for path in emit_diagrams(report, args.out_dir):
                print(f"  wrote {path}")
            reports.append(report)

    report_path = args.out_dir / "report.json"
    write_report(reports, report_path)
    print(f"\nWrote {report_path}")

    if not args.no_write_registry:
        registry = build_registry(reports)
        # Merge with the registry on disk: heads fitted in an earlier run
        # (e.g. change_mask on LEVIR-CD val) survive a run that only refits
        # landcover; a head refitted here replaces its old entry (and leaves
        # `rejected` if it now passes, or `heads` if it now fails).
        if args.registry.exists():
            old = json.loads(args.registry.read_text(encoding="utf-8"))
            touched = set(registry["heads"]) | set(registry["rejected"])
            for section in ("heads", "rejected"):
                for key, entry in (old.get(section) or {}).items():
                    if key not in touched:
                        registry[section][key] = entry
        args.registry.parent.mkdir(parents=True, exist_ok=True)
        args.registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        print(f"Wrote {args.registry}  "
              f"({len(registry['heads'])} calibrated, "
              f"{len(registry['rejected'])} rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

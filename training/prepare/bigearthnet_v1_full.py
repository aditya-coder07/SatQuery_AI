"""Prepare the complete BigEarthNet-S2 v1.0 (19-label) benchmark from the
HDF5 mirror, with the official torchgeo train/val/test split.

Source: `lc-col/bigearthnet` on Hugging Face - an Apache-2.0 converter
(github.com/lccol/bigearthnet-conversion) over torchgeo's BigEarthNet
(Sumbul et al. split lists; 269,695 / 123,723 / 125,866 patches, snow and
cloud patches removed). Licence of the data: CDLA-Permissive-1.0
(BigEarthNet). Every file is verified against the sha256 the Hub records
for its LFS blob before it is unpacked.

Provenance findings folded into this preparer (audit L1, revised):

* The converter writes shards in split-list order, and the split lists are
  grouped by Sentinel-2 acquisition, so any *prefix* of shards is a
  geographic subset (the first 60k train patches are one region; the last
  test shard is another). Partial mirrors are therefore not random
  samples - which is what made the Phase 1-6 subset numbers so low on
  `test_p8`. Only the complete split is representative.
* `compute_block_size` in the converter has an off-by-one
  (`dataset_size - block_start_idx + 1`), so the last shard of every split
  carries one all-zero phantom row with an empty label. The mapping CSV is
  the row source of truth: rows beyond it are quarantined, never read.

Output layout (`--out`, default `data/ben_v1_full`)::

    raw/bigearthnet_{split}_p{k}.hdf5[.gz]   verified mirror
    bigearthnet_hdf5_{split}.csv             patch id -> shard, row
    {split}_images.u16.npy                   (N,12,120,120) uint16 memmap, CSV order
    {split}_labels19.npy / {split}_labels43.npy
    {split}_ids.txt                          patch id per row
    manifests/stats.json                     counts, hashes, quarantine, licence

Usage::

    python training/prepare/bigearthnet_v1_full.py --stage all [--splits train val test]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = "lc-col/bigearthnet"
SPLITS = ("train", "val", "test")
LICENCE = "CDLA-Permissive-1.0 (BigEarthNet); mirror converter Apache-2.0"
PREPROCESSING_VERSION = "ben_v1_full/1 (uint16 memmap, CSV order, phantom rows dropped)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def hub_listing() -> dict[str, dict]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(REPO, files_metadata=True)
    out = {}
    for s in info.siblings:
        if s.lfs is not None:
            out[s.rfilename] = {"sha256": s.lfs.sha256, "size": s.lfs.size}
    return out


def files_for(split: str, listing: dict) -> list[str]:
    shards = sorted((k for k in listing if k.startswith(f"bigearthnet_{split}_p") and k.endswith(".hdf5.gz")),
                    key=lambda k: int(k.rsplit("_p", 1)[1].split(".")[0]))
    return [f"bigearthnet_hdf5_{split}.csv"] + shards


def stage_download(out: Path, splits, listing) -> None:
    from huggingface_hub import hf_hub_download

    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for name in files_for(split, listing):
            dest = raw / name
            meta = listing[name]
            if dest.exists() and dest.stat().st_size == meta["size"]:
                continue
            hdf5 = raw / name[:-3]
            if name.endswith(".gz") and (raw / (hdf5.name + ".verified")).exists():
                continue  # already unpacked and verified; the .gz was released
            t0 = time.time()
            hf_hub_download(REPO, name, repo_type="dataset", local_dir=raw)
            print(f"downloaded {name} {meta['size'] / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)


def stage_verify(out: Path, splits, listing) -> None:
    raw = out / "raw"
    for split in splits:
        for name in files_for(split, listing):
            src = raw / name
            marker = raw / (name + ".sha256ok")
            if marker.exists() or not src.exists():
                continue
            digest = sha256_file(src)
            if digest != listing[name]["sha256"]:
                raise SystemExit(f"sha256 mismatch for {name}: {digest} != {listing[name]['sha256']}")
            marker.write_text(digest, encoding="utf-8")
            print(f"verified {name}", flush=True)


def stage_unpack(out: Path, splits, listing, release_gz: bool) -> None:
    import h5py

    raw = out / "raw"
    for split in splits:
        for name in files_for(split, listing):
            if not name.endswith(".gz"):
                continue
            gz, hdf5 = raw / name, raw / name[:-3]
            verified = raw / (hdf5.name + ".verified")
            if verified.exists():
                continue
            if not (raw / (name + ".sha256ok")).exists():
                raise SystemExit(f"{name} is not sha256-verified; run --stage verify first")
            t0 = time.time()
            if shutil.which("pigz"):
                with hdf5.open("wb") as fh:
                    subprocess.run(["pigz", "-dc", str(gz)], stdout=fh, check=True)
            else:
                with gzip.open(gz, "rb") as src, hdf5.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
            with h5py.File(hdf5, "r") as h:
                n = h["images"].shape[0]
                assert h["images"].shape[1:] == (12, 120, 120), h["images"].shape
                assert h["labels19"].shape == (n, 19) and h["labels43"].shape == (n, 43)
                last = np.asarray(h["images"][n - 1])
                _ = np.asarray(h["images"][0])
            verified.write_text(json.dumps({"rows": n, "last_row_all_zero": bool(np.abs(last).sum() == 0),
                                            "sha256_hdf5": sha256_file(hdf5)}), encoding="utf-8")
            print(f"unpacked {name}: {n} rows in {time.time() - t0:.0f}s", flush=True)
            if release_gz:
                gz.unlink()  # sha-verified, unpacked and re-read; regenerable from the Hub


def read_csv(path: Path) -> list[tuple[str, str, int]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        pid, shard, idx = line.split(",")
        rows.append((pid, shard, int(idx)))
    return rows


def stage_pack(out: Path, splits) -> dict:
    """Write each split as one uint16 memmap in CSV order (phantom rows dropped)."""
    import h5py

    raw = out / "raw"
    report = {}
    for split in splits:
        rows = read_csv(raw / f"bigearthnet_hdf5_{split}.csv")
        n = len(rows)
        img_path = out / f"{split}_images.u16.npy"
        done = out / f"{split}.packed.json"
        if done.exists():
            report[split] = json.loads(done.read_text(encoding="utf-8"))
            continue
        images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint16, shape=(n, 12, 120, 120))
        labels19 = np.zeros((n, 19), dtype=np.uint8)
        labels43 = np.zeros((n, 43), dtype=np.uint8)
        ids = []
        by_shard: dict[str, list[tuple[int, int]]] = {}
        for r, (pid, shard, idx) in enumerate(rows):
            by_shard.setdefault(shard, []).append((r, idx))
            ids.append(pid)
        clipped = 0
        quarantined = []
        t0 = time.time()
        for shard, pairs in by_shard.items():
            with h5py.File(raw / shard, "r") as h:
                stored = h["images"].shape[0]
                if stored != len(pairs):
                    quarantined.append({"shard": shard, "stored_rows": stored, "csv_rows": len(pairs),
                                        "reason": "rows beyond the mapping CSV (converter off-by-one)"})
                local = np.array([q for _, q in pairs])
                target = np.array([q for q, _ in pairs])
                assert np.all(np.diff(local) > 0), shard
                for s in range(0, len(local), 2000):
                    sl = slice(s, s + 2000)
                    blk = np.asarray(h["images"][local[sl][0]:local[sl][-1] + 1])
                    blk = blk[local[sl] - local[sl][0]]
                    clipped += int((blk > 65535).sum())
                    images[target[sl]] = np.clip(np.rint(blk), 0, 65535).astype(np.uint16)
                labels19[target] = np.asarray(h["labels19"])[local]
                labels43[target] = np.asarray(h["labels43"])[local]
            print(f"packed {shard} ({len(pairs)} rows, {time.time() - t0:.0f}s)", flush=True)
        images.flush()
        del images
        np.save(out / f"{split}_labels19.npy", labels19)
        np.save(out / f"{split}_labels43.npy", labels43)
        (out / f"{split}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        acq = Counter(pid.rsplit("_", 2)[0] for pid in ids)
        rep = {"n": n, "clipped_values": clipped, "quarantined": quarantined, "empty_labels": int((labels19.sum(1) == 0).sum()),
               "n_acquisitions": len(acq), "label19_frequency": [round(float(v), 4) for v in labels19.mean(0)],
               "sha256_images": sha256_file(img_path), "sha256_labels19": sha256_file(out / f"{split}_labels19.npy"),
               "sha256_ids": sha256_file(out / f"{split}_ids.txt")}
        done.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        report[split] = rep
    return report


def stage_stats(out: Path, splits, listing, report) -> None:
    raw = out / "raw"
    shards = {}
    for split in splits:
        for name in files_for(split, listing):
            entry = dict(listing[name])
            v = raw / (name[:-3] + ".verified") if name.endswith(".gz") else None
            if v is not None and v.exists():
                entry.update(json.loads(v.read_text(encoding="utf-8")))
            shards[name] = entry
    ids = {s: set((out / f"{s}_ids.txt").read_text(encoding="utf-8").split()) for s in splits if (out / f"{s}_ids.txt").exists()}
    overlap = {f"{a}&{b}": len(ids[a] & ids[b]) for a in ids for b in ids if a < b}
    stats = {"dataset": "BigEarthNet-S2 v1.0, 19-label nomenclature (43-label also stored)",
             "source": f"https://huggingface.co/datasets/{REPO} (torchgeo split lists; converter github.com/lccol/bigearthnet-conversion)",
             "license": LICENCE, "preprocessing_version": PREPROCESSING_VERSION,
             "splits": {s: report.get(s) for s in splits}, "split_id_overlap": overlap,
             "leakage": "official split lists; patch ids pairwise disjoint (see split_id_overlap); "
                        "test never read for selection",
             "shards": shards, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "manifests").mkdir(exist_ok=True)
    (out / "manifests" / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({"splits": {s: {k: report[s][k] for k in ("n", "quarantined", "empty_labels", "n_acquisitions")}
                                 for s in report}, "overlap": overlap}, indent=1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("data/ben_v1_full"))
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--stage", choices=["download", "verify", "unpack", "pack", "stats", "all"], default="all")
    p.add_argument("--keep-gz", action="store_true", help="keep the .gz after a verified unpack")
    args = p.parse_args()
    listing = hub_listing()
    if args.stage in ("download", "all"):
        stage_download(args.out, args.splits, listing)
    if args.stage in ("verify", "all"):
        stage_verify(args.out, args.splits, listing)
    if args.stage in ("unpack", "all"):
        stage_unpack(args.out, args.splits, listing, release_gz=not args.keep_gz)
    report = {}
    if args.stage in ("pack", "all"):
        report = stage_pack(args.out, args.splits)
    if args.stage in ("stats", "all"):
        if not report:
            report = {s: json.loads((args.out / f"{s}.packed.json").read_text(encoding="utf-8"))
                      for s in args.splits if (args.out / f"{s}.packed.json").exists()}
        stage_stats(args.out, args.splits, listing, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

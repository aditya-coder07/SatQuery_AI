"""Full-BigEarthNet preparer (pack stage), the memmap reader, and the v3
trainer on the full layout - all on a synthetic 12-band corpus with the
converter's phantom trailing row reproduced."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("torch")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.prepare import bigearthnet_v1_full as prep  # noqa: E402
from training.v3.ben_memmap import MemmapBigEarthNet, prefetch  # noqa: E402


def _write_split(raw: Path, split: str, n: int, per_shard: int, rng):
    rows = ["s2_folder,s2_hdf5_file,index"]
    k = 0
    for start in range(0, n, per_shard):
        m = min(per_shard, n - start)
        last = start + m >= n
        stored = m + 1 if last else m  # converter off-by-one on the last shard
        with h5py.File(raw / f"bigearthnet_{split}_p{k}.hdf5", "w") as h:
            img = np.zeros((stored, 12, 120, 120), dtype="float32")
            img[:m] = rng.integers(0, 12000, (m, 12, 120, 120)).astype("float32")
            img[0, 0, 0, 0] = 70000.0  # one out-of-range value to be clipped
            lab = np.zeros((stored, 19), dtype="int64")
            lab[:m] = rng.integers(0, 2, (m, 19))
            lab[:m, 4] = 1
            h.create_dataset("images", data=img)
            h.create_dataset("labels19", data=lab)
            h.create_dataset("labels43", data=np.zeros((stored, 43), dtype="int64"))
        for i in range(m):
            rows.append(f"S2A_MSIL2A_2017_{split}_{start + i}_1,bigearthnet_{split}_p{k}.hdf5,{i}")
        k += 1
    (raw / f"bigearthnet_hdf5_{split}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("ben")
    raw = out / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    _write_split(raw, "train", 14, 6, rng)
    _write_split(raw, "val", 5, 6, rng)
    _write_split(raw, "test", 7, 6, rng)
    report = prep.stage_pack(out, ("train", "val", "test"))
    return out, report


def test_pack_drops_phantom_rows_and_clips(corpus):
    out, report = corpus
    assert report["train"]["n"] == 14 and report["test"]["n"] == 7
    assert report["train"]["quarantined"][0]["stored_rows"] == report["train"]["quarantined"][0]["csv_rows"] + 1
    assert report["train"]["empty_labels"] == 0
    assert report["train"]["clipped_values"] >= 1
    imgs = np.load(out / "train_images.u16.npy", mmap_mode="r")
    assert imgs.shape == (14, 12, 120, 120) and imgs.dtype == np.uint16
    assert imgs[0, 0, 0, 0] == 65535
    ids = (out / "train_ids.txt").read_text().split()
    assert len(ids) == 14 and ids[6].endswith("_6_1")
    # rows are in CSV order across shards: row 6 is shard p1 row 0
    with h5py.File(out / "raw" / "bigearthnet_train_p1.hdf5") as h:
        assert np.array_equal(imgs[6][1:], np.asarray(h["images"][0])[1:].astype(np.uint16))


def test_memmap_reader_matches_getitem(corpus):
    out, _ = corpus
    ds = MemmapBigEarthNet(out, "train", subset=np.array([3, 1, 9]))
    assert len(ds) == 3
    xb, yb = ds.batch(np.array([2, 0]))
    x0, y0 = ds[0]
    assert xb.shape == (2, 12, 120, 120) and np.allclose(xb[0], x0) and np.array_equal(yb[0], y0)
    got = list(prefetch(lambda i: i * 2, range(5), depth=2))
    assert got == [0, 2, 4, 6, 8]


def test_trainer_full_layout_selects_on_val(corpus, tmp_path):
    out, _ = corpus
    ckpt = tmp_path / "ck"
    cmd = [sys.executable, str(ROOT / "training" / "train_landcover_v3.py"), "--data", str(out), "--ckpt-dir",
           str(ckpt), "--epochs", "2", "--batch-size", "4", "--no-pretrained", "--val-limit", "4"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    m = json.loads((ckpt / "metrics.json").read_text())
    assert m["selection"] == "val micro mAP" and m["n_val"] == 4
    assert m["test_at_best_val"]["epoch"] in (1, 2)
    assert (ckpt / "best.pt").exists() and (ckpt / "last.pt").exists() and (ckpt / "final.pt").exists()
    r2 = subprocess.run(cmd + ["--resume", "--epochs", "3"], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert "resumed at epoch 2" in r2.stdout

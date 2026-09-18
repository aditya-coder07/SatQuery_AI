"""BigEarthNet-S2 v1.0 full-split reader over the uint16 memmaps written by
`training/prepare/bigearthnet_v1_full.py`.

Same `batch(idx)` / `__getitem__` / `__len__` surface as
`track_a_full.ShardedBigEarthNet`, so the v3 trainer and `compute_stats`
work unchanged; the difference is that nothing is pulled into RAM up
front - the 93 GB train memmap is served through the page cache, and a
one-thread prefetcher keeps the next batches ready while the GPU works.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

REFLECTANCE_SCALE = 10000.0


class MemmapBigEarthNet:
    def __init__(self, root: Path, split: str, stats=None, subset: np.ndarray | None = None):
        self.images = np.load(root / f"{split}_images.u16.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels19.npy").astype("float32")
        self.ids = (root / f"{split}_ids.txt").read_text(encoding="utf-8").split()
        assert len(self.ids) == len(self.labels) == self.images.shape[0]
        self.subset = None if subset is None else np.sort(np.asarray(subset))
        self.stats = stats

    def __len__(self) -> int:
        return len(self.subset) if self.subset is not None else self.images.shape[0]

    def _norm(self, image: np.ndarray) -> np.ndarray:
        image = image.astype("float32") / REFLECTANCE_SCALE
        if self.stats is not None:
            mean, std = self.stats
            shape = (1, -1, 1, 1) if image.ndim == 4 else (-1, 1, 1)
            image = (image - mean.reshape(shape)) / std.reshape(shape)
        return image

    def batch(self, idx: np.ndarray):
        idx = np.sort(np.asarray(idx))
        rows = self.subset[idx] if self.subset is not None else idx
        return self._norm(self.images[rows]), self.labels[rows]

    def __getitem__(self, i: int):
        row = int(self.subset[i]) if self.subset is not None else int(i)
        return self._norm(np.asarray(self.images[row])), self.labels[row]

    def close(self) -> None:
        pass


def prefetch(fn, items, depth: int = 4):
    """Yield fn(item) for each item, computing `depth` items ahead on a thread."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = object()

    def worker():
        for it in items:
            q.put(fn(it))
        q.put(stop)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        out = q.get()
        if out is stop:
            return
        yield out

"""Near-duplicate images across two manifests, by perceptual hash.

Leakage check for the benchmark audit: an image that appears (or nearly
appears) in both a training and a test manifest inflates the test score.
The check hashes every distinct image path in each manifest with a 16x16
difference hash (256 bits, grayscale, resized with antialiasing), then
reports pairs whose Hamming distance is at most `--max-distance` (default 8
of 256, i.e. visually the same image up to compression/resampling), plus
exact byte-identical files by sha256.

Output: a JSON report with counts and the first 200 offending pairs, and an
exit code of 0 either way - this is a report, not a gate; what to do with
an official split that shares images by design is a documentation decision
(`docs/research/benchmark_audit.md`).

Usage::

    python evaluation/near_duplicates.py --a data/dior_rsvg_official/manifests/train.jsonl \
        --b data/dior_rsvg_official/manifests/test.jsonl --out artifacts/benchmark_reports/dup_dior_train_test.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def dhash(path: Path, size: int = 16) -> int:
    from PIL import Image

    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype="int16")
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def images_of(manifest: Path) -> list[Path]:
    root = manifest.parent.parent
    seen, out = set(), []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for rel in (r.get("images") or [r["image"]]):
            if rel not in seen:
                seen.add(rel)
                out.append(root / rel)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-distance", type=int, default=8)
    args = p.parse_args()

    a_imgs, b_imgs = images_of(args.a), images_of(args.b)
    a_h = {q: dhash(q) for q in a_imgs}
    b_h = {q: dhash(q) for q in b_imgs}
    a_s = {sha256(q): q for q in a_imgs}
    exact = [(str(q), str(a_s[sha256(q)])) for q in b_imgs if sha256(q) in a_s]
    # Bucket by the top 16 bits to keep the pairwise search tractable, then
    # confirm by full Hamming distance; near-duplicates share leading bits
    # almost always, and the exact-sha pass catches the rest.
    buckets: dict[int, list] = defaultdict(list)
    for q, h in a_h.items():
        buckets[h >> 240].append((q, h))
    near = []
    for qb, hb in b_h.items():
        for qa, ha in buckets.get(hb >> 240, []):
            d = hamming(ha, hb)
            if d <= args.max_distance and str(qa) != str(qb):
                near.append({"b": str(qb), "a": str(qa), "distance": d})
    same_path = len({str(q) for q in a_imgs} & {str(q) for q in b_imgs})
    report = {"a": str(args.a), "b": str(args.b), "n_a": len(a_imgs), "n_b": len(b_imgs),
              "same_path": same_path, "exact_duplicates": len(exact), "near_duplicates": len(near),
              "max_distance": args.max_distance, "examples_exact": exact[:200], "examples_near": near[:200]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("examples")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

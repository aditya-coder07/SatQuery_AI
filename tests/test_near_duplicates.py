"""dHash near-duplicate check: identical, re-encoded, and different images."""

import json

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from evaluation import near_duplicates as nd  # noqa: E402


def _img(path, seed, jpeg_quality=None):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (8, 8, 3), dtype="uint8")
    im = Image.fromarray(base).resize((128, 128), Image.BILINEAR)  # smooth structure, not noise
    if jpeg_quality:
        im.save(path, quality=jpeg_quality)
    else:
        im.save(path)
    return path


def _manifest(path, images):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"id": str(i), "image": f"images/{p.name}", "task": "<CAPTION>",
                                          "question": "q", "target": "t"}) for i, p in enumerate(images)) + "\n")
    return path


def test_exact_and_near_duplicates_are_found_and_distinct_images_are_not(tmp_path, monkeypatch):
    imgs = tmp_path / "ds" / "images"
    imgs.mkdir(parents=True)
    a1 = _img(imgs / "a1.png", 1)
    a2 = _img(imgs / "a2.png", 2)
    b_same = imgs / "b_same.png"
    b_same.write_bytes(a1.read_bytes())          # byte-identical
    b_reenc = _img(imgs / "b_reenc.jpg", 1, 85)   # same picture, JPEG
    b_new = _img(imgs / "b_new.png", 3)
    ma = _manifest(tmp_path / "ds" / "manifests" / "train.jsonl", [a1, a2])
    mb = _manifest(tmp_path / "ds" / "manifests" / "test.jsonl", [b_same, b_reenc, b_new])
    out = tmp_path / "rep.json"
    import sys

    argv = sys.argv
    sys.argv = ["x", "--a", str(ma), "--b", str(mb), "--out", str(out)]
    try:
        assert nd.main() == 0
    finally:
        sys.argv = argv
    rep = json.loads(out.read_text())
    assert rep["exact_duplicates"] == 1
    near_b = {e["b"].split("images")[-1][1:] for e in rep["examples_near"]}
    assert "b_reenc.jpg" in near_b and "b_same.png" in near_b
    assert "b_new.png" not in near_b

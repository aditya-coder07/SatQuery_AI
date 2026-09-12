"""Tests for the Qwen2.5-VL grounding format and the official DIOR-RSVG preparer.

Neither needs torch or a model: they pin the parts that decide whether a
box is scored in the right frame, which is where a grounding number goes
silently wrong.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from training.common import vlm_grounding as vg
from training.prepare import dior_rsvg_official as prep


class TestFormat:
    def test_render_then_parse_roundtrip(self):
        text = vg.render_answer([12.4, 30.6, 200.0, 250.0], "ship")
        assert text.startswith("```json") and '"label": "ship"' in text
        assert vg.parse_box(text) == [12.0, 31.0, 200.0, 250.0]

    def test_parse_tolerates_missing_fences_and_whitespace(self):
        assert vg.parse_box('[{"bbox_2d": [ 1 , 2 ,3,4 ]}]') == [1.0, 2.0, 3.0, 4.0]

    def test_parse_swaps_inverted_corners(self):
        assert vg.parse_box("[50, 60, 10, 20]") == [10.0, 20.0, 50.0, 60.0]

    def test_parse_none_when_no_box(self):
        assert vg.parse_box("I cannot see it.") is None
        assert vg.parse_box("[1, 2, 3]") is None

    def test_scale_box_maps_between_frames(self):
        # 800 -> 812 (the 28-multiple the processor resizes DIOR to) and back
        up = vg.scale_box([100, 200, 300, 400], 812 / 800, 812 / 800)
        back = vg.scale_box(up, 800 / 812, 800 / 812)
        assert [round(v, 6) for v in back] == [100, 200, 300, 400]

    def test_resized_hw_reads_patch_grid(self):
        assert vg.resized_hw([1, 58, 58]) == (812, 812)

    def test_iou(self):
        assert vg.iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
        assert vg.iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)
        assert vg.iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_assistant_start_finds_last_marker(self):
        marker = [7, 8, 9]
        ids = [1, 7, 8, 9, 2, 3, 7, 8, 9, 4, 5]
        assert vg.assistant_start(ids, marker) == 9
        with pytest.raises(ValueError):
            vg.assistant_start([1, 2, 3], marker)

    def test_size_buckets(self):
        assert vg.size_bucket([0, 0, 40, 40], 800, 800) == "small(<1%)"
        assert vg.size_bucket([0, 0, 200, 200], 800, 800) == "medium(1-10%)"
        assert vg.size_bucket([0, 0, 400, 400], 800, 800) == "large(>10%)"

    def test_user_prompt_strips_trailing_period(self):
        assert vg.user_prompt("the red car.").endswith("the red car in the image and output its bounding box coordinates in JSON format.")


XML = """<?xml version="1.0"?>
<annotation><filename>{name}.jpg</filename><size><width>800</width><height>800</height><depth>3</depth></size>
{objects}</annotation>"""
OBJ = """<object><name>{cat}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox><description>{desc}</description></object>"""


def _make_raw(tmp: Path) -> Path:
    raw = tmp / "raw"
    raw.mkdir()
    ann = tmp / "Annotations"
    ann.mkdir()
    # 3 images, 5 objects total; sorted order 00001, 00002, 00003
    specs = {
        "00001": [("ship", 10, 10, 50, 50, "a ship"), ("ship", 100, 100, 200, 220, "the other ship")],
        "00002": [("dam", 0, 0, 800, 800, "a large dam")],
        "00003": [("car", 5, 5, 6, 6, "tiny car"), ("bad", 20, 20, 20, 30, "degenerate box")],
    }
    imgs = tmp / "JPEGImages"
    imgs.mkdir()
    for name, objs in specs.items():
        (ann / f"{name}.xml").write_text(XML.format(name=name, objects="".join(
            OBJ.format(cat=c, x1=a, y1=b, x2=c2, y2=d, desc=e) for c, a, b, c2, d, e in objs)))
        (imgs / f"{name}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    with zipfile.ZipFile(raw / "Annotations.zip", "w") as z:
        for f in ann.iterdir():
            z.write(f, f"Annotations/{f.name}")
    with zipfile.ZipFile(raw / "JPEGImages.zip", "w") as z:
        for f in imgs.iterdir():
            z.write(f, f"JPEGImages/{f.name}")
    (raw / "train.txt").write_text("0\n2\n3\n")
    (raw / "val.txt").write_text("4\n")
    (raw / "test.txt").write_text("1\n")
    return raw


class TestPreparer:
    def test_flat_index_matches_reference_loader(self, tmp_path):
        raw = _make_raw(tmp_path)
        out = tmp_path / "out"
        import sys

        argv = sys.argv
        sys.argv = ["x", "--raw", str(raw), "--out", str(out)]
        try:
            assert prep.main() == 0
        finally:
            sys.argv = argv
        rows = {s: [json.loads(l) for l in (out / "manifests" / f"{s}.jsonl").read_text().splitlines()]
                for s in ("train", "val", "test")}
        # index 1 = second object of 00001 -> test; index 2 = 00002's dam -> train
        assert [r["id"] for r in rows["test"]] == ["dior_rsvg:00001:1"]
        assert rows["test"][0]["question"] == "the other ship"
        assert {r["id"] for r in rows["train"]} == {"dior_rsvg:00001:0", "dior_rsvg:00002:0", "dior_rsvg:00003:0"}
        stats = json.loads((out / "manifests" / "stats.json").read_text())
        assert stats["n_objects"] == 5
        assert stats["image_overlap"]["train_test"] == 1  # 00001 on both sides
        assert stats["splits"]["val"]["validator"]["n_bad"] == 1  # degenerate box reported, not dropped
        assert rows["val"][0]["target"]["bbox_xyxy"] == [20, 20, 20, 30]
        for r in rows["train"]:
            assert r["task"] == "<GROUNDING>" and r["license"] == "CC-BY-NC-4.0" and r["synthetic"] is False

    def test_overlapping_split_files_are_rejected(self, tmp_path):
        raw = _make_raw(tmp_path)
        (raw / "val.txt").write_text("1\n")
        import sys

        argv = sys.argv
        sys.argv = ["x", "--raw", str(raw), "--out", str(tmp_path / "o")]
        try:
            assert prep.main() == 1
        finally:
            sys.argv = argv

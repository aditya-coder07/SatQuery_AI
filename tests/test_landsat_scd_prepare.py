"""Landsat-SCD preparer: augmented copies whose A/B files are spelled
"ZheDang" while the label is "Zhedang" resolve to the files on disk; a
truly missing file stops the preparer instead of producing a dead path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _make(root: Path, names_ab: list[str], names_label: list[str]) -> None:
    for sub, names in (("A", names_ab), ("B", names_ab), ("label", names_label)):
        (root / sub).mkdir(parents=True)
        for n in names:
            (root / sub / f"{n}.png").write_bytes(b"x")


def _run(root: Path, out: Path):
    return subprocess.run([sys.executable, str(ROOT / "training" / "prepare" / "landsat_scd.py"), "--root", str(root),
                           "--out", str(out)], capture_output=True, text=True, cwd=ROOT)


def test_case_mismatch_resolves_to_files_on_disk(tmp_path):
    originals = [f"From1990To1993_{i:02d}" for i in range(10)]
    root = tmp_path / "ds"
    _make(root, originals + ["From1990To1993_00ZheDang1"], originals + ["From1990To1993_00Zhedang1"])
    out = tmp_path / "index.json"
    r = _run(root, out)
    assert r.returncode == 0, r.stderr
    idx = json.loads(out.read_text())
    rows = [x for s in idx["splits"].values() for x in s if x["augmented"]]
    if rows:  # the copy follows its original; only present when 00 landed in train
        assert rows[0]["a"].endswith("A/From1990To1993_00ZheDang1.png")
        assert rows[0]["label"].endswith("label/From1990To1993_00Zhedang1.png")


def test_missing_file_is_refused(tmp_path):
    originals = [f"From1990To1993_{i:02d}" for i in range(10)]
    root = tmp_path / "ds"
    _make(root, originals[:-1], originals)  # last original has a label but no A/B
    r = _run(root, tmp_path / "index.json")
    assert r.returncode != 0 and "do not exist on disk" in (r.stderr + r.stdout)

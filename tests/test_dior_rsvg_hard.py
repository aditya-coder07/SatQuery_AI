"""Hard-negative subset: only expressions whose image holds >=2 distinct
same-category objects; duplicate expressions for one object do not count."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_hard_subset_rule(tmp_path):
    man = tmp_path / "manifests"
    man.mkdir()
    rows = [
        {"id": "a1", "image": "i1.jpg", "question": "left ship", "target": {"bbox_xyxy": [0, 0, 10, 10], "category": "ship"}},
        {"id": "a2", "image": "i1.jpg", "question": "right ship", "target": {"bbox_xyxy": [50, 0, 60, 10], "category": "ship"}},
        {"id": "b1", "image": "i2.jpg", "question": "the ship", "target": {"bbox_xyxy": [0, 0, 10, 10], "category": "ship"}},
        {"id": "b2", "image": "i2.jpg", "question": "the vessel", "target": {"bbox_xyxy": [1, 1, 10, 10], "category": "ship"}},
        {"id": "c1", "image": "i2.jpg", "question": "the dam", "target": {"bbox_xyxy": [20, 20, 40, 40], "category": "dam"}},
    ]
    (man / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "training" / "prepare" / "dior_rsvg_hard.py"), "--root", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    hard = [json.loads(l) for l in (man / "train_hard.jsonl").read_text().splitlines()]
    assert sorted(h["id"] for h in hard) == ["a1", "a2"]  # i2's two ship rows are one object; the dam is alone
    assert all(h["metadata"]["n_same_category"] == 2 for h in hard)
    stats = json.loads((man / "stats.json").read_text())
    assert stats["hard_negatives"]["train"]["n_rows"] == 2

"""Risk-coverage and AURC over the real heads (plan task 3.6).

Reuses the logits `evaluation/calibrate.py` cached for task 3.3, so this
needs neither a GPU nor the datasets on disk once that has run once.

Writes `docs/assets/abstention/selective.json` and one risk-coverage SVG per
signal.

Usage:
    python evaluation/selective.py
    python evaluation/selective.py --cache-dir artifacts/calibration/logits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.abstention import (  # noqa: E402
    SelectiveResult,
    evaluate_selective,
    risk_coverage_svg,
    write_results,
)
from evaluation.calibration import sigmoid  # noqa: E402

REPORT_DIR = Path("docs/assets/abstention")
CACHE_DIR = Path("artifacts/calibration/logits")


def landcover_signal(logits: np.ndarray, labels: np.ndarray):
    """Per-decision confidence for a multi-label head.

    Each (patch, class) pair is one binary decision. Its confidence is
    `max(p, 1-p)` - how far from undecided the head was - and it is correct
    when the thresholded decision matches the label. Treating the whole
    19-class vector as one prediction would make almost everything wrong and
    say nothing about which individual calls to trust.
    """
    probs = sigmoid(logits).ravel()
    truth = np.asarray(labels, dtype="float64").ravel()
    confidence = np.maximum(probs, 1.0 - probs)
    correct = ((probs >= 0.5).astype("float64") == truth).astype("float64")
    return confidence, correct


NL_SPLITS = ("queries.jsonl", "queries_test.jsonl", "queries_final.jsonl")

LANDCOVER_NOTE = (
    "Phase 3 land-cover head (Track A, ckpt_step_2814), per (patch, class) "
    "decision on the official BigEarthNet test shard: does the head's own "
    "sigmoid confidence rank its correct calls above its incorrect ones. The "
    "deployed v3 head (landcover_full) has not been re-scored here. The "
    "deployed tool reports the mean calibrated probability of the classes it "
    "asserts as its confidence and abstains per class below its decision "
    "threshold (configs/thresholds.v3.yaml)."
)
ROUTER_NOTE = (
    "Deployed router ({classifier}: config gate + classifier + rules), top-1 "
    "probability against whether the chosen task is in the query's accepted "
    "set, on the three hand-written NL splits (evaluation/nl/, never used "
    "to build the template bank), n={n}."
)


def router_signal():
    """The live router over the held-out NL benchmark splits."""
    import json as _json
    import tempfile

    from evaluation.nl_understanding_eval import manifests
    from satquery.controller.intent import CLASSIFIER_NAME
    from satquery.controller.matrix_loader import load_matrix
    from satquery.controller.router import Router

    nl_dir = Path(__file__).resolve().parent / "nl"
    rows = []
    for name in NL_SPLITS:
        text = (nl_dir / name).read_text(encoding="utf-8")
        rows += [_json.loads(line) for line in text.splitlines() if line.strip()]
    router = Router(load_matrix(Path("configs/capability_matrix.yaml")))
    confidence, correct = [], []
    with tempfile.TemporaryDirectory() as tmp:
        by_config = manifests(Path(tmp))
        for row in rows:
            decision = router.decide(row["query"], by_config[row["config"]], history=row.get("history"))
            confidence.append(decision.prediction.top1 if decision.prediction else 0.0)
            correct.append(float(decision.plan.tasks[0] in row["accept"]))
    note = ROUTER_NOTE.format(classifier=CLASSIFIER_NAME, n=len(rows))
    return np.asarray(confidence), np.asarray(correct), note


SIGNALS = {"landcover": landcover_signal}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--signals", nargs="+", default=["router", *sorted(SIGNALS)],
                   choices=["router", *sorted(SIGNALS)])
    args = p.parse_args()

    out = args.out_dir / "selective.json"
    previous = {}
    if out.exists():
        previous = {r["name"]: r for r in json.loads(out.read_text(encoding="utf-8"))}

    results: list[SelectiveResult] = []
    carried: list[dict] = []
    for name in args.signals:
        if name == "router":
            confidence, correct, note = router_signal()
        else:
            cached = args.cache_dir / f"{name}.npz"
            if not cached.exists():
                if name in previous:
                    # Keep the measured result; only its description is refreshed.
                    print(f"{name}: no cached logits at {cached}; keeping the recorded result")
                    carried.append({**previous[name], "note": LANDCOVER_NOTE})
                else:
                    print(f"skipping {name}: no cached logits at {cached}.", file=sys.stderr)
                continue
            blob = np.load(cached, allow_pickle=False)
            confidence, correct = SIGNALS[name](blob["logits"], blob["labels"])
            note = LANDCOVER_NOTE
        result = evaluate_selective(confidence, correct, name, note)
        results.append(result)
        print(result.summary())
        for target, coverage in result.coverage_at_risk.items():
            shown = "unreachable" if coverage is None else f"{coverage:.1%}"
            print(f"    coverage at {target}: {shown}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / f"{name}_risk_coverage.svg"
        path.write_text(risk_coverage_svg(result), encoding="utf-8")
        print(f"    wrote {path}")

    if not results and not carried:
        print("no signals scored", file=sys.stderr)
        return 1

    write_results(results, out)
    if carried:
        merged = json.loads(out.read_text(encoding="utf-8")) + carried
        out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Natural-language understanding benchmark for the router (2026-09-20).

Runs the real `Router` (config gating + classifier + parameter extraction)
over `evaluation/nl/queries.jsonl`: hand-written queries in the wording a
user actually types, labelled with the set of tasks that would serve them.
Nothing in that file is a template of `satquery/synth/query_bank.py`, and
nothing from it may be copied into the bank - the file is the held-out
measurement of whether the bank generalises, and it stops measuring that
the moment a query appears in both.

Scored per query:

* **routing** - the selected task is in the query's `accept` list;
* **object** - when the query names a thing to locate, the extracted
  referring phrase contains it (case-insensitive);
* **classes** - when the query names land-cover classes, the extracted set
  equals the labelled set;
* **quantity / spatial / temporal / image** - equality when labelled;
* **follow-ups** - rows with `history` are resolved against it first.

Usage::

    python evaluation/nl_understanding_eval.py [--out artifacts/benchmark_reports/nl_understanding.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.scenes import (  # noqa: E402
    build_msi_6band, build_msi_6band_t2, build_rgb_3band, build_sar_dualpol,
)
from satquery.controller.matrix_loader import load_matrix  # noqa: E402
from satquery.controller.router import Router  # noqa: E402
from satquery.ingest import ingest  # noqa: E402

QUERIES = Path(__file__).resolve().parent / "nl" / "queries.jsonl"


def manifests(tmp: Path) -> dict:
    return {
        "SINGLE": ingest([build_rgb_3band(tmp / "s.tif")]),
        "BITEMPORAL_PAIR": ingest([build_msi_6band(tmp / "t1.tif"), build_msi_6band_t2(tmp / "t2.tif")]),
        "CROSSMODAL_PAIR": ingest([build_msi_6band(tmp / "o.tif"), build_sar_dualpol(tmp / "sar.tif")]),
    }


def evaluate(router: Router, rows: list[dict], by_config: dict) -> dict:
    hits: dict[str, list[int]] = defaultdict(list)
    failures: list[dict] = []
    for row in rows:
        manifest = by_config[row["config"]]
        decision = router.decide(row["query"], manifest, history=row.get("history"))
        u = decision.understanding
        task = decision.plan.tasks[0]
        ok = task in row["accept"]
        hits["routing"].append(int(ok))
        group = row["id"][0]
        hits[f"routing/{group}"].append(int(ok))
        detail = {"id": row["id"], "query": row["query"], "task": task, "accept": row["accept"],
                  "top1": decision.prediction.top1 if decision.prediction else None,
                  "resolved": u.resolved_query if u else None}
        if not ok:
            failures.append({**detail, "field": "routing"})
        checks = {
            "object": lambda: u is not None and u.object_filter is not None
            and row["object"].lower() in u.object_filter.lower(),
            "classes": lambda: u is not None and sorted(u.classes or []) == sorted(row["classes"]),
            "quantity": lambda: u is not None and u.quantity == row["quantity"],
            "spatial": lambda: u is not None and u.spatial_scope is not None
            and row["spatial"].replace("-", " ").lower() in u.spatial_scope.replace("-", " ").lower(),
            "temporal": lambda: u is not None and u.temporal_relation == row["temporal"],
            "image": lambda: u is not None and u.image_index == row["image"],
        }
        for field, check in checks.items():
            if field in row:
                good = bool(check())
                hits[field].append(int(good))
                if not good:
                    failures.append({**detail, "field": field, "expected": row[field],
                                     "got": getattr(u, {"object": "object_filter", "spatial": "spatial_scope",
                                                       "temporal": "temporal_relation", "image": "image_index"}
                                                    .get(field, field), None) if u else None})
    summary = {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in sorted(hits.items())}
    return {"n": len(rows), "summary": summary, "failures": failures}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--queries", type=Path, default=QUERIES)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    tmp = Path(tempfile.mkdtemp(prefix="nl_eval_"))
    router = Router(load_matrix("configs/capability_matrix.yaml"))
    report = evaluate(router, rows, manifests(tmp))
    report["queries"] = str(args.queries)
    for k, v in report["summary"].items():
        print(f"{k:16} {v['accuracy']:.4f}  (n={v['n']})")
    for f in report["failures"]:
        print(f"  MISS {f['field']:8} {f['id']:5} {f['query']!r} -> {f.get('task')} "
              f"{'expected ' + json.dumps(f.get('expected')) + ' got ' + json.dumps(f.get('got')) if 'expected' in f else ''}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

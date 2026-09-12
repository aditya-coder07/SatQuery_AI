"""Render docs/research/training_summary.md from the experiment registry.

One row per experiment (latest record per id), grouped into training runs
and evaluations, with the lineage the prompt asks for:
BASELINE -> EXPERIMENT -> BEST CHECKPOINT -> BEST SCORE. Numbers are read
from the registry only; nothing is typed by hand, so the document cannot
drift from the ledger.

Usage::

    python scripts/registry_report.py [--registry artifacts/experiment_registry/registry.jsonl]
        [--out docs/research/training_summary.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import registry  # noqa: E402


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        keep = {k: x for k, x in v.items() if isinstance(x, (int, float, str)) and not isinstance(x, bool)}
        return "; ".join(f"{k} {fmt(x)}" for k, x in list(keep.items())[:6])
    if v is None:
        return "—"
    return str(v)[:60]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--registry", type=Path, default=registry.DEFAULT_PATH)
    p.add_argument("--out", type=Path, default=Path("docs/research/training_summary.md"))
    args = p.parse_args()
    recs = sorted(registry.load(args.registry).values(), key=lambda r: (str(r.get("timestamp") or ""), r["experiment_id"]))
    train = [r for r in recs if not r["experiment_id"].startswith(("eval_", "phase5-rsvqa", "phase5-v1_v2", "robustness"))]
    evals = [r for r in recs if r["experiment_id"].startswith(("eval_", "phase5-rsvqa", "robustness"))]

    lines = ["# Training summary — generated from the experiment registry", "",
             f"Source: `{args.registry}` ({len(recs)} experiments). Regenerate with "
             "`python scripts/registry_report.py`. Do not edit by hand.", "",
             "## Training runs", "",
             "| experiment_id | model / architecture | status | steps | GPU-h | checkpoint | validation | test | notes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in train:
        hours = (r.get("duration_s") or 0) / 3600
        lines.append("| " + " | ".join([
            r["experiment_id"], f"{fmt(r.get('model'))} / {fmt(r.get('architecture'))}", fmt(r.get("status")),
            fmt(r.get("training_steps")), f"{hours:.2f}" if hours else "—", fmt(r.get("checkpoint")),
            fmt(r.get("validation_metrics")), fmt(r.get("test_metrics")), fmt(r.get("notes"))]) + " |")
    lines += ["", "## Evaluations", "",
              "| experiment_id | checkpoint | headline | details |", "|---|---|---|---|"]
    for r in evals:
        tm = r.get("test_metrics") or {}
        head = tm.get("headline") if isinstance(tm, dict) else None
        if head is None and isinstance(tm, dict):
            for k in ("acc@0.5", "micro_accuracy", "f1", "map_all_bands", "miou"):
                if k in tm:
                    head = tm[k]
                    break
        lines.append(f"| {r['experiment_id']} | {fmt(r.get('checkpoint'))} | {fmt(head)} | {fmt(tm)} |")
    lines += ["", "## Lineage", "",
              "| Tool | BASELINE (Phase 5) | EXPERIMENT | BEST CHECKPOINT | BEST SCORE | DELTA |", "|---|---|---|---|---|---|"]
    by_id = {r["experiment_id"]: r for r in recs}
    lineage = [
        ("change_mask", "phase5-change_mask", "f1", None, "checkpoints/v3/change_mask/best.pt"),
        ("optsar_fusion", "phase5-optsar_fusion", "complementarity_gain", None, "checkpoints/v3/optsar_fusion/best.pt"),
        ("grounding", "phase5-grounding_pre", "acc@0.5", None, "checkpoints/v3/grounding_vlm_r16/adapter_best"),
        ("landcover", "phase5-track_a", "map_all_bands", None, "checkpoints/v3/landcover/final.pt"),
    ]
    for tool, base_id, key, _, ckpt in lineage:
        base = (by_id.get(base_id) or {}).get("validation_metrics") or {}
        base_v = base.get(key)
        best = None
        for r in recs:
            ck = str(r.get("checkpoint") or "")
            if ck.replace("\\", "/").startswith(ckpt.rsplit("/", 1)[0]) and r.get("status") == "done":
                m = r.get("test_metrics") or r.get("validation_metrics") or {}
                if isinstance(m, dict):
                    for k in (key, "headline", "gain_miou", "f1", "acc@0.5", "map_all_bands"):
                        if k in m and isinstance(m[k], (int, float)):
                            best = (r["experiment_id"], m[k])
                            break
        if best and isinstance(base_v, (int, float)):
            lines.append(f"| {tool} | {base_v:.4f} | {best[0]} | {ckpt} | {best[1]:.4f} | {best[1] - base_v:+.4f} |")
        else:
            lines.append(f"| {tool} | {fmt(base_v)} | pending | {ckpt} | — | — |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(train)} runs, {len(evals)} evaluations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

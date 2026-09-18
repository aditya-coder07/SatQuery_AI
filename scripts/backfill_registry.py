"""Backfill the experiment registry from the Phase 5 artifacts.

The registry (`training/common/registry.py`) starts on 2026-09-12; every run
before it is recorded in `docs/assets/phase5/` as metrics + campaign state.
This script turns those into registry records once, so lineage is complete
from the v1 baselines onward. It is idempotent: records carry deterministic
ids (`phase5-<run>`), and re-running rewrites nothing that already exists.

Usage::

    python scripts/backfill_registry.py [--registry artifacts/experiment_registry/registry.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import registry  # noqa: E402

PHASE5 = Path("docs/assets/phase5")
CAMPAIGN = Path("configs/campaign.yaml")

# What the Phase 5 comparison treats as each run's headline metric.
HEADLINE = {
    "track_a": ("map_all_bands", "retention"),
    "track_a_nodropout": ("map_all_bands", "retention"),
    "grounding": ("acc@0.5", "miou"),
    "grounding_pre": ("acc@0.5", "miou"),
    "change_mask": ("f1", "iou"),
    "caption": ("bleu4_sentence_mean",),
    "caption_pre": ("bleu4_sentence_mean",),
    "change_caption": ("bleu4_changed", "bleu4_aggregate"),
    "change_vqa": ("miou_change_classes",),
    "change_vqa_scratch": ("miou_change_classes",),
    "optsar_fusion": ("complementarity_gain", "fused"),
    "track_b_vqa": ("val_loss",),
    "track_b_vqa_es": ("val_loss",),
}


def flat_metrics(m: dict, keys: tuple[str, ...]) -> dict:
    out = {}
    for k in keys:
        if k in m:
            out[k] = m[k]
        elif "final" in m and k in m["final"]:
            out[k] = m["final"][k]
        elif "best" in m and isinstance(m["best"], dict) and k in m["best"]:
            out[k] = m["best"][k]
    return out or {k: v for k, v in m.items() if isinstance(v, (int, float))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=registry.DEFAULT_PATH)
    args = p.parse_args()

    existing = registry.load(args.registry)
    state = json.loads((PHASE5 / "campaign_state.json").read_text())
    runs = state.get("runs", state)
    n_new = 0
    for run_dir in sorted(d for d in PHASE5.iterdir() if d.is_dir()):
        run = run_dir.name
        exp_id = f"phase5-{run}"
        if exp_id in existing:
            continue
        metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
        st = runs.get(run, {})
        rec = dict(
            experiment_id=exp_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.get("finished_at", 0))) if st.get("finished_at") else "2026-09-12",
            git_commit=None,  # campaign ran across several commits; see docs/phase5-full-training.md
            model=run, architecture="v2 (training/v2/architectures.py)" if not run.startswith("track_b") else "qwen2.5-vl-3b + qlora",
            status=st.get("status", "done"), duration_s=st.get("seconds"),
            checkpoint=f"checkpoints/v2/{run}", validation_metrics=flat_metrics(metrics, HEADLINE.get(run, ())),
            notes="backfilled from docs/assets/phase5; hyperparameters in configs/campaign.yaml",
            hardware={"gpu": "NVIDIA L40S (shared)", "host": "compute01.node"},
        )
        registry.record(args.registry, **rec)
        n_new += 1

    # Phase 6 runs made by the legacy trainers (no registry hook of their
    # own): one record per docs/assets/phase6/<run> that has none yet.
    phase6 = Path("docs/assets/phase6")
    if phase6.exists():
        for run_dir in sorted(d for d in phase6.iterdir() if d.is_dir()):
            exp_id = f"phase6-{run_dir.name}"
            if exp_id in existing or not (run_dir / "metrics.json").exists():
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text())
            meta = json.loads((run_dir / "run_metadata.json").read_text()) if (run_dir / "run_metadata.json").exists() else {}
            if "experiment_id" in meta:  # trainer already registered itself
                continue
            head = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            best = metrics.get("test_at_best_val") if isinstance(metrics.get("test_at_best_val"), dict) else None
            registry.record(args.registry, experiment_id=exp_id, model=run_dir.name,
                            architecture=meta.get("arch") or "see run_metadata", status="done",
                            hyperparameters={k: v for k, v in meta.items() if k not in ("task",)},
                            checkpoint=f"checkpoints/v3/{run_dir.name}/best.pt" if best else f"checkpoints/v3/{run_dir.name}",
                            test_metrics=best or head, validation_metrics=metrics.get("best_val"),
                            duration_s=metrics.get("gpu_hours", 0) * 3600 or None,
                            notes="backfilled from docs/assets/phase6", hardware={"gpu": "NVIDIA L40S (shared)"})
            n_new += 1

    official = PHASE5 / "rsvqa_lr_official_test.json"
    if official.exists() and "phase5-rsvqa_official_test" not in existing:
        d = json.loads(official.read_text())
        registry.record(args.registry, experiment_id="phase5-rsvqa_official_test", model="qwen2.5-vl-3b + qlora",
                        status="done", checkpoint="checkpoints/v2/track_b_vqa/adapter_final",
                        test_metrics={arm: v["published_convention"] for arm, v in d["arms"].items()},
                        notes="official RSVQA-LR test, all arms; see docs/assets/phase5/rsvqa_lr_official_test.json")
        n_new += 1
    comp = PHASE5 / "v1_v2_comparison.json"
    if comp.exists() and "phase5-v1_v2_comparison" not in existing:
        registry.record(args.registry, experiment_id="phase5-v1_v2_comparison", status="done",
                        test_metrics=json.loads(comp.read_text()), notes="v1 vs v2 headline table")
        n_new += 1
    print(f"backfilled {n_new} records into {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

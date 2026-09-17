#!/usr/bin/env bash
# VRSBench val grounding for arm E at its training pixel budget (min_pixels
# 1024x1024); paired offline against arm D' (vrsbench_val_grounding_armD2).
LOG=logs/queue_vrsbench_eval3.log
MIN_FREE_GB=${MIN_FREE_GB:-10}
source scripts/cluster_unit_lib.sh
log "vrsbench eval (arm E) started"
run eval_vrsbench_ground_armE $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/vrsbench/manifests/val_grounding.jsonl \
  --arms lora_hires=checkpoints/v3/grounding_vlm_hires/adapter_best --min-pixels 1048576 \
  --out artifacts/benchmark_reports/vrsbench_val_grounding_armE.json --batch 8
log "vrsbench eval (arm E) finished"

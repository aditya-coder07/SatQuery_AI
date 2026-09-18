#!/usr/bin/env bash
# VRSBench val grounding, arms A and C only (the base arm's 16k pass was
# lost to the width KeyError fixed in vlm_task_eval.py on 2026-09-15).
LOG=logs/queue_vrsbench_eval.log
MIN_FREE_GB=${MIN_FREE_GB:-10}
source scripts/cluster_unit_lib.sh
log "vrsbench eval unit started"
run eval_vrsbench_ground2 $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/vrsbench/manifests/val_grounding.jsonl \
  --arms lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best lora_vrs=checkpoints/v3/grounding_vlm_vrs/adapter_best \
  --out artifacts/benchmark_reports/vrsbench_val_grounding.json --batch 16
log "vrsbench eval unit finished"

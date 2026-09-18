#!/usr/bin/env bash
# VRSBench val grounding, arm C (deployed) vs arm D' (hard negatives from C).
LOG=logs/queue_vrsbench_eval2.log
MIN_FREE_GB=${MIN_FREE_GB:-10}
source scripts/cluster_unit_lib.sh
log "vrsbench eval unit started"
run eval_vrsbench_ground_armD2 $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/vrsbench/manifests/val_grounding.jsonl \
  --arms lora_vrs=checkpoints/v3/grounding_vlm_vrs/adapter_best lora_hard_vrs=checkpoints/v3/grounding_vlm_hard_vrs/adapter_best \
  --out artifacts/benchmark_reports/vrsbench_val_grounding_armD2.json --batch 16
log "vrsbench eval unit finished"

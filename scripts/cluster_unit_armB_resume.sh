#!/usr/bin/env bash
# RESUME variant (2026-09-14 15:10): all three trainers received an external
# SIGKILL at 14:58:36 (not cgroup OOM: oom_kill 0, 431 GB RAM free); this
# continues from adapter_last/train_state.pt with --resume.
# Grounding arm B (+ trainable visual merger), relaunch through the launch
# lock (2026-09-14 11:05, started beside arm C on the user's instruction):
# batch 2 x accum 8 and expandable segments so it fits in ~12 GB.
LOG=logs/queue_armB_resume.log
MIN_FREE_GB=${MIN_FREE_GB:-14}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
log "arm B unit started"
run train_ground_lora_merger $PY training/train_grounding_vlm.py --model $BASE --data data/dior_rsvg_official   --ckpt-dir checkpoints/v3/grounding_vlm_r16_merger --epochs 1 --batch-size 2 --grad-accum 8 --lr 1e-4   --val-limit 400 --val-every 800 --save-every 400 --workers 6 --quant none --train-merger   --notes "arm B: LoRA r16 + trainable visual merger, official DIOR-RSVG train, 1 epoch, batch 2x8"
run eval_ground_merger $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official   --arms lora_r16_merger=checkpoints/v3/grounding_vlm_r16_merger/adapter_best   --out artifacts/benchmark_reports/dior_rsvg_official_armB.json --batch 16
log "arm B unit finished"

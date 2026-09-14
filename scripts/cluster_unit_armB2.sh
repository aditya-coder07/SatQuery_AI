#!/usr/bin/env bash
# Grounding arm B (+ trainable visual merger), relaunch through the launch
# lock (2026-09-14). A 3B LoRA job reserves ~24-32 GB, so it needs the card
# mostly free: MIN_FREE_GB 28.
LOG=logs/queue_armB2.log
MIN_FREE_GB=${MIN_FREE_GB:-28}
source scripts/cluster_unit_lib.sh
log "arm B unit started"
run train_ground_lora_merger $PY training/train_grounding_vlm.py --model $BASE --data data/dior_rsvg_official   --ckpt-dir checkpoints/v3/grounding_vlm_r16_merger --epochs 1 --batch-size 4 --grad-accum 4 --lr 1e-4   --val-limit 400 --val-every 800 --save-every 400 --workers 6 --quant none --train-merger   --notes "arm B: LoRA r16 + trainable visual merger, official DIOR-RSVG train, 1 epoch, batch 4x4"
run eval_ground_merger $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official   --arms lora_r16_merger=checkpoints/v3/grounding_vlm_r16_merger/adapter_best   --out artifacts/benchmark_reports/dior_rsvg_official_armB.json --batch 16
log "arm B unit finished"

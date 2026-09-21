#!/usr/bin/env bash
# Grounding arms (2026-09-21): DIOR-RSVG 0.7323 -> toward GeoGround 0.777.
#   F   arm E continued at 1280x1280 (min_pixels 1638400): resolution was the
#       largest lever so far (D' 0.702 -> E 0.732 at 1024^2, small objects
#       0.529 -> 0.578) and small objects are 42% of the test. Deferred since
#       2026-09-18; recipe unchanged from scripts/cluster_unit_post1.sh.
#   F2  F continued one more epoch at lr 2e-5 on the same mix: the 1-epoch
#       schedules have never been extended; val is the check.
#   Each arm is scored on the official test at its training resolution and
#   on VRSBench val (the regression number that must not fall below 0.6594).
LOG=logs/queue_ground_sota.log
MIN_FREE_GB=${MIN_FREE_GB:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
log "grounding sota queue started"
TRAIN="data/dior_rsvg_official/manifests/train.jsonl data/dior_rsvg_official/manifests/train_hard.jsonl data/vrsbench/manifests/train_grounding.jsonl"

run train_ground_hires1280 $PY training/train_vlm_sft.py --model $BASE \
  --train $TRAIN --train-weight 0.5 0.5 0.2 \
  --val data/dior_rsvg_official/manifests/val.jsonl --val-limit 400 \
  --init-adapter checkpoints/v3/grounding_vlm_hires/adapter_best \
  --ckpt-dir checkpoints/v3/grounding_vlm_hires1280 --epochs 1 --batch-size 1 --grad-accum 16 --lr 2e-5 \
  --min-pixels 1638400 --val-every 400 --save-every 200 --workers 6 --quant none \
  --notes "arm F: arm E continued at min_pixels 1280x1280, train 0.5x + hard 0.5x + VRSBench 0.2x"
if [ -f checkpoints/v3/grounding_vlm_hires1280/adapter_best/adapter_model.safetensors ]; then
run eval_ground_armF $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_hires1280=checkpoints/v3/grounding_vlm_hires1280/adapter_best \
  --min-pixels 1638400 --out artifacts/benchmark_reports/dior_rsvg_official_armF.json --batch 4
run eval_vrs_armF $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/vrsbench/manifests/val_grounding.jsonl \
  --arms lora_hires1280=checkpoints/v3/grounding_vlm_hires1280/adapter_best --min-pixels 1638400 \
  --out artifacts/benchmark_reports/vrsbench_val_grounding_armF.json --batch 4

run train_ground_hires1280_e2 $PY training/train_vlm_sft.py --model $BASE \
  --train $TRAIN --train-weight 0.5 0.5 0.2 \
  --val data/dior_rsvg_official/manifests/val.jsonl --val-limit 400 \
  --init-adapter checkpoints/v3/grounding_vlm_hires1280/adapter_best \
  --ckpt-dir checkpoints/v3/grounding_vlm_hires1280_e2 --epochs 1 --batch-size 1 --grad-accum 16 --lr 2e-5 \
  --min-pixels 1638400 --val-every 400 --save-every 200 --workers 6 --quant none --seed 43 \
  --notes "arm F2: arm F continued one more epoch at 1280x1280 (seed 43 for a fresh order)"
if [ -f checkpoints/v3/grounding_vlm_hires1280_e2/adapter_best/adapter_model.safetensors ]; then
run eval_ground_armF2 $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_hires1280_e2=checkpoints/v3/grounding_vlm_hires1280_e2/adapter_best \
  --min-pixels 1638400 --out artifacts/benchmark_reports/dior_rsvg_official_armF2.json --batch 4
run eval_vrs_armF2 $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/vrsbench/manifests/val_grounding.jsonl \
  --arms lora_hires1280_e2=checkpoints/v3/grounding_vlm_hires1280_e2/adapter_best --min-pixels 1638400 \
  --out artifacts/benchmark_reports/vrsbench_val_grounding_armF2.json --batch 4
fi
fi
log "grounding sota queue finished"

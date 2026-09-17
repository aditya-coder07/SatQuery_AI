#!/usr/bin/env bash
# Post-programme queue 1 (2026-09-18): the remaining cheap/medium arms, in
# order of expected gain per GPU-hour, each through the launch lock.
#   L1  land cover, class-balanced BCE (pos_weight <= 5) + noise aug 0.15,
#       full official split, val-selected -> macro-mAP tail (2 GPU-h)
#   F   grounding arm E continued at 1280x1280 input (min_pixels 1638400),
#       train 0.5x + hard 0.5x + VRSBench 0.2x, lr 2e-5 (~14 h) -> official test
#   S   seed-variance: arm D' recipe re-run at seed 43 (~4 h) -> official test
LOG=logs/queue_post1.log
MIN_FREE_GB=${MIN_FREE_GB:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
log "post-programme queue 1 started"

run train_landcover_v3_balanced $PY training/train_landcover_v3.py --data data/ben_v1_full \
  --ckpt-dir checkpoints/v3/landcover_full_balanced --epochs 30 --batch-size 128 --val-limit 20000 \
  --class-balanced 5 --noise-aug 0.15 \
  --notes "full official split; class-balanced pos_weight (cap 5) + gaussian noise aug 0.15; targets macro mAP tail"

run train_ground_hires1280 $PY training/train_vlm_sft.py --model $BASE \
  --train data/dior_rsvg_official/manifests/train.jsonl data/dior_rsvg_official/manifests/train_hard.jsonl data/vrsbench/manifests/train_grounding.jsonl \
  --train-weight 0.5 0.5 0.2 \
  --val data/dior_rsvg_official/manifests/val.jsonl --val-limit 400 \
  --init-adapter checkpoints/v3/grounding_vlm_hires/adapter_best \
  --ckpt-dir checkpoints/v3/grounding_vlm_hires1280 --epochs 1 --batch-size 1 --grad-accum 16 --lr 2e-5 \
  --min-pixels 1638400 --val-every 400 --save-every 200 --workers 6 --quant none \
  --notes "arm F: arm E continued at min_pixels 1280x1280, train 0.5x + hard 0.5x + VRSBench 0.2x"
if [ -f checkpoints/v3/grounding_vlm_hires1280/adapter_best/adapter_model.safetensors ]; then
run eval_ground_armF $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_hires1280=checkpoints/v3/grounding_vlm_hires1280/adapter_best \
  --min-pixels 1638400 --out artifacts/benchmark_reports/dior_rsvg_official_armF.json --batch 4
fi

run train_ground_hard_vrs_s43 $PY training/train_vlm_sft.py --model $BASE \
  --train data/dior_rsvg_official/manifests/train_hard.jsonl data/dior_rsvg_official/manifests/train.jsonl data/vrsbench/manifests/train_grounding.jsonl \
  --train-weight 1.0 0.2 0.3 \
  --val data/dior_rsvg_official/manifests/val.jsonl --val-limit 400 \
  --init-adapter checkpoints/v3/grounding_vlm_vrs/adapter_best \
  --ckpt-dir checkpoints/v3/grounding_vlm_hard_vrs_s43 --epochs 1 --batch-size 4 --grad-accum 4 --lr 3e-5 \
  --val-every 400 --save-every 200 --workers 6 --quant none --seed 43 \
  --notes "seed-variance: arm D' recipe at seed 43"
if [ -f checkpoints/v3/grounding_vlm_hard_vrs_s43/adapter_best/adapter_model.safetensors ]; then
run eval_ground_armD2_s43 $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_hard_s43=checkpoints/v3/grounding_vlm_hard_vrs_s43/adapter_best \
  --out artifacts/benchmark_reports/dior_rsvg_official_armD2_s43.json --batch 16
fi
log "post-programme queue 1 finished"

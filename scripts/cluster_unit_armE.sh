#!/usr/bin/env bash
# Grounding arm E (2026-09-16): resolution. Every analysis put the DIOR-RSVG
# gap on small objects (0.53 vs 0.88 Acc@0.5 for large; 4x downscale -0.107),
# so the deployed arm D' adapter is continued with the processor's
# min_pixels raised to 1024x1024 (800-px DIOR images are UPSCALED, ~1.7x the
# visual tokens per object), on DIOR-RSVG train x0.6 + hard negatives x0.5 +
# VRSBench grounding x0.2, lr 3e-5. The official test is run at the same
# pixel budget (single arm; paired against D' offline from the predictions).
LOG=logs/queue_armE.log
MIN_FREE_GB=${MIN_FREE_GB:-20}
INIT_ADAPTER=${INIT_ADAPTER:-checkpoints/v3/grounding_vlm_hard_vrs/adapter_best}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
log "arm E unit started; init adapter $INIT_ADAPTER; min_pixels 1048576"
[ -f "$INIT_ADAPTER/adapter_model.safetensors" ] || { log "STOPPED: $INIT_ADAPTER missing"; exit 1; }
run train_ground_hires $PY training/train_vlm_sft.py --model $BASE \
  --train data/dior_rsvg_official/manifests/train.jsonl data/dior_rsvg_official/manifests/train_hard.jsonl data/vrsbench/manifests/train_grounding.jsonl \
  --train-weight 0.6 0.5 0.2 \
  --val data/dior_rsvg_official/manifests/val.jsonl --val-limit 400 \
  --init-adapter "$INIT_ADAPTER" \
  --ckpt-dir checkpoints/v3/grounding_vlm_hires --epochs 1 --batch-size 2 --grad-accum 8 --lr 3e-5 \
  --min-pixels 1048576 --val-every 400 --save-every 200 --workers 6 --quant none \
  --notes "arm E: arm D' continued at min_pixels 1024x1024 (upscaled DIOR), train 0.6x + hard 0.5x + VRSBench 0.2x"
[ -f checkpoints/v3/grounding_vlm_hires/adapter_best/adapter_model.safetensors ] || { log "STOPPED: arm E adapter_best missing"; exit 1; }
run eval_ground_armE $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_hires=checkpoints/v3/grounding_vlm_hires/adapter_best \
  --min-pixels 1048576 --out artifacts/benchmark_reports/dior_rsvg_official_armE.json --batch 8
log "arm E unit finished"

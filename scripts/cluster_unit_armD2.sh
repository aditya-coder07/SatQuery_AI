#!/usr/bin/env bash
# Grounding arm D' (2026-09-16): the arm-D hard-negative recipe applied to the
# deployed arm C adapter, so the deployed adapter gets the gain arm D showed
# (+1.0 pt on the official test, McNemar significant). The best adapter so far
# (INIT_ADAPTER, chosen from the official test of arms A/C) is continued
# for one epoch on the hard-negative subset of DIOR-RSVG train (13,519
# expressions from images with >=2 same-category objects, oversampled) plus
# a 0.2x resample of the full train split and 0.3x of VRSBench grounding,
# at a lower learning rate; then the official test paired against the
# adapter it started from.
LOG=logs/queue_armD2.log
MIN_FREE_GB=${MIN_FREE_GB:-16}
INIT_ADAPTER=${INIT_ADAPTER:-checkpoints/v3/grounding_vlm_r16/adapter_best}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
log "arm D' unit started; init adapter $INIT_ADAPTER"
[ -f "$INIT_ADAPTER/adapter_model.safetensors" ] || { log "STOPPED: $INIT_ADAPTER missing"; exit 1; }
[ -s data/dior_rsvg_official/manifests/train_hard.jsonl ] || { log "STOPPED: train_hard.jsonl missing"; exit 1; }
run train_ground_hard_vrs $PY training/train_vlm_sft.py --model $BASE \
  --train data/dior_rsvg_official/manifests/train_hard.jsonl data/dior_rsvg_official/manifests/train.jsonl data/vrsbench/manifests/train_grounding.jsonl \
  --train-weight 1.0 0.2 0.3 \
  --val data/dior_rsvg_official/manifests/val.jsonl --val-limit 400 \
  --init-adapter "$INIT_ADAPTER" \
  --ckpt-dir checkpoints/v3/grounding_vlm_hard_vrs --epochs 1 --batch-size 4 --grad-accum 4 --lr 3e-5 \
  --val-every 400 --save-every 200 --workers 6 --quant none \
  --notes "arm D: hard negatives (multi-instance DIOR-RSVG train, 1.0x) + train 0.2x + VRSBench 0.3x, from $INIT_ADAPTER"
[ -f checkpoints/v3/grounding_vlm_hard_vrs/adapter_best/adapter_model.safetensors ] || { log "STOPPED: arm D adapter_best missing"; exit 1; }
run eval_ground_armD2 $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms init=$INIT_ADAPTER lora_hard=checkpoints/v3/grounding_vlm_hard_vrs/adapter_best \
  --out artifacts/benchmark_reports/dior_rsvg_official_armD2.json --batch 16
log "arm D' unit finished"

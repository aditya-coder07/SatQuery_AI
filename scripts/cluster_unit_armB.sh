#!/usr/bin/env bash
# Grounding arm B (+ trainable visual merger) as its own unit (2026-09-13
# 03:10); waits only for VRAM. Batch 4 x accum 4 so it can share the card.
LOG=logs/queue_armB.log
MIN_FREE_GB=${MIN_FREE_GB:-16}
set -u
cd /scratch/home/adi01/satquery
PY=/scratch/home/adi01/satquery/.conda-env/bin/python
BASE=models/qwen25_vl_3b
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
wait_vram() {  # sized from FREE VRAM on the shared card, never total
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$(( (total - used) / 1024 ))
    if [ "$free" -ge "$MIN_FREE_GB" ]; then return 0; fi
    log "waiting for VRAM: ${free} GB free < ${MIN_FREE_GB} GB"; sleep 120
  done
}
run() { local name=$1; shift; wait_vram; log "START $name"
  if "$@" >> "logs/$name.log" 2>&1; then log "DONE $name (exit 0)"; else log "FAILED $name (exit $?)"; fi; }

log "arm B unit started"
run train_ground_lora_merger $PY training/train_grounding_vlm.py --model $BASE --data data/dior_rsvg_official   --ckpt-dir checkpoints/v3/grounding_vlm_r16_merger --epochs 1 --batch-size 4 --grad-accum 4 --lr 1e-4   --val-limit 400 --val-every 800 --save-every 400 --workers 6 --quant none --train-merger   --notes "arm B: LoRA r16 + trainable visual merger, official DIOR-RSVG train, 1 epoch, batch 4x4"
run eval_ground_merger $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official   --arms lora_r16_merger=checkpoints/v3/grounding_vlm_r16_merger/adapter_best   --out artifacts/benchmark_reports/dior_rsvg_official_armB.json --batch 16
log "arm B unit finished"

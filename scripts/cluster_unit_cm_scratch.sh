#!/usr/bin/env bash
# change_mask v3 scratch ablation as its own unit (2026-09-13 03:10).
LOG=logs/queue_cm_scratch.log
MIN_FREE_GB=${MIN_FREE_GB:-10}
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

log "cm-scratch unit started"
run train_cm_v3_scratch $PY training/train_change_mask.py --index data/levircd/index.json   --ckpt-dir checkpoints/v3/change_mask_scratch --arch v3 --no-pretrained --dim 64 --epochs 40 --batch-size 16   --lr 2e-4 --loss bce_dice --augment --cosine --amp --workers 6 --select-on-val --save-every 445
log "cm-scratch unit finished"

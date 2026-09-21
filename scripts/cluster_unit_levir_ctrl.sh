#!/usr/bin/env bash
# LEVIR control arm (2026-09-21): plain continuation of the champion, no
# boundary term, lr 1e-5, 6 epochs - does ANY continuation beat the champion?
LOG=logs/queue_levir_ctrl.log
MIN_FREE_GB=${MIN_FREE_GB:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
log "levir control started"
run train_change_mask_cont $PY training/train_change_mask.py --index data/levircd/index.json \
  --ckpt-dir checkpoints/v3/change_mask_cont --arch v3 --dim 64 --epochs 6 --batch-size 16 --lr 1e-5 \
  --loss bce_dice --augment --cosine --amp --workers 6 --select-on-val \
  --init-weights checkpoints/v3/change_mask/best.pt
if [ -f checkpoints/v3/change_mask_cont/best.pt ]; then
run eval_levir_cont_tta $PY evaluation/change_mask_official_eval.py \
  --checkpoint checkpoints/v3/change_mask_cont/best.pt --val-split val --tta \
  --out artifacts/benchmark_reports/levircd_test_cont_tta.json
fi
log "levir control finished"

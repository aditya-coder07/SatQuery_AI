#!/usr/bin/env bash
# LEVIR-CD arm (2026-09-21): close the last 0.7 points to ChangeGCC (0.921).
#   Measured first, no training: 8-fold dihedral TTA + the val-selected
#   threshold (0.75) takes the frozen champion from F1 0.9038 to 0.9139 on
#   the official test (evaluation/change_mask_official_eval.py --tta).
#   This arm fine-tunes the champion for the boundary tail the re-score found
#   (10th-percentile per-tile F1 0.42 on changed tiles): bce_dice plus BCE
#   re-weighted x3 on a 3-px band around label edges, fresh cosine schedule
#   at 1/4 of the original lr, val-selected epoch, then the same TTA +
#   val-threshold evaluation. Any gain is reported beside the plain number.
LOG=logs/queue_levir_sota.log
MIN_FREE_GB=${MIN_FREE_GB:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
log "levir sota queue started"

run eval_levir_tta_baseline $PY evaluation/change_mask_official_eval.py \
  --checkpoint checkpoints/v3/change_mask/best.pt --val-split val --tta \
  --out artifacts/benchmark_reports/levircd_test_tta_v3_cluster.json

run train_change_mask_boundary $PY training/train_change_mask.py --index data/levircd/index.json \
  --ckpt-dir checkpoints/v3/change_mask_boundary --arch v3 --dim 64 --epochs 12 --batch-size 16 --lr 5e-5 \
  --loss bce_dice_boundary --augment --cosine --amp --workers 6 --select-on-val \
  --init-weights checkpoints/v3/change_mask/best.pt
if [ -f checkpoints/v3/change_mask_boundary/best.pt ]; then
run eval_levir_boundary $PY evaluation/change_mask_official_eval.py \
  --checkpoint checkpoints/v3/change_mask_boundary/best.pt --val-split val \
  --out artifacts/benchmark_reports/levircd_test_boundary.json
run eval_levir_boundary_tta $PY evaluation/change_mask_official_eval.py \
  --checkpoint checkpoints/v3/change_mask_boundary/best.pt --val-split val --tta \
  --out artifacts/benchmark_reports/levircd_test_boundary_tta.json
fi
log "levir sota queue finished"

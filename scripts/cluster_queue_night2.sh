#!/usr/bin/env bash
# Phase 6 queue, night 2: waits for night 1 to finish, then runs the cheap
# ablations and the second grounding arm. Same conventions as night 1.
set -u
cd /scratch/home/adi01/satquery
PY=/scratch/home/adi01/satquery/.conda-env/bin/python
LOG=logs/queue_night2.log
MIN_FREE_GB=${MIN_FREE_GB:-18}
BASE=models/qwen25_vl_3b
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
wait_unit() { while systemctl --user is-active --quiet "$1"; do sleep 60; done; }
wait_vram() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$(( (total - used) / 1024 ))
    [ "$free" -ge "$MIN_FREE_GB" ] && return 0
    log "waiting for VRAM: ${free} GB free < ${MIN_FREE_GB} GB"; sleep 120
  done
}
run() { local name=$1; shift; wait_vram; log "START $name"
  if "$@" >> "logs/$name.log" 2>&1; then log "DONE $name (exit 0)"; else log "FAILED $name (exit $?)"; fi; }

log "queue 2 started; waiting for sq-queue-night1"
wait_unit sq-queue-night1
log "night 1 finished"

# A. Ablation: change_mask v3 with the same trunk from scratch (weights=None).
run train_cm_v3_scratch $PY training/train_change_mask.py --index data/levircd/index.json   --ckpt-dir checkpoints/v3/change_mask_scratch --arch v3 --no-pretrained --dim 64 --epochs 40 --batch-size 16   --lr 2e-4 --loss bce_dice --augment --cosine --amp --workers 6 --select-on-val --save-every 445

# B. Ablation: landcover v3 trunk from scratch.
run train_landcover_v3_scratch $PY training/train_landcover_v3.py --data data/ben_full \
  --ckpt-dir checkpoints/v3/landcover_scratch --epochs 30 --batch-size 128 --no-pretrained

# C. Grounding arm B: LoRA + trainable visual merger, official train, 1 epoch.
run train_ground_lora_merger $PY training/train_grounding_vlm.py --model $BASE --data data/dior_rsvg_official \
  --ckpt-dir checkpoints/v3/grounding_vlm_r16_merger --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 \
  --val-limit 400 --val-every 400 --save-every 200 --workers 8 --quant none --train-merger \
  --notes "arm B: LoRA r16 + trainable visual merger, official DIOR-RSVG train, 1 epoch"
run eval_ground_merger $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_r16_merger=checkpoints/v3/grounding_vlm_r16_merger/adapter_best \
  --out artifacts/benchmark_reports/dior_rsvg_official_armB.json --batch 24

# D. Robustness of the selected change_mask under the deployed loader is
#    covered by sq-robust-cm-v3; nothing else here.
log "queue 2 finished"

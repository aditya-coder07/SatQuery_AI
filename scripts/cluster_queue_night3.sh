#!/usr/bin/env bash
# Phase 6 queue, night 3 (2026-09-12, replaces night 2 after the L1 revision
# and the VRSBench fetch). Runs on compute01 under `systemd-run --user`,
# after night 1 (grounding eval, VQA, caption, change-caption arms).
#
# Priority order from the charter: grounding first. Arm C continues the
# arm-A adapter on the complete DIOR-RSVG train plus VRSBench grounding
# (CC-BY-4.0, crops from DIOR-RSVG val/test images quarantined), capped at
# native resolution (arm A's test: small objects 0.51 vs large 0.89 Acc@0.5,
# so input pixels are not where to save compute); then the official test
# (paired McNemar against arm A) and the VRSBench val split. The
# change_mask scratch ablation and grounding arm B (+merger) follow.
#
# Every step waits for MIN_FREE_GB of free VRAM (sized from free, never
# total); a failed step is logged and the queue continues.
set -u
cd /scratch/home/adi01/satquery
PY=/scratch/home/adi01/satquery/.conda-env/bin/python
LOG=logs/queue_night3.log
MIN_FREE_GB=${MIN_FREE_GB:-18}
BASE=models/qwen25_vl_3b

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
wait_unit() { while systemctl --user is-active --quiet "$1"; do sleep 60; done; }
wait_vram() {
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

# Gate: arm C continues arm A's adapter, so arm A must have finished cleanly
# (exit 0 in its log), its adapter_best must exist and load, and the official
# test report for lora_r16 must be present with 7,500 items. A failed gate
# stops this queue (exit 1) instead of guessing; nothing downstream runs.
gate_arm_a() {
  local ad=checkpoints/v3/grounding_vlm_r16/adapter_best
  local rep=artifacts/benchmark_reports/dior_rsvg_official_phase6.json
  grep -q "DONE eval_ground_official (exit 0)" logs/queue_night1.log || { log "GATE FAIL: eval_ground_official did not finish with exit 0"; return 1; }
  [ -f "$ad/adapter_config.json" ] && [ -f "$ad/adapter_model.safetensors" ] || { log "GATE FAIL: $ad incomplete"; return 1; }
  [ -f "$rep" ] || { log "GATE FAIL: $rep missing"; return 1; }
  $PY - "$rep" "$ad" <<'PYEOF' || { log "GATE FAIL: report/adapter verification failed"; return 1; }
import json, sys
rep, ad = sys.argv[1], sys.argv[2]
d = json.load(open(rep))
arms = d.get("arms") or {}
a = arms.get("lora_r16") or {}
n = a.get("n") or d.get("n") or 0
acc = a.get("acc@0.5")
assert acc is not None and n == 7500, (acc, n)
from safetensors import safe_open
with safe_open(f"{ad}/adapter_model.safetensors", "pt") as f:
    keys = list(f.keys())
assert len(keys) > 100, len(keys)
print(f"GATE OK: arm A lora_r16 acc@0.5={acc:.4f} on n={n}; adapter tensors={len(keys)}")
PYEOF
}

log "queue 3 started; waiting for sq-queue-night1"
wait_unit sq-queue-night1
log "night 1 finished"
if ! gate_arm_a >> "$LOG" 2>&1; then log "queue 3 STOPPED at the arm-A gate; nothing else run"; exit 1; fi

# A. Grounding arm C: arm A adapter continued on DIOR-RSVG train (40% resample,
#    already seen once) + VRSBench grounding (all clean rows), 1 epoch.
run train_ground_vrs $PY training/train_vlm_sft.py --model $BASE \
  --train data/dior_rsvg_official/manifests/train.jsonl data/vrsbench/manifests/train_grounding.jsonl \
  --train-weight 0.4 1.0 \
  --val data/dior_rsvg_official/manifests/val.jsonl data/vrsbench/manifests/val_grounding.jsonl --val-limit 400 \
  --init-adapter checkpoints/v3/grounding_vlm_r16/adapter_best \
  --ckpt-dir checkpoints/v3/grounding_vlm_vrs --epochs 1 --batch-size 8 --grad-accum 2 --lr 5e-5 \
  --val-every 400 --save-every 200 --workers 8 --quant none \
  --notes "arm C: arm A adapter continued on DIOR-RSVG train (0.4x) + VRSBench grounding (CC-BY-4.0, quarantined), native resolution"
run eval_ground_armC $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best lora_vrs=checkpoints/v3/grounding_vlm_vrs/adapter_best \
  --out artifacts/benchmark_reports/dior_rsvg_official_armC.json --batch 24
run eval_vrsbench_ground $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/vrsbench/manifests/val_grounding.jsonl \
  --arms base=BASE lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best lora_vrs=checkpoints/v3/grounding_vlm_vrs/adapter_best \
  --out artifacts/benchmark_reports/vrsbench_val_grounding.json --batch 24

# B. Ablation: change_mask v3 with the same trunk from scratch.
run train_cm_v3_scratch $PY training/train_change_mask.py --index data/levircd/index.json \
  --ckpt-dir checkpoints/v3/change_mask_scratch --arch v3 --no-pretrained --dim 64 --epochs 40 --batch-size 16 \
  --lr 2e-4 --loss bce_dice --augment --cosine --amp --workers 6 --select-on-val --save-every 445

# C. Grounding arm B: LoRA + trainable visual merger, official train, 1 epoch.
run train_ground_lora_merger $PY training/train_grounding_vlm.py --model $BASE --data data/dior_rsvg_official \
  --ckpt-dir checkpoints/v3/grounding_vlm_r16_merger --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 \
  --val-limit 400 --val-every 400 --save-every 200 --workers 8 --quant none --train-merger \
  --notes "arm B: LoRA r16 + trainable visual merger, official DIOR-RSVG train, 1 epoch"
run eval_ground_merger $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_r16_merger=checkpoints/v3/grounding_vlm_r16_merger/adapter_best \
  --out artifacts/benchmark_reports/dior_rsvg_official_armB.json --batch 24

log "queue 3 finished"

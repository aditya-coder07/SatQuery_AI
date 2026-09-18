#!/usr/bin/env bash
# Grounding arm C as its own unit (2026-09-13 03:10): starts as soon as the
# arm-A gate passes and enough VRAM is free - it no longer waits for the
# rest of night 1. Batch 4 x accum 4 (same effective batch as arm A) so it
# fits beside a second VLM job on the shared card.
LOG=logs/queue_armC.log
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

log "arm C unit started"
gate_arm_a >> "$LOG" 2>&1 || { log "STOPPED at the arm-A gate"; exit 1; }
run train_ground_vrs $PY training/train_vlm_sft.py --model $BASE   --train data/dior_rsvg_official/manifests/train.jsonl data/vrsbench/manifests/train_grounding.jsonl   --train-weight 0.4 1.0   --val data/dior_rsvg_official/manifests/val.jsonl data/vrsbench/manifests/val_grounding.jsonl --val-limit 400   --init-adapter checkpoints/v3/grounding_vlm_r16/adapter_best   --ckpt-dir checkpoints/v3/grounding_vlm_vrs --epochs 1 --batch-size 4 --grad-accum 4 --lr 5e-5   --val-every 800 --save-every 400 --workers 6 --quant none   --notes "arm C: arm A adapter continued on DIOR-RSVG train (0.4x) + VRSBench grounding (CC-BY-4.0, quarantined), native resolution, batch 4x4"
run eval_ground_armC $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official   --arms lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best lora_vrs=checkpoints/v3/grounding_vlm_vrs/adapter_best   --out artifacts/benchmark_reports/dior_rsvg_official_armC.json --batch 16
run eval_vrsbench_ground $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/vrsbench/manifests/val_grounding.jsonl   --arms base=BASE lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best lora_vrs=checkpoints/v3/grounding_vlm_vrs/adapter_best   --out artifacts/benchmark_reports/vrsbench_val_grounding.json --batch 16
log "arm C unit finished"

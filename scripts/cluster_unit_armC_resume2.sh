#!/usr/bin/env bash
# RESUME variant (2026-09-14 15:10): all three trainers received an external
# SIGKILL at 14:58:36 (not cgroup OOM: oom_kill 0, 431 GB RAM free); this
# continues from adapter_last/train_state.pt with --resume.
# Grounding arm C, relaunch 2 (2026-09-14 10:20) with the launch lock; the
# first launch OOM'd when three VLM jobs started within 30 s.
LOG=logs/queue_armC_resume2.log
MIN_FREE_GB=${MIN_FREE_GB:-16}
source scripts/cluster_unit_lib.sh
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

log "arm C unit (relaunch 2) started"
gate_arm_a >> "$LOG" 2>&1 || { log "STOPPED at the arm-A gate"; exit 1; }
run train_ground_vrs $PY training/train_vlm_sft.py --model $BASE   --train data/dior_rsvg_official/manifests/train.jsonl data/vrsbench/manifests/train_grounding.jsonl   --train-weight 0.4 1.0   --val data/dior_rsvg_official/manifests/val.jsonl data/vrsbench/manifests/val_grounding.jsonl --val-limit 400   --init-adapter checkpoints/v3/grounding_vlm_r16/adapter_best   --ckpt-dir checkpoints/v3/grounding_vlm_vrs --epochs 1 --batch-size 4 --grad-accum 4 --lr 5e-5   --val-every 800 --save-every 400 --workers 6 --quant none --resume   --notes "arm C: arm A adapter continued on DIOR-RSVG train (0.4x) + VRSBench grounding (CC-BY-4.0, quarantined), native resolution, batch 4x4"
[ -f checkpoints/v3/grounding_vlm_vrs/adapter_best/adapter_model.safetensors ] || { log "STOPPED: arm C adapter_best missing; evals skipped"; exit 1; }
run eval_ground_armC $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official   --arms lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best lora_vrs=checkpoints/v3/grounding_vlm_vrs/adapter_best   --out artifacts/benchmark_reports/dior_rsvg_official_armC.json --batch 16
run eval_vrsbench_ground $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/vrsbench/manifests/val_grounding.jsonl   --arms base=BASE lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best lora_vrs=checkpoints/v3/grounding_vlm_vrs/adapter_best   --out artifacts/benchmark_reports/vrsbench_val_grounding.json --batch 16
log "arm C unit finished"

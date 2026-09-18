#!/usr/bin/env bash
# Phase 6 queue, night 4: the unified multitask adapter (charter §12), run
# only after the specialist arms have valid baselines (night 1 + night 3).
# One LoRA r16 on the shared Qwen2.5-VL-3B base over every task, with
# per-manifest weights chosen so no task exceeds ~35% of an epoch and the
# 160k-row VQA sources do not dominate (raw sizes: VQA 136k, grounding 60k,
# caption 62k rows, change caption 34k). ~100k samples/epoch.
#
# Then every official test the specialists were scored on, with the
# unified adapter as one more arm, so negative transfer is measured per
# task on the same items (paired tests where the evaluator supports them).
set -u
cd /scratch/home/adi01/satquery
PY=/scratch/home/adi01/satquery/.conda-env/bin/python
LOG=logs/queue_night4.log
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

gate_specialists() {
  for f in artifacts/benchmark_reports/dior_rsvg_official_phase6.json artifacts/benchmark_reports/rsvqa_lr_official_phase6.json \
           artifacts/benchmark_reports/rsicd_test_vlm.json artifacts/benchmark_reports/levircc_test_vlm.json; do
    [ -f "$f" ] || { log "GATE FAIL: $f missing (specialist baseline absent)"; return 1; }
  done
  for m in data/dior_rsvg_official/manifests/train.jsonl data/vrsbench/manifests/train_grounding.jsonl \
           data/vrsbench/manifests/train_vqa.jsonl data/vrsbench/manifests/train_caption.jsonl \
           data/rsvqa_lr_official/manifests/train.jsonl data/rsicd/manifests/train.jsonl \
           data/levir_mci/manifests/train.jsonl data/instruct_mix_v2/manifests/train_no_rsvqa.jsonl; do
    [ -s "$m" ] || { log "GATE FAIL: $m missing"; return 1; }
  done
  log "GATE OK: specialist baselines and all eight training manifests present"
}

log "queue 4 started; waiting for sq-queue-night3"
wait_unit sq-queue-night3
gate_specialists || { log "queue 4 STOPPED at the specialist gate"; exit 1; }

# A. Unified multitask adapter, fresh LoRA r16, balanced sampling, 640-px cap.
#    Weights -> samples/epoch: DIOR 0.5 (13.5k) + VRS-ground 0.5 (16.7k) = 30k grounding;
#    RSVQA 0.3 (17.2k) + VRS-VQA 0.2 (15.7k) + mix 1.0 (3k) = 36k VQA; RSICD 0.25 (10.9k) +
#    VRS-cap 0.5 (9.2k) = 20k caption; LEVIR-CC 0.4 (13.6k) change caption. ~100k rows.
run train_unified_vlm $PY training/train_vlm_sft.py --model $BASE \
  --train data/dior_rsvg_official/manifests/train.jsonl data/vrsbench/manifests/train_grounding.jsonl \
          data/rsvqa_lr_official/manifests/train.jsonl data/vrsbench/manifests/train_vqa.jsonl \
          data/instruct_mix_v2/manifests/train_no_rsvqa.jsonl \
          data/rsicd/manifests/train.jsonl data/vrsbench/manifests/train_caption.jsonl \
          data/levir_mci/manifests/train.jsonl \
  --train-weight 0.5 0.5 0.3 0.2 1.0 0.25 0.5 0.4 \
  --val data/dior_rsvg_official/manifests/val.jsonl data/rsvqa_lr_official/manifests/val.jsonl \
        data/rsicd/manifests/val.jsonl data/levir_mci/manifests/val.jsonl --val-limit 300 \
  --ckpt-dir checkpoints/v3/unified_vlm --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 \
  --max-pixels 409600 --val-every 500 --save-every 250 --workers 8 --quant none \
  --notes "unified multitask LoRA r16: grounding/VQA/caption/change-caption, balanced weights, 640px cap"

# B. Same official tests as the specialists, unified adapter as an extra arm.
run eval_unified_ground $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best unified=checkpoints/v3/unified_vlm/adapter_best \
  --out artifacts/benchmark_reports/dior_rsvg_official_unified.json --batch 24
run eval_unified_vqa $PY evaluation/rsvqa_official_eval.py --base $BASE --data data/rsvqa_lr_official \
  --arms v3_official=checkpoints/v3/vqa_official/adapter_best unified=checkpoints/v3/unified_vlm/adapter_best \
  --out artifacts/benchmark_reports/rsvqa_lr_official_unified.json
run eval_unified_caption $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/rsicd/manifests/test.jsonl \
  --arms caption_lora=checkpoints/v3/caption_vlm/adapter_best unified=checkpoints/v3/unified_vlm/adapter_best \
  --out artifacts/benchmark_reports/rsicd_test_unified.json --batch 32
run eval_unified_cc $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/levir_mci/manifests/test.jsonl \
  --arms cc_lora=checkpoints/v3/change_caption_vlm/adapter_best unified=checkpoints/v3/unified_vlm/adapter_best \
  --out artifacts/benchmark_reports/levircc_test_unified.json --batch 16
log "queue 4 finished"

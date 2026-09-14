#!/usr/bin/env bash
# Unified multitask adapter (relaunch with the launch lock, 2026-09-14).
LOG=logs/queue_unified2.log
MIN_FREE_GB=${MIN_FREE_GB:-16}
source scripts/cluster_unit_lib.sh
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

log "unified unit started; polling the specialist gate"
until gate_specialists >> "$LOG" 2>&1; do sleep 600; done
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
  --ckpt-dir checkpoints/v3/unified_vlm --epochs 1 --batch-size 4 --grad-accum 4 --lr 1e-4 \
  --max-pixels 409600 --val-every 1000 --save-every 500 --workers 6 --quant none \
  --notes "unified multitask LoRA r16: grounding/VQA/caption/change-caption, balanced weights, 640px cap"

# B. Same official tests as the specialists, unified adapter as an extra arm.
run eval_unified_ground $PY evaluation/grounding_official_eval.py --base $BASE --data data/dior_rsvg_official \
  --arms lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best unified=checkpoints/v3/unified_vlm/adapter_best \
  --out artifacts/benchmark_reports/dior_rsvg_official_unified.json --batch 16
run eval_unified_vqa $PY evaluation/rsvqa_official_eval.py --base $BASE --data data/rsvqa_lr_official \
  --arms v3_official=checkpoints/v3/vqa_official/adapter_best unified=checkpoints/v3/unified_vlm/adapter_best \
  --out artifacts/benchmark_reports/rsvqa_lr_official_unified.json
run eval_unified_caption $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/rsicd/manifests/test.jsonl \
  --arms caption_lora=checkpoints/v3/caption_vlm/adapter_best unified=checkpoints/v3/unified_vlm/adapter_best \
  --out artifacts/benchmark_reports/rsicd_test_unified.json --batch 16
run eval_unified_cc $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/levir_mci/manifests/test.jsonl \
  --arms cc_lora=checkpoints/v3/change_caption_vlm/adapter_best unified=checkpoints/v3/unified_vlm/adapter_best \
  --out artifacts/benchmark_reports/levircc_test_unified.json --batch 16
log "unified unit finished"

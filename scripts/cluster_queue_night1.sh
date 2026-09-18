#!/usr/bin/env bash
# Phase 6 overnight queue, night 1 (2026-09-12). Runs on compute01 under
# `systemd-run --user`. Waits for the grounding LoRA run to finish, then
# executes the VLM experiments one after another - the L40S is shared and
# only one 3B LoRA job fits beside other users' work.
#
# Every step waits until at least MIN_FREE_GB of VRAM is free (never sizes
# from total VRAM, never touches another user's job) and logs to
# logs/queue_night1.log. A failed step is logged and the queue continues;
# nothing here deletes anything.
set -u
cd /scratch/home/adi01/satquery
PY=/scratch/home/adi01/satquery/.conda-env/bin/python
LOG=logs/queue_night1.log
MIN_FREE_GB=${MIN_FREE_GB:-18}
BASE=models/qwen25_vl_3b

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

wait_unit() {  # wait for a systemd user unit to leave the active state
  while systemctl --user is-active --quiet "$1"; do sleep 60; done
}

wait_vram() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$(( (total - used) / 1024 ))
    if [ "$free" -ge "$MIN_FREE_GB" ]; then return 0; fi
    log "waiting for VRAM: ${free} GB free < ${MIN_FREE_GB} GB"
    sleep 120
  done
}

run() {  # run <name> <command...>
  local name=$1; shift
  wait_vram
  log "START $name"
  if "$@" >> "logs/$name.log" 2>&1; then log "DONE $name (exit 0)"; else log "FAILED $name (exit $?)"; fi
}

log "queue started; waiting for sq-train-ground-lora-e1"
wait_unit sq-train-ground-lora-e1
log "grounding run finished"

# 1. Grounding: official test, both arms in one report (paired McNemar).
run eval_ground_official $PY evaluation/grounding_official_eval.py --base $BASE \
  --data data/dior_rsvg_official \
  --arms zero_shot=BASE lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best \
  --out artifacts/benchmark_reports/dior_rsvg_official_phase6.json --batch 24

# 2. VQA: official RSVQA-LR train + aligned-label instruction mix (no rsvqa rows).
run train_vqa_official $PY training/train_vlm_sft.py --model $BASE \
  --train data/rsvqa_lr_official/manifests/train.jsonl data/instruct_mix_v2/manifests/train_no_rsvqa.jsonl \
  --val data/rsvqa_lr_official/manifests/val.jsonl data/instruct_mix_v2/manifests/val.jsonl \
  --val-limit 1000 --ckpt-dir checkpoints/v3/vqa_official --epochs 1 --batch-size 16 --grad-accum 1 \
  --lr 1e-4 --val-every 500 --save-every 250 --workers 8 --quant none \
  --notes "VQA arm: fresh LoRA r16, official RSVQA-LR train 57k + instruct_mix_v2 (aligned WHU labels)"

# 3. VQA official test in the deployed (4-bit) load path, new arm vs deployed v2.
run eval_vqa_official $PY evaluation/rsvqa_official_eval.py --base $BASE \
  --data data/rsvqa_lr_official \
  --arms v3_official=checkpoints/v3/vqa_official/adapter_best v2_deployed=checkpoints/v2/track_b_vqa/adapter_final \
  --out artifacts/benchmark_reports/rsvqa_lr_official_phase6.json

# 4. Caption: RSICD SFT, then test.
run train_caption_vlm $PY training/train_vlm_sft.py --model $BASE \
  --train data/rsicd/manifests/train.jsonl --val data/rsicd/manifests/val.jsonl --val-limit 300 \
  --ckpt-dir checkpoints/v3/caption_vlm --epochs 1 --batch-size 16 --grad-accum 1 --lr 1e-4 \
  --val-every 400 --save-every 200 --workers 8 --quant none --notes "caption arm: RSICD 5 refs/image, fresh LoRA r16"
run eval_caption_vlm $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/rsicd/manifests/test.jsonl \
  --arms base=BASE caption_lora=checkpoints/v3/caption_vlm/adapter_best \
  --out artifacts/benchmark_reports/rsicd_test_vlm.json --batch 32

# 5. Change caption: LEVIR-CC two-image SFT, then test.
run train_change_caption_vlm $PY training/train_vlm_sft.py --model $BASE \
  --train data/levir_mci/manifests/train.jsonl --val data/levir_mci/manifests/val.jsonl --val-limit 300 \
  --ckpt-dir checkpoints/v3/change_caption_vlm --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 \
  --val-every 400 --save-every 200 --workers 8 --quant none --notes "change caption arm: LEVIR-CC 5 refs/pair, two images, fresh LoRA r16"
run eval_change_caption_vlm $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/levir_mci/manifests/test.jsonl \
  --arms base=BASE cc_lora=checkpoints/v3/change_caption_vlm/adapter_best \
  --out artifacts/benchmark_reports/levircc_test_vlm.json --batch 16

log "queue finished"

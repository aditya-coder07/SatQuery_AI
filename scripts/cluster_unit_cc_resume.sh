#!/usr/bin/env bash
# Change-caption arm: resume the night-1 run killed by the 2026-09-14 reboot
# at step 2,040/2,129 (adapter_last + train_state.pt), then the official test.
LOG=logs/queue_cc_resume.log
MIN_FREE_GB=${MIN_FREE_GB:-14}
set -u
cd /scratch/home/adi01/satquery
PY=/scratch/home/adi01/satquery/.conda-env/bin/python
BASE=models/qwen25_vl_3b
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
wait_vram() { while true; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); free=$(( (total - used) / 1024 )); [ "$free" -ge "$MIN_FREE_GB" ] && return 0; log "waiting for VRAM: ${free} GB free < ${MIN_FREE_GB} GB"; sleep 120; done; }
run() { local name=$1; shift; wait_vram; log "START $name"; if "$@" >> "logs/$name.log" 2>&1; then log "DONE $name (exit 0)"; else log "FAILED $name (exit $?)"; fi; }
log "cc-resume unit started"
run train_change_caption_vlm $PY training/train_vlm_sft.py --model $BASE \
  --train data/levir_mci/manifests/train.jsonl --val data/levir_mci/manifests/val.jsonl --val-limit 300 \
  --ckpt-dir checkpoints/v3/change_caption_vlm --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 \
  --val-every 400 --save-every 200 --workers 8 --quant none --resume \
  --notes "change caption arm: LEVIR-CC 5 refs/pair, two images, fresh LoRA r16 (resumed after the 2026-09-14 reboot)"
run eval_change_caption_vlm $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/levir_mci/manifests/test.jsonl \
  --arms base=BASE cc_lora=checkpoints/v3/change_caption_vlm/adapter_best \
  --out artifacts/benchmark_reports/levircc_test_vlm.json --batch 16
log "cc-resume unit finished"

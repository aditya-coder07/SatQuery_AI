#!/usr/bin/env bash
# 7B base (Qwen2.5-VL-7B-Instruct, Apache-2.0, 16.6 GB): download, then the
# arm-E recipe as QLoRA (4-bit base, LoRA r16) so it fits beside other jobs,
# from scratch (no 3B adapter to continue), DIOR-RSVG train + hard + VRSBench,
# 1024x1024 input; official test at the same budget in the 4-bit path.
LOG=logs/queue_7b.log
MIN_FREE_GB=${MIN_FREE_GB:-18}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
B7=models/qwen25_vl_7b
log "7B unit started"
if [ ! -f "$B7/model.safetensors.index.json" ]; then
  log "downloading Qwen/Qwen2.5-VL-7B-Instruct"
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='$B7', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json'])" >> "$LOG" 2>&1 \
    || { log "download failed"; exit 1; }
  log "download done: $(du -sh $B7 | cut -f1)"
fi
run train_ground_7b $PY training/train_vlm_sft.py --model $B7 \
  --train data/dior_rsvg_official/manifests/train.jsonl data/dior_rsvg_official/manifests/train_hard.jsonl data/vrsbench/manifests/train_grounding.jsonl \
  --train-weight 1.0 0.5 0.5 \
  --val data/dior_rsvg_official/manifests/val.jsonl --val-limit 400 \
  --ckpt-dir checkpoints/v3/grounding_7b_hires --epochs 1 --batch-size 2 --grad-accum 8 --lr 1e-4 \
  --min-pixels 1048576 --val-every 400 --save-every 200 --workers 6 --quant 4bit \
  --notes "7B QLoRA r16: DIOR-RSVG train 1.0x + hard 0.5x + VRSBench 0.5x, 1024px input, Apache-2.0 base"
if [ -f checkpoints/v3/grounding_7b_hires/adapter_best/adapter_model.safetensors ]; then
run eval_ground_7b $PY evaluation/grounding_official_eval.py --base $B7 --data data/dior_rsvg_official \
  --arms lora_7b=checkpoints/v3/grounding_7b_hires/adapter_best \
  --min-pixels 1048576 --out artifacts/benchmark_reports/dior_rsvg_official_7b.json --batch 8
fi
log "7B unit finished"

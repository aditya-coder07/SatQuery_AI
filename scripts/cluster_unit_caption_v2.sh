#!/usr/bin/env bash
# Caption / change-caption improvement arms (prepared 2026-09-20; launch when
# the cluster account is renewed). Each through the launch lock, in order of
# expected gain per GPU-hour. Rationale: docs/research/model_improvement_report.md.
#
#   C3  caption arm: the deployed recipe (fresh LoRA r16, RSICD train, 5 refs
#       as separate examples) run for 3 epochs with the checkpoint selected on
#       val_unseen (the 4xx val images none of whose references is a verbatim
#       train caption; training/prepare/rsicd_val_unseen.py). The official val
#       rewards memorised training sentences (32% of its references are train
#       captions vs 11% on test), which is why the deployed adapter read 0.42 on
#       val and 0.256 on test; selecting on it picks the step that memorised
#       most. Hypothesis: with an honest selection signal, 2-3 epochs beat 1.
#       Reported on the official test, as before. ~3x the 1-epoch cost.
#   CC2 change-caption arm: the deployed recipe continued from
#       change_caption_vlm/adapter_best for one more epoch at lr 3e-5, full-val
#       selection; the original run lost its last 89 steps to a reboot.
#   Both are scored on the official test splits afterwards by the same
#   evaluator that produced the deployed numbers (rsicd_test_vlm.json,
#   levircc_test_vlm.json), so the comparison is like for like.
LOG=logs/queue_caption_v2.log
MIN_FREE_GB=${MIN_FREE_GB:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/cluster_unit_lib.sh
log "caption v2 queue started"

run prep_rsicd_val_unseen $PY training/prepare/rsicd_val_unseen.py --manifests data/rsicd/manifests
run train_caption_e3 $PY training/train_vlm_sft.py --model $BASE \
  --train data/rsicd/manifests/train.jsonl \
  --val data/rsicd/manifests/val_unseen.jsonl --val-limit 100000 \
  --ckpt-dir checkpoints/v3/caption_vlm_e3 --epochs 3 --batch-size 16 --grad-accum 1 --lr 1e-4 \
  --val-every 800 --save-every 400 --workers 8 --quant none --eval-batch 32 \
  --notes "caption arm C3: deployed recipe x3 epochs, checkpoint selected on val_unseen (no memorisable references)"
if [ -f checkpoints/v3/caption_vlm_e3/adapter_best/adapter_model.safetensors ]; then
run eval_caption_e3 $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/rsicd/manifests/test.jsonl \
  --arms caption_e3=checkpoints/v3/caption_vlm_e3/adapter_best caption_lora=checkpoints/v3/caption_vlm/adapter_best \
  --out artifacts/benchmark_reports/rsicd_test_vlm_e3.json --batch 32 --max-new-tokens 64
fi

run train_change_caption_cont $PY training/train_vlm_sft.py --model $BASE \
  --train data/levir_mci/manifests/train.jsonl \
  --val data/levir_mci/manifests/val.jsonl --val-limit 100000 \
  --init-adapter checkpoints/v3/change_caption_vlm/adapter_best \
  --ckpt-dir checkpoints/v3/change_caption_vlm_cont --epochs 1 --batch-size 8 --grad-accum 2 --lr 3e-5 \
  --val-every 500 --save-every 250 --workers 8 --quant none --eval-batch 16 \
  --notes "change-caption arm CC2: continued from adapter_best one epoch at lr 3e-5, full-val selection"
if [ -f checkpoints/v3/change_caption_vlm_cont/adapter_best/adapter_model.safetensors ]; then
run eval_change_caption_cont $PY evaluation/vlm_task_eval.py --base $BASE --manifest data/levir_mci/manifests/test.jsonl \
  --arms cc_cont=checkpoints/v3/change_caption_vlm_cont/adapter_best cc_lora=checkpoints/v3/change_caption_vlm/adapter_best \
  --out artifacts/benchmark_reports/levircc_test_vlm_cont.json --batch 16 --max-new-tokens 64
fi
log "caption v2 queue finished"

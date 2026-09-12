# Training summary — generated from the experiment registry

Source: `artifacts\experiment_registry\registry.jsonl` (29 experiments). Regenerate with `python scripts/registry_report.py`. Do not edit by hand.

## Training runs

| experiment_id | model / architecture | status | steps | GPU-h | checkpoint | validation | test | notes |
|---|---|---|---|---|---|---|---|---|
| phase5-change_mask | change_mask / v2 (training/v2/architectures.py) | done | — | 1.86 | checkpoints/v2/change_mask | f1 0.8550; iou 0.7467 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-track_a | track_a / v2 (training/v2/architectures.py) | done | — | 2.34 | checkpoints/v2/track_a | map_all_bands 0.3150; retention 0.9262 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-track_a_nodropout | track_a_nodropout / v2 (training/v2/architectures.py) | done | — | 2.76 | checkpoints/v2/track_a_nodropout | map_all_bands 0.3015; retention 0.8775 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-optsar_fusion | optsar_fusion / v2 (training/v2/architectures.py) | done | — | 0.09 | checkpoints/v2/optsar_fusion | complementarity_gain -0.0301; fused 0.7420 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-track_b_vqa | track_b_vqa / qwen2.5-vl-3b + qlora | done | — | 3.79 | checkpoints/v2/track_b_vqa |  | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-grounding | grounding / v2 (training/v2/architectures.py) | done | — | 0.66 | checkpoints/v2/grounding | acc@0.5 0.1262; miou 0.1650 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-caption | caption / v2 (training/v2/architectures.py) | done | — | 0.32 | checkpoints/v2/caption | bleu4_sentence_mean 0.2255 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-change_caption | change_caption / v2 (training/v2/architectures.py) | done | — | 0.79 | checkpoints/v2/change_caption | bleu4_aggregate 0.5746; bleu4_changed 0.1641 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-change_vqa | change_vqa / v2 (training/v2/architectures.py) | done | — | 1.90 | checkpoints/v2/change_vqa | miou_change_classes 0.2933 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-caption_pre | caption_pre / v2 (training/v2/architectures.py) | done | — | — | checkpoints/v2/caption_pre | bleu4_sentence_mean 0.2658 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-change_vqa_scratch | change_vqa_scratch / v2 (training/v2/architectures.py) | done | — | — | checkpoints/v2/change_vqa_scratch | miou_change_classes 0.1730 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-grounding_pre | grounding_pre / v2 (training/v2/architectures.py) | done | — | — | checkpoints/v2/grounding_pre | acc@0.5 0.1604; miou 0.1974 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| phase5-track_b_vqa_es | track_b_vqa_es / qwen2.5-vl-3b + qlora | done | — | — | checkpoints/v2/track_b_vqa_es | best_step 1500; n_val 160; steps_trained 2250; val_loss_best 0.1301; val_loss_final 0.1395 | — | backfilled from docs/assets/phase5; hyperparameters in confi |
| optsar_v3-20260912-164402-805e5d | optsar_fusion_v3 / dual ImageNet-R50 + gated FPN triad | done | 3880 | 0.19 | checkpoints/v3/optsar_fusion/best.pt | fused 0.1963; fused_no_optical 0.1718; fused_no_sar 0.1966; gain_miou -0.0022; optical 0.1986; sar 0.1831 | — | — |
| optsar_v3-20260912-170418-f87e9c | optsar_fusion_v3 / dual ImageNet-R50 + gated FPN triad | done | 3880 | 0.20 | checkpoints/v3/optsar_fusion/best.pt | fused 0.4681; fused_no_optical 0.1073; fused_no_sar 0.4090; gain_miou 0.0214; optical 0.4467; sar 0.3883 | — | — |
| phase6-change_mask | change_mask / v3 | done | — | — | checkpoints/v3/change_mask/best.pt | epoch 34; f1 0.9055; iou 0.8274; precision 0.8720; recall 0.9417 | f1 0.9038; iou 0.8244; precision 0.8758; recall 0.9336 | backfilled from docs/assets/phase6 |
| landcover_v3-20260912-184016-e485da | landcover_v3 / SSL4EO-S12 MoCo ResNet-50 (12-band) + linear head | done | 14040 | 0.71 | checkpoints/v3/landcover/final.pt | — | f1_micro@0.5 0.3813; map_all_bands 0.3394; map_cartosat_4band 0.3099; map_micro_all_bands 0.4057; retention 0.9131 | — |
| landcover_v3-20260912-192918-a2f944 | landcover_v3 / SSL4EO-S12 MoCo ResNet-50 (12-band) + linear head | done | 10530 | 0.53 | checkpoints/v3/landcover_holdout/final.pt | — | f1_micro@0.5 0.6313; map_all_bands 0.5362; map_cartosat_4band 0.4753; map_micro_all_bands 0.6851; retention 0.8864 | — |
| grounding_vlm-20260912-163618-f20498 | models/qwen25_vl_3b / qwen2.5-vl+lora | done | 1686 | 6.38 | checkpoints/v3/grounding_vlm_r16/adapter_best | acc@0.25 0.7825; acc@0.5 0.6950; acc@0.7 0.5425; miou 0.6084; n 400; parse_rate 1.0000 | — | arm A resumed at step 400 with a 1-epoch cosine schedule |
| landcover_v3-20260912-220043-7cab7d | landcover_v3 / SSL4EO-S12 MoCo ResNet-50 (12-band) + linear head | done | 63180 | 2.05 | checkpoints/v3/landcover_full/best.pt | map_micro 0.8845 | f1_micro@0.5 0.7852; map_all_bands 0.7923; map_cartosat_4band 0.7463; map_micro_all_bands 0.8850; retention 0.9419 | full official BigEarthNet-S2 v1.0 split (269,695 train); SSL |
| vlm_sft-20260913-003408-b79298 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/vqa_official | — | — | VQA arm: fresh LoRA r16, official RSVQA-LR train 57k + instr |
| grounding_vlm-20260912-143014-f66340 | models/qwen25_vl_3b / qwen2.5-vl+lora | superseded | — | — | checkpoints/v3/grounding_vlm_r16 | — | — | first arm-A launch (2 epochs); stopped at step 400 and resum |

## Evaluations

| experiment_id | checkpoint | headline | details |
|---|---|---|---|
| eval_grounding_zero_shot-20260912-152132-f7801a | BASE | 0.3823 | acc@0.5 0.3823; benchmark DIOR-RSVG official test split; miou 0.3816 |
| phase5-rsvqa_official_test | checkpoints/v2/track_b_vqa/adapter_final | — |  |
| robustness_change_mask-20260912-184237-758558 | checkpoints/v3/change_mask/best.pt | — |  |
| eval_levircd_independent-20260912-202439-24919c | checkpoints/v3/change_mask/best.pt | 0.9038 | f1 0.9038; iou 0.8244; n 2048; precision 0.8758; recall 0.9336 |
| eval_grounding_lora_r16-20260913-003359-7170fe | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.6877 | acc@0.5 0.6877; benchmark DIOR-RSVG official test split; miou 0.6028 |
| eval_grounding_zero_shot-20260913-003359-e34422 | BASE | 0.3823 | acc@0.5 0.3823; benchmark DIOR-RSVG official test split; miou 0.3816 |

## Lineage

| Tool | BASELINE (Phase 5) | EXPERIMENT | BEST CHECKPOINT | BEST SCORE | DELTA |
|---|---|---|---|---|---|
| change_mask | 0.8550 | eval_levircd_independent-20260912-202439-24919c | checkpoints/v3/change_mask/best.pt | 0.9038 | +0.0488 |
| optsar_fusion | -0.0301 | optsar_v3-20260912-170418-f87e9c | checkpoints/v3/optsar_fusion/best.pt | 0.0214 | +0.0516 |
| grounding | 0.1604 | eval_grounding_lora_r16-20260913-003359-7170fe | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.6877 | +0.5273 |
| landcover | 0.3150 | landcover_v3-20260912-220043-7cab7d | checkpoints/v3/landcover_full/best.pt | 0.7923 | +0.4774 |

# Training summary — generated from the experiment registry

Source: `artifacts\experiment_registry\registry.jsonl` (71 experiments). Regenerate with `python scripts/registry_report.py`. Do not edit by hand.

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
| grounding_vlm-20260912-143014-f66340 | models/qwen25_vl_3b / qwen2.5-vl+lora | superseded | — | — | checkpoints/v3/grounding_vlm_r16 | — | — | first arm-A launch (2 epochs); stopped at step 400 and resum |
| scd_landsat-20260913-004701-b35192 | scd_landsat_v3 / siamese ImageNet-R50 + FPN over [fa, fb, |fa-fb|], 10-way ch | done | 25640 | 0.67 | checkpoints/v3/scd_landsat/best.pt | epoch 39; score 0.4922 | f1_change_binary 0.8507; iou_change_binary 0.7403; miou_all 0.6077; miou_change_types 0.5726; oa 0.9289; score 0.5044 | — |
| vlm_sft-20260913-003408-b79298 | models/qwen25_vl_3b / qwen2.5-vl+lora | done | 3764 | 3.63 | checkpoints/v3/vqa_official/adapter_best | n 1533; selection 0.9330; step 3500 | — | VQA arm: fresh LoRA r16, official RSVQA-LR train 57k + instr |
| scd_landsat-20260913-025242-ab732f | scd_landsat_v3 / siamese ImageNet-R50 + FPN over [fa, fb, |fa-fb|], 10-way ch | done | 51280 | 1.98 | checkpoints/v3/scd_landsat_e80/best.pt | epoch 79; score 0.5402 | f1_change_binary 0.8699; iou_change_binary 0.7698; miou_all 0.6545; miou_change_types 0.6234; oa 0.9393; score 0.5526 | — |
| vlm_sft-20260913-052822-1d5938 | models/qwen25_vl_3b / qwen2.5-vl+lora | done | 2729 | 0.75 | checkpoints/v3/caption_vlm/adapter_best | n 300; selection 0.4231; step 2729 | — | caption arm: RSICD 5 refs/image, fresh LoRA r16 |
| vlm_sft-20260913-062621-be33a2 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/change_caption_vlm | — | — | change caption arm: LEVIR-CC 5 refs/pair, two images, fresh  |
| grounding_vlm-20260914-101402-77f2b0 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_r16_merger | — | — | arm B: LoRA r16 + trainable visual merger, official DIOR-RSV |
| vlm_sft-20260914-101404-a58960 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_vrs | — | — | arm C: arm A adapter continued on DIOR-RSVG train (0.4x) + V |
| vlm_sft-20260914-101409-c5da98 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/change_caption_vlm | — | — | change caption arm: LEVIR-CC 5 refs/pair, two images, fresh  |
| grounding_vlm-20260914-101425-11448d | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_r16_merger | — | — | arm B: LoRA r16 + trainable visual merger, official DIOR-RSV |
| grounding_vlm-20260914-102757-9e6b90 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_r16_merger | — | — | arm B: LoRA r16 + trainable visual merger, official DIOR-RSV |
| vlm_sft-20260914-103246-a4909d | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_vrs | — | — | arm C: arm A adapter continued on DIOR-RSVG train (0.4x) + V |
| grounding_vlm-20260914-110437-d00458 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_r16_merger | — | — | arm B: LoRA r16 + trainable visual merger, official DIOR-RSV |
| vlm_sft-20260914-132653-5fb8da | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/unified_vlm | — | — | unified multitask LoRA r16: grounding/VQA/caption/change-cap |
| vlm_sft-20260914-150500-98eeed | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_vrs | — | — | arm C: arm A adapter continued on DIOR-RSVG train (0.4x) + V |
| grounding_vlm-20260914-151558-b86256 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_r16_merger | — | — | arm B: LoRA r16 + trainable visual merger, official DIOR-RSV |
| vlm_sft-20260914-151828-0cb48a | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_vrs | — | — | arm C: arm A adapter continued on DIOR-RSVG train (0.4x) + V |
| grounding_vlm-20260914-152929-65d884 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_r16_merger | — | — | arm B: LoRA r16 + trainable visual merger, official DIOR-RSV |
| vlm_sft-20260914-153433-8ea711 | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/unified_vlm | — | — | unified multitask LoRA r16: grounding/VQA/caption/change-cap |
| grounding_vlm-20260915-104831-ef1556 | models/qwen25_vl_3b / qwen2.5-vl+lora | done | 1686 | 5.73 | checkpoints/v3/grounding_vlm_r16_merger/adapter_best | acc@0.25 0.7825; acc@0.5 0.6875; acc@0.7 0.5300; miou 0.5974; n 400; parse_rate 1.0000 | — | arm B: LoRA r16 + trainable visual merger, official DIOR-RSV |
| vlm_sft-20260915-105659-22128d | models/qwen25_vl_3b / qwen2.5-vl+lora | done | 2764 | 7.17 | checkpoints/v3/grounding_vlm_vrs/adapter_best | n 800; selection 0.6562; step 2400 | — | arm C: arm A adapter continued on DIOR-RSVG train (0.4x) + V |
| vlm_sft-20260915-105203-0a378c | models/qwen25_vl_3b / qwen2.5-vl+lora | done | 6243 | 11.65 | checkpoints/v3/unified_vlm/adapter_best | n 1200; selection 0.6382; step 5000 | — | unified multitask LoRA r16: grounding/VQA/caption/change-cap |
| vlm_sft-20260915-200235-f49006 | models/qwen25_vl_3b / qwen2.5-vl+lora | done | 1809 | 8.21 | checkpoints/v3/grounding_vlm_hard/adapter_best | n 400; selection 0.7200; step 1809 | — | arm D: hard negatives (multi-instance DIOR-RSVG train, 1.0x) |
| vlm_sft-20260916-052737-61181d | models/qwen25_vl_3b / qwen2.5-vl+lora | running | — | — | checkpoints/v3/grounding_vlm_hard_vrs | — | — | arm D: hard negatives (multi-instance DIOR-RSVG train, 1.0x) |

## Evaluations

| experiment_id | checkpoint | headline | details |
|---|---|---|---|
| eval_grounding_zero_shot-20260912-152132-f7801a | BASE | 0.3823 | acc@0.5 0.3823; benchmark DIOR-RSVG official test split; miou 0.3816 |
| phase5-rsvqa_official_test | checkpoints/v2/track_b_vqa/adapter_final | — |  |
| robustness_change_mask-20260912-184237-758558 | checkpoints/v3/change_mask/best.pt | — |  |
| eval_levircd_independent-20260912-202439-24919c | checkpoints/v3/change_mask/best.pt | 0.9038 | f1 0.9038; iou 0.8244; n 2048; precision 0.8758; recall 0.9336 |
| eval_grounding_lora_r16-20260913-003359-7170fe | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.6877 | acc@0.5 0.6877; benchmark DIOR-RSVG official test split; miou 0.6028 |
| eval_grounding_zero_shot-20260913-003359-e34422 | BASE | 0.3823 | acc@0.5 0.3823; benchmark DIOR-RSVG official test split; miou 0.3816 |
| robustness_landcover-20260913-020129-cb0509 | checkpoints/v3/landcover_full/best.pt | — | clean_micro 0.8868; n 20000; worst noise_0.3; worst_micro 0.7168 |
| eval_grounding_lora_r16-20260913-044528-19d72c | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.6780 | acc@0.5 0.6780; headline 0.6780; manifest data/dior_rsvg_official/manifests/test.jsonl; miou 0.5959; n 7500 |
| eval_caption_base-20260913-062614-c979f4 | BASE | 0.0223 | headline 0.0223; manifest data/rsicd/manifests/test.jsonl; n 1093; unique_fraction 0.9762 |
| eval_caption_caption_lora-20260913-062614-316da0 | checkpoints/v3/caption_vlm/adapter_best | 0.2560 | headline 0.2560; manifest data/rsicd/manifests/test.jsonl; n 1093; unique_fraction 0.6002 |
| eval_change_caption_base-20260914-104343-9166dd | BASE | 0.0136 | headline 0.0136; manifest data/levir_mci/manifests/test.jsonl; n 1929; unique_fraction 0.4277 |
| eval_change_caption_cc_lora-20260914-104343-617ff7 | checkpoints/v3/change_caption_vlm/adapter_best | 0.6045 | headline 0.6045; manifest data/levir_mci/manifests/test.jsonl; n 1929; unique_fraction 0.1379 |
| robustness_grounding-20260914-125259-efe254 | checkpoints/v3/grounding_vlm_r16/adapter_best | — | clean 0.6660; n 1000; worst hflip; worst_acc 0.4250 |
| eval_grounding_lora_r16_merger-20260915-175044-8661e4 | checkpoints/v3/grounding_vlm_r16_merger/adapter_best | 0.6736 | acc@0.5 0.6736; benchmark DIOR-RSVG official test split; miou 0.5938 |
| eval_grounding_lora_r16-20260915-195352-d5a9e6 | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.6896 | acc@0.5 0.6896; benchmark DIOR-RSVG official test split; miou 0.6027 |
| eval_grounding_lora_vrs-20260915-195352-dc548d | checkpoints/v3/grounding_vlm_vrs/adapter_best | 0.6880 | acc@0.5 0.6880; benchmark DIOR-RSVG official test split; miou 0.6041 |
| eval_grounding_lora_r16-20260916-010903-191ca2 | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.6896 | acc@0.5 0.6896; benchmark DIOR-RSVG official test split; miou 0.6027 |
| eval_grounding_unified-20260916-010903-022359 | checkpoints/v3/unified_vlm/adapter_best | 0.6379 | acc@0.5 0.6379; benchmark DIOR-RSVG official test split; miou 0.5645 |
| eval_grounding_lora_r16-20260916-012747-79e0cd | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.5011 | acc@0.5 0.5011; headline 0.5011; manifest data/vrsbench/manifests/val_grounding.jsonl; miou 0.4564; n 16159 |
| eval_grounding_lora_vrs-20260916-012747-460359 | checkpoints/v3/grounding_vlm_vrs/adapter_best | 0.6325 | acc@0.5 0.6325; headline 0.6325; manifest data/vrsbench/manifests/val_grounding.jsonl; miou 0.5388; n 16159 |
| eval_caption_caption_lora-20260916-025701-748338 | checkpoints/v3/caption_vlm/adapter_best | 0.2546 | headline 0.2546; manifest data/rsicd/manifests/test.jsonl; n 1093; unique_fraction 0.6048 |
| eval_caption_unified-20260916-025701-e7de0f | checkpoints/v3/unified_vlm/adapter_best | 0.2349 | headline 0.2349; manifest data/rsicd/manifests/test.jsonl; n 1093; unique_fraction 0.5050 |
| eval_change_caption_cc_lora-20260916-032421-753ba2 | checkpoints/v3/change_caption_vlm/adapter_best | 0.6045 | headline 0.6045; manifest data/levir_mci/manifests/test.jsonl; n 1929; unique_fraction 0.1379 |
| eval_change_caption_unified-20260916-032421-31ae9d | checkpoints/v3/unified_vlm/adapter_best | 0.5949 | headline 0.5949; manifest data/levir_mci/manifests/test.jsonl; n 1929; unique_fraction 0.1037 |
| eval_grounding_init-20260916-052523-26658a | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.6896 | acc@0.5 0.6896; benchmark DIOR-RSVG official test split; miou 0.6027 |
| eval_grounding_lora_hard-20260916-052523-fda30c | checkpoints/v3/grounding_vlm_hard/adapter_best | 0.6992 | acc@0.5 0.6992; benchmark DIOR-RSVG official test split; miou 0.6131 |

## Lineage

| Tool | BASELINE (Phase 5) | EXPERIMENT | BEST CHECKPOINT | BEST SCORE | DELTA |
|---|---|---|---|---|---|
| change_mask | 0.8550 | eval_levircd_independent-20260912-202439-24919c | checkpoints/v3/change_mask/best.pt | 0.9038 | +0.0488 |
| optsar_fusion | -0.0301 | optsar_v3-20260912-170418-f87e9c | checkpoints/v3/optsar_fusion/best.pt | 0.0214 | +0.0516 |
| grounding | 0.1604 | eval_grounding_init-20260916-052523-26658a | checkpoints/v3/grounding_vlm_r16/adapter_best | 0.6896 | +0.5292 |
| landcover | 0.3150 | landcover_v3-20260912-220043-7cab7d | checkpoints/v3/landcover_full/best.pt | 0.7923 | +0.4774 |

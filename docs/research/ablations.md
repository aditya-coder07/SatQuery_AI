# Ablations — Phase 6

What caused each gain, one factor at a time where the arms exist. All rows
are measured by this project; sources are the report/metrics files named.
"Same data" means the same split and manifest hash.

## Change detection (LEVIR-CD official test, change-class F1 / IoU)

| Arm | Encoder | Loss / schedule | Aug | Selection | F1 | IoU | Source |
|---|---|---|---|---|---|---|---|
| v1 (Phase 3) | 0.05M scratch | BCE pos_weight, 4 ep | none | last | 0.5597 | 0.3886 | `checkpoints/change_mask/metrics.json` |
| v2 (Phase 5) | scratch residual, 60 ep | BCE pos_weight | none | last | 0.8550 | 0.7467 | `docs/assets/phase5/change_mask` |
| **v3** | ImageNet ResNet-50 siamese + FPN, 40 ep | BCE pos_weight + Dice, cosine, bf16 | dihedral + date swap | best val F1 | **0.9038** | **0.8244** | `docs/assets/phase6/change_mask` |
| v3, final-epoch weights | same | same | same | last | 0.9005 | 0.8190 | same file |
| v3 `--no-pretrained` | same trunk **from scratch**, 40 ep | same | same | best val F1 | 0.8751 | 0.7780 | `docs/assets/phase6/change_mask_scratch` (2026-09-14) |

Of the +4.9 F1 over v2, ImageNet initialisation is worth **+2.9** (the
scratch arm, same decoder/loss/augmentation/selection, reaches 0.8751);
the decoder + Dice + augmentation account for the remaining +2.0; the
val-selection alone is worth +0.3.

## Optical-SAR fusion (WHU-OPT-SAR, 7-class mIoU; per-tile gain = fused − optical)

| Arm | Labels | Split | Encoders | Task | optical | SAR | fused | gain | Source |
|---|---|---|---|---|---|---|---|---|---|
| v2 (Phase 5) | **misaligned** | tile-random (scene-leaky) | scratch | tile presence mAP | 0.772 | — | 0.742 | −0.030 | `docs/assets/phase5/optsar_fusion` |
| v3 on old labels | **misaligned** | scene-disjoint | ImageNet R50 ×2 | per-pixel | 0.199 | 0.183 | 0.196 | −0.002 | `docs/assets/phase6/optsar_fusion_mislabelled` |
| **v3** | aligned (re-cut) | scene-disjoint | ImageNet R50 ×2 | per-pixel | 0.447 | 0.388 | **0.468** | **+0.021** (per-tile +0.009, CI [+0.006, +0.012]) | `docs/assets/phase6/optsar_fusion` |
| v3, fused head with SAR zeroed | aligned | scene-disjoint | same | per-pixel | — | — | 0.409 | −0.038 vs optical | same file |

The label fix is the whole story: identical architecture and split, labels
one tile off → no learnable signal; aligned → a positive gain with a CI
that excludes zero. Modality dropout gives a fused head that degrades
gracefully without SAR (0.409, still above SAR-only) but not to optical
level: a "selective use" router should fall back to the optical head when
SAR is absent, which the tool's payload now makes possible.

## Grounding (DIOR-RSVG official test, Acc@0.5)

| Arm | Data | Model | Acc@0.5 | Source |
|---|---|---|---|---|
| Phase 5 CNN (pretrained) | 6,359 expressions **of the test split**, 15% self-made holdout | ImageNet R50 + GRU, box regression | 0.160 (Category C) | `docs/assets/phase5/grounding_pre` |
| Qwen2.5-VL-3B zero-shot | none | base | **0.3823** [0.371, 0.394] | `artifacts/benchmark_reports/dior_rsvg_official_zero_shot.json` |
| **A**: + LoRA r16 (LLM only), official train, 1 epoch, batch 8×2 | 26,991 | adapter | **0.6877** [0.677, 0.699]; image-disjoint 0.805; small/medium/large 0.509 / 0.782 / 0.888 | `dior_rsvg_official_phase6.json` |
| **B**: A's recipe + trainable visual merger, batch 2×8 (killed twice, resumed) | 26,991 | adapter + merger | 0.6736; image-disjoint 0.800; small/medium/large 0.491 / 0.770 / 0.878 — lower on every size bucket | `dior_rsvg_official_armB.json` |
| **C**: A's adapter continued on DIOR-RSVG ×0.4 + VRSBench grounding (33,440, CC-BY-4.0), native resolution | 44,236 | adapter | DIOR-RSVG 0.6880 vs arm A 0.6896 (McNemar n.s.); **VRSBench val 0.6325 vs 0.5011** (clean subset 0.586 vs 0.446; DOTA-only 0.550 vs 0.395) | `dior_rsvg_official_armC.json`, `vrsbench_val_grounding{,_subsets}.json` |
| **D**: arm A continued on the hard-negative subset (13,519 multi-instance train expressions ×1.0) + train ×0.2 + VRSBench ×0.3, lr 3e-5 | 29k | adapter | **0.6992** vs arm A 0.6896 in the same run — McNemar 216/288, χ² 10.0, **significant**; small objects 0.526 vs 0.510; wrong_object 365 vs 387, wrong_box 1,043 vs 1,099; val 0.72 | `dior_rsvg_official_armD.json` |
| **D′**: the same recipe from arm C | 29k | adapter | **0.7020** vs arm C 0.688 (McNemar 200/305, χ² 21.4, significant); **VRSBench val 0.6396** vs 0.6325 (clean 0.5946 vs 0.5863); small objects 0.529; val 0.72 — deployed until arm E clears VRSBench | `dior_rsvg_official_armD2.json`, `vrsbench_val_grounding_armD2{,_subsets}.json` |
| **E**: arm D′ continued with the processor's `min_pixels` raised to 1024² (800-px DIOR upscaled, ≈1.7× visual tokens), train ×0.6 + hard ×0.5 + VRSBench ×0.2, lr 3e-5 | 29.6k | adapter | **0.7323** vs D′ 0.7020 — McNemar 429/656, χ² 47.1, **significant**; small objects **0.578** (D′ 0.529), medium 0.819, large 0.895; image-disjoint 0.840; wrong_box 887 (D′ 1,027); val 0.765 | `dior_rsvg_official_armE.json` (VRSBench at the same budget pending) |
| Unified multitask (grounding + VQA + caption + change caption, balanced) | 100k rows | one adapter | **0.6379** on DIOR-RSVG — −5.2 pts vs the specialist, McNemar 661/273 significant: negative transfer on grounding | `dior_rsvg_official_unified.json` |
| 4-bit deployed path (arm A) | — | — | 0.678 (−1.0 pt vs bf16) | `dior_rsvg_official_lora_r16_4bit.json` |

Reading of arm B: training the merger did not help at one epoch — the
small-object bucket, the one it was meant to move, fell 1.8 pts; the run
also used batch 2×8 instead of 8×2 (same effective batch) after the admin
kills, so the two arms are not a perfectly controlled pair. Reading of arm C: 33k extra CC-BY-4.0 referring expressions at 512 px
changed nothing on DIOR-RSVG (±0.2 pt, n.s.) but lifted VRSBench val by
+13 pts (+14 on the clean subset, +16 on DOTA imagery the DIOR-RSVG
adapter never saw) — the same DIOR-RSVG score with far better
generalisation to other phrases and imagery. **Arm C is selected for
deployment** (tie on the official benchmark, dominant off it). Arm D confirmed the failure-directed loop: oversampling the multi-instance
expressions (the `wrong_object` teacher) gave +1.0 pt on the official test
with the misses moving exactly where the taxonomy pointed. Arm D′ applied it to arm C and gained on both benchmarks (+1.4 on
DIOR-RSVG, +0.7 on VRSBench). Arm E then confirmed the resolution
hypothesis the size breakdown and the 4× downscale test had made:
upscaling DIOR to 1024² is worth **+3.0 pts** on the official test, five of
them on small objects — the largest single gain after the base fine-tune.
A second epoch, a larger budget (1280²) and the 7B base remain.

## Unified multitask adapter vs specialists (official tests, 2026-09-16)

One LoRA r16 on the shared Qwen2.5-VL-3B base over grounding + VQA +
caption + change caption (≈100k rows/epoch, balanced weights 0.5/0.5/0.3/
0.2/1.0/0.25/0.5/0.4 so no task exceeds ~35%; 6,243 steps; resumed twice
after admin kills). Every task lost against its specialist; the loss is
large only for grounding.

| Task (official test) | Specialist | Unified | Δ | Test |
|---|---|---|---|---|
| DIOR-RSVG Acc@0.5 (n 7,500) | 0.6896 | **0.6379** | −5.2 pts | McNemar 661/273, χ² 160, significant |
| RSVQA-LR published convention (n 7,057) | 0.9119 [0.905, 0.918] | 0.9053 [0.898, 0.912] | −0.7 pt | CIs overlap |
| RSICD corpus BLEU-4 / CIDEr-D (n 1,093) | 0.2546 / 0.793 | 0.2349 / 0.745 | −2.0 pts | — |
| LEVIR-CC corpus BLEU-4 all / changed (n 1,929) | 0.6045 / 0.322 | 0.5949 / 0.284 | −1.0 / −3.8 pts | — |

Decision (charter §12): **specialists stay deployed** on the shared base;
the unified adapter (`checkpoints/v3/unified_vlm/adapter_best`, backed up)
is kept as the single-adapter fallback for a deployment that cannot
afford four adapter switches. The grounding loss is the negative transfer
the charter asked to measure: box-format outputs compete with free-text
tasks for the same LoRA capacity at r16; a per-task rank or a
grounding-only second stage would be the next arm if a unified model were
required.

## Captioning (corpus BLEU-4, 5 references)

| Task | Arm | BLEU-4 | CIDEr-D | Source |
|---|---|---|---|---|
| RSICD | v2 caption_pre (specialist, ImageNet R50 + transformer decoder) | 0.1744 | 0.562 | `artifacts/benchmark_reports/rescoring/rsicd_caption_pre.json` |
| RSICD | Qwen2.5-VL-3B zero-shot (base) | 0.022 | 0.065 | `artifacts/benchmark_reports/rsicd_test_vlm.json` — the base does not speak RSICD's caption register at all |
| RSICD | Qwen2.5-VL-3B + LoRA r16 (RSICD train, 1 epoch, 2,729 steps, 45 min) | **0.256** (ROUGE-L 0.486, meteor_exact 0.484) | **0.793** | same report; val (300) BLEU-4 0.42 — the val/test gap is not duplicate leakage (`dup_rsicd_train_val.json`: 0), it is the 300-item val sample; unique-caption fraction on test 0.60 |
| LEVIR-CC | v1 mask-conditioned GRU (**GT mask at test**) | 0.384 (changed half 0.222) | 1.29 | `artifacts/benchmark_reports/rescoring/levircc_change_caption_v1.json` |
| LEVIR-CC | Qwen2.5-VL-3B + LoRA r16, two images, no mask (val-selected step 2,000/2,129) | official test running (2026-09-14) | | val (300) corpus BLEU-4 0.623, CIDEr-D 1.32 at step 2,000 |

## Land cover (BigEarthNet-19 test shard, 5,867 patches; macro mAP / 4-band retention)

| Arm | Encoder | Data | macro mAP (micro) | retention | Source |
|---|---|---|---|---|---|
| v2 (Phase 5) | scratch band-agnostic + FiLM GSD | 65,867 patches, 40 ep | 0.315 | 0.926 | `docs/assets/phase5/track_a` |
| v3 | SSL4EO-S12 MoCo ResNet-50 (12-band) | same 60,000 patches, 30 ep, band dropout 0.3 | 0.339 (micro 0.406) | 0.913 | `docs/assets/phase6/landcover` — Category B under geographic shift (L1 revised) |
| v3 `--no-pretrained` | same trunk from scratch | same | planned (night 2) | | |
| v3, in-subset holdout | SSL4EO-S12 | train p0–p2 (45,000) → test p3 (15,000; one acquisition shared) | 0.536 (micro 0.685) | 0.886 | `docs/assets/phase6/landcover_holdout` — Category B; p3 is 3 acquisitions p0–p2 barely cover |
| v2 rescored on p3 | Phase 5 v2 | trained on p0–p3, so p3 is **in its training set** | 0.900 | 0.625 | `artifacts/benchmark_reports/rescoring/track_a_v2_on_holdout_p3.json` — **not a holdout for v2**; kept only as evidence that the shard's labels are learnable (contradicts the first L1 reading) |
| v3, full official split | SSL4EO-S12 | 269,695 train / 20,000-item val subsample / 125,866 test | **0.792 macro / 0.885 micro** (best-val epoch 13; final epoch 0.792 / 0.878) | 0.94 | `docs/assets/phase6/landcover_full` — the only Category A row in this table; 2.05 GPU-h |

The subset rows above disagree with each other by 0.2–0.6 mAP for one
reason: which acquisitions the shards happen to contain. The full split
is the number.

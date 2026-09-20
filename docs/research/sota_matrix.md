# SOTA matrix — external results and comparability

**Started 2026-09-12; living document.** Every external figure is quoted
from a paper and **not reproduced by this project**. Comparability class
per prompt §22: **A** = same benchmark, same official split, same metric,
same modality; **B** = same benchmark but a differing split/metric/protocol
detail (stated); **C** = not comparable. "Beats SOTA" may only be claimed
against an A row.

Our figures: "Phase 5" = deployed as of 2026-09-12 morning; "Phase 6" =
this programme, updated as results land (`artifacts/benchmark_reports/`).

## Benchmark matrix

| Benchmark | Metric | Ours Phase 5 | **Ours Phase 6 (best so far)** | Published best (A/B) | Gap to best | Class | Notes |
|---|---|---|---|---|---|---|---|
| RSVQA-LR official test | published-convention acc (presence+comp+RU) | 0.8947 [0.887, 0.902] | **0.9119** [0.905, 0.918] (LoRA r16 on the official train, 4-bit deployed path; presence 0.913, comp 0.912, RU 0.88; count 0.270) | GeoChat 90.70 avg; UniRS 92.63; RingMo-Agent 93.10 (presence) | +1.2 vs GeoChat / −1.5 vs UniRS | **A** (official test, same convention) | count reported separately: 0.2195 |
| DIOR-RSVG official test | Acc@0.5 | 0.160 **(C — trained on test split, self-made subset)** | **0.7323** arm E (deployed; 7B QLoRA arm 0.7392, McNemar vs E χ² 2.29 n.s., not selected) (arm D′ + 1024² input; McNemar vs D′ χ² 47.1, significant; small objects 0.578) [≈0.722, 0.742]; arm D′ 0.7020; arm D 0.6992; arm A 0.6896; arm C 0.6880; arm B 0.6736; unified 0.6379; zero-shot 0.3823; image-disjoint 0.840 | LQVG 83.4; LPVA 82.8; GeoGround 77.7; EarthGPT 76.7 | −10.1 (arm E) | **A** for Phase 6 | image-disjoint subset 0.4764 |
| VRSBench val, visual grounding (16,159 refs; CC-BY-4.0) | Acc@0.5 | — | **0.6594** [0.652, 0.667] arm E (deployed; 1024² input); arm D′ 0.6396; arm C 0.6325; arm A 0.501; clean subset (11,316 refs whose DIOR source is not in DIOR-RSVG train) **0.6237** (D′ 0.5946); DOTA-only 0.5965 | VRSBench paper: LLaVA-1.5-ft 51.2, GeoChat-ft ~49.8, Mini-Gemini-ft 42.4; EarthDial 55.6 | +7 (clean) to +11 (full) | **A** (official val, same metric) — quote the clean subset when the adapter was trained on DIOR-RSVG train | arm C = arm A continued on DIOR-RSVG ×0.4 + VRSBench train grounding (quarantined) |
| LEVIR-CD official test | change F1 / IoU | 0.855 / 0.747 | **0.9038 / 0.8244** (v3, best-val epoch 34; P 0.876 R 0.934; independently re-scored, CI [0.899, 0.908]; 0.9093 / 0.8336 at the val-selected threshold 0.8) | ChangeGCC 92.06 / 85.28; AGCD 90.34 / 82.38; BIT 89.3; ChangeFormer 90.4 | −6.5 (Phase 5) | **B** | ours: 256-px tiles of the official test, F1 on change class pixels; papers use the same tiling but some report on 512 px |
| RSICD test | corpus BLEU-4 / CIDEr-D (5 refs) | **0.1744 / 0.562** (caption_pre rescored; the 0.266 was sentence-mean) | **0.256 / 0.793** (VLM LoRA r16 on RSICD train; ROUGE-L 0.486, meteor_exact 0.484; base zero-shot 0.022 / 0.065) | specialist captioners ~0.30–0.45 BLEU-4, CIDEr 1.5–2.5+ | −13 to −28 pts | **A** (corpus BLEU-4, official test list) | ROUGE-L 0.396 |
| LEVIR-CC test | corpus BLEU-4 all / changed-half (5 refs) | **0.384 / 0.222** (v1 rescored; 0.569 was 1-ref sentence-mean); CIDEr-D 1.29 | **0.6045 / 0.322** (VLM two-image LoRA, images only, no mask; CIDEr-D 1.29, ROUGE-L 0.712, meteor_exact 0.685; base zero-shot 0.014) | SAGE-CC 65.5; KCFI 65.3; Chg2Cap 64.4 | −27 pts | **B→A** (5-ref corpus BLEU-4 now; but v1 is scored with the GT change mask as input — oracle-mask, papers see only the two images) | fair comparison needs predicted masks; the VLM arm sees images only |
| SECOND (semantic change) | mIoU (change classes) | 0.293 | — (not retrained: unlicensed) | GSTM-SCD mIoU 73.5 / SeK 24.2; Semantic-CD 75.1 / 23.9 | n/a | **C** | metric definitions differ (theirs include no-change); dataset unlicensed |
| Landsat-SCD test (CC BY 4.0; self-made 3:1:1 split by original, 477 test pairs) | 10-way change-type mIoU / SeK / Score = 0.3 mIoU + 0.7 SeK; binary change F1 | none | **0.623 / 0.509 / 0.553**; binary F1 0.870, OA 0.939 (siamese R50 + FPN, **80 ep**, val-selected epoch 79; the 40-ep arm gave 0.573 / 0.460 / 0.504) | no published number on this split; the Landsat-SCD paper decodes per-date maps and reports on its own split | n/a | **B** | replaces SECOND as the licensed semantic-change row; val still rising at epoch 39 → longer schedule is the next arm |
| CDVQA test1 | overall acc | 0.6061 (Phase 3) | — | CDVQA paper baseline ~0.62–0.65 | −2 to −4 | **A** | oracle over GT change maps 0.997 |
| BigEarthNet-19 (S2 v1.0, torchgeo split) | mAP (micro, 12 bands) | 0.315 on 11% of patches (geographic prefix of the train shards, scored on the last test shard: Category B under shift, L1 revised) | **0.885 micro / 0.792 macro** on the complete official test (125,866 patches; val-selected epoch 13 of 30; F1 micro 0.785; 4-band retention 0.94) | SeaMo 88.54; SpectralGPT ~88; ResNet-50 ~70–80 (full data) | −0.04 to SeaMo (micro) | **A** for the full-split row (same benchmark, official split, micro mAP as the papers) — check each paper's train fraction before calling it a tie | per-class AP: marine 0.998, arable 0.948, coniferous 0.947 … beaches 0.518, coastal wetlands 0.594, industrial 0.599 |
| WHU-OPT-SAR (fusion) | mIoU optical / SAR / fused, 7 classes | −0.030 tile-mAP gain (leaky split, **misaligned labels** — void) | **0.447 / 0.388 / 0.468** (gain **+0.021**; per-tile +0.009, CI [+0.006, +0.012]), scene-disjoint, aligned labels | MCANet ~0.53, ASANet ~0.60 mIoU (full data, 512 px, 8 classes incl. background) | −8 to −13 absolute | **B** (our 256 px, 7 classes w/o background, 1,935-tile subset, scene-disjoint) | the *gain* is the deliverable; papers report +1–3 |

## Frontier general-purpose VLMs (zero-shot, remote-sensing benchmarks)

Not fine-tuned on any of the above; quoted from benchmark papers.

| Model | Benchmark | Result | Class vs ours |
|---|---|---|---|
| GPT-4o | GEOBench-VLM (ICCV 2025) | best on object classification; "struggles" on grounding, counting | C (different benchmark) |
| GPT-5 | UHR remote-sensing MLLM benchmark (arXiv 2512.17319) | best overall 74.0/58.0/66.0 on sub-tasks | C |
| Gemini 2.5 Pro | same | best on low-level perception | C |
| Qwen2-VL / Qwen2.5-VL | GEOBench-VLM; RSVQA-LR zero-shot in several papers | strong on event detection, non-optical; RSVQA-LR zero-shot far below fine-tuned | **A on DIOR-RSVG** for our own zero-shot measurement (0.3823) |

## Published-result cards (for the A/B rows)

| Paper | Year | Arch | Train data | Split | Metric | Modality | Pretrain | External data | Class |
|---|---|---|---|---|---|---|---|---|---|
| GeoChat (CVPR 2024) | 2024 | LLaVA-1.5 (Vicuna-7B) + CLIP-L | GeoChat-Instruct 318k (RSVQA-LR train incl.) | RSVQA-LR official test | per-type acc, count excluded | RGB | LLaVA | yes | A |
| UniRS | 2024 | InternVL-based | multi-temporal instruct | RSVQA-LR test | avg acc | RGB | yes | yes | A |
| LQVG | 2024 | Swin + BERT + query decoder | DIOR-RSVG train | official test | Pr@0.5 | RGB | ImageNet/BERT | no | A |
| GeoGround | 2024 | LVLM (7B) | 161k grounding instruct incl. DIOR-RSVG train | official test | Acc@0.5 | RGB | yes | yes | A |
| ChangeGCC | 2025 | conv global context | LEVIR-CD train | test | F1/IoU | RGB | ImageNet | no | B (tile size unstated) |
| AGCD | 2025 | SAM-based | LEVIR-CD train | test | F1/IoU | RGB | SAM | no | B |
| SAGE-CC / KCFI / Chg2Cap | 2023–25 | ResNet + transformer decoders | LEVIR-CC train | test (1,929) | corpus BLEU-4, 5 refs | RGB | ImageNet | no | B vs our sentence-mean |
| SeaMo | 2024 | ViT MAE, season-aware | BEN 10% fine-tune | BEN test | micro mAP | S2 12-band | SSL on S2 | no | B (we use 11% but different subset) |

Sources: see `docs/research/SATQUERY_SOTA_FINAL_REPORT.md` §References (URLs
collected 2026-09-12 from the web search in this session).

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
| RSVQA-LR official test | published-convention acc (presence+comp+RU) | 0.8947 [0.887, 0.902] | — (running: SFT on official train) | GeoChat 90.70 avg; UniRS 92.63; RingMo-Agent 93.10 (presence) | −1.2 / −3.2 | **A** (official test, same convention) | count reported separately: 0.2195 |
| DIOR-RSVG official test | Acc@0.5 | 0.160 **(C — trained on test split, self-made subset)** | **0.3823 zero-shot** [0.371, 0.394]; LoRA arm running | LQVG 83.4; LPVA 82.8; GeoGround 77.7; EarthGPT 76.7 | −45 (zero-shot) | **A** for Phase 6 | image-disjoint subset 0.4764 |
| LEVIR-CD official test | change F1 / IoU | 0.855 / 0.747 | v3 running (val F1 0.90 @ epoch 5) | ChangeGCC 92.06 / 85.28; AGCD 90.34 / 82.38; BIT 89.3; ChangeFormer 90.4 | −6.5 (Phase 5) | **B** | ours: 256-px tiles of the official test, F1 on change class pixels; papers use the same tiling but some report on 512 px |
| RSICD test | BLEU-4 | 0.266 (sentence-mean, smoothed, 5 refs) | — | specialist captioners ~0.30–0.45 corpus BLEU-4; RSGPT higher | not directly | **B** | sentence-mean ≠ corpus BLEU; corpus evaluator to be added |
| LEVIR-CC test | BLEU-4 aggregate / changed-half | 0.569 / 0.306 | — | SAGE-CC 65.5; KCFI 65.3; Chg2Cap 64.4 (aggregate, 1,929 pairs) | −8 aggregate | **B** | papers: corpus BLEU-4 over 5 refs; ours sentence-mean, 1 ref |
| SECOND (semantic change) | mIoU (change classes) | 0.293 | — | GSTM-SCD mIoU 73.5 / SeK 24.2; Semantic-CD 75.1 / 23.9 | n/a | **C** | metric definitions differ (theirs include no-change); dataset unlicensed |
| CDVQA test1 | overall acc | 0.6061 (Phase 3) | — | CDVQA paper baseline ~0.62–0.65 | −2 to −4 | **A** | oracle over GT change maps 0.997 |
| BigEarthNet-19 (v2) | mAP (micro, 12 bands) | 0.315 on 11% of patches | — | SeaMo 88.54; SpectralGPT ~88; ResNet-50 ~70–80 (full data) | −55 | **B** | data-limited (65,867 patches); macro vs micro to be pinned |
| WHU-OPT-SAR (fusion) | fused − best single | −0.030 (tile-random split, leaky) | — | ASANet / MCANet report fused > optical by ~1–3 mIoU | n/a | **C** (our split is scene-leaky; their task is segmentation) | to be re-split by scene |

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

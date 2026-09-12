# SatQuery AI — Phase 6 SOTA programme: final report

**Status: IN PROGRESS — 2026-09-12 22:00 (full-valid-data phase started).** Rows marked *pending* have a
run queued or running on `compute01`; every other number is measured and
its artefact is named. This document is regenerated as results land; the
per-benchmark ledger is `docs/research/sota_matrix.md`, the run ledger is
`docs/research/training_summary.md` (generated from the registry).

## 1. Headline table

| Benchmark (official split) | Metric | CURRENT BASELINE (Phase 5, deployed 2026-09-12 a.m.) | BEST NEW RESULT | PUBLISHED BEST | ABS. IMPROVEMENT | REL. | COMPARABILITY | STATISTICAL CONFIDENCE |
|---|---|---|---|---|---|---|---|---|
| RSVQA-LR test | published-convention acc | 0.8947 [0.887, 0.902] | *pending* (SFT on official train) | 90.70 GeoChat / 92.63 UniRS | — | — | A | Wilson CI + McNemar vs constant |
| DIOR-RSVG test | Acc@0.5 | 0.160 (**Category C**: trained on the test split) | **0.6877** [0.677, 0.699] (LoRA r16, official train; zero-shot 0.3823) | 83.4 LQVG / 77.7 GeoGround | +0.53 vs Phase 5 (+0.31 vs zero-shot) | +330% | A (Phase 6) | bootstrap 95% CI; McNemar 2,465 fixed / 174 broken, p ≪ 0.001 |
| LEVIR-CD test | change F1 / IoU | 0.8550 / 0.7467 | **0.9038 / 0.8244** (independently verified; 0.9093 / 0.8336 at the val-selected threshold) | 92.06 / 85.28 ChangeGCC | +0.049 / +0.078 | +5.7% / +10.4% | B (256-px tiling) | 2,048 tiles; val-selected; per-tile bootstrap CI [0.899, 0.908]; robustness suite |
| WHU-OPT-SAR (scene-disjoint) | fused − optical mIoU | −0.030 (tile-mAP, misaligned labels: **void**) | **+0.021** (0.447 → 0.468) | +1–3 mIoU reported by MCANet/ASANet | n/a (baseline void) | — | B | per-tile paired bootstrap CI [+0.006, +0.012] |
| RSICD test | corpus BLEU-4 / CIDEr-D | 0.1744 / 0.562 (rescored; 0.266 was sentence-mean) | *pending* (VLM SFT) | ~0.30–0.45 / 1.5–2.5+ | — | — | A | bootstrap CI |
| LEVIR-CC test | corpus BLEU-4 (5 refs) | 0.384 / changed 0.222 (v1, oracle mask; v2 0.388 / 0.231 — no regression) | *pending* (VLM, images only) | 65.5 SAGE-CC | — | — | B→A | bootstrap CI |
| BigEarthNet-19 (S2 v1.0) official test | micro mAP / macro mAP / 4-band retention | 0.315 macro on a geographic subset shard (Category B under shift; L1 revised) | **0.885 micro / 0.792 macro / retention 0.94** (SSL4EO-S12 trunk, complete official split, val-selected) | 88.5 SeaMo (micro), ~88.2 SpectralGPT | +0.48 macro vs Phase 5 (not comparable: different test) | — | **A** (official split, micro mAP) | 125,866 test patches; calibration ECE 0.0085 → 0.0030 on val; assertion threshold 0.69 at precision 0.90 / recall 0.61 |
| Landsat-SCD test (CC BY 4.0) | change-type mIoU / SeK / Score | none (SECOND unlicensed) | **0.573 / 0.460 / 0.504** (binary change F1 0.851) | no comparable published number on this split (GSTM-SCD-class methods report SeK ~20–30 on SECOND with a different decomposition) | — | — | B | 477 test pairs; val-selected |
| CDVQA test1 | overall acc | 0.6061 | unchanged | — | — | — | A | — |

## 2. What changed and why (chronological, evidence-first)

1. **Audit before compute.** Two data findings invalidated published
   numbers: the grounding mirror held only the official test split (every
   grounding score to date was trained on 85% of it) and every WHU-OPT-SAR
   label tile was cut one tile off its imagery (every fusion score to date
   was measured against unrelated pixels). Both are documented, fixed, and
   the old artefacts kept (`docs/research/dataset_audit.md`, findings G1,
   G2, F1, F2).
2. **Grounding** moved from a from-scratch CNN box regressor to the
   deployed Qwen2.5-VL base with a task adapter, on the official split.
3. **Change detection** gained a pretrained encoder, Dice, augmentation
   and val selection: +4.9 F1.
4. **Fusion** was re-asked as per-pixel segmentation with pretrained dual
   encoders, gated fusion and modality dropout, on aligned labels and a
   scene-disjoint split: the first positive complementarity result in the
   project, with a CI that excludes zero.
5. **Captioning** metrics were replaced by corpus conventions (the
   sentence-mean BLEU was ~9 points optimistic on RSICD) before any new
   captioner was trained.
6. **Licensing**: Landsat-SCD (CC BY 4.0) replaces SECOND as the
   semantic-change benchmark; the licence register is `licensing.md`.

## 3. DATA USED

| Task | Train | Selection | Test | Licence |
|---|---|---|---|---|
| grounding | DIOR-RSVG official train 26,991 | official val (400-item subsample) | official test 7,500 | CC-BY-NC-4.0 |
| VQA | RSVQA-LR official train 57,223 + instruct_mix_v2 (3,010, aligned WHU labels, no rsvqa rows) | official val (1,000-item subsample) | official test 10,004 | CC-BY-4.0 / mixed |
| change mask | LEVIR-CD train 7,120 tiles | official val 1,024 | official test 2,048 | academic-only |
| fusion | WHU-OPT-SAR 29 scenes / 1,557 tiles | 7 scenes / 378 tiles | same (no separate test) | unstated |
| caption | RSICD train 8,734 × 5 refs | RSICD val 300 | RSICD test 1,093 | unstated |
| change caption | LEVIR-CC train 6,815 × 5 refs | val 300 | test 1,929 | academic-only |
| landcover | BigEarthNet-S2 v1.0 official train 269,695 (subset 60,000 before 2026-09-12 20:00) | official val (20,000-item subsample; a second 20,000 subsample for calibration/threshold) | official test 125,866 | CDLA-Permissive-1.0 |
| semantic change | Landsat-SCD 1,431 originals + 3,703 aug copies | 477 originals | 477 originals | CC BY 4.0 |

## 4. MODEL USED

Qwen2.5-VL-3B-Instruct (Qwen research licence; no `trust_remote_code`) +
task LoRA adapters r16 on the language tower, sharing one 4-bit base in
deployment; ImageNet ResNet-50 (torchvision, BSD-3) siamese encoders for
change and fusion; SSL4EO-S12 MoCo ResNet-50 (CC-BY-4.0) for land cover.

## 5. COMPUTE USED

Single shared NVIDIA L40S 46 GB (`compute01`, RHEL 9), other users'
jobs present throughout (5–10 GB); sizes chosen from free VRAM. GPU-hours
per run are in the registry; Phase 6 day 1: ≈ 12 GPU-h measured so far.

## 6. KNOWN LIMITATIONS

* Grounding, caption and change-caption adapters are trained on
  non-commercial / academic-only datasets: the SIH deployment is fine, a
  commercial release of those adapters is not.
* Fusion is measured on a 1,935-tile subset of WHU-OPT-SAR at 256 px; the
  absolute mIoU is below full-data papers; the *gain* is the claim.
* Change captioning's Phase 5 number used the ground-truth mask at test
  time; the fair (images-only) number is the VLM arm's.
* Land cover before 20:00 was measured on a geographic prefix of the
  mirror (audit L1, revised); the complete official split (146 GB
  compressed, sha256-verified) was pulled the same evening and every
  land-cover number from here on is on it.
* Everything is one seed (42). Seed variance has not been measured; CIs
  are over test items, not over training runs.
* The unified (shared-adapter) study, HPO beyond defaults, hard-negative
  loops and the 7B base are not yet run; see `ablations.md` for the queue.

## 7. References

Collected 2026-09-12 (web search in this session): GeoChat (CVPR 2024),
UniRS (arXiv 2412.20742), RingMo-Agent (2507.20776), RS-HyRe-R1
(2604.17504), LQVG (LANMNG/LQVG), GeoGround (2411.11904), RSGround-R1
(2601.21634), ChangeGCC (Springer 2025), AGCD (2504.12619), ChangeRWKV
(2603.19606), SAGE-CC (2511.21420), KCFI (2409.12612), Chg2Cap, SeaMo
(2412.19237), SpectralGPT (2311.07113), GSTM-SCD (ISPRS 2025), Semantic-CD
(2501.06808), GEOBench-VLM (ICCV 2025), UHR RS MLLM benchmark (2512.17319),
SSL4EO-S12 (2211.07044), Landsat-SCD (IJDE 2022; figshare 19946135),
DIOR-RSVG (Zhan et al. 2023), WHU-OPT-SAR (Li et al. 2022).

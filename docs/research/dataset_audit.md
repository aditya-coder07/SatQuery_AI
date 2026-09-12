# Dataset audit — 2026-09-12

Every dataset on the cluster (`compute01:~/satquery/data`, 79 GB) and what
was found when its splits and integrity were checked before any new GPU
work. Sizes are `du -sh`. Licences are in `licensing.md`. Findings that
change a number or a plan are marked **FINDING**.

| Key | Size | Split used so far | Official? | Integrity / leakage result |
|---|---|---|---|---|
| `rsvqa_lr_official` | 189 M | official test, 10,004 q / 100 img | **yes** | typed by dataset; constant baseline reported alongside. Clean. |
| `rsvqa_lr_2k` | 337 M | 90/10 of the HF **validation** redistribution | no (val) | train-only use since Phase 4; never scored as a benchmark again |
| `dior_rsvg` (parquet mirror) | 1.9 G | 85/15 by image of **the official test split** | **no** | **FINDING G1:** mirror holds only `test-*` shards (7,500 rows). Phase 5 grounding trained on 6,359 official-test expressions; card said "~38k". Numbers 0.076 / 0.126 / 0.160 are Category C. **Superseded.** |
| `dior_rsvg_official` (new) | 5.4 G | official 26,991 / 3,829 / 7,500 | **yes** | validator: 0 bad rows. **FINDING G2:** official protocol is object-level, so 4,206 of 6,102 test images also appear in train with a different phrase; reported as-is (comparability) plus an image-disjoint subset (1,896 images). |
| `rsicd` | 502 M | official test 1,093 | yes | vocab from train only; BLEU is sentence-mean smoothed, 5 refs — **not corpus BLEU**, so Category B against papers |
| `levircd` | 737 M | official train/val/test, 256 px tiles | yes | clean; F1 on change class only, tiles never cross splits |
| `levir_mci` | 5.4 G | official test 1,929 | yes | half of test is "no change"; both halves reported since Phase 5 fix |
| `cdvqa` | 122 M | official test1, 39,686 q / 968 pairs | yes | 968 test ids never read during training (verified in Phase 5) |
| `second` | 2.3 G | CDVQA's own train ids, 1,600 / 400 | n/a | **no licence** — weights unpublishable; benchmark use flagged |
| `whu_opt_sar` | 2.9 G | random **by tile**, 1,548 / 387 | no | **FINDING F1 (leakage):** all 36 source scenes appear on both sides. Tiles are 512 px crops of ~5,500×3,700 scenes; neighbours share texture and season. Validation is optimistic; the −0.030 fusion gain is measured on a leaky split and must be re-measured scene-disjoint. Tile names encode `SCENE_ROW_COL`, so a scene split is possible. |
| `whu_opt_sar` labels | — | — | — | **FINDING F2 (label misalignment, critical):** `prepare/whu_opt_sar.py` cut each label tile at `row*512, col*512` treating the `_RR_CC` suffix as 0-based; the mirror's indices are **1-based** (1..6 × 1..9). Every label tile was one tile down and one right of its imagery. Detected by the NDWI test (water-labelled pixels vs the rest: old labels +0.001, re-cut labels **+0.184** over 322 tiles; six crop hypotheses in `prepare/whu_opt_sar_relabel.py`). Consequences: every `optsar_fusion` result (v1 −0.006, v2 −0.030, v3-on-old-labels −0.002) was trained and scored on labels unrelated to the pixels — the "fusion adds nothing" conclusion is **void**, not negative; the WHU rows of `instruct_mix` (2,786 VQA answers such as "forest covers about 78%" + 122 refusals) were wrong, so the deployed VQA adapter learned from mislabelled SAR/optical questions. Fixed: `prepared/lbl_v2/`, `index_v2*.json`, `instruct_mix_v2/`. Old artefacts kept for reproducibility. |
| `ben_full` | 43 G | prepared HDF5 shards, 65,867 patches (11% of 590k) | partial | official BEN v2 split ids honoured within the subset; clean. Data-limited, not leaky. |
| `instruct_mix` | 1.2 M | pointers into `whu_opt_sar` + `rsvqa_lr_2k` | n/a | inherits F1 for its `whu_opt_sar` rows (VQA-style questions over tiles) |

## Integrity checks run

* **Boxes / masks / captions:** DIOR-RSVG official — 38,320 objects parsed,
  0 out-of-bounds or degenerate boxes, 0 empty phrases, 17,402 images all
  present (`data/dior_rsvg_official/manifests/stats.json`).
* **Split-file overlap:** DIOR-RSVG index files are pairwise disjoint (the
  preparer refuses to run otherwise; tested).
* **Manifest hashes:** sha256 of each JSONL is in `stats.json`; the
  evaluator records them in every report, and the registry keys eval
  records on the test-manifest hash.
* **Old mirror vs official test:** not yet cross-checked row-by-row; both
  report 7,500 rows and the first rows match by id and box. Low priority
  because the mirror is retired.

## Consequences

1. Grounding is re-baselined on the official split from today. Zero-shot
   Qwen2.5-VL-3B: **0.3823 Acc@0.5** on the official test
   (`artifacts/benchmark_reports/dior_rsvg_official_zero_shot.json`).
2. Fusion (`optsar_fusion`) is re-split by scene AND re-labelled (F2)
   before any new fusion experiment; the Phase 5 −0.030 figure is kept as
   "leaky split, misaligned labels" history. With aligned labels the v3
   triad shows fused > optical from epoch 3 onward (first real
   complementarity signal in the project; final numbers in
   `docs/assets/phase6/optsar_fusion/`).
2b. `instruct_mix_v2` replaces `instruct_mix` for every new VQA run.
3. Caption BLEU stays sentence-mean for continuity but a corpus-BLEU /
   CIDEr / METEOR / ROUGE-L evaluator is added so papers can be compared
   (Category A needs corpus BLEU-4 on the standard test list).
4. Nothing was deleted. The retired parquet mirror stays under
   `data/dior_rsvg` for reproducibility of the Phase 5 numbers.

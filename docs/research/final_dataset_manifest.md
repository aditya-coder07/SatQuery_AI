# Final dataset manifest

Generated 2026-09-16 06:47 by `scripts/final_dataset_manifest.py` on the data host. Machine-readable copy: `artifacts/dataset_manifests/FINAL_DATASET_MANIFEST.json`. Nothing in the *train* column may be used unless `usable_for` includes train.

| key | dataset / version | licence | usable for | train | val | test | valid | quarantined | leakage | manifest hash |
|---|---|---|---|---|---|---|---|---|---|---|
| `dior_rsvg_official` | DIOR-RSVG — authors' Google Drive release, official train/val/test.txt | CC-BY-NC-4.0 | train+eval (non-commercial) | 26991 | 3829 | 7500 | 38320 | — | index files pairwise disjoint (enforced); 4,206 test images also in train BY THE OFFICIAL PROTOCOL (object-level split) - reported, not altered; 22 source-level | b0446929d0d9 |
| `dior_rsvg` | DIOR-RSVG (HF parquet mirror) — test shards only | CC-BY-NC-4.0 | none (retired: holds only the official test split; finding G1) | — | — | 7500 | 0 | 7500 official test rows - must never be trained on | was the Phase 5 training source (Category C numbers) | 44136fa355b3 |
| `rsvqa_lr_official` | RSVQA-LR — Zenodo 6344334 official lists | CC-BY-4.0 | train+eval | 57223 | 10005 | 10004 | 77232 | — | train/val/test images disjoint (0/0/0); selection on val only | 0a58c0fd2e0a |
| `rsvqa_lr_2k` | RSVQA-LR (HF 2k redistribution of the VALIDATION split) — dmarsili/RSVQA-LR-2k | CC-BY-4.0 | none (retired: official train replaces it; it is official-val material) | — | — | — | 0 | validation-split material; no longer trained on | disjoint from official test | 44136fa355b3 |
| `instruct_mix_v2` | SatQuery instruction mix — instruct_mix_v2 | derived: WHU-OPT-SAR (unstated) + RSVQA-LR (CC-BY-4.0) | train (train_no_rsvqa) | 4807 | 533 | — | 8350 | — | WHU rows scene-split; rsvqa rows are HF-val material (excluded from train_no_rsvqa) | 09419a7c6ce5 |
| `instruct_mix` | SatQuery instruction mix — instruct_mix | derived: WHU-OPT-SAR (unstated) + RSVQA-LR (CC-BY-4.0) | none (superseded) | 4806 | 534 | — | 0 | SUPERSEDED: built on misaligned WHU labels (F2) | WHU rows scene-split; rsvqa rows are HF-val material (excluded from train_no_rsvqa) | 7847d0508c7c |
| `vrsbench` | VRSBench — xiang709/VRSBench (per-image annotations; official EVAL files) | CC-BY-4.0 | train+eval | 130565 | — | — | 130565 | 11825 DIOR source image is in the DIOR-RSVG official val or test split | 8452 DIOR-RSVG val/test source images blocked from train; val rows on DIOR-RSVG train images flagged (grounding 4843 of 16159) | 1f0c0391c2f9 |
| `rsicd` | RSICD — HF parquet mirror, official split names | unstated (research use) | train+eval | 43670 | 1094 | 1093 | 45857 | — | train vs test: 0 exact, 0 near duplicates (dup_rsicd_train_test.json) | a155fd86f386 |
| `levir_mci` | LEVIR-CC / LEVIR-MCI — LEVIR-MCI release | academic-only | train+eval (non-commercial) | 34075 | 1333 | 1929 | 37337 | — | official split; GT masks never used by the VLM arm | 3b722575cfe6 |
| `levircd` | LEVIR-CD — official split, 256-px tiles | academic-only | train+eval (non-commercial) | 7120 | 1024 | 2048 | 10192 | — | tiles never cross splits; val for selection+calibration only | 4164846dbf39 |
| `whu_opt_sar` | WHU-OPT-SAR — lbl_v2 (1-based re-cut) + scene-disjoint split | unstated (research use) | train+eval | 1557 | 378 | — | 1935 | F1 leakage + F2 misalignment | scene-disjoint (29/7 scenes); NDWI alignment +0.184 vs +0.001 (relabel_report.json) | 63ded73f8af0 |
| `ben_v1_full` | BigEarthNet-S2 v1.0 (19-label) — lc-col/bigearthnet HDF5 mirror, torchgeo split lists | CDLA-Permissive-1.0 (BigEarthNet); mirror converter Apache-2.0 | train+eval | 269695 | 123723 | 125866 | 519284 | rows beyond the mapping CSV (converter off-by-one); rows beyond the mapping CSV (converter off-by-one); rows beyond the mapping CSV (converter off-by-one) | official split lists; patch ids pairwise disjoint (see split_id_overlap); test never read for selection; id overlap {'train&val': 0, 'test&train': 0, 'test&val' | 28a6effeddd2 |
| `ben_full` | BigEarthNet-S2 v1.0 (partial mirror) — train p0-p3 + test p8 | CDLA-Permissive-1.0 | none (superseded by ben_v1_full; geographic prefix, L1 revised) | 60000 | — | 5866 | 0 | 1 phantom zero row in test_p8 (converter off-by-one); 65866 not a representative sample of the official split | official split ids | 44136fa355b3 |
| `landsat_scd` | Landsat-SCD — figshare 19946135 v1 | CC BY 4.0 | train+eval | 5134 | 477 | 477 | 6088 | — | originals cut 3:1:1 by seed 42; augmented copies follow their original and are train-only; 0 copies without an original dropped | 3309783109b0 |
| `second` | SECOND — release | NONE STATED - blocked | none (licence unresolved; replaced by Landsat-SCD) | — | — | — | 0 | no licence; not trained on | n/a | 44136fa355b3 |
| `cdvqa` | CDVQA — GitHub annotations (Apache-2.0) over SECOND imagery | annotations Apache-2.0; imagery unlicensed | eval only (legacy Phase 5 number); no training | — | — | — | 0 | imagery licence unresolved | test ids never trained on (Phase 5 verified) | 6bdfcca0c3e9 |

## Rules applied

* Test splits are never read by a trainer; selection uses the split marked val.
* Quarantined rows are listed with the reason and are never sampled.
* `manifest_hash` is the sha256 over the per-file sha256s and is what the experiment registry records as `dataset_manifest_hash`.
* Licence-blocked data (SECOND imagery) is not trained on; CDVQA is evaluation-only history.

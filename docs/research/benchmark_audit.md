# Benchmark audit — Phase 6

For every number this programme reports: which split, how many items, the
exact metric, where the items come from, what leakage check was run, and
the comparability class against the literature. Manifest hashes are in each
dataset's `manifests/stats.json` and copied into every report under
`artifacts/benchmark_reports/`.

| Benchmark | Split | n | Metric (exact) | Provenance | Leakage checks | Class |
|---|---|---|---|---|---|---|
| RSVQA-LR | official test | 10,004 q / 100 img | exact match after `normalise_answer`; published convention = presence + comp + rural/urban micro-acc (count excluded, reported separately); Wilson 95% CI; McNemar vs train-fitted per-type constant | Zenodo 6344334 lists | train/val/test images disjoint (0/0/0 overlap, `rsvqa_official_train.py`); selection on official val only | A |
| DIOR-RSVG | official test | 7,500 expr / 6,102 img | Acc@0.25/0.5/0.7 (IoU with the single GT box), mIoU; bootstrap 95% CI; image-disjoint subset (1,896 img) as the stricter number | authors' Drive, `train/val/test.txt` indices over sorted XML objects | index files pairwise disjoint (enforced); 4,206 test images also in train **by the official protocol** (reported, not altered); old mirror (test-only) retired | A |
| LEVIR-CD | official test | 2,048 tiles (256 px) | change-class F1 / IoU / P / R at threshold 0.5, pixels pooled over tiles | Phase 3 preparer, official folders | tiles never cross splits; val used only for selection and calibration | B (papers vary tile size) |
| WHU-OPT-SAR | scene-disjoint val (7 of 36 scenes) | 378 tiles | 7-class pixel mIoU per arm; complementarity = fused − max(optical, SAR); per-tile paired bootstrap on fused − optical | authors' release; labels re-cut (F2) | scene split (F1); NDWI alignment test of labels (+0.184 vs +0.001) | B |
| RSICD | official test | 1,093 img × 5 refs | corpus BLEU-1..4 (no smoothing, closest-ref BP), ROUGE-L (β 1.2), CIDEr-D (n 1–4, σ 6, DF over test refs); bootstrap CI | HF parquet mirror, official split names | vocab from train only (specialist); VLM arm never reads test | A |
| LEVIR-CC | official test | 1,929 pairs × 5 refs | as RSICD; changed (964) / unchanged (965) halves | LEVIR-MCI index | VLM arm sees images only (no GT mask) — the Phase 5 specialist used the GT mask at test | B (specialist) / A (VLM) |
| BigEarthNet-19 v2 | test shard of the prepared subset | 5,867 patches | macro mAP over 19 classes (Phase 5 convention) + micro mAP + F1@0.5; 4-band retention = mAP(B02,B03,B04,B08 only) / mAP(12) | reBEN HDF5 shards | official split ids honoured within the subset; no selection on test | B (11% subset; micro mAP is the paper convention) |
| Landsat-SCD | test (477 originals) | 477 pairs | OA, mIoU over 10 change-type classes, binary change IoU/F1, SeK, Score = 0.3 mIoU + 0.7 SeK; pair bootstrap | figshare 19946135 (CC BY 4.0) | augmented copies never cross splits (grouped by original) | B (papers decode per-date maps) |
| CDVQA | official test1 | 39,686 q / 968 pairs | overall accuracy | GitHub annotations + SECOND imagery | test ids never read in training (Phase 5 verified) | A (imagery unlicensed) |

## Contamination checks not yet run

* Perceptual-hash near-duplicate search between DIOR-RSVG train and test
  images (the official split shares images by design; near-duplicates of
  *different* images are the open question).
* Cross-dataset overlap between RSICD and DIOR (both Google-Earth-derived;
  a cheap phash pass is queued for day 2).

## Reporting rules applied

* A result on a self-made split is labelled as such and never placed in
  an A row.
* Every VLM number is from greedy decoding with the deployed prompt
  format; bf16 for training-time evaluation, 4-bit for the deployed-path
  parity check (RSVQA already; grounding queued).
* No number is quoted without n; CIs are bootstrap over items unless the
  metric has a closed form (Wilson for accuracies).

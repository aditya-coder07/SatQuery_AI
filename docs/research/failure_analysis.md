# Failure analysis — Phase 6

Living document; every section names the report it was computed from.
Taxonomy definitions are in the evaluators (`evaluation/grounding_official_eval.py`,
`evaluation/vlm_task_eval.py`).

## Grounding — DIOR-RSVG official test, zero-shot Qwen2.5-VL-3B

Source: `artifacts/benchmark_reports/dior_rsvg_official_zero_shot.json`
(n = 7,500, Acc@0.5 0.3823).

| Failure class (misses only, 4,633) | Count | Share | Reading |
|---|---|---|---|
| wrong_box (IoU < 0.1, not another annotated object) | 2,058 | 44% | the model does not know the category or the phrase: windmill 3%, bridge 7%, harbor 10%, overpass 11%, toll station 14% |
| scale_error (centre inside GT, IoU < 0.5) | 1,202 | 26% | right place, wrong extent — mostly large diffuse classes (golf field, airport, dam) |
| localisation_drift (0.1 ≤ IoU < 0.5) | 901 | 19% | near misses; fine-tuning on the pretraining box format should collect most of these |
| wrong_object (IoU ≥ 0.5 with another labelled object) | 472 | 10% | the phrase disambiguation ("the ship on the left") is ignored |

By object size: large (>10% of image) 0.577, medium 0.400, small (<1%) 0.273
— 42% of the test set is small. IoU histogram is bimodal (2,530 below 0.1;
2,006 above 0.7): the model either finds the object or not; there is little
mass in between, so improvements have to come from recognition, not from
box refinement.

**What this directs:** train on the official 26,991 expressions with the
category as the answer label (free classification signal for the
domain-specific classes), keep native 800-px resolution (small objects), and
after the first LoRA arm measure `wrong_object` separately - it is the
class that phrase-conditioning must fix and the one that would justify
adding hard negatives (same image, other object of the same class).

### After arm A (LoRA r16, official train, 1 epoch) — 2026-09-13 00:34

Source: `artifacts/benchmark_reports/dior_rsvg_official_phase6.json`
(n = 7,500; **Acc@0.5 0.6877 [0.677, 0.699]**, Acc@0.25 0.769, Acc@0.7
0.545, mIoU 0.603, parse 1.0; image-disjoint subset 0.805 [0.788, 0.820],
n = 2,223). McNemar vs zero-shot: 2,465 items fixed, 174 broken, χ² 1987,
p ≪ 0.001. Val (400) 0.695 at the selected step 1,600, so val and test
agree within noise.

| Failure class (misses, 2,342 = 31% of test) | Count | Share | Was (zero-shot) |
|---|---|---|---|
| wrong_box | 1,085 | 46% | 2,058 |
| scale_error | 649 | 28% | 1,202 |
| wrong_object | 390 | 17% | 472 |
| localisation_drift | 218 | 9% | 901 |

* **Size is the axis.** Large 0.888, medium 0.782, **small (<1% of the
  image) 0.509** — small objects are 42% of the test set and contribute
  most misses. The IoU histogram is still bimodal (1,475 below 0.1).
* **Categories:** bridge 0.33, harbor 0.42, overpass 0.46, train station
  0.57, vehicle 0.62 at the bottom; stadium 0.91, airport 0.87, golf field
  0.86, airplane 0.85 at the top. The elongated/linear classes (bridge,
  overpass, harbor) fail as `wrong_box` — the box format has no orientation
  and the phrases rarely disambiguate which span of a river or road.
* **wrong_object fell only 17%** (472 → 390): phrase-conditioning is the
  least-improved class, which is the argument for hard negatives (same
  image, another object of the same class, contrastive phrasing) and for
  VRSBench's referring expressions, which are written to be discriminative.

**What this directs (arms B/C, queued):** keep native resolution (a 640-px
cap was removed from arm C for this reason), add VRSBench's 33k
discriminative expressions (arm C), try the trainable visual merger (arm
B) for the small-object end, and build the hard-negative set from the
390 `wrong_object` + 1,085 `wrong_box` items' images for the retrain that
follows.

## Change detection — LEVIR-CD, v3

Source: `docs/assets/phase6/change_mask/metrics.json`, independently
re-scored by `evaluation/change_mask_official_eval.py` (weights loaded
through the deployed tool's `_Handle`, metrics recomputed from PIL-read
tiles): **F1 0.9038 / IoU 0.8244 reproduced to the fourth decimal**;
per-tile bootstrap 95% CI F1 [0.899, 0.908], IoU [0.817, 0.832]; 2,048
tiles, 935 with change (`artifacts/benchmark_reports/levircd_test_independent_v3.json`).
The result is frozen as the champion (2026-09-12 21:00); it is retrained
only for a demonstrated weakness.

Weaknesses found by the re-score:

* **Over-prediction of change.** Precision 0.876 / recall 0.934 at 0.5,
  and F1 keeps rising with the threshold on both val and test (val: 0.9055
  at 0.5 → 0.9116 at 0.8; test 0.9038 → 0.9093 / IoU 0.8336 at the
  val-selected 0.8, P 0.910 R 0.909). The Dice term pushes probabilities
  up; the val-chosen operating point recovers +0.55 F1 with no retraining
  and is a legitimate deployment setting (chosen on val, not test). The
  0.5 number stays the headline for comparability.
* **A hard tail.** On tiles with change, the 10th-percentile per-tile F1
  is 0.42 (median 0.91): about a tenth of changed tiles are scored badly -
  small or thin structures and seasonal/shadow confusers. Those tiles are
  the hard-negative set for any retraining; the boundary-quality metric is
  still to be added.

## Change captioning — the "v2 regression" investigated (2026-09-12 21:20)

Phase 5 kept the v1 GRU captioner because v2 scored 0.1641 on the changed
half against v1's 0.3063. Re-scored with the corpus evaluator on the
official LEVIR-CC test (5 references; `artifacts/benchmark_reports/rescoring/
levircc_change_caption_{v1,v2}.json`):

| Arm (both mask-conditioned, GT mask at test) | unique captions | 1-ref changed (old metric) | 5-ref corpus BLEU-4 all / changed | CIDEr-D all / changed | ROUGE-L changed |
|---|---|---|---|---|---|
| v1 GRU | 85 | 0.1702 | 0.384 / 0.222 | 1.291 / 0.354 | 0.406 |
| v2 masked transformer | 638 | 0.1641 | **0.388 / 0.231** | 1.297 / 0.347 | 0.381 |

There is no regression. The 0.3063 figure was never reproducible from
code (Phase 2 finding D2); against the reproducible v1 number (0.170) the
two arms are within noise on every corpus metric, and v2 produces 7.5×
more distinct captions (v1 collapsed onto 85 templates, which the 1-ref
sentence-mean BLEU rewarded). What both arms share is the real
limitation: they read the ground-truth change mask at test time and are
still at 0.22–0.23 changed-half BLEU-4 against 0.65 for image-only SOTA.
The mask-conditioned family is not where the gap closes; the VLM arm
(two images, no mask; night-1 queue) is, and an explicit-change variant
of it (a third, |T1−T2| difference image) is the ablation to run if the
plain two-image adapter under-describes change.

## Land cover — BigEarthNet-S2 v1.0 official test, v3 full-split (2026-09-13)

Source: `docs/assets/phase6/landcover_full/metrics.json` (micro mAP 0.885 /
macro 0.792, n = 125,866) and the corruption suite
`artifacts/benchmark_reports/ben_robustness_landcover_v3.json`
(20,000-patch test subsample, `evaluation/robustness_landcover.py`).

* **Where the macro number is lost:** beaches/dunes AP 0.52, coastal
  wetlands 0.59, industrial units 0.60, natural grassland 0.61 — the four
  rarest classes; marine 0.998, arable 0.95, coniferous 0.95 at the top.
  The micro/macro gap (0.885 vs 0.792) is a class-frequency gap, not a
  modelling one; a class-balanced loss or logit adjustment is the arm
  that would move macro without touching micro.
* **Robustness (Δ micro mAP vs clean 0.8868):** dihedral transforms exact
  (−0.0001); any single band dropped ≤ −0.002 (band-dropout training
  worked — B11 is the most-relied-on band); Cartosat 4-band −0.026
  (retention 0.97 on this subsample, 0.94 on the full test); brightness
  ×0.8 / ×1.2 −0.039 / −0.027; a 40×40 bright cloud patch −0.078;
  gaussian noise σ 0.1 −0.008 but σ 0.3 **−0.170** — sensor noise far
  above Sentinel-2's is the one failure mode; a noise augmentation at
  σ ≤ 0.2 is cheap insurance for the next run.
* **Calibration:** ECE 0.0085 → 0.0030 (affine, official val); the
  assertion threshold 0.69 yields precision 0.90 at recall 0.61, so the
  deployed tool now asserts most present classes instead of 0.3% of them.

## Fusion — WHU-OPT-SAR, v3 (aligned labels)

Source: `docs/assets/phase6/optsar_fusion/metrics.json`. Per-class IoU
optical → fused: farmland 0.612 → 0.619, city 0.488 → 0.491, village
0.342 → 0.357, water 0.625 → 0.655, forest 0.827 → 0.830, road 0.188 →
0.242, others 0.046 → 0.083. SAR contributes where optical is ambiguous
(road vs bare soil, water vs shadow); "others" is unlearnable at this size.
The fused head loses 0.06 mIoU when SAR is zeroed at test time and 0.36
when optical is zeroed, so the trained selective use is: optical carries,
SAR corrects.

## Data failures found by the audit (not model failures)

* DIOR-RSVG: trained on the test split (G1). Fixed by the official release.
* WHU-OPT-SAR: labels one tile off (F2). Fixed by re-cutting; every
  fusion conclusion before 2026-09-12 is void.
* WHU-OPT-SAR: scene leakage across the tile-random split (F1). Fixed by
  the scene-disjoint index.

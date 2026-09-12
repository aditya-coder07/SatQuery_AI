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

*Post-training analysis will be appended from
`artifacts/benchmark_reports/dior_rsvg_official_phase6.json`.*

## Change detection — LEVIR-CD, v3

Source: `docs/assets/phase6/change_mask/metrics.json`. Precision 0.876 /
recall 0.934 at threshold 0.5 on the best-val weights: the remaining error
is false positives (shadows, roof re-surfacing, seasonal ground) more than
misses. Calibration (`docs/assets/calibration_v3/`) will move the operating
point; a boundary-quality metric is still to be added.

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

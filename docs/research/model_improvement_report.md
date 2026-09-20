# MODEL IMPROVEMENT REPORT — accuracy + natural-language understanding pass (2026-09-20)

Rule kept throughout: diagnose → controlled experiment → change → same
official split → regression suite → deploy. Nothing below was selected on
a favourable subset; every number names its split, its protocol and its
artefact.

## Current baseline (deployed, `configs/deploy.v3.yaml`, all official splits)

| Task | Tool / checkpoint | Benchmark, split, metric | Score | Published reference (class) |
|---|---|---|---|---|
| VQA | `vqa_official/adapter_best` (LoRA r16 on Qwen2.5-VL-3B, NF4) | RSVQA-LR test, published-convention acc | 0.9119 [0.905, 0.918] | UniRS 0.926, GeoChat 0.907 (A) |
| Grounding | `grounding_vlm_hires/adapter_best` (arm E, 1024²) | DIOR-RSVG test, Acc@0.5 | 0.7323 | LQVG 0.834, GeoGround 0.777 (A) |
| Grounding | same | VRSBench val, Acc@0.5 | 0.6594 (clean 0.6237) | EarthDial 0.556 (A) |
| Caption | `caption_vlm/adapter_best` | RSICD test, corpus BLEU-4 / CIDEr-D | 0.256 / 0.793 | RSGPT ≈ 0.30 / ≈ 1.0 (A) |
| Change caption | `change_caption_vlm/adapter_best` | LEVIR-CC test, BLEU-4 all / changed | 0.6045 / 0.322 | SAGE-CC 0.655 (B→A) |
| Change mask | `change_mask/best.pt` | LEVIR-CD test, F1 / IoU | 0.9038 / 0.8244 | ChangeGCC 0.921 (B) |
| Land cover | `landcover_full/best.pt` | BigEarthNet-S2 test, micro / macro mAP | 0.885 / 0.792 | SeaMo 0.885 (A) |
| Opt-SAR fusion | `optsar_fusion/best.pt` | WHU-OPT-SAR (scene-disjoint), fused−optical mIoU | +0.021 | +0.01–0.03 (B) |
| Change VQA | `v2/change_vqa/best.pt` (held) | CDVQA test1 acc | 0.606 | ≈ 0.62–0.65 (A) |
| **NL routing** | `tfidf_logreg_v1` | `evaluation/nl/queries.jsonl` (hand-written) | **71.7%** (two-date inputs 58.3%) | — |

## Weakest tasks, ranked by gap × user impact

1. **Natural-language understanding** — 28% of ordinary queries reached a
   tool that could not answer them; on two-image inputs 42%. Parameters
   (the object to find, the classes to map, which image) were never
   extracted, so even correctly routed queries ran with the wrong inputs.
2. **Captioning** (−5 BLEU-4 / −0.2–0.3 CIDEr to the published range).
3. **Change captioning** (−5 BLEU-4).
4. **Grounding** (−10 Acc@0.5 to the best specialist; already +4 over the
   best VLM competitor on VRSBench). Small objects (0.578) remain the axis.
5. Change VQA is held on an unlicensed benchmark; the licensed replacement
   (Landsat-SCD) is a different task. Not touched here.

Change mask, land cover and fusion are within noise of their references;
the class-balanced land-cover arm was already tried and rejected
(2026-09-18). Not touched.

## Root-cause analysis

| Task | Bottleneck | Evidence | Category |
|---|---|---|---|
| NL routing | classifier blind to the input configuration; no parameter extraction; no conversation state; template bank had no implicit-change or open "analyse" shapes | 40-query probe then 180-query benchmark: every miss on pairs was an implicit change question → `SINGLE_VQA`; grounding prompt contained the whole sentence ("Locate the Find the airport in this image in the image…") | router / query parsing |
| Caption | (a) served path OK; (b) 1-epoch cosine schedule ended with val still rising (0.4183 → 0.4231 over the last 700 of 2,729 steps); (c) greedy decoding produces the generic caption for 40% of test images (`unique_fraction` 0.60); (d) reported val BLEU-4 0.42 (random 300) vs test 0.256 — unexplained gap, measured on the full val below | metrics.json / run_metadata.json of `caption_vlm`; `rsicd_test_vlm.json` | insufficient training + decoding; possibly val/test shift |
| Change caption | last 89 of 2,129 steps lost to a reboot (lr < 1e-6 by then, so negligible); images only, no explicit difference signal; 5-ref LEVIR-CC captions are short and templated | ledger 2026-09-14; `levircc_test_vlm.json` | mild undertraining; task formulation |
| Grounding | small objects (42% of test) 0.578; resolution was the last lever (arm E +3 pts); the 7B arm gained +0.7 pts n.s. | failure_analysis.md; `dior_rsvg_official_armE.json`; 7B log | model capacity at native resolution; diminishing returns |
| Grounding (served) | referring phrase = whole user sentence | code read; phrase-format experiment below | inference / query parsing |

## Experiments performed

Compute: the cluster account (`adi01@172.16.1.161`) reports **expired**
on 2026-09-20, so no training arm could be launched. Everything below is
inference-only on the laptop's RTX 4050 (6 GB) in the deployed NF4
precision, or CPU.

| # | Experiment | Split / n | Result | Artefact |
|---|---|---|---|---|
| E1 | NL routing v1 → v2 (config token + cue features + bank shapes + extraction + follow-ups) | dev 180 (tuned) / test 63 (single shot, then folded) / final 50 (single shot) | 71.7 → **99.4%**; 66.7 → **84.1%** single-shot (96.8% after folding); 72.0 → **90.0%** single-shot | `nl_understanding_v2_queries*.json` |
| E2 | Grounding phrase format: bare vs user sentence vs extracted | DIOR-RSVG test subsample, 150, seed 0, arm E, 1024² NF4 | see below | `grounding_phrase_format.json` |
| E3 | Caption decoding: greedy vs beam 3/5 vs no-repeat | RSICD val (selection) → test (report), `caption_vlm/adapter_best`, NF4 | see below | `rsicd_decoding_{val,test}.json` |

### E2 — grounding phrase format

`artifacts/benchmark_reports/grounding_phrase_format.json` — 150 official
DIOR-RSVG test expressions (seed 0), arm E adapter, NF4, 1024², one run:

| Condition (what the adapter is asked to locate) | Acc@0.5 [95% CI] | mIoU | McNemar vs bare |
|---|---|---|---|
| `bare` — the annotated expression | 0.807 [0.740, 0.867] | 0.704 | — |
| `sentence` — wrapped as a user types it, passed unchanged (**pre-fix served path**) | 0.800 [0.733, 0.860] | 0.700 | 3 lost / 2 gained, n.s. |
| `extracted` — the same sentence through `extract_object` + `referring_expression` (**served path now**) | 0.820 [0.760, 0.873] | 0.716 | 1 lost / 3 gained, n.s. |

Reading, stated plainly: **the whole-sentence prompt cost less than the
audit assumed** — 0.7 points on this subsample, inside the noise — because
the language model reads through the wrapper. The extractor does no harm
(+1.3 over bare, +2.0 over the sentence, both n.s.) and hands the adapter
the phrase form it was trained on, which is the reason to keep it; it is
not an accuracy lever. The first extractor version, which rebuilt the
phrase from a bare object and a normalised position, was caught by this
experiment's example dump before scoring (it dropped relational tails such
as "of the tennis court at the bottom") and replaced by the residual-based
one. The subsample scores above the full-test 0.732 because it is a
subsample; it is a comparison of formats, not a benchmark.

### E4 — why the captioner's val and test disagree (data, not model)

Measured on the RSICD parquet release (`data/rsicd/data`), CPU:

| | val (1,094) | test (1,093) |
|---|---|---|
| nearest train image, dHash-256 Hamming distance, 5th / 50th percentile | 84 / 94 | 85 / 94 |
| reference sentences that are verbatim train captions | **32.3%** | 11.3% |
| images with ≥ 1 reference seen in train | 55.4% | 35.1% |
| images with all 5 references seen in train | **18.4%** | 0.9% |
| distinct references / references | 59.5% | 84.8% |

The images are equally novel; the *captions* are not. RSICD's val
references repeat training sentences three times as often as the test's,
so a captioner is rewarded on val for reproducing training phrasing and
on test for describing the image. That is the 0.40–0.42 (val) vs 0.256
(test) gap, and it means "val still rising at the end of the schedule" is
partly memorisation, not learning — checkpoint selection on this val
picks the step that memorised most. LEVIR-CC does not have the asymmetry
(val 61% / test 60% verbatim; the unchanged pairs are templated in both).

Consequence for the queued caption arm: selection moves to
`val_unseen.jsonl` (`training/prepare/rsicd_val_unseen.py`: the val images
with no reference verbatim in train; ≈ 45% of val), reporting stays on
the official test. The decoding choice in E3 is likewise read on the
unseen subset of the val sample, not on the full sample.

### E3 — caption decoding

(filled in from `artifacts/benchmark_reports/rsicd_decoding_{val,test}.json`)

## Training changes

None trained (compute). Prepared and queued, with the hypothesis each
tests, in `scripts/cluster_unit_caption_v2.sh`:

* **C3** caption: deployed recipe × 3 epochs, checkpoint selected on the
  *full* official val. Hypothesis: undertraining (val still rising at the
  end of the 1-epoch schedule). Expected: +1–3 BLEU-4 if right; the
  official test decides.
* **CC2** change caption: continue `change_caption_vlm/adapter_best` one
  epoch at lr 3e-5, full-val selection. Hypothesis: mild undertraining.
* Deferred from before and unchanged: grounding arm F (1280²), seed-43
  repeat.

## NLP / query-understanding changes

`docs/research/nl_understanding.md` has the full design. In short:
`satquery/controller/understanding.py` (cues, extraction, follow-up
resolution, IR assembly), `satquery/contracts/understanding.py`
(`QueryUnderstanding`), classifier v2 in `intent.py`, router passes the
resolved query and the config token and fills `classes`, executor passes
`_query` (resolved), `_phrase`, `_image_index`, `_spatial_scope`,
`_classes`; tools read `_phrase` (grounder) and `selected_image`
(VQA, caption, grounder, land cover); API takes `history`; the query page
keeps the conversation and shows "Understood as …".

## Benchmark results

| Benchmark | Baseline | New | Δ abs | Δ rel | Config | Checkpoint | Data | Seed | Inference |
|---|---|---|---|---|---|---|---|---|---|
| NL routing, dev 180 | 0.717 | 0.994 | +0.277 | +38.7% | classifier v2 | fitted at start | `evaluation/nl/queries.jsonl` | 20260829 (bank), 0 (split) | CPU |
| NL routing, test 63 (single shot) | 0.667 | 0.841 | +0.174 | +26.1% | classifier v2 | — | `queries_test.jsonl` | — | CPU |
| NL routing, test 63 (after folding) | 0.667 | 0.968 | +0.301 | +45.1% | classifier v2 | — | — | — | CPU |
| NL routing, final 50 (single shot, never tuned on) | 0.720 | 0.900 | +0.180 | +25.0% | classifier v2 | — | `queries_final.jsonl` | — | CPU |
| Grounding, served prompt format (E2), DIOR-RSVG test subsample 150 | 0.800 (sentence) | 0.820 (extracted) | +0.020 | +2.5% (n.s.) | arm E, `min_pixels` 1048576 | `grounding_vlm_hires/adapter_best` | `data/dior_rsvg` test parquet | 0 | NF4, greedy, 4050 |
| Caption decoding (E3) | — | — | — | — | — | — | — | — | — |

(E2/E3 rows completed below when the runs finish.)

## Regression results

* Unit + integration suite: 1,580 passed, 7 skipped (`pytest tests
  --ignore=tests/test_soak.py`), including 55 new tests
  (`tests/test_understanding.py`, `tests/test_api.py::TestConversationHistory`).
* Golden traces: regenerated once. Behavioural diffs (all others differ
  only in classifier name/scores and the new `understanding` block):
  `adversarial_instruction_override` (`SINGLE_GROUND` → abstain),
  `adversarial_parameter_injection` (`SINGLE_VQA` → abstain),
  `adversarial_code_injection` (`SINGLE_VQA` → abstain),
  `adversarial_out_of_scope` ("weather forecast for this location":
  `SINGLE_VQA` → abstain), `adversarial_tool_coercion`
  (`SINGLE_LANDCOVER` → `SINGLE_CAPTION`), `crossmodal_fusion`
  (`classes` now `[built_up]` from the query, was the matrix default).
* RSVQA-LR protected number: untouched — the VQA adapter, processor and
  system prompt are unchanged; the VQA tool receives the resolved
  question (identical to the typed one for any standalone query).
* Deployment: `scripts/verify_deploy.py` on `deploy.v3.yaml` (GPU, local)
  and `deploy.cpu.yaml` — see the verification section of the PR.
* Frontend: `tsc --noEmit` clean, `next build` clean; the conversation
  thread and "Understood as" line verified in the browser against the
  local CPU API (follow-up "Is it near the water?" → "Is the road near the
  water?", `SINGLE_VQA`).

## Final selected checkpoints

Unchanged: every row of `configs/deploy.v3.yaml` keeps its checkpoint. The
selection that changed is *decoding* for the captioner if E3 justifies it
(chosen on val, confirmed on test) and the *served prompt* for the
grounder (E2).

## Deployment changes

* `satquery/tools/grounding.py`: `_phrase` (extracted referring
  expression) replaces the raw sentence; falls back with a warning.
* `satquery/tools/{rs_vqa,caption,grounding,landcover}.py`: `_image_index`;
  `landcover.py` answers the asked-for classes first (`_classes`,
  `REQUEST_CLASS_LABELS`).
* API: `history` form field on `/runs` and `/runs/stream`; trace field
  `routing.understanding`.
* Frontend: conversation thread, "Understood as" line; `Turn` /
  `Understanding` types.
* No new weights, no new environment variables, no change to
  `deploy.v3.yaml` / `deploy.cpu.yaml`.

## Remaining limitations

* The accuracy arms (C3, CC2, F, s43) are queued, not run: cluster access
  is gone. The report's benchmark table therefore carries the NL layer and
  the two inference experiments only.
* The NL benchmark is small (243 queries, one author). The test-split
  single-shot number (84.1%) is the number to quote for generalisation.
* Extraction is English, pattern-based; typo-heavy grounding queries
  ("were r the bildings") still route to VQA.
* `classes` from the query shapes the land-cover answer (asked-for
  classes first, in the head's labels) but does not change the head's
  decision threshold per class.
* Multi-turn resolution reads the latest completed turn only.

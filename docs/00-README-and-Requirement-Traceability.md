# SatQuery AI — Build Plan Index & Requirement Traceability

**PS 26167 · ISRO / Department of Space · SIH 2026 · Software · Space Technology**
**SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

Document 0 of 6 · Written 2026-08-27 · **Read this first.**

---

## 1. What this document set is

Six documents. A consolidated build plan produced by merging five independent design passes over PS 26167 and resolving their disagreements, then adding the decisions none of them made.

| Doc | Contents | Primary reader |
|---|---|---|
| **`00`** (this file) | Index, **PS requirement traceability matrix**, unverified-claims register, week-0 checklist, first-two-weeks action list | Everyone, day 1 |
| **`01-Solution-Architecture-and-System-Design.md`** | Six design axioms, five-layer architecture, ingest & compatibility gate, tool registry, verification & confidence, trace, dual-mode operation, tech stack, repo layout, non-functional targets | Backend, geo, frontend |
| **`02-Agentic-Workflow-and-Orchestration.md`** | Why a constrained planner beats free-form tool-calling, nine-task taxonomy, routing, the capability matrix, plan validation, executor, worked traces for all five PS representative queries, hard cases, testing | Backend lead |
| **`03-Models-and-Datasets.md`** | Compute reality and the three hard rules, two-track resolution-bridged adaptation, per-tool model recommendations with fallbacks, dataset plan, splits and leakage, evaluation plan, ablations, **week-0 verification gate** | ML leads |
| **`04-Implementation-Plan.md`** | Three organising principles, team allocation, four phases W0–W14 with exit criteria, descope ladder, 12-row risk register, 7-minute demo script, weekly cadence | Team lead, everyone |
| **`05-Innovation-and-Extra-Features.md`** | Extras beyond mandatory scope, tiered by marks-per-hour, with an explicit declined list and a "build only these five" summary | Team lead, M6 |

**Reading order for a new team member:** this file, then `04` §2 (your role) and `04` §3–7 (the phase you are in), then whichever of `01`–`03` covers your components.

---

## 2. The submission in five sentences

An interactive assistant that accepts single optical/SAR imagery, co-registered optical–SAR pairs, or bi-temporal pairs, plus a free-text question, and answers it by orchestrating a registry of small remote-sensing-adapted specialist models. Adaptation runs on **two tracks** — a band-agnostic multi-sensor encoder adapted on BigEarthNet at 10 m, and high-resolution object-level instruction tuning — bridged by band-presence masking, random band dropout and GSD-conditioned augmentation, because the mandated training data and the ISRO evaluation data are 10–20× apart in ground sample distance. Orchestration is a **constrained planner over a version-controlled capability matrix**, not free-form LLM tool-calling, giving a provable illegal-plan rate of zero — which matters because the PS grades the observable trace and explicitly does not grade internal reasoning. Every neural output is independently checked by a **physics verifier** built from classical remote-sensing indices and SAR backscatter statistics, with documented fallbacks for the fact that Cartosat-2S has no SWIR band and RISAT's frequency is unconfirmed. Confidence is calibrated and reported as three separate components, quantitative answers come from deterministic computation rather than generation, prose passes an entailment gate against structured evidence, and every answer ships a georeferenced evidence pack an analyst can open in QGIS.

---

## 3. PS requirement traceability matrix

Every clause of the PS mapped to a component, a document section, a test, a
metric and a **current status**. **This table is the contract.**

**Refreshed 2026-08-30.** The original table was written at plan time (W0) and
its Test and Metric columns described what *would* be measured. Every row now
carries what *has* been measured, with the artifact it came from. Nothing here
is marked satisfied without a number in the repository behind it.

### 3.0 How to read the status column

| Status | Means |
|---|---|
| **VERIFIED** | The PS clause is satisfied: component built, wired into the pipeline, covered by tests, and evidence in the repo. |
| **VERIFIED (weak)** | Satisfied, but the measured number is poor. Said so it cannot be mistaken for strength. |
| **VERIFIED (negative)** | Built and measured, and the measurement contradicts the design hypothesis it was built to test. Reported as-is. The PS clause is still satisfied; our claim about it is not. |
| **PARTIAL** | Satisfied in part, with the missing part named. |
| **MISSING** | Not built, or built and unmeasured. |
| **BLOCKED** | Cannot be satisfied from here — an external dependency, or an input the PS says will not be disclosed. |

Rows reviewed before 2026-08-30 used **MET** for **VERIFIED**; where both appear
they mean the same thing.

**Source of truth: [`docs/ps-26167.md`](ps-26167.md).** The authoritative PS
text is now in the repository, and this table was checked against it
clause-by-clause on 2026-08-30. That closes the former limitation L12.

**The check found four real defects in this table**, all now corrected: two of
the five "PS representative queries" were not the PS's and one PS query was
missing entirely (§3.3); the deliverables list claimed six PS deliverables
where the PS states two (§3.4); one clause of the PS's controller list had no
row at all (**M9**); and I6 was presented as a PS requirement when the PS says
nothing about scene size (§3.2). A traceability matrix built from paraphrase
drifts in exactly this way, which is the argument for keeping the PS in git.

### 3.1 Mandatory functional scope

| # | PS requirement (paraphrased) | Component | Test | Measured result | Status |
|---|---|---|---|---|---|
| **M1** | **Adapt at least one visual / VL component to remote sensing data.** *A generic LLM/VLM without RS adaptation is explicitly not acceptable.* | **Track A** band-agnostic encoder → `landcover_v1`, `optsar_fusion_v1`; **Track B** VLM QLoRA → `rs_vqa_v1`, `caption_v1` | `test_track_a.py` (24), `test_track_a_full.py` (15), `test_stage_a2.py` (17), `test_training.py` (43), `test_track_b_eval.py` (9) | Track A BigEarthNet **mAP 0.2854** all-bands / **0.2573** Cartosat-4-band (retention **0.9015**); Stage A2 WHU-OPT-SAR **mAP 0.7759**; Stage A3 frozen probe 0.1151 → finetuned **0.2880** (**gain +0.1729**); Track B `rsvqa_lr` exact **0.4510 → 0.6425** on an identical held-out split | **VERIFIED** |
| **M2** | **Single-image VQA — mandatory** | `rs_vqa_v1` (QLoRA adapter on Qwen2.5-VL-3B) | `test_vqa_tool.py` (14), `test_track_b_eval.py` (9) | `rsvqa_lr` exact match **0.6425** (n=207); full val exact **0.381**, token-F1 **0.7913** (n=534) | **VERIFIED** |
| **M3** | **At least one of** captioning **or** grounding | **Both built:** `caption_v1`, `grounding_v1` | `test_learned_tools.py` (22) | Caption BLEU-4 **0.2446** (n=1093) but only **146 unique captions / 13.4%** diversity; Grounding mIoU **0.1405**, Acc@0.5 **0.0762**, Acc@0.7 **0.0088** (n=1141) | **VERIFIED (weak)** — mandatory needs one; captioning is the stronger arm, grounding is near-floor |
| **M4** | **Bi-temporal change description or change-VQA — mandatory** | **Both built:** `change_caption_v1`, `change_vqa_v1` (deterministic index path + semantic-change path) | `test_change_caption.py` (13), `test_change_vqa_semantic.py` (12), `test_semantic_change.py` (39), `test_cdvqa_prepare.py` (18) | Change caption BLEU-4 **0.3063** on changed pairs (the aggregate 0.5686 is inflated by the trivially-unchanged half and is not the figure to quote); **CDVQA 0.5380** over the full prescribed split (39,686 questions, 968 pairs, **100% coverage**) against a per-type majority baseline of **0.5084**, ceiling **0.9975** | **VERIFIED** |
| **M5** | Change map — **optional bonus** | `change_mask_v1` | `test_learned_tools.py`, `test_fault_injection.py` (14) | LEVIR-CD **F1 0.5597**, IoU 0.3886, precision 0.4426, recall 0.7613 | **VERIFIED** (optional) |
| **M6** | **Cross-modal complementary extraction** from a co-registered optical + SAR pair | `optsar_fusion_v1`, triad mode + complementarity score | `test_learned_tools.py`, `test_ablations.py` (8) | WHU-OPT-SAR triad: optical **0.7778**, SAR 0.7410, fused 0.7714 → **complementarity gain −0.0064**. Fusion does **not** beat optical alone | **VERIFIED (negative)** — the capability is built, wired and measured; the hypothesis it was built to test is disconfirmed |
| **M7** | **Agentic layer**: interpret query, determine task, select tool from a **predefined registry**, execute with **permitted parameters** | Controller: config gate → Tier-1 classifier → capability matrix → validated plan → executor | `test_router.py` (29), `test_adversarial_routing.py` (15), `test_matrix_validate.py` (3), `test_controller_e2e.py` | **Illegal-plan rate 0 / 600 plans** across 3 configurations, 200 adversarial queries; unpermitted-parameter rate 0; `satquery matrix --validate` green in CI. Routing accuracy on the never-tuned CLEAN_HOLDOUT: **0.6552** (n=29) | **VERIFIED** for the graded property (illegal-plan rate); routing accuracy is **weak** and is the system's weakest measured component |
| **M9** | **Controller shall "combine textual and spatial outputs, estimate confidence, and return visual evidence"** *(added 2026-08-30: this clause of the PS's controller list had no row of its own)* | Narrative synthesiser + evidence pack + three-component confidence combiner + georeferenced COG overlays | `test_evidence_pack.py` (16), `test_confidence_weights.py` (15), `test_abstention.py` (25), `test_calibration.py` (28) | Every answer carries a confidence with three named components and a limiting-component explanation; evidence pack exports GeoJSON + COG + `evidence.json` as a ZIP that opens in QGIS; change-mask calibration ECE **0.0668 → 0.0034** | **VERIFIED** |
| **M8** | **Auditable execution summary**: selected task, tool names, key parameters. *Internal reasoning not required.* | Trace writer + SSE stream + trace panel; `rationale_tag` enums, no free-form reasoning | `test_golden_traces.py` (38 over **31 golden trace files**), `test_contracts.py` (6), `test_api.py` (18) | Every run carries task, tool names, versions, params, per-step confidence, runtime and warnings; Pydantic-validated schema; 31 golden traces byte-compared | **VERIFIED** |

### 3.2 Defined input scope

| # | PS requirement | Component | Test | Status |
|---|---|---|---|---|
| **I1** | Single optical/MSI **or** single SAR image | `config = SINGLE`; modality inferred from band count, dtype, histogram, local CoV and metadata — **never the filename** | `test_ingest.py` (41), `test_real_products.py` (24) | **VERIFIED** — exercised on real Cartosat-2E MX and EOS-04 FRS-1/MRS products |
| **I2** | Co-registered optical + SAR **pair** | `config = CROSSMODAL_PAIR`; **footprint overlap measured and gated** (`check_footprint_overlap`, `Router.unmet_requirements`) since 2026-08-30 | `test_ingest.py` (47, incl. `TestFootprintOverlap`), `test_real_products.py` | **VERIFIED for "same area"** — a disjoint pair now fails the check and cannot reach `XMODAL_JOINT_EXTRACT`. **Sub-pixel co-registration is still unverified**: the shift estimator reports ~38 px on an identically-footprinted pair, so `max_coreg_shift_px` is deliberately not gated — see **L16** |
| **I3** | **Bi-temporal** pair of the same area | `config = BITEMPORAL_PAIR`; **"same area" now gated by footprint overlap** at the matrix's 80% for the temporal tasks. Missing dates remain a **WARN** rather than an abstention — see **L16** | `test_ingest.py` (47), `test_abstention.py` (25) | **VERIFIED for "same area"**; date provenance is disclosed rather than enforced |
| **I4** | GeoTIFF / TIFF for geospatial imagery | `IngestMode.OPERATIONAL` | `test_ingest.py` | **VERIFIED** |
| **I5** | **PNG / JPEG only for prescribed public benchmarks** | `IngestMode.BENCHMARK` requires a named benchmark; PNG in operational mode is rejected with the rule quoted | `test_ingest.py` | **VERIFIED** — and load-bearing: before this landed (2026-08-29) **no prescribed benchmark image could enter the pipeline at all**. Verified end to end on CDVQA PNGs, where `crs_present` records **WARN**, not FAIL |
| **I6** | ~~Inputs may be full scenes far larger than any model window~~ **Not a PS clause.** The PS's input scope specifies image *configurations* and *formats*, and says nothing about scene size | Tile pyramid + query-conditioned coarse-to-fine retrieval | `test_tiling.py` (27) | **VERIFIED as built, but beyond PS scope.** Kept because a 7687×7640 Cartosat scene is real and the PS's evaluation set is Cartosat — but it must not be counted as a satisfied PS requirement |

### 3.3 Representative queries — **the PS's own five, verbatim**

**Corrected 2026-08-30 against `docs/ps-26167.md`.** The previous version of this
table listed a set that was *not* the PS's. Two entries were wrong and one PS
query was missing entirely:

* *"How many aircraft are visible in this image?"* was listed as a PS query. **It is not in the PS.** It nevertheless drove a real design decision — arithmetic counting over detections rather than generative counting — which remains sound, but it is our query, not theirs.
* *"Highlight the water body referred to in the query."* — a **grounding** query, and the PS's only one — **was absent from the table**.
* The change query was recorded as *"Describe the changes between these two images."*, dropping the PS's *"and where did the change occur?"* That clause is load-bearing: it asks for spatial localisation, not only description.

Routing measured 2026-08-30 by running the verbatim PS strings through the
Tier-1 classifier and, for query 3, the full controller.

| # | PS query (verbatim) | Routes to | top-1 | Status |
|---|---|---|---|---|
| 1 | *"Describe the land-cover and major objects visible in this image."* | `SINGLE_CAPTION` | 0.504 | **VERIFIED** — routes correctly; answer quality is caption BLEU-4 0.2446 |
| 2 | *"Highlight the water body referred to in the query."* | `SINGLE_GROUND` | 0.758 | **PARTIAL** — routes correctly, and this is the PS's only grounding query, but grounding is at Acc@0.5 **0.0762**. Routing is asserted; localisation quality is the open weakness |
| 3 | *"What changed between these two dates, and where did the change occur?"* | `TEMPORAL_CHANGE_DESC` | 0.643 | **VERIFIED** — fixed 2026-08-30, see below |
| 4 | *"Use the optical and SAR images together to identify built-up and water-covered regions."* | `XMODAL_JOINT_EXTRACT` | 0.973 | **VERIFIED** |
| 5 | *"Has the built-up area increased, decreased, or remained unchanged?"* | `TEMPORAL_CHANGE_VQA` | 0.699 | **VERIFIED** — three-way answer from a signed area difference against a stated significance threshold |

**Query 3 was the one to fix, and it is fixed (2026-08-30).** It previously
selected `TEMPORAL_CHANGE_MAP`, whose plan is `index_engine_v1 →
change_mask_v1` and whose answer is *"Produced a change mask; see the exported
raster artifact."* — **where** without **what**. `_CHANGE_MAP` owned every
"where" phrasing in the query bank, so the PS's "and where" pulled the whole
query there.

The fix is training data, not a special case: eight compound *"what changed
**and** where"* templates were added to `TEMPORAL_CHANGE_DESC`, whose plan is
`index_engine_v1 → change_mask_v1 → change_caption_v1` and which therefore
returns **both** the prose and the same georeferenced mask. **None of the eight
is the PS string** — the PS query is the acceptance test, and a template equal
to the test string would prove memorisation rather than generalisation. A plain
*"Show me where the changes occurred"* still routes to `TEMPORAL_CHANGE_MAP`
at 0.950, which is the distinction the fix turns on and is asserted in a test.

**What the fix cost, stated plainly.** Raw Tier-1 accuracy on the never-tuned
CLEAN_HOLDOUT fell **0.6552 → 0.5862** (n=29, three items flipped). All three
flipped items were **already below the confidence gate** (top1 0.252–0.296
against a 0.35 threshold) both before and after, which is the band where the
router ignores the classifier and falls back to the configuration default — so
**system behaviour on all three is unchanged**. The drop is real on the raw
metric and does not correspond to a behavioural regression. Illegal-plan rate
re-measured after the change: still **0 / 600**.

**All five PS queries are now golden traces** (`ps_q1`…`ps_q5`), and
`TestPSRepresentativeQueries` asserts each one *behaviourally* rather than
byte-wise — a golden pins whatever the system does, including doing the wrong
thing consistently, which is exactly how L13 survived three phases. The
query-3 assertion was verified to fail when the fix is reverted.

### 3.4 Deliverables

**Corrected 2026-08-30.** The PS lists exactly **two** deliverables. The
previous table listed six, treating documentation, evaluation results and
model cards as PS requirements. They are not. They may still be worth
producing — but they are **team-chosen**, and a matrix that cannot tell the
difference will mis-order Phase 4.

| PS deliverable (verbatim) | Status | Evidence |
|---|---|---|
| *"An interactive GUI or web application with an agentic remote-sensing AI backend."* | **VERIFIED** | Next.js frontend (typechecks, builds, 5 routes) + FastAPI/SSE backend + constrained-planner controller; three Docker images build and run; the containerised API answered a real Cartosat-2E query |
| *"Codes and models including test and demonstration."* | **PARTIAL** | **Code:** repo with CI, 825 tests, 31 golden traces, ADR 001 — VERIFIED. **Models:** eight trained checkpoints with `metrics.json` and `run_metadata.json` — present, but publication is licence-blocked (SECOND states no licence; SpaceNet 6 is share-alike). **Test:** VERIFIED. **Demonstration:** `scripts/make_demo_bundle.py` exists and **has never been run**; the 7-minute script in `04` §10 has never been rehearsed — **MISSING** |

**Not PS deliverables** (team-chosen; keep, but do not call them requirements):
technical report, model cards, slide deck, this document set. The PS asks for
a *demonstration*, which is the one Phase-4 item it does require.

### 3.4b Expected-solution checklist (PS "The solution should include")

| PS item (verbatim) | Status | Evidence |
|---|---|---|
| *"Input upload and compatibility checking."* | **VERIFIED** | Layer 0; `test_ingest.py` (41), `test_api.py` (18), `test_api_limits.py` (18) |
| *"A remote-sensing-adapted vision-language component."* | **VERIFIED** | Track B QLoRA adapter; `rsvqa_lr` 0.4510 → 0.6425 |
| *"Specialist tools for VQA, captioning or grounding, change understanding, and optical–SAR analysis."* | **VERIFIED** | All four families built and wired: `rs_vqa_v1`, `caption_v1`/`grounding_v1`, `change_caption_v1`/`change_vqa_v1`/`change_mask_v1`, `optsar_fusion_v1` |
| *"An agentic controller for task routing, tool execution, and output integration."* | **VERIFIED** | Illegal-plan rate 0/600 |
| *"Visual evidence, confidence information, execution summaries, and downloadable reports."* | **VERIFIED** | Georeferenced COG overlays + evidence-pack ZIP; three-component confidence; trace; PDF at `/runs/{id}/report.pdf` (`test_evidence_pack.py` 16, `test_report_pages.py` 18, `test_confidence_weights.py` 15) |

### 3.5 Evaluation conditions

| PS condition | Status | Evidence |
|---|---|---|
| **PS assigns each benchmark a role:** *"VRSBench and RSVQA will be used to evaluate single-image captioning, grounding, and visual question answering, while CDVQA will be used to evaluate multitemporal change-based visual question answering."* | **PARTIAL** | **RSVQA-LR: used** (official split, n=207, exact 0.6425). **CDVQA: used at 100% coverage** (39,686 questions / 968 pairs, 0.5380). **VRSBench: not evaluated** — annotations only, imagery lives in DOTA, not on disk. Note the PS assigns **captioning and grounding** to VRSBench/RSVQA, so the caption BLEU-4 0.2446 and grounding Acc@0.5 0.0762 in §3.1 are **not** on the prescribed splits |
| *"BigEarthNet.txt will serve as the primary dataset for adapting image–text representations"* (Background) vs *"using BigEarthNet.txt **or other open-source training data**"* (Mandatory Scope) | **PARTIAL** | Track A adapted on BigEarthNet **imagery + 19 labels**, not on BigEarthNet.txt, the image–text corpus. The **mandatory** clause permits this explicitly; the **Background** states an expectation we did not meet. Defensible, and a judge may ask — the substitution should be justified in writing, not discovered |
| **Private ISRO/SAC set**: pre-georeferenced, co-registered **Cartosat-2S + RISAT SAR** | **PARTIAL** | Real Cartosat-2E MX and EOS-04 FRS-1/MRS products on disk, held out and never trained on; ingest, tiling and the SWIR-free path exercised on them. **Which RISAT SAC will use is still unconfirmed** — see §3.6 |
| Reference answers, labels, boxes, masks | **VERIFIED** | All four annotation types implemented and emitted; `evaluation/schemas.py`, `test_evaluation.py` (29) |
| *"Evaluation annotations will not be disclosed to participating teams."* | **BLOCKED — by design, and this is the right kind of blocked** | Nothing can be tuned against the ISRO/SAC set. It makes the never-trained-on Bhoonidhi hold-out the closest available proxy, and it is the reason the cross-sensor generalisation result matters more than any in-distribution number |
| Scores normalised before combining | **VERIFIED** (as a design driver) | Drives breadth-over-depth and the descope ladder |
| Large-scale batch evaluation | **VERIFIED** | Headless batch runner; `run_batch` on every tool; `test_batch_and_metrics.py` (22) |

### 3.6 Known limitations, carried deliberately

Every item here is measured or reproduced, not suspected. A judge who finds one
of these should find it already written down.

| # | Limitation | Evidence | Consequence |
|---|---|---|---|
| **L1** | **An unresolved flaky test.** `test_swir_free_path_exercised_on_real_cartosat` failed once under the no-torch CI simulation on 2026-08-29 and again on 2026-08-30, and passed on every other run including immediately after. The failing run took 272 s against ~105 s typical, which *suggests* I/O contention during concurrent Docker builds — that is a hypothesis, not a diagnosis | Two observed failures; current state 700 passed / 18 skipped / 0 failed | A CI run could go red without a code change. Not diagnosed |
| **L2** | **`band_stats.json` is gitignored.** `checkpoints/` is excluded from git, so the land-cover head's normalisation statistics are not in the repository. A fresh clone cannot load `landcover_v1` until they are regenerated | `.gitignore:4`; `training/track_a_full.py` writes the file | Regenerate with `compute_stats(seed=0, sample=2000)` over the four BigEarthNet train shards. Verified 2026-08-30 as **deterministic** — the regenerated file is bit-identical to the base run's. **Requires `data/ben_full` (45 GB) to be present** |
| **L3** | **Task 3.1 refusal is a negative result.** Refusal recall **0.4118** decomposes into **5/5 (100%) on lexical refusals** and **2/12 (16.7%) on image-conditional ones**. The model learned to refuse when the *question* is impossible on its face, not when the *image* is the reason — which is the half that matters | `docs/assets/refusal/track_b_fullval.json`; false-refusal rate 0.0077, lexical-shortcut probe 0.1667 | The system can be induced to answer a question its imagery cannot support. Three candidate causes (refusal fraction, epoch count, learning rate) and the run separates none |
| **L4** | **Which RISAT the evaluation set uses is unconfirmed — and the PS says to keep it that way.** `ps-26167.md` states the PS "does not specify the exact RISAT mission/product" and that the project "should not assume a specific RISAT variant... unless independently confirmed", requiring the implementation to "remain **sensor-configurable**". **Reframed 2026-08-30:** the requirement is configurability, which adaptive rather than absolute σ⁰ thresholds satisfy — not identification | `docs/ps-26167.md` §"Authoritative Sensor Note"; `docs/verification.md` §"Which RISAT" | **The narrowing to EOS-04 must not become a baked-in assumption.** It stays as background for the Stage A3 decision only. If SAC confirms RISAT-2B/2BR1, verification item 8 inverts and Stage A3 should be redone against 0.25 m X-band SAR (~2–4 GPU-h + downloads) |
| **L5** | **Optical–SAR fusion does not beat optical alone.** Complementarity gain **−0.0064** | `checkpoints/optsar_fusion/metrics.json` | M6 is satisfied as a *capability*; the claim that fusion helps is disconfirmed on WHU-OPT-SAR and must not be asserted |
| **L6** | **The two-track ablation is not comparable.** The two tracks were trained and evaluated on different tasks, so no controlled comparison exists | `docs/assets/ablations/ablations.json`, status `not_comparable` | The two-track design decision is *reasoned*, not *demonstrated*. Do not claim it is proven |
| **L7** | **Tier-1 routing accuracy is 0.5862** on the never-tuned holdout (n=29), down from 0.6552 as the measured cost of the L13 fix. Every flipped item sits below the confidence gate, where the router already ignores the classifier, so no system behaviour changed | `satquery/synth/holdout.py`; measured 2026-08-30 | The weakest measured component, and n=29 makes it a smoke test rather than a benchmark. The config gate keeps a misroute from becoming an illegal plan, so it degrades rather than fails — re-verified at 0/600 |
| **L8** | **`landcover_v1` asserts on ~0.25% of decisions** at 91% precision; at threshold 0.5 the head is worse than always predicting negative (0.2064 vs 0.1834) | `configs/thresholds.yaml`, `docs/phase1-status.md` | Correct behaviour for a head with mAP 0.285, but thin for a demo. The narrative synthesiser carries land-cover answers |
| **L9** | **Tier-2 LLM tie-break is unbuilt.** `llm_tiebreak_invoked` is always `false` | Trace schema | An honest flag on an unbuilt feature; not one of the 14 Phase-3 tasks |
| **L10** | **CDVQA's remaining headroom is the segmenter.** 0.5380 achieved against a 0.9975 oracle ceiling — **93% of the gap is semantic-change segmentation** (change-class mIoU 0.2636) | `artifacts/cdvqa/`, `checkpoints/change_vqa/metrics.json` | The answer layer contributes no measurable error; further gains are one well-posed segmentation problem |
| **L11** | **VRSBench is not evaluated.** It ships annotations only; its imagery lives in DOTA, which is not on disk | `docs/verification.md` item 9 | One of three prescribed benchmarks has no number |
| ~~**L12**~~ | ~~The verbatim PS text is not in the repository.~~ **CLOSED 2026-08-30.** `docs/ps-26167.md` is now the in-repo source of truth and this matrix was checked against it clause-by-clause | `docs/ps-26167.md` | The check found four defects — see §3.0. That is the measured cost of having run on paraphrase for three phases |
| ~~**L13**~~ | ~~A "what changed and where" query answers only "where".~~ **CLOSED 2026-08-30** by adding compound "what and where" templates to `TEMPORAL_CHANGE_DESC`. Query 3 now runs `index_engine_v1 → change_mask_v1 → change_caption_v1` and returns both | §3.3; `TestPSRepresentativeQueries::test_q3_answers_both_what_and_where` | Cost: raw CLEAN_HOLDOUT accuracy 0.6552 → 0.5862, entirely inside the low-confidence band the router already ignores. Illegal-plan rate still 0/600 |
| **L16** | **PARTLY CLOSED 2026-08-30.** `min_overlap_pct` and `min_bands_optical` are now **enforced** — `check_footprint_overlap` measures the overlap and `Router.unmet_requirements` excludes any task whose declared threshold is unmet, with the reason surfaced. The disjoint optical+SAR pair now abstains instead of being fused. **Two gates remain deliberately unenforced, on evidence:** `max_coreg_shift_px`, because the cross-modal shift estimator reports **38.1 px on a pair with identical footprints, 100% overlap, the same CRS and the same GSD** — twenty times the matrix's 2.0 px limit, so gating on it would refuse well-formed pairs and the estimator's absolute accuracy across modalities is unvalidated; and `require_dates`, because enforcing it would refuse change analysis on every undated pair including the prescribed benchmarks (CDVQA ships undated PNGs), where the existing `temporal_order` WARN is the honest disclosure. Both stay declared in the matrix and open here. **Original finding:** the matrix's input gates were declared but never enforced. `RequiresSchema` in `satquery/controller/matrix_loader.py` declares only `config` with `extra: "allow"`, so `min_overlap_pct`, `max_coreg_shift_px`, `require_dates` and `min_bands_optical` are parsed into the model and **read by nothing**. `Router.legal_tasks` gates on `requires.config` alone. Measured 2026-08-30: an optical + SAR pair written **60 km apart** routes to `XMODAL_JOINT_EXTRACT`, answers, and raises no failing check | `configs/capability_matrix.yaml` lines 57/79/96; `satquery/controller/router.py:120`; demo bundle beat `incompatible_pair` | **The system will confidently fuse two images of different places** - the exact failure mode the project's own axioms are written against. It also means the demo's scripted opening beat ("footprint overlap 0.41, below the required 0.70") does not exist. I2 and I3 are downgraded to PARTIAL on this basis. The PNG-in-operational-mode rejection **does** work and is the honest opening beat until this is fixed |
| ~~**L14**~~ | ~~None of the PS's five representative queries is a golden trace.~~ **CLOSED 2026-08-30.** All five are now goldens **and** carry behavioural assertions; a test also checks the strings still appear verbatim in `docs/ps-26167.md`, so drift between the matrix and the PS becomes a test failure | `tests/test_golden_traces.py` (50 tests, 36 goldens) | The matrix is now self-verifying on this row |
| **L15** | **The demonstration is a PS deliverable and does not exist.** *"Codes and models including test and demonstration"* — `scripts/make_demo_bundle.py` has never been run and the 7-minute script has never been rehearsed | §3.4 | The only Phase-4 item the PS actually names. The technical report and model cards, which the plan treated as deliverables, are **not** PS requirements |

---

## 4. Unverified claims register

No network access was available while writing these documents. Most items below remain unconfirmed. **Update 2026-08-27:** the user supplied the BigEarthNet.txt paper (`2603.29630v2`) and the dataset's HuggingFace card directly, so **item 1 is now fully verified** — contents *and* licence (**CDLA-Permissive-1.0**, a permissive open-data licence; see the row and `03` §4.1). The remaining load-bearing unknowns are items 2 (Cartosat SWIR) and 3 (RISAT band), both needing a real Bhoonidhi product.

| # | Claim | Status | Consequence if false |
|---|---|---|---|
| 1 | BigEarthNet.txt at `txt.bigearth.net`: 464,044 co-registered S1/S2 pairs, 9.6 M annotations incl. captions + VQA (binary + MCQ) + referring expressions; benchmark subset 1,082 pairs / 15,029 annotations; arXiv 2603.29630 | **VERIFIED 2026-08-27 — paper + HF card; licence = CDLA-Permissive-1.0** | Fully resolved; licence permits use. Download = 467 MB Parquet (9,553,962 rows = 1 per annotation, **text only**); S1/S2 imagery is a separate reBEN pull (large — size storage around it). Format-only fallback if that pull is impractical: BigEarthNet v2 + GeoChat-Instruct + VRSBench; Track A unaffected |
| 2 | Cartosat-2S MX is 4-band VNIR with **no SWIR** → MNDWI and NDBI unavailable | **RESOLVED 2026-08-29 — the assumption holds.** Read from a real product's `BAND_META.txt`: `NoOfBands=4`, `BandNumbers=1234`, `PixelSpacing=1.6 m`, `SatID=CARTOSAT-2E` | The SWIR-free fallback paths are the operative ones, not a contingency |
| 3 | The RISAT in the ISRO set: RISAT-1 (C-band) vs RISAT-2B/2BR1 (X-band) | **PARTLY RESOLVED — measured for EOS-04, narrowed for the set.** A real EOS-04 product reads `radarCenterFrequency = 5.40 GHz` (C-band, within 0.09% of Sentinel-1). Which RISAT *SAC* will use is narrowed by elimination but unconfirmed — see limitation **L4** | **Still load-bearing.** Mitigated by adaptive rather than absolute σ⁰ thresholds |
| 4 | CROMA / DOFA checkpoints downloadable under a permissive licence | Unverified | Fall back to torchgeo SSL weights |
| 5 | Change-Agent / LEVIR-MCI weights available | Unverified | TinyCD + separate caption head, ~1 extra GPU-h |
| 6 | Qwen2.5-VL-3B is still the best ≤4B VLM in Aug 2026 | **Open — knowledge ends May 2025.** Note: the BigEarthNet.txt paper (early 2026) built its RS baseline on **InternVL3-1B** — evaluate it as an alternate | Substitute per the six criteria in `03` §2.1; drop-in by design |
| 7 | SpaceNet 6 / Umbra / Capella high-res SAR accessible and licensed | **RESOLVED 2026-08-29 to a "no" that is more useful than a yes.** All three are accessible and permissively licensed, but all three are **X-band** (9.69 GHz measured from a real Umbra product) against EOS-04's C-band — a 1.79× wavelength ratio | Stage A3 ran optical-only **on evidence, not on unavailability**. Inverts under L4 |
| 8 | GeoChat-7B / RS-LLaVA / LHRS-Bot downloadable for zero-shot baselines | Unverified | Baseline against the un-finetuned base VLM only |
| 9 | SIH 2026 calendar: internal deadline, finale dates, submission format | **STILL UNKNOWN — the oldest open item, and the one that decides what Phase 4 can contain.** Needs the team or the organisers | Compress the `04` phase plan proportionally, cutting from the top of the descope ladder |
| 10 | Published benchmark numbers cited anywhere in these docs | Unverified | Treat every number as directional until the harness produces your own |

**Rule for the team: no GPU-hour is spent on any path whose enabling claim is still unverified.** The full 12-item gate with owners and deadlines is in `03` §6, and `docs/verification.md` carries the resolutions — **7 of 12 resolved** as of 2026-08-30.

**Update 2026-08-30.** Items 2, 3 (partly) and 7 above are resolved from primary
evidence: real Bhoonidhi product metadata and a real Umbra STAC record, not from
reading a web page. Item 9 remains open and is now the binding constraint on
Phase 4 — every other open row has a costed fallback, and this one does not.

---

## 5. The first two weeks, as a checklist

Copy this into your tracker. Nothing here needs a GPU except item 11.

**Week 0**

- [ ] Run the 12-item verification gate (`03` §6); write every answer into `docs/verification.md`
- [ ] Register for **ISRO Bhoonidhi**; download real Cartosat-2S MX + PAN and RISAT products
- [ ] **Open a real Cartosat product and read its band list from the metadata** — resolves register item 2
- [ ] **Determine which RISAT and which mode** — resolves register item 3
- [ ] Confirm the SIH 2026 timeline and submission format — resolves register item 9
- [ ] Repo scaffold, `docker-compose.yml`, `Makefile`, CI, pre-commit
- [ ] **Freeze the Pydantic contracts** (`InputManifest`, `Plan`, `ToolResult`, `Trace`); tag `contracts-v1`
- [ ] Write ADR 001: LangGraph vs hand-rolled executor. **Decide once.**
- [ ] Inventory GPU accounts and quotas; start the weekly hour-tracking sheet
- [ ] Assign the six roles (`04` §2)

**Week 1**

- [ ] **Stub all nine tools** with schema-valid fake data; full pipeline runs end to end on stubs
- [ ] `capability_matrix.yaml` v1 with all nine tasks; `satquery matrix --validate` wired into CI
- [ ] `scripts/fetch_datasets.py` — download and mirror every P0 dataset to shared storage
- [ ] `scripts/fetch_models.py` — download and **sha256-verify** base checkpoints; `HF_HUB_OFFLINE=1` smoke test
- [ ] **Prove a T4 QLoRA run starts, checkpoints, and resumes after a deliberate kill** — do not skip this
- [ ] Synthetic query bank: ~60 templates per task, expanded to 3–5 k paraphrases
- [ ] Frontend shell running against stubs: upload, OpenLayers viewer, trace panel

**Gate to leave Phase 0:** stubbed pipeline produces a valid trace end to end; every verification item answered in writing; a training run has been resumed successfully.

---

## 6. The six decisions that define this submission

If you remember nothing else from these documents, remember these. Each is a deliberate choice against a plausible alternative, and each is defensible out loud.

1. **Two adaptation tracks, bridged.** Because BigEarthNet is 10 m and Cartosat-2S is ~1.6 m, and no single-track model spans that gap. The two-track ablation is the empirical proof, not a claim.

2. **A constrained planner, not a free-form LLM agent.** Because the PS grades the observable trace and explicitly does not grade reasoning, so determinism plus a provable illegal-plan rate of zero strictly dominates. The LLM is retained as a tie-breaker inside an already-legal set.

3. **Physics verifies neural, not the reverse.** Classical indices and SAR statistics are the independent referee — including a SAR-primary built-up path because **Cartosat-2S has no SWIR**, and adaptive rather than absolute σ⁰ thresholds because **RISAT's frequency is unconfirmed**. This is the trustworthiness story, and it costs zero GPU.

4. **Deterministic computation for every number; generation only for prose, and gated.** Counts come from detections, areas from the affine transform, increase/decrease from a signed subtraction against a stated significance threshold. Prose is generated once, from structured facts, and each sentence must be entailed by the evidence.

5. **Contracts in week 1 so four non-ML members never wait on a GPU.** Nine stub tools returning schema-valid data means the API, the executor, the matrix, the frontend, the trace, the evidence pack, the eval CLI and the golden tests are all built in parallel from week 2. This is the decision that plays to the team's real strength and it is the one that prevents the standard SIH failure.

6. **Breadth before depth, always.** Five mandatory areas, normalised scores. Every mandatory capability produces a real answer by W9 before anything is optimised. One zero costs more than any amount of polish gains.

---

*End of index. Begin with `01-Solution-Architecture-and-System-Design.md`.*

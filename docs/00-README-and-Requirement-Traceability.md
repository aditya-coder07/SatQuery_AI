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
| **MET** | Component built, wired into the pipeline, covered by tests, and a metric exists in the repo. |
| **MET (weak)** | Same, but the measured number is poor. Stated so it cannot be mistaken for strength. |
| **MET (negative)** | Built and measured, and the measurement contradicts the design hypothesis. Reported as-is. |
| **PARTIAL** | Some named sub-part is missing, and the missing part is named. |
| **NOT MET** | Not built, or built and unmeasured. |

**An honesty note about this table's source.** The verbatim PS 26167 text is
**not in this repository**. Every "PS requirement" below is a paraphrase made
at plan time from the PS. The paraphrases have been stable across six
documents and five design passes, but a clause could have been mis-transcribed
at W0 and nothing here would catch it. **Before submission, re-read this table
against the official PS text.** That is an evidence gap, not a satisfied row.

### 3.1 Mandatory functional scope

| # | PS requirement (paraphrased) | Component | Test | Measured result | Status |
|---|---|---|---|---|---|
| **M1** | **Adapt at least one visual / VL component to remote sensing data.** *A generic LLM/VLM without RS adaptation is explicitly not acceptable.* | **Track A** band-agnostic encoder → `landcover_v1`, `optsar_fusion_v1`; **Track B** VLM QLoRA → `rs_vqa_v1`, `caption_v1` | `test_track_a.py` (24), `test_track_a_full.py` (15), `test_stage_a2.py` (17), `test_training.py` (43), `test_track_b_eval.py` (9) | Track A BigEarthNet **mAP 0.2854** all-bands / **0.2573** Cartosat-4-band (retention **0.9015**); Stage A2 WHU-OPT-SAR **mAP 0.7759**; Stage A3 frozen probe 0.1151 → finetuned **0.2880** (**gain +0.1729**); Track B `rsvqa_lr` exact **0.4510 → 0.6425** on an identical held-out split | **MET** |
| **M2** | **Single-image VQA — mandatory** | `rs_vqa_v1` (QLoRA adapter on Qwen2.5-VL-3B) | `test_vqa_tool.py` (14), `test_track_b_eval.py` (9) | `rsvqa_lr` exact match **0.6425** (n=207); full val exact **0.381**, token-F1 **0.7913** (n=534) | **MET** |
| **M3** | **At least one of** captioning **or** grounding | **Both built:** `caption_v1`, `grounding_v1` | `test_learned_tools.py` (22) | Caption BLEU-4 **0.2446** (n=1093) but only **146 unique captions / 13.4%** diversity; Grounding mIoU **0.1405**, Acc@0.5 **0.0762**, Acc@0.7 **0.0088** (n=1141) | **MET (weak)** — mandatory needs one; captioning is the stronger arm, grounding is near-floor |
| **M4** | **Bi-temporal change description or change-VQA — mandatory** | **Both built:** `change_caption_v1`, `change_vqa_v1` (deterministic index path + semantic-change path) | `test_change_caption.py` (13), `test_change_vqa_semantic.py` (12), `test_semantic_change.py` (39), `test_cdvqa_prepare.py` (18) | Change caption BLEU-4 **0.3063** on changed pairs (the aggregate 0.5686 is inflated by the trivially-unchanged half and is not the figure to quote); **CDVQA 0.5380** over the full prescribed split (39,686 questions, 968 pairs, **100% coverage**) against a per-type majority baseline of **0.5084**, ceiling **0.9975** | **MET** |
| **M5** | Change map — **optional bonus** | `change_mask_v1` | `test_learned_tools.py`, `test_fault_injection.py` (14) | LEVIR-CD **F1 0.5597**, IoU 0.3886, precision 0.4426, recall 0.7613 | **MET** (optional) |
| **M6** | **Cross-modal complementary extraction** from a co-registered optical + SAR pair | `optsar_fusion_v1`, triad mode + complementarity score | `test_learned_tools.py`, `test_ablations.py` (8) | WHU-OPT-SAR triad: optical **0.7778**, SAR 0.7410, fused 0.7714 → **complementarity gain −0.0064**. Fusion does **not** beat optical alone | **MET (negative)** — the capability is built, wired and measured; the hypothesis it was built to test is disconfirmed |
| **M7** | **Agentic layer**: interpret query, determine task, select tool from a **predefined registry**, execute with **permitted parameters** | Controller: config gate → Tier-1 classifier → capability matrix → validated plan → executor | `test_router.py` (29), `test_adversarial_routing.py` (15), `test_matrix_validate.py` (3), `test_controller_e2e.py` | **Illegal-plan rate 0 / 600 plans** across 3 configurations, 200 adversarial queries; unpermitted-parameter rate 0; `satquery matrix --validate` green in CI. Routing accuracy on the never-tuned CLEAN_HOLDOUT: **0.6552** (n=29) | **MET** for the graded property (illegal-plan rate); routing accuracy is **weak** and is the system's weakest measured component |
| **M8** | **Auditable execution summary**: selected task, tool names, key parameters. *Internal reasoning not required.* | Trace writer + SSE stream + trace panel; `rationale_tag` enums, no free-form reasoning | `test_golden_traces.py` (38 over **31 golden trace files**), `test_contracts.py` (6), `test_api.py` (18) | Every run carries task, tool names, versions, params, per-step confidence, runtime and warnings; Pydantic-validated schema; 31 golden traces byte-compared | **MET** |

### 3.2 Defined input scope

| # | PS requirement | Component | Test | Status |
|---|---|---|---|---|
| **I1** | Single optical/MSI **or** single SAR image | `config = SINGLE`; modality inferred from band count, dtype, histogram, local CoV and metadata — **never the filename** | `test_ingest.py` (41), `test_real_products.py` (24) | **MET** — exercised on real Cartosat-2E MX and EOS-04 FRS-1/MRS products |
| **I2** | Co-registered optical + SAR **pair** | `config = CROSSMODAL_PAIR`; overlap ≥ 70%, gradient-domain phase correlation (raw-intensity correlation fails across modalities) | `test_ingest.py`, `test_real_products.py` | **MET** |
| **I3** | **Bi-temporal** pair of the same area | `config = BITEMPORAL_PAIR`; dates from metadata, **abstain rather than guess** when absent | `test_ingest.py`, `test_abstention.py` (25) | **MET** |
| **I4** | GeoTIFF / TIFF for geospatial imagery | `IngestMode.OPERATIONAL` | `test_ingest.py` | **MET** |
| **I5** | **PNG / JPEG only for prescribed public benchmarks** | `IngestMode.BENCHMARK` requires a named benchmark; PNG in operational mode is rejected with the rule quoted | `test_ingest.py` | **MET** — and load-bearing: before this landed (2026-08-29) **no prescribed benchmark image could enter the pipeline at all**. Verified end to end on CDVQA PNGs, where `crs_present` records **WARN**, not FAIL |
| **I6** | Inputs may be full scenes far larger than any model window | Tile pyramid + query-conditioned coarse-to-fine retrieval | `test_tiling.py` (27) | **MET** — exercised on the real 7687×7640 Cartosat scene |

### 3.3 Representative queries (the PS's own five — acceptance tests)

All five route correctly and are covered by golden traces. Routing is verified;
answer *quality* is the per-capability metric in §3.1.

| Query | Route | Status |
|---|---|---|
| *"How many aircraft are visible in this image?"* | `SINGLE_VQA` (counting) | Routes; count is arithmetic over detections, never generated. Bounded by grounding's weak Acc@0.5 (0.0762) |
| *"Describe the land-cover characteristics of this scene."* | `SINGLE_CAPTION` | Routes; narrative grounded in index statistics, entailment-gated |
| *"Use the optical and SAR images together to identify built-up and water regions."* | `XMODAL_JOINT_EXTRACT` | Routes; SAR-primary built-up path fires when NDBI is unavailable, and says so in the trace |
| *"Describe the changes between these two images."* | `TEMPORAL_CHANGE_DESC` | Routes; mask-conditioned caption, BLEU-4 0.3063 on changed pairs |
| *"Has the built-up area increased, decreased, or remained unchanged?"* | `TEMPORAL_CHANGE_VQA` | Routes; georeferenced area difference against a stated significance threshold |

### 3.4 Deliverables

| PS deliverable | Status | Evidence |
|---|---|---|
| Working interactive system | **MET** | Next.js + FastAPI + SSE; frontend typechecks and builds; three Docker images build and run; the containerised API answered a real Cartosat query |
| **Input upload and compatibility checking** | **MET** | Layer 0 complete; an explicitly named PS deliverable |
| Models | **PARTIAL** | Eight trained checkpoints with `metrics.json` and `run_metadata.json`. **Model cards not written.** Publication is additionally blocked by licence: SECOND states **no licence at all**, SpaceNet 6 is share-alike |
| Code | **MET** | Repo with CI, 825 tests, golden traces, ADR 001 |
| Documentation | **PARTIAL** | This document set + `phase1-status.md` + `verification.md`. **Technical report not written.** |
| Evaluation results | **MET** | `evaluation/harness.py` → one JSON report; artifacts under `docs/assets/` for calibration, abstention, adversarial, entailment, ablations, soak, refusal |

### 3.5 Evaluation conditions

| PS condition | Status | Evidence |
|---|---|---|
| Prescribed benchmark subsets (VRSBench, RSVQA, CDVQA) | **PARTIAL** | **RSVQA-LR: used** (official split, n=207). **CDVQA: used at 100% coverage** (39,686 questions / 968 pairs). **VRSBench: not evaluated** — it ships annotations only and its imagery lives in DOTA, which is not on disk |
| **Private ISRO/SAC set**: pre-georeferenced, co-registered **Cartosat-2S + RISAT SAR** | **PARTIAL** | Real Cartosat-2E MX and EOS-04 FRS-1/MRS products on disk, held out and never trained on; ingest, tiling and the SWIR-free path exercised on them. **Which RISAT SAC will use is still unconfirmed** — see §3.6 |
| Reference answers, labels, boxes, masks | **MET** | All four annotation types implemented and emitted; `evaluation/schemas.py`, `test_evaluation.py` (29) |
| Scores normalised before combining | **MET** (as a design driver) | Drives breadth-over-depth and the descope ladder |
| Large-scale batch evaluation | **MET** | Headless batch runner; `run_batch` on every tool; `test_batch_and_metrics.py` (22) |

### 3.6 Known limitations, carried deliberately

Every item here is measured or reproduced, not suspected. A judge who finds one
of these should find it already written down.

| # | Limitation | Evidence | Consequence |
|---|---|---|---|
| **L1** | **An unresolved flaky test.** `test_swir_free_path_exercised_on_real_cartosat` failed once under the no-torch CI simulation on 2026-08-29 and again on 2026-08-30, and passed on every other run including immediately after. The failing run took 272 s against ~105 s typical, which *suggests* I/O contention during concurrent Docker builds — that is a hypothesis, not a diagnosis | Two observed failures; current state 700 passed / 18 skipped / 0 failed | A CI run could go red without a code change. Not diagnosed |
| **L2** | **`band_stats.json` is gitignored.** `checkpoints/` is excluded from git, so the land-cover head's normalisation statistics are not in the repository. A fresh clone cannot load `landcover_v1` until they are regenerated | `.gitignore:4`; `training/track_a_full.py` writes the file | Regenerate with `compute_stats(seed=0, sample=2000)` over the four BigEarthNet train shards. Verified 2026-08-30 as **deterministic** — the regenerated file is bit-identical to the base run's. **Requires `data/ben_full` (45 GB) to be present** |
| **L3** | **Task 3.1 refusal is a negative result.** Refusal recall **0.4118** decomposes into **5/5 (100%) on lexical refusals** and **2/12 (16.7%) on image-conditional ones**. The model learned to refuse when the *question* is impossible on its face, not when the *image* is the reason — which is the half that matters | `docs/assets/refusal/track_b_fullval.json`; false-refusal rate 0.0077, lexical-shortcut probe 0.1667 | The system can be induced to answer a question its imagery cannot support. Three candidate causes (refusal fraction, epoch count, learning rate) and the run separates none |
| **L4** | **Which RISAT the evaluation set uses is unconfirmed.** Narrowed by elimination — RISAT-1 decommissioned 2017, RISAT-1B/EOS-09 failed at launch 2025, RISAT-2B/2BR1 are X-band but "data not ordinarily available to the public" and absent from Bhoonidhi's civil catalogue, leaving EOS-04 as the only openly-served candidate | `docs/verification.md` §"Which RISAT" | If it is RISAT-2B/2BR1, verification item 8 **inverts**: Umbra/Capella/SpaceNet 6 become the right Stage A3 sources and that stage should be redone against 0.25 m X-band SAR (~2–4 GPU-h + downloads) |
| **L5** | **Optical–SAR fusion does not beat optical alone.** Complementarity gain **−0.0064** | `checkpoints/optsar_fusion/metrics.json` | M6 is satisfied as a *capability*; the claim that fusion helps is disconfirmed on WHU-OPT-SAR and must not be asserted |
| **L6** | **The two-track ablation is not comparable.** The two tracks were trained and evaluated on different tasks, so no controlled comparison exists | `docs/assets/ablations/ablations.json`, status `not_comparable` | The two-track design decision is *reasoned*, not *demonstrated*. Do not claim it is proven |
| **L7** | **Tier-1 routing accuracy is 0.6552** on the never-tuned holdout (n=29) | `satquery/synth/holdout.py`; measured 2026-08-30 | The weakest measured component. The config gate keeps a misroute from becoming an illegal plan, so it degrades rather than fails |
| **L8** | **`landcover_v1` asserts on ~0.25% of decisions** at 91% precision; at threshold 0.5 the head is worse than always predicting negative (0.2064 vs 0.1834) | `configs/thresholds.yaml`, `docs/phase1-status.md` | Correct behaviour for a head with mAP 0.285, but thin for a demo. The narrative synthesiser carries land-cover answers |
| **L9** | **Tier-2 LLM tie-break is unbuilt.** `llm_tiebreak_invoked` is always `false` | Trace schema | An honest flag on an unbuilt feature; not one of the 14 Phase-3 tasks |
| **L10** | **CDVQA's remaining headroom is the segmenter.** 0.5380 achieved against a 0.9975 oracle ceiling — **93% of the gap is semantic-change segmentation** (change-class mIoU 0.2636) | `artifacts/cdvqa/`, `checkpoints/change_vqa/metrics.json` | The answer layer contributes no measurable error; further gains are one well-posed segmentation problem |
| **L11** | **VRSBench is not evaluated.** It ships annotations only; its imagery lives in DOTA, which is not on disk | `docs/verification.md` item 9 | One of three prescribed benchmarks has no number |
| **L12** | **The verbatim PS text is not in the repository.** Every requirement above is a plan-time paraphrase | This document | A mis-transcribed clause at W0 would be invisible. Re-read against the official PS before submission |

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

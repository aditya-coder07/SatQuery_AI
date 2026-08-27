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

Every clause of the PS mapped to a component, a document section, a test and a metric. **This table is the contract.** Anything in it without a green metric by W9 is a gap, and under normalised score combination a gap costs more than any amount of depth elsewhere gains.

### 3.1 Mandatory functional scope

| # | PS requirement (paraphrased) | Component | Spec | Test | Metric |
|---|---|---|---|---|---|
| **M1** | **Adapt at least one visual / vision-language component to remote sensing data.** Using BigEarthNet.txt or any open-source training data. *A generic LLM/VLM without RS adaptation is explicitly not acceptable.* | **Track A** band-agnostic encoder (BigEarthNet) → `landcover_v1`, `optsar_fusion_v1`; **Track B** VLM QLoRA on RS instruction mix → `rs_vqa_v1`, `caption_v1` | `03` §2 | `training/track_a_encoder.py`, `training/track_b_vlm_qlora.py`; two-track ablation | BigEarthNet test mAP; VQA/caption gains over the un-adapted base VLM |
| **M2** | **Single-image VQA — mandatory** | `rs_vqa_v1` | `01` §4, `03` §3 | RSVQA-LR/HR test, VRSBench VQA test | Accuracy overall + per question type |
| **M3** | **At least one of** captioning **or** grounding / referring localisation | **Both built:** `caption_v1` and `grounding_v1` | `01` §4, `03` §3 | VRSBench caption test; DIOR-RSVG + VRSBench referring | BLEU-4/METEOR/ROUGE-L/CIDEr; Acc@0.5, Acc@0.7, mIoU |
| **M4** | **Bi-temporal change description or change-VQA — mandatory** | **Both built:** `change_caption_v1` and `change_vqa_v1` (with a deterministic template path that cannot score zero) | `01` §4, `02` §7 Q4/Q5, `03` §3 | LEVIR-CC test; CDVQA test | BLEU-4/CIDEr; CDVQA accuracy per question type |
| **M5** | Change map — **optional bonus**, where mask annotations exist | `change_mask_v1` | `01` §4 | LEVIR-CD, WHU-CD test | F1, IoU, precision, recall |
| **M6** | **Cross-modal complementary information extraction** from a co-registered optical/MSI + SAR pair | `optsar_fusion_v1` in **triad mode** (optical-only / SAR-only / fused) + **complementarity score** | `01` §5.2, `05` §1.1 | WHU-OPT-SAR test; triad ablation | mIoU per class; **gain over best single modality**; modality agreement IoU |
| **M7** | **Agentic layer**: interpret the query, determine the required task, select the model/tool from a **predefined registry**, execute with **permitted parameters** | Controller: config gate → two-tier intent classifier → capability matrix → validated plan → executor | `02` entire | 200-query adversarial suite; 30 golden trace tests; `satquery matrix --validate` in CI | Routing accuracy; **illegal-plan rate = 0**; unpermitted-parameter rate = 0 |
| **M8** | **Auditable execution summary** with selected task, model/tool names, key parameters. *Internal reasoning traces neither required nor evaluated.* | Trace writer + SSE stream + live trace panel; `rationale_tag` enums instead of free-form reasoning | `01` §6, `02` §10 | Golden trace tests; Pydantic-validated trace schema | 100 % of runs carry every required field |

### 3.2 Defined input scope

| # | PS requirement | Component | Spec | Test |
|---|---|---|---|---|
| **I1** | Single optical/MSI **or** single SAR image | `config = SINGLE`; adaptive modality inference from band count, dtype, histogram, local CoV and metadata — **never from the filename** | `01` §2.2 | 20 real files incl. Cartosat MX/PAN and RISAT |
| **I2** | Co-registered optical/MSI + SAR **pair** | `config = CROSSMODAL_PAIR`; overlap ≥ 70 %, co-registration residual check via **gradient-domain** phase correlation (raw-intensity correlation fails across modalities) | `01` §2.4–2.5 | Pair compatibility suite |
| **I3** | **Bi-temporal** pair of the same area | `config = BITEMPORAL_PAIR`; dates parsed from metadata, **abstain rather than guess** when absent | `01` §2.4 | Missing-date and same-date cases |
| **I4** | GeoTIFF / TIFF for geospatial imagery | `IngestMode.OPERATIONAL` | `01` §2.1 | Format gate tests |
| **I5** | **PNG / JPEG accepted *only* for the prescribed public benchmark datasets** | `IngestMode.BENCHMARK` requires a named benchmark; PNG in operational mode is **rejected with the rule quoted** | `01` §2.1 | Rejection test — and it is the demo's opening move |
| **I6** | Inputs may be full scenes far larger than any model window | Tile pyramid + query-conditioned coarse-to-fine retrieval, bounded cost independent of scene size | `01` §2.7, `05` §2.4 | 8000×8000 scene answered in bounded time |

### 3.3 Representative queries (the PS's own five — these are acceptance tests)

| Query | Route | Mechanism | Spec |
|---|---|---|---|
| *"How many aircraft are visible in this image?"* | `SINGLE_VQA` (counting) | `grounding_v1` detect → NMS → **arithmetic count** → template. **Never a generative count.** | `02` §7 Q1 |
| *"Describe the land-cover characteristics of this scene."* | `SINGLE_CAPTION` | `index_engine_v1` statistics → `landcover_v1` → `caption_v1` narrative from structured fractions → entailment gate | `02` §7 Q2 |
| *"Use the optical and SAR images together to identify built-up and water-covered regions."* | `XMODAL_JOINT_EXTRACT` | index engine (NDWI + GLCM + adaptive σ⁰) → fusion triad → narrative → verifier. **NDBI unavailable on 4-band Cartosat → SAR-primary built-up path, logged.** | `02` §7 Q3, `01` Axiom 2 |
| *"Describe the changes between these two images."* | `TEMPORAL_CHANGE_DESC` | per-date indices → `change_mask_v1` → mask-conditioned `change_caption_v1` → entailment gate → GeoTIFF evidence | `02` §7 Q4 |
| *"Has the built-up area increased, decreased, or remained unchanged?"* | `TEMPORAL_CHANGE_VQA` | change mask → **georeferenced area difference in hectares** → significance threshold → three-way template answer | `02` §7 Q5 |

### 3.4 Deliverables

| PS deliverable | Where it comes from |
|---|---|
| Working interactive system | Next.js + FastAPI + SSE; `make demo` |
| **Input upload and compatibility checking** | Layer 0 in full — `01` §2, an explicitly named PS deliverable, and the demo's opening beat |
| Models | Published weights + model cards for all four trainings; `04` §4.5 |
| Code | `satquery-ai` repo per `01` §9, with CI, golden tests and ADRs |
| Documentation | This document set + technical report + this traceability matrix |
| Evaluation results | `evaluation/harness.py` → one JSON report covering every row of `03` §5, plus four ablations |

### 3.5 Evaluation conditions

| PS condition | How the plan handles it |
|---|---|
| Prescribed **public benchmark test subsets** (VRSBench, RSVQA, CDVQA) | Official splits used throughout; `--eval-mode` emits predictions in reference format for all four annotation types |
| **Private ISRO/SAC set**: pre-georeferenced, co-registered **Cartosat-2S optical + RISAT SAR** | The two-track resolution bridge exists for exactly this; Bhoonidhi products held out as an untrained cross-sensor check; adaptive σ⁰ thresholds cover the C-band/X-band ambiguity |
| Reference **answers, labels, bounding boxes, masks** | All four output types implemented and emitted by the batch runner |
| **Scores normalised before combining** | Drives the breadth-over-depth priority and the descope-ladder ordering — `01` Axiom 5, `04` §8 |
| Large-scale batch evaluation implied | **Headless batch runner from day one**, `run_batch` on every tool, `--fast` verification skip — `01` Axiom 4, §7 |

---

## 4. Unverified claims register

No network access was available while writing these documents. Most items below remain unconfirmed. **Update 2026-08-27:** the user supplied the BigEarthNet.txt paper (`2603.29630v2`) and the dataset's HuggingFace card directly, so **item 1 is now fully verified** — contents *and* licence (**CDLA-Permissive-1.0**, a permissive open-data licence; see the row and `03` §4.1). The remaining load-bearing unknowns are items 2 (Cartosat SWIR) and 3 (RISAT band), both needing a real Bhoonidhi product.

| # | Claim | Status | Consequence if false |
|---|---|---|---|
| 1 | BigEarthNet.txt at `txt.bigearth.net`: 464,044 co-registered S1/S2 pairs, 9.6 M annotations incl. captions + VQA (binary + MCQ) + referring expressions; benchmark subset 1,082 pairs / 15,029 annotations; arXiv 2603.29630 | **VERIFIED 2026-08-27 — paper + HF card; licence = CDLA-Permissive-1.0** | Fully resolved; licence permits use. Download = 467 MB Parquet (9,553,962 rows = 1 per annotation, **text only**); S1/S2 imagery is a separate reBEN pull (large — size storage around it). Format-only fallback if that pull is impractical: BigEarthNet v2 + GeoChat-Instruct + VRSBench; Track A unaffected |
| 2 | Cartosat-2S MX is 4-band VNIR with **no SWIR** → MNDWI and NDBI unavailable | **Assumption, high confidence** | **Load-bearing.** If SWIR exists it is pure upside — enable both index paths |
| 3 | The RISAT in the ISRO set: RISAT-1 (C-band, ~3–50 m) vs RISAT-2B/2BR1 (X-band, ~0.35–4 m) | **Unknown** | **Load-bearing.** Already mitigated by adaptive rather than absolute σ⁰ thresholds |
| 4 | CROMA / DOFA checkpoints downloadable under a permissive licence | Unverified | Fall back to torchgeo SSL weights |
| 5 | Change-Agent / LEVIR-MCI weights available | Unverified | TinyCD + separate caption head, ~1 extra GPU-h |
| 6 | Qwen2.5-VL-3B is still the best ≤4B VLM in Aug 2026 | **Open — knowledge ends May 2025.** Note: the BigEarthNet.txt paper (early 2026) built its RS baseline on **InternVL3-1B** — evaluate it as an alternate | Substitute per the six criteria in `03` §2.1; drop-in by design |
| 7 | SpaceNet 6 / Umbra / Capella high-res SAR accessible and licensed | Unverified | Stage A3 runs optical-only; document the limitation |
| 8 | GeoChat-7B / RS-LLaVA / LHRS-Bot downloadable for zero-shot baselines | Unverified | Baseline against the un-finetuned base VLM only |
| 9 | SIH 2026 calendar: internal deadline, finale dates, submission format | **Unknown** | Compress the `04` phase plan proportionally, cutting from the top of the descope ladder |
| 10 | Published benchmark numbers cited anywhere in these docs | Unverified | Treat every number as directional until the harness produces your own |

**Rule for the team: no GPU-hour is spent on any path whose enabling claim is still unverified.** The full 12-item gate with owners and deadlines is in `03` §6.

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

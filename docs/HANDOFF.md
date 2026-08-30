# Session handoff — 2026-08-29

State: branch `phase-0-closeout`, **855 tests passing**, working tree clean,
**pushed** to PR #2. **Phase 4 is audited and largely closed — read
`docs/phase4-status.md` first**: 5 of 8 tasks DONE, 3 PARTIAL, and the missing
halves are a person in a room (narrated rehearsals on the venue laptop), a
licence decision (publishing weights), and a screen recording (the backup
video). Nothing else is blocking.

New in Phase 4: `docs/technical-report.md`, `docs/model-cards.md`,
`docs/deck.md`, `docs/judge-qa.md`, `docs/rehearsal.md`, `docs/code-freeze.md`,
`docs/phase4-status.md`, and `docs/ps-26167.md` — the authoritative PS text,
now in git, which is what made the traceability check possible.

**Update 2026-08-30.** The two items this file listed as highest-value were
worked, and CDVQA went from not-on-disk to fully measured in three passes.
Read the **last three sections** of `docs/phase1-status.md` in order; each
corrects the one before it.

- **CDVQA now scores 0.5380** at 100% coverage (39,686 questions, 968 pairs)
  against a per-type majority baseline of 0.5084 and an oracle ceiling of
  **0.9975**. The first measurement was 0.0000 and the second was 0.4439 -
  *below* the baseline. Only the third beats it.
- **The whole benchmark is arithmetic over a pair of semantic change maps**,
  so the neural problem is one segmentation task. 93% of the remaining
  headroom is the segmenter's 0.2636 change-class mIoU.
- **A 20-point routing gap was found and closed.** Calling the tool directly
  scored 0.5701 where the full controller scored 0.3616, because only 67.4%
  of CDVQA questions reached the tool that answers them - `change_to_what`
  reached it 0.000 of the time. The two paths now agree to six decimals.
- **The SAR sensor question is narrowed by elimination** but still needs one
  sentence from the team; see §"Which RISAT" in `docs/verification.md`.

Read `docs/phase1-status.md` first. It carries every measured number with its
caveats, in dated sections, and later sections correct earlier ones.
`docs/verification.md` tracks the 12-item gate (**6 resolved**).

---

## What happened this session

Phase 3 was built end to end (all 14 tasks addressed, 13 complete), then
audited for correctness, security, packaging and completeness. The audit found
more than the build did.

### Phase 3 headline results

| result | number |
|---|---|
| agent vs monolith — impossible plans | **148/600 (24.7%) ungated, 0/600 gated** |
| land-cover head at threshold 0.5 | **worse than always predicting negative** (0.2064 vs 0.1834) |
| E-AURC: router vs land-cover | 0.0405 vs 0.0966 (raw AURC makes them look equal) |
| entailment gate, clean suite, hybrid | **96%**, all 9 contradictions caught |
| deterministic gate cost | **+1.9 ms/query**; NLI **+2,625 ms** (22× the pipeline) |
| change-mask calibration | ECE 0.0668 → **0.0034** (affine, not temperature) |

### Task 3.1 is the one incomplete task

Track B retrained on a 5,340-example mix (1,654 SAR, 4.57% refusals).
**VQA improved with no regression** — `rsvqa_lr` exact match 0.4510 → 0.6373 on
the distribution both models trained on.

**Refusal is a negative result.** Recall 0.4118 decomposes into **5/5 (100%)
on lexical refusals** and **2/12 (16.7%) on the image-conditional one**. The
model learned to refuse when the *question* is impossible on its face and did
not learn to refuse when the *image* is the reason. That is the harder half.
Loss plateaued from step ~45 over ~half an epoch; refusal fraction, epoch count
and learning rate are three candidate causes and this run separates none.

---

## The habit that mattered

Four measurement artifacts were caught **in my own work**, and they are the
reason to trust the rest:

1. The first hybrid entailment gate scored **identically** to deterministic
   alone — the precedence rule meant NLI was never consulted on the six cases
   that mattered.
2. The verifier ablation first reported **+440 ms/query** for the gate. That was
   cold-start cost attributed to the verifier. With warm-up: +1.9 ms.
3. The soak test at the plan's **20 iterations** reports +0.2445 MB/query; at
   120 with warm-up excluded it is +0.0239. The plan's own number would have
   produced a false leak alarm.
4. A confidence stressor wrote zeros without setting the raster's nodata value,
   so it moved the wrong component — a bug in the *measurement* that read as a
   failure of the *system*.

Where a suite was used to change the design, it is marked burnt and a fresh one
written (`TUNED_CASES` vs `CLEAN_CASES` in `evaluation/entailment_bench.py`).
**Before concluding a model failed, check the split can answer the question.**

---

## Audit findings (all fixed)

**Security** — four defects reachable by an unauthenticated caller: unbounded
uploads; temp directories never deleted; the blanket handler converting a
deliberate 413 into a 500; and the **SSE path reaching past the controller**,
which had drifted so streamed answers silently lost the task-3.8 exclusion
notice — the path the frontend uses.

**Packaging** — both Docker images **could not build** (`python:3.11-slim` while
rasterio 1.5.1 declares `requires_python >=3.12`, verified against PyPI).
Compose set `PROFILE` where the loader reads `SATQUERY_PROFILE`. Pillow was
undeclared despite being a direct runtime import. `reportlab` was missing from
pyproject, which CI installs from. Added a `train` extra — the whole GPU stack
was undeclared.

**Completeness** — Phase 2 tasks 2.5, 2.7, 2.8 were incomplete: the models were
trained and their metrics published, and **no tool module existed**, so three
checkpoint directories were unreachable from a query. Now wired.

Wiring them exposed a **verifier defect**: `extract_claims` returned only the
*first* subject in a sentence, so "a bridge is over a river with some green
trees" was checked on vegetation only and the water claim never at all.

---

## Open — in priority order

1. **Confirm which SAR sensor ISRO/SAC will use.** Still open, but
   **narrowed by elimination on 2026-08-30** (§"Which RISAT" in
   `docs/verification.md`): of the four candidates, RISAT-1 was decommissioned
   in 2017, RISAT-1B/EOS-09 failed at launch in 2025, and RISAT-2B/2BR1 are
   X-band but carry "data not ordinarily available to the public" and appear
   nowhere in Bhoonidhi's civil catalogue - leaving **EOS-04 as the only
   RISAT that is both operational and openly served**. The counter-argument
   is honest: Cartosat-2S at 0.65-1.6 m pairs better with RISAT-2B's 0.35 m
   than with EOS-04's 2.5 m, and SAC can reach restricted data. One sentence
   from the team still decides it. Verification item 8 is resolved to a *no*:
   high-res SAR is freely available and permissively licensed, but Umbra,
   Capella and SpaceNet 6 are all **X-band** (9.69 GHz measured from a real
   product) while EOS-04 is **C-band** (5.40 GHz) — a 1.79× wavelength ratio
   against the 0.09% Sentinel-1 match. **If the sensor is RISAT-2B/2BR1 instead,
   those are X-band and this inverts** — Stage A3 should then be redone against
   0.25 m SAR rather than the optical-only arm that shipped.
2. **Verification item 10** — SIH deadline and submission format. Needs the team.
3. **The Cartosat priced-data risk** — for the team lead.
4. **Review and merge [PR #2](https://github.com/aditya-coder07/SatQuery_AI/pull/2)** - PR #1 is already merged; the handoff's
   instruction to merge it was stale.

### Known gaps, deliberately left

- **A flake I could not explain.** `test_swir_free_path_exercised_on_real_cartosat`
  failed once under the CI simulation and passed on three subsequent full runs.
  One in four is not "fine"; it is unresolved.
- **Tier-2 LLM tiebreak unbuilt** — `llm_tiebreak_invoked` is always `false`.
  Not one of the 14 tasks; an unbuilt feature with an honest flag.
- **`landcover_v1` asserts on ~0.25% of decisions** at 91% precision. Correct
  behaviour for a head with mAP 0.285, stated in `configs/thresholds.yaml`.
- **The entailment bench has no multi-subject sentences**, so it cannot exercise
  the case the real captioner produces immediately. Adding them to the clean
  suite would burn it.
- **`artifacts/run_*/` grows unbounded** (gitignored runtime output; the temp
  uploads *are* bounded).
- **3.1's refusal half** — see above.

---

## Environment

- Local RTX 4050 with CUDA, torch 2.13+cu126, bitsandbytes, peft, accelerate.
  `training/track_b_vlm_qlora.py`'s "cannot run here" docstring was stale and is
  corrected — it runs locally.
- `gh` is installed but not on the Git Bash PATH: call it as
  `"/c/Program Files/GitHub CLI/gh.exe"`.
- Gitignored but on disk: BigEarthNet shards, WHU-OPT-SAR, LEVIR-CD/MCI, RSICD,
  DIOR-RSVG, the Bhoonidhi products, **CDVQA** (`data/cdvqa/` - annotations,
  72 extracted image pairs and ~2.4 GB of mirror shards), `checkpoints/`,
  `models/` (including the 370 MB MNLI checkpoint for the gate's NLI backend).
- Learned tools are all **opt-in by environment variable** and fall back to
  stubs, which is what keeps CI green: `SATQUERY_CAPTION`, `SATQUERY_GROUNDING`,
  `SATQUERY_CHANGE_CAPTION`, `SATQUERY_LANDCOVER`, `SATQUERY_CHANGE_MASK`,
  `SATQUERY_FUSION`, `SATQUERY_NLI`, `SATQUERY_VQA_BASE`/`_ADAPTER`.
- `make report` regenerates every evaluation artifact under `docs/assets/`.
- **CI has no torch.** Verify with the block-import simulation before claiming a
  green CI — it caught a module-scope import that made the whole package
  unimportable, which the normal suite could not.

---

# Problem-statement traceability (checked 2026-08-29)

The PS text was re-read against the code. Mandatory scope is covered; the gaps
are in the **prescribed evaluation datasets**, which is how the PS says the
work will be scored.

## Mandatory functional scope — covered

| PS requirement | status |
|---|---|
| RS adaptation of a visual/VL component | Track A on BigEarthNet imagery, Track B QLoRA on the instruct mix |
| Single-image VQA (mandatory) | `rs_vqa_v1` + deterministic `change_vqa_v1` |
| **Plus** captioning **or** grounding | **both** wired (`caption_v1`, `grounding_v1`) |
| Change description **or** change VQA (mandatory) | both (`change_caption_v1`, `change_vqa_v1`) |
| Spatial change map (optional) | `change_mask_v1`, georeferenced COG |
| Cross-modal optical–SAR extraction | `optsar_fusion_v1`, triad mode, complementarity now in the trace |
| Agentic orchestration | router + capability matrix + validated plans; **0/600 illegal plans** |
| GUI / web app | Next.js: run view, comparators, map, `/models`, `/benchmarks`, `/runs/{id}` |
| Visual evidence, confidence, execution summary, downloadable report | map + evidence pack, three-component confidence, full trace, PDF |
| GeoTIFF/TIFF, **PNG/JPEG for benchmarks** | fixed 2026-08-29 — see below |

## Gaps against the PS — ordered by evaluation risk

1. ~~**CDVQA is not on disk and never evaluated.**~~ **Resolved 2026-08-30 -
   and the answer is a zero.** The official annotations are Apache-2.0 and
   `curl`-able; imagery came from a webdataset mirror that
   `training/prepare/cdvqa.py` verifies against them sample by sample. First
   measurement: **0.0000 exact match on all eight question types**, 2,900
   questions, 34.5% coverage. Checked for a scoring artifact and it is not one
   - a deliberately lenient rescoring gives 0.0076. The causes are structural:
   CDVQA imagery is **RGB, so all four classical indices are unavailable**;
   seven of eight question types need a **semantic change head that does not
   exist**; and the one class-agnostic type never reaches the change mask's
   measured percentage because tools do not see each other's outputs. Full
   write-up, including what would move it and what should not, at the end of
   `docs/phase1-status.md`.
2. **BigEarthNet.txt (the image–text corpus) was never used.** The PS Background
   calls it "the primary dataset for adapting image–text representations". We
   adapted on BigEarthNet *imagery + 19 labels* instead. The Mandatory Scope
   says "BigEarthNet.txt **or any open source training data**", so this is
   defensible — but it is a stated expectation, and a judge may ask. Decide
   whether to run an adaptation pass on it or to justify the substitution in
   the report.
3. **VRSBench evaluation is partial.** Item 9: it ships annotations only, with
   imagery in DOTA and DIOR. DIOR is on disk, DOTA is not.
4. **`landcover_v1` asserts on ~0.25% of decisions.** Fine for honesty, thin for
   a demo. The narrative synthesiser carries land-cover answers.

## Fixed while checking

**PNG/JPEG benchmark inputs were rejected outright.** The PS admits them "only
for the prescribed public benchmark datasets", and those are ungeoreferenced by
construction — RSVQA and VRSBench ship plain rasters. `check_crs_present`
FAILed them in every mode, so **no prescribed benchmark image could enter the
pipeline at all**. In `IngestMode.BENCHMARK` a missing CRS is now a WARN: the
answer is valid, nothing downstream can georeference or co-register, and the
trace says so. Operational mode still refuses. Verified end to end on a
PNG (`SINGLE_VQA`, no abstention, WARN recorded).

## Evaluation-set note

The ISRO/SAC set is **pre-georeferenced, co-registered Cartosat-2S + RISAT
pairs**. Two measured facts bear on it: EOS-04 is C-band at 5.40 GHz (item 5),
and the cross-sensor test found vegetation agreement collapsing from +0.476 at
10 m to **-0.135 at native 1.6 m** — resolution, not bands, is the dominant
gap, which is why Stage A2/A3 exist.

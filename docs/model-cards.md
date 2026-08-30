# Model cards — SatQuery AI

**Plan task 4.5.** One card per trained component. Every number is read from
the `metrics.json` and `run_metadata.json` written by the training run that
produced the checkpoint; nothing here is estimated, rounded up, or carried
over from a paper.

**Read the weakest line in each card.** Several of these models are honestly
poor, and the card says so rather than reporting only the flattering figure.
A reviewer who finds a weakness we did not disclose has learned something
about our reporting, not about the model.

## Publication status — read before distributing anything

| Component | Weights publishable? | Why |
|---|---|---|
| `change_vqa_v1` semantic head | **BLOCKED** | Trained on **SECOND**, which states **no licence at all** — not a restrictive one, none. See `docs/verification.md` §"NEW RISK — SECOND states no licence". |
| Everything else | **Undecided — do not publish yet** | Each was trained on an openly licensed corpus, but no licence has been chosen for our own weights and no per-dataset redistribution check has been done. |

The PS's deliverable is *"codes and models including test and demonstration"*.
Code and tests are in the repository. **Weights are not published**, and the
blocker above is the reason for one of them; the rest await a decision that is
the team's to make. This is recorded rather than quietly skipped.

---

## Track A — band-agnostic encoder → `landcover_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/track_a_full_base/` |
| Task | Multi-label land cover, BigEarthNet-19 |
| Training data | BigEarthNet v2 imagery + 19 labels, 30,000 patches, 3 epochs, lr 1e-3, dim 64, seed 42 |
| Key design | Band-presence masking + **random band dropout 0.3** + GSD conditioning |
| **mAP, all 12 bands** | **0.2854** |
| **mAP, Cartosat 4-band subset** | **0.2573** |
| **Retention (4-band / 12-band)** | **0.9015** |

**What the retention number is for.** The model must survive losing eight of
twelve bands, because Cartosat-2S MX has four. Band dropout is what buys that,
and it was ablated rather than assumed:

| arm | mAP all bands | mAP 4-band | retention |
|---|---|---|---|
| with dropout (`track_a_dropout`) | 0.4171 | 0.3765 | **0.9025** |
| without (`track_a_nodropout`) | 0.4310 | 0.3639 | 0.8443 |

Dropout costs 1.4 points of full-band mAP and buys 5.8 points of retention.
That is the trade the design claims, measured.

**Weakness:** mAP 0.2854 is low. At threshold 0.5 the head is **worse than
always predicting negative** (0.2064 against 0.1834), which is why
`landcover_v1` asserts on only ~0.25% of decisions at 91% precision and the
narrative synthesiser carries land-cover answers. See limitation L8.

**Not shipped in git:** `band_stats.json` lives beside the weights and
`checkpoints/` is gitignored, so a fresh clone cannot load this head until the
statistics are regenerated with `compute_stats(seed=0, sample=2000)` over the
BigEarthNet train shards — which needs the 45 GB corpus. Limitation L2.

### Stage A2 — WHU-OPT-SAR transfer

| | fine-tuned | frozen probe |
|---|---|---|
| mAP | **0.7759** | 0.7206 |

Fine-tuning beats a frozen probe by 5.5 points, which is the evidence that the
transfer is doing work rather than the head fitting the labels alone.

### Stage A3 — high-resolution transfer

| | |
|---|---|
| Frozen probe mAP | 0.1151 |
| Fine-tuned mAP | **0.2880** |
| **Adaptation gain** | **+0.1729** |

Ran **optical-only**, on evidence: verification item 8 measured every
accessible high-resolution SAR source as X-band against EOS-04's C-band. That
was the plan's documented fallback, chosen because the measurement said so and
not because the data was unavailable.

---

## Track B — `rs_vqa_v1` (QLoRA adapter)

| | |
|---|---|
| Checkpoint | `checkpoints/track_b_v1/adapter_final` |
| Base | Qwen2.5-VL-3B-Instruct, 4-bit |
| Training data | 4,806-example RS instruction mix, lr 1e-4, effective batch 8, 300 steps, seed 42 |
| **`rsvqa_lr` exact match** | **0.6425** (n=207) |
| Full held-out val, exact match | 0.3810 (n=534) |
| Full held-out val, token F1 | 0.7913 |

**The comparison that matters** — v0 and v1 on an *identical* held-out split:

| | v0 | v1 |
|---|---|---|
| `rsvqa_lr` exact match | 0.4510 | **0.6425** |

**Weakness — refusal is a negative result.** Recall **0.4118** decomposes into
**5/5 (100%) on lexical refusals** and **2/12 (16.7%) on image-conditional
ones**. The model learned to refuse when the *question* is impossible on its
face and did not learn to refuse when the *image* is the reason — the harder
and more useful half. False-refusal rate 0.0077; lexical-shortcut probe
0.1667. Limitation L3.

**Not evaluated on VRSBench**, which the PS assigns to VQA alongside RSVQA:
VRSBench ships annotations only and its imagery lives in DOTA, not on disk.
Limitation L11.

---

## `caption_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/caption/` |
| Training data | RSICD, 8,734 examples, 8 epochs, dim 192, vocab 1,781 |
| **BLEU-4 (sentence mean)** | **0.2446** (n=1,093) |
| Unique captions | **146 of 1,093 — 13.4%** |

**Weakness, and it is the diversity number, not the BLEU.** The model emits
146 distinct captions across 1,093 images. It has learned the corpus's common
sentences well enough to score, and it is not describing each scene
individually. Quote the 13.4% alongside the 0.2446 or the BLEU misleads.

**Not on the prescribed split:** the PS assigns captioning to VRSBench, which
is not evaluated. This number is on RSICD.

---

## `grounding_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/grounding/` |
| Training data | DIOR-RSVG, 6,359 examples, 5 epochs, dim 128, **backbone from scratch** |
| **Acc@0.5** | **0.0762** (n=1,141) |
| Acc@0.7 | 0.0088 |
| mIoU | 0.1405 |

**This is the weakest model in the system and it should not be presented as
working.** Acc@0.5 of 0.0762 means roughly nine in ten referring expressions
are not localised. It satisfies the PS's M3 only because M3 requires
captioning **or** grounding and captioning is the stronger arm — but the PS's
own representative query *"Highlight the water body referred to in the query"*
routes here, so the routing is right and the answer is usually not.

`run_metadata` records `split_note: NO published split in this mirror`, so the
split is ours and the number is not comparable to published DIOR-RSVG results.

---

## `change_mask_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/change_mask/` |
| Architecture | TinyCD-style siamese, **49,185 parameters** |
| Training data | LEVIR-CD, 7,120 tiles, 4 epochs, `pos_weight` 10.11 |
| **F1 (change class)** | **0.5597** |
| IoU | 0.3886 |
| Precision / Recall | 0.4426 / **0.7613** |

Scored on the **change class only**: LEVIR-CD is heavily imbalanced and
overall pixel accuracy would sit near 0.98 for a model that predicts "nothing
changed" everywhere.

The precision/recall split is a design choice worth stating: at 0.44/0.76 the
detector over-calls change. For a screening tool that surfaces candidates to
an analyst, recall is the cheaper error.

**Calibration:** ECE **0.0668 → 0.0034** after an *affine* fit. Temperature
scaling alone did not work on this head, and the calibration report records
which transform was accepted rather than assuming the usual one.

---

## `change_caption_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/change_caption/` |
| Training data | LEVIR-CC, 6,815 examples, 6 epochs, dim 128, vocab 377 |
| **BLEU-4, changed pairs** | **0.3063** (n=964) |
| BLEU-4, unchanged pairs | 0.9706 (n=965) |
| BLEU-4, aggregate | 0.5686 |

**Quote 0.3063, never 0.5686.** The aggregate is inflated by the trivially
unchanged half, where the reference is a fixed "no difference" sentence the
model learns to emit verbatim. The checkpoint's own `metrics.json` carries
this warning in a `note` field. 85 unique captions.

---

## `change_vqa_v1` — semantic change head

| | |
|---|---|
| Checkpoint | `checkpoints/change_vqa/best.pt` |
| Architecture | Siamese **ImageNet-pretrained ResNet-18** encoder, two per-date decoders |
| Training data | SECOND via **CDVQA's own train ids** — 1,600 pairs, 400 val, **968 test ids never read** |
| Pixel accuracy | 0.7528 |
| mIoU | 0.3323 |
| **Change-class mIoU** | **0.2636** |

Pretraining ablation, same data and schedule:

| encoder | change-class mIoU |
|---|---|
| from scratch (`change_vqa_scratch`) | 0.1691 |
| **ImageNet ResNet-18** | **0.2636** (+56% relative) |

**End-to-end on the PS's prescribed benchmark**, full split, 39,686 questions
over 968 pairs at **100% coverage**:

| | accuracy |
|---|---|
| per-type majority baseline (fitted on train, applied to test) | 0.5084 |
| **system** | **0.5380** |
| oracle over ground-truth change maps | **0.9975** |

**The three numbers must be read together.** +3.0 points over a constant is a
real but small win; the 0.9975 oracle says the answer layer contributes no
measurable error and **93% of the remaining headroom is this segmenter**.
Earlier iterations scored 0.0000 and then 0.4439 — *below* the baseline — and
both are recorded in `docs/phase1-status.md` rather than deleted.

**Weights are not publishable.** See the table at the top.

---

## `optsar_fusion_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/optsar_fusion/` |
| Training data | WHU-OPT-SAR, 1,548 tiles, 5 epochs, dim 32 |
| Optical only | **0.7778** |
| SAR only | 0.7410 |
| Fused | 0.7714 |
| **Complementarity gain** | **−0.0064** |

**A negative result, reported as one.** Fusion does not beat optical alone on
this dataset. The PS's M6 asks the system to *extract complementary
information from a co-registered pair*, and it does — the triad runs, the
per-modality numbers are produced, and the complementarity score is reported
in the trace. What is **not** supported is the claim that fusing helps, and
that claim must not be made on stage.

`run_metadata` records `split_method: deterministic random by tile; NOT
geographic`, so tiles from one scene can span train and test and the absolute
numbers are optimistic. The *comparison* between the three arms is unaffected,
which is what the ablation is for.

---

## `index_engine_v1` — not a model

Deterministic: NDVI, NDWI, MNDWI, NDBI, σ⁰, VH/VV, GLCM texture, CoV, adaptive
Otsu/GMM thresholding, SWIR-free fallbacks. No learned parameters, no training
data, no failure mode beyond bad input. It is the independent referee the
neural outputs are checked against, and it is why quantitative answers in this
system come from arithmetic rather than generation.

# The ten hardest questions, and the honest answers

**Plan task 4.8.** These are the questions a reviewer who reads carefully will
actually ask. Several have uncomfortable answers; those are the ones worth
rehearsing, because the alternative is discovering them on stage.

**The rule for all ten: lead with the number, including when it is bad.** A
judge who finds a weakness we concealed has learned something about our
reporting. A judge who hears it from us first has learned something about our
rigour.

---

### 1. "Your grounding accuracy is 7.6%. Isn't the system broken?"

**No, and the number is real — Acc@0.5 = 0.0762, Acc@0.7 = 0.0088.** It is the
weakest component we have.

Three things make it survivable rather than fatal. The PS's M3 requires
captioning **or** grounding, and captioning is the stronger arm at BLEU-4
0.2446. The backbone is trained *from scratch* — `run_metadata` records
`backbone: from scratch (no remote code)` — and when we replaced a
from-scratch encoder with an ImageNet-pretrained one on the change segmenter,
change-class mIoU rose 56% relative. So the fix is identified, not mysterious.
And the split is ours, not a published one, so this is not comparable to
published DIOR-RSVG numbers in either direction.

What we will not claim: that grounding works well. The PS's own query
*"Highlight the water body referred to in the query"* routes here correctly
and is usually localised wrongly.

---

### 2. "You built optical–SAR fusion. Does it actually help?"

**No. Complementarity gain is −0.0064: fusion (0.7714) is worse than optical
alone (0.7778).**

We report the triad — optical-only, SAR-only, fused — precisely so this is
visible rather than hidden inside a single fused number. The PS's M6 asks the
system to *extract complementary information from a co-registered pair*, and
it does: the three arms run, the per-modality numbers are produced, and the
complementarity score is in the trace. What is not supported is the claim that
fusing helps on WHU-OPT-SAR, and we do not make it.

The split is also `deterministic random by tile; NOT geographic`, so tiles from
one scene can span train and test and the absolute numbers are optimistic. The
*comparison between arms* is unaffected, which is what the ablation is for.

---

### 3. "Your CDVQA score is 0.5380. A constant scores 0.5084. Why should I be impressed?"

**You shouldn't be impressed by the margin. You should be interested in the
decomposition.**

The oracle over ground-truth change maps scores **0.9975**, which says the
answer layer contributes no measurable error and the entire remaining gap —
93% of it — is one segmentation model at change-class mIoU 0.2636. That turns
a vague "improve the VQA" into a well-posed problem with a published
literature.

The history matters more than the number. The first measurement was
**0.0000**. The second was **0.4439** — *below* the constant baseline — and we
reported it as a failure, with the baseline script printing "The head does NOT
beat the per-type majority baseline". Only the third beat it. All three are in
`docs/phase1-status.md`; none was deleted.

---

### 4. "How do I know your agentic layer is doing anything a prompt couldn't?"

**Measured: the same classifier, ungated, selects an impossible task on
148 of 600 plans — 24.7%. Gated, 0 of 600.**

The guarantee is structural, not statistical. The legal task set is computed
from the **images**, never from the query text, so no phrasing can widen it —
which is why 200 adversarial queries across three input configurations produce
zero illegal plans rather than "few". A prompt-based agent has no such
property; it has a strong tendency.

We also do not claim the routing is *good*: Tier-1 accuracy on a never-tuned
29-item holdout is 0.5862. A misroute degrades the answer. It cannot produce
an illegal plan.

---

### 5. "The PS names BigEarthNet.txt as the primary dataset. Did you use it?"

**No. We adapted on BigEarthNet imagery plus its 19 labels, not on
BigEarthNet.txt, the image–text corpus.**

The Mandatory Scope says "using BigEarthNet.txt **or other open source
training data**", so the requirement is met. The Background states the
expectation we did not meet, and we would rather say that than hope it is not
noticed.

Why: BigEarthNet.txt is 467 MB of text with imagery in a separate large reBEN
pull, and Track A's job is a band-agnostic *encoder* over 12 spectral bands —
the labels carry that signal directly. The image–text corpus is the right
choice for a captioning-first design; ours is verifier-first.

---

### 6. "Your confidence says HIGH and the system still abstained. Which is it?"

That was a real defect and it is fixed. An abstained run now renders **"Status:
Abstained"**, names the failing check, and shows the headline confidence as
**"not applicable — the run abstained"**, with the three components still
visible because they are the diagnosis.

The underlying arithmetic was always right — the run abstained on *input
validation*, not on low confidence — and we deliberately did **not** change the
combiner to make the interface look better. The confidence describes an answer
that was never returned; the interface now says so.

---

### 7. "You claim co-registration checking. Prove it."

**Partly.** Footprint overlap is measured and enforced: an optical and a SAR
scene written 60 km apart is refused with `footprint_overlap 0%` named, and
that gate is in the demo's opening beat.

Sub-pixel co-registration is **not** verified, and here is the measurement that
stopped us: on a pair with *identical* footprints — 100% overlap, same CRS,
same GSD — the gradient-domain phase correlation reports **38.1 px** against
the matrix's 2.0 px limit. Enforcing that gate refused well-formed pairs. So
the estimator is a relative quality signal whose absolute accuracy across
modalities is unvalidated, and gating on it would have been a lie dressed as
rigour.

We have no real co-registered optical–SAR pair to settle it — our Cartosat and
EOS-04 products are not co-located. That is the next acquisition.

---

### 8. "Which RISAT did you build for? Cartosat-2S pairs with a 0.35 m sensor."

**The PS does not specify, and it explicitly tells us not to assume one.**

So the implementation is sensor-configurable: adaptive rather than absolute σ⁰
thresholds, which is why the RISAT band question never sat on the critical
path.

What we do know, from primary evidence rather than a web page: a real EOS-04
product reads `radarCenterFrequency = 5.40 GHz` — C-band, within **0.09%** of
Sentinel-1, so S1-trained backscatter behaviour transfers almost directly. And
every accessible high-resolution SAR source (Umbra, Capella, SpaceNet 6) is
**X-band at 9.69 GHz** — a 1.79× wavelength ratio. That measurement is why
Stage A3 ran optical-only: the plan's documented fallback, chosen on evidence
rather than on unavailability.

If SAC confirms RISAT-2B or 2BR1, that inverts and Stage A3 should be redone
against 0.25 m X-band SAR. Roughly 2–4 GPU-hours plus downloads.

---

### 9. "You only evaluated two of the three prescribed benchmarks."

**Correct.** RSVQA-LR and CDVQA are evaluated; **VRSBench is not.**

VRSBench ships annotations only — 142,390 LLaVA-style rows referencing imagery
that lives in DOTA and DIOR. DIOR is on disk; DOTA is not. So the benchmark
the PS assigns to captioning and grounding has no number from us, which also
means our BLEU-4 of 0.2446 (RSICD) and Acc@0.5 of 0.0762 (DIOR-RSVG) are **not
on the prescribed split**.

It is a gap, it is one download away, and it is recorded as limitation L11
rather than glossed.

---

### 10. "What is the single thing most likely to be wrong in what you have shown me?"

Two candidates, and we would name both.

**The two-track adaptation ablation is `not_comparable`.** The two tracks were
trained and evaluated on different tasks, so no controlled comparison exists.
The central design decision of this project — a band-agnostic encoder plus an
instruction-tuned VLM, bridged — is *reasoned*, not *demonstrated*. Three of
the four planned ablations produced results; this one produced an admission.

**A flaky test we cannot explain.** `test_swir_free_path_exercised_on_real_cartosat`
has failed twice under the no-torch CI simulation and passed on every other
run, including immediately after each failure. The failing runs were slower
(272 s against ~105 s typical), which *suggests* I/O contention — a hypothesis,
not a diagnosis. One in four is not "fine", and it is listed as unresolved.

---

### "Can you load the models you have just described?"

**Mostly yes — and the exception is worth hearing.** On 2026-08-30 a Phase-0
test harness deleted `checkpoints/`; on 2026-08-31 the tree was restored from a
volume shadow copy, 4.542 GB, verified bit-exact against a SHA-256 recorded
before the deletion. Six of eight learned tools load and answer:
`landcover_v1`, `change_mask_v1`, `change_caption_v1`, `optsar_fusion_v1`, the
semantic `change_vqa_v1`, and the Track B adapter.

**Seven of eight, and the eighth is the honest part of the answer.** **CORRECTED 2026-09-01: seven of eight load, not eight.** The Track B QLoRA adapter is destroyed - `adapter_model.safetensors` is 148,712,776 bytes of which the first 148,701,184 are NUL (99.9922%), and the same is true of all eleven adapter files, 1.636 GB in total. The earlier claim came from a verification that loaded the 61 `.pt` files and only *hashed* the safetensors, so a whole model's weights were reported as recovered without ever being opened. Re-verified by loading every weight file: **64 load (10.784 GB), 11 fail (1.636 GB)**, the failures being exactly the adapters. See `docs/00` section 3.6 **L32**.

We found it by running a real image through the system rather than by trusting the recovery report, which is the same habit that caught the four measurement artifacts in section 'The habit that mattered'. Twelve small JSON
sidecars came back as NUL bytes: the snapshot captured their size but not
their contents. Three of them were load-bearing — the caption and grounding
vocabularies and the multires band statistics — and a vocabulary of zeros
decodes every token id to nothing, so we would rather have had an error than a
silent empty caption. The availability check now parses its sidecars instead
of checking they exist.

We regenerated all three from the project's own code and the datasets on disk,
and **validated each by reproducing a published number rather than by looking
at it**: the caption evaluation returned BLEU-4 **0.24460787515482577** against
the published **0.24460787515482577**; grounding reproduced Acc@0.5 and Acc@0.7
bit-exactly; and the multires evaluation, which recomputes those statistics
independently, returned mAP identical to seventeen significant digits at all
four GSD levels. A vocabulary of the right size but the wrong order would load
without complaint and decode different words — that is exactly why the test is
the metric and not the file.

**No measured number changed through any of this.** They were read from the
`metrics.json` each run wrote; those files restored intact and were checked
against the published figures rather than the reverse.

---

### "Can you load the models you have just described?"

**No, and the reason is on the record.** On 2026-08-30 a Phase-0 test harness,
`training/run_checkpoint_test.py`, deleted `checkpoints/`. It hardcoded that
directory, called `shutil.rmtree` on it unconditionally, and had no argument
parser — so passing `--help` to check whether it ran did not print help, it ran
the program. At least 2.75 GB of trained weights were destroyed, and recovery
failed everywhere it could be attempted without administrator rights.

What that does and does not mean:

* **Every measured number in this deck and in the technical report survives.**
  They were read from the `metrics.json` each training run wrote, and they are
  in `docs/assets/`, `docs/phase1-status.md` and `docs/model-cards.md`, all in
  git. **Nothing was re-derived, re-estimated or quietly adjusted.** If any
  number in front of you had been affected, this answer would say so.
* **What you are seeing run is the stub-and-index configuration** — the same
  one CI has always used, the same one the 958-test suite exercises, and the
  same one the demo bundle was verified under at 9 / 9 beats. The deterministic
  index engine, the router, the plan validator, the verifier, the abstention
  policy, the confidence combiner and the evidence pack are all real and all
  independent of the lost weights.
* **The `/models` page is empty**, because it reads its numbers off disk rather
  than carrying its own copy — which is the property that page was built for.
  An empty registry is the page telling the truth.
* **What is genuinely unavailable is a live neural answer** from the captioner,
  the grounder, the change captioner, the semantic change head, the land-cover
  head, the fusion head or the Track B adapter.

We would rather be asked this than have it discovered. The failure was ours,
the containment is in the repository — the harness now refuses `checkpoints/`
by name, refuses any directory holding files it did not write, and a test fails
if **any** runnable script lacks an argument parser — and the lesson worth
stating is the one the incident actually teaches: `checkpoints/` was gitignored
and had no backup, so a single `rmtree` was terminal.

---

## Two questions we want to be asked

**"Why is your land-cover head asserting on only 0.25% of decisions?"**
Because at threshold 0.5 it is worse than always predicting negative — 0.2064
against 0.1834 — and `configs/thresholds.yaml` says so in a comment. A model
that knows when not to speak is the one you can deploy.

**"Why is your soak test 120 iterations when the plan says 20?"**
Because at 20 it reports +0.2445 MB/query and at 120 with warm-up excluded it
reports +0.0239. The plan's own number would have produced a false leak alarm.
We changed the measurement, not the system.

# Natural-language understanding — diagnosis, design, measurement (2026-09-20)

Scope: the layer between a person's sentence and the plan the controller
executes. Everything here runs on CPU, trains in ~3 s at process start,
and is measured on a hand-written benchmark that is never fed to the
training templates.

## 1. Diagnosis — what the deployed system did with ordinary language

Measured on the deployed router (`tfidf_logreg_v1`) before any change, on
the held-out set written for this audit (`evaluation/nl/queries.jsonl`,
180 queries in the wording people type, labelled with the tasks that
would serve them; `queries_test.jsonl`, 63 more, written afterwards and
scored once):

| Split | Routed to an acceptable task | Single image | Two dates | Optical+SAR | Follow-ups |
|---|---|---|---|---|---|
| dev (180) | **71.7%** | 86.3% | **58.3%** | 60.0% | 53.3% |
| test (63) | **66.7%** | 84.0% | **50.0%** | 60.0% | 50.0% |

Four defects, all structural rather than "needs more templates":

1. **The classifier never saw the input configuration.** "did anything
   get built?", "are there any new structures?", "new roads?", "what got
   demolished", "what is different?" on a *two-image* input all routed to
   `SINGLE_VQA`, whose tool then answered about one image. The words are
   the same on one image (a presence question) and on a pair (a change
   question); only the configuration disambiguates, and the model had no
   way to know it.
2. **No parameters were ever extracted.** The tools received the raw
   sentence under `_query`. The grounding tool used it as the referring
   phrase, so *"Find the airport in this image."* reached the adapter as
   the prompt **"Locate the Find the airport in this image in the image
   and output its bounding box…"** — an adapter trained on DIOR-RSVG noun
   phrases ("the airport", "the plane on the left"). Land cover always ran
   with the matrix default `classes: [built_up, water]` regardless of the
   question. A question about "the second image" was answered about the
   first.
3. **No conversation state.** Every message was classified alone;
   "Where exactly?" after a change question routed to `SINGLE_GROUND` with
   the phrase "Where exactly?".
4. **Out-of-scope and open requests** fell on the class prior: "Analyze
   this satellite image and tell me the important things." abstained;
   "What is the weather forecast for this location?" went to the grounder
   because of the word *location*.

## 2. Design — NL → IR → plan

```
query (+ history)
  │
  ├─ resolve_follow_up ──► resolved_query          "Where exactly?" → "Show me where the changes occurred."
  │                                                (deterministic rewrite; note recorded)
  ├─ config gating ──────► legal tasks              (unchanged: the illegal-plan guarantee)
  ├─ classifier v2 ──────► task                     features: word+char n-grams + 21 lexical cues,
  │     input = "[bitemporal] <resolved query>"     config token in front; restricted to legal tasks
  ├─ understand() ───────► QueryUnderstanding       object_filter, referring_expression, spatial_scope,
  │                                                classes, quantity, temporal_relation, image_index
  ├─ plan params ────────► classes (validated against the matrix's enum_subset)
  └─ runtime params ─────► _query (resolved), _phrase, _image_index, _spatial_scope, _classes
```

`QueryUnderstanding` (`satquery/contracts/understanding.py`) is a typed,
validated pydantic model, carried in `trace.routing.understanding`,
returned by both run endpoints and shown on the query page ("Understood
as …" with chips). A trace now answers "what did the system think I
asked?" without reading logs.

Design rules, and why:

* **Cues are features, not rules.** `understanding.CUE_NAMES` (change,
  new, gone, two_images, locate, caption, classify, fuse, mask, count,
  fraction, area, comparison, question, greeting, anaphor, …) are 21
  binary features appended to the TF-IDF blocks of the same logistic
  regression. A cue word can shift a decision only in proportion to what
  the bank taught it; "location" alone does not make a grounding request.
* **The configuration is a token, and every example is also trained
  without one.** Change tasks are legal only on a bitemporal pair, fusion
  only on a cross-modal pair, the single-image tasks everywhere; each bank
  example gets one tokened copy cycling through its task's legal
  configurations. The router classifies the *bare* sentence too, to keep
  detecting "asked for a change mask on one image" (`config_excluded`).
* **Extraction never invents.** `extract_object` strips intent
  scaffolding (leads such as "can you find the", "where is", "how many";
  trails such as "in this image", "are visible", "increased, decreased or
  remained unchanged") and returns `None` when what remains still opens
  with a request verb ("describe", "classify", "produce"). The grounder
  falls back to the sentence as typed, with a warning, exactly as before.
* **The referring expression keeps the user's relational words.**
  `referring_expression` is the stripped residual with the article
  normalised — "the tennis court on the left of the tennis court at the
  bottom" — not a rebuilt "object + normalised position". The first
  version rebuilt it and lost the relational tail; the phrase-format
  experiment (§4) is what showed it.
* **A follow-up becomes a sentence.** `resolve_follow_up` rewrites
  "Where exactly?", "Only show buildings.", "And the dam?", "Is it near
  the coast?", "how many?", "as a mask please" against the previous
  turn's task and object into a standalone query, which is then gated,
  classified and validated like any other. The rewrite and its reason are
  in the trace (`follow_up`, `resolution`). Fresh sentences are never
  rewritten. The API accepts `history` (JSON list of earlier turns) on
  `/runs` and `/runs/stream`; the query page keeps the thread and sends
  it, and starts a new conversation when the scenes change.
* **One named image of a pair.** "Is there a bridge in the second image?"
  → `image_index 1`; the single-image tools read
  `imaging.selected_image(manifest, params)`. "… more built up than the
  first" names both → `None` → the task's default.

## 3. Measurement

Protocol: `evaluation/nl_understanding_eval.py` runs the real `Router`
over synthetic manifests of the three configurations. A query scores when
the selected task is in its `accept` list; extraction fields score
against labels where given. **Nothing in the benchmark files is in the
template bank** — `tests/test_understanding.py` fails if any expansion of
the bank equals a benchmark query (it did, 17 times, during development;
each was rephrased on the bank side).

| Split | v1 routing | **v2 routing** | object | classes | quantity | spatial | temporal | image |
|---|---|---|---|---|---|---|---|---|
| dev, 180 (tuned against) | 71.7% | **99.4%** | 100% (34) | 100% (13) | 100% (20) | 100% (9) | 100% (54) | 100% (6) |
| test, 63 — **single shot, before any fix from it** | 66.7% | **84.1%** | 100% (13) | 100% (5) | 80% (5) | 100% (3) | 76% (17) | 100% (5) |
| test, 63 — after folding its failure *shapes* into the bank (no longer held out) | — | 96.8% | 100% | 100% | 100% | 100% | 100% | 100% |
| **final, 50 — written after every change, scored once, never tuned on** | 72.0% | **90.0%** | 89% (9) | 100% (5) | 100% (7) | 100% (3) | 82% (11) | 100% (4) |

Reading: the single-shot numbers (**+17.5** on the test split, 66.7 →
84.1; **+18.0** on the final split, 72.0 → 90.0) are the honest
generalisation estimate for the v2 layer; the 99.4 / 96.8 figures are the
state after tuning and are what the CI floors pin (0.97 / 0.93). The
final split's five misses: "Could you look for a helipad?", "Object
detection for windmills." and the bare follow-up "show me" abstain;
"How much land got paved?" abstains; "Is there any deforestation?" on a
pair goes to single-image VQA. They are left as they are - the split is
the measurement.
Two-image inputs went from 50–58% to 98–100%; follow-ups from ~50% to
100% on both splits. The residual misses are typo strings ("were r the
bildings", "were is teh airprot" → VQA rather than grounding) and one
abstention ("Sum this up in one line." on the test split before tuning).

By-group, v1 → v2 (dev): single 86.3 → 100; two-date 58.3 → 100;
optical+SAR 60 → 100; follow-up 53.3 → 100; technical/typo/adversarial
73.3 → 93.3.

Per-query records: `artifacts/benchmark_reports/nl_understanding_v2_queries.json`,
`…_queries_test.json`.

## 4. Grounding phrase format (evaluation/grounding_phrase_format.py)

Controlled comparison on the deployed grounding adapter
(`grounding_vlm_hires/adapter_best`, arm E) in the deployed precision
(NF4, 1024² input) over a seeded 150-expression subsample of the official
DIOR-RSVG test: the annotated phrase (`bare`), the phrase wrapped the way
a user types it and passed unchanged (`sentence` — the pre-fix served
path), and the same sentence through the extractor (`extracted` — the
served path now). Acc@0.5: bare 0.807, sentence 0.800, extracted 0.820;
every pairwise McNemar n.s. (`artifacts/benchmark_reports/grounding_phrase_format.json`).
So the wrapper cost about 0.7 points, inside the noise: the adapter reads
through it. The extractor is kept because it does no harm and gives the
adapter its training-time phrase form, not because it moves accuracy. A
subsample is a comparison of prompt formats, not a benchmark number.

## 5. What did not change

The classifier family (TF-IDF + logistic regression, fitted at process
start), the illegal-plan guarantee (config gating → classifier restricted
to legal tasks → matrix validation), the abstention policy, and every tool
contract. The golden traces were regenerated once (classifier name and
scores change; five adversarial cases now abstain or caption where they
used to run a tool on nonsense — recorded per case in the PR).

## 6. Limitations

* The extractor is English-only and pattern-based; a sentence outside its
  scaffolding patterns yields no object, and the grounder gets the
  sentence as typed (with a warning), which is the previous behaviour.
* Follow-up resolution reads the latest completed turn only, and the
  turn's *object*; it does not resolve references to an earlier answer's
  content ("the second one you mentioned").
* `classes` narrows the *answer*, not the head: `landcover_v1` opens with
  one sentence per asked-for class (present as which BigEarthNet labels /
  undecided / none asserted) and still lists everything it asserted; the
  head's decision is not changed by the question.
* Spatial scope is extracted for the trace and the grounder; the VQA
  model receives it inside the question only.

# Phase 4 (W13–W14) — status

Audit of plan tasks 4.1–4.8. Every DONE carries the artifact that proves it;
every PARTIAL names precisely what is missing; every BLOCKED names what blocks
it and who can unblock it.

**Three of the eight cannot be closed by the build.** They need a person in a
room, a licence decision, or a screen recorder. Marking them DONE would be the
kind of claim this project has spent four phases not making.

---

## Summary

| # | Task | Status |
|---|---|---|
| 4.1 | Curated demo bundle | **DONE** |
| 4.2 | Ten 7-minute rehearsals, incl. offline venue laptop | **PARTIAL** |
| 4.3 | Technical report | **DONE** |
| 4.4 | PS requirement traceability matrix | **DONE** |
| 4.5 | Model cards + published weights | **PARTIAL** (cards done; weights BLOCKED) |
| 4.6 | Deck + recorded backup video | **PARTIAL** (deck done; video BLOCKED) |
| 4.7 | Code freeze | **DONE** |
| 4.8 | Ten hardest judge Q&As | **DONE** |

**5 done, 3 partial, 0 wholly blocked.** Of the three partials, the missing
halves are: ten *narrated* run-throughs on the venue laptop, a licence
decision on weights, and a screen recording.

---

## 4.1 — Curated demo bundle · **DONE**

**Evidence:** `scripts/make_demo_bundle.py`, `data/demo_bundle/manifest.json`,
`tests/test_demo_bundle.py` (8 tests).

Nine inputs covering every beat, **three on real Bhoonidhi products**
(Cartosat-2E MX, EOS-04 FRS-1 HH, and the 7687×7640 Cartosat scene). Verified
**9 / 9 beats behave as scripted** via `--verify`, which runs each input
through the real controller and asserts the beat it exists to produce.

The plan asked for 8; there are 9, because the scripted opening rejection
turned out to need two inputs — the overlap rejection *and* the PNG rejection
— for reasons recorded under 4.2 below.

The bundle found three defects that testing had not: the PS's built-up query
abstaining, the opaque mask overlay, and a bi-temporal fixture containing no
change. All three are fixed.

## 4.2 — Ten rehearsals, including offline on the venue laptop · **PARTIAL**

**Evidence:** `scripts/rehearse.py`, `docs/assets/rehearsal/online.json`,
`docs/assets/rehearsal/offline.json`, `docs/rehearsal.md`.

**Done:** twenty automated rehearsals — ten online, ten with the socket layer
blocked. **All beats behaved in all twenty runs.** Offline median 116.6 s
against online 118.5 s, so the system does not need the network, measured.

**Missing, and it is the half the plan actually means:** narrated run-throughs
timed against the spoken script, on the venue laptop. A person is required.

**One finding the team must plan around:** the two real-Cartosat beats take
**≈56 s each** — essentially their whole slot. Every other beat finishes in
under 3 seconds. `docs/rehearsal.md` gives three options in preference order.

## 4.3 — Technical report · **DONE**

**Evidence:** `docs/technical-report.md`.

Architecture, all metrics, four ablations (two measured, one measured-negative,
one `not_comparable`), calibration, AURC/E-AURC, abstention, soak, and
fourteen limitations. Every number cited to the `metrics.json` or
`docs/assets/` artifact that produced it.

It leads on the CDVQA correction — 0.0000 → 0.4439 (below baseline, reported
as failure) → 0.5380 against a 0.9975 oracle — because a corrected measurement
is the strongest evidence of method the project has.

## 4.4 — PS requirement traceability matrix · **DONE**

**Evidence:** `docs/00-README-and-Requirement-Traceability.md` §3;
`docs/ps-26167.md` as the in-repo source of truth.

Refreshed with measured status per requirement, then **checked clause-by-clause
against the authoritative PS text**, which found four defects in it: two
"representative queries" that were not the PS's and one missing entirely; six
claimed deliverables where the PS states two; a controller clause with no row;
and I6 presented as a PS requirement when the PS says nothing about scene size.
All corrected. A test now asserts the PS's five queries still appear verbatim
in `ps-26167.md`, so drift becomes a test failure.

## 4.5 — Model cards + published weights · **PARTIAL**

**Evidence:** `docs/model-cards.md`.

**Done:** a card per trained component — Track A (+ Stage A2/A3), Track B,
caption, grounding, change mask, change caption, semantic change head,
optical–SAR fusion — each with training provenance, metrics, and its weakest
number stated rather than omitted.

**BLOCKED — weights are not published.** The semantic change head is trained on
**SECOND, which states no licence at all** (not a restrictive one — none), so
redistributing weights derived from it is an unresolved question. The rest
await a licence decision the team must make. The PS deliverable is *"codes and
models"*; code and tests are in the repository, weights are not.

**Unblocked by:** an email to the SECOND authors (addresses in
`docs/verification.md`) and a team decision on our own licence.

## 4.6 — Deck + recorded backup video · **PARTIAL**

**Evidence:** `docs/deck.md`.

**Done:** the deck's full content — nine slides plus backup slides, with the
words to say and the citation behind every claim. Slide 8 is "What does not
work" and lists six weaknesses with numbers.

**BLOCKED — the recorded backup video does not exist.** It needs a screen
recording of a human driving the demo with narration, on the machine that will
be used. It is insurance against a live failure and cannot be produced by the
build.

**Unblocked by:** one recording session. The bundle builds in one command and
`rehearse.py` reports per-beat timings, so the recording is a session rather
than a construction task.

## 4.7 — Code freeze · **DONE**

**Evidence:** `docs/code-freeze.md`, and the SHA below.

Defines what may still change (bug fixes, evidence, demo material), the
four-point bar a bug fix must clear, and an explicit out-of-scope list — the
CDVQA segmenter, grounding, refusal, VRSBench, `max_coreg_shift_px`, and the
router — each with why it is tempting and why not now.

## 4.8 — Ten hardest judge Q&As · **DONE**

**Evidence:** `docs/judge-qa.md`.

Ten questions with honest answers, chosen as the ones a careful reviewer
actually asks: the 7.6% grounding accuracy, the negative fusion gain, the
narrow CDVQA margin, whether the agentic layer earns itself, the BigEarthNet.txt
substitution, the confidence-above-abstention contradiction, the
co-registration claim, the RISAT question, the missing third benchmark, and
"what is most likely wrong in what you have shown me".

Plus two questions we want to be asked, where the honest answer is the
strongest one.

---

## Verification at freeze

| gate | result |
|---|---|
| Full test suite | **855 passed** |
| No-torch CI simulation | **730 passed, 18 skipped, 0 failed** |
| Illegal-plan rate | **0 / 600** |
| Matrix validation | successful |
| Demo bundle | **9 / 9 beats** |
| Rehearsals | **20 / 20 clean** (10 online, 10 offline) |
| Frontend | typechecks and builds |
| Docker | three images build; API container serves a real query |
| `pip-audit` | no known vulnerabilities |

## What Phase 4 changed about the plan's own claims

Three items in the plan turned out to be wrong when checked, and are corrected
rather than carried forward:

1. **The PS lists two deliverables, not six.** The technical report, model
   cards and this document set are team-chosen and worth having; they are not
   PS requirements. The *demonstration* is, and it did not exist until 4.1.
2. **Two of the plan's five "PS representative queries" were not the PS's**,
   and the PS's only grounding query was missing from the matrix entirely.
3. **The 7-minute script cites a fusion gain of "+14% IoU on built-up".** The
   measured gain is **−0.0064** — fusion does not beat optical alone. That
   line must not be spoken; `docs/deck.md` replaces it with the real number.

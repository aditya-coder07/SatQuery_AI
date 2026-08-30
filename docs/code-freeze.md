# Code freeze — W13

**Plan task 4.7: "Freeze code. Only bug fixes after W13."**

## What freeze means here

From the freeze commit onward, the only changes that go in are:

* **bug fixes** — something measured to be wrong is made right;
* **evidence** — a measurement, a document, a model card, a rehearsal record;
* **demo material** — inputs and scripts for the demonstration.

Not permitted: new capabilities, new tools, retraining that changes a
published number, refactors, dependency bumps that are not security fixes.

**A number in `docs/` may only change if the run that produced it is
re-executed and the new value is recorded with its date.** `phase1-status.md`
is append-only in spirit: later sections correct earlier ones, and nothing is
deleted when it turns out to be wrong. That property is the reason the CDVQA
history — 0.0000, then 0.4439, then 0.5380 — is still readable.

## State at freeze

| | |
|---|---|
| Branch | `phase-0-closeout` |
| Tests | **855 passing** |
| No-torch CI simulation | 730 passed, 18 skipped, 0 failed |
| Illegal-plan rate | **0 / 600** |
| Matrix validation | successful |
| Frontend | typechecks and builds |
| Docker | three images build; API container serves a real query |
| Dependency audit | `pip-audit`: no known vulnerabilities |
| Demo bundle | 9 / 9 beats behave as scripted |
| Working tree | clean |

Freeze commit: the annotated tag **`phase-4-freeze`**. Resolve it with
`git rev-list -n1 phase-4-freeze`; a SHA written into the file it commits
would necessarily name the previous commit.

## The bug-fix bar

A change qualifies as a bug fix if all four hold:

1. Something is **measured** to be wrong — a failing test, a wrong number, a
   defect reproduced in the browser or against the live API.
2. The fix is **scoped to that defect** and does not add capability.
3. The **full regression set** is re-run: `pytest` (855), the no-torch CI
   simulation, `evaluation/adversarial.py` for the 0/600 guarantee, matrix
   validation, and `make_demo_bundle.py --verify` if any beat could be
   affected.
4. The defect and its fix are **recorded** in `docs/00` §3.6.

Every Phase-4 fix so far met that bar. Three were found by rehearsing rather
than by testing — the PS's built-up query abstaining, the opaque mask overlay,
and the bi-temporal fixture with no change in it — which is the argument for
rehearsing at all.

## Explicitly out of scope after freeze

These are known, documented, and **must not** be started now. Each is real
work with a real regression surface, and the risk of breaking a working demo
exceeds the gain.

| Item | Why it is tempting | Why not now |
|---|---|---|
| CDVQA segmenter (0.2636 mIoU, 0.9975 ceiling) | 93% of the headroom | GPU-hours, and it changes a published number |
| Grounding (Acc@0.5 0.0762) | Weakest component | A retrain, not a fix |
| Image-conditional refusal (2/12) | An open negative result | Needs a designed ablation |
| VRSBench | Closes the third prescribed benchmark | Needs the DOTA download |
| `max_coreg_shift_px` enforcement | Completes the input gate | The estimator is unvalidated — see L16 |
| Tier-1 routing (0.5862) | Weakest measured number | Touching the router risks the 0/600 guarantee |

## The one thing that may still need code

The **recorded backup video** (task 4.6) is not produced. If recording it
exposes a defect, fixing that defect is in scope under the bar above. Nothing
else about the video requires code.

## Unfreezing

Only the team lead, and only with the reason written into this file. If a
change cannot be justified in one sentence here, it does not go in before the
finale.

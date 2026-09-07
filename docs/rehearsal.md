# Rehearsal record

**Plan task 4.2: "Rehearse the 7-minute demo ten times, including on the
actual venue laptop with networking off."**

That item has two halves and only one of them is automatable. This file
records what was measured, and states plainly what was not.

## What was measured — on the HOST, 2026-08-30

> **These are host numbers and they are no longer the ones to plan against.**
> The demo runs in a container, and one in-container pass measured **239.8 s
> against this table's 118.5 s median** - near the slow end of the observed
> host range (111.0-268.2 s), on a **single run**. See "In-container timings"
> below, which supersedes this section for pacing. This section is kept
> because it is a 20-run record and the container figures are n=1.

`scripts/rehearse.py` executes every beat of the `docs/04` §10 script through
the real controller, in the scripted order, and checks that each beat produces
what the script says it produces. Twenty rehearsals: ten online, ten with the
socket layer blocked.

| | online | offline |
|---|---|---|
| Rehearsals | 10 | 10 |
| **All beats behaved, every run** | **yes** | **yes** |
| Median total system time | 118.5 s | **116.6 s** |
| Fastest / slowest run | 111.0 s / 268.2 s | 106.0 s / 135.9 s |
| First (cold) run | 126.5 s | 135.9 s |
| Runs 2–10, median | 116.6 s | 116.3 s |

Artifacts: `docs/assets/rehearsal/online.json`, `.../offline.json`.

**Offline is not slower.** 116.6 s against 118.5 s median, and the offline
spread is *tighter* (max 135.9 s against 268.2 s). The online outlier is the
network being attempted; with sockets blocked there is nothing to wait for.
The system does not need the internet, and that is now measured rather than
claimed.

## Per-beat timings on the host, and the problem they found

| beat | median | slot | |
|---|---|---|---|
| 0:30 rejection — incompatible pair | 0.03 s | 40 s | ✅ |
| 0:30 rejection — PNG in operational mode | 0.01 s | 40 s | ✅ |
| 1:10 cross-modal flagship | 0.29 s | 70 s | ✅ |
| **2:20 single optical, real Cartosat** | **56.59 s** | 50 s | ❌ **over** |
| 2:20 single SAR, real EOS-04 | 2.32 s | 50 s | ✅ |
| 3:10 bi-temporal — what changed and where | 0.23 s | 60 s | ✅ |
| 3:10 bi-temporal — increased or decreased | 0.24 s | 60 s | ✅ |
| 4:50 abstention — clouded optical | 0.28 s | 50 s | ✅ |
| **5:40 the large scene, real Cartosat** | **55.36 s** | 60 s | ⚠ marginal |

**The finding: the two real-Cartosat beats cost about 56 seconds each.**
*(Superseded 2026-09-07 — in-container they cost **73.16 s** and **66.47 s**,
and a third beat, the cross-modal flagship, joins them at **83.35 s**. See
below.)* That
is the full 7687×7640, four-band product going through ingest, tiling and the
index engine — it is honest work, not a bug, and it is roughly the entire slot
those beats have in a seven-minute script. Together they are **112 of the
118 seconds** of system time in a rehearsal; every other beat finishes in
under three seconds.

### What to do about it, in preference order

1. **Pre-warm both Cartosat runs before the demo starts** and show the stored
   `/runs/{id}` permalinks. The GUI renders a stored run identically to a live
   one — verified in the browser — so nothing about the demo looks different.
2. **Narrate over it.** 56 seconds is enough to explain Axiom 2 (no SWIR on
   Cartosat, so NDBI is unavailable and the SAR-primary path fires) while the
   trace fills. This is the option that shows real work happening.
3. **Do not** substitute a synthetic scene for the Cartosat beat. Real
   target-sensor imagery is the most convincing thing in the demo, and the
   whole point of holding those products out of training.

The other seven beats total under 2 seconds, so the script has slack
everywhere except here.

## In-container timings — 2026-09-07. **Plan against these.**

Measured with the same `scripts/rehearse.py`, unmodified, one pass per
configuration, inside the running containers. **This is where the demo
actually runs**, so these supersede the host figures above for pacing.

**Caveat that travels with them: n=1 per configuration**, against the host
table's median of ten. Treat individual beats as indicative and the shape -
which beats dominate - as solid, because it reproduces the host's shape.

| beat | slot | **GPU (in-container)** | CPU fallback | |
|---|---|---|---|---|
| 0:30 rejection - incompatible pair | 40 s | 0.29 s | 0.31 s | ✅ |
| 0:30 rejection - PNG in operational mode | 40 s | 2.26 s | 0.03 s | ✅ |
| **1:10 cross-modal flagship** | 70 s | **83.35 s** | 0.71 s | ❌ **over by 13.4 s** |
| **2:20 single optical, real Cartosat** | 50 s | **73.16 s** | 48.48 s | ❌ **over by 23.2 s** |
| 2:20 single SAR, real EOS-04 | 50 s | 11.59 s | 7.58 s | ✅ |
| 3:10 bi-temporal - what changed and where | 60 s | 0.88 s | 0.22 s | ✅ |
| 3:10 bi-temporal - increased or decreased | 60 s | 0.91 s | 0.18 s | ✅ |
| 4:50 abstention - clouded optical | 50 s | 0.89 s | 0.56 s ⚠ **answers, does not abstain** - see `docs/00` **L36** | ✅ |
| **5:40 the large scene** | 60 s | **66.47 s** | 46.10 s | ❌ **over by 6.5 s** |
| **total system time** | | **239.80 s** | **104.17 s** | |

All nine beats produced their scripted result in both configurations.

### The slot math

| | |
|---|---|
| Demo slot | **420 s** (7 minutes) |
| System time, in-container GPU | **239.80 s** - **57% of the slot** |
| System time, on the host (for comparison) | 118.5 s - 28% |
| **Left for narration** | **180 s (3:00)** across nine beats, down from 5:02 |

**The seven-minute script still fits end to end** - 239.8 s of 420 s - but it
no longer fits *as written*, because the three beats above overrun their own
slots by **43.0 s combined**. Everything scheduled after 1:10 drifts late by
that much unless the narration below absorbs it. The other six beats total
**6.5 s**, so there is no slack to borrow from anywhere else.

Two facts worth knowing before optimising the wrong thing:

* **The Cartosat cost is raster I/O, not inference.** That beat still takes
  48.5 s with every learned tool stubbed, so roughly two thirds of it is
  reading a 7687x7640 four-band product off disk.
* **The cross-modal cost is entirely the model.** 83.35 s on GPU against
  0.71 s stubbed.

### Filler cues for the three over-budget beats

The pauses are real work and they are long enough to be uncomfortable in
silence. One talking point each, all of them things the audience should hear
anyway:

* **1:10 cross-modal flagship (83 s).** *While this runs, say:* "It is
  computing the triad - optical alone, SAR alone, and fused - as three
  separate passes, because that is the only way to measure whether fusion
  actually helps. Ours says it does not: optical 0.7778, fused 0.7714, a gain
  of minus 0.0064. We report all three rather than one fused number that
  hides it."
* **2:20 single optical, real Cartosat (73 s).** *While this runs, say:*
  "This is a real Cartosat-2S product, 7687 by 7640 pixels, four bands, going
  through ingest, tiling and the index engine. Most of this wait is reading
  the scene off disk, not the model - we measured it at 48 seconds with every
  learned tool switched off. It is the price of not substituting a synthetic
  image for the sensor you actually care about."
* **5:40 the large scene (66 s).** *While this runs, say:* "Same product
  path, and while it works: every neural claim in the answer is checked
  against NDVI, NDWI, NDBI and SAR backscatter computed deterministically
  from the pixels. The verifier has no opinion and no training - it is
  arithmetic - which is what makes a confident wrong answer catchable rather
  than merely unlikely."

**Still the better option where it is available:** pre-warm the two real-product
runs before the demo and show the stored `/runs/{id}` permalinks, as the host
section recommends. The cues above are for when a judge asks to see it run live.

## What was NOT measured, and is not done

**This is not ten rehearsals in the sense the plan means.** A rehearsal is a
person driving the demo and speaking to a clock. What ran here is the system's
half: the beats execute, in order, repeatably, within budget except where
noted. Reporting this as "task 4.2 complete" would be a false claim.

Specifically not covered:

* **Narration and timing against the spoken script.** The 7-minute budget is
  dominated by speech, not compute, and nothing here measures whether the
  words fit.
* **The actual venue laptop.** These ran on the development machine. The plan
  names the venue laptop because that is where surprises happen — different
  GPU or none, different screen, a locked-down network.
* **Recovery from interruption.** A judge asking a question mid-beat is the
  most likely live failure and cannot be simulated.
* **The GUI path end to end.** Beats were driven through the controller; a
  separate browser pass verified the run view, three-component confidence, map
  overlays and PDF link, but not the upload-and-watch-the-trace-stream flow
  ten times over.

**What the team must still do:** ten timed run-throughs with narration, at
least one on the venue laptop with its network off, and one recorded (task
4.6's backup video, also open).

## How to re-run

```bash
python scripts/make_demo_bundle.py --out data/demo_bundle --verify
python scripts/rehearse.py --runs 10 --out docs/assets/rehearsal/online.json
python scripts/rehearse.py --runs 10 --offline --out docs/assets/rehearsal/offline.json
```

The script exits non-zero if any beat misbehaves or exceeds its slot, so it is
usable as a pre-demo check on the venue machine — which is the cheapest way to
find out that the venue laptop is slower than this one.

# Rehearsal record

**Plan task 4.2: "Rehearse the 7-minute demo ten times, including on the
actual venue laptop with networking off."**

That item has two halves and only one of them is automatable. This file
records what was measured, and states plainly what was not.

## What was measured

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

## Per-beat timings, and the problem they found

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

**The finding: the two real-Cartosat beats cost about 56 seconds each.** That
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

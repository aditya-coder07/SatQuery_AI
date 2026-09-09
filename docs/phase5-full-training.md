# Phase 5 — full training of all nine tools on cluster GPU

**Written 2026-09-09.** This document is a *plan and a harness*, not a result.
No number in it is measured. When runs complete, their numbers go into a new
dated section of `docs/phase1-status.md` and new cards in `docs/model-cards.md`
— **never as an edit to a v1 number**, for the reason in §1.

---

## 1. This is post-freeze work, and how it stays legitimate

`docs/code-freeze.md` permits bug fixes, evidence and demo material, and
explicitly forbids "new capabilities, new tools, **retraining that changes a
published number**, refactors".

A full retrain of all nine tools is squarely the forbidden clause. It is being
done anyway, as a deliberate decision, because the compute constraint the whole
plan was built around no longer holds: `docs/03` §1 opens with "the team has
free-tier Colab and Kaggle only — T4 (16 GB) and occasionally P100", and every
model in the registry is small because of that sentence. College cluster GPU
access removes it.

**The freeze is honoured structurally rather than by exemption.** Four rules,
each enforced by something other than good intentions:

| rule | enforced by |
|---|---|
| No v1 checkpoint directory is written to | every `ckpt_dir` in `configs/campaign.yaml` is under `checkpoints/v2/`, asserted in `tests/test_campaign.py` |
| No v1 architecture changes | v2 lives in `training/v2/architectures.py`; `--arch` defaults to `v1`, asserted per-trainer |
| No published number is edited | v2 results are published as a new dated section; the v1 number is the baseline the v2 number is *compared against*, so overwriting it destroys the comparison |
| v1 stays loadable | checkpoints carry `arch` in `extra`; absent means v1, so all seven loadable v1 checkpoints load unchanged |

The one thing this cannot preserve is the claim that `phase-4-freeze` is the
last commit that touched training code. It is not, and this document is the
record of that.

---

## 2. Why new architectures rather than longer training

Retraining the existing models for longer would not have worked, and the
project's own measurements say why.

| tool | v1 measured | published range | cause |
|---|---|---|---|
| `landcover_v1` | mAP **0.2854** | ~0.65–0.85 | dim-64 4-layer CNN on 30k of ~590k patches |
| `grounding_v1` | Acc@0.5 **0.0762** | ~0.70–0.80 | **global-average-pools before regressing the box** |
| `change_vqa_v1` (scratch) | mIoU 0.1691 | — | 1,600 training pairs, no pretraining |

The grounding row is the clearest case. `docs/model-cards.md` already names it:
pooling the feature map to a vector discards every pixel coordinate, so the
model "can only learn an average box". That is not a training-budget problem.
Ten thousand GPU-hours on the same architecture would produce the same average
box.

So each v2 model changes what the measurement identified and nothing else. The
data loaders, losses, metrics and `--eval-only` guards are v1 code, unmodified.
**That is what makes a v2 number and a v1 number measurements of the same
thing.**

| tool | v1 params | v2 params | what changed |
|---|---|---|---|
| `landcover_v1` | 0.42 M | 5.40 M | residual trunk; per-band features survive to depth; FiLM at every stage, not once |
| `grounding_v1` | 0.55 M | 18.98 M | **no pooling**: phrase cross-attends over the feature map, box read from where attention landed; auxiliary heatmap |
| `change_mask_v1` | 0.05 M | 0.83 M | differences at every scale, decoder with concat skips |
| `change_caption_v1` | 0.32 M | 25.39 M | transformer decoder with cross-attention over the difference map |
| `change_vqa_v1` | 1.02 M | 3.79 M | multi-scale skips into two unshared per-date decoders |
| `optsar_fusion_v1` | 0.13 M | 6.26 M | cross-attention **before** pooling, both directions |
| `caption_v1` | 1.56 M | 26.17 M | transformer decoder with cross-attention |
| `rs_vqa_v1` | — | — | **unchanged architecture**; it was always real QLoRA on Qwen2.5-VL-3B |
| `index_engine_v1` | — | — | deterministic NumPy; no training run, by design |

### What is deliberately not adopted

* **No `trust_remote_code`.** `training/train_grounding.py` records the decision
  to build a fallback grounder rather than execute Python fetched from a model
  repo. A bigger GPU is not a reason to reverse a security decision, so
  Florence-2 is still out and the v2 grounder is still built from parts.
* **No `timm` / `torchgeo` pretrained weights.** A cluster notebook may have no
  outbound network, and a run that dies at `from_pretrained` after 63 GB is
  staged is the worst available failure. Stated as a trade, not as a free
  choice: on the two small corpora it is probably the wrong one, which is why
  `change_vqa` keeps `--pretrained` and the v2 arm is an ablation.

---

## 3. The environment this is built for

JupyterHub, notebook-only, on a shared cluster. That constraint drives the
whole harness, because a notebook session gets killed — idle timeout, wall
clock, closed tab, node reclaimed.

**The notebook holds no state.** `notebooks/satquery_campaign.ipynb` calls
`campaign.run_all()` and nothing else; state lives in `runs/campaign_state.json`
and in checkpoint directories that already survive a kill.

| module | what it decides |
|---|---|
| `training/cluster/env_probe.py` | precision, attention impl, batch shape — from the card actually allocated |
| `training/cluster/stage_data.py` | is the data here and intact, and which runs does that unblock |
| `training/cluster/campaign.py` | dependency order, crash recovery, session budgets |

Three design points that are not obvious, each written because the naive
version was wrong:

**Effective batch is invariant across cards.** `micro_batch` scales with VRAM
and `grad_accum` is derived from the target, so an A100 gives 4×4 and a T4
gives 1×16 — both effective 16. Without this, two runs of the same config on
two nodes would be solving different optimisation problems and their metrics
would not be comparable, which a campaign spread over whatever node is free
cannot afford.

**Budgets are advisory.** The first version refused to start a run longer than
the remaining session. Sessions are 4–12 hours and the longest run here is
estimated at 20, so it would have reported "nothing ready to run" indefinitely
and trained nothing. Every trainer checkpoints; partial progress is kept;
`strict=True` restores the refusing behaviour for a node that must be handed
back on time.

**Stale runs are reclaimed, live ones are not.** A killed session leaves a run
marked `running` forever. A heartbeat plus an owner record distinguishes "the
session died" from "it is running on another node" — the second must never be
restarted, because two processes writing one checkpoint directory corrupts it.

---

## 4. The runs

Ten runs, eight of them producing a deployed checkpoint, two of them ablation
arms. `index_engine_v1` has no run.

| run | tool | data | est. h | note |
|---|---|---|---|---|
| `track_a` | `landcover_v1` | BigEarthNet full | 14 | the encoder two tools share |
| `track_a_nodropout` | — | BigEarthNet full | 14 | **ablation**: the band-dropout claim was single-seed |
| `optsar_fusion` | `optsar_fusion_v1` | WHU-OPT-SAR | 6 | PS-mandatory |
| `track_b_vqa` | `rs_vqa_v1` | instruction mix | 20 | **recovery** — the v1 adapter is destroyed |
| `track_b_caption` | `caption_v1` | instruction mix | 10 | second adapter, same frozen base |
| `grounding` | `grounding_v1` | DIOR-RSVG | 8 | largest expected gain |
| `change_mask` | `change_mask_v1` | LEVIR-CD | 7 | cheapest real run; good first test |
| `change_caption` | `change_caption_v1` | LEVIR-MCI | 8 | |
| `change_vqa` | `change_vqa_v1` | CDVQA/SECOND | 5 | `--pretrained`, deliberately |
| `change_vqa_scratch_v2` | — | CDVQA/SECOND | 5 | **ablation**: does v2 capacity beat a pretrained stem on 1,600 pairs? |

**~97 GPU-hours estimated** for one complete pass. `docs/03` §1.2 budgeted
55–95 for the v1 pass and 150–220 with realistic iteration; the same multiplier
applies here.

`track_b_vqa` is the only run that is *recovery* rather than improvement.
`docs/model-cards.md` records its adapter as 99.99% NUL bytes — 148,701,184 of
148,712,776 — so `rs_vqa_v1` currently cannot be loaded or published at all.

### Two results that would be honest failures

Worth stating in advance, so that neither gets quietly reinterpreted afterwards:

1. **`optsar_fusion`'s fused head not beating `max(optical, sar)`.** The
   three-output shape exists to make that falsifiable. If the gain is not
   there, the cross-attention is decoration and the report says so.
2. **`change_vqa_scratch_v2` losing to `--pretrained`.** Expected, on 1,600
   pairs. It is run because the alternative is assuming it a second time.

---

## 5. Running it

```bash
# on the machine that has prepared data (63.18 GB across nine datasets)
python training/cluster/stage_data.py build --root data
rsync -a --info=progress2 data/ cluster:/scratch/$USER/data/

# on the cluster, in the notebook or a terminal
python training/cluster/env_probe.py
python training/cluster/stage_data.py verify --root /scratch/$USER/data
python training/cluster/campaign.py status
python training/cluster/campaign.py --budget-minutes 480 all
```

Then re-run the same commands after every session death. The queue resumes.

**On transferring 63 GB.** Three of the nine directories are ~88,000 small
files (`levircd`, `levir_mci`, `bigearthnet_14k`), and rsync of many small
files is dominated by per-file round trips. A tar stream is usually much
faster:

```bash
tar cf - data/levircd data/levir_mci | ssh cluster 'tar xf - -C /scratch/$USER/'
```

Verify with digests afterwards either way. `stage_data.py verify` opens and
hashes every file rather than trusting size and mtime, because a zero-filled
file keeps its size — the exact failure that produced twelve NUL-byte sidecars
in the 2026-08-30 incident and survived a verification that hashed without
opening.

---

## 6. What is not done

Stated plainly rather than left to be discovered:

* **No run has been executed.** There is no GPU here big enough and the cluster
  is yours. Every number in this repository is still a v1 number.
* **Wall-clock estimates are estimates**, extrapolated from v1 run times and
  parameter counts. They are not measured on your hardware, and `est_hours`
  only affects scheduling preference, never correctness.
* **The inference-side tools have not been re-pointed at v2 checkpoints.** They
  read `arch` from `extra` and dispatch correctly, and the plumbing is tested,
  but no `SATQUERY_*` variable points at a v2 directory because no v2 directory
  has contents. That switch is a deployment decision to make per tool once
  numbers exist — the shape of the decision is in `docs/checkpoint-decision.md`.
* **Licensing is unchanged.** `change_vqa_v1` weights remain unpublishable:
  SECOND states no licence at all. Retraining does not change that.

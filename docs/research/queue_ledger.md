# Cluster queue ledger — 2026-09-12 (night 1 → night 3)

Every job that is running or queued on `compute01` (`adi01@172.16.1.161`,
`~/satquery`), in dependency order. Nothing else of ours runs on the card.
All jobs are `systemd-run --user --unit=<unit> --same-dir --collect`
units; `MainPID` is the unit's main process (the wrapper for queued
steps; the python process is its child and appears in `nvidia-smi`).
Queue scripts are committed verbatim: `scripts/cluster_queue_night1.sh`,
`scripts/cluster_queue_night3.sh` (night 2 was stopped at 21:33 before it
ran anything, and replaced by night 3), `scripts/cluster_landcover_full.sh`,
`scripts/cluster_run_when_free.sh`.

**GPU assumptions (shared NVIDIA L40S, 46,068 MiB).** Another user
(dev01) holds 3–7 GB at any time. Every step sizes from *free* VRAM at
launch: VLM steps wait for ≥ 18 GB free (`MIN_FREE_GB`), the land-cover
run for ≥ 8 GB, the SCD run for ≥ 14 GB, single-model evals for ≥ 4–10 GB.
Only one 3B-LoRA job runs at a time (24 GB bf16). No job of another user
is ever signalled.

**Verification rule.** A job counts as finished only when all four hold:
(1) the wrapper/queue log has `EXIT 0` / `DONE <name> (exit 0)`; (2) the
checkpoint listed below exists and loads; (3) the metrics artefact exists
with the expected `n`; (4) the registry has a `done` record with the
checkpoint sha256. Status column values: RUNNING, WAITING, DONE (verified),
FAILED, STOPPED.

## Scheduling change (2026-09-13 03:10, on the user's instruction)

Jobs no longer wait for a whole "night" queue to finish. Night 3 and night
4 were stopped (both still waiting, nothing run) and replaced by one unit
per job, each of which waits only on its own gate and on free VRAM:
`scripts/cluster_unit_armC.sh` (gate: arm A verified), `cluster_unit_cm_scratch.sh`,
`cluster_unit_armB.sh`, `cluster_unit_unified.sh` (gate: the four specialist
reports, polled every 10 min). VLM units use batch 4 × accum 4 (same
effective batch) so two can share the card; whichever unit finds the VRAM
first starts, the others keep waiting. Night 1 continues unchanged.
Deployment of these units is pending the cluster link (down at 03:12).

## Jobs

| # | Unit / PID | Command (exact, see script) | Dataset manifest (sha256 prefix) | Config | Expected runtime | Checkpoint | Log | Depends on | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `sq-train-ground-lora-e1` / 816558 (started 16:36) | `python training/train_grounding_vlm.py --model models/qwen25_vl_3b --data data/dior_rsvg_official --ckpt-dir checkpoints/v3/grounding_vlm_r16 --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 --val-limit 400 --val-every 400 --save-every 200 --workers 8 --quant none --resume` | DIOR-RSVG official train `2cbc6e7e…` (26,991) / val subsample 400 | LoRA r16 α32 on LLM linears, bf16, cosine, 1 epoch = 1,686 steps | ended 22:58 (20 s/step once the land-cover run shared the card) | `checkpoints/v3/grounding_vlm_r16/{adapter_best,adapter_last,adapter_final}` | `logs/train_ground_lora_r16.log` | — | 24 GB, running |
| 2 | `sq-prep-ben-full` / 863170 (20:05) | `python training/prepare/bigearthnet_v1_full.py --stage all` | HF `lc-col/bigearthnet` (39 files, 146 GB, LFS sha256-verified) | pack → uint16 memmaps, CSV order | download+unpack done 21:00; packing ≈ 22:20 | `data/ben_v1_full/{train,val,test}_images.u16.npy` + `manifests/stats.json` | `logs/prep_ben_v1_full.log` | — | CPU/disk only |
| 3 | `sq-train-landcover-v3-full` / 866532 (20:20, waiting) | `bash scripts/cluster_landcover_full.sh` → `cluster_run_when_free.sh 8 train_landcover_v3_full python training/train_landcover_v3.py --data data/ben_v1_full --ckpt-dir checkpoints/v3/landcover_full --epochs 30 --batch-size 128 --val-limit 20000` | `data/ben_v1_full/manifests/stats.json` (269,695 / 123,723 / 125,866) | SSL4EO-S12 MoCo R50, band dropout 0.3, AdamW 1e-3 head / 1e-4 trunk, cosine, bf16, val-selected | 2,107 steps/epoch ≈ 6–8 min → ≈ 3.5–4 h + test | `checkpoints/v3/landcover_full/{best,last,final}.pt`, `metrics.json` | `logs/landcover_full_queue.log`, `logs/train_landcover_v3_full.log` | #2 (manifest must exist; refuses otherwise) | needs ≥ 8 GB free; ~4 GB used; runs beside the VLM job |
| 4 | `sq-train-scd-landsat` / 920985 (relaunched 00:50; the 17:55 wrapper died at 23:00 reading a script edited under it — see Stopped — and the first relaunch failed on 1,058 dead paths, fixed in the preparer) | `cluster_run_when_free.sh 14 train_scd_landsat python training/train_scd_landsat.py --ckpt-dir checkpoints/v3/scd_landsat --epochs 40 --batch-size 8 --workers 6` | `data/landsat_scd/index.json` `405e92c8…` (5,134 / 477 / 477; same split as `a7dd275f…`, paths resolved case-insensitively) | siamese R50 + FPN, 10-way change-type head, median-freq weights | ≈ 2–3 h once it starts | `checkpoints/v3/scd_landsat/best.pt` | `logs/train_scd_landsat.log` | free VRAM ≥ 14 GB (i.e. after #1 ends and before/around night-1 evals) | ≈ 10 GB |
| 5 | `sq-queue-night1` / 827471 (17:09, waiting on #1) step 1 `eval_ground_official` | `python evaluation/grounding_official_eval.py --base models/qwen25_vl_3b --data data/dior_rsvg_official --arms zero_shot=BASE lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best --out artifacts/benchmark_reports/dior_rsvg_official_phase6.json --batch 24` | DIOR-RSVG official test (7,500 expr; hashes recorded in report) | greedy, bf16, both arms, paired McNemar | ≈ 45–60 min | report only | `logs/eval_ground_official.log`, `logs/queue_night1.log` | #1 | ≥ 18 GB free |
| 6 | night-1 step 2 `train_vqa_official` (RUNNING since 00:34; step 3,120/3,764 at 02:55; val published-convention acc 0.930 at step 2,500 on the 1,533-item val sample) | `python training/train_vlm_sft.py --model … --train data/rsvqa_lr_official/manifests/train.jsonl data/instruct_mix_v2/manifests/train_no_rsvqa.jsonl --val … --val-limit 1000 --ckpt-dir checkpoints/v3/vqa_official --epochs 1 --batch-size 16 --grad-accum 1 --lr 1e-4 --val-every 500 --save-every 250 --workers 8 --quant none` | RSVQA-LR official train `755cd50d…` (57,223) + instruct_mix_v2 `1982a780…` (3,010) | fresh LoRA r16, 1 epoch = 3,765 steps | ≈ 3–4 h | `checkpoints/v3/vqa_official/adapter_best` | `logs/train_vqa_official.log` | #5 | ≥ 18 GB free |
| 7 | night-1 step 3 `eval_vqa_official` | `python evaluation/rsvqa_official_eval.py --base … --data data/rsvqa_lr_official --arms v3_official=checkpoints/v3/vqa_official/adapter_best v2_deployed=checkpoints/v2/track_b_vqa/adapter_final --out artifacts/benchmark_reports/rsvqa_lr_official_phase6.json` | RSVQA-LR official test (10,004 q) | 4-bit deployed path, both arms | ≈ 40 min | report | `logs/eval_vqa_official.log` | #6 | ≥ 18 GB |
| 8 | night-1 step 4 `train_caption_vlm` | `train_vlm_sft.py --train data/rsicd/manifests/train.jsonl --val …/val.jsonl --val-limit 300 --ckpt-dir checkpoints/v3/caption_vlm --epochs 1 --batch-size 16 --grad-accum 1 --lr 1e-4 --val-every 400 --save-every 200` | RSICD train `67c61c33…` (43,670 rows) | fresh LoRA r16, 2,730 steps | ≈ 3 h | `checkpoints/v3/caption_vlm/adapter_best` | `logs/train_caption_vlm.log` | #7 | ≥ 18 GB |
| 9 | night-1 step 5 `eval_caption_vlm` | `vlm_task_eval.py --manifest data/rsicd/manifests/test.jsonl --arms base=BASE caption_lora=checkpoints/v3/caption_vlm/adapter_best --out artifacts/benchmark_reports/rsicd_test_vlm.json --batch 32` | RSICD test (1,093 × 5 refs) | corpus BLEU/ROUGE-L/CIDEr-D/meteor_exact | ≈ 20 min | report | `logs/eval_caption_vlm.log` | #8 | ≥ 18 GB |
| 10 | night-1 step 6 `train_change_caption_vlm` | `train_vlm_sft.py --train data/levir_mci/manifests/train.jsonl --val …/val.jsonl --val-limit 300 --ckpt-dir checkpoints/v3/change_caption_vlm --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4` | LEVIR-CC train `2333113e…` (34,075 rows, two images) | fresh LoRA r16, 2,130 steps | ≈ 4–5 h | `checkpoints/v3/change_caption_vlm/adapter_best` | `logs/train_change_caption_vlm.log` | #9 | ≥ 18 GB |
| 11 | night-1 step 7 `eval_change_caption_vlm` | `vlm_task_eval.py --manifest data/levir_mci/manifests/test.jsonl --arms base=BASE cc_lora=… --out artifacts/benchmark_reports/levircc_test_vlm.json --batch 16` | LEVIR-CC test (1,929 × 5) | changed/unchanged halves | ≈ 30 min | report | `logs/eval_change_caption_vlm.log` | #10 | ≥ 18 GB |
| 12 | `sq-queue-night3` / 916873 (restarted 00:38 with arm C at native resolution; waiting on #5–#11) **gate** `gate_arm_a` | checks `DONE eval_ground_official (exit 0)`, `adapter_best/{adapter_config.json,adapter_model.safetensors}` (>100 tensors), report has `arms.lora_r16.acc@0.5` with n = 7,500 | — | — | seconds | — | `logs/queue_night3.log` | #5 | none; **queue exits 1 on failure, arm C never starts** |
| 13 | night-3 step A `train_ground_vrs` (arm C) | `train_vlm_sft.py --train data/dior_rsvg_official/manifests/train.jsonl data/vrsbench/manifests/train_grounding.jsonl --train-weight 0.4 1.0 --val …/val.jsonl data/vrsbench/manifests/val_grounding.jsonl --val-limit 400 --init-adapter checkpoints/v3/grounding_vlm_r16/adapter_best --ckpt-dir checkpoints/v3/grounding_vlm_vrs --epochs 1 --batch-size 8 --grad-accum 2 --lr 5e-5 --max-pixels 409600 --val-every 400 --save-every 200 --workers 8 --quant none` | DIOR-RSVG train ×0.4 (10,796) + VRSBench grounding `bc82fd58…` (33,440 clean rows) | arm-A adapter continued, native resolution (640-px cap dropped after arm A's small-object result), 2,765 steps; smoke-tested 21:45 (4-bit, 4 rows) | ≈ 10–12 h | `checkpoints/v3/grounding_vlm_vrs/adapter_best` | `logs/train_ground_vrs.log` | #12 | ≥ 18 GB |
| 14 | night-3 step A2 `eval_ground_armC` + `eval_vrsbench_ground` | official DIOR-RSVG test, arms lora_r16 vs lora_vrs (paired McNemar) → `dior_rsvg_official_armC.json`; VRSBench val grounding (16,159) base / lora_r16 / lora_vrs → `vrsbench_val_grounding.json` | DIOR-RSVG test; VRSBench `val_grounding.jsonl` | greedy | ≈ 1 h + 1.5 h | reports | `logs/eval_ground_armC.log`, `logs/eval_vrsbench_ground.log` | #13 | ≥ 18 GB |
| 15 | night-3 step B `train_cm_v3_scratch` | `train_change_mask.py --index data/levircd/index.json --ckpt-dir checkpoints/v3/change_mask_scratch --arch v3 --no-pretrained --dim 64 --epochs 40 --batch-size 16 --lr 2e-4 --loss bce_dice --augment --cosine --amp --workers 6 --select-on-val --save-every 445` | LEVIR-CD index (7,120 / 1,024 / 2,048) | ablation: same trunk from scratch | ≈ 2 h | `checkpoints/v3/change_mask_scratch/best.pt` | `logs/train_cm_v3_scratch.log` | #14 | ≈ 8 GB |
| 16 | night-3 step C `train_ground_lora_merger` (arm B) + `eval_ground_merger` | `train_grounding_vlm.py … --ckpt-dir checkpoints/v3/grounding_vlm_r16_merger --epochs 1 … --train-merger`; then official test → `dior_rsvg_official_armB.json` | DIOR-RSVG official train | LoRA r16 + trainable merger, 1,686 steps | ≈ 7 h + 1 h | `checkpoints/v3/grounding_vlm_r16_merger/adapter_best` | `logs/train_ground_lora_merger.log`, `logs/eval_ground_merger.log` | #15 | ≥ 18 GB |
| 17 | `sq-landcover-post` (queued 22:30) | `bash scripts/cluster_landcover_post.sh`: gate (EXIT 0, best/final/metrics/band_stats present, n_test 125,866) → `calibrate.py --heads landcover --ben-data data/ben_v1_full --ben-split val --ben-limit 20000 --track-a-ckpt checkpoints/v3/landcover_full/best.pt` → `assertion_threshold.py` on the calibrated val logits → `verify_deploy.py --only landcover` | official val subsample 20,000 (seed 1) | affine/temperature fit, ECE before/after; threshold at ≥0.90 precision | ≈ 10 min | none (read-only on weights) | `logs/landcover_post.log` | #3 | ≈ 3 GB (inference) |
| 18 | `sq-queue-night4` (queued 22:40, waits on #12–#16) **gate** `gate_specialists` then `train_unified_vlm` + four evals | `train_vlm_sft.py` over 8 manifests with weights 0.5 0.5 0.3 0.2 1.0 0.25 0.5 0.4 (≈100k rows/epoch: 30k grounding, 36k VQA, 20k caption, 13.6k change caption), fresh LoRA r16, 640-px cap; then DIOR-RSVG / RSVQA-LR / RSICD / LEVIR-CC official tests with the unified adapter beside each specialist | DIOR `2cbc6e7e…`, VRSBench train_{grounding,vqa,caption}, RSVQA `755cd50d…`, mix `1982a780…`, RSICD `67c61c33…`, LEVIR-CC `2333113e…` | `scripts/cluster_queue_night4.sh` | ≈ 12 h train + 3 h evals | `checkpoints/v3/unified_vlm/adapter_best` | `logs/queue_night4.log`, `logs/train_unified_vlm.log`, `logs/eval_unified_*.log` | #16 (and the four specialist reports must exist) | ≥ 18 GB |
| 19 | `sq-eval-ground-4bit` / 935244 (queued 01:52; waits for ≥ 12 GB free so it never displaces a training job) | `cluster_run_when_free.sh 12 eval_ground_4bit python evaluation/vlm_task_eval.py --base models/qwen25_vl_3b --manifest data/dior_rsvg_official/manifests/test.jsonl --arms lora_r16=checkpoints/v3/grounding_vlm_r16/adapter_best --quant 4bit --out artifacts/benchmark_reports/dior_rsvg_official_lora_r16_4bit.json --batch 16` | DIOR-RSVG official test (7,500) | deployed 4-bit load path, greedy | ≈ 1 h | report only | `logs/eval_ground_4bit.log` | free VRAM only | ≈ 6 GB |
| 20 | `sq-train-scd-landsat-e80` (queued 01:58; waits for ≥ 14 GB) | `cluster_run_when_free.sh 14 train_scd_landsat_e80 python training/train_scd_landsat.py --ckpt-dir checkpoints/v3/scd_landsat_e80 --epochs 80 --batch-size 8 --workers 6` | `data/landsat_scd/index.json` `405e92c8…` | same arm, 2× schedule — justified by val score still rising at epoch 39/40 | ≈ 1.4 h | `checkpoints/v3/scd_landsat_e80/best.pt` | `logs/train_scd_landsat_e80.log` | free VRAM (≥ 14 GB) | ≈ 10 GB |
| 21 | `sq-robust-landcover` (queued 02:05; waits for ≥ 6 GB) | `cluster_run_when_free.sh 6 robust_landcover python evaluation/robustness_landcover.py --data data/ben_v1_full --ckpt checkpoints/v3/landcover_full/best.pt --limit 20000 --batch 128 --out artifacts/benchmark_reports/ben_robustness_landcover_v3.json` | official test subsample 20,000 (seed 0) | 21 corruption conditions (noise, brightness, dihedral, per-band drop, 4-band, cloud patch); smoke-tested on 256 patches | ≈ 25 min | report only | `logs/robust_landcover.log` | free VRAM | ≈ 3 GB |

## Reboot (2026-09-14 ≈ 09:24) and relaunch

compute01 rebooted (uptime 44 min at 10:08); every user unit died. What had
finished before it is verified below; `train_change_caption_vlm` was at
step 2,040/2,129 (adapter_best = val-selected step 2,000, lr < 1e-6) and
is scored from that adapter; nothing else was running. All remaining jobs
were relaunched as per-job units at 10:12. Three VLM units passed their
VRAM checks within 30 s of each other and two OOM'd while loading;
`scripts/cluster_unit_lib.sh` now serialises launches with a lock held for
5 min after each start, and arm C / unified were relaunched through it
(`cluster_unit_armC2.sh`, `cluster_unit_unified2.sh`).

| Unit (10:12–10:20) | Command | Log | State at 10:25 |
|---|---|---|---|
| `sq-unit-cc_resume` / 30606 | `scripts/cluster_unit_cc_resume.sh` — resume train (OOM'd at load, FAILED) → `eval_change_caption_vlm` on `adapter_best` | `logs/queue_cc_resume.log` | DONE 10:43 (exit 0): `levircc_test_vlm.json` n 1,929, cc_lora BLEU-4 0.6045 / changed 0.322; adapter sha256 in `checksums.tsv`, backed up |
| `sq-unit-cm_scratch` / 31626 | `scripts/cluster_unit_cm_scratch.sh` | `logs/queue_cm_scratch.log`, `logs/train_cm_v3_scratch.log` | DONE 10:38 (exit 0): `change_mask_scratch/best.pt` test F1 0.8751 / IoU 0.7780 (val-selected epoch 32); backed up |
| `sq-unit-armB` → `sq-unit-armB2` → **`sq-unit-armB3`** (11:04, on the user's instruction to start now) | `scripts/cluster_unit_armB3.sh`: batch 2 × accum 8 (same effective batch 16), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`, MIN_FREE_GB 14, launch lock | `logs/queue_armB3.log`, `logs/train_ground_lora_merger.log` | TRAINING beside arm C since 11:04 (13.4 GB; 26 s/step → ≈ 12 h) |
| `sq-unit-armC2` / 75357 | `scripts/cluster_unit_armC2.sh` (gate OK: arm A 0.6877; batch 4×4; launch lock) | `logs/queue_armC2.log`, `logs/train_ground_vrs.log` | TRAINING since 10:32 (2,764 steps, 13.9 s/step, ≈ 24 GB) → evals |
| `sq-unit-unified2` → `unified3` → **`sq-unit-unified4`** (13:26, on the user's instruction to start now) | `scripts/cluster_unit_unified4.sh`: batch 2 × accum 8, expandable segments, eval batch 8, MIN_FREE_GB 12, launch lock; gate OK | `logs/queue_unified4.log`, `logs/train_unified_vlm.log` | TRAINING beside arms C and B since 13:26 (6,243 steps, 9.2 s/step → ≈ 16 h; 11.3 GB; card at 42.3/46 GB, no OOM) |
| `sq-verify-deploy` | `cluster_run_when_free.sh 10 verify_deploy_v3 python scripts/verify_deploy.py --map configs/deploy.v3.yaml` | `logs/verify_deploy_v3.log` | DONE 10:29 (exit 0): **8/8 tools load** through their deployed loaders (rs_vqa v3 adapter, grounding/caption/change_caption adapters on the shared base, change_mask v3, fusion v3, landcover v3, change_vqa v2) |
| 22 | `sq-robust-grounding` (queued 11:05; ≥ 8 GB) | `cluster_run_when_free.sh 8 robust_grounding python evaluation/robustness_grounding.py --base models/qwen25_vl_3b --adapter checkpoints/v3/grounding_vlm_r16/adapter_best --quant 4bit --manifest data/dior_rsvg_official/manifests/test.jsonl --limit 1000 --batch 16 --out artifacts/benchmark_reports/dior_rsvg_robustness_lora_r16.json` | DIOR-RSVG official test, 1,000-item subsample (seed 0) | 9 conditions (JPEG, blur, brightness, noise, hflip with box flipped, 4× downscale); smoke-tested on 16 | ≈ 50 min | report only | `logs/robust_grounding.log` | free VRAM | ≈ 6 GB |

## External kills (2026-09-14 14:58:36 and 15:14:40)

All three trainers (arm C at step 1,060, arm B at 600, unified at 580)
received SIGKILL in the same second at 14:58:36 — not a cgroup OOM
(`memory.events` oom_kill 0; 431 GB RAM free), not our scripts. They
were relaunched with `--resume` from `adapter_last`/`train_state.pt`
(`scripts/cluster_unit_{armC,armB,unified}_resume.sh`) at 15:04; the
resumed arm C was killed again at 15:14:40 while alone on the card. `last`
shows a root login at 10:05 and a root process `python original.py` has
been using the GPU since 15:13 — an administrator is working on the card.
This is the charter's "another user's GPU job" case: nothing of theirs is
touched, our jobs run at reduced footprint, and the user is asked to
check with the administrator. Lost: ≈ 1 h of arm C, 1 h of arm B, 15 min
of unified (the resume points are the last saves).

## Third kill and server outage (2026-09-14 18:35 → 2026-09-15 09:24)

The three resumed trainers were SIGKILLed again at 18:35:36 (arm C at step
1,600, arm B at 860, unified at 1,620); the units then attempted their
evaluations, which were killed at 18:48 / 18:52, and the node went down
until a reboot at 09:24 on the 15th (a root process `python
qvit_pipeline.py` has been on the GPU since 10:13). Relaunched 10:47 as
`sq-unit-{armC,armB,unified}-resume3` from the last saves (arm C 1,600, arm
B 800, unified 1,500). Cumulative loss to external kills: ≈ 4 GPU-hours.

## Finished today (verified)

| Job | Exit | Checkpoint | Metrics | Registry |
|---|---|---|---|---|
| #22 `sq-robust-grounding` | EXIT 0 12:53 | read-only | `dior_rsvg_robustness_lora_r16.json` n 1,000: photometric ≤ ±0.011, blur σ2 −0.098, 4× downscale −0.107, hflip −0.241 (positional phrases) | `robustness_grounding-…` done |
| night-1 #6 `train_vqa_official` | DONE (exit 0) 04:12 | `checkpoints/v3/vqa_official/adapter_best` (sha256 f3e4522d…, backed up) | val (1,533) published-convention 0.933 at step 3,500 | `vlm_sft-20260913-003408-b79298` done, 3,764 steps |
| night-1 #7 `eval_vqa_official` | DONE (exit 0) 05:28 | — | `rsvqa_lr_official_phase6.json`: n 10,004 both arms; v3 0.9119 [0.905, 0.918] vs v2 0.8947 [0.887, 0.902]; all types up | — |
| night-1 #8 `train_caption_vlm` | DONE (exit 0) 06:13 | `checkpoints/v3/caption_vlm/adapter_best` (sha256 638efb8f…, backed up) | val (300) corpus BLEU-4 0.423 | `vlm_sft-20260913-052822-1d5938` done, 2,729 steps |
| night-1 #9 `eval_caption_vlm` | DONE (exit 0) 06:26 | — | `rsicd_test_vlm.json`: n 1,093; LoRA BLEU-4 0.256 / CIDEr-D 0.793; base 0.022 | `eval_caption_caption_lora-…` |
| #19 `sq-eval-ground-4bit` | EXIT 0 04:45 | — | `dior_rsvg_official_lora_r16_4bit.json`: n 7,500, Acc@0.5 0.678 (bf16 0.6877) | eval record |
| #20 `sq-train-scd-landsat-e80` | EXIT 0 04:51 | `checkpoints/v3/scd_landsat_e80/best.pt` (sha256 b9d62e81…, backed up) | test mIoU 0.623 / SeK 0.509 / Score 0.553 (477 pairs); val 0.540 at epoch 79 | `scd_landsat-20260913-025242-ab732f` done, 51,280 steps |

## Finished 2026-09-12/13 (verified)

| Job | Exit | Checkpoint | Metrics | Registry |
|---|---|---|---|---|
| `sq-robust-landcover` (#21) | EXIT 0 (02:01) | read-only | `ben_robustness_landcover_v3.json` (n 20,000): dihedral exact, single-band drops ≤ −0.002, 4-band −0.026, cloud patch −0.078, noise σ0.3 −0.170 | `robustness_landcover-…` done |
| `sq-train-scd-landsat` (#4, relaunched 00:47) | EXIT 0 (01:27); 40 epochs, 0.67 GPU-h | `checkpoints/v3/scd_landsat/best.pt` (epoch 39; sha256 497968ad…), `last.pt` | `metrics.json` test: mIoU 0.573 / SeK 0.460 / Score 0.504 / binary F1 0.851 (477 pairs); val score 0.4922 | `scd_landsat-20260913-004701-…` done |
| `sq-train-landcover-v3-full` (#3) | EXIT 0 (00:03); 30 epochs, 63,180 steps, 2.05 GPU-h | `checkpoints/v3/landcover_full/best.pt` (epoch 13; sha256 1d775c0c…), `final.pt`, `band_stats.json` | `metrics.json`: test@best-val micro mAP 0.885 / macro 0.792 / retention 0.94 on n_test 125,866; val (20,000) micro 0.8845 | `landcover_v3-20260912-220043-7cab7d` done |
| `sq-landcover-post` (#17) | all steps exit 0 (00:04) | read-only | `configs/calibration.v3.json` SINGLE_LANDCOVER affine ECE 0.0085 → 0.0030 (n_fit/eval 10,000/10,000, official val); threshold 0.69 → precision 0.9027 / recall 0.6135 (380,000 decisions); `verify_deploy --only landcover` OK (LandcoverV3, 23.6M params) | — |
| night-1 #5 `eval_ground_official` | DONE (exit 0) 00:34; 2,318 s generate, peak 11 GB | — | `dior_rsvg_official_phase6.json`: n 7,500 both arms; lora_r16 Acc@0.5 0.6877 [0.677, 0.699]; zero_shot 0.3823; McNemar significant | `eval_grounding_lora_r16-20260913-003359-7170fe` |
| `sq-train-ground-lora-e1` (arm A) | unit completed 22:59 (7 h 17 min CPU, 11.6 G peak RAM); `done: best val` in log | `checkpoints/v3/grounding_vlm_r16/adapter_best` (step 1,600; sha256 6584fcba…), `adapter_final` (1,686), `train_state.pt` | val (400) Acc@0.5 **0.695**, Acc@0.25 0.7825, mIoU 0.608, parse 1.0 — up from 0.6475@400 | `grounding_vlm-20260912-163618-f20498` done, 1,686 steps |
| `sq-train-landcover-v3-holdout` | EXIT 0 (20:01) | `checkpoints/v3/landcover_holdout/final.pt` | `metrics.json` mAP 0.536 / micro 0.685 (n_test 15,000) | done |
| `sq-eval-tracka-v2-holdout-b8` | EXIT 0 (20:19) | v2 `checkpoints/v2/track_a` (eval-only) | `rescoring/track_a_v2_on_holdout_p3.json` 0.900 (p3 in v2's train set — not a holdout for v2) | — |
| independent LEVIR-CD re-score (foreground, 20:40) | 0 | `checkpoints/v3/change_mask/best.pt` sha256 739077ad… | `levircd_test_independent_v3.json` F1 0.9038 (n 2,048) | `eval_levircd_independent_*` |
| `sq-eval-cc-v2` | EXIT 0 (21:20) | `checkpoints/v2/change_caption` (eval-only) | `rescoring/v2cc/levircc_change_caption_v2.json` (n 1,929) | — |
| `sq-dl-vrsbench`, `sq-prep-vrsbench` | 0 / 0 (21:32) | — | `data/vrsbench/manifests/stats.json` (130,565 train / 11,825 quarantined) | — |
| arm-C smoke (foreground, 4-bit, 4 rows, `/tmp/smoke_vrs`) | 0 | discarded (under /tmp, excluded from registry) | val selection 0.5 on 8 items — pipeline check only | excluded |

## Stopped

* `sq-train-scd-landsat` (first instance, 17:55) — ended 00:00 without
  running its command: bash reads a script incrementally and
  `cluster_run_when_free.sh` was edited (exit-code fix) while this wrapper
  was still in its wait loop. Rule from now on: never edit a wrapper script
  that a waiting unit is executing; copy to a new name instead. Its second
  instance (00:43) exited 1 on `FileNotFoundError` for
  `A/From1990To2001_01Zhedang2.png`: the release spells occlusion copies
  `ZheDang` under A/ and B/ but `Zhedang` under label/; the preparer now
  resolves names against the files on disk (index sha `405e92c8…`, same
  split membership) and refuses missing files.

* `sq-queue-night2` — stopped 21:33 while still waiting on night 1; it had
  run nothing. Its steps A and C are carried into night 3 (#15, #16); its
  step B (land-cover scratch on the geographic-prefix subset) is dropped
  because that subset is no longer a benchmark (audit L1, revised).

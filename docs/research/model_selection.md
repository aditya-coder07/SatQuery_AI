# Model selection — Phase 6

Rule (prompt §28): a Phase 6 artefact replaces the deployed one for a tool
only when **all** of the following hold; otherwise the Phase 5 artefact is
held and the reason recorded in `configs/deploy.v3.yaml`.

1. **Better on the official split** of the tool's benchmark, on the
   headline metric, with a 95% CI that does not include the deployed
   value (bootstrap or Wilson), or a paired test (McNemar for hits) at
   p < 0.05.
2. **No regression on a protected number.** RSVQA-LR official test
   published-convention accuracy must not fall below 0.8947 for any change
   that touches the VQA path (a new adapter on the shared base, a change to
   the processor, a new system prompt). Grounding/caption adapters attach to
   the same base but do not alter the VQA adapter, so they cannot regress it.
3. **Loads through the real tool loader** (`scripts/verify_deploy.py`) with
   the sidecars it needs, and the golden traces still pass (or are
   regenerated with a stated reason).
4. **Calibrated or honestly uncalibrated.** A new head is either fitted in
   `configs/calibration.v3.json` (ECE reported before/after) or ships with
   `no_assertion` / `mean_asserted_probability` semantics and the
   abstention policy stays in force.
5. **Deployable within the demo budget**: single L40S/consumer GPU, cold
   start and warm latency measured (`evaluation/soak.py`), no
   `trust_remote_code`, licence recorded.

Specialist vs unified: a shared adapter replaces a task adapter only if it
is within the CI of the specialist on that task **and** better on at least
one other, measured on the same official splits. Otherwise specialists are
kept and the router (`configs/capability_matrix.yaml`) sends each task to
its own adapter on the shared base - which costs one base load and one
adapter switch per request, not a second model.

## Decisions so far (2026-09-12)

| Tool | Candidate | Deployed (Phase 5) | Decision | Evidence |
|---|---|---|---|---|
| change_mask | v3 best.pt | v2 | **select — frozen champion** | LEVIR-CD test F1 0.9038 / IoU 0.8244 independently reproduced through the deployed loader (CI [0.899, 0.908]); calibration v3 fitted on val (ECE 0.0011); val-selected threshold 0.8 → 0.9093 available as an operating point |
| optsar_fusion | v3 best.pt | v2 | **select** | first positive complementarity, CI excludes zero; v2's number was on misaligned labels |
| grounding | LoRA r16 adapter (arm A) | v2 CNN (0.160, Category C) | **select arm C** (`grounding_vlm_vrs/adapter_best`): DIOR-RSVG 0.6880 = arm A (n.s.), VRSBench val 0.6325 vs 0.501; arm B rejected; unified rejected for grounding (0.638); arm D pending | official test 0.6877 [0.677, 0.699] vs zero-shot 0.3823, McNemar p ≪ 0.001; 4-bit deployed path 0.678 (−1.0 pt, parity accepted) |
| rs_vqa | official-train SFT (`vqa_official/adapter_best`) | v2 adapter (0.8947) | **select** | 0.9119 [0.905, 0.918] vs 0.8947 [0.887, 0.902] on the official test in the 4-bit deployed path; no type regressed (rule 2 satisfied); count 0.270 vs 0.220 |
| caption | VLM caption adapter (`caption_vlm/adapter_best`) | v2 caption_pre (corpus BLEU-4 0.1744) | **select** | RSICD test corpus BLEU-4 0.256 / CIDEr-D 0.793 / ROUGE-L 0.486; tool v1.1.0 serves it from the shared base |
| change_caption | VLM two-image adapter (`change_caption_vlm/adapter_best`, step 2,000 val-selected of 2,129 — the last 89 steps were lost to the 2026-09-14 reboot, lr < 1e-6 by then) | v1 (0.384, oracle-mask) | **select** | LEVIR-CC test 0.6045 all / 0.322 changed (images only) vs v1's 0.384 / 0.222 with the GT mask; tool v1.1.0 serves it from the shared base; verify_deploy OK |
| landcover | v3 SSL4EO trunk, full official split (`landcover_full/best.pt`) | v2 (0.315 on a geographic subset) | **select** | official test micro 0.885 / macro 0.792; retention 0.94 (≥ 0.9 rule met); calibrated on val (ECE 0.0030); threshold 0.69 → precision 0.90 at recall 0.61 (v2: 0.0026); verify_deploy OK |
| change_vqa | Landsat-SCD model (`scd_landsat_e80/best.pt`, mIoU 0.623 / SeK 0.509 on the licensed benchmark; 40-ep arm 0.573 / 0.460) | v2 (SECOND) | benchmark replacement only | CDVQA tool path unchanged (its questions are SECOND's classes); SECOND licence unresolved; the Landsat-SCD model is the licensed semantic-change reference, not a drop-in for the CDVQA tool |

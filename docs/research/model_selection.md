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
| change_mask | v3 best.pt | v2 | **select** | LEVIR-CD test F1 0.9038 vs 0.855; loads; calibration queued |
| optsar_fusion | v3 best.pt | v2 | **select** | first positive complementarity, CI excludes zero; v2's number was on misaligned labels |
| grounding | LoRA r16 adapter | v2 CNN (0.160, Category C) | pending official test | zero-shot alone is 0.3823 (A) |
| rs_vqa | official-train SFT | v2 adapter (0.8947) | pending official test | rule 2 applies |
| caption | VLM caption adapter | v2 caption_pre (corpus BLEU-4 0.1744) | pending | |
| change_caption | VLM two-image adapter | v1 (0.384, oracle-mask) | pending | VLM arm sees images only - the fair comparison |
| landcover | v3 SSL4EO trunk | v2 (0.315) | pending | retention on 4 bands must stay ≥ 0.9 |
| change_vqa | Landsat-SCD model | v2 (SECOND) | benchmark replacement only | CDVQA tool path unchanged; SECOND licence unresolved |

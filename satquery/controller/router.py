from satquery.contracts.plan import Plan, PlanStep, TaskID, RationaleTag
from satquery.contracts.input_manifest import InputManifest
from satquery.controller.matrix_loader import CapabilityMatrix
from satquery.tools.stubs import REGISTRY

CONFIG_TO_LEGAL_TASKS = {
    "SINGLE": ["SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER", "CLARIFY_OR_ABSTAIN"],
    "CROSSMODAL_PAIR": ["XMODAL_JOINT_EXTRACT", "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER", "CLARIFY_OR_ABSTAIN"],
    "BITEMPORAL_PAIR": ["TEMPORAL_CHANGE_DESC", "TEMPORAL_CHANGE_VQA", "TEMPORAL_CHANGE_MAP", "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER", "CLARIFY_OR_ABSTAIN"]
}

class Router:
    def __init__(self, matrix: CapabilityMatrix):
        self.matrix = matrix

    def route(self, query: str, manifest: InputManifest) -> Plan:
        # Phase 0: Hardcode routing to SINGLE_VQA for now
        task_id: TaskID = "SINGLE_VQA"
        task_config = self.matrix.tasks[task_id]
        
        tool_name = "rs_vqa_v1"
        
        step = PlanStep(
            step_id="step_1",
            tool=tool_name,
            tool_version="0.1.0-stub",
            inputs=[],
            params={"answer_mode": "template"},
            rationale_tag=RationaleTag.VQA_INFERENCE,
            on_failure="abort"
        )
        
        legal_tasks = CONFIG_TO_LEGAL_TASKS.get(manifest.config, [])
        
        plan = Plan(
            run_id=manifest.run_id,
            legal_tasks=legal_tasks,
            tasks=[task_id],
            steps=[step],
            fallbacks=task_config.fallbacks,
            matrix_version=self.matrix.version,
            estimated_vram_mb=4000,
            estimated_runtime_ms=200
        )
        return plan

from pathlib import Path
from typing import Any

from satquery.contracts.tool_result import ToolResult, ToolPayload, Artifact
from satquery.contracts.input_manifest import InputManifest
from satquery.tools.base import ToolProtocol


class StubPayload(ToolPayload):
    data: dict[str, Any]


class RSVQAStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "answer": "fake answer string",
        })
        return ToolResult(
            tool="rs_vqa",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=0.85,
            confidence_method="threshold_rule",
            model_card="stub_vqa_model",
            runtime_ms=120,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class CaptionStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "caption": "A satellite image showing a fake landscape.",
        })
        return ToolResult(
            tool="caption",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=0.9,
            confidence_method="threshold_rule",
            model_card="stub_caption_model",
            runtime_ms=150,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class GroundingStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "bounding_boxes": [{"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
        })
        artifact = Artifact(
            key="grounding_bboxes",
            kind="geojson",
            path=Path("/fake/grounding.geojson"),
            description="Fake bounding boxes"
        )
        return ToolResult(
            tool="grounding",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[artifact],
            confidence=0.95,
            confidence_method="threshold_rule",
            model_card="stub_grounding_model",
            runtime_ms=100,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class LandcoverStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={"classes": ["forest", "water"]})
        artifact = Artifact(
            key="landcover_map",
            kind="geotiff",
            path=Path("/fake/landcover.tif"),
            description="Fake landcover map"
        )
        return ToolResult(
            tool="landcover",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[artifact],
            confidence=0.8,
            confidence_method="threshold_rule",
            model_card="stub_landcover_model",
            runtime_ms=200,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class OptSARFusionStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={"fusion_status": "success"})
        artifact = Artifact(
            key="fused_image",
            kind="geotiff",
            path=Path("/fake/optsar_fusion.tif"),
            description="Fake optical-SAR fusion"
        )
        return ToolResult(
            tool="optsar_fusion",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[artifact],
            confidence=0.88,
            confidence_method="threshold_rule",
            model_card="stub_fusion_model",
            runtime_ms=500,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class ChangeMaskStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={"changed_pixels": 420})
        artifact = Artifact(
            key="change_mask",
            kind="png",
            path=Path("/fake/change_mask.png"),
            description="Fake change mask"
        )
        return ToolResult(
            tool="change_mask",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[artifact],
            confidence=0.89,
            confidence_method="threshold_rule",
            model_card="stub_change_mask_model",
            runtime_ms=300,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class ChangeCaptionStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "caption": "A new building was constructed in the center.",
        })
        return ToolResult(
            tool="change_caption",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=0.92,
            confidence_method="threshold_rule",
            model_card="stub_change_caption_model",
            runtime_ms=180,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class ChangeVQAStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "answer": "Yes, there is a new road.",
        })
        return ToolResult(
            tool="change_vqa",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=0.87,
            confidence_method="threshold_rule",
            model_card="stub_change_vqa_model",
            runtime_ms=170,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class IndexEngineStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "NDVI": 0.65,
            "NDWI": -0.2,
        })
        return ToolResult(
            tool="index_engine",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=1.0,
            confidence_method="deterministic",
            model_card="math",
            runtime_ms=10,
            warnings=[]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


# index_engine_v1 is the first tool promoted from stub to real (plan task 1.2).
# Imported here rather than at module top to keep the stub definitions above
# free of any dependency on the real implementation.
from satquery.tools.index_engine import IndexEngine  # noqa: E402
from satquery.tools.change_vqa import ChangeVQATemplate  # noqa: E402


def _fusion_tool():
    """Real triad when a checkpoint is configured, else the stub."""
    from satquery.tools.optsar_fusion import OptSARFusionTool, is_available

    return OptSARFusionTool() if is_available()[0] else OptSARFusionStub()


def _change_mask_tool():
    """Real detector when a checkpoint is configured, else the stub."""
    from satquery.tools.change_mask import ChangeMaskTool, is_available

    return ChangeMaskTool() if is_available()[0] else ChangeMaskStub()

# rs_vqa_v1 uses the real QLoRA adapter only when SATQUERY_VQA_BASE and
# SATQUERY_VQA_ADAPTER are both set and the GPU stack is importable.
# Otherwise the stub stays, so CI and GPU-less machines keep a green suite
# rather than half-loading a model and answering badly.
def _vqa_tool():
    from satquery.tools.rs_vqa import RSVQATool, is_available

    available, _reason = is_available()
    return RSVQATool() if available else RSVQAStub()


def _landcover_tool():
    """Real Track A head when SATQUERY_LANDCOVER is set, else the stub.

    Wired with selective prediction rather than a 0.5 threshold - task 3.6
    measured that this head is worse than trivial at 0.5.
    """
    from satquery.tools.landcover import LandcoverTool, is_available

    return LandcoverTool() if is_available()[0] else LandcoverStub()


def _caption_tool():
    """Real captioner when SATQUERY_CAPTION is set, else the stub (task 2.8)."""
    from satquery.tools.caption import CaptionTool, is_available

    return CaptionTool() if is_available()[0] else CaptionStub()


def _grounding_tool():
    """Real grounder when SATQUERY_GROUNDING is set, else the stub (2.7)."""
    from satquery.tools.grounding import GroundingTool, is_available

    return GroundingTool() if is_available()[0] else GroundingStub()


def _change_caption_tool():
    """Real change captioner when SATQUERY_CHANGE_CAPTION is set (task 2.5)."""
    from satquery.tools.change_caption import ChangeCaptionTool, is_available

    return ChangeCaptionTool() if is_available()[0] else ChangeCaptionStub()


REGISTRY = {
    "rs_vqa_v1": _vqa_tool(),
    "caption_v1": _caption_tool(),
    "grounding_v1": _grounding_tool(),
    "landcover_v1": _landcover_tool(),
    "optsar_fusion_v1": _fusion_tool(),
    "change_mask_v1": _change_mask_tool(),
    "change_caption_v1": _change_caption_tool(),
    "change_vqa_v1": ChangeVQATemplate(),
    "index_engine_v1": IndexEngine(),
}

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

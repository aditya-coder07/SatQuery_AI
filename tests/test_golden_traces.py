"""Golden trace regression tests (plan task 1.11).

Ten fixed cases spanning every input configuration and the abstention path.
Each produces a trace that is normalised (volatile fields removed) and
compared byte-for-byte against a stored golden file. Any unintended change to
routing, tool selection, rationale tags, verification or confidence banding
breaks these immediately.

Regenerate deliberately after an intended change:
    pytest tests/test_golden_traces.py --update-goldens
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from satquery.controller.pipeline import Controller

GOLDEN_DIR = Path(__file__).parent / "golden_traces"

# Fields that legitimately differ between runs and carry no behavioural
# meaning. Everything else is compared exactly.
VOLATILE_KEYS = {
    "run_id", "timestamp_utc", "runtime_ms", "path", "output_dir",
}


def normalise(value):
    """Strip volatile fields so the comparison is about behaviour, not timing."""
    if isinstance(value, dict):
        return {
            k: ("<volatile>" if k in VOLATILE_KEYS else normalise(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalise(v) for v in value]
    if isinstance(value, float):
        # Guard against platform-dependent float noise in index statistics.
        return round(value, 4)
    return value


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


# (case name, query, fixture names for the input images)
CASES = [
    ("single_vqa", "How many buildings are visible?", ["msi_6band"]),
    ("single_caption", "Describe this image.", ["msi_6band"]),
    ("single_ground", "Show me where the roads are.", ["msi_6band"]),
    ("single_landcover", "Classify the land cover.", ["msi_6band"]),
    ("single_vnir_swir_free", "Classify the land cover.", ["msi_4band"]),
    ("crossmodal_fusion", "Combine the optical and radar images to find buildings.",
     ["msi_6band", "sar_dualpol"]),
    ("bitemporal_change_desc", "Describe what changed between the two images.",
     ["msi_6band", "msi_6band_t2"]),
    ("bitemporal_change_map", "Produce a change mask.",
     ["msi_6band", "msi_6band_t2"]),
    ("abstain_no_crs", "How many buildings are visible?", ["no_crs_raster"]),
    ("abstain_vague", "hmm", ["msi_6band"]),
]


@pytest.fixture(scope="module")
def controller():
    return Controller()


@pytest.mark.parametrize("name,query,fixtures", CASES, ids=[c[0] for c in CASES])
def test_golden_trace(name, query, fixtures, controller, request, tmp_path, pytestconfig):
    paths = [request.getfixturevalue(f) for f in fixtures]
    trace = controller.run(
        paths,
        query,
        run_id="fixed_run_id",
        tool_params={},
    )
    actual = normalise(json.loads(trace.model_dump_json()))

    path = golden_path(name)
    if pytestconfig.getoption("--update-goldens"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True), encoding="utf-8")
        pytest.skip(f"golden {name} regenerated")

    if not path.exists():
        pytest.fail(
            f"missing golden for {name}; regenerate with --update-goldens"
        )

    expected = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected, (
        f"trace for {name} changed. If intended, rerun with --update-goldens."
    )


class TestGoldenCoverage:
    def test_all_configs_covered(self):
        counts = {len(f) for _, _, f in CASES}
        assert counts == {1, 2}, "goldens must cover single and pair inputs"

    def test_abstention_covered(self):
        assert any(name.startswith("abstain") for name, _, _ in CASES)

    def test_ten_cases(self):
        assert len(CASES) == 10

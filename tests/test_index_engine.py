"""End-to-end tests for the real `index_engine_v1` tool (plan task 1.2)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from satquery.ingest import ingest
from satquery.tools.index_engine import IndexEngine
from satquery.tools.stubs import REGISTRY


@pytest.fixture
def engine():
    return IndexEngine()


def run_on(engine, paths, tmp_path, **params):
    manifest = ingest(paths)
    params.setdefault("output_dir", str(tmp_path / "artifacts"))
    return manifest, engine.run(manifest, params)


class TestRegistryWiring:
    def test_registry_uses_the_real_engine_not_the_stub(self):
        assert isinstance(REGISTRY["index_engine_v1"], IndexEngine)

    def test_reports_deterministic_confidence(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        assert result.confidence == 1.0
        assert result.confidence_method == "deterministic"
        assert result.version == "1.0.0"


class TestSixBandOptical:
    def test_computes_all_four_optical_indices(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        indices = result.payload.data["indices"]
        assert {"ndvi", "ndwi", "mndwi", "ndbi"} <= set(indices)

    def test_every_index_has_stats_and_a_threshold(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        for name, entry in result.payload.data["indices"].items():
            if name == "cov":
                continue
            assert "stats" in entry, name
            assert "threshold" in entry, name
            assert entry["threshold_method"] in {"otsu", "gmm", "fixed_prior"}

    def test_index_values_within_physical_range(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        for name in ("ndvi", "ndwi", "mndwi", "ndbi"):
            stats = result.payload.data["indices"][name]["stats"]
            assert stats["min"] >= -1.0
            assert stats["max"] <= 1.0

    def test_no_substitution_needed_when_swir_present(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        subs = " ".join(result.payload.data["substitutions"])
        assert "MNDWI unavailable" not in subs
        assert "NDBI unavailable" not in subs


class TestSwirFreePath:
    """A 4-band VNIR product - the assumed Cartosat-2S MX case."""

    def test_swir_indices_absent(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        indices = result.payload.data["indices"]
        assert "mndwi" not in indices
        assert "ndbi" not in indices

    def test_vnir_indices_still_computed(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        assert {"ndvi", "ndwi"} <= set(result.payload.data["indices"])

    def test_builtup_proxy_substituted_for_ndbi(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        assert "builtup_proxy" in result.payload.data["indices"]

    def test_substitutions_are_named_explicitly(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        subs = " ".join(result.payload.data["substitutions"])
        assert "MNDWI unavailable" in subs
        assert "NDBI unavailable" in subs

    def test_proxy_lowers_reliability_via_warning(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        assert any("SWIR-free proxy" in w for w in result.warnings)


class TestSAR:
    def test_sigma0_and_ratio_computed(self, engine, sar_dualpol, tmp_path):
        _, result = run_on(engine, [sar_dualpol], tmp_path)
        indices = result.payload.data["indices"]
        assert "sigma0_vv" in indices
        assert "vh_vv_ratio_db" in indices

    def test_cov_computed_for_sar(self, engine, sar_dualpol, tmp_path):
        _, result = run_on(engine, [sar_dualpol], tmp_path)
        assert "cov" in result.payload.data["indices"]

    def test_cross_pol_ratio_is_negative_for_typical_backscatter(
        self, engine, sar_dualpol, tmp_path
    ):
        """VH is weaker than VV for most surfaces, so the ratio is below 0 dB."""
        _, result = run_on(engine, [sar_dualpol], tmp_path)
        mean = result.payload.data["indices"]["vh_vv_ratio_db"]["stats"]["mean"]
        assert mean < 0


class TestCrossModal:
    def test_optical_and_sar_indices_both_present(
        self, engine, msi_4band, sar_dualpol, tmp_path
    ):
        _, result = run_on(engine, [msi_4band, sar_dualpol], tmp_path)
        indices = result.payload.data["indices"]
        assert "ndvi" in indices          # from optical
        assert "sigma0_vv" in indices     # from SAR

    def test_sar_strengthens_the_swir_free_builtup_proxy(
        self, engine, msi_4band, sar_dualpol, tmp_path
    ):
        """With SAR available the proxy must say it used the sigma0 term."""
        _, result = run_on(engine, [msi_4band, sar_dualpol], tmp_path)
        subs = " ".join(result.payload.data["substitutions"])
        assert "sar_sigma0" in subs


class TestArtifacts:
    def test_writes_readable_cogs(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        assert result.artifacts
        for art in result.artifacts:
            assert art.path.exists(), art.key
            with rasterio.open(art.path) as src:
                assert src.count == 1
                assert src.dtypes[0] == "float32"

    def test_cogs_are_georeferenced_to_the_source(self, engine, msi_6band, tmp_path):
        manifest, result = run_on(engine, [msi_6band], tmp_path)
        ndvi_art = next(a for a in result.artifacts if a.key == "ndvi")
        with rasterio.open(ndvi_art.path) as out, rasterio.open(msi_6band) as src:
            assert out.crs == src.crs
            assert out.transform == src.transform
            assert (out.height, out.width) == (src.height, src.width)

    def test_cog_driver_used(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        with rasterio.open(result.artifacts[0].path) as src:
            assert src.driver in {"COG", "GTiff"}  # COG reads back as GTiff

    def test_artifact_values_match_recomputed_index(self, engine, msi_6band, tmp_path):
        """The written raster must actually contain the index, not a placeholder."""
        from satquery.ingest.reader import read_canonical_band
        from satquery.verify import ndvi as ndvi_fn

        manifest, result = run_on(engine, [msi_6band], tmp_path)
        art = next(a for a in result.artifacts if a.key == "ndvi")
        with rasterio.open(art.path) as src:
            written = src.read(1)
        img = manifest.images[0]
        expected = ndvi_fn(
            read_canonical_band(img, "RED"), read_canonical_band(img, "NIR")
        )
        assert np.allclose(written, expected.astype("float32"), equal_nan=True)

    def test_write_artifacts_can_be_disabled(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path, write_artifacts=False)
        assert result.artifacts == []
        assert result.payload.data["indices"]  # stats still computed


class TestDegradation:
    def test_pan_only_input_computes_no_spectral_indices(
        self, engine, pan_1band, tmp_path
    ):
        _, result = run_on(engine, [pan_1band], tmp_path)
        indices = result.payload.data["indices"]
        assert "ndvi" not in indices
        assert any("no indices computable" in w for w in result.warnings)

    def test_glcm_still_computed_for_pan(self, engine, pan_1band, tmp_path):
        """Texture needs only one band, so it must survive where indices cannot."""
        _, result = run_on(engine, [pan_1band], tmp_path)
        assert result.payload.data["glcm"]

    def test_does_not_raise_on_degenerate_input(self, engine, tiny_raster, tmp_path):
        manifest = ingest([tiny_raster])
        result = engine.run(manifest, {"output_dir": str(tmp_path)})
        assert result.tool == "index_engine"

    def test_batch_matches_individual_runs(self, engine, msi_6band, tmp_path):
        manifest = ingest([msi_6band])
        params = {"output_dir": str(tmp_path / "a"), "write_artifacts": False}
        batch = engine.run_batch([manifest, manifest], params)
        assert len(batch) == 2
        assert (
            batch[0].payload.data["indices"]["ndvi"]["stats"]
            == batch[1].payload.data["indices"]["ndvi"]["stats"]
        )

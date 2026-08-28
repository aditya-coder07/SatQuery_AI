"""API, SSE streaming and run-store tests (plan task 1.5)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from satquery.api import main as api_main
from satquery.api.store import RunStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client with an isolated on-disk run store."""
    store = RunStore(tmp_path / "runs.db")
    monkeypatch.setattr(api_main, "_store", store)
    with TestClient(api_main.app) as c:
        yield c
    store.close()


def upload(path, name="image.tif"):
    return ("images", (name, path.read_bytes(), "image/tiff"))


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestSynchronousRun:
    def test_single_image_run_returns_trace(self, client, msi_6band):
        r = client.post(
            "/runs",
            data={"query": "Describe this image."},
            files=[upload(msi_6band)],
        )
        assert r.status_code == 200
        trace = r.json()
        assert trace["query"] == "Describe this image."
        assert trace["routing"]["selected_task"]
        assert trace["confidence"]["band"] in {"HIGH", "MEDIUM", "LOW"}

    def test_trace_contains_real_ingest_detail(self, client, msi_6band):
        r = client.post(
            "/runs", data={"query": "Classify the land cover."},
            files=[upload(msi_6band)],
        )
        ingest = r.json()["ingest"]
        assert ingest["config"] == "SINGLE"
        assert ingest["images"][0]["modality"] == "MSI"
        assert ingest["index_availability"]["ndvi"] is True

    def test_pair_run(self, client, msi_6band, sar_dualpol):
        r = client.post(
            "/runs",
            data={"query": "Combine the optical and radar images."},
            files=[upload(msi_6band, "opt.tif"), upload(sar_dualpol, "sar.tif")],
        )
        assert r.status_code == 200
        assert r.json()["ingest"]["config"] == "CROSSMODAL_PAIR"

    def test_rejects_zero_images(self, client):
        r = client.post("/runs", data={"query": "hi"}, files=[])
        assert r.status_code in (400, 422)

    def test_rejects_three_images(self, client, msi_6band):
        r = client.post(
            "/runs", data={"query": "hi"},
            files=[upload(msi_6band, f"{i}.tif") for i in range(3)],
        )
        assert r.status_code == 400

    def test_abstains_on_bad_input(self, client, no_crs_raster):
        r = client.post(
            "/runs", data={"query": "How many buildings?"},
            files=[upload(no_crs_raster, "nocrs.tif")],
        )
        assert r.status_code == 200
        trace = r.json()
        assert trace["abstained"] is True
        assert "crs_present" in trace["abstain_reason"]

    def test_traversal_filename_cannot_escape_upload_dir(self, client, msi_6band):
        """A malicious filename must not write outside the work directory."""
        r = client.post(
            "/runs", data={"query": "Describe this image."},
            files=[("images", ("../../evil.tif", msi_6band.read_bytes(), "image/tiff"))],
        )
        assert r.status_code == 200
        for img in r.json()["ingest"]["images"]:
            assert ".." not in img["path"]


class TestSSEStreaming:
    def _events(self, response) -> list[tuple[str, dict]]:
        events, name = [], None
        for line in response.text.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: ") and name:
                events.append((name, json.loads(line[len("data: "):])))
                name = None
        return events

    def test_stream_emits_stages_in_order(self, client, msi_6band):
        r = client.post(
            "/runs/stream", data={"query": "Classify the land cover."},
            files=[upload(msi_6band)],
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        names = [n for n, _ in self._events(r)]
        assert names[0] == "run_started"
        assert names[-1] == "complete"
        # Ingest and routing are known before any tool runs, so they must
        # precede the first step event.
        assert names.index("ingest") < names.index("step")
        assert names.index("routing") < names.index("step")
        for stage in ("verification", "confidence"):
            assert stage in names

    def test_stream_step_events_carry_tool_output(self, client, msi_6band):
        r = client.post(
            "/runs/stream", data={"query": "Classify the land cover."},
            files=[upload(msi_6band)],
        )
        steps = [d for n, d in self._events(r) if n == "step"]
        assert steps
        assert all("tool" in s and "runtime_ms" in s for s in steps)

    def test_stream_final_event_matches_stored_trace(self, client, msi_6band):
        r = client.post(
            "/runs/stream", data={"query": "Describe this image."},
            files=[upload(msi_6band)],
        )
        complete = [d for n, d in self._events(r) if n == "complete"][0]
        stored = client.get(f"/runs/{complete['run_id']}").json()
        assert stored["status"] == "complete"
        assert stored["trace"]["answer"] == complete["answer"]

    def test_stream_is_valid_json_throughout(self, client, msi_4band):
        """Every data line must parse - NaN would break this."""
        r = client.post(
            "/runs/stream", data={"query": "Classify the land cover."},
            files=[upload(msi_4band)],
        )
        events = self._events(r)
        assert events  # parsing above would have raised on invalid JSON


class TestRunStore:
    def test_run_persisted_and_retrievable(self, client, msi_6band):
        r = client.post(
            "/runs", data={"query": "Describe this image."},
            files=[upload(msi_6band)],
        )
        run_id = r.json()["run_id"]
        stored = client.get(f"/runs/{run_id}")
        assert stored.status_code == 200
        assert stored.json()["query"] == "Describe this image."

    def test_missing_run_is_404(self, client):
        assert client.get("/runs/does_not_exist").status_code == 404

    def test_list_runs(self, client, msi_6band):
        for i in range(3):
            client.post(
                "/runs", data={"query": f"query {i}"}, files=[upload(msi_6band)]
            )
        runs = client.get("/runs").json()["runs"]
        assert len(runs) >= 3

    def test_list_respects_limit(self, client, msi_6band):
        for i in range(3):
            client.post("/runs", data={"query": f"q{i}"}, files=[upload(msi_6band)])
        assert len(client.get("/runs?limit=2").json()["runs"]) == 2

    def test_store_roundtrips_trace(self, tmp_path, msi_6band):
        from satquery.controller.pipeline import Controller

        store = RunStore(tmp_path / "x.db")
        trace = Controller().run([msi_6band], "Describe this image.")
        store.create(trace.run_id, trace.query)
        store.complete(trace.run_id, trace)
        record = store.get(trace.run_id)
        assert record["trace"]["run_id"] == trace.run_id
        assert record["confidence"] == pytest.approx(trace.confidence.final)
        store.close()

    def test_failed_run_recorded(self, tmp_path):
        store = RunStore(tmp_path / "y.db")
        store.create("r1", "q")
        store.fail("r1", "boom")
        record = store.get("r1")
        assert record["status"] == "failed"
        assert record["error"] == "boom"
        store.close()

"""FastAPI application with SSE trace streaming (plan task 1.5).

Endpoints:
    GET  /health              liveness
    POST /runs                run the pipeline, return the full trace
    POST /runs/stream         run the pipeline, stream the trace live as SSE
    GET  /runs                recent runs
    GET  /runs/{run_id}       one stored run including its trace

The streaming endpoint runs the pipeline on a worker thread and drains its
events through a queue, so the client sees ingest, routing and each tool step
as they happen rather than waiting for the whole run.
"""

from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Iterator

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from satquery.contracts.trace import Trace
from satquery.controller.pipeline import Controller
from satquery.api.store import RunStore

app = FastAPI(title="SatQuery AI API", version="0.2.0-phase1")

# The frontend is served from a different origin (:3000) than the API (:8000),
# so the browser blocks fetch() without an explicit CORS allowance. Origins are
# configurable and default to local development only - never "*", which would
# also have to disable credentials and is wrong for a deployed system.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "SATQUERY_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Built once: the controller fits the intent classifier on construction, and
# refitting per request would dominate latency.
_controller: Controller | None = None
_store: RunStore | None = None
_lock = threading.Lock()

MAX_IMAGES = 2

# Per-file upload cap. Without one, `shutil.copyfileobj` writes whatever the
# client sends, so a single request can fill the disk - and neither Starlette
# nor uvicorn imposes a default. 256 MB comfortably clears a full Cartosat-2E
# scene (7687x7640 px, 4 bands, uint16 ~ 470 MB uncompressed but far less as a
# compressed GeoTIFF) while bounding the damage.
MAX_UPLOAD_BYTES = int(os.getenv("SATQUERY_MAX_UPLOAD_BYTES", 256 * 1024 * 1024))
UPLOAD_CHUNK = 1024 * 1024

# How many completed run directories to keep. Each holds the uploaded rasters
# and the artifacts the run produced, and nothing deleted them: a demo session
# grew the temp directory without bound and kept user-supplied imagery on disk
# indefinitely. They cannot be deleted immediately - /runs/{id}/preview and
# /runs/{id}/report.pdf both read from them - so the fix is bounded retention,
# oldest evicted first.
MAX_RETAINED_RUNS = int(os.getenv("SATQUERY_MAX_RETAINED_RUNS", 20))
# Sentinel that closes the SSE stream from the worker thread.
_DONE = object()


def get_controller() -> Controller:
    global _controller
    if _controller is None:
        with _lock:
            if _controller is None:
                _controller = Controller()
    return _controller


def get_store() -> RunStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = RunStore()
    return _store


def _save_uploads(files: list[UploadFile], dest: Path) -> list[Path]:
    """Persist uploads under `dest`, using sanitised names and a size cap."""
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, upload in enumerate(files):
        # Never trust a client-supplied filename for a path: take the basename
        # only, so "../../etc/passwd" cannot escape the upload directory.
        safe = Path(upload.filename or f"image_{i}").name
        target = dest / f"{i:02d}_{safe}"
        written = 0
        # Copied in chunks with a running total rather than in one call: the
        # point is to stop BEFORE the disk fills, so the limit has to be
        # enforced during the write, not checked afterwards.
        with target.open("wb") as fh:
            while chunk := upload.file.read(UPLOAD_CHUNK):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    fh.close()
                    shutil.rmtree(dest, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"{safe} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} "
                        f"MB per-file upload limit",
                    )
                fh.write(chunk)
        paths.append(target)
    return paths


def _prune_run_dirs(keep: int = MAX_RETAINED_RUNS) -> None:
    """Delete the oldest run directories beyond `keep`.

    Best-effort and never fatal: failing to reclaim disk must not fail the run
    that just succeeded. A directory still being read by an in-flight preview
    request will refuse to delete on Windows, which is the correct outcome -
    it is simply retried on the next run.
    """
    root = Path(tempfile.gettempdir())
    try:
        dirs = sorted(
            # Both prefixes: /runs creates satquery_run_*, and
            # /runs/{id}/report.pdf creates satquery_report_*. The second was
            # leaking too.
            (
                p for p in root.iterdir()
                if p.is_dir() and p.name.startswith(("satquery_run_", "satquery_report_"))
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in dirs[keep:]:
        shutil.rmtree(stale, ignore_errors=True)


def _sse(event: str, data: dict | list) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _client_safe_error(exc: Exception) -> str:
    """A message safe to send to a client.

    Raw exception text from rasterio/GDAL embeds absolute server paths, which
    discloses the filesystem layout to anyone who can upload a bad file. The
    full detail is still recorded in the run store for operators; only the
    outward-facing string is reduced.
    """
    name = type(exc).__name__
    if isinstance(exc, (ValueError, KeyError)):
        # These carry our own validation messages, which are safe and useful.
        return str(exc).replace(str(Path.cwd()), ".")
    return f"the input could not be processed ({name})"


@app.get("/health")
def health_check():
    return {"status": "ok", "version": app.version}


@app.post("/runs")
async def create_run(
    query: str = Form(...),
    images: list[UploadFile] = File(...),
):
    """Run the pipeline synchronously and return the complete trace."""
    if not 1 <= len(images) <= MAX_IMAGES:
        raise HTTPException(400, f"expected 1 to {MAX_IMAGES} images, got {len(images)}")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    work_dir = Path(tempfile.mkdtemp(prefix=f"satquery_{run_id}_"))
    store = get_store()
    store.create(run_id, query)

    try:
        paths = _save_uploads(images, work_dir)
        trace = get_controller().run(paths, query, run_id=run_id)
    except HTTPException as exc:
        # A deliberate status - 413 for an oversized upload - must survive.
        # The blanket handler below was converting it to a 500, which told the
        # client "server error" for what is squarely a client error.
        store.fail(run_id, exc.detail)
        raise
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        store.fail(run_id, str(exc))  # full detail retained server-side
        raise HTTPException(500, _client_safe_error(exc)) from exc

    store.complete(run_id, trace)
    _prune_run_dirs()
    return json.loads(trace.model_dump_json())


@app.post("/runs/stream")
async def stream_run(
    query: str = Form(...),
    images: list[UploadFile] = File(...),
):
    """Run the pipeline and stream trace stages as server-sent events."""
    if not 1 <= len(images) <= MAX_IMAGES:
        raise HTTPException(400, f"expected 1 to {MAX_IMAGES} images, got {len(images)}")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    work_dir = Path(tempfile.mkdtemp(prefix=f"satquery_{run_id}_"))
    paths = _save_uploads(images, work_dir)

    store = get_store()
    store.create(run_id, query)
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            from satquery.ingest import ingest as ingest_fn

            controller = get_controller()
            manifest = ingest_fn(paths, run_id=run_id)
            # Through the controller, not around it. Reaching into
            # router/executor here duplicated the pipeline and the copies
            # drifted - config_excluded was never passed, so the streamed
            # answer silently lost the task 3.8 exclusion notice.
            trace = controller.run_on_manifest(
                manifest,
                query,
                on_event=lambda name, data: events.put((name, data)),
            )
            store.complete(run_id, trace)
            events.put(("complete", json.loads(trace.model_dump_json())))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            store.fail(run_id, str(exc))  # full detail retained server-side
            events.put(
                ("error", {"run_id": run_id, "message": _client_safe_error(exc)})
            )
        finally:
            events.put(_DONE)
            _prune_run_dirs()

    threading.Thread(target=worker, daemon=True).start()

    def generate() -> Iterator[str]:
        yield _sse("run_started", {"run_id": run_id, "query": query})
        while True:
            item = events.get()
            if item is _DONE:
                break
            name, data = item
            yield _sse(name, data)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/runs/{run_id}/preview/{role}")
def get_preview(run_id: str, role: str, max_edge: int = 512):
    """Render one of a run's input images as a PNG.

    Browsers cannot display GeoTIFF, and the bi-temporal swipe and
    optical-SAR blend comparators (task 2.12) need something they can draw.
    This reuses the same `to_rgb_preview` the VQA tool feeds the model, so
    what the user compares is what the model saw - band selection and
    stretch included, rather than a separately-tuned display rendering that
    could look better or worse than the actual input.
    """
    from satquery.ingest.reader import read_image
    from satquery.tools.imaging import to_rgb_preview

    record = get_store().get(run_id)
    if record is None or not record.get("trace"):
        raise HTTPException(404, f"no such run: {run_id}")

    images = record["trace"]["ingest"]["images"]
    match = next((i for i in images if i.get("role") == role), None)
    if match is None:
        raise HTTPException(
            404,
            f"no image with role {role!r}; available: "
            f"{[i.get('role') for i in images]}",
        )

    path = Path(match["path"])
    if not path.exists():
        raise HTTPException(410, "the source image is no longer on disk")

    try:
        image, _ = to_rgb_preview(read_image(path), max_edge=max(64, min(max_edge, 2048)))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, _client_safe_error(exc)) from exc

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/models")
def get_model_registry():
    """Model registry page data (task 3.12).

    Read-only and assembled from what the training runs wrote. Registered
    before /runs/{run_id} would matter only if the paths collided; they do
    not, but the ordering is kept explicit anyway.
    """
    from satquery.report.registry import model_registry

    return model_registry()


@app.get("/benchmarks")
def get_benchmarks():
    """Benchmark page data (task 3.12): every Phase 3 measurement."""
    from satquery.report.registry import benchmarks

    return benchmarks()


@app.get("/runs/{run_id}/report.pdf")
def get_run_report(run_id: str):
    """Render a completed run to a PDF (task 3.12)."""
    from fastapi.responses import FileResponse

    from satquery.report.pdf_report import export_pdf

    record = get_store().get(run_id)
    if record is None or not record.get("trace"):
        raise HTTPException(404, f"no completed run {run_id}")

    trace = Trace.model_validate(
        record["trace"] if isinstance(record["trace"], dict)
        else json.loads(record["trace"])
    )
    out_dir = Path(tempfile.mkdtemp(prefix=f"satquery_report_{run_id}_"))
    try:
        pdf = export_pdf(trace, out_dir / f"{run_id}.pdf")
    except RuntimeError as exc:
        # reportlab is optional; say so rather than returning a 500.
        raise HTTPException(503, str(exc)) from exc
    return FileResponse(
        pdf, media_type="application/pdf", filename=f"{run_id}.pdf"
    )


@app.get("/runs")
def list_runs(limit: int = 50):
    return {"runs": get_store().list(limit=min(max(limit, 1), 500))}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    record = get_store().get(run_id)
    if record is None:
        raise HTTPException(404, f"no such run: {run_id}")
    return record

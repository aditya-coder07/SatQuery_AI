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
from fastapi.responses import StreamingResponse

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
    """Persist uploads under `dest`, using sanitised names."""
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, upload in enumerate(files):
        # Never trust a client-supplied filename for a path: take the basename
        # only, so "../../etc/passwd" cannot escape the upload directory.
        safe = Path(upload.filename or f"image_{i}").name
        target = dest / f"{i:02d}_{safe}"
        with target.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        paths.append(target)
    return paths


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
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        store.fail(run_id, str(exc))  # full detail retained server-side
        raise HTTPException(500, _client_safe_error(exc)) from exc

    store.complete(run_id, trace)
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
            plan = controller.router.route(query, manifest)
            prediction = (
                None
                if manifest.blocking_failures
                else getattr(controller.router, "last_prediction", None)
            )
            trace = controller.executor.execute(
                plan, manifest, query, prediction=prediction,
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


@app.get("/runs")
def list_runs(limit: int = 50):
    return {"runs": get_store().list(limit=min(max(limit, 1), 500))}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    record = get_store().get(run_id)
    if record is None:
        raise HTTPException(404, f"no such run: {run_id}")
    return record

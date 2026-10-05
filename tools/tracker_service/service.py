"""Standalone synoptics object-tracking service (loopback only).

Owns the tracker models and the per-run track state machine. It knows nothing about guidance, providers,
sessions, cookies or rate limits: those live in the application. Every request carries the ``run_id`` it
belongs to, so GPU state is held **per run** and work for an abandoned run can never mutate an active one.

Endpoints (private to the app; the app owns the public ``/api/track/*`` contract):

* ``GET  /health``      readiness + pinned artefacts + configuration + work counters
* ``POST /runs/start``  ``{run_id, target}``                          (register only, no frame)
* ``POST /runs/frame``  ``{run_id, frame_seq, frame_b64, seed_box?}``  (seed_box: first frame only)
* ``POST /runs/stop``   ``{run_id}``

State machine (frozen):

* ``acquiring`` -- no identity yet.
* ``tracking``  -- a usable mask this frame; ``box`` is present.
* ``occluded``  -- the model's own presence head reports the target not observable (movement alone is never
  treated as loss); ``box`` is absent, the previous box is never re-shown.
* ``lost``      -- not observable beyond ``occluded_max_frames``. Nothing is re-acquired silently: the run
  stays lost with no box until the caller opens a new run whose first frame carries a ``seed_box``. A new
  identity acquired after a loss (the only path that exists for it) bumps ``generation`` and issues a **new**
  ``track_id``; a boxless frame never claims a new identity.
* ``unavailable`` is the app's state for an unreachable/misconfigured service, never produced here.

``version`` increases on every accepted frame. ``confidence`` is ``sigmoid(object_score_logits)`` and is
**uncalibrated**: a ranking/display score, never a probability.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from models import BOX_THRESHOLD, OCCLUDED_MAX_FRAMES, SAM2_COMPILE, TRACK_KEEP_SCORE  # noqa: E402

MAX_FRAME_BYTES = 4 * 1024 * 1024
#: Tracker steps after the warm-up acquire (compile only happens on the first encoder call; a few more steps
#: also bring the memory-attention path to its steady shapes).
WARMUP_STEPS = 3


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class StartBody(_Body):
    """Register a run. A run's first frame (and its optional user seed) travels on ``/runs/frame``."""

    run_id: str = Field(min_length=1, max_length=64)
    target: str = Field(min_length=1, max_length=40)
    #: True when the caller is OPENING a new run under this id (rather than retrying a registration): any
    #: state this service still holds under the id -- for example from an earlier application session that
    #: reused the id -- is dropped instead of being inherited by the new run.
    fresh: bool = False


class FrameBody(_Body):
    run_id: str = Field(min_length=1, max_length=64)
    frame_seq: int = Field(ge=0)
    frame_b64: str = Field(min_length=1)
    #: The user's manual selection on THIS frame, normalised 0..1 in the uploaded image. Accepted on the
    #: first frame of the run only (``409 seed_not_first_frame`` otherwise).
    seed_box: dict[str, float] | None = None


class StopBody(_Body):
    run_id: str = Field(min_length=1, max_length=64)


@dataclass(slots=True)
class RunState:
    run_id: str
    target: str
    track_id: str = ""
    generation: int = 0
    version: int = 0
    state: str = "acquiring"
    transition: str = "none"
    source: str | None = None
    last_seq: int = -1
    occluded_run: int = 0
    had_lock: bool = False
    seed_box: dict[str, float] | None = None
    session: Any = None
    frames_seen: int = 0
    #: Set when the run is retired (stopped, or its application session ended). Admission is gone; an
    #: in-flight step still finishes with the session reference it already holds, and its answer is
    #: discarded by the caller's run fence.
    retired: bool = False
    last_box: dict[str, float] | None = None
    last_score: float | None = None
    created: float = field(default_factory=time.monotonic)


class Runtime:
    """Holds the models and the runs, and serialises GPU work."""

    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.sam2: Any = None
        self.grounder: Any = None
        self.processor: Any = None
        self.meta: dict[str, Any] = {}
        #: Exact work counters, per class. A run that has lost its target keeps answering without running
        #: any model, so an HTTP answer count is never an inference count.
        self.inferences: dict[str, int] = {"sam2_steps": 0, "grounder_calls": 0, "frames_without_model_work": 0}
        self.runs: dict[str, RunState] = {}
        self.gpu_lock = asyncio.Lock()
        self.clock = time.monotonic()
        # Two-GPU placement (frozen design): GPU 0 acquires/re-acquires, GPU 1 tracks. The tracker is the
        # per-frame workload; the grounder runs only on cold start and on explicit user re-selection, so
        # GPU 0 is busy intermittently while GPU 1 carries the steady state. No simultaneous-overlap claim.
        self.sam2_device = os.environ.get("TRACKER_SAM2_DEVICE", "cuda:1")
        self.grounder_device = os.environ.get("TRACKER_GROUNDER_DEVICE", "cuda:0")

    async def load(self) -> None:
        import torch

        from models import POLICY, load_grounding, load_sam2

        try:
            self.sam2 = await asyncio.to_thread(load_sam2, self.sam2_device)
            self.processor, self.grounder = await asyncio.to_thread(load_grounding, self.grounder_device)
            warmup = await asyncio.to_thread(self._warm_up) if SAM2_COMPILE != "off" else None
            self.meta = {
                "sam2": {**getattr(self.sam2, "_tracker_meta", {}), "placed_on": self.sam2_device},
                "grounder": {**getattr(self.grounder, "_tracker_meta", {}), "placed_on": self.grounder_device},
                "policy": POLICY,
                "warmup": warmup,
                "placement": {
                    "sam2_parameter_device": str(next(self.sam2.parameters()).device),
                    "grounder_parameter_device": str(next(self.grounder.parameters()).device),
                    "visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
                    "device_count": torch.cuda.device_count(),
                    "cuda_memory_allocated_mib": {
                        i: round(torch.cuda.memory_allocated(i) / 2**20, 1)
                        for i in range(torch.cuda.device_count())
                    },
                },
            }
            self.ready = True
            self.error = None
        except Exception as exc:  # fail closed: readiness stays false and /health carries the reason
            self.error = f"{type(exc).__name__}: {exc}"[:300]
            self.ready = False

    def _warm_up(self) -> dict[str, Any]:
        """Pay the compile before readiness: one grounder call, one acquire and a few steps on a synthetic frame.

        The throwaway session is never registered as a run and the work counters are not touched, so /health
        still counts only real requests. Readiness waits for this, so no user frame ever carries the compile.
        """
        import torch
        from PIL import ImageDraw

        from models import CausalSam2Session, ground

        image = Image.new("RGB", (1280, 720), (128, 128, 128))
        ImageDraw.Draw(image).rectangle([520, 260, 760, 460], fill=(30, 90, 200))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        raw = buf.getvalue()
        started = time.monotonic()
        ground(self.processor, self.grounder, image, "object")
        grounder_ms = (time.monotonic() - started) * 1e3
        session = CausalSam2Session(self.sam2)
        with _tracker_autocast():
            first = session.acquire(raw, {"x": 0.4, "y": 0.35, "width": 0.2, "height": 0.3})
            steps = [session.step(raw).infer_ms for _ in range(WARMUP_STEPS)]
        torch.cuda.synchronize()
        return {
            "total_ms": round((time.monotonic() - started) * 1e3, 1),
            "grounder_ms": round(grounder_ms, 1),
            "first_sam2_ms": round(first.infer_ms, 1),
            "step_ms": [round(ms, 1) for ms in steps],
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        }


runtime = Runtime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load both pinned models before serving; a load failure leaves readiness false with the reason."""
    await runtime.load()
    yield


app = FastAPI(title="synoptics tracker", docs_url=None, redoc_url=None, lifespan=lifespan)


def _decode(data_b64: str) -> bytes:
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="frame_not_base64")
    if not raw or len(raw) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=422, detail="frame_size")
    try:
        Image.open(io.BytesIO(raw)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="frame_not_image")
    return raw


def _clamp_box(box: dict[str, float]) -> dict[str, float]:
    x = min(max(box["x"], 0.0), 1.0)
    y = min(max(box["y"], 0.0), 1.0)
    w = min(max(box["width"], 1e-6), 1.0 - x)
    h = min(max(box["height"], 1e-6), 1.0 - y)
    return {"x": round(x, 6), "y": round(y, 6), "width": round(w, 6), "height": round(h, 6)}


def _state_body(run: RunState) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "target": run.target,
        "track_id": run.track_id,
        "generation": run.generation,
        "state": run.state,
        "transition": run.transition,
        "box": run.last_box if run.state == "tracking" else None,
        "source": run.source,
        "frame_seq": run.last_seq,
        "version": run.version,
        "confidence": run.last_score,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ready": runtime.ready and runtime.error is None,
        "error": runtime.error,
        "device": runtime.meta.get("sam2", {}).get("device"),
        "uptime_s": round(time.monotonic() - runtime.clock, 1),
        "runs": len(runtime.runs),
        "inferences": dict(runtime.inferences),
        "cuda_max_memory_reserved_mib": _peak_reserved_mib(),
        **runtime.meta,
    }


def _peak_reserved_mib() -> dict[int, float]:
    """Peak reserved CUDA memory per visible device since start (this process only)."""
    import torch

    if not torch.cuda.is_available():
        return {}
    return {i: round(torch.cuda.max_memory_reserved(i) / 2**20, 1) for i in range(torch.cuda.device_count())}


@app.post("/runs/start")
async def start(body: StartBody) -> dict[str, Any]:
    """Register a run. Idempotent for the same run id; the model work starts at its first frame."""
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="tracker_not_ready")
    existing = runtime.runs.get(body.run_id)
    if existing is not None and not body.fresh:
        return {"run_id": existing.run_id, "target": existing.target, "active": True}
    if existing is not None:
        # Retire it without touching a step that may be running: that step keeps the reference it holds and
        # its answer is discarded, because the run is no longer registered.
        existing.retired = True
    run = RunState(run_id=body.run_id, target=body.target,
                   track_id=f"t-{uuid.uuid4().hex[:12]}")
    runtime.runs[body.run_id] = run
    return {"run_id": run.run_id, "target": run.target, "active": True}


@app.post("/runs/frame")
async def frame(body: FrameBody) -> dict[str, Any]:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="tracker_not_ready")
    # Registry lookup is the admission fence: a retired run is no longer registered.
    run = runtime.runs.get(body.run_id)
    if run is None:
        raise HTTPException(status_code=409, detail="stale_run")
    seed = dict(body.seed_box) if body.seed_box is not None else None
    return await _process_frame(run, body.frame_seq, _decode(body.frame_b64), seed)


@app.post("/runs/stop")
async def stop(body: StopBody) -> dict[str, Any]:
    """Retire a run's admission first, then release its GPU state.

    Removing it from the registry is the fence: no new frame can join, and a queued frame is refused under
    the GPU lock. A step already running keeps the session reference it holds and finishes normally — its
    answer is discarded by the caller's run fence — and only then, under that same lock, is the state
    dropped and the cache released. So a stop never mutates a session out from under the GPU thread, and the
    caller never waits on the release.
    """
    run = runtime.runs.pop(body.run_id, None)
    if run is None:
        raise HTTPException(status_code=409, detail="not_active_run")
    run.retired = True
    async with runtime.gpu_lock:
        run.session = None
        run.last_box = None
        await asyncio.to_thread(_empty_cache)
    return {"run_id": run.run_id, "stopped": True}


def _empty_cache() -> None:
    import torch

    torch.cuda.empty_cache()


def _tracker_autocast():
    """Every tracker step runs under BF16 autocast — the measured P2 configuration, one place, no switch.

    The grounder is deliberately NOT wrapped: GroundingDINO acquisition and re-acquisition stay FP32, exactly
    as measured (the SAM 3 grounder applies its own BF16 autocast inside ``models.ground``).
    """
    import torch

    return torch.autocast("cuda", dtype=torch.bfloat16)


async def _process_frame(run: RunState, seq: int, raw: bytes,
                         seed_box: dict[str, float] | None = None) -> dict[str, Any]:
    async with runtime.gpu_lock:
        # Admission is decided under the lock: a run retired while this frame waited must not start GPU
        # work, and a duplicate sequence must not be processed twice.
        if run.retired:
            raise HTTPException(status_code=409, detail="stale_run")
        if seq <= run.last_seq:
            raise HTTPException(status_code=409, detail="stale_frame")
        if seed_box is not None:
            if run.frames_seen:
                raise HTTPException(status_code=409, detail="seed_not_first_frame")
            run.seed_box = dict(seed_box)
        result = await asyncio.to_thread(_advance, run, seq, raw)
    return result


def _advance(run: RunState, seq: int, raw: bytes) -> dict[str, Any]:
    """The whole track step, on the GPU thread. Synchronous by design: one frame in, one state out."""
    from models import CausalSam2Session, distinct_instances, ground

    started = time.monotonic()
    run.frames_seen += 1
    run.last_seq = seq

    if run.session is None:
        # ---- acquisition -----------------------------------------------------------------
        # Cold start may lock automatically. After a real LOSS nothing is re-acquired silently: the run
        # stays `lost` with no box until the user re-selects (a NEW run whose first frame carries a
        # seed_box). Transient occlusion is recovered through SAM's own memory while it stays inside the
        # occlusion window, so no reset happens for that case.
        if run.seed_box is not None:
            box, source = _clamp_box(run.seed_box), "user"
        elif run.had_lock:
            # The target is gone for this run: no grounder, no SAM step — the answer costs transport only.
            runtime.inferences["frames_without_model_work"] += 1
            run.state, run.transition = "lost", "none"
            run.version += 1
            run.last_box, run.last_score = None, None
            return _with_ms(_state_body(run), started)
        else:
            runtime.inferences["grounder_calls"] += 1
            candidates = ground(runtime.processor, runtime.grounder, _pil(raw), run.target)
            if distinct_instances(candidates):
                run.state, run.transition, run.source = "acquiring", "ambiguous", None
                run.version += 1
                run.last_box, run.last_score = None, None
                return _with_ms(_state_body(run), started)
            best = candidates[0] if candidates else None
            if best is None or best.score < BOX_THRESHOLD:
                run.state, run.transition = "acquiring", "none"
                run.version += 1
                run.last_box, run.last_score = None, None
                return _with_ms(_state_body(run), started)
            box, source = best.box_norm_xywh, "grounder"

        session = CausalSam2Session(runtime.sam2)
        runtime.inferences["sam2_steps"] += 1
        with _tracker_autocast():
            res = session.acquire(raw, box)
        run.seed_box = None
        run.session = session
        run.source = source
        run.occluded_run = 0
        if res.box_norm_xywh is None:
            run.state, run.transition = "acquiring", "none"
            run.last_box, run.last_score = None, None
        else:
            run.state = "tracking"
            if run.had_lock:
                # A new identity acquired after a loss. The frozen contract ties the generation bump and
                # the new track id to this event and to nothing else, so a boxless frame never claims a
                # new identity. Under the declared lost policy this path is only reachable through a
                # deliberate re-acquisition.
                run.generation += 1
                run.track_id = f"t-{uuid.uuid4().hex[:12]}"
                run.transition = "reacquired"
            else:
                run.transition = "acquired"
            run.last_box, run.last_score = res.box_norm_xywh, round(res.score, 4)
            run.had_lock = True
        run.version += 1
        return _with_ms(_state_body(run), started)

    # ---- steady state --------------------------------------------------------------------
    runtime.inferences["sam2_steps"] += 1
    with _tracker_autocast():
        res = run.session.step(raw)
    observable = res.box_norm_xywh is not None and res.score >= TRACK_KEEP_SCORE
    if observable:
        run.state, run.transition = "tracking", "tracking"
        run.occluded_run = 0
        run.last_box, run.last_score = res.box_norm_xywh, round(res.score, 4)
        run.had_lock = True
    else:
        run.occluded_run += 1
        run.last_box, run.last_score = None, round(res.score, 4)
        if run.occluded_run <= OCCLUDED_MAX_FRAMES:
            run.state, run.transition = "occluded", "occluded"
        else:
            run.state, run.transition = "lost", "lost"
            run.session = None                      # drop the whole session: no memory reuse after loss
            run.occluded_run = 0
    run.version += 1
    return _with_ms(_state_body(run), started)


def _pil(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.load()
    return image


def _with_ms(body: dict[str, Any], started: float) -> dict[str, Any]:
    body["compute_ms"] = round((time.monotonic() - started) * 1e3, 2)
    return body


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "service:app",
        host=os.environ.get("TRACKER_HOST", "127.0.0.1"),
        port=int(os.environ.get("TRACKER_PORT", "8090")),
        log_level=os.environ.get("TRACKER_LOG_LEVEL", "info"),
        workers=1,
    )


if __name__ == "__main__":
    main()

"""Standalone object tracking: the app-side admission, fencing and transport for the local tracker service.

The models do not run in this process. They run in a separate local service reached over loopback, and this
module owns exactly the parts the application must own:

* **admission and fencing** — ``start_seq`` per application session, ``frame_seq`` per run, and the active
  run identity, so a delayed start or a duplicate frame can never move the active run;
* **transport** — one HTTP call per frame, forwarded verbatim (the service decides track state);
* **contract mapping** — the service's answer is projected onto the frozen ``/api/track/*`` wire, and a
  malformed or non-conforming answer fails closed instead of reaching the client.

It holds no GPU state and no model state: the service keeps per-run state, so the work of an abandoned run
can never mutate the active run's state. No provider (paid) credential, key or slot is involved anywhere in
this path, and no frame leaves the machine — the endpoint is validated as loopback.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.app.errors import ApiFailure, StageConfigError
from backend.app.loopback import loopback_endpoint
from backend.app.tracking_contracts import (
    TrackControlResponse,
    TrackFrameResponse,
)

#: Where the standalone tracker service listens. Loopback-only, and **required**: an unset variable disables
#: the lane rather than probing an ambient service, so a test run or a deployment that never configured a
#: tracker can never reach one by accident. The follower's ``AISW_FOLLOW_LOCAL_URL`` follows the same rule.
ENV_TRACKER_URL = "AISW_TRACKER_URL"

#: How long a readiness probe result is reused, so ``/api/health`` polling cannot hammer the service.
HEALTH_TTL_S = 2.0
HEALTH_TIMEOUT_S = 2.0
CONTROL_TIMEOUT_S = 15.0
#: The first frame of a run performs acquisition (grounder plus target encoding) and is the slow step; a
#: steady frame is tens of milliseconds. This is a fail-closed ceiling, not an expected duration.
FRAME_TIMEOUT_S = 30.0
#: Abandoning a run when its application session goes away must never delay the session path.
RELEASE_TIMEOUT_S = 2.0


def tracker_endpoint() -> str:
    """The configured tracker base URL, or a named configuration failure.

    ``AISW_TRACKER_URL`` is a **prerequisite**, like the other local-stage endpoints: with it unset the lane
    is disabled and nothing is probed, so an unconfigured process cannot reach a tracker that happens to be
    running on the machine. A configured value must be loopback.
    """
    configured = os.getenv(ENV_TRACKER_URL, "").strip()
    if not configured:
        raise StageConfigError(f"{ENV_TRACKER_URL} is not set; the tracker lane is disabled")
    return loopback_endpoint(configured, ENV_TRACKER_URL)


def _service_error(response: httpx.Response) -> str | None:
    """The service's own closed-set error token, when it sent one."""
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"][:64]
    return None


class TrackerClient:
    """Loopback transport to the tracker service. Every failure is a closed-set API failure."""

    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client if client is not None else httpx.AsyncClient()
        self._owns_client = client is None
        self._probe: tuple[float, bool, str | None] = (0.0, False, None)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def ready(self) -> tuple[bool, str | None]:
        """Short-cached readiness plus the reason it is not ready. Never raises.

        Readiness is about the tracker service alone: it is reported independently of paid provider
        readiness in both directions, and an unreachable service is simply not ready.
        """
        now = time.monotonic()
        seen, ready, error = self._probe
        if seen and now - seen < HEALTH_TTL_S:
            return ready, error
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=HEALTH_TIMEOUT_S)
            payload = response.json() if response.status_code == 200 else {}
            if not isinstance(payload, dict):
                payload = {}
            ready = bool(payload.get("ready")) and response.status_code == 200
            error = None if ready else payload.get("error") or f"health_status_{response.status_code}"
        except (httpx.HTTPError, ValueError):
            ready, error = False, "tracker_unreachable"
        self._probe = (now, ready, None if error is None else str(error)[:120])
        return ready, self._probe[2]

    async def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            response = await self.client.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        except httpx.TimeoutException:
            raise ApiFailure(504, "tracker_timeout") from None
        except httpx.HTTPError:
            raise ApiFailure(503, "tracker_unavailable") from None
        if response.status_code == 409:
            # The service fences its own runs too. The application fences first, so reaching this means the
            # two disagree; the service's token is authoritative and is passed through unchanged.
            raise ApiFailure(409, _service_error(response) or "stale_run")
        if response.status_code != 200:
            raise ApiFailure(503, "tracker_unavailable")
        try:
            body = response.json()
        except ValueError:
            raise ApiFailure(503, "tracker_unavailable") from None
        if not isinstance(body, dict):
            raise ApiFailure(503, "tracker_unavailable")
        return body

    async def start(self, run_id: str, target: str, *, fresh: bool = False) -> dict[str, Any]:
        """Register a run. ``fresh`` claims a NEW run under this id: the service drops any state it still
        holds under it (a recycled id from an earlier application session must never be inherited)."""
        return await self._post(
            "/runs/start", {"run_id": run_id, "target": target, "fresh": fresh}, CONTROL_TIMEOUT_S
        )

    async def stop(self, run_id: str) -> dict[str, Any]:
        return await self._post("/runs/stop", {"run_id": run_id}, CONTROL_TIMEOUT_S)

    async def frame(
        self,
        run_id: str,
        frame_seq: int,
        image_base64: str,
        seed_box: dict[str, float] | None,
    ) -> tuple[dict[str, Any], float]:
        """Forward one frame. Returns the service's answer and the admission->response elapsed time."""
        payload: dict[str, Any] = {
            "run_id": run_id,
            "frame_seq": frame_seq,
            "frame_b64": image_base64,
        }
        if seed_box is not None:
            payload["seed_box"] = seed_box
        started = time.monotonic()
        body = await self._post("/runs/frame", payload, FRAME_TIMEOUT_S)
        return body, (time.monotonic() - started) * 1e3


def build_tracker_client(client: httpx.AsyncClient | None = None) -> TrackerClient:
    """Build the tracker transport from the environment. Raises ``StageConfigError`` when misconfigured."""
    return TrackerClient(tracker_endpoint(), client=client)


@dataclass(slots=True)
class TrackRun:
    """One run of the active session. The service owns the model; this owns identity and fencing."""

    run_id: str
    target: str
    start_seq: int
    last_frame_seq: int | None = None
    answered: int = 0
    version: int = -1
    active: bool = True
    #: Set once the service has been told to drop this run's state, so a release is attempted at most once.
    released: bool = False
    #: Serialises the frames of THIS run only. It is never taken by control, so stopping or replacing a run
    #: never waits for a frame that is in flight; that frame's answer is rejected by the run fence instead.
    inflight: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class TrackerRunRegistry:
    """Per-application-session run admission. Every rejection is a 409 carrying the expected value.

    Every mutation here is synchronous and therefore atomic with respect to the event loop: no operation
    spans an ``await``, which is what lets control retire a run while one of its frames is still in flight.
    """

    latest_start_seq: int | None = None
    run: TrackRun | None = None
    #: Every run this session closed, for the session's lifetime, so a stopped or superseded run_id can
    #: never be restarted (no window, no eviction).
    closed: set[str] = field(default_factory=set)

    @property
    def active_run_id(self) -> str | None:
        run = self.run
        return run.run_id if run is not None and run.active else None

    def run_for(self, run_id: str) -> TrackRun:
        """Resolve the run a frame claims to belong to, without changing any state."""
        run = self.run
        if run is None or not run.active or run.run_id != run_id:
            raise ApiFailure(409, "stale_run", extra={"active_run_id": self.active_run_id})
        return run

    def apply_start(self, *, run_id: str, start_seq: int, target: str) -> tuple[TrackRun, TrackRun | None, bool]:
        """Admit a start, or reject it as stale. A new run replaces the previous one.

        ``target`` is the caller's own non-blank text (the route refuses anything else before this point);
        there is no default, because a default would be a class name nobody chose.

        Returns the new run, the run it retired (its GPU state must be released) and whether this start
        OPENED a run rather than re-asserting an existing one. Only an open claims a fresh registration at
        the service, so an id recycled from another session cannot inherit its state.
        """
        run = self.run
        if run is not None and run.active and run.run_id == run_id:
            # Idempotent only for the same active run, the same start_seq AND the same target.
            if self.latest_start_seq == start_seq and run.target == target:
                return run, None, False
            raise ApiFailure(409, "stale_start", extra={"latest_start_seq": self.latest_start_seq})
        if run_id in self.closed:
            raise ApiFailure(409, "stale_start", extra={"latest_start_seq": self.latest_start_seq})
        if self.latest_start_seq is not None and start_seq <= self.latest_start_seq:
            raise ApiFailure(409, "stale_start", extra={"latest_start_seq": self.latest_start_seq})
        retired = None
        if run is not None and run.active:
            run.active = False
            self.closed.add(run.run_id)
            retired = run
        self.latest_start_seq = start_seq
        self.run = TrackRun(run_id=run_id, target=target, start_seq=start_seq)
        return self.run, retired, True

    def apply_stop(self, *, run_id: str) -> TrackRun:
        """Close the active run. Stopping anything else says so explicitly."""
        run = self.run
        if run is None or not run.active or run.run_id != run_id:
            raise ApiFailure(409, "not_active_run", extra={"active_run_id": self.active_run_id})
        run.active = False
        self.closed.add(run_id)
        return run

    def admit_frame(self, run: TrackRun, *, frame_seq: int, has_seed: bool) -> None:
        """Decide a frame against the run it was resolved to. Called with the run's in-flight lock held."""
        if self.run is not run or not run.active:
            # Retired or replaced while this frame waited its turn.
            raise ApiFailure(409, "stale_run", extra={"active_run_id": self.active_run_id})
        if has_seed and run.answered:
            raise ApiFailure(409, "seed_not_first_frame")
        if run.last_frame_seq is not None and frame_seq <= run.last_frame_seq:
            raise ApiFailure(409, "stale_frame", extra={"expected_seq_after": run.last_frame_seq})

    def accepted(self, run: TrackRun, *, frame_seq: int, version: int) -> None:
        """Record an answered frame. Called only after the service answered it and the fence passed."""
        if version <= run.version:
            # The frozen wire requires a version that strictly increases per accepted frame. A service
            # answer that does not is not usable, so it fails closed rather than being renumbered here.
            raise ApiFailure(503, "tracker_unavailable")
        run.last_frame_seq = frame_seq
        run.answered += 1
        run.version = version


def project_frame(
    body: dict[str, Any],
    *,
    run: TrackRun,
    frame_id: str,
    frame_seq: int,
    ingest_age_ms: int,
) -> TrackFrameResponse:
    """Project the service's answer onto the frozen frame contract, failing closed on anything else.

    The service's echoed identity must be the frame this request asked about: a stale answer (a different
    run, a different sequence, or a different target) is never stamped onto the current frame.
    """
    if body.get("run_id") != run.run_id or body.get("frame_seq") != frame_seq or body.get("target") != run.target:
        raise ApiFailure(503, "tracker_unavailable")
    try:
        return TrackFrameResponse(
            run_id=run.run_id,
            target=str(body["target"]),
            track_id=str(body["track_id"]),
            generation=body["generation"],
            state=body["state"],
            transition=body["transition"],
            box=body.get("box"),
            source=body.get("source"),
            frame_id=frame_id,
            frame_seq=frame_seq,
            version=body["version"],
            confidence=body.get("confidence"),
            ingest_age_ms=max(0, ingest_age_ms),
        )
    except (KeyError, TypeError, ValueError):
        raise ApiFailure(503, "tracker_unavailable") from None


def project_control(run: TrackRun) -> TrackControlResponse:
    return TrackControlResponse(run_id=run.run_id, active=run.active, target=run.target)


async def release_run(run: TrackRun | None, tracker: TrackerClient) -> None:
    """Abandon a captured run at the service, best effort and bounded. Idempotent per run.

    The run is already closed in this application wherever this is called: it only releases the service's
    per-run state (a tracking run holds the tracker's frame memory). It must never delay the caller, so the
    wait is bounded here and every failure is swallowed — the local fence is what makes the run inert.
    """
    if run is None or run.released:
        return
    run.released = True
    try:
        await asyncio.wait_for(tracker.stop(run.run_id), timeout=RELEASE_TIMEOUT_S)
    except (ApiFailure, StageConfigError, asyncio.TimeoutError, OSError):
        pass

"""Standalone tracking API tests: admission, fencing, isolation, and the tracker-ready session gate.

Asserted at the observable boundary only: a request is accepted with a specific body, or rejected with a
specific status and error code, and the frame the tracker is asked about is the frame the client named. The
tracker service is replaced by an in-process fake that speaks the same wire, so nothing here calls a model.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from backend.app.main import app, store
from backend.app.provider import GeminiProvider
from backend.app.tracking import TrackerClient
from backend.app.tracking_contracts import MANUAL_TARGET

TRACKER_URL = "http://127.0.0.1:8090"
ORIGIN = {"Origin": "http://127.0.0.1:8000"}
#: The target every start in this module names explicitly. The application has no implicit default any
#: more, so a start always has to carry a target the caller chose.
TARGET = "laptop"


def make_test_jpeg(width: int = 640, height: int = 480, color: str = "blue") -> str:
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


class NoProvider:
    """A deployment with no paid provider key at all: only the local tracker lane exists."""

    provider_name = "none"
    api_key = ""
    model = "none"


class FakeTracker:
    """A stand-in tracker service: the same wire, recording what the application sent it."""

    def __init__(self, *, ready: bool = True, answer: dict[str, Any] | None = None, status: int = 200,
                 transport_error: Exception | None = None) -> None:
        self.ready = ready
        self.answer = answer
        self.status = status
        self.transport_error = transport_error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.runs: dict[str, dict[str, Any]] = {}
        #: When set, /runs/frame waits on it before answering, so a test can hold a frame in flight.
        self.frame_gate: asyncio.Event | None = None

    # -- wire -----------------------------------------------------
    async def handler(self, request: httpx.Request) -> httpx.Response:
        if self.transport_error is not None:
            raise self.transport_error
        payload: dict[str, Any] = json.loads(request.content) if request.content else {}
        self.calls.append((request.url.path, payload))
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"ready": self.ready, "error": None if self.ready else "sam2_pin_mismatch"})
        if path in ("/runs/start", "/runs/frame") and not self.ready:
            return httpx.Response(503, json={"error": "tracker_not_ready"})
        if path == "/runs/start":
            existing = self.runs.get(payload["run_id"])
            if existing is not None and not payload.get("fresh"):
                return httpx.Response(200, json={"run_id": payload["run_id"], "target": existing["target"],
                                                 "active": True})
            self.runs[payload["run_id"]] = {"frames": 0, "version": 0, "target": payload["target"]}
            return httpx.Response(200, json={"run_id": payload["run_id"], "target": payload["target"], "active": True})
        if path == "/runs/stop":
            if self.runs.pop(payload["run_id"], None) is None:
                return httpx.Response(409, json={"error": "not_active_run"})
            return httpx.Response(200, json={"run_id": payload["run_id"], "stopped": True})
        if path == "/runs/frame":
            run = self.runs.get(payload["run_id"])
            if run is None:
                return httpx.Response(409, json={"error": "stale_run"})
            if self.frame_gate is not None:
                await self.frame_gate.wait()
            run["frames"] += 1
            run["version"] += 1
            return httpx.Response(self.status, json=dict(self.answer or self.answer_for(payload, run)))
        raise AssertionError(f"unexpected tracker path {path}")

    def answer_for(self, payload: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
        first = run["frames"] == 1
        return {
            "run_id": payload["run_id"],
            "target": run["target"],
            "track_id": "t-1",
            "generation": 0,
            "state": "tracking",
            "transition": "acquired" if first else "tracking",
            "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1},
            "source": "user" if payload.get("seed_box") else "grounder",
            "frame_seq": payload["frame_seq"],
            "version": run["version"],
            "confidence": 0.87,
        }

    def paths(self) -> list[str]:
        return [path for path, _payload in self.calls]

    def frames(self) -> list[dict[str, Any]]:
        return [payload for path, payload in self.calls if path == "/runs/frame"]


def install(fake: FakeTracker) -> None:
    """Point the application's tracker transport at a fake service, with a fresh readiness cache."""
    store.tracker_client = TrackerClient(
        TRACKER_URL, client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    )
    store.tracker_error = None


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch):
    """A fake tracker service and a provider with no key, so only the tracker lane can open a session."""
    monkeypatch.setenv("DEMO_ACCESS_CODE", "test-code")
    monkeypatch.delenv("AISW_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("AISW_TRACKER_URL", raising=False)
    fake = FakeTracker()
    previous = (store.provider, store.tracker_client, store.tracker_error)
    store.provider = NoProvider()
    install(fake)
    store.sessions.clear()
    store.creations.clear()
    try:
        yield fake
    finally:
        store.provider, store.tracker_client, store.tracker_error = previous
        store.sessions.clear()
        store.creations.clear()


def api_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000")


async def open_session(api: httpx.AsyncClient, code: str = "test-code") -> tuple[dict[str, str], str]:
    response = await api.post("/api/session", json={"access_code": code}, headers=ORIGIN)
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].split(";", 1)[0]
    headers = {**ORIGIN, "Cookie": cookie}
    return headers, response.json()["session_id"]


def control_body(session_id: str, run_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "run_id": run_id,
        "action": "start",
        "start_seq": 1,
        "target": TARGET,
        **extra,
    }


def frame_body(session_id: str, run_id: str, seq: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "run_id": run_id,
        "frame_id": f"cam-{seq}",
        "frame_seq": seq,
        "image_base64": make_test_jpeg(),
        **extra,
    }


async def start_run(api: httpx.AsyncClient, headers: dict[str, str], session_id: str, run_id: str,
                    start_seq: int = 1, **extra: Any) -> httpx.Response:
    return await api.post("/api/track/control", json=control_body(session_id, run_id, start_seq=start_seq, **extra),
                          headers=headers)


def test_session_opens_with_a_tracker_but_no_paid_provider(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            health = await api.get("/api/health")
            assert health.status_code == 200
            assert health.json()["ready"] is False
            assert health.json()["tracker_ready"] is True

            headers, session_id = await open_session(api)
            assert session_id
            started = await start_run(api, headers, session_id, "run-1")
            assert started.status_code == 200
            assert started.json() == {"run_id": "run-1", "active": True, "target": TARGET}

    asyncio.run(exercise())


def test_session_is_refused_when_neither_lane_is_ready(tracker: FakeTracker) -> None:
    tracker.ready = False
    install(tracker)  # fresh readiness cache for the flipped service

    async def exercise() -> None:
        async with api_client() as api:
            health = await api.get("/api/health")
            assert health.json() == {**health.json(), "ready": False, "tracker_ready": False}
            refused = await api.post("/api/session", json={"access_code": "test-code"}, headers=ORIGIN)
            assert refused.status_code == 503
            assert refused.json() == {"error": "service_unavailable"}

    asyncio.run(exercise())


def test_paid_readiness_and_tracker_readiness_are_independent(tracker: FakeTracker) -> None:
    class PaidProvider:
        provider_name = "fake"
        api_key = "test-only"
        model = "fake-model"

    store.provider = PaidProvider()
    tracker.ready = False
    install(tracker)

    async def exercise() -> None:
        async with api_client() as api:
            health = await api.get("/api/health")
            assert health.json()["ready"] is True
            assert health.json()["tracker_ready"] is False
            # The paid lane alone still opens a session (its own gate is untouched by tracking).
            headers, session_id = await open_session(api)
            # ... but a tracking start is the tracker's to refuse.
            refused = await start_run(api, headers, session_id, "run-1")
            assert refused.status_code == 503
            assert refused.json() == {"error": "tracker_unavailable"}

    asyncio.run(exercise())


def test_a_delayed_start_cannot_replace_a_newer_run(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            assert (await start_run(api, headers, session_id, "run-a", start_seq=1)).status_code == 200
            assert (await start_run(api, headers, session_id, "run-b", start_seq=2)).status_code == 200

            delayed = await start_run(api, headers, session_id, "run-a", start_seq=1)
            assert delayed.status_code == 409
            assert delayed.json() == {"error": "stale_start", "latest_start_seq": 2}

            stale = await api.post("/api/track/frame", json=frame_body(session_id, "run-a"), headers=headers)
            assert stale.status_code == 409
            assert stale.json() == {"error": "stale_run", "active_run_id": "run-b"}
            assert "state" not in stale.json() and "box" not in stale.json()

            live = await api.post("/api/track/frame", json=frame_body(session_id, "run-b"), headers=headers)
            assert live.status_code == 200
            assert live.json()["run_id"] == "run-b"

    asyncio.run(exercise())


def test_start_is_idempotent_only_for_the_same_run_and_target(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            first = await start_run(api, headers, session_id, "run-a", start_seq=1)
            assert first.status_code == 200
            repeated = await start_run(api, headers, session_id, "run-a", start_seq=1)
            assert repeated.status_code == 200
            assert repeated.json() == first.json()
            assert tracker.paths().count("/runs/start") == 2  # the service's own start is idempotent too

            retargeted = await start_run(api, headers, session_id, "run-a", start_seq=1, target="cup.")
            assert retargeted.status_code == 409
            assert retargeted.json() == {"error": "stale_start", "latest_start_seq": 1}

            same_seq_new_run = await start_run(api, headers, session_id, "run-b", start_seq=1)
            assert same_seq_new_run.status_code == 409
            assert same_seq_new_run.json() == {"error": "stale_start", "latest_start_seq": 1}

    asyncio.run(exercise())


def test_a_stopped_run_cannot_be_restarted(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a", start_seq=1)
            stopped = await api.post(
                "/api/track/control",
                json={"session_id": session_id, "run_id": "run-a", "action": "stop"},
                headers=headers,
            )
            assert stopped.status_code == 200
            assert stopped.json() == {"run_id": "run-a", "active": False, "target": TARGET}
            assert "run-a" not in tracker.runs

            restarted = await start_run(api, headers, session_id, "run-a", start_seq=2)
            assert restarted.status_code == 409
            assert restarted.json() == {"error": "stale_start", "latest_start_seq": 1}

            again = await api.post(
                "/api/track/control",
                json={"session_id": session_id, "run_id": "run-a", "action": "stop"},
                headers=headers,
            )
            assert again.status_code == 409
            assert again.json() == {"error": "not_active_run", "active_run_id": None}

    asyncio.run(exercise())


def test_stale_frame_sequence_is_rejected_with_the_expected_sequence(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            assert (await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)).status_code == 200

            duplicate = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            assert duplicate.status_code == 409
            assert duplicate.json() == {"error": "stale_frame", "expected_seq_after": 0}

            older = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0, frame_id="cam-late"),
                                   headers=headers)
            assert older.status_code == 409
            # The tracker was never asked about a frame this application rejected.
            assert len(tracker.frames()) == 1

            assert (await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 1), headers=headers)).status_code == 200

    asyncio.run(exercise())


def test_concurrent_duplicate_frames_admit_exactly_one(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            body = frame_body(session_id, "run-a", 0)
            first, second = await asyncio.gather(
                api.post("/api/track/frame", json=body, headers=headers),
                api.post("/api/track/frame", json=dict(body), headers=headers),
            )
            codes = sorted([first.status_code, second.status_code])
            assert codes == [200, 409]
            winner = first if first.status_code == 200 else second
            loser = second if first.status_code == 200 else first
            assert loser.json() == {"error": "stale_frame", "expected_seq_after": 0}
            assert winner.json()["frame_seq"] == 0 and winner.json()["frame_id"] == "cam-0"
            assert len(tracker.frames()) == 1

    asyncio.run(exercise())


def test_a_recycled_run_id_never_inherits_an_earlier_sessions_state(tracker: FakeTracker) -> None:
    """A run id is client-minted and may be reused in a later session: the service must not inherit it."""

    async def exercise() -> None:
        async with api_client() as api:
            # A registration left behind by an earlier application session, mid-run.
            tracker.runs["run-a"] = {"frames": 9, "version": 9, "target": TARGET}
            headers, session_id = await open_session(api)
            started = await start_run(api, headers, session_id, "run-a")
            assert started.status_code == 200
            assert [payload["fresh"] for path, payload in tracker.calls if path == "/runs/start"] == [True]
            assert tracker.runs["run-a"]["frames"] == 0

            first = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            assert first.status_code == 200 and first.json()["version"] == 1

            # Re-asserting the SAME active run is a retry, not a fresh registration.
            again = await start_run(api, headers, session_id, "run-a")
            assert again.status_code == 200
            assert [payload["fresh"] for path, payload in tracker.calls if path == "/runs/start"] == [True, False]

    asyncio.run(exercise())


def test_a_wrong_upstream_frame_is_never_stamped_as_current(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            answer = tracker.answer_for({"run_id": "run-a", "frame_seq": 0},
                                        {"frames": 0, "version": 0, "target": TARGET})
            for wrong in ({"frame_seq": 9}, {"run_id": "run-b"}, {"target": "cup."}):
                tracker.answer = {**answer, **wrong}
                refused = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0),
                                         headers=headers)
                assert refused.status_code == 503
                assert refused.json() == {"error": "tracker_unavailable"}
            # The rejected frames advanced nothing: the sequence is still the client's to retry.
            tracker.answer = answer
            accepted = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            assert accepted.status_code == 200
            assert accepted.json()["frame_seq"] == 0

    asyncio.run(exercise())


async def _wait_for_inflight(tracker: FakeTracker, expected: int = 1) -> None:
    for _ in range(200):
        if len(tracker.frames()) >= expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the frame never reached the tracker service")


def test_selection_never_reaches_the_paid_lane_without_a_provider_key(tracker: FakeTracker) -> None:
    """A tracker-only session must be refused LOCALLY by /api/track/select: no provider socket is opened."""
    attempted: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempted.append(str(request.url))
        return httpx.Response(500, json={})

    async def exercise() -> None:
        previous_client = store.client
        store.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        # The real provider implementation, on the guarded client, with no key: if the request ever reached
        # it, the transport above would record the socket and the error would be the provider's, not ours.
        store.provider = GeminiProvider(store.client, api_key="")
        try:
            async with api_client() as api:
                health = await api.get("/api/health")
                assert health.json()["ready"] is False
                assert health.json()["tracker_ready"] is True
                headers, session_id = await open_session(api)  # tracker-only session, no paid key anywhere

                refused = await api.post("/api/track/select", json={
                    "session_id": session_id, "consent_ai": True,
                    "scene": {"frame_id": "cam-1", "image_base64": make_test_jpeg()},
                    "user_goal": "다음 작업을 안내해줘",
                }, headers=headers)
                assert refused.status_code == 503
                assert refused.json() == {"error": "service_unavailable"}
                assert attempted == [], f"a paid provider request was attempted: {attempted}"
        finally:
            await store.client.aclose()
            store.client = previous_client

    asyncio.run(exercise())


def test_a_stop_retires_a_run_whose_frame_is_in_flight(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            tracker.frame_gate = asyncio.Event()
            pending = asyncio.create_task(
                api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            )
            await _wait_for_inflight(tracker)

            # The stop must not wait for the frame that is in flight.
            stopped = await asyncio.wait_for(
                api.post("/api/track/control",
                         json={"session_id": session_id, "run_id": "run-a", "action": "stop"}, headers=headers),
                timeout=5,
            )
            assert stopped.status_code == 200
            assert stopped.json()["active"] is False
            assert "run-a" not in tracker.runs

            tracker.frame_gate.set()
            late = await asyncio.wait_for(pending, timeout=5)
            assert late.status_code == 409
            assert late.json() == {"error": "stale_run", "active_run_id": None}

    asyncio.run(exercise())


def test_ending_a_session_discards_a_frame_that_is_in_flight(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            tracker.frame_gate = asyncio.Event()
            pending = asyncio.create_task(
                api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            )
            await _wait_for_inflight(tracker)
            assert (await api.post("/api/session/end", headers=headers)).status_code == 200

            tracker.frame_gate.set()
            late = await asyncio.wait_for(pending, timeout=5)
            assert late.status_code == 401
            assert late.json() == {"error": "session_expired"}

    asyncio.run(exercise())


def test_a_replacement_start_releases_the_old_run_and_refuses_its_pending_frame(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a", start_seq=1)
            tracker.frame_gate = asyncio.Event()
            pending = asyncio.create_task(
                api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            )
            await _wait_for_inflight(tracker)

            replaced = await asyncio.wait_for(
                start_run(api, headers, session_id, "run-b", start_seq=2), timeout=5
            )
            assert replaced.status_code == 200
            # The replaced run's GPU state was released, and only then was the new run registered.
            stops = [payload["run_id"] for path, payload in tracker.calls if path == "/runs/stop"]
            assert stops == ["run-a"], tracker.paths()
            assert list(tracker.runs) == ["run-b"]

            tracker.frame_gate.set()
            late = await asyncio.wait_for(pending, timeout=5)
            assert late.status_code == 409
            assert late.json() == {"error": "stale_run", "active_run_id": "run-b"}

            # The new run is fully usable afterwards.
            tracker.frame_gate = None
            fresh = await api.post("/api/track/frame", json=frame_body(session_id, "run-b", 0), headers=headers)
            assert fresh.status_code == 200 and fresh.json()["run_id"] == "run-b"

    asyncio.run(exercise())


def test_a_closed_run_id_never_restarts_within_the_session(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            # Eleven runs, each stopped: no window, so the first run id is still refused at the end.
            for index in range(11):
                assert (await start_run(api, headers, session_id, f"run-{index}", start_seq=index + 1)).status_code == 200
                assert (await api.post("/api/track/control",
                                       json={"session_id": session_id, "run_id": f"run-{index}", "action": "stop"},
                                       headers=headers)).status_code == 200
            for index in (0, 5, 10):
                refused = await start_run(api, headers, session_id, f"run-{index}", start_seq=50 + index)
                assert refused.status_code == 409
                assert refused.json() == {"error": "stale_start", "latest_start_seq": 11}

    asyncio.run(exercise())


def test_seed_box_is_accepted_only_on_the_first_frame(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            seed = {"x": 0.28, "y": 0.20, "width": 0.42, "height": 0.24}
            first = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0, seed_box=seed),
                                   headers=headers)
            assert first.status_code == 200
            assert first.json()["source"] == "user"
            assert tracker.frames()[0]["seed_box"] == seed

            late = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 1, seed_box=seed),
                                  headers=headers)
            assert late.status_code == 409
            assert late.json() == {"error": "seed_not_first_frame"}
            assert len(tracker.frames()) == 1

    asyncio.run(exercise())


def test_control_rejects_frame_and_seed_fields(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            for extra in ({"seed_box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}},
                          {"image_base64": make_test_jpeg()},
                          {"frame_seq": 0}):
                refused = await api.post("/api/track/control", json=control_body(session_id, "run-a", **extra),
                                         headers=headers)
                assert refused.status_code == 422
                assert refused.json() == {"error": "invalid_request"}
            assert not [path for path in tracker.paths() if path.startswith("/runs")]

    asyncio.run(exercise())


def test_runs_are_isolated_per_application_session(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers_a, session_a = await open_session(api)
            headers_b, session_b = await open_session(api)
            assert session_a != session_b

            await start_run(api, headers_a, session_a, "shared-id")
            foreign = await api.post("/api/track/frame", json=frame_body(session_b, "shared-id"), headers=headers_b)
            assert foreign.status_code == 409
            assert foreign.json() == {"error": "stale_run", "active_run_id": None}

            # The same run id in the other session is that session's run, and neither sees a frame answered
            # for the other: each frame names the session it belongs to.
            await start_run(api, headers_b, session_b, "shared-id")
            assert (await api.post("/api/track/frame", json=frame_body(session_b, "shared-id"), headers=headers_b)).status_code == 200
            assert (await api.post("/api/track/frame", json=frame_body(session_a, "shared-id"), headers=headers_a)).status_code == 200

            mismatch = await api.post("/api/track/frame", json=frame_body(session_b, "shared-id"), headers=headers_a)
            assert mismatch.status_code == 401
            assert mismatch.json() == {"error": "session_mismatch"}

    asyncio.run(exercise())


def test_box_is_present_exactly_while_tracking(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            answer = {
                "run_id": "run-a", "target": TARGET, "track_id": "t-1", "generation": 0,
                "state": "occluded", "transition": "occluded", "box": None, "source": "grounder",
                "frame_seq": 0, "version": 7, "confidence": 0.22,
            }
            tracker.answer = answer
            occluded = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            assert occluded.status_code == 200
            assert occluded.json() == {
                "run_id": "run-a", "target": TARGET, "track_id": "t-1", "generation": 0,
                "state": "occluded", "transition": "occluded", "box": None, "source": "grounder",
                "frame_id": "cam-0", "frame_seq": 0, "version": 7, "confidence": 0.22,
                "ingest_age_ms": occluded.json()["ingest_age_ms"],
            }
            assert occluded.json()["ingest_age_ms"] >= 0

            # A tracker answer that violates the frozen invariant is never forwarded to the client.
            tracker.answer = {**answer, "state": "tracking", "transition": "tracking", "box": None, "version": 8}
            invalid = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 1), headers=headers)
            assert invalid.status_code == 503
            assert invalid.json() == {"error": "tracker_unavailable"}
            # ... and the frame it failed on is not recorded as answered, so the client may retry it.
            tracker.answer = {**answer, "state": "tracking", "transition": "tracking", "frame_seq": 1,
                              "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1}, "version": 9}
            retried = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 1), headers=headers)
            assert retried.status_code == 200
            assert retried.json()["box"] == {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1}

    asyncio.run(exercise())


def test_a_non_increasing_version_fails_closed(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            tracker.answer = {
                "run_id": "run-a", "target": TARGET, "track_id": "t-1", "generation": 0,
                "state": "tracking", "transition": "tracking",
                "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1},
                "source": "grounder", "frame_seq": 0, "version": 0, "confidence": 0.5,
            }
            assert (await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)).status_code == 200
            stalled = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 1), headers=headers)
            assert stalled.status_code == 503
            assert stalled.json() == {"error": "tracker_unavailable"}

    asyncio.run(exercise())


def test_transport_failure_keeps_the_frame_retryable(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")

            tracker.transport_error = httpx.ConnectError("connection refused")
            unavailable = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            assert unavailable.status_code == 503
            assert unavailable.json() == {"error": "tracker_unavailable"}

            tracker.transport_error = httpx.ReadTimeout("timed out")
            timed_out = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            assert timed_out.status_code == 504
            assert timed_out.json() == {"error": "tracker_timeout"}

            tracker.transport_error = None
            recovered = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            assert recovered.status_code == 200
            assert recovered.json()["frame_seq"] == 0

    asyncio.run(exercise())


def test_the_tracker_lane_is_disabled_when_the_endpoint_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient default: an unconfigured process reports a named prerequisite and probes nothing."""
    monkeypatch.setenv("DEMO_ACCESS_CODE", "test-code")
    monkeypatch.delenv("AISW_TRACKER_URL", raising=False)
    previous = (store.provider, store.tracker_client, store.tracker_error)
    store.provider = NoProvider()
    store.tracker_client = None
    store.tracker_error = None
    store.sessions.clear()

    async def exercise() -> None:
        async with api_client() as api:
            health = await api.get("/api/health")
            assert health.status_code == 200
            assert health.json()["tracker_ready"] is False

            refused = await api.post("/api/session", json={"access_code": "test-code"}, headers=ORIGIN)
            assert refused.status_code == 503
            assert refused.json() == {"error": "service_unavailable"}

    try:
        asyncio.run(exercise())
    finally:
        store.provider, store.tracker_client, store.tracker_error = previous
        store.sessions.clear()


def test_a_non_loopback_tracker_url_fails_closed(tracker: FakeTracker, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AISW_TRACKER_URL", "http://10.0.0.5:8090")
    store.tracker_client = None
    store.tracker_error = None

    async def exercise() -> None:
        async with api_client() as api:
            health = await api.get("/api/health")
            assert health.status_code == 200
            assert health.json()["tracker_ready"] is False

            refused = await api.post("/api/session", json={"access_code": "test-code"}, headers=ORIGIN)
            assert refused.status_code == 503  # no paid key either, and the tracker lane is misconfigured
            assert refused.json() == {"error": "service_unavailable"}

    asyncio.run(exercise())


def test_ending_a_session_abandons_its_run(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0), headers=headers)
            assert "run-a" in tracker.runs

            ended = await api.post("/api/session/end", headers=headers)
            assert ended.status_code == 200
            assert "run-a" not in tracker.runs

    asyncio.run(exercise())


def test_a_start_without_a_target_is_refused_and_registers_nothing(tracker: FakeTracker) -> None:
    """The old implicit default is gone: nothing starts, and nothing is spent or sent on the client's behalf."""

    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            absent = await api.post(
                "/api/track/control",
                json={"session_id": session_id, "run_id": "run-a", "action": "start", "start_seq": 1},
                headers=headers,
            )
            assert absent.status_code == 422
            assert absent.json() == {"error": "target_required"}
            blank = await api.post(
                "/api/track/control",
                json={"session_id": session_id, "run_id": "run-a", "action": "start", "start_seq": 1,
                      "target": "   "},
                headers=headers,
            )
            assert blank.status_code == 422
            assert blank.json() == {"error": "target_required"}
            # A refused start is not an admitted run: no service call, and the sequence is still free.
            assert [path for path in tracker.paths() if path.startswith("/runs")] == []
            assert (await start_run(api, headers, session_id, "run-a", start_seq=1)).status_code == 200

    asyncio.run(exercise())


def test_a_drawn_box_starts_a_run_with_no_provider_and_no_typed_class(tracker: FakeTracker) -> None:
    """The manual correction path: the user's box is the target, named by its provenance, never by a class."""

    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            started = await start_run(api, headers, session_id, "run-a", target=MANUAL_TARGET)
            assert started.status_code == 200
            assert started.json() == {"run_id": "run-a", "active": True, "target": MANUAL_TARGET}

            seed = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}
            first = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 0, seed_box=seed),
                                   headers=headers)
            assert first.status_code == 200
            assert first.json()["state"] == "tracking"
            assert first.json()["source"] == "user"
            assert first.json()["target"] == MANUAL_TARGET
            assert [payload["target"] for path, payload in tracker.calls if path == "/runs/start"] == [MANUAL_TARGET]
            assert tracker.frames()[-1]["seed_box"] == seed

    asyncio.run(exercise())


# -- binary frames (Content-Type: image/jpeg, fields in the query string) ------------------------------------


def frame_query(session_id: str, run_id: str, seq: int = 0, **extra: Any) -> dict[str, Any]:
    return {"session_id": session_id, "run_id": run_id, "frame_id": f"cam-{seq}", "frame_seq": str(seq), **extra}


async def post_binary_frame(api: httpx.AsyncClient, headers: dict[str, str], params: Any, body: bytes,
                            content_type: str = "image/jpeg") -> httpx.Response:
    return await api.post("/api/track/frame", params=params, content=body,
                          headers={**headers, "Content-Type": content_type})


def test_a_binary_frame_reaches_the_tracker_as_the_same_image_as_json(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            jpeg_b64 = make_test_jpeg()
            jpeg = base64.b64decode(jpeg_b64)

            binary = await post_binary_frame(api, headers, frame_query(session_id, "run-a", 0), jpeg)
            assert binary.status_code == 200
            assert binary.json()["frame_id"] == "cam-0"
            assert binary.json()["frame_seq"] == 0

            as_json = await api.post("/api/track/frame", json=frame_body(session_id, "run-a", 1, image_base64=jpeg_b64),
                                     headers=headers)
            assert as_json.status_code == 200
            sent = tracker.frames()
            assert [frame["frame_seq"] for frame in sent] == [0, 1]
            # The tracker cannot tell the two forms apart: same base64 bytes, no seed.
            assert sent[0]["frame_b64"] == sent[1]["frame_b64"] == jpeg_b64
            assert "seed_box" not in sent[0] or sent[0]["seed_box"] is None

            # Admission is shared: a replayed binary frame is as stale as a replayed JSON one.
            replay = await post_binary_frame(api, headers, frame_query(session_id, "run-a", 1), jpeg)
            assert replay.status_code == 409
            assert replay.json() == {"error": "stale_frame", "expected_seq_after": 1}

    asyncio.run(exercise())


@pytest.mark.parametrize("params", [
    {"run_id": "run-a", "frame_id": "cam-0", "frame_seq": "0"},  # no session
    {"frame_seq": "0", "frame_id": "cam-0", "run_id": "run-a", "session_id": "S", "extra": "1"},  # unknown field
    {"frame_seq": "-1"},
    {"frame_seq": "01"},
    {"frame_seq": "+1"},
    {"frame_seq": "1.0"},
    {"frame_seq": ""},
    {"frame_id": ""},
    {"seed_box": '{"x":0,"y":0,"width":0.5,"height":0.5}'},  # seeds are JSON-only
])
def test_a_binary_frame_with_a_bad_query_is_refused_before_any_run_state(tracker: FakeTracker, params: dict[str, str]) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            query = {**frame_query(session_id, "run-a", 0), **params}
            if query.get("session_id") == "S":
                query["session_id"] = session_id
            if "session_id" not in params and params.get("run_id") == "run-a":
                query.pop("session_id")
            response = await post_binary_frame(api, headers, query, base64.b64decode(make_test_jpeg()))
            assert response.status_code == 422
            assert response.json()["error"] == "invalid_request"
            assert tracker.frames() == []

    asyncio.run(exercise())


def test_a_binary_frame_repeating_a_query_field_is_refused(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            params = [*frame_query(session_id, "run-a", 0).items(), ("frame_seq", "1")]
            response = await post_binary_frame(api, headers, params, base64.b64decode(make_test_jpeg()))
            assert response.status_code == 422
            assert tracker.frames() == []

    asyncio.run(exercise())


def test_a_binary_frame_is_validated_as_a_jpeg_and_bounded(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            query = frame_query(session_id, "run-a", 0)

            not_jpeg = await post_binary_frame(api, headers, query, b"\x89PNG\r\n\x1a\n" + b"0" * 64)
            assert (not_jpeg.status_code, not_jpeg.json()["error"]) == (422, "invalid_image")

            tiny = await post_binary_frame(api, headers, query, base64.b64decode(make_test_jpeg(64, 64)))
            assert (tiny.status_code, tiny.json()["error"]) == (422, "invalid_image_dimensions")

            oversized = await post_binary_frame(api, headers, query, b"\xff\xd8\xff" + b"\0" * 1_500_000)
            assert (oversized.status_code, oversized.json()["error"]) == (422, "image_too_large")

            too_large_body = await post_binary_frame(api, headers, query, b"\xff\xd8\xff" + b"\0" * (4 * 1024 * 1024))
            assert (too_large_body.status_code, too_large_body.json()["error"]) == (413, "body_too_large")

            other_type = await post_binary_frame(api, headers, query, base64.b64decode(make_test_jpeg()), "image/png")
            assert (other_type.status_code, other_type.json()["error"]) == (415, "json_required")
            assert tracker.frames() == []

    asyncio.run(exercise())


def test_a_binary_frame_for_another_session_is_refused(tracker: FakeTracker) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, session_id = await open_session(api)
            await start_run(api, headers, session_id, "run-a")
            other = "00000000-0000-4000-8000-000000000000"
            response = await post_binary_frame(api, headers, frame_query(other, "run-a", 0),
                                               base64.b64decode(make_test_jpeg()))
            assert (response.status_code, response.json()["error"]) == (401, "session_mismatch")
            assert tracker.frames() == []

    asyncio.run(exercise())

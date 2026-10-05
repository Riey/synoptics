"""Startup target-selection tests: the one paid call, its honest declines, and its fences.

Every test drives the real ASGI application with a fail-closed canned transport: a real provider object
whose HTTP client is an ``httpx.MockTransport`` (so the prompt, the envelope parsing and the usage
accounting under test are the production ones) plus a recording tracker stub. Nothing here opens a socket,
reads a credential, or touches a GPU — a provider attempt is countable, so "exactly one call" and "no call
at all" are asserted from the recorded requests rather than from a mock's cooperation.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from typing import Any, Callable

import httpx
import pytest
from PIL import Image

from backend.app.main import app, store
from backend.app.provider import DeepSeekProvider, GeminiProvider
from backend.app.tracking import TrackerClient

TRACKER_URL = "http://127.0.0.1:8090"
ORIGIN = {"Origin": "http://127.0.0.1:8000"}
GOAL = "부품 조립을 돕고 있어"
CONTEXT = "책상 위에서 노트북을 분해하는 중"
SELECTED = {"status": "selected", "target": "laptop", "rationale": "책상 위 노트북이 작업 대상입니다."}


def make_test_jpeg(width: int = 640, height: int = 480, color: str = "blue") -> str:
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


class RecordingTracker:
    """A tracker service that records what reached it, so "no tracker call" is an observation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.runs: dict[str, dict[str, Any]] = {}

    async def handler(self, request: httpx.Request) -> httpx.Response:
        payload: dict[str, Any] = json.loads(request.content) if request.content else {}
        self.calls.append((request.url.path, payload))
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True, "error": None})
        if request.url.path == "/runs/start":
            existing = self.runs.get(payload["run_id"])
            if existing is not None and not payload.get("fresh"):
                return httpx.Response(200, json={"run_id": payload["run_id"], "target": existing["target"],
                                                 "active": True})
            self.runs[payload["run_id"]] = {"target": payload["target"], "version": 0}
            return httpx.Response(200, json={"run_id": payload["run_id"], "target": payload["target"],
                                             "active": True})
        if request.url.path == "/runs/frame":
            run = self.runs.get(payload["run_id"])
            if run is None:
                return httpx.Response(409, json={"error": "stale_run"})
            run["version"] += 1
            # The answer echoes the run's own target: the application refuses an answer that names another.
            return httpx.Response(200, json={
                "run_id": payload["run_id"], "target": run["target"], "track_id": "t-1", "generation": 0,
                "state": "tracking", "transition": "acquired",
                "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1},
                "source": "user" if payload.get("seed_box") else "grounder",
                "frame_seq": payload["frame_seq"], "version": run["version"], "confidence": 0.8,
            })
        if request.url.path == "/runs/stop":
            if self.runs.pop(payload["run_id"], None) is None:
                return httpx.Response(409, json={"error": "not_active_run"})
            return httpx.Response(200, json={"run_id": payload["run_id"], "stopped": True})
        raise AssertionError(f"unexpected tracker path {request.url.path}")

    def paths(self) -> list[str]:
        return [path for path, _payload in self.calls]

    def run_paths(self) -> list[str]:
        """/runs/** only: /health is readiness, which opening a session legitimately performs."""
        return [path for path in self.paths() if path.startswith("/runs")]


class KeylessProvider:
    """The production shape of a deployment with no paid key: readiness is a property, not an exception."""

    provider_name = "none"
    model = "none"
    api_key = ""

    async def select_target(self, _request: Any) -> tuple[dict[str, Any], None]:
        raise AssertionError("a provider with no key must never be called")


class ProviderStub:
    """A canned transport behind a real provider object, recording every outbound attempt.

    The handler may be sync or async; an async one is what lets a test hold a call in flight.
    """

    def __init__(self, provider: Any, handler: Callable[[httpx.Request], Any]) -> None:
        self.requests: list[httpx.Request] = []
        provider.api_key = "test-only-key"

        async def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            result = handler(request)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        provider.client = httpx.AsyncClient(transport=httpx.MockTransport(record))
        self.provider = provider

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def deepseek(handler: Callable[[httpx.Request], Any]) -> ProviderStub:
    return ProviderStub(DeepSeekProvider(httpx.AsyncClient(), api_key="test-only-key", model="deepseek-flash"),
                        handler)


def gemini(handler: Callable[[httpx.Request], Any]) -> ProviderStub:
    return ProviderStub(GeminiProvider(httpx.AsyncClient(), api_key="test-only-key", model="gemini-test"),
                        handler)


def canned(body: dict[str, Any], status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return handler


def chat_answer(arguments: dict[str, Any], *, finish_reason: str = "tool_calls",
                usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """A chat-completions answer carrying exactly one track_target call."""
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "track_target", "arguments": json.dumps(arguments)},
                }],
            },
        }],
        "usage": usage if usage is not None else {"prompt_tokens": 1200, "completion_tokens": 40,
                                                 "total_tokens": 1240, "prompt_cache_hit_tokens": 1000},
    }


def holding(answer: dict[str, Any], entered: asyncio.Event,
            release: asyncio.Event) -> Callable[[httpx.Request], Any]:
    """An async handler that reports when the call arrived and answers only once released."""

    async def handler(_request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(200, json=answer)

    return handler


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    """A recording tracker plus a fresh session store; the provider is installed by each test."""
    monkeypatch.setenv("DEMO_ACCESS_CODE", "test-code")
    monkeypatch.delenv("AISW_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("AISW_TRACKER_URL", raising=False)
    tracker = RecordingTracker()
    previous = (store.provider, store.tracker_client, store.tracker_error)
    store.tracker_client = TrackerClient(
        TRACKER_URL, client=httpx.AsyncClient(transport=httpx.MockTransport(tracker.handler))
    )
    store.tracker_error = None
    store.sessions.clear()
    store.creations.clear()
    try:
        yield tracker
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
    return {**ORIGIN, "Cookie": cookie}, response.json()["session_id"]


def select_body(session_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "consent_ai": True,
        "scene": {"frame_id": "cam-1", "image_base64": make_test_jpeg()},
        "user_goal": GOAL,
        "context": CONTEXT,
        **extra,
    }


async def select(api: httpx.AsyncClient, headers: dict[str, str], session_id: str, **extra: Any) -> httpx.Response:
    return await api.post("/api/track/select", json=select_body(session_id, **extra), headers=headers)


def test_deepseek_select_names_one_object_and_reports_its_usage(env: RecordingTracker) -> None:
    """One call, the task text really in the prompt, and the provider's own token counts passed through."""

    async def exercise() -> None:
        stub = deepseek(canned(chat_answer(SELECTED)))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await select(api, headers, session_id)

            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "selected"
            assert body["target"] == "laptop"
            assert body["frame_id"] == "cam-1"
            assert body["provider"] == "deepseek"
            assert body["model"] == "deepseek-flash"
            assert body["usage"] == {"input_tokens": 1200, "output_tokens": 40, "total_tokens": 1240,
                                     "cached_input_tokens": 1000, "thought_tokens": None, "tool_use_tokens": None}

            assert len(stub.requests) == 1
            sent = stub.bodies()[0]
            assert sent["tools"][0]["function"]["name"] == "track_target"
            # The task reaches the model: goal, context and the frame it is about are all in the request.
            text = json.dumps(sent["messages"], ensure_ascii=False)
            assert GOAL in text and CONTEXT in text and "cam-1" in text
            assert "data:image/jpeg;base64," in text
            # The analysis answers with a name only; it starts no run by itself.
            assert env.run_paths() == []

    asyncio.run(exercise())


def test_gemini_select_uses_its_own_envelope(env: RecordingTracker) -> None:
    """The other adapter's tool-call shape reaches the same contract, through the same route."""
    body = {
        "steps": [{"type": "function_call", "name": "track_target", "arguments": SELECTED}],
        "usage": {"total_input_tokens": 900, "total_output_tokens": 30},
    }

    async def exercise() -> None:
        stub = gemini(canned(body))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await select(api, headers, session_id)

            assert response.status_code == 200
            assert response.json()["status"] == "selected"
            assert response.json()["target"] == "laptop"
            assert response.json()["provider"] == "gemini"
            assert response.json()["usage"]["input_tokens"] == 900
            assert len(stub.requests) == 1
            assert stub.bodies()[0]["tools"][0]["name"] == "track_target"
            assert env.run_paths() == []

    asyncio.run(exercise())


@pytest.mark.parametrize("arguments", [
    {"status": "no_target", "rationale": "화면에 추적할 만한 물체가 없습니다."},
    {"status": "uncertain", "rationale": "두 물체가 비슷하게 보여 하나를 고를 수 없습니다."},
])
def test_a_declined_selection_is_a_success_that_starts_nothing(env: RecordingTracker,
                                                               arguments: dict[str, Any]) -> None:
    """Declining is an answer, not an error — and it must not start a tracker run."""

    async def exercise() -> None:
        stub = deepseek(canned(chat_answer(arguments)))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await select(api, headers, session_id)

            assert response.status_code == 200
            body = response.json()
            assert body["status"] == arguments["status"]
            assert body["target"] is None
            assert body["rationale"] == arguments["rationale"]
            assert body["usage"]["input_tokens"] == 1200  # the call was still paid for, and is still reported
            assert len(stub.requests) == 1
            assert env.run_paths() == []

    asyncio.run(exercise())


@pytest.mark.parametrize("arguments", [
    {"status": "selected", "target": None},
    {"status": "selected", "target": "   "},
    {"status": "selected", "rationale": "laptop입니다."},
    {"status": "no_target", "target": "laptop", "rationale": "없습니다."},
    {"status": "no_target", "rationale": "   "},
])
def test_a_self_contradictory_answer_fails_closed_instead_of_inventing_a_target(
    env: RecordingTracker, arguments: dict[str, Any]
) -> None:
    """A target that contradicts its status, or an empty rationale, is refused — never repaired here."""

    async def exercise() -> None:
        stub = deepseek(canned(chat_answer(arguments)))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await select(api, headers, session_id)

            assert response.status_code == 502
            assert response.json() == {"error": "invalid_provider_output", "reason": "schema"}
            assert len(stub.requests) == 1  # no repair call
            assert env.run_paths() == []

    asyncio.run(exercise())


def test_a_truncated_answer_is_refused_without_a_second_call(env: RecordingTracker) -> None:
    async def exercise() -> None:
        stub = deepseek(canned(chat_answer(SELECTED, finish_reason="length")))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await select(api, headers, session_id)

            assert response.status_code == 502
            assert response.json() == {"error": "invalid_provider_output", "reason": "truncated"}
            assert len(stub.requests) == 1
            assert env.run_paths() == []

    asyncio.run(exercise())


@pytest.mark.parametrize("provider_name", ["deepseek", "gemini"])
def test_a_provider_failure_costs_exactly_one_attempt_and_starts_nothing(env: RecordingTracker,
                                                                        provider_name: str) -> None:
    async def exercise() -> None:
        handler = canned({"error": {"message": "upstream"}}, status=500)
        stub = deepseek(handler) if provider_name == "deepseek" else gemini(handler)
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await select(api, headers, session_id)

            assert response.status_code == 502
            assert response.json() == {"error": "provider_unavailable", "reason": "upstream_status"}
            assert len(stub.requests) == 1
            assert env.run_paths() == []

    asyncio.run(exercise())


def test_an_unconfigured_provider_refuses_before_any_socket(env: RecordingTracker) -> None:
    async def exercise() -> None:
        store.provider = KeylessProvider()
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await select(api, headers, session_id)

            assert response.status_code == 503
            assert response.json() == {"error": "service_unavailable"}
            assert env.run_paths() == []

    asyncio.run(exercise())


def test_a_held_call_makes_a_second_select_refuse_instead_of_queueing(env: RecordingTracker) -> None:
    """The manual cooldown cannot be what protects a long first call, so an overlap is refused outright.

    The clock that would otherwise refuse the second request is deliberately rewound before it arrives, so
    the in-flight guard is the only thing that can stop it — and it must stop it without a second attempt.
    """

    async def exercise() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        stub = deepseek(holding(chat_answer(SELECTED), entered, release))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            first = asyncio.ensure_future(select(api, headers, session_id))
            await entered.wait()
            for session in store.sessions.values():
                session.last_analyze_at -= 60  # as if the first call had already outlived the cooldown

            second = await select(api, headers, session_id)
            assert second.status_code == 503
            assert second.json() == {"error": "provider_busy"}
            assert len(stub.requests) == 1

            release.set()
            assert (await first).status_code == 200
            assert len(stub.requests) == 1
            assert env.run_paths() == []

    asyncio.run(exercise())


def test_a_session_that_ends_while_selecting_gets_no_answer(env: RecordingTracker) -> None:
    async def exercise() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        stub = deepseek(holding(chat_answer(SELECTED), entered, release))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            pending = asyncio.ensure_future(select(api, headers, session_id))
            await entered.wait()
            assert (await api.post("/api/session/end", headers=headers)).status_code == 200

            release.set()
            response = await pending
            assert response.status_code == 401
            assert response.json() == {"error": "session_expired"}
            assert env.run_paths() == []

    asyncio.run(exercise())


def test_consent_and_session_identity_are_checked_before_the_provider(env: RecordingTracker) -> None:
    async def exercise() -> None:
        stub = deepseek(canned(chat_answer(SELECTED)))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            refused = await select(api, headers, session_id, consent_ai=False)
            assert refused.status_code == 400
            assert refused.json() == {"error": "ai_consent_required"}

            mismatch = await api.post(
                "/api/track/select",
                json=select_body("123e4567-e89b-42d3-a456-426614174000"),
                headers=headers,
            )
            assert mismatch.status_code == 401
            assert mismatch.json() == {"error": "session_mismatch"}

            assert stub.requests == []

    asyncio.run(exercise())


def test_an_immediate_repeat_select_is_paced_by_the_manual_cooldown(env: RecordingTracker) -> None:
    """The startup analysis spends the same manual budget as a deliberate guidance click."""

    async def exercise() -> None:
        stub = deepseek(canned(chat_answer(SELECTED)))
        store.provider = stub.provider
        async with api_client() as api:
            headers, session_id = await open_session(api)
            assert (await select(api, headers, session_id)).status_code == 200
            again = await select(api, headers, session_id)
            assert again.status_code == 429
            assert again.json()["error"] == "rate_limited"
            assert len(stub.requests) == 1
            assert env.run_paths() == []

    asyncio.run(exercise())

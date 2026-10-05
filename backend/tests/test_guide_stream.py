"""Streamed /api/guide/plan: the early-hint scanner, the provider's SSE reader and the route's SSE contract.

Same discipline as ``test_guide.py``: the real ASGI application and a real ``DeepSeekProvider`` whose HTTP
client is an ``httpx.MockTransport``; here the transport answers ``stream: true`` requests with chunked
chat-completions SSE lines. Nothing opens a socket or reaches a paid API.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from backend.app import intent, main
from backend.app.intent import PlanHintScanner, plan_stream_hints
from backend.app.main import app, store
from backend.app.provider import (
    DeepSeekProvider,
    GuideStreamFinal,
    InvalidOutputStage,
    ProviderInvalidOutput,
    ProviderRateLimited,
    ProviderUnavailable,
)
from backend.tests.test_guide import (
    ORIGIN,
    PLAN,
    api_client,
    chat_answer,
    env,  # noqa: F401  (fixture)
    install,
    make_test_jpeg,
    mutate,
    open_session,
    review_step,
    unpace,
)

SSE = {"Accept": "text/event-stream"}
USAGE = {"prompt_tokens": 900, "completion_tokens": 50, "total_tokens": 950}
ARGS = json.dumps(PLAN, ensure_ascii=False)


# ------------------------------------------------------------------------------------------- scanner


def test_hints_appear_only_once_each_string_is_closed() -> None:
    target_end = ARGS.index('"cup"') + len('"cup"')
    say = PLAN["steps"][0]["say"]
    say_end = ARGS.index(json.dumps(say, ensure_ascii=False)) + len(json.dumps(say, ensure_ascii=False))
    for cut in range(len(ARGS) + 1):
        target, first_say = plan_stream_hints(ARGS[:cut])
        assert target == ("cup" if cut >= target_end else None), cut
        assert first_say == (say if cut >= say_end else None), cut


def test_incremental_feeding_equals_the_pure_function_at_every_split() -> None:
    for size in (1, 2, 3, 7, 64):
        scanner = PlanHintScanner()
        for start in range(0, len(ARGS), size):
            scanner.feed(ARGS[start:start + size])
            assert (scanner.target, scanner.first_say) == plan_stream_hints(ARGS[:start + size])
        assert (scanner.target, scanner.first_say) == ("cup", PLAN["steps"][0]["say"])


def test_escapes_are_decoded_and_never_cut() -> None:
    plan = mutate(PLAN, selection={"status": "selected", "target": 'mug "a"\\b', "rationale": "r"},
                  steps=[{"id": "s1", "say": "컵\n을 é \U0001F600 옮기세요", "commands": [], "done_when": "x"}])
    text = json.dumps(plan)  # ASCII escapes, including a surrogate pair
    assert "\\ud83d\\ude00" in text
    for cut in range(len(text) + 1):
        target, first_say = plan_stream_hints(text[:cut])
        assert target in (None, 'mug "a"\\b')
        assert first_say in (None, "컵\n을 é \U0001F600 옮기세요")
    assert plan_stream_hints(text) == ('mug "a"\\b', "컵\n을 é \U0001F600 옮기세요")


def test_only_the_two_paths_count() -> None:
    # Key order differs from the schema, "target" occurs as a command anchor value, a later step has a say,
    # and a decoy "say"/"target" sits in another object.
    text = json.dumps({
        "goal_when": "target",
        "decoy": {"target": "wrong", "say": "wrong"},
        "steps": [
            {"id": "s1", "commands": [{"kind": "label", "anchor": "target", "text": "say"}], "say": "첫 단계"},
            {"id": "s2", "say": "둘째 단계", "commands": [], "done_when": "y"},
        ],
        "selection": {"status": "selected", "rationale": "target", "target": "lamp"},
    }, ensure_ascii=False)
    assert plan_stream_hints(text) == ("lamp", "첫 단계")


def test_no_hint_from_a_non_selection_blank_or_overlong_value() -> None:
    assert plan_stream_hints(json.dumps({"selection": {"status": "no_target", "rationale": "없음"}, "steps": []})) \
        == (None, None)
    assert plan_stream_hints('{"selection":{"target":"   "},"steps":[{"say":""}]}') == (None, None)
    long_target = "x" * (intent.HINT_TARGET_MAX + 1)
    long_say = "가" * (intent.HINT_SAY_MAX + 1)
    assert plan_stream_hints(json.dumps({"selection": {"target": long_target}, "steps": [{"say": long_say}]})) \
        == (None, None)


def test_malformed_text_stops_the_scan() -> None:
    assert plan_stream_hints('{"selection"]: {"target": "cup"}}') == (None, None)
    assert plan_stream_hints('{"selection": {"target": "c\\x"}}') == (None, None)  # invalid escape
    assert plan_stream_hints('}{"selection": {"target": "cup"}}') == (None, None)


# ------------------------------------------------------------------------------- provider SSE reader


def sse_lines(arguments: str, *, pieces: int = 12, finish_reason: Any = "tool_calls", name: str = "guide_plan",
              usage: dict[str, Any] | None = USAGE, done: bool = True) -> list[bytes]:
    """A chat-completions stream for one tool call, the argument text split into ``pieces``."""
    def chunk(choice: dict[str, Any] | None, **extra: Any) -> bytes:
        body = {"id": "c1", "object": "chat.completion.chunk", "choices": [choice] if choice else [], **extra}
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode()

    out = [b": keep-alive\n\n", chunk({"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": [
        {"index": 0, "id": "call_1", "type": "function", "function": {"name": name, "arguments": ""}}]},
        "finish_reason": None})]
    step = max(1, -(-len(arguments) // pieces))
    for start in range(0, len(arguments), step):
        out.append(chunk({"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": arguments[start:start + step]}}]}, "finish_reason": None}))
    if finish_reason is not None:
        out.append(chunk({"index": 0, "delta": {}, "finish_reason": finish_reason}))
    if usage is not None:
        out.append(chunk(None, usage=usage))
    if done:
        out.append(b"data: [DONE]\n\n")
    return out


class StreamStub:
    """A real DeepSeek provider whose transport answers JSON calls from ``answers`` and stream calls by
    replaying ``stream_lines`` (optionally pausing before line ``hold_at`` until ``release`` is set)."""

    def __init__(self, stream_lines: list[bytes] | httpx.Response, *, answers: dict[str, Any] | None = None,
                 hold_at: int | None = None) -> None:
        self.stream_lines = stream_lines
        self.answers = {"guide_plan": PLAN, **(answers or {})}
        self.hold_at = hold_at
        self.release = asyncio.Event()
        self.held = asyncio.Event()
        self.requests: list[dict[str, Any]] = []
        self.provider = DeepSeekProvider(httpx.AsyncClient(), api_key="test-only-key", model="deepseek-flash")
        self.provider.client = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    async def body(self) -> AsyncIterator[bytes]:
        assert isinstance(self.stream_lines, list)
        for index, line in enumerate(self.stream_lines):
            if index == self.hold_at:
                self.held.set()
                await self.release.wait()
            yield line

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("stream"):
            if isinstance(self.stream_lines, httpx.Response):
                return self.stream_lines
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=self.body())
        tool = body["tools"][0]["function"]["name"] if body["tool_choice"] == "auto" else body["tool_choice"]["function"]["name"]
        answer = self.answers[tool]
        return httpx.Response(200, json=answer if "choices" in answer else chat_answer(tool, answer))


async def drain(provider: DeepSeekProvider) -> tuple[list[str], GuideStreamFinal]:
    from backend.app.guide_contracts import GuidePlanRequest

    request = GuidePlanRequest.model_validate({
        "session_id": "00000000-0000-4000-8000-000000000000", "consent_ai": True,
        "scene": {"frame_id": "cam-1", "image_base64": make_test_jpeg()}, "user_goal": "목표"})
    pieces: list[str] = []
    final: GuideStreamFinal | None = None
    async for item in provider.guide_plan_stream(request, make_test_jpeg()):
        if isinstance(item, GuideStreamFinal):
            final = item
        else:
            assert final is None
            pieces.append(item)
    assert final is not None
    return pieces, final


def test_provider_stream_yields_argument_pieces_then_the_final_answer() -> None:
    async def exercise() -> None:
        reasoning = b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"private reasoning"},"finish_reason":null}]}\n\n'
        stub = StreamStub([reasoning, *sse_lines(ARGS)])
        pieces, final = await drain(stub.provider)
        assert len(pieces) == 12 and "".join(pieces) == ARGS
        assert final.arguments == PLAN
        assert final.usage is not None and final.usage.output_tokens == 50
        sent = stub.requests[0]
        assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
        # The stream carries the same shared upper planning profile as the non-streamed plan call.
        assert sent["tool_choice"] == "auto"
        assert sent["thinking"] == {"type": "enabled"}

    asyncio.run(exercise())


@pytest.mark.parametrize("lines, error, stage", [
    (sse_lines(ARGS, finish_reason="length"), ProviderInvalidOutput, InvalidOutputStage.TRUNCATED),
    (sse_lines(ARGS, finish_reason="content_filter"), ProviderInvalidOutput, InvalidOutputStage.TOOL_ENVELOPE),
    (sse_lines(ARGS, finish_reason=None), ProviderInvalidOutput, InvalidOutputStage.ENVELOPE),
    (sse_lines(ARGS, name="guide_confirm"), ProviderInvalidOutput, InvalidOutputStage.TOOL_ENVELOPE),
    (sse_lines(ARGS[:-1]), ProviderInvalidOutput, InvalidOutputStage.TOOL_JSON),
    (sse_lines(ARGS)[:2] + [b"data: {not json\n\n"], ProviderInvalidOutput, InvalidOutputStage.ENVELOPE),
    ([b'data: {"error": {"message": "x"}}\n\n'], ProviderUnavailable, None),
    (httpx.Response(500, text="boom"), ProviderUnavailable, None),
    (httpx.Response(429, headers={"retry-after": "7"}), ProviderRateLimited, None),
])
def test_provider_stream_failures_are_the_provider_exceptions(lines: Any, error: type[Exception],
                                                                stage: InvalidOutputStage | None) -> None:
    async def exercise() -> None:
        with pytest.raises(error) as raised:
            await drain(StreamStub(lines).provider)
        if stage is not None:
            assert raised.value.stage == stage  # type: ignore[attr-defined]

    asyncio.run(exercise())


def test_provider_stream_accepts_stop_as_a_finish_reason() -> None:
    async def exercise() -> None:
        _pieces, final = await drain(StreamStub(sse_lines(ARGS, finish_reason="stop", usage=None)).provider)
        assert final.arguments == PLAN and final.usage is None

    asyncio.run(exercise())


# ------------------------------------------------------------------------------------------- route


def parse_sse(text: str) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    """(events, ping count) from a complete SSE body."""
    events: list[tuple[str, dict[str, Any]]] = []
    pings = 0
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        if block == ": ping":
            pings += 1
            continue
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events, pings


def use(stub: StreamStub) -> StreamStub:
    store.provider = stub.provider
    return stub


def plan_body(session_id: str, **extra: Any) -> dict[str, Any]:
    return {"session_id": session_id, "consent_ai": True,
            "scene": {"frame_id": "cam-1", "image_base64": make_test_jpeg()},
            "user_goal": "책상 위 물건을 정리하고 싶어", "context": "작업 공간 정리", **extra}


def test_streamed_plan_sends_target_then_first_say_then_the_json_answer(env) -> None:  # noqa: F811
    async def exercise() -> None:
        stub = use(StreamStub(sse_lines(ARGS, pieces=40)))
        async with api_client() as api:
            headers, session_id = await open_session(api)
            plain = await api.post("/api/guide/plan", json=plan_body(session_id), headers=headers)
            unpace(session_id)
            streamed = await api.post("/api/guide/plan", json=plan_body(session_id), headers={**headers, **SSE})
            unpace(session_id)

            assert plain.status_code == 200 and plain.headers["content-type"].startswith("application/json")
            assert streamed.status_code == 200
            assert streamed.headers["content-type"].startswith("text/event-stream")
            assert streamed.headers["cache-control"] == "no-cache"
            events, _pings = parse_sse(streamed.text)
            assert [kind for kind, _ in events] == ["partial", "partial", "final"]
            assert events[0][1] == {"target": "cup"}
            assert events[1][1] == {"first_say": PLAN["steps"][0]["say"]}
            final = events[2][1]
            # Identical to the JSON body apart from the plan identity the second call was issued.
            expected = plain.json()
            assert final["plan_revision"] == expected["plan_revision"] + 1
            assert {k: v for k, v in final.items() if k not in {"plan_id", "plan_revision"}} == \
                {k: v for k, v in expected.items() if k not in {"plan_id", "plan_revision"}}
            stored = next(iter(store.sessions.values())).guide
            assert stored.plan is not None and stored.plan.plan_id == final["plan_id"]
            assert stored.plan_inflight is False
            assert [bool(body.get("stream")) for body in stub.requests] == [False, True]
            # The non-streamed request is the historical one, byte for byte: no stream fields.
            assert "stream" not in stub.requests[0] and "stream_options" not in stub.requests[0]

    asyncio.run(exercise())


def test_streamed_truncation_is_an_error_event_with_the_json_body_and_status(env) -> None:  # noqa: F811
    async def exercise() -> None:
        use(StreamStub(sse_lines(ARGS, finish_reason="length")))
        async with api_client() as api:
            headers, session_id = await open_session(api)
            streamed = await api.post("/api/guide/plan", json=plan_body(session_id), headers={**headers, **SSE})
            assert streamed.status_code == 200
            events, _ = parse_sse(streamed.text)
            # The hints were already out; the terminal event says the answer is refused.
            assert [kind for kind, _ in events] == ["partial", "partial", "error"]
            assert events[-1][1] == {"error": "invalid_provider_output", "reason": "truncated", "status": 502}
            guide = next(iter(store.sessions.values())).guide
            assert guide.plan is None and guide.plan_inflight is False

    asyncio.run(exercise())


def test_streamed_schema_refusal_matches_the_json_route(env) -> None:  # noqa: F811
    bad = json.dumps(mutate(PLAN, steps=[]), ensure_ascii=False)

    async def exercise() -> None:
        use(StreamStub(sse_lines(bad), answers={"guide_plan": mutate(PLAN, steps=[])}))
        async with api_client() as api:
            headers, session_id = await open_session(api)
            plain = await api.post("/api/guide/plan", json=plan_body(session_id), headers=headers)
            unpace(session_id)
            streamed = await api.post("/api/guide/plan", json=plan_body(session_id), headers={**headers, **SSE})
            events, _ = parse_sse(streamed.text)
            assert events == [("partial", {"target": "cup"}), ("error", {**plain.json(), "status": plain.status_code})]
            assert plain.status_code == 502

    asyncio.run(exercise())


def test_streamed_rate_limit_is_an_error_event(env) -> None:  # noqa: F811
    async def exercise() -> None:
        use(StreamStub(sse_lines(ARGS)))
        async with api_client() as api:
            headers, session_id = await open_session(api)
            first = await api.post("/api/guide/plan", json=plan_body(session_id), headers={**headers, **SSE})
            second = await api.post("/api/guide/plan", json=plan_body(session_id), headers={**headers, **SSE})
            assert parse_sse(first.text)[0][-1][0] == "final"
            assert parse_sse(second.text)[0] == [("error", {"error": "rate_limited", "status": 429})]

    asyncio.run(exercise())


def test_checks_before_the_stream_answer_as_plain_json(env) -> None:  # noqa: F811
    async def exercise() -> None:
        stub = use(StreamStub(sse_lines(ARGS)))
        async with api_client() as api:
            headers, session_id = await open_session(api)
            refused = await api.post("/api/guide/plan", json=plan_body(session_id, consent_ai=False),
                                     headers={**headers, **SSE})
            assert refused.status_code == 400 and refused.json() == {"error": "ai_consent_required"}
            stub.provider.api_key = ""
            keyless = await api.post("/api/guide/plan", json=plan_body(session_id), headers={**headers, **SSE})
            assert keyless.status_code == 503 and keyless.json() == {"error": "service_unavailable"}
            assert stub.requests == []

    asyncio.run(exercise())


def test_streamed_plan_holds_the_in_flight_guard_and_pings(env) -> None:  # noqa: F811
    env.setattr(main, "STREAM_HEARTBEAT_S", 0.01)

    async def exercise() -> None:
        stub = use(StreamStub(sse_lines(ARGS, pieces=40), hold_at=5))
        async with api_client() as api:
            headers, session_id = await open_session(api)
            pending = asyncio.create_task(
                api.post("/api/guide/plan", json=plan_body(session_id), headers={**headers, **SSE}))
            await stub.held.wait()
            busy = await api.post("/api/guide/plan", json=plan_body(session_id), headers=headers)
            assert busy.status_code == 503 and busy.json() == {"error": "provider_busy"}
            await asyncio.sleep(0.05)
            stub.release.set()
            streamed = await pending
            events, pings = parse_sse(streamed.text)
            assert pings >= 1
            assert events[-1][0] == "final"
            assert len(stub.requests) == 1

    asyncio.run(exercise())


def test_client_disconnect_mid_stream_does_not_corrupt_the_plan_store(env) -> None:  # noqa: F811
    """The client leaves after the first-say hint; the call still finishes and is stored (as the JSON path
    would), the in-flight flag, the session lock and the provider slot are released, and the next plan works."""
    say_line = None
    lines = sse_lines(ARGS, pieces=40)
    for index in range(len(lines)):
        if plan_stream_hints("".join(
                json.loads(line[6:])["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                for line in lines[1:index + 1] if line.startswith(b"data: {") and b'"tool_calls"' in line))[1]:
            say_line = index + 1
            break
    assert say_line is not None

    async def exercise() -> None:
        stub = use(StreamStub(lines, hold_at=say_line))
        async with api_client() as api:
            headers, session_id = await open_session(api)
        cookie = headers["Cookie"]
        body = json.dumps(plan_body(session_id)).encode()
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/api/guide/plan", "raw_path": b"/api/guide/plan", "root_path": "",
            "query_string": b"", "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000),
            "headers": [(b"host", b"127.0.0.1:8000"), (b"origin", ORIGIN["Origin"].encode()),
                        (b"cookie", cookie.encode()), (b"content-type", b"application/json"),
                        (b"accept", b"text/event-stream"), (b"content-length", str(len(body)).encode())],
        }
        disconnect = asyncio.Event()
        sent: list[dict[str, Any]] = []
        delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)
            if b"first_say" in message.get("body", b""):
                disconnect.set()

        await asyncio.wait_for(app(scope, receive, send), 5)
        streamed = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()
        events, _ = parse_sse(streamed)
        assert [kind for kind, _ in events] == ["partial", "partial"]  # no terminal event reached the client
        assert sent[0]["status"] == 200

        session = next(iter(store.sessions.values()))
        assert session.guide.plan_inflight is True  # the call is still running
        stub.release.set()
        for _ in range(500):
            if not main._plan_tasks:
                break
            await asyncio.sleep(0.01)
        assert not main._plan_tasks
        assert session.guide.plan_inflight is False
        assert not session.lock.locked()
        assert not store.provider_slots.locked()
        stored = session.guide.plan
        assert stored is not None and stored.plan_revision == 1
        assert [step.id for step in stored.steps] == ["s1", "s2"]

        unpace(session_id)
        async with api_client() as api:
            again = await api.post("/api/guide/plan", json=plan_body(session_id), headers=headers)
            assert again.status_code == 200, again.text
            assert again.json()["plan_revision"] == 2

    asyncio.run(exercise())


def test_without_accept_the_route_is_plain_json_even_when_the_client_accepts_both(env) -> None:  # noqa: F811
    async def exercise() -> None:
        stub = use(StreamStub(sse_lines(ARGS)))
        async with api_client() as api:
            headers, session_id = await open_session(api)
            plain = await api.post("/api/guide/plan", json=plan_body(session_id),
                                   headers={**headers, "Accept": "application/json"})
            unpace(session_id)
            both = await api.post("/api/guide/plan", json=plan_body(session_id),
                                  headers={**headers, "Accept": "application/json, text/event-stream;q=0.9"})
            assert plain.headers["content-type"].startswith("application/json")
            assert both.headers["content-type"].startswith("text/event-stream")
            assert [bool(body.get("stream")) for body in stub.requests] == [False, True]

    asyncio.run(exercise())


def test_streamed_plan_drops_ungrounded_evidence_before_validation(env) -> None:  # noqa: F811
    """The streamed path shares ``guide_plan_answer``, so the raw citation filter applies there too."""
    answer = mutate(PLAN, steps=[review_step(1, evidence={"material_id": "m13"})])

    async def exercise() -> None:
        use(StreamStub(sse_lines(json.dumps(answer, ensure_ascii=False))))
        async with api_client() as api:
            headers, session_id = await open_session(api)
            streamed = await api.post("/api/guide/plan", json=plan_body(session_id), headers={**headers, **SSE})
            assert streamed.status_code == 200
            events, _pings = parse_sse(streamed.text)
            assert events[-1][0] == "final"
            assert events[-1][1]["steps"][0]["evidence"] is None
            assert next(iter(store.sessions.values())).guide.plan.steps[0].evidence is None

    asyncio.run(exercise())

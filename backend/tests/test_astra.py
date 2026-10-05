"""OpenAI Plan transport contracts; fake API keys and in-memory HTTP only."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend.app.astra import (
    AstraProvider, MAX_FUNCTION_ARGUMENTS_BYTES, OPENAI_RESPONSES_URL,
)
from backend.app.guide_contracts import GuidePlanRequest
from backend.app.provider import (
    GuideStreamFinal, InvalidOutputStage, ProviderConfigError, ProviderInvalidOutput,
    ProviderRateLimited, ProviderUnavailable,
)
from backend.tests.test_guide import GOAL, PLAN, make_test_jpeg


def request(**overrides) -> GuidePlanRequest:
    body = {"session_id": "11111111-1111-4111-8111-111111111111", "consent_ai": True,
            "plan_model": "astra:high", "scene": {"frame_id": "cam-1", "image_base64": make_test_jpeg()},
            "user_goal": GOAL}
    body.update(overrides)
    return GuidePlanRequest(**body)


def call(arguments=PLAN, **overrides):
    return {"id": "fc_1", "call_id": "call_1", "type": "function_call", "name": "guide_plan",
            "status": "completed", "arguments": json.dumps(arguments, ensure_ascii=False), **overrides}


def sse(*events):
    return "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events)


def stream(item=None):
    item = item or call()
    return sse(
        {"type": "response.output_item.added", "item": {**item, "arguments": "", "status": "in_progress"}},
        {"type": "response.reasoning_summary_text.delta", "delta": "PRIVATE_REASONING"},
        {"type": "response.output_text.delta", "delta": "NOT_A_FUNCTION"},
        {"type": "response.function_call_arguments.delta", "item_id": item["id"], "delta": item["arguments"]},
        {"type": "response.output_item.done", "item": item},
        {"type": "response.completed", "response": {"status": "completed", "output": [item],
            "usage": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
                      "input_tokens_details": {"cached_tokens": 20},
                      "output_tokens_details": {"reasoning_tokens": 12}}}},
    )


def response(content=None):
    return httpx.Response(200, content=content if content is not None else stream(),
                          headers={"content-type": "text/event-stream"})


def test_plan_uses_api_issuer_and_only_function_arguments_reach_sse():
    async def exercise():
        seen = []

        def handle(req):
            assert str(req.url) == OPENAI_RESPONSES_URL
            assert req.headers["authorization"] == "Bearer test-openai-key"
            assert "ChatGPT-Account-ID" not in req.headers
            body = json.loads(req.content)
            assert body["model"] == "gpt-6-astra" and body["reasoning"] == {"effort": "high"}
            assert body["store"] is False and body["parallel_tool_calls"] is False
            assert body["tool_choice"] == {"type": "function", "name": "guide_plan"}
            assert body["max_output_tokens"] > 0
            seen.append(req)
            return response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            provider = AstraProvider(client, api_key="test-openai-key")
            pieces = [piece async for piece in provider.guide_plan_stream(request(), "prepared-image")]
        final = pieces[-1]
        assert isinstance(final, GuideStreamFinal) and final.arguments == PLAN
        assert json.loads("".join(pieces[:-1])) == PLAN
        assert len(seen) == 1
        assert final.usage.input_tokens == 100 and final.usage.output_tokens == 40
        assert final.usage.total_tokens == 140 and final.usage.thought_tokens == 12
        assert final.usage.cached_input_tokens == 20  # reasoning is already inside output, not added twice

    asyncio.run(exercise())


def test_api_key_file_can_be_installed_without_reusing_codex_auth(tmp_path, monkeypatch):
    key_file = tmp_path / "openai.key"
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(key_file))
    # A valid-looking Codex login must not make the API-key lane ready.
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "auth.json").write_text(json.dumps({"tokens": {
        "access_token": "not-an-api-key", "refresh_token": "never-refresh", "account_id": "account"}}))
    monkeypatch.setenv("HOME", str(tmp_path))

    async def exercise():
        def handle(req):
            assert req.headers["authorization"] == "Bearer installed-api-key"
            return response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            provider = AstraProvider(client)
            assert not provider.configured
            with pytest.raises(ProviderConfigError):
                await provider.guide_plan(request(), "image")
            key_file.write_text("installed-api-key\n")
            assert provider.configured
            output, _usage = await provider.guide_plan(request(), "image")
            assert output == PLAN
            key_file.unlink()
            assert not provider.configured

    asyncio.run(exercise())


@pytest.mark.parametrize("status,error", [(401, ProviderConfigError), (403, ProviderConfigError),
                                          (429, ProviderRateLimited), (500, ProviderUnavailable),
                                          (307, ProviderUnavailable)])
def test_api_failure_never_retries_falls_back_or_follows_redirect(status, error):
    async def exercise():
        seen = []

        def handle(req):
            seen.append(str(req.url))
            return httpx.Response(status, headers={"location": "https://untrusted.invalid/key", "retry-after": "7"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
            with pytest.raises(error) as exc:
                await AstraProvider(client, api_key="test-only").guide_plan(request(), "image")
            if status == 429:
                assert exc.value.retry_after == "7"
        assert seen == [OPENAI_RESPONSES_URL]

    asyncio.run(exercise())


@pytest.mark.parametrize("content,stage", [
    (sse({"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}),
     InvalidOutputStage.TRUNCATED),
    (sse({"type": "response.completed", "response": {"status": "completed", "output": []}}),
     InvalidOutputStage.TOOL_ENVELOPE),
    (stream(call(name="guide_follow")), InvalidOutputStage.TOOL_ENVELOPE),
    (stream(call(arguments="[1,2]")), InvalidOutputStage.TOOL_JSON),
    (sse({"type": "response.output_item.done", "item": call()}), InvalidOutputStage.ENVELOPE),
    (sse({"type": "response.output_item.done", "item": call()},
         {"type": "response.completed", "response": {"status": "completed", "output": [call(), call(id="fc_2")]}}),
     InvalidOutputStage.TOOL_ENVELOPE),
    (sse({"type": "response.function_call_arguments.delta", "delta": "x" * (MAX_FUNCTION_ARGUMENTS_BYTES + 1)}),
     InvalidOutputStage.TOOL_ENVELOPE),
], ids=["truncated", "prose-only", "wrong-tool", "non-object", "interrupted", "extra-call", "oversized-delta"])
def test_incomplete_or_unusable_response_is_never_accepted(content, stage):
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _req: response(content))) as client:
            with pytest.raises(ProviderInvalidOutput) as exc:
                await AstraProvider(client, api_key="test-only").guide_plan(request(), "image")
            assert exc.value.stage == stage

    asyncio.run(exercise())


def test_confirm_preserves_before_current_image_order_and_plan_policy():
    from backend.app.guide import GuideSessionState
    from backend.app.guide_contracts import GuideConfirmRequest, GuideStep
    from backend.tests.test_guide import confirm_body, CONFIRM

    stored = GuideSessionState().install(steps=[GuideStep.model_validate(s) for s in PLAN["steps"]],
        goal_when=PLAN["goal_when"], user_goal=GOAL, context=None, task_revision=1, plan_model="astra:high")
    req_body = confirm_body("11111111-1111-4111-8111-111111111111", {
        **PLAN, "plan_id": stored.plan_id, "plan_revision": stored.plan_revision}, trigger="target_left")
    req_body["before_scene"] = {"frame_id": "before", "image_base64": make_test_jpeg()}
    req = GuideConfirmRequest.model_validate(req_body)

    async def exercise():
        def handle(post):
            body = json.loads(post.content)
            images = [p["image_url"] for p in body["input"][0]["content"] if p["type"] == "input_image"]
            assert images == ["data:image/jpeg;base64,BEFORE", "data:image/jpeg;base64,CURRENT"]
            assert body["tool_choice"]["name"] == "guide_confirm"
            return response(stream(call({**CONFIRM, "inferred_done": "unsure"}, name="guide_confirm")))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            out, _usage = await AstraProvider(client, api_key="test-only").guide_confirm(req, stored, "CURRENT", "BEFORE")
            assert out["inferred_done"] == "unsure"

    asyncio.run(exercise())


def test_cancelling_a_call_closes_its_response():
    class WaitingStream(httpx.AsyncByteStream):
        def __init__(self):
            self.started = asyncio.Event()
            self.closed = False

        async def __aiter__(self):
            self.started.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            self.closed = True

    async def exercise():
        waiting = WaitingStream()
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, stream=waiting))) as client:
            task = asyncio.create_task(AstraProvider(client, api_key="test-only").guide_plan(request(), "image"))
            await waiting.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert waiting.closed

    asyncio.run(exercise())


def test_streamed_credit_exhaustion_is_a_configuration_failure():
    """Replay the real API's HTTP-200/SSE billing failure, without another paid call."""
    content = sse(
        {"type": "error", "error": {"code": "credit_balance_exhausted", "type": "insufficient_quota"}},
        {"type": "response.failed", "response": {"error": {"code": "credit_balance_exhausted"}}},
    )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _req: response(content))) as client:
            with pytest.raises(ProviderConfigError):
                await AstraProvider(client, api_key="test-only").guide_plan(request(), "image")

    asyncio.run(exercise())


#: One completed hosted web search with a real source (`include: web_search_call.action.sources`).
WEB = {"id": "ws_1", "type": "web_search_call", "status": "completed",
       "action": {"type": "search", "query": "led driver pinout",
                  "sources": [{"type": "url", "url": "https://example.com/driver", "title": "Driver manual"},
                              {"type": "url", "url": "https://example.org/partial", "title": "Partial page"}]}}


def test_research_plan_uses_hosted_web_search_and_returns_verified_sources():
    from backend.app import intent
    from backend.app.guide_contracts import GuidePlanToolOutput, ResearchSource

    opened = {"id": "ws_2", "type": "web_search_call", "status": "completed",
              "action": {"type": "open_page", "url": "https://opened.example/page"}}
    retained = ResearchSource(url="https://retained.example/manual", title="보관 매뉴얼",
                              summary="이전 조사에서 확인한 내용")
    req = request(assisted=True, research=True)
    req._research_sources = [retained]
    arguments = {**PLAN, "research_sources": [
        {"url": "https://example.com/driver", "title": "Driver manual", "summary": "핀 배열 설명"},
        {"url": "https://forged.invalid/made-up", "title": "지어낸 출처", "summary": "조사하지 않은 내용"},
        {"url": "javascript:alert(1)", "title": "스크립트", "summary": "주소가 아님"},
        {"url": "https://opened.example/page", "title": "Opened page", "summary": "열어서 확인한 내용"},
        {"url": "https://retained.example/manual", "title": retained.title, "summary": retained.summary},
        {"url": "https://example.org/partial", "title": "Partial page"},
    ]}
    item = call(arguments)

    async def exercise():
        seen = []

        def handle(post):
            body = json.loads(post.content)
            assert body["instructions"] == intent.ASSISTED_PLAN_SYSTEM_PROMPT
            assert body["tools"][0] == {"type": "web_search"} and body["tools"][1]["name"] == "guide_plan"
            assert body["tool_choice"] == "auto" and body["max_tool_calls"] == 2
            assert body["include"] == ["web_search_call.action.sources"]
            assert body["store"] is False and body["parallel_tool_calls"] is False
            assert body["reasoning"] == {"effort": "high"}
            seen.append(body)
            return response(sse(
                {"type": "response.output_item.added", "item": {**WEB, "status": "in_progress"}},
                {"type": "response.reasoning_summary_text.delta", "delta": "PRIVATE_REASONING"},
                {"type": "response.output_item.done", "item": WEB},
                {"type": "response.output_item.done", "item": opened},
                {"type": "response.function_call_arguments.delta", "item_id": item["id"],
                 "delta": item["arguments"]},
                {"type": "response.output_item.done", "item": item},
                {"type": "response.completed", "response": {"status": "completed", "output": [WEB, opened, item],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}},
            ))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            out, usage = await AstraProvider(client, api_key="test-only").guide_plan(req, "prepared-image")
        # Kept: the searched URL, the opened page and the request's own retained source. Dropped: a forged URL,
        # a non-http "URL", and a source whose fields do not satisfy the shared contract — never repaired.
        assert [source["url"] for source in out["research_sources"]] == [
            "https://example.com/driver", "https://opened.example/page", "https://retained.example/manual"]
        assert out["research_sources"][0] == {"url": "https://example.com/driver",
                                              "title": "Driver manual", "summary": "핀 배열 설명"}
        assert "forged.invalid" not in json.dumps(out, ensure_ascii=False)
        assert "javascript" not in json.dumps(out, ensure_ascii=False)
        # The production route validates the raw answer against this exact contract.
        strict = GuidePlanToolOutput.model_validate(out)
        assert [source.url for source in strict.research_sources] == [
            "https://example.com/driver", "https://opened.example/page", "https://retained.example/manual"]
        assert out["steps"] == PLAN["steps"] and out["goal_when"] == PLAN["goal_when"]
        assert usage.total_tokens == 15 and len(seen) == 1

    asyncio.run(exercise())


def test_assisted_plan_without_research_keeps_the_forced_function_call():
    from backend.app import intent

    async def exercise():
        def handle(post):
            body = json.loads(post.content)
            assert body["instructions"] == intent.ASSISTED_PLAN_SYSTEM_PROMPT
            assert [tool["type"] for tool in body["tools"]] == ["function"]
            assert body["tool_choice"] == {"type": "function", "name": "guide_plan"}
            assert "include" not in body and "max_tool_calls" not in body
            return response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            out, _usage = await AstraProvider(client, api_key="test-only").guide_plan(
                request(assisted=True), "image")
        assert out == PLAN

    asyncio.run(exercise())


@pytest.mark.parametrize("sources", [
    [{"url": "https://forged.invalid/a", "title": "a", "summary": "b"}, "https://not-a-dict"],
    "https://not-a-list",
    None,
], ids=["forged-and-scalar", "not-a-list", "null"])
def test_plain_plan_discards_every_source_it_cannot_ground(sources):
    from backend.app import intent

    item = call({**PLAN, "research_sources": sources})

    async def exercise():
        def handle(post):
            body = json.loads(post.content)
            assert body["instructions"] == intent.PLAN_SYSTEM_PROMPT
            assert [tool["type"] for tool in body["tools"]] == ["function"]
            assert "include" not in body and "max_tool_calls" not in body
            return response(stream(item))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            out, _usage = await AstraProvider(client, api_key="test-only").guide_plan(request(), "image")
        assert out["research_sources"] == []
        assert out["steps"] == PLAN["steps"]

    asyncio.run(exercise())


def test_reference_images_stay_separate_from_the_current_scene():
    reference = make_test_jpeg()
    req = request(assisted=True, reference_images=[
        {"frame_id": "ref-a", "image_base64": reference, "label": "closeup-1"},
        {"frame_id": "ref-b", "image_base64": make_test_jpeg()},
    ])

    async def exercise():
        def handle(post):
            body = json.loads(post.content)
            content = body["input"][0]["content"]
            assert [part["type"] for part in content] == [
                "input_text", "input_text", "input_image", "input_text", "input_image",
                "input_text", "input_image"]
            assert "[별도 참고 사진 1]" in content[1]["text"]
            assert "ref-a" in content[1]["text"] and "label=closeup-1" in content[1]["text"]
            assert content[2]["image_url"] == f"data:image/jpeg;base64,{reference}"
            assert "[별도 참고 사진 2]" in content[3]["text"] and "ref-b" in content[3]["text"]
            assert content[5]["text"].startswith("[현재 장면 사진]") and "참고" not in content[5]["text"]
            assert content[6]["image_url"] == "data:image/jpeg;base64,PREPARED-CURRENT"
            return response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            out, _usage = await AstraProvider(client, api_key="test-only").guide_plan(req, "PREPARED-CURRENT")
        assert out == PLAN

    asyncio.run(exercise())


def test_research_events_never_leak_into_streamed_hints():
    item = call()

    async def exercise():
        def handle(post):
            return response(sse(
                {"type": "response.output_item.added", "item": {**WEB, "status": "in_progress"}},
                {"type": "response.reasoning_summary_text.delta", "delta": "PRIVATE_REASONING"},
                {"type": "response.output_text.delta", "delta": "NOT_A_HINT"},
                {"type": "response.output_text.annotation.added",
                 "annotation": {"type": "url_citation", "url": "https://example.com/driver",
                                "title": "Driver manual"}},
                {"type": "response.output_item.done", "item": WEB},
                {"type": "response.function_call_arguments.delta", "item_id": item["id"],
                 "delta": item["arguments"]},
                {"type": "response.output_item.done", "item": item},
                {"type": "response.completed", "response": {"status": "completed", "output": [WEB, item]}},
            ))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            pieces = [piece async for piece in AstraProvider(client, api_key="test-only").guide_plan_stream(
                request(assisted=True, research=True), "image")]
        hints = "".join(pieces[:-1])
        assert hints == item["arguments"]
        assert "PRIVATE_REASONING" not in hints and "example.com" not in hints and "NOT_A_HINT" not in hints
        assert isinstance(pieces[-1], GuideStreamFinal) and pieces[-1].arguments == PLAN

    asyncio.run(exercise())


@pytest.mark.parametrize("content,stage", [
    (sse({"type": "response.output_item.done", "item": WEB},
         {"type": "response.completed", "response": {"status": "completed", "output": [WEB]}}),
     InvalidOutputStage.TOOL_ENVELOPE),
    (sse({"type": "response.output_item.done", "item": WEB},
         {"type": "response.output_item.done", "item": call()},
         {"type": "response.output_item.done", "item": call(id="fc_2")},
         {"type": "response.completed",
          "response": {"status": "completed", "output": [WEB, call(), call(id="fc_2")]}}),
     InvalidOutputStage.TOOL_ENVELOPE),
    (sse({"type": "response.output_item.done", "item": WEB},
         {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}),
     InvalidOutputStage.TRUNCATED),
    (sse({"type": "response.output_item.done", "item": WEB}), InvalidOutputStage.ENVELOPE),
], ids=["web-only-no-function", "web-plus-extra-call", "web-then-truncated", "web-then-unterminated"])
def test_web_research_never_replaces_the_one_complete_function_call(content, stage):
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _req: response(content))) as client:
            with pytest.raises(ProviderInvalidOutput) as exc:
                await AstraProvider(client, api_key="test-only").guide_plan(
                    request(assisted=True, research=True), "image")
            assert exc.value.stage == stage

    asyncio.run(exercise())

"""Tests for provider transport envelopes, malformed response type safety, and usage parsing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from backend.app.provider import (
    DeepSeekProvider,
    GeminiProvider,
    InvalidOutputStage,
    ProviderInvalidOutput,
    ProviderRateLimited,
    ProviderUnavailable,
    UnavailableStage,
    parse_usage,
)
from backend.app.tracking_contracts import TrackSelectRequest


#: The transport is shared by every call; the startup target selection is the simplest call that uses it.
TOOL_NAME = "track_target"

TOOL_OUTPUT = {
    "status": "selected",
    "target": "laptop",
    "rationale": "사용자가 노트북을 다루고 있습니다.",
}


def make_select_request() -> TrackSelectRequest:
    return TrackSelectRequest.model_validate({
        "session_id": "123e4567-e89b-42d3-a456-426614174000",
        "consent_ai": True,
        "scene": {"frame_id": "frame-1", "image_base64": "valid_jpeg"},
        "user_goal": "다음 작업 안내",
        "context": "합성 실습 맥락",
    })


def test_deepseek_sends_nested_envelope_and_parses_cache_usage() -> None:
    sent_request: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_request
        sent_request = json.loads(request.content)
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": TOOL_NAME,
                                    "arguments": json.dumps(TOOL_OUTPUT),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        }
        return httpx.Response(200, json=body)

    async def exercise() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = DeepSeekProvider(client, api_key="test-key", model="deepseek-flash")
        args, usage = await provider.select_target(make_select_request())
        assert isinstance(args, dict)
        assert args["status"] == "selected"
        assert args["target"] == "laptop"

        tools = sent_request["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == TOOL_NAME
        assert "status" in tools[0]["function"]["parameters"]["properties"]

        user_content = sent_request["messages"][1]["content"]
        prompt = "\n".join(part["text"] for part in user_content if part["type"] == "text")
        assert "현재 장면 frame_id: frame-1" in prompt
        images = [part for part in user_content if part["type"] == "image_url"]
        assert len(images) == 1  # the current scene only
        assert sent_request["messages"][0]["content"].startswith("You are a tracking assistant")

        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cached_input_tokens == 40
        assert usage.thought_tokens == 10
        await client.aclose()

    asyncio.run(exercise())


def test_gemini_sends_flat_envelope_and_parses_call() -> None:
    sent_request: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_request
        sent_request = json.loads(request.content)
        return httpx.Response(200, json={
            "steps": [{"type": "function_call", "name": TOOL_NAME, "arguments": TOOL_OUTPUT}],
            "usage": {"total_input_tokens": 120, "total_output_tokens": 60, "total_tokens": 180},
        })

    async def exercise() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GeminiProvider(client, api_key="test-gemini-key", model="gemini-3.8-flash")
        args, usage = await provider.select_target(make_select_request())
        assert isinstance(args, dict)
        assert args["status"] == "selected"

        tools = sent_request["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == TOOL_NAME
        assert "parameters" in tools[0]
        assert "status" in tools[0]["parameters"]["properties"]
        assert "function" not in tools[0]
        assert sent_request["store"] is False

        contents = sent_request["input"][0]["content"]
        texts = "\n".join(part["text"] for part in contents if part["type"] == "text")
        assert "현재 장면 frame_id: frame-1" in texts
        assert len([part for part in contents if part["type"] == "image"]) == 1

        assert usage is not None
        assert usage.input_tokens == 120
        assert usage.output_tokens == 60
        assert usage.total_tokens == 180
        await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("malformed_body", "stage"),
    [
        ({"choices": []}, InvalidOutputStage.ENVELOPE),
        ({"choices": [{"message": {"tool_calls": []}}]}, InvalidOutputStage.TOOL_ENVELOPE),
        ({"choices": [{"message": {"tool_calls": [{"function": {"name": "wrong_tool", "arguments": "{}"}}]}}]}, InvalidOutputStage.TOOL_ENVELOPE),
        ({"choices": [{"message": {"tool_calls": [{"function": {"name": TOOL_NAME, "arguments": "not_json"}}]}}]}, InvalidOutputStage.TOOL_JSON),
    ],
)
def test_deepseek_malformed_responses_raise_provider_invalid_output(malformed_body: dict[str, Any], stage: InvalidOutputStage) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=malformed_body)

    async def exercise() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = DeepSeekProvider(client, api_key="test-key")
        with pytest.raises(ProviderInvalidOutput) as exc:
            await provider.select_target(make_select_request())
        # The stage names the accepting rule that refused it; the message carries nothing else.
        assert exc.value.stage is stage
        assert str(exc.value) == stage.value
        await client.aclose()

    asyncio.run(exercise())


def test_a_non_json_provider_body_is_an_envelope_failure() -> None:
    """An error page or a proxy's HTML answer is not a completion envelope."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    async def exercise() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = DeepSeekProvider(client, api_key="test-key")
        with pytest.raises(ProviderInvalidOutput) as exc:
            await provider.select_target(make_select_request())
        assert exc.value.stage is InvalidOutputStage.ENVELOPE
        await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("malformed_body", "stage"),
    [
        ({"steps": "not_a_list"}, InvalidOutputStage.ENVELOPE),
        ({"steps": []}, InvalidOutputStage.TOOL_ENVELOPE),
        ({"steps": [{"type": "function_call", "name": "wrong_tool", "arguments": {}}]}, InvalidOutputStage.TOOL_ENVELOPE),
        ({"steps": [{"type": "function_call", "name": TOOL_NAME, "arguments": "not_dict"}]}, InvalidOutputStage.TOOL_ENVELOPE),
    ],
)
def test_gemini_malformed_responses_raise_provider_invalid_output(malformed_body: dict[str, Any], stage: InvalidOutputStage) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=malformed_body)

    async def exercise() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GeminiProvider(client, api_key="test-key")
        with pytest.raises(ProviderInvalidOutput) as exc:
            await provider.select_target(make_select_request())
        assert exc.value.stage is stage
        await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("provider_name", ["deepseek", "gemini"])
def test_upstream_error_status_is_not_a_transport_failure(provider_name: str) -> None:
    """A 5xx from the provider (or a gateway) is an upstream status, not a network failure: an answer
    was produced and refused, which is a different thing to report than 'nothing came back'."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    async def exercise() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = (
            DeepSeekProvider(client, api_key="test-key")
            if provider_name == "deepseek"
            else GeminiProvider(client, api_key="test-key")
        )
        with pytest.raises(ProviderUnavailable) as exc:
            await provider.select_target(make_select_request())
        assert exc.value.stage is UnavailableStage.UPSTREAM_STATUS
        await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("provider_name", ["deepseek", "gemini"])
def test_a_connect_failure_is_a_transport_failure(provider_name: str) -> None:
    """No response at all is the transport stage, and it stays distinguishable from an error status."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    async def exercise() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = (
            DeepSeekProvider(client, api_key="test-key")
            if provider_name == "deepseek"
            else GeminiProvider(client, api_key="test-key")
        )
        with pytest.raises(ProviderUnavailable) as exc:
            await provider.select_target(make_select_request())
        assert exc.value.stage is UnavailableStage.TRANSPORT
        await client.aclose()

    asyncio.run(exercise())


def test_retry_after_header_is_preserved_on_rate_limit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, json={})

    async def exercise() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = DeepSeekProvider(client, api_key="test-key")
        with pytest.raises(ProviderRateLimited) as exc:
            await provider.select_target(make_select_request())
        assert exc.value.retry_after == "7"
        await client.aclose()

    asyncio.run(exercise())


def test_usage_parsing_rejects_bool_and_malformed() -> None:
    # bool is a subclass of int in Python, but StrictInt rejects it
    assert parse_usage({"prompt_tokens": True}) is None
    assert parse_usage({"prompt_tokens": False}) is None
    assert parse_usage("not_a_dict") is None


def test_usage_parsing_preserves_zero() -> None:
    # Legitimate 0 must NOT be dropped by 'or'
    usage = parse_usage({
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
    })
    assert usage is not None
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0
    assert usage.cached_input_tokens == 0


def test_a_truncated_completion_is_refused_instead_of_parsed() -> None:
    """A call that ran into the ceiling cannot be a valid tool call; it must never be partially parsed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{
                "finish_reason": "length",
                "message": {"role": "assistant", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": TOOL_NAME, "arguments": '{"status": "sel'}}]},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 768},
        })

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DeepSeekProvider(client, api_key="test-only", model="deepseek-flash")
            with pytest.raises(ProviderInvalidOutput) as exc:
                await provider.select_target(make_select_request())
            # The output ceiling is its own stage: it is not the same failure as an unusable answer.
            assert exc.value.stage is InvalidOutputStage.TRUNCATED

    asyncio.run(exercise())

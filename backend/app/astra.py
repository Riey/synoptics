"""Plan-only OpenAI Responses adapter: gpt-6-astra on the shared upper planning profile (HIGH reasoning).

Credentials are API keys only (OPENAI_API_KEY or OPENAI_API_KEY_FILE). No Codex login,
OAuth refresh, alternative endpoint, inference retries, or model fallback. The same
strict Responses parser serves JSON and SSE callers; reasoning never becomes a hint.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from backend.app.api_contracts import Usage
from backend.app.provider import (
    UPPER_MAX_TOKENS, UPPER_TIMEOUT_S, GuideStreamFinal, InvalidOutputStage, ProviderConfigError, ProviderError,
    ProviderInvalidOutput, ProviderRateLimited, ProviderTimeout, ProviderUnavailable,
    UnavailableStage, parse_strict_json, valid_http_date,
)

if TYPE_CHECKING:
    from backend.app.guide import StoredPlan
    from backend.app.guide_contracts import GuidePlanRequest, GuideConfirmRequest, GuideTalkRequest

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
#: OpenAI's dial for the upper planning profile; DeepSeek reaches the same profile by enabling thinking.
REASONING_EFFORT = "high"
MAX_FUNCTION_ARGUMENTS_BYTES = 128 * 1024
DEFAULT_KEY_FILE = "~/.config/synoptics/OPENAI_API_KEY.txt"
#: Requested provider-side search limit. Live probing observed more items; this is NOT a hard billing cap.
RESEARCH_MAX_TOOL_CALLS = 2
WEB_SEARCH_INCLUDE = ("web_search_call.action.sources",)
#: The plan contract's own cap on returned sources, applied after verification.
MAX_RESEARCH_SOURCES = 6
#: Defensive bound on URLs retained from one stream's tool metadata.
MAX_TRACKED_WEB_URLS = 256


def load_openai_key() -> str:
    """Read only the explicitly named OpenAI API credential; never log it or read Codex auth."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    path = os.getenv("OPENAI_API_KEY_FILE", "").strip() or DEFAULT_KEY_FILE
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _retry_after(headers: httpx.Headers) -> str:
    delay = headers.get("retry-after", "30")
    return delay if delay.isdecimal() or valid_http_date(delay) else "30"


def _image_part(image_b64: str) -> dict[str, Any]:
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}


def responses_tool(schema: dict[str, Any]) -> dict[str, Any]:
    fn = schema["function"]
    # Keep the application's optional fields optional; validate the returned arguments ourselves.
    return {"type": "function", "name": fn["name"], "description": fn["description"],
            "parameters": fn["parameters"], "strict": False}


def _http_url(value: Any) -> str | None:
    """``value`` as an absolute, credential-free http(s) URL within the source bound, else ``None``."""
    if not isinstance(value, str) or not value or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
        valid = valid and parsed.username is None and parsed.password is None
    except ValueError:
        return None
    return value if valid else None


def _filter_research_sources(arguments: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Drop a model's ``research_sources`` entries that no real web tool result grounded.

    ``allowed`` holds the URLs actually observed from this call's search/open-page metadata plus the request's
    own retained verified sources. A model-written URL is never evidence on its own, so an entry is kept only
    when its URL is in ``allowed`` AND its title/summary satisfy the shared ``ResearchSource`` contract; the
    model's own words are preserved, never rewritten or invented. A malformed list becomes empty.
    """
    if "research_sources" not in arguments:
        return arguments
    from backend.app.guide_contracts import ResearchSource  # contracts import this provider's lane lazily
    raw = arguments.get("research_sources")
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url")
            if not isinstance(url, str) or url not in allowed or url in seen:
                continue
            try:
                source = ResearchSource.model_validate(
                    {"url": url, "title": entry.get("title"), "summary": entry.get("summary")})
            except ValidationError:
                continue
            seen.add(url)
            kept.append(source.model_dump(mode="json"))
    arguments["research_sources"] = kept[:MAX_RESEARCH_SOURCES]
    return arguments


def _nonneg(mapping: Any, key: str) -> int | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    return value if type(value) is int and value >= 0 else None


def responses_usage(data: Any) -> Usage | None:
    """The Responses ``usage`` object as the app's ``Usage``; ``None`` when the provider sent no numbers."""
    if not isinstance(data, dict):
        return None
    input_tokens = _nonneg(data, "input_tokens")
    output_tokens = _nonneg(data, "output_tokens")
    total_tokens = _nonneg(data, "total_tokens")
    cached = _nonneg(data.get("input_tokens_details"), "cached_tokens")
    thoughts = _nonneg(data.get("output_tokens_details"), "reasoning_tokens")
    if all(value is None for value in (input_tokens, output_tokens, total_tokens, cached, thoughts)):
        return None
    try:
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached,
            thought_tokens=thoughts,
        )
    except Exception:
        return None


_RATE_LIMIT_CODES = frozenset({"rate_limit_exceeded", "rate_limited", "rate_limit", "too_many_requests"})
_QUOTA_CODES = frozenset({"insufficient_quota", "credit_balance_exhausted", "quota_exceeded",
                          "usage_limit_reached", "usage_not_included"})
_AUTH_CODES = frozenset({"invalid_api_key", "unauthorized", "invalid_token", "token_expired", "account_deactivated",
                         "authentication_error"})
_REFUSAL_CODES = frozenset({"content_filter", "content_policy_violation", "cyber_policy", "bio_policy",
                            "misalignment_policy_violation", "invalid_prompt"})


def _error_for(code: Any, retry_after: str = "30") -> ProviderError:
    """Map a streamed failure's error code onto the shared provider taxonomy (no code is ever echoed)."""
    token = code.lower() if isinstance(code, str) else ""
    if token in _RATE_LIMIT_CODES:
        return ProviderRateLimited(retry_after)
    if token in _QUOTA_CODES:
        return ProviderConfigError("OpenAI plan cannot serve this request")
    if token in _AUTH_CODES:
        return ProviderConfigError("OpenAI credentials are not accepted")
    if token in _REFUSAL_CODES:
        return ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
    if token == "context_length_exceeded":
        return ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
    if token == "max_output_tokens":
        return ProviderInvalidOutput(InvalidOutputStage.TRUNCATED)
    return ProviderUnavailable(UnavailableStage.UPSTREAM_STATUS)


@dataclass(slots=True)
class _FunctionCall:
    name: str
    call_id: str | None
    arguments: str
    item_id: str | None


class _ResponsesStream:
    """Accumulates one forced function call from Responses SSE ``data:`` lines.

    Fail-closed like the non-streamed envelope: non-JSON data or an event without a string ``type`` is
    ``envelope``; a second function-call item, a different name, or an incomplete call is ``tool_envelope``;
    the turn is only usable after ``response.completed``, and an unterminated stream is ``envelope``. Only
    ``response.function_call_arguments.delta`` pieces are returned for the caller's display hints — reasoning
    and message deltas are consumed and dropped, never surfaced as plan fragments. ``web_search_call`` items and
    ``url_citation`` annotations are retained in ``web_sources`` as verified URL/title metadata; a message's
    prose or hidden reasoning never becomes a hint or a source.
    """

    __slots__ = ("tool_name", "item_id", "call", "completed", "usage", "retry_after", "argument_bytes",
                 "web_sources")

    def __init__(self, tool_name: str, *, retry_after: str = "30") -> None:
        self.tool_name = tool_name
        self.item_id: str | None = None
        self.call: _FunctionCall | None = None
        self.completed = False
        self.usage: Usage | None = None
        self.retry_after = retry_after
        self.argument_bytes = 0
        #: URL -> title for every source a real web tool result named in this call (`""` when it gave none).
        self.web_sources: dict[str, str] = {}

    def line(self, line: str) -> str | None:
        """One SSE line; returns the argument text it adds to the streamed hints (or ``None``)."""
        if not line.startswith("data:"):
            return None  # blank separators, ': keep-alive' comments, `event:` fields
        data = line[5:]
        if data.startswith(" "):
            data = data[1:]
        stripped = data.strip()
        if not stripped or stripped == "[DONE]":
            # The Responses stream is complete only at `response.completed`; a bare terminator ends it
            # without an answer and `finish` refuses the stream.
            return None
        try:
            event = parse_strict_json(data)
        except ProviderInvalidOutput as exc:
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE) from exc
        kind = event.get("type")
        if not isinstance(kind, str):
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        if kind == "response.output_item.added":
            self._added(event)
        elif kind == "response.output_item.done":
            self._done(event)
        elif kind == "response.function_call_arguments.delta":
            return self._arguments_delta(event)
        elif kind == "response.completed":
            self._completed(event)
        elif kind == "response.output_text.annotation.added":
            self._annotations(event.get("annotation"))
        elif kind == "response.incomplete":
            self._incomplete(event)
        elif kind == "response.failed":
            error = event.get("response")
            code = error.get("error", {}).get("code") if isinstance(error, dict) and isinstance(error.get("error"), dict) else None
            raise _error_for(code, self.retry_after)
        elif kind == "error":
            error = event.get("error")
            code = error.get("code") if isinstance(error, dict) else event.get("code")
            raise _error_for(code, self.retry_after)
        # Anything else — response.created, in_progress, output_text/reasoning deltas, metadata — is not a
        # plan fragment and is deliberately consumed without reaching the caller.
        return None

    def _added(self, event: dict[str, Any]) -> None:
        item = event.get("item")
        self._web_item(item)
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return
        if self.call is not None:
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        item_id = item.get("id")
        if self.call is None:
            if self.item_id is None and isinstance(item_id, str):
                self.item_id = item_id
            elif isinstance(item_id, str) and item_id != self.item_id:
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        name = item.get("name")
        if isinstance(name, str) and name and name != self.tool_name:
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)

    def _done(self, event: dict[str, Any]) -> None:
        item = event.get("item")
        self._web_item(item)
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return
        if self.call is not None:
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        name = item.get("name")
        if not isinstance(name, str) or name != self.tool_name:
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        arguments = item.get("arguments")
        if not isinstance(arguments, str) or not arguments.strip():
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        status = item.get("status")
        if status is not None and status != "completed":
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        item_id = item.get("id")
        if isinstance(item_id, str) and self.item_id is not None and item_id != self.item_id:
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        call_id = item.get("call_id")
        self.call = _FunctionCall(name, call_id if isinstance(call_id, str) else None, arguments,
                                  item_id if isinstance(item_id, str) else self.item_id)

    def _arguments_delta(self, event: dict[str, Any]) -> str | None:
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return None
        self.argument_bytes += len(delta.encode("utf-8"))
        if self.argument_bytes > MAX_FUNCTION_ARGUMENTS_BYTES:
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        item_id = event.get("item_id")
        if isinstance(item_id, str):
            if self.item_id is None:
                self.item_id = item_id
            elif item_id != self.item_id:
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        return delta

    def _completed(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, dict):
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        # Billing evidence survives refusal of an unusable final answer.
        self.usage = responses_usage(response.get("usage"))
        status = response.get("status")
        if status is not None and status != "completed":
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        if "output" in response:
            output = response["output"]
            if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
            calls = [item for item in output if item.get("type") == "function_call"]
            if len(calls) != 1 or self.call is None:
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
            call = calls[0]
            if (call.get("name") != self.call.name or call.get("arguments") != self.call.arguments
                    or call.get("status") not in (None, "completed")
                    or call.get("id") != self.call.item_id):
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
            for item in output:
                self._web_item(item)
        self.completed = True

    def _incomplete(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        if isinstance(response, dict):
            self.usage = responses_usage(response.get("usage"))
        reason = None
        if isinstance(response, dict) and isinstance(response.get("incomplete_details"), dict):
            reason = response["incomplete_details"].get("reason")
        if reason == "content_filter":
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        raise ProviderInvalidOutput(InvalidOutputStage.TRUNCATED)

    def _web_item(self, item: Any) -> None:
        """Retain source metadata from one output item: web-search actions and cited-URL annotations."""
        if not isinstance(item, dict):
            return
        if item.get("type") == "web_search_call":
            self._web_action(item.get("action"))
        elif item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        self._annotations(part.get("annotations"))

    def _web_action(self, action: Any) -> None:
        """The ``search``/``open_page``/``find_in_page`` action's own URL and its ``sources`` list."""
        if not isinstance(action, dict):
            return
        self._record_url(action.get("url"), None)
        sources = action.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    self._record_url(source.get("url"), source.get("title"))

    def _annotations(self, annotations: Any) -> None:
        """``url_citation`` annotations the tool attached to a message; the model's prose is never parsed."""
        if not isinstance(annotations, list):
            return
        for annotation in annotations:
            if isinstance(annotation, dict) and annotation.get("type") == "url_citation":
                self._record_url(annotation.get("url"), annotation.get("title"))

    def _record_url(self, url: Any, title: Any) -> None:
        verified = _http_url(url)
        if verified is None or len(self.web_sources) >= MAX_TRACKED_WEB_URLS or self.web_sources.get(verified):
            return
        self.web_sources[verified] = title if isinstance(title, str) else ""

    def finish(self) -> GuideStreamFinal:
        if not self.completed:
            # The stream ended (or the reader stopped) before `response.completed`: no complete answer exists.
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        if self.call is None:
            # A completed turn with no function call is prose, a refusal, or a different tool.
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        arguments = self.call.arguments
        if len(arguments.encode("utf-8")) > MAX_FUNCTION_ARGUMENTS_BYTES:
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        return GuideStreamFinal(parse_strict_json(arguments), self.usage)


class AstraProvider:
    """A second upper lane for Plan tasks; basic and follower providers remain independent."""

    provider_name = "openai"
    model = "gpt-6-astra"

    def __init__(self, client: httpx.AsyncClient, *, api_key: str | None = None) -> None:
        self.client = client
        self._key = api_key

    def _api_key(self) -> str:
        return self._key.strip() if self._key is not None else load_openai_key()

    @property
    def configured(self) -> bool:
        return bool(self._api_key())

    def _payload(self, *, system_prompt: str, text: str, frame_id: str, image_b64: str,
                 tool_schema: dict[str, Any], tool_name: str,
                 before: tuple[str, str] | None = None,
                 references: Sequence[tuple[str, str | None, str]] = (), research: bool = False,
                 ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        if before is not None:
            content.extend([{"type": "input_text", "text": f"[이전 장면 사진] frame_id: {before[0]} (대상 추적을 시작할 때의 화면)"},
                            _image_part(before[1])])
        # Reference photos are separate evidence, never merged into the current-scene collage; each keeps the
        # same number and label the plan prompt gives it.
        for index, (ref_frame, ref_label, ref_b64) in enumerate(references, start=1):
            label = f", label={ref_label}" if ref_label else ""
            content.extend([{"type": "input_text",
                             "text": f"[별도 참고 사진 {index}] frame_id: {ref_frame}{label} (현재 장면이 아닌 참고 자료)"},
                            _image_part(ref_b64)])
        content.extend([{"type": "input_text", "text": f"[현재 장면 사진] frame_id: {frame_id} (이 이미지가 판단의 근거)"},
                        _image_part(image_b64)])
        if research:
            # One HTTP request with hosted research: at most `RESEARCH_MAX_TOOL_CALLS` built-in searches, then
            # the same single guide_plan function call. `auto` lets the model research before answering.
            tools: list[dict[str, Any]] = [{"type": "web_search"}, responses_tool(tool_schema)]
            tool_choice: Any = "auto"
        else:
            tools = [responses_tool(tool_schema)]
            tool_choice = {"type": "function", "name": tool_name}
        payload: dict[str, Any] = {
            "model": self.model, "stream": True, "store": False,
            "instructions": system_prompt,
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
            "reasoning": {"effort": REASONING_EFFORT},
            "max_output_tokens": UPPER_MAX_TOKENS,
        }
        if research:
            payload["max_tool_calls"] = RESEARCH_MAX_TOOL_CALLS
            payload["include"] = list(WEB_SEARCH_INCLUDE)
        return payload

    async def _stream(self, payload: dict[str, Any], tool_name: str,
                      state: _ResponsesStream | None = None) -> AsyncIterator[str | GuideStreamFinal]:
        key = self._api_key()
        if not key:
            raise ProviderConfigError("OpenAI API key is not configured")
        state = state if state is not None else _ResponsesStream(tool_name)
        response: httpx.Response | None = None
        deadline = asyncio.get_running_loop().time() + UPPER_TIMEOUT_S
        try:
            async with asyncio.timeout_at(deadline):
                request = self.client.build_request(
                    "POST", OPENAI_RESPONSES_URL, json=payload,
                    headers={"Authorization": f"Bearer {key}", "Accept": "text/event-stream"},
                    timeout=httpx.Timeout(UPPER_TIMEOUT_S),
                )
                response = await self.client.send(request, stream=True, follow_redirects=False)
            if response.status_code in (401, 403):
                raise ProviderConfigError("OpenAI API credentials or model access were rejected")
            if response.status_code == 429:
                raise ProviderRateLimited(_retry_after(response.headers))
            if response.status_code != 200:
                raise ProviderUnavailable(UnavailableStage.UPSTREAM_STATUS)
            state.retry_after = _retry_after(response.headers)
            lines = response.aiter_lines()
            while not state.completed:
                try:
                    async with asyncio.timeout_at(deadline):
                        line = await anext(lines)
                except StopAsyncIteration:
                    break
                fragment = state.line(line)
                if fragment is not None:
                    yield fragment
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ProviderTimeout("OpenAI Plan request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(UnavailableStage.TRANSPORT) from exc
        finally:
            if response is not None:
                await response.aclose()
        yield state.finish()

    async def _call(self, payload: dict[str, Any], tool_name: str) -> tuple[dict[str, Any], Usage | None]:
        final = None
        async for item in self._stream(payload, tool_name):
            if isinstance(item, GuideStreamFinal):
                final = item
        if final is None:
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        return final.arguments, final.usage

    def _plan_payload(self, request: GuidePlanRequest, image_b64: str) -> dict[str, Any]:
        from backend.app import intent
        system_prompt = intent.ASSISTED_PLAN_SYSTEM_PROMPT if request.assisted else intent.PLAN_SYSTEM_PROMPT
        references = [(image.frame_id, image.label, image.image_base64) for image in request.reference_images]
        return self._payload(system_prompt=system_prompt, text=intent.plan_prompt(request),
                             frame_id=request.scene.frame_id, image_b64=image_b64,
                             tool_schema=intent.PLAN_TOOL_SCHEMA, tool_name=intent.PLAN_TOOL_NAME,
                             references=references, research=request.research)

    @staticmethod
    def _verified_plan_arguments(request: GuidePlanRequest, arguments: dict[str, Any],
                                 state: _ResponsesStream) -> dict[str, Any]:
        """Keep only the model's sources whose URL this call's tool results or the request's own verified
        research grounded; a URL the model wrote on its own is never evidence."""
        allowed = set(state.web_sources)
        allowed.update(source.url for source in request._research_sources)
        return _filter_research_sources(arguments, allowed)

    async def guide_plan(self, request: GuidePlanRequest, image_b64: str) -> tuple[dict[str, Any], Usage | None]:
        from backend.app import intent
        state = _ResponsesStream(intent.PLAN_TOOL_NAME)
        final = None
        async for item in self._stream(self._plan_payload(request, image_b64), intent.PLAN_TOOL_NAME, state=state):
            if isinstance(item, GuideStreamFinal):
                final = item
        if final is None:
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        return self._verified_plan_arguments(request, final.arguments, state), final.usage

    async def guide_plan_stream(self, request: GuidePlanRequest,
                                image_b64: str) -> AsyncIterator[str | GuideStreamFinal]:
        from backend.app import intent
        state = _ResponsesStream(intent.PLAN_TOOL_NAME)
        async for item in self._stream(self._plan_payload(request, image_b64), intent.PLAN_TOOL_NAME, state=state):
            if isinstance(item, GuideStreamFinal):
                yield GuideStreamFinal(self._verified_plan_arguments(request, item.arguments, state), item.usage)
            else:
                yield item

    async def guide_confirm(self, request: GuideConfirmRequest, plan: StoredPlan, image_b64: str,
                            before_b64: str | None = None) -> tuple[dict[str, Any], Usage | None]:
        from backend.app import intent
        system, schema = intent.confirm_tool(request.trigger, assisted=plan.assisted)
        before = (request.before_scene.frame_id, before_b64) if request.before_scene is not None and before_b64 else None
        payload = self._payload(system_prompt=system, text=intent.confirm_prompt(request, plan),
                                frame_id=request.scene.frame_id, image_b64=image_b64,
                                tool_schema=schema, tool_name=intent.CONFIRM_TOOL_NAME, before=before)
        return await self._call(payload, intent.CONFIRM_TOOL_NAME)

    async def guide_talk(self, request: GuideTalkRequest, plan: StoredPlan,
                         image_b64: str) -> tuple[dict[str, Any], Usage | None]:
        from backend.app import intent
        system, schema = intent.talk_tool(request.replan_allowed, assisted=plan.assisted)
        payload = self._payload(system_prompt=system, text=intent.talk_prompt(request, plan),
                                frame_id=request.scene.frame_id, image_b64=image_b64,
                                tool_schema=schema, tool_name=intent.TALK_TOOL_NAME)
        return await self._call(payload, intent.TALK_TOOL_NAME)

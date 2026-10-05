"""Local follower: the ``guide_follow`` judgement from a llama.cpp server on this machine (loopback only).

The follower writes no instruction and no goal status (see ``guide_contracts.GuideFollowToolOutput``), which
is what makes a small local model acceptable here: its whole answer is a checklist of closed-set verdicts
(one per checked plan step — under the ``classic`` core every step from the current one to the last, under
``sequential`` the current step alone, under ``graph`` the whole plan (earlier steps included);
``StoredPlan.checklist_ids`` decides) plus the anchor and goal verdicts,
every one of them checked by the route before the client sees it. There is no free-text field to write.

Transport: an OpenAI-compatible ``/v1/chat/completions`` endpoint. The default mode sends
``response_format: {type: "json_schema", strict: true}`` whose schema is the follow tool's parameter schema
with ``additionalProperties: false`` on every object, narrowed to the request (``narrowed_follow_schema``: the
checklist ids as an ordered tuple of consts, the sent anchor id as a const), so llama.cpp's grammar constrains
the answer to exactly that object, the model has no room to write prose first, and it cannot write a wrong
step or anchor id. The ``tool`` mode sends the same narrowed schema as a forced tool instead; llama.cpp refuses a ``tool_choice`` OBJECT (its log: ``type must be string``), so the
call is forced with the string ``"required"``. Measured 2026-10-01 (Qwen3.5-4B Q4_K_M, llama.cpp b11146,
enable_thinking off, ceiling 200, same frames, n=37 each): ``tool`` mode was schema-valid 17/37 (the other
20 wrote a Korean reasoning sentence before the call and hit the ceiling, ``finish_reason=length``),
``json_schema`` 37/37; end to end through the route, ``json_schema`` 4/4. Either way the answer is parsed
strictly and every failure maps onto the provider failure taxonomy, so the route answers a local failure
exactly like a DeepSeek one (502 ``invalid_provider_output`` with its stage, 502 ``provider_unavailable``,
504 ``provider_timeout``).

The ``kv`` mode drops the JSON contract altogether: a short prompt pair (``intent.FOLLOW_KV_SYSTEM_PROMPT``,
``intent.follow_kv_prompt``) and a GBNF ``grammar`` that admits exactly one line ``s1=<v> s2=<v> a1=<v> g=<v>``
(``kv_grammar``), expanded to the tool-output shape by ``kv_arguments`` and checked by the route like the
other modes. Measured 2026-10-02 (llama.cpp b11146, same 58 calls over 3 clips): on Qwen3.5-4B Q4 the
answer went from 50 tokens / 180 ms (json_schema) to 15 tokens / 49 ms with 89 -> 91/108 step verdicts and
6 -> 1/60 false ``yes``; on Qwen3.6-35B-A3B Q4 it is 96/108, 0/60 false ``yes``, ~300 ms server time per
call on one RTX 5090. The short prompt costs nothing in accuracy (same 96/108) and saves ~270 prompt tokens.

Configuration (environment only; values are never logged):

* ``AISW_FOLLOW_PROVIDER`` — ``local`` (default), ``clef`` (``clef_follow.py``) or ``deepseek``. Read by the application, not here.
* ``AISW_FOLLOW_LOCAL_URL`` — required for ``local``: the full chat-completions URL, loopback host only
  (``loopback.py``), e.g. ``http://127.0.0.1:8081/v1/chat/completions``. Redirects are never followed.
* ``AISW_FOLLOW_LOCAL_MODEL`` — required for ``local``: the model name requested and reported.
* ``AISW_FOLLOW_LOCAL_MODE`` — ``json_schema`` (default), ``tool`` or ``kv``.
* ``AISW_FOLLOW_LOCAL_TIMEOUT_S`` — optional, default ``10``; positive number.
* ``AISW_FOLLOW_LOCAL_MAX_TOKENS`` — optional, default ``800``; positive integer.

Missing or unusable configuration raises ``StageConfigError`` when the follower is built (at startup when
``AISW_FOLLOW_PROVIDER`` is default or ``local``); it never falls back to a cloud call.
"""

from __future__ import annotations

import asyncio
import copy
import os
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from backend.app.api_contracts import Usage
from backend.app.intent import (
    FOLLOW_KV_SYSTEM_PROMPT,
    FOLLOW_SYSTEM_PROMPT,
    FOLLOW_TOOL_NAME,
    FOLLOW_TOOL_SCHEMA,
    follow_kv_prompt,
    follow_prompt,
)
from backend.app.loopback import loopback_endpoint
from backend.app.provider import (
    InvalidOutputStage,
    ProviderConfigError,
    ProviderInvalidOutput,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    UnavailableStage,
    _chat_arguments,
    check_guide_finish_reason,
    parse_strict_json,
    parse_usage,
)
from backend.app.errors import StageConfigError

if TYPE_CHECKING:
    from backend.app.guide import StoredPlan
    from backend.app.guide_contracts import GuideFollowRequest


ENV_URL = "AISW_FOLLOW_LOCAL_URL"
ENV_MODEL = "AISW_FOLLOW_LOCAL_MODEL"
ENV_MODE = "AISW_FOLLOW_LOCAL_MODE"
ENV_TIMEOUT_S = "AISW_FOLLOW_LOCAL_TIMEOUT_S"
ENV_MAX_TOKENS = "AISW_FOLLOW_LOCAL_MAX_TOKENS"

MODE_TOOL = "tool"
MODE_JSON_SCHEMA = "json_schema"
#: One grammar-constrained line, ``s1=no s2=yes a1=yes g=no``, expanded to the tool-output shape here.
#: Same verdicts as the JSON modes with the answer at 9-15 tokens instead of 50-450 (``kv_grammar``).
MODE_KV = "kv"
MODES = (MODE_TOOL, MODE_JSON_SCHEMA, MODE_KV)
#: ``json_schema``: the only mode that stayed schema-valid on the 4B with the current prompt (see above).
DEFAULT_MODE = MODE_JSON_SCHEMA
DEFAULT_TIMEOUT_S = 10.0
#: Same ceiling as the DeepSeek follower (``intent.FOLLOW_MAX_TOKENS``). 1e local spike (llama.cpp b11146,
#: Qwen3.5 0.8B and 4B, no ``note``, n=20 frames per model and mode): 0/20 tool-mode answers fit a 60 cap,
#: and 20/20 were schema-valid at a 160 cap in both modes. With the phase-4 prompt the 4B's json_schema
#: answers were 70-71 tokens and its valid tool-mode answers 85-86 (server log, 2026-10-01). A hit is
#: refused as truncated. Raised with the follow ceiling to 600 when the answer became a checklist of up to
#: twelve steps (§15 cap, 2026-10-04; a 12-entry checklist cannot fit the old 260), then to 800 for the
#: sixteen-step cap (§15, 2026-10-05; four more entries at about 15 tokens each). **Unmeasured estimate**,
#: to be re-measured on a 16-step plan.
DEFAULT_MAX_TOKENS = 800

#: Readiness probe: cached like the tracker probe, so /api/health polling cannot hammer the model server.
HEALTH_TTL_S = 2.0
HEALTH_TIMEOUT_S = 2.0

LOCAL_PROVIDER_NAME = "local"


def _closed(node: Any) -> Any:
    """A copy of a JSON schema in which every object node has ``additionalProperties: false``.

    The follow tool schema is already closed (Pydantic ``extra="forbid"``); this makes the property hold for
    the grammar the local server compiles even if a contract change ever left one object open.
    """
    if isinstance(node, dict):
        out = {key: _closed(value) for key, value in node.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_closed(item) for item in node]
    return node


#: The follow tool parameters, closed: the generic (documented) local schema. Each request sends a copy
#: narrowed to that request (``narrowed_follow_schema``); this object is never mutated.
FOLLOW_JSON_SCHEMA: dict[str, Any] = _closed(copy.deepcopy(FOLLOW_TOOL_SCHEMA["function"]["parameters"]))


def narrowed_follow_schema(step_ids: list[str], anchor_ids: list[str]) -> dict[str, Any]:
    """``FOLLOW_JSON_SCHEMA`` narrowed to one request, so the grammar cannot write a wrong id.

    * ``step_checks`` becomes a tuple: ``prefixItems`` with one closed item per id of the checklist
      (``current_step`` .. last step), each ``step_id`` a ``const`` in plan order, ``visible`` the same enum;
      ``minItems == maxItems == len(step_ids)``. There is no ``items`` key: llama.cpp's schema-to-grammar
      converter reads ``items`` before ``prefixItems`` when both are present, and with ``prefixItems`` alone
      it emits exactly that sequence.
    * ``anchor_verdicts`` keeps its 0..n cardinality (a verdict about a box-less anchor may be omitted), but
      ``anchor_id`` is the ``const`` of the anchor sent (an ``enum`` if there were ever several); with no
      anchor sent, ``maxItems`` is 0.

    The route still checks the id list against the plan (``502 schema``) whatever the follower returns.
    """
    schema = copy.deepcopy(FOLLOW_JSON_SCHEMA)
    properties = schema["properties"]

    checks = properties["step_checks"]
    item = checks.pop("items")
    prefix = []
    for step_id in step_ids:
        fixed = copy.deepcopy(item)
        fixed["properties"]["step_id"] = {"type": "string", "const": step_id}
        prefix.append(fixed)
    checks["prefixItems"] = prefix
    checks["minItems"] = checks["maxItems"] = len(step_ids)

    verdicts = properties["anchor_verdicts"]
    if anchor_ids:
        verdicts["items"]["properties"]["anchor_id"] = (
            {"type": "string", "const": anchor_ids[0]} if len(anchor_ids) == 1
            else {"type": "string", "enum": list(anchor_ids)}
        )
        verdicts["maxItems"] = min(verdicts.get("maxItems", len(anchor_ids)), len(anchor_ids))
    else:
        verdicts["maxItems"] = 0
    return schema


KV_VALUES = ("yes", "no", "unsure")


def kv_keys(step_ids: list[str], anchor_ids: list[str]) -> list[str]:
    """The kv answer's keys in order: the checklist, the sent anchors, then ``g`` (goal_seen)."""
    return [*step_ids, *anchor_ids, "g"]


def kv_grammar(keys: list[str]) -> str:
    """GBNF for exactly ``k1=<v> k2=<v> ...`` with ``<v>`` in ``KV_VALUES``: the model can write nothing else.

    Measured 2026-10-02 (Qwen3.5-4B / Qwen3.6-35B-A3B, llama.cpp b11146, n=58 calls each): 15 answer tokens
    for a 3-step checklist, 9 for one step, vs 50 (json_schema) and 450 (tool). The keyless variant (values
    only, in order) was 6 tokens but lost the key-to-step binding on both models (4B: 68/108 step verdicts,
    21/60 false ``yes``; 35B: 6 false goal ``yes``), so the keys stay.
    """
    sequence = ' " " '.join(f'"{key}=" v' for key in keys)
    values = " | ".join(f'"{value}"' for value in KV_VALUES)
    return f"root ::= {sequence}\nv ::= {values}"


def kv_arguments(content: str, step_ids: list[str], anchor_ids: list[str]) -> dict[str, Any]:
    """The tool-output shape from one kv line, or ``ProviderInvalidOutput(tool_json)`` when the line is not
    exactly the expected keys in order (a server that ignored the grammar)."""
    keys = kv_keys(step_ids, anchor_ids)
    tokens = content.split()
    if len(tokens) != len(keys):
        raise ProviderInvalidOutput(InvalidOutputStage.TOOL_JSON)
    values: dict[str, str] = {}
    for key, token in zip(keys, tokens, strict=True):
        name, sep, value = token.partition("=")
        if not sep or name != key or value not in KV_VALUES:
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_JSON)
        values[key] = value
    return {
        "step_checks": [{"step_id": step_id, "visible": values[step_id]} for step_id in step_ids],
        "anchor_verdicts": [{"anchor_id": anchor_id, "matches": values[anchor_id]} for anchor_id in anchor_ids],
        "goal_seen": values["g"],
    }


def _health_url(chat_url: str) -> str:
    """llama.cpp's ``GET /health`` on the same scheme and authority as the chat endpoint."""
    parts = urlsplit(chat_url)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


async def post_loopback_json(client: httpx.AsyncClient, url: str, payload: dict[str, Any], timeout_s: float,
                             name: str) -> dict[str, Any]:
    """One POST to a loopback model server, no retry, every failure mapped onto the provider taxonomy.

    Shared by the local followers (``LocalFollower``, ``clef_follow.ClefFollower``). ``name`` appears only in
    the exception message, never a value from the request or the answer.
    """
    try:
        async with asyncio.timeout(timeout_s):
            response = await client.post(
                url,
                json=payload,
                timeout=httpx.Timeout(timeout_s),
                follow_redirects=False,
            )
            if response.status_code == 429:
                delay = response.headers.get("retry-after")
                raise ProviderRateLimited(delay if delay is not None and delay.isdecimal() else "30")
            if response.status_code in {401, 403}:
                raise ProviderConfigError(f"{name} rejected the request")
            response.raise_for_status()
            raw = response.text
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise ProviderTimeout(f"{name} timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderUnavailable(UnavailableStage.UPSTREAM_STATUS) from exc
    except httpx.HTTPError as exc:
        raise ProviderUnavailable(UnavailableStage.TRANSPORT) from exc
    try:
        return parse_strict_json(raw)
    except ProviderInvalidOutput as exc:
        raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE) from exc


class LocalFollower:
    """One ``guide_follow`` call per request against a loopback llama.cpp server; no retries."""

    provider_name = LOCAL_PROVIDER_NAME
    #: Not a paid analysis: the route keeps it out of the paid rate lane and the provider slots.
    paid = False

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        model: str,
        mode: str = DEFAULT_MODE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if mode not in MODES:
            raise StageConfigError(f"{ENV_MODE} must be one of {', '.join(repr(m) for m in MODES)}")
        self.client = client
        self.url = loopback_endpoint(url, ENV_URL)
        self.model = model
        self.mode = mode
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._probe: tuple[float, bool] = (0.0, False)

    def payload(self, request: GuideFollowRequest, plan: StoredPlan, image_b64: str) -> dict[str, Any]:
        step_ids = plan.checklist_ids(request.current_step)
        anchor_ids = [anchor.anchor_id for anchor in request.anchors]
        kv = self.mode == MODE_KV
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": follow_kv_prompt(request, plan) if kv else follow_prompt(request, plan)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": FOLLOW_KV_SYSTEM_PROMPT if kv else FOLLOW_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if kv:
            body["grammar"] = kv_grammar(kv_keys(step_ids, anchor_ids))
            return body
        schema = narrowed_follow_schema(step_ids, anchor_ids)
        if self.mode == MODE_TOOL:
            tool = copy.deepcopy(FOLLOW_TOOL_SCHEMA)
            tool["function"]["parameters"] = schema
            body["tools"] = [tool]
            # A string, never an object: llama.cpp rejects the object form.
            body["tool_choice"] = "required"
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": FOLLOW_TOOL_NAME,
                    "strict": True,
                    "schema": schema,
                },
            }
        return body

    async def guide_follow(self, request: GuideFollowRequest, plan: StoredPlan,
                           image_b64: str) -> tuple[dict[str, Any], Usage | None]:
        body = await self._post(self.payload(request, plan, image_b64))
        arguments = self._arguments(body)
        if self.mode == MODE_KV:
            arguments = kv_arguments(arguments, plan.checklist_ids(request.current_step),
                                     [anchor.anchor_id for anchor in request.anchors])
        return arguments, parse_usage(body.get("usage"))

    def _arguments(self, body: dict[str, Any]) -> Any:
        """The follow arguments from either JSON mode's answer, or the kv mode's answer line (a ``str``, the
        caller expands it with ``kv_arguments``), with the same fail-closed stages as DeepSeek."""
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        # ``length`` is truncated; ``tool_calls`` (tool mode) and ``stop`` (json_schema mode) are answers;
        # any other finish reason is refused as ``tool_envelope``.
        check_guide_finish_reason(choice.get("finish_reason"))
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        tool_calls = message.get("tool_calls")
        content = message.get("content")
        if self.mode in (MODE_JSON_SCHEMA, MODE_KV):
            # Exactly one JSON object (or one kv line) as the whole content, nothing else: no tool call, no
            # prose around it, no code fence (a grammar-constrained answer has none of these). Only
            # surrounding whitespace is tolerated. ``parse_strict_json`` also refuses duplicate keys and
            # non-finite numbers.
            if tool_calls or not isinstance(content, str) or not content.strip():
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
            return content.strip() if self.mode == MODE_KV else parse_strict_json(content.strip())
        if tool_calls:
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
            return _chat_arguments(tool_calls[0], FOLLOW_TOOL_NAME)
        # Tool mode without a tool call: a chat template whose tool-call parser left the call in the content.
        # Parsed strictly; anything else (e.g. reasoning prose) is refused.
        if not isinstance(content, str) or not content.strip():
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_ENVELOPE)
        return parse_strict_json(content.strip())

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await post_loopback_json(self.client, self.url, payload, self.timeout_s, "local follower")

    async def ready(self) -> bool:
        """Short-cached readiness of the local server (``GET /health`` == 200). Never raises."""
        now = time.monotonic()
        seen, ready = self._probe
        if seen and now - seen < HEALTH_TTL_S:
            return ready
        try:
            response = await self.client.get(
                _health_url(self.url), timeout=HEALTH_TIMEOUT_S, follow_redirects=False
            )
            ready = response.status_code == 200
        except httpx.HTTPError:
            ready = False
        self._probe = (now, ready)
        return ready


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise StageConfigError(f"{name} is not a number") from exc
    if not value > 0 or value == float("inf"):
        raise StageConfigError(f"{name} must be a positive number")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise StageConfigError(f"{name} is not an integer") from exc
    if value <= 0:
        raise StageConfigError(f"{name} must be positive")
    return value


def build_local_follower(client: httpx.AsyncClient) -> LocalFollower:
    """Build the local follower from its environment, or raise ``StageConfigError`` naming the variable."""
    url = (os.getenv(ENV_URL) or "").strip()
    if not url:
        raise StageConfigError(f"{ENV_URL} is not set")
    model = (os.getenv(ENV_MODEL) or "").strip()
    if not model:
        raise StageConfigError(f"{ENV_MODEL} is not set")
    mode = (os.getenv(ENV_MODE) or "").strip().lower() or DEFAULT_MODE
    return LocalFollower(
        client,
        url=url,
        model=model,
        mode=mode,
        timeout_s=_positive_float(ENV_TIMEOUT_S, DEFAULT_TIMEOUT_S),
        max_tokens=_positive_int(ENV_MAX_TOKENS, DEFAULT_MAX_TOKENS),
    )

"""HTTP client for the existing demo backend (aisw-hybrid-talk ``backend/app/main.py``), used as the real engine's upstream.

The demo backend is a browser API: a session cookie (``visual_coach_session``, ``SameSite=Strict``) and an
``Origin`` check (``Origin`` must equal ``scheme://Host`` of the request, main.py:352-362). One
``UpstreamClient`` = one upstream session = one live session. The cookie is replayed as an explicit ``Cookie``
header instead of through httpx's jar, because the backend marks it ``Secure`` whenever it does not consider the
peer loopback (main.py:365-373) and a ``Secure`` cookie is never sent back over plain HTTP by a jar.

Errors become ``UpstreamError(status, code, body, retry_after_ms, reason)``; transport failures have status 0
(code ``network``). Nothing here retries: the engine's policy (the browser's ``handleCallError``) decides.

The guide lane's plan routes are ``POST /api/guide/plan`` (SSE), ``GET /api/guide/plan/current``, and the two
plan-adoption routes ``POST /api/guide/plan/approve`` / ``POST /api/guide/plan/revert`` — each of the latter
returns the ``plan/current`` shape (``plan_id``, ``plan_revision``, ``task_revision``, ``steps``, ``goal_when``,
``user_goal``) and a ``409 stale_plan`` when the named pair is not the session's current one.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

COOKIE = "visual_coach_session"
EVENT_STREAM = "text/event-stream"
HEALTH_TTL_S = 10.0
HEALTH_FAIL_TTL_S = 3.0
# The Plan provider has a 180 s total deadline; allow its final result/error to reach Live.
PLAN_CALL_TIMEOUT_S = 190.0


class UpstreamError(Exception):
    def __init__(self, status: int, code: str, body: dict | None = None, retry_after_ms: int | None = None,
                 reason: str | None = None):
        super().__init__(f"{status} {code}")
        self.status = status
        self.code = code
        self.body = body or {}
        self.retry_after_ms = retry_after_ms
        self.reason = reason


def origin_of(base_url: str) -> str:
    """``scheme://Host`` exactly as httpx will send ``Host`` (lower-cased host, default port dropped)."""
    url = httpx.URL(base_url)
    return f"{url.scheme}://{url.netloc.decode('ascii')}"


def _retry_after_ms(value: str | None) -> int | None:
    """planStream.ts:96-100: seconds -> ms."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return int(seconds * 1000) if seconds >= 0 else None


def error_from_body(status: int, body: Any, retry_after_ms: int | None = None) -> UpstreamError:
    """planStream.ts:107-112."""
    record = body if isinstance(body, dict) else {}
    code = record.get("error") if isinstance(record.get("error"), str) else "request_failed"
    reason = record.get("reason") if isinstance(record.get("reason"), str) else None
    return UpstreamError(status, code, record, retry_after_ms, reason)


def is_plan_response(value: Any) -> bool:
    """planStream.ts:115-125."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("plan_id"), str)
        and isinstance(value.get("plan_revision"), int)
        and isinstance(value.get("steps"), list)
        and isinstance(value.get("selection"), dict)
        and isinstance(value.get("needs_clarification"), bool)
    )


class SseParser:
    """planStream.ts:43-89 ``createSseParser`` over already-split lines."""

    def __init__(self) -> None:
        self.event = ""
        self.data: list[str] = []

    def line(self, line: str) -> tuple[str, str] | None:
        if line == "":
            out = (self.event or "message", "\n".join(self.data)) if self.data else None
            self.event, self.data = "", []
            return out
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self.event = value
        elif field == "data":
            self.data.append(value)
        return None


class UpstreamClient:
    def __init__(self, base_url: str, *, access_code: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, call_timeout_s: float = 60.0,
                 track_timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.origin = origin_of(self.base_url)
        self.access_code = access_code
        self.call_timeout_s = call_timeout_s
        self.track_timeout_s = track_timeout_s
        self.client = httpx.AsyncClient(base_url=self.base_url, transport=transport,
                                        headers={"Origin": self.origin}, timeout=call_timeout_s)
        self.session_id: str | None = None
        self._token: str | None = None
        self._start_seq = 0

    # ------------------------------------------------------------ plumbing

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if self._token is not None:
            headers["Cookie"] = f"{COOKIE}={self._token}"
        return headers

    async def _post(self, path: str, body: dict, *, timeout: float | None = None) -> dict:
        try:
            resp = await self.client.post(path, json=body, headers=self._headers(),
                                          timeout=timeout or self.call_timeout_s)
        except httpx.HTTPError as exc:
            raise UpstreamError(0, "network", {"detail": type(exc).__name__}) from exc
        return self._read(resp)

    @staticmethod
    def _read(resp: httpx.Response) -> dict:
        try:
            body = resp.json()
        except ValueError:
            body = None
        if resp.status_code >= 400:
            raise error_from_body(resp.status_code, body, _retry_after_ms(resp.headers.get("retry-after")))
        if not isinstance(body, dict):
            raise UpstreamError(502, "invalid_response")
        return body

    # ------------------------------------------------------------ session

    def invalidate_session(self) -> None:
        self.session_id = None
        self._token = None

    async def ensure_session(self, code_required: bool) -> str:
        if self.session_id is not None:
            return self.session_id
        body: dict[str, Any] = {}
        if code_required and self.access_code:
            body["access_code"] = self.access_code
        try:
            resp = await self.client.post("/api/session", json=body, timeout=self.call_timeout_s)
        except httpx.HTTPError as exc:
            raise UpstreamError(0, "network", {"detail": type(exc).__name__}) from exc
        data = self._read(resp)
        token = resp.cookies.get(COOKIE)
        if not token or not isinstance(data.get("session_id"), str):
            raise UpstreamError(502, "session_without_cookie")
        self.client.cookies.clear()  # replayed explicitly (see module doc)
        self.session_id, self._token = data["session_id"], token
        self._start_seq = 0
        return self.session_id

    async def end_session(self) -> None:
        if self._token is None:
            return
        try:
            await self.client.post("/api/session/end", json={}, headers=self._headers(), timeout=5.0)
        except httpx.HTTPError:
            pass
        self.invalidate_session()

    async def aclose(self) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------ tracking

    async def track_start(self, run_id: str, target: str) -> dict:
        """``start_seq`` is strictly increasing per upstream session (useObjectTrack.ts:618-624)."""
        assert self.session_id is not None
        self._start_seq += 1
        return await self._post("/api/track/control", {
            "session_id": self.session_id, "run_id": run_id, "action": "start",
            "start_seq": self._start_seq, "target": target,
        }, timeout=self.track_timeout_s * 2)

    async def track_stop(self, run_id: str) -> None:
        if self.session_id is None:
            return
        try:
            await self._post("/api/track/control", {"session_id": self.session_id, "run_id": run_id,
                                                    "action": "stop"}, timeout=self.track_timeout_s)
        except UpstreamError:
            pass  # best effort, like the browser (useObjectTrack.ts:352-354)

    async def track_frame(self, run_id: str, frame_id: str, frame_seq: int, image_b64: str,
                          seed_box: dict | None = None) -> dict:
        assert self.session_id is not None
        body: dict[str, Any] = {"session_id": self.session_id, "run_id": run_id, "frame_id": frame_id,
                                "frame_seq": frame_seq, "image_base64": image_b64}
        if seed_box is not None:
            body["seed_box"] = seed_box
        return await self._post("/api/track/frame", body, timeout=self.track_timeout_s)

    # ------------------------------------------------------------ guide

    async def plan(self, payload: dict, on_partial: Callable[[str, str], None]) -> dict:
        """``POST /api/guide/plan`` with ``Accept: text/event-stream`` (planStream.ts:155-228)."""
        headers = self._headers({"Accept": EVENT_STREAM})
        # §15: every plan request carries ``plan_model`` and both planning profiles run under the same 180 s
        # provider deadline, so the upper read budget is the plan budget — no longer tied to review mode.
        timeout_s = max(self.call_timeout_s, PLAN_CALL_TIMEOUT_S)
        try:
            async with self.client.stream("POST", "/api/guide/plan", json=payload, headers=headers,
                                          timeout=httpx.Timeout(timeout_s, read=timeout_s)) as resp:
                ctype = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if ctype != EVENT_STREAM:
                    await resp.aread()
                    body = self._read(resp)
                    if not is_plan_response(body):
                        raise UpstreamError(502, "invalid_plan_response")
                    return body
                parser = SseParser()
                saw_target = saw_say = False

                async def events():
                    async for line in resp.aiter_lines():
                        event = parser.line(line.rstrip("\r"))
                        if event is not None:
                            yield event
                    flushed = parser.line("")  # planStream.ts:213 flush at EOF
                    if flushed is not None:
                        yield flushed

                async for name, data in events():
                    if name not in ("partial", "final", "error"):
                        continue  # planStream.ts:207 unknown event names are ignored
                    try:
                        parsed = json.loads(data)
                    except ValueError:
                        if name == "partial":
                            continue  # a malformed hint is only a missing hint
                        raise UpstreamError(502, "stream_malformed") from None
                    if name == "partial" and isinstance(parsed, dict):
                        if not saw_target and isinstance(parsed.get("target"), str) and parsed["target"].strip():
                            saw_target = True
                            on_partial("target", parsed["target"])
                        if not saw_say and isinstance(parsed.get("first_say"), str) and parsed["first_say"].strip():
                            saw_say = True
                            on_partial("first_say", parsed["first_say"])
                    elif name == "final":
                        if not is_plan_response(parsed):
                            raise UpstreamError(502, "invalid_plan_response")
                        return parsed
                    elif name == "error":
                        status = parsed.get("status") if isinstance(parsed, dict) else None
                        status = status if isinstance(status, int) and status >= 400 else 502
                        raise error_from_body(status, parsed)
                raise UpstreamError(502, "stream_incomplete")
        except httpx.HTTPError as exc:
            raise UpstreamError(0, "network", {"detail": type(exc).__name__}) from exc

    async def plan_current(self) -> dict:
        """``GET /api/guide/plan/current``: the recovery read after a ``409 stale_plan`` (no provider call)."""
        assert self.session_id is not None
        try:
            resp = await self.client.get("/api/guide/plan/current", params={"session_id": self.session_id},
                                         headers=self._headers(), timeout=self.track_timeout_s * 2)
        except httpx.HTTPError as exc:
            raise UpstreamError(0, "network", {"detail": type(exc).__name__}) from exc
        return self._read(resp)

    async def plan_approve(self, payload: dict) -> dict:
        """``POST /api/guide/plan/approve``: the client-reviewed plan becomes the session's current plan."""
        return await self._post("/api/guide/plan/approve", payload)

    async def plan_revert(self, payload: dict) -> dict:
        """``POST /api/guide/plan/revert``: install the session's previous plan under a new ``plan_revision``."""
        return await self._post("/api/guide/plan/revert", payload)

    async def follow(self, payload: dict) -> dict:
        return await self._post("/api/guide/follow", payload)

    async def confirm(self, payload: dict) -> dict:
        # §15: the plan's stored profile drives confirm/replan too, so this call gets the same upper budget.
        return await self._post("/api/guide/confirm", payload,
                                timeout=max(self.call_timeout_s, PLAN_CALL_TIMEOUT_S))

    async def talk(self, payload: dict) -> dict:
        return await self._post("/api/guide/talk", payload,
                                timeout=max(self.call_timeout_s, PLAN_CALL_TIMEOUT_S))


class UpstreamHealth:
    """``GET /api/health`` cached ~10 s (failures ~3 s). Shared by the app's health route and every engine."""

    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.clock = clock
        self._value: dict | None = None
        self._at = -1e18
        self._ok = False

    def cached(self) -> dict | None:
        return self._value

    async def get(self) -> dict | None:
        ttl = HEALTH_TTL_S if self._ok else HEALTH_FAIL_TTL_S
        if self.clock() - self._at < ttl:
            return self._value
        try:
            async with httpx.AsyncClient(base_url=self.base_url, transport=self.transport, timeout=3.0) as client:
                resp = await client.get("/api/health")
            body = resp.json() if resp.status_code == 200 else None
            self._value = body if isinstance(body, dict) else None
        except (httpx.HTTPError, ValueError):
            self._value = None
        self._ok = self._value is not None
        self._at = self.clock()
        return self._value



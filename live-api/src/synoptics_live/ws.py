"""The live WebSocket (§4-§6, §9): hello/auth, framing, credit, gating, dispatch to the engine."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from typing import Any

from PIL import Image
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocket

from .contracts import (
    CLIENT_MSG,
    CLIENT_TYPES,
    CORE_GRAPH,
    PROTOCOL,
    ByeMsg,
    ErrorMsg,
    FrameHeader,
    HelloMsg,
    Limits,
    Prefs,
)
from . import plan_policy
from .engine import EngineReject
from .envelope import EnvelopeError, unpack
from .sessions import LiveSession, SessionStore, tokens_equal
from .settings import Settings

log = logging.getLogger("synoptics_live.ws")

CLOSE_POLICY = 1008
CLOSE_AUTH = 4001
CLOSE_EXPIRED = 4002
CLOSE_PROTOCOL = 4008
CLOSE_TOO_LARGE = 4009

#: §15: the only client messages that may legitimately carry reference photos. Every other control message
#: keeps the ordinary bound, so image input enlarges the WebSocket control limit only where it must.
REFERENCE_MESSAGE_TYPES = frozenset({"start", "plan_answer"})

_CLOSE = object()


class Connection:
    """One accepted socket. Every send goes through a queue drained by a single sender task."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closing = False
        self.frame_task: asyncio.Task | None = None
        self.outstanding_since: float | None = None
        self.violations = 0

    def send_text(self, text: str) -> None:
        if not self.closing:
            self.queue.put_nowait(("text", text))

    def send_model(self, model: BaseModel) -> None:
        self.send_text(model.model_dump_json())

    def send_bytes(self, data: bytes) -> None:
        if not self.closing:
            self.queue.put_nowait(("bytes", data))

    def error(self, code: str, message: str, *, retryable: bool = True, fatal: bool = False,
              retry_after_ms: int | None = None) -> None:
        self.send_model(ErrorMsg(code=code, message=message, retryable=retryable, fatal=fatal,
                                 retry_after_ms=retry_after_ms))

    def close(self, code: int, reason: str | None) -> None:
        """Send ``bye{reason}`` (if given) after everything already queued, then close with ``code``."""
        if self.closing:
            return
        self.closing = True
        self.queue.put_nowait((_CLOSE, code, reason))

    async def run_sender(self) -> None:
        try:
            while True:
                item = await self.queue.get()
                if item[0] is _CLOSE:
                    _, code, reason = item
                    if reason:
                        await self.ws.send_text(ByeMsg(reason=reason).model_dump_json())
                    await self.ws.close(code)
                    return
                kind, data = item
                if kind == "text":
                    await self.ws.send_text(data)
                else:
                    await self.ws.send_bytes(data)
        except Exception as exc:  # peer gone
            log.debug("sender stopped: %r", exc)


async def _reject(ws: WebSocket, code: int, reason: str, error: ErrorMsg | None = None) -> None:
    try:
        if error is not None:
            await ws.send_text(error.model_dump_json())
        await ws.send_text(ByeMsg(reason=reason).model_dump_json())
        await ws.close(code)
    except Exception:
        pass


def _text_of(msg: dict[str, Any]) -> str | None:
    return msg.get("text")


def origin_allowed(settings: Settings, origin: str | None) -> bool:
    if settings.any_origin:
        return True
    return origin is not None and origin in settings.cors_origins


async def live_endpoint(ws: WebSocket, sid: str, store: SessionStore, settings: Settings, limits: Limits) -> None:
    if not origin_allowed(settings, ws.headers.get("origin")):
        await ws.close(CLOSE_POLICY)  # before accept: the handshake is refused (HTTP 403)
        return
    await ws.accept()

    # ---- hello
    try:
        first = await asyncio.wait_for(ws.receive(), settings.hello_timeout_s)
    except asyncio.TimeoutError:
        await _reject(ws, CLOSE_PROTOCOL, "protocol_violation")
        return
    if first["type"] == "websocket.disconnect":
        return
    text = _text_of(first)
    if text is None:
        await _reject(ws, CLOSE_PROTOCOL, "protocol_violation")
        return
    if len(text.encode("utf-8")) > settings.max_message_bytes:
        await _reject(ws, CLOSE_TOO_LARGE, "message_too_large")
        return
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict) or obj.get("type") != "hello":
            raise ValueError("first message is not hello")
        hello = HelloMsg.model_validate(obj)
    except (ValueError, ValidationError):
        await _reject(ws, CLOSE_PROTOCOL, "protocol_violation")
        return

    session = await store.get(sid)
    if session is None:
        if store.was_ended(sid):
            await _reject(ws, CLOSE_EXPIRED, "session_expired")
        else:
            await _reject(ws, CLOSE_AUTH, "auth_failed")
        return
    if not tokens_equal(hello.token, session.token):
        await _reject(ws, CLOSE_AUTH, "auth_failed")
        return
    if hello.protocol != PROTOCOL:
        await _reject(
            ws, CLOSE_PROTOCOL, "unsupported_protocol",
            ErrorMsg(code="unsupported_protocol", message=f"서버 프로토콜은 {PROTOCOL}입니다",
                     retryable=False, fatal=True),
        )
        return

    # ---- live
    conn = Connection(ws)
    sender = asyncio.create_task(conn.run_sender())
    await session.attach(conn, hello, limits)
    handler = _Handler(conn, session, settings, limits)
    receiver = asyncio.create_task(handler.receive_loop())
    try:
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        if receiver in done and conn.closing and not sender.done():
            try:
                await asyncio.wait_for(asyncio.shield(sender), 2.0)
            except asyncio.TimeoutError:
                pass
    finally:
        for task in (sender, receiver, conn.frame_task):
            if task is not None and not task.done():
                task.cancel()
        await session.detach(conn)


class _Handler:
    def __init__(self, conn: Connection, session: LiveSession, settings: Settings, limits: Limits):
        self.conn = conn
        self.session = session
        self.settings = settings
        self.limits = limits

    @property
    def engine(self):
        return self.session.engine

    async def receive_loop(self) -> None:
        ws = self.conn.ws
        while not self.conn.closing:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            self.session.touch()
            data = msg.get("bytes")
            if data is not None:
                if len(data) > self.settings.max_message_bytes:
                    self.conn.close(CLOSE_TOO_LARGE, "message_too_large")
                    return
                await self.on_binary(data)
            else:
                text = msg.get("text") or ""
                size = len(text.encode("utf-8"))
                if size > self.settings.max_message_bytes and not self._reference_message_allowed(text, size):
                    self.conn.close(CLOSE_TOO_LARGE, "message_too_large")
                    return
                await self.on_text(text)

    def _reference_message_allowed(self, text: str, size: int) -> bool:
        """§15: a message that may carry reference photos gets the larger (still bounded) envelope — and only
        that message type. Every per-photo bound is enforced later, before any provider call."""
        if size > self.settings.max_reference_message_bytes:
            return False
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(obj, dict) and obj.get("type") in REFERENCE_MESSAGE_TYPES

    # ------------------------------------------------------------ text

    async def on_text(self, text: str) -> None:
        conn = self.conn
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            conn.close(CLOSE_PROTOCOL, "protocol_violation")
            return
        if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
            conn.close(CLOSE_PROTOCOL, "protocol_violation")
            return
        kind = obj["type"]
        if kind not in CLIENT_TYPES:
            conn.error("invalid_message", f"알 수 없는 메시지 type: {kind[:40]}", retryable=False)
            return
        try:
            msg = CLIENT_MSG.validate_python(obj)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ())[1:]) or kind
            conn.error("invalid_message", f"{kind}: {where}: {first.get('msg', 'invalid')}"[:300], retryable=False)
            return
        try:
            await self.dispatch(msg)
        except EngineReject as rej:
            conn.error(rej.code, rej.message, retryable=rej.retryable, retry_after_ms=rej.retry_after_ms)

    async def dispatch(self, msg) -> None:
        conn, session, engine = self.conn, self.session, self.engine
        snap = session.snapshot
        kind = msg.type
        if kind == "hello":
            conn.error("invalid_message", "hello는 연결마다 한 번만 보냅니다", retryable=False)
        elif kind == "start":
            if msg.consent_ai is not True:
                conn.error("consent_required", "AI 사용 동의가 필요합니다", retryable=False)
                return
            session.consent = True
            await engine.on_start(msg.goal, msg.context, plan_mode=bool(msg.plan_mode),
                                  plan_model=msg.plan_model, core_mode=msg.core_mode,
                                  materials=tuple(msg.materials or ()),
                                  reference_images=tuple(msg.reference_images or ()))
        elif kind == "stop":
            await engine.on_stop()
        # ---- §15 Plan 모드
        elif kind == "plan_answer":
            # §15: the fence (the phase and the current question id) lives in the engine, which owns the run.
            await engine.on_plan_answer(msg.clarification_id, msg.answer,
                                        None if msg.reference_images is None else tuple(msg.reference_images))
        elif kind == "plan_edit":
            if snap.phase == "reviewing":
                await engine.on_plan_edit(msg)
            elif snap.plan is not None:
                conn.error("plan_locked", "승인된 계획은 고칠 수 없습니다. 새로 시작하거나 초안을 다시 만드세요.",
                           retryable=False)
            else:
                conn.error("not_reviewing", "검토 중인 계획이 없습니다", retryable=False)
        elif kind in ("plan_approve", "plan_discard"):
            if snap.phase != "reviewing":
                conn.error("not_reviewing", "검토 중인 계획이 없습니다", retryable=False)
            elif kind == "plan_approve":
                await engine.on_plan_approve()
            else:
                await engine.on_plan_discard()
        elif kind == "run_pause":
            if snap.phase != "running":
                conn.error("not_running", "진행 중인 가이드가 없습니다")
            elif snap.paused:
                conn.error("not_running", "이미 일시정지 상태입니다")
            else:
                await engine.on_run_pause()
        elif kind == "run_resume":
            if snap.phase != "running":
                conn.error("not_running", "진행 중인 가이드가 없습니다")
            elif not snap.paused:
                conn.error("not_running", "일시정지 상태가 아닙니다")
            else:
                await engine.on_run_resume()
        elif kind == "step_ack":
            if snap.phase != "running":
                conn.error("not_running", "진행 중인 가이드가 없습니다")
            elif snap.paused:
                conn.error("not_running", "일시정지 중입니다. 재개한 뒤 확인해 주세요")
            else:
                await engine.on_step_ack(msg.step_id)
        elif kind in ("proposal_accept", "proposal_reject"):
            if snap.proposal is None:
                conn.error("no_proposal", "제안된 변경안이 없습니다", retryable=False)
            else:
                await engine.on_proposal(accept=kind == "proposal_accept")
        elif kind == "talk":
            if msg.consent_ai is False:
                conn.error("consent_required", "AI 사용 동의가 필요합니다", retryable=False)
            elif snap.phase == "planning":
                conn.error("plan_in_flight", "계획을 세우는 중입니다")
            elif snap.phase != "running":
                conn.error("not_running", "진행 중인 가이드가 없습니다")
            elif snap.paused:
                conn.error("not_running", "일시정지 중입니다. 재개한 뒤 말씀해 주세요")
            elif not session.consent:
                conn.error("consent_required", "AI 사용 동의가 필요합니다", retryable=False)
            elif snap.talk_pending:
                conn.error("talk_busy", "이전 질문에 답하는 중입니다")
            else:
                await engine.on_talk(msg.utterance)
        elif kind in ("follow_now", "replan_now"):
            if snap.phase == "planning":
                conn.error("plan_in_flight", "계획을 세우는 중입니다")
            elif snap.phase != "running":
                conn.error("not_running", "진행 중인 가이드가 없습니다")
            elif snap.paused:
                conn.error("not_running", "일시정지 중입니다. 재개한 뒤 다시 시도해 주세요")
            elif kind == "follow_now":
                await engine.on_follow_now()
            else:
                await engine.on_replan_now()
        elif kind == "confirm_done":
            pending = [] if snap.plan is None else (
                plan_policy.graph_completion_pending(snap.plan, snap.steps_done)
                if snap.core_mode == CORE_GRAPH else
                plan_policy.required_pending(snap.plan, step_index=snap.step_index,
                                             skipped=snap.steps_skipped, user_done=snap.steps_user_done)
            )
            if snap.phase == "planning":
                conn.error("plan_in_flight", "계획을 세우는 중입니다")
            elif snap.paused:
                conn.error("not_running", "일시정지 중입니다. 재개한 뒤 확인해 주세요")
            elif pending:
                conn.error("required_check", f"필수 조건이 남아 있어 완료로 넘길 수 없습니다: {', '.join(pending)}")
            elif snap.completion != "confirmed":
                conn.error("not_running", "아직 완료 확인 단계가 아닙니다")
            else:
                await engine.on_confirm_done()
        elif kind == "prefs":
            update = {k: v for k, v in (("voice_out", msg.voice_out), ("tts", msg.tts), ("visuals", msg.visuals))
                      if v is not None}
            session.prefs = Prefs(**{**session.prefs.model_dump(), **update})
            await engine.on_prefs(session.prefs)
        elif kind == "played":
            log.info("played %s via %s (session %s)", msg.line_id, msg.via, session.sid[:8])

    # ------------------------------------------------------------ binary (frames)

    def _frame_error(self, code: str, message: str) -> None:
        self.conn.error(code, message[:300], retryable=True)

    async def on_binary(self, data: bytes) -> None:
        try:
            raw_header, payload = unpack(data)
        except EnvelopeError as exc:
            self._frame_error("frame_invalid", f"envelope: {exc}")
            return
        try:
            header = FrameHeader.model_validate(raw_header)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ())) or "header"
            self._frame_error("frame_invalid", f"frame header {where}: {first.get('msg', 'invalid')}")
            return
        cap = self.limits.hi_max_bytes if header.hi_req else self.limits.stream_max_bytes
        if len(payload) > cap:
            self._frame_error("frame_too_large", f"seq {header.seq}: {len(payload)} bytes > {cap}")
            return
        problem = jpeg_problem(payload, header.w, header.h)
        if problem:
            self._frame_error("frame_invalid", f"seq {header.seq}: {problem}")
            return

        if header.hi_req is not None:
            if header.hi_req not in self.session.issued_hi:
                self._frame_error("frame_invalid", f"seq {header.seq}: unknown hi_req {header.hi_req}")
                return
            self.session.issued_hi.remove(header.hi_req)
            try:
                await self.engine.on_hi_frame(header, payload)
            except EngineReject as rej:
                self.conn.error(rej.code, rej.message, retryable=rej.retryable)
            return

        conn = self.conn
        now = time.monotonic()
        if conn.frame_task is not None and not conn.frame_task.done():
            assert conn.outstanding_since is not None
            if now - conn.outstanding_since < self.settings.credit_stale_s:
                conn.violations += 1
                if conn.violations >= self.settings.credit_violation_limit:
                    conn.close(CLOSE_PROTOCOL, "protocol_violation")
                    return
                conn.error("rate_limited", f"seq {header.seq}: 이전 프레임의 track을 받기 전에 보낸 프레임이라 버렸습니다")
                return
            conn.frame_task.cancel()  # the old frame is stale; the new one replaces it
        conn.outstanding_since = now
        conn.frame_task = asyncio.create_task(self._process_frame(header, payload))

    async def _process_frame(self, header: FrameHeader, payload: bytes) -> None:
        conn = self.conn
        try:
            track = await self.engine.on_frame(header, payload)
        except EngineReject as rej:
            conn.outstanding_since = None
            conn.error(rej.code, rej.message, retryable=rej.retryable)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("engine.on_frame failed")
            conn.outstanding_since = None
            conn.error("tracker_unavailable", "추적 처리 중 오류가 났습니다")
            return
        conn.outstanding_since = None
        conn.send_model(track)


def jpeg_problem(payload: bytes, w: int, h: int) -> str | None:
    """None if ``payload`` is a JPEG whose header says ``w``x``h``; otherwise what is wrong."""
    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        return "payload is not a JPEG (no FFD8 marker)"
    try:
        with Image.open(io.BytesIO(payload)) as img:
            fmt, size = img.format, img.size
    except Exception as exc:
        return f"JPEG header unreadable: {type(exc).__name__}"
    if fmt != "JPEG":
        return f"payload decodes as {fmt}, not JPEG"
    if size != (w, h):
        return f"header says {w}x{h} but the JPEG is {size[0]}x{size[1]}"
    return None

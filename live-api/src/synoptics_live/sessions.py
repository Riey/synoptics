"""Sessions: creation policy (access code, rate limit, capacity), expiry, and the per-session live state.

A ``LiveSession`` is the engine's sink: it owns ``rev``, the latest ``state`` snapshot, line/req ids and the
TTS hand-off, and forwards to whichever connection is currently attached (at most one).
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import math
import secrets
import time
import uuid
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Callable, Literal

from .contracts import AudioHeader, CaptureHiMsg, GuideState, HelloMsg, Prefs, ReadyMsg, SayMsg
from .envelope import pack
from .tts import TtsBridge, pick_format

if TYPE_CHECKING:
    from .engine import Engine
    from .settings import Settings
    from .ws import Connection

log = logging.getLogger("synoptics_live.sessions")

TOMBSTONE_TTL_S = 3600.0
TOMBSTONE_MAX = 4096


def now_ms() -> int:
    return int(time.time() * 1000)


def tokens_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


class SessionError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


class RateLimiter:
    """Sliding window: at most ``limit`` attempts per ``window_s`` per key."""

    def __init__(self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}

    def hit(self, key: str) -> float | None:
        """Record an attempt. Returns None if allowed, else seconds until the next attempt is allowed."""
        now = self.clock()
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] >= self.window_s:
            q.popleft()
        if len(q) >= self.limit:
            return self.window_s - (now - q[0])
        q.append(now)
        if len(self._hits) > 10_000:  # forget idle keys
            for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s]:
                del self._hits[k]
        return None


class LiveSession:
    def __init__(self, sid: str, token: str, settings: Settings, tts: TtsBridge | None, clock=time.monotonic):
        self.sid = sid
        self.token = token
        self.settings = settings
        self.tts = tts
        self._clock = clock
        self.last_active = clock()
        self.ended = False
        self.rev = 0
        self.snapshot = GuideState()
        self.conn: Connection | None = None
        self.disconnected_at: float | None = None
        self.paused = False
        self.engine: Engine | None = None
        self.prefs = Prefs()
        self.audio_accept: list[str] | None = None
        self.consent = False
        self.issued_hi: deque[str] = deque(maxlen=8)
        self._line = 0
        self._hi = 0
        self._bg: set[asyncio.Task] = set()

    # ------------------------------------------------------------ activity / expiry

    def touch(self) -> None:
        self.last_active = self._clock()

    def expiry_reason(self) -> str | None:
        """Why this session should end now, if it should (never while a connection is attached)."""
        if self.conn is not None:
            return None
        now = self._clock()
        if self.disconnected_at is not None and now - self.disconnected_at > self.settings.resume_grace_s:
            return "resume grace elapsed"
        if now - self.last_active > self.settings.idle_ttl_s:
            return "idle"
        return None

    def expires_at_ms(self) -> int:
        remaining = self.settings.idle_ttl_s - (self._clock() - self.last_active)
        return now_ms() + int(max(remaining, 0) * 1000)

    # ------------------------------------------------------------ connection attach / detach

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def attach(self, conn: Connection, hello: HelloMsg, limits) -> None:
        old = self.conn
        if old is not None:
            old.close(4003, "replaced")
        self.conn = conn
        self.disconnected_at = None
        self.audio_accept = hello.audio_accept
        self.touch()
        conn.send_model(ReadyMsg(rev=self.rev, server_time=now_ms(), limits=limits))
        conn.send_text(self._state_text())
        if self.paused and self.engine is not None:
            self.paused = False
            await self.engine.on_resume()
        if self.tts is not None:
            self._spawn(self.tts.ready())

    async def detach(self, conn: Connection) -> None:
        if self.conn is not conn:
            return  # replaced by a newer connection
        self.conn = None
        self.touch()
        if self.ended:
            return
        self.disconnected_at = self._clock()
        if self.engine is not None and not self.paused:
            self.paused = True
            await self.engine.on_pause()

    async def end(self) -> None:
        self.ended = True
        for task in list(self._bg):
            task.cancel()
        if self.engine is not None:
            await self.engine.close()

    # ------------------------------------------------------------ EngineSink

    def _state_text(self) -> str:
        body = self.snapshot.model_dump_json()
        return f'{{"type":"state","rev":{self.rev},{body[1:]}'

    def state(self, state: GuideState) -> None:
        self.rev += 1
        self.snapshot = state
        if self.conn is not None:
            self.conn.send_text(self._state_text())

    def _wants_audio(self) -> bool:
        return (
            self.tts is not None
            and self.prefs.voice_out
            and self.prefs.tts == "server"
            and self.tts.ready_now()
        )

    def say(self, text: str, mode: Literal["replace", "append"]) -> str:
        self._line += 1
        line_id = f"L{self._line}"
        conn = self.conn
        if conn is None:
            return line_id  # paused: pending lines are not re-sent after a reconnect (§9)
        audio = "pending" if self._wants_audio() else "none"
        conn.send_model(SayMsg(line_id=line_id, text=text, mode=mode, audio=audio))
        if audio == "pending":
            self._spawn(self._deliver_audio(conn, line_id, text))
        return line_id

    async def _deliver_audio(self, conn: Connection, line_id: str, text: str) -> None:
        assert self.tts is not None
        clip = await self.tts.synthesize(text, pick_format(self.audio_accept))
        if clip is None or self.conn is not conn or conn.closing:
            return
        header = AudioHeader(line_id=line_id, mime=clip.mime, duration_ms=clip.duration_ms)
        conn.send_bytes(pack(header.model_dump(), clip.data))

    def capture_hi(self) -> str:
        self._hi += 1
        req_id = f"h{self._hi}"
        self.issued_hi.append(req_id)
        if self.conn is not None:
            self.conn.send_model(CaptureHiMsg(req_id=req_id))
        return req_id


class SessionStore:
    def __init__(
        self,
        settings: Settings,
        engine_factory: Callable[[LiveSession], Engine],
        engine_ok: bool,
        tts: TtsBridge | None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self.engine_factory = engine_factory
        self.engine_ok = engine_ok
        self.tts = tts
        self.clock = clock
        self.sessions: dict[str, LiveSession] = {}
        self._tombstones: OrderedDict[str, float] = OrderedDict()
        self.limiter = RateLimiter(settings.create_per_ip_per_min, 60.0, clock)

    @property
    def access_code_required(self) -> bool:
        return self.settings.access_code is not None

    @property
    def access_mode(self) -> str:
        return "code" if self.access_code_required else "open"

    async def create(self, access_code: str | None, client_ip: str) -> LiveSession:
        await self.sweep()
        wait = self.limiter.hit(client_ip)
        if wait is not None:
            raise SessionError(
                429, "rate_limited", "세션 생성 요청이 너무 잦습니다", {"Retry-After": str(max(1, math.ceil(wait)))}
            )
        if self.settings.access_code is not None:
            if not access_code or not tokens_equal(access_code, self.settings.access_code):
                raise SessionError(401, "invalid_access_code", "접근 코드가 맞지 않습니다")
        if not self.engine_ok:
            raise SessionError(503, "service_unavailable", "서버 설정 문제로 세션을 만들 수 없습니다")
        if len(self.sessions) >= self.settings.max_sessions:
            raise SessionError(503, "capacity", "동시 세션 수가 상한에 도달했습니다")
        sid = uuid.uuid4().hex
        session = LiveSession(sid, secrets.token_urlsafe(32), self.settings, self.tts, self.clock)
        session.engine = self.engine_factory(session)
        self.sessions[sid] = session
        return session

    async def get(self, sid: str) -> LiveSession | None:
        await self.sweep()
        return self.sessions.get(sid)

    def was_ended(self, sid: str) -> bool:
        return sid in self._tombstones

    async def end(self, session: LiveSession, *, close_code: int = 1000, reason: str = "session_ended") -> None:
        if session.ended:
            return
        self.sessions.pop(session.sid, None)
        self._tombstones[session.sid] = self.clock()
        while len(self._tombstones) > TOMBSTONE_MAX:
            self._tombstones.popitem(last=False)
        if session.conn is not None:
            session.conn.close(close_code, reason)
        await session.end()

    async def sweep(self) -> None:
        now = self.clock()
        for session in list(self.sessions.values()):
            why = session.expiry_reason()
            if why is not None:
                log.info("session %s expired (%s)", session.sid[:8], why)
                await self.end(session, close_code=4002, reason="session_expired")
        while self._tombstones:
            sid, at = next(iter(self._tombstones.items()))
            if now - at < TOMBSTONE_TTL_S:
                break
            self._tombstones.popitem(last=False)

    async def shutdown(self) -> None:
        for session in list(self.sessions.values()):
            if session.conn is not None:
                session.conn.close(1012, "server_restart")
            await session.end()
        self.sessions.clear()

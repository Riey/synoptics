"""The engine contract the live WebSocket layer drives.

One engine instance per session; it outlives individual WebSocket connections (paused while disconnected).
The WS layer owns the protocol (auth, framing, credit, gating against the last ``state`` snapshot, ``rev``,
line/req ids, TTS); the engine owns the guide: it decides what the state is, what to say and when it needs a
high-resolution frame, and answers every accepted stream frame with a ``track`` message.

The engine never sends to the socket itself. It reports through an ``EngineSink``:

* ``state(GuideState)`` — the FULL snapshot after every change; the server bumps ``rev`` and forwards it.
* ``say(text, mode)`` — a sentence to read; the server assigns ``line_id`` and handles TTS. Returns line_id.
* ``capture_hi()`` — ask the client for one high-resolution frame; returns the ``req_id``.

A refusal the client should see as ``error{code}`` is raised as ``EngineReject``.
"""

from __future__ import annotations

from typing import Literal, Protocol

from .contracts import (
    DEFAULT_CORE_MODE,
    DEFAULT_PLAN_MODEL,
    FrameHeader,
    GuideState,
    MaterialInput,
    PlanEditMsg,
    Prefs,
    ReferenceImage,
    TrackMsg,
)


class EngineReject(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = True, retry_after_ms: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms


class EngineSink(Protocol):
    def state(self, state: GuideState) -> None: ...

    def say(self, text: str, mode: Literal["replace", "append"]) -> str: ...

    def capture_hi(self) -> str: ...


class Engine(Protocol):
    async def on_start(self, goal: str, context: str | None, *, plan_mode: bool = False,
                       plan_model: str = DEFAULT_PLAN_MODEL, core_mode: str = DEFAULT_CORE_MODE,
                       materials: tuple[MaterialInput, ...] = (),
                       reference_images: tuple[ReferenceImage, ...] = ()) -> None: ...

    async def on_stop(self) -> None: ...

    # ---- §15 Plan 모드: 질문 왕복·검토·승인·일시정지·사용자 확인·변경안
    async def on_plan_answer(self, clarification_id: str, answer: str,
                             reference_images: tuple[ReferenceImage, ...] | None = None) -> None: ...
    async def on_plan_edit(self, edit: PlanEditMsg) -> None: ...
    async def on_plan_approve(self) -> None: ...
    async def on_plan_discard(self) -> None: ...
    async def on_run_pause(self) -> None: ...
    async def on_run_resume(self) -> None: ...
    async def on_step_ack(self, step_id: str) -> None: ...
    async def on_proposal(self, *, accept: bool) -> None: ...

    async def on_talk(self, utterance: str) -> None: ...

    async def on_follow_now(self) -> None: ...

    async def on_replan_now(self) -> None: ...

    async def on_confirm_done(self) -> None: ...

    async def on_prefs(self, prefs: Prefs) -> None: ...

    async def on_frame(self, header: FrameHeader, jpeg: bytes) -> TrackMsg: ...

    async def on_hi_frame(self, header: FrameHeader, jpeg: bytes) -> None: ...

    async def on_pause(self) -> None: ...

    async def on_resume(self) -> None: ...

    async def close(self) -> None: ...


def make_context(kind: str, settings, transport=None):
    """Per-app shared state of an engine kind (the real engine's upstream config and health cache), or None."""
    if kind == "real" and engine_available(kind, settings):
        from .real_engine import RealConfig, RealContext

        return RealContext(RealConfig(upstream_url=settings.upstream_url, access_code=settings.upstream_access_code),
                           transport=transport)
    return None


def make_engine(kind: str, sink: EngineSink, settings, context=None) -> Engine:
    if kind == "mock":
        from .mock_engine import MockConfig, MockEngine

        return MockEngine(sink, MockConfig(speed=settings.mock_speed, step_s=settings.mock_step_s))
    if kind == "real":
        from .real_engine import RealEngine

        if context is None:
            context = make_context(kind, settings)
        if context is None:
            raise ValueError("the real engine needs LIVE_UPSTREAM_URL")
        return RealEngine(sink, context)
    raise ValueError(f"unknown engine {kind!r} (known: 'mock', 'real')")


def engine_available(kind: str, settings=None) -> bool:
    if kind == "mock":
        return True
    if kind == "real":
        return bool(settings is not None and getattr(settings, "upstream_url", None))
    return False


__all__ = ["Engine", "EngineReject", "EngineSink", "GuideState", "make_context", "make_engine", "engine_available"]

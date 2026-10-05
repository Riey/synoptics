"""FastAPI app: ``synoptics_live.app:app`` (REST §3, WS §4, reference client at /demo/)."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import ValidationError

from .contracts import (
    PLAN_MODELS,
    PROTOCOL,
    BuildInfo,
    CreateSessionRequest,
    CreateSessionResponse,
    EndedResponse,
    HealthResponse,
    Limits,
)
from .engine import engine_available, make_context, make_engine
from .sessions import SessionError, SessionStore, tokens_equal
from .settings import Settings
from .tts import HttpTtsBridge, TtsBridge
from .ws import live_endpoint

log = logging.getLogger("synoptics_live")
STATIC = Path(__file__).parent / "static"
_DEFAULT: Any = object()


def _first(value: str | None) -> str | None:
    if not value:
        return None
    value = value.split(",")[0].strip()
    return value or None


def client_ip(request: Request) -> str:
    for name in ("cf-connecting-ip", "x-real-ip"):
        v = _first(request.headers.get(name))
        if v:
            return v
    return request.client.host if request.client else "unknown"


def live_url(request: Request, sid: str, path_prefixes: dict[str, str] | None = None) -> str:
    proto = (_first(request.headers.get("x-forwarded-proto")) or request.url.scheme).lower()
    host = _first(request.headers.get("x-forwarded-host")) or request.headers.get("host") or request.url.netloc
    prefix = (_first(request.headers.get("x-forwarded-prefix")) or "").rstrip("/")
    if prefix and not prefix.startswith("/"):
        prefix = ""
    if not prefix and path_prefixes:
        # Proxies that strip their mount path without announcing it (Tailscale Serve --set-path).
        prefix = path_prefixes.get(host.lower(), "")
    scheme = "wss" if proto in ("https", "wss") else "ws"
    return f"{scheme}://{host}{prefix}/v1/sessions/{sid}/live"


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def create_app(settings: Settings | None = None, tts: TtsBridge | None = _DEFAULT, *,
               upstream_transport: Any = None) -> FastAPI:
    """``upstream_transport``: an httpx transport for the real engine's upstream (tests)."""
    settings = settings or Settings.from_env()
    if tts is _DEFAULT:
        tts = (
            HttpTtsBridge(settings.tts_url, settings.tts_voice, settings.ffmpeg)
            if settings.tts_url
            else None
        )
    limits = Limits(resume_grace_ms=int(settings.resume_grace_s * 1000))
    engine_ok = engine_available(settings.engine, settings)
    engine_ctx = make_context(settings.engine, settings, transport=upstream_transport) if engine_ok else None
    store = SessionStore(
        settings,
        engine_factory=lambda session: make_engine(settings.engine, session, settings, engine_ctx),
        engine_ok=engine_ok,
        tts=tts,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await store.shutdown()
        if tts is not None and hasattr(tts, "aclose"):
            await tts.aclose()

    app = FastAPI(title="synoptics Live API", version=PROTOCOL, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.settings = settings
    app.state.store = store
    app.state.limits = limits
    app.state.tts = tts
    app.state.engine_ctx = engine_ctx
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Retry-After"],
        max_age=600,
    )

    @app.get("/v1/health")
    async def health() -> JSONResponse:
        await store.sweep()
        tts_ready = bool(tts is not None and await tts.ready())
        # §15: the mock engine serves every configured profile; a real engine reports the upstream's map.
        plan_models = {name: bool(engine_ok) for name in PLAN_MODELS}
        guide_ok = plan_ok = follow_ok = tracker_ok = ready = engine_ok
        if engine_ok and engine_ctx is not None:
            # Real engine: readiness is the upstream demo backend's (/api/health, cached ~10 s).
            up = await engine_ctx.health.get() or {}
            raw_models = up.get("plan_models")
            raw_models = raw_models if isinstance(raw_models, dict) else {}
            plan_models = {name: raw_models.get(name) is True for name in PLAN_MODELS}
            guide_ok = bool(up.get("guide_ready"))
            plan_ok = any(plan_models.values())  # aggregate: any configured profile makes a plan possible
            follow_ok = bool(up.get("follow_ready"))
            tracker_ok = bool(up.get("tracker_ready"))
            ready = bool(up.get("ready")) and guide_ok and tracker_ok
        body = HealthResponse(
            ready=ready,
            access_mode=store.access_mode,
            access_code_required=store.access_code_required,
            guide_ready=guide_ok,
            plan_ready=plan_ok,
            plan_models=plan_models,
            follow_ready=follow_ok,
            tracker_ready=tracker_ok,
            tts_ready=tts_ready,
            tts_voice=tts.voice_label if (tts is not None and tts_ready) else None,
            build=BuildInfo(commit=settings.build_commit, time=settings.build_time),
            engine=settings.engine,
        )
        return JSONResponse(body.model_dump(), headers={"Cache-Control": "no-store"})

    @app.post("/v1/sessions")
    async def create_session(request: Request) -> JSONResponse:
        raw = await request.body()
        try:
            data = json.loads(raw) if raw.strip() else {}
            req = CreateSessionRequest.model_validate(data)
        except (ValueError, ValidationError):
            return _error(400, "invalid_request", "요청 본문이 올바른 JSON이 아닙니다")
        try:
            session = await store.create(req.access_code, client_ip(request))
        except SessionError as exc:
            return _error(exc.status, exc.code, exc.message, exc.headers)
        body = CreateSessionResponse(
            session_id=session.sid,
            token=session.token,
            expires_at=session.expires_at_ms(),
            live_url=live_url(request, session.sid, settings.path_prefixes),
        )
        return JSONResponse(body.model_dump(), status_code=201, headers={"Cache-Control": "no-store"})

    @app.delete("/v1/sessions/{sid}")
    async def delete_session(sid: str, request: Request) -> JSONResponse:
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
        session = await store.get(sid)
        if session is None:
            return _error(404, "session_not_found", "세션이 없거나 이미 끝났습니다")
        if not token or not tokens_equal(token, session.token):
            return _error(401, "invalid_token", "토큰이 맞지 않습니다")
        await store.end(session, close_code=1000, reason="session_ended")
        return JSONResponse(EndedResponse().model_dump())

    @app.websocket("/v1/sessions/{sid}/live")
    async def live(ws: WebSocket, sid: str) -> None:
        await live_endpoint(ws, sid, store, settings, limits)

    @app.get("/demo", include_in_schema=False)
    async def demo_redirect() -> RedirectResponse:
        return RedirectResponse("demo/", status_code=308)  # relative: survives the proxy prefix

    @app.get("/demo/", include_in_schema=False)
    async def demo() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()

"""Ephemeral, same-origin guidance API with bounded image and provider calls."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import ipaddress
import json
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from backend.app.api_contracts import (
    EndSessionResponse,
    AccessMode,
    SessionRequest,
    SessionResponse,
)
from backend.app.astra import AstraProvider
from backend.app.bugreport import build_bugreport_router
from backend.app.build import build_info
from backend.app.errors import ApiFailure, StageConfigError, StageError
from backend.app.guide import GuideSessionState, StoredPlan, stale_plan
from backend.app.guide_contracts import (
    AnchorEcho,
    CORE_MODE_SEQUENTIAL,
    DEFAULT_CORE_MODE,
    DEFAULT_PLAN_MODEL,
    CoreMode,
    GuideConfirmRequest,
    GuideConfirmResponse,
    GuideConfirmToolOutput,
    GuideFollowRequest,
    GuideFollowResponse,
    GuideFollowToolOutput,
    GuideHealthResponse,
    GuidePlanApproveRequest,
    GuidePlanCurrentResponse,
    GuidePlanRequest,
    GuidePlanResponse,
    GuidePlanRevertRequest,
    GuidePlanToolOutput,
    GuideTalkRequest,
    GuideTalkResponse,
    GuideTalkToolOutput,
    MaterialInput,
    PlanAnswer,
    PlanModel,
    ResearchSource,
    check_grounded_evidence,
    drop_ungrounded_evidence,
)
from backend.app.intent import PlanHintScanner, confirm_allows_replan, confirm_checklist_ids, prepare_frame_b64
from backend.app.plan_manual import MANUAL_REQUEST_MAX_BYTES, ManualImportRequest, extract_manual
from backend.app.clef_follow import ClefFollower, build_clef_follower
from backend.app.local_follow import LocalFollower, build_local_follower
from backend.app.provider import (
    DeepSeekProvider,
    GuideStreamFinal,
    InvalidOutputStage,
    ProviderConfigError,
    ProviderError,
    ProviderInvalidOutput,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    UnavailableStage,
    get_provider,
)
from backend.app.tracking import (
    TrackerClient,
    TrackerRunRegistry,
    build_tracker_client,
    project_control,
    project_frame,
    release_run,
)
from backend.app.tracking_contracts import (
    TargetSelectionToolOutput,
    TrackControlRequest,
    TrackControlResponse,
    TrackFrameQuery,
    TrackFrameRequest,
    TrackFrameResponse,
    TrackSelectRequest,
    TrackSelectResponse,
)
from backend.app.tts import TtsRequest, build_tts_router


logger = logging.getLogger(__name__)


MAX_GUIDANCE_BODY = 10 * 1024 * 1024
MAX_STANDARD_BODY = 64 * 1024
#: One frame plus its seed box: the image itself is capped at MAX_IMAGE bytes by validate_jpeg, so this is
#: the base64-inflated envelope with room to spare.
MAX_TRACK_BODY = 4 * 1024 * 1024
MAX_IMAGE = 1_500_000
SESSION_IDLE = 15 * 60
COOKIE = "visual_coach_session"
# Explicit, single-switch local-only deployment mode. Only the exact value "1" enables it, so a
# typo or an unrelated truthy value can never silently drop the access-code requirement.
LOCAL_ONLY_ENV = "AISW_LOCAL_ONLY"
# Explicit open-access mode: sessions without any code, for a deployment whose network boundary is the
# authentication (a tailnet-only Tailscale Serve). Same exact-"1" rule. It conflicts with local-only mode
# (one says "proxied traffic welcome", the other "never proxied"), so both together refuse every session.
OPEN_ACCESS_ENV = "AISW_OPEN_ACCESS"
#: Which model answers ``/api/guide/follow``: ``local`` (default, a loopback llama.cpp server, see
#: ``local_follow.py``), ``clef`` (a loopback Clef decision server, see ``clef_follow.py``) or ``deepseek``
#: (the paid provider). Any other value disables the follow route.
FOLLOW_PROVIDER_ENV = "AISW_FOLLOW_PROVIDER"
FOLLOW_PROVIDERS = ("local", "clef", "deepseek")
#: The followers that run on this machine: no paid rate lane, no provider slot, readiness by their own probe.
LOOPBACK_FOLLOWERS = (LocalFollower, ClefFollower)


@dataclass(slots=True)
class Session:
    """Session state: rate limits, the current task text and its revision, and the per-lane run state."""

    id: str
    expires: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    calls: deque[float] = field(default_factory=deque)
    analyses: deque[float] = field(default_factory=deque)
    last_analyze_at: float = 0
    active: bool = True
    task_goal: str | None = None
    task_context: str | None = None
    #: The §15 reference materials of the CURRENT task (``None`` when the plan start sent none). Compared on
    #: every plan start: a changed list under the same goal/context is a new task (``task_revision`` bumps and
    #: the plan is cleared), so a plan can never inherit authority from a different source. Task identity only:
    #: the material text is never echoed or logged, and the plan keeps its own copy for replan prompts.
    task_materials: list[MaterialInput] | None = None
    #: Which upper planning profile the CURRENT task uses (``deepseek:high`` or ``astra:high``). It joins the
    #: task identity alongside the goal/context/materials: a plan start that changes it is a new task, so a
    #: plan made for one choice can never be inherited, revised or reverted back into by the other.
    task_plan_model: PlanModel = DEFAULT_PLAN_MODEL
    #: Which guide core the CURRENT task runs (``classic``, ``sequential`` or ``graph``). It joins the task
    #: identity the same way ``task_plan_model`` does: changing it under the same goal/context is a new task, so
    #: the old plan and its history are cleared and can never be inherited, revised or reverted into by the
    #: other core.
    task_core_mode: CoreMode = DEFAULT_CORE_MODE
    task_assisted: bool = False
    task_answers: list[PlanAnswer] = field(default_factory=list)
    task_sources: list[ResearchSource] = field(default_factory=list)
    task_revision: int = 0
    # Standalone tracking is scoped to this session: its own run registry (start_seq/frame_seq fencing).
    # The model state lives in the tracker service, never here. No lock is held here across an await: every
    # registry mutation is a single synchronous call, which is what lets a stop retire a run whose frame is
    # still in flight (that frame's answer is then rejected by the post-await fence).
    tracker: TrackerRunRegistry = field(default_factory=TrackerRunRegistry)
    #: True while a startup target-selection call is in flight for this session. It is read and written
    #: without an await in between, so two overlapping selections cannot both reach the provider; the
    #: second is refused at once rather than queueing on the session lock and dispatching a second paid
    #: call once the first has been running longer than the manual cooldown. It says nothing about any
    #: tracker run, and the guide lane does not consult it.
    select_inflight: bool = False
    #: The anchored guide lane's plan (text only) and its in-flight flags. Cleared with the task.
    guide: GuideSessionState = field(default_factory=GuideSessionState)


class Store:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.creations: dict[str, deque[float]] = {}
        self.guard = asyncio.Lock()
        self.provider_slots = asyncio.Semaphore(4)
        self.client = httpx.AsyncClient()
        self.provider = get_provider(self.client)
        # The Plan-only OpenAI API lane is independent of the basic provider. Construction reads no key;
        # missing credentials are a per-call failure, never a reason to dispatch to the basic lane.
        self.astra = AstraProvider(self.client)
        # The standalone tracker transport, built once on first use so its environment is read once. It is
        # a separate lane: no provider slot, no paid credential, and no failure of one implies the other.
        self.tracker_client: TrackerClient | None = None
        self.tracker_error: StageError | None = None
        # The local follower, built once (at startup when selected) so its environment is read once. A
        # configuration failure is recorded, reported by health, and refuses the follow route; never degrades.
        self.local_follower: LocalFollower | None = None
        self.local_follower_error: StageError | None = None
        self.clef_follower: ClefFollower | None = None
        self.clef_follower_error: StageError | None = None

    def upper_provider(self, plan_model: PlanModel) -> Any:
        """The adapter for the selected ``plan_model``: the OpenAI lane for ``astra:high``, else the basic paid one.

        Read-only and infallible: the choice is a field, never a probe. Whether the chosen lane is actually
        usable is checked separately (``plan_model_ready``), so a missing credential is a clean refusal, never
        a silent fallback to the other provider.
        """
        return self.astra if plan_model == "astra:high" else self.provider

    def follower(self) -> LocalFollower | ClefFollower | Any:
        """The configured follower (the paid provider, the local or the Clef one), or a configuration failure."""
        choice = follow_provider_choice()
        if choice is None:
            raise StageConfigError(f"{FOLLOW_PROVIDER_ENV} must be 'local', 'clef' or 'deepseek'")
        if choice == "deepseek":
            return self.provider
        if choice == "clef":
            if self.clef_follower is None:
                if self.clef_follower_error is None:
                    try:
                        self.clef_follower = build_clef_follower(self.client)
                    except StageError as exc:
                        self.clef_follower_error = exc
                        raise
                else:
                    raise self.clef_follower_error
            return self.clef_follower
        if self.local_follower is None:
            if self.local_follower_error is None:
                try:
                    self.local_follower = build_local_follower(self.client)
                except StageError as exc:
                    self.local_follower_error = exc
                    raise
            else:
                raise self.local_follower_error
        return self.local_follower

    def tracker(self) -> TrackerClient:
        """The tracker transport, or the recorded configuration failure. Fails closed, never degrades."""
        if self.tracker_client is None:
            if self.tracker_error is None:
                try:
                    self.tracker_client = build_tracker_client(self.client)
                except StageError as exc:
                    self.tracker_error = exc
                    raise
            else:
                raise self.tracker_error
        return self.tracker_client

    async def close(self) -> None:
        await self.client.aclose()

    async def abandon_tracker_run(self, session: Session) -> None:
        """Stop the session's tracker run, if any, without letting a tracker failure delay the caller."""
        run = session.tracker.run
        if run is not None:
            run.active = False
        try:
            await release_run(run, self.tracker())
        except StageError:
            pass

    async def sweep(self) -> None:
        while True:
            await asyncio.sleep(1)
            now = time.monotonic()
            expired: list[Session] = []
            async with self.guard:
                for token, session in list(self.sessions.items()):
                    if session.expires <= now:
                        session.active = False
                        session.calls.clear()
                        session.analyses.clear()
                        session.guide.clear()
                        session.task_goal = None
                        session.task_context = None
                        session.task_materials = None
                        session.task_plan_model = DEFAULT_PLAN_MODEL
                        session.task_core_mode = DEFAULT_CORE_MODE
                        session.task_revision += 1
                        del self.sessions[token]
                        expired.append(session)
                for ip, attempts in list(self.creations.items()):
                    while attempts and attempts[0] <= now - 60:
                        attempts.popleft()
                    if not attempts:
                        del self.creations[ip]
            # Outside the guard: releasing a run talks to another process and must not hold session state.
            for session in expired:
                await self.abandon_tracker_run(session)

    async def session(self, token: str | None) -> Session:
        if not token:
            raise ApiFailure(401, "session_required")
        async with self.guard:
            now = time.monotonic()
            session = self.sessions.get(token)
            if not session or not session.active or session.expires <= now:
                self.sessions.pop(token, None)
                raise ApiFailure(401, "session_expired")
            session.expires = now + SESSION_IDLE
            return session

    async def count_creation(self, ip: str) -> None:
        async with self.guard:
            now = time.monotonic()
            calls = self.creations.setdefault(ip, deque())
            while calls and calls[0] <= now - 60:
                calls.popleft()
            if len(calls) >= 5:
                raise ApiFailure(429, "rate_limited", max(1, int(calls[0] + 60 - now) + 1))
            calls.append(now)

    async def create(self) -> tuple[str, Session]:
        async with self.guard:
            if len(self.sessions) >= 32:
                raise ApiFailure(429, "sessions_full", 60)
            now = time.monotonic()
            token = uuid.uuid4().hex + uuid.uuid4().hex
            session = Session(str(uuid.uuid4()), now + SESSION_IDLE)
            self.sessions[token] = session
            return token, session

    async def end(self, token: str) -> None:
        async with self.guard:
            session = self.sessions.pop(token, None)
            if session:
                session.expires = 0
                session.active = False
                session.calls.clear()
                session.analyses.clear()
                session.guide.clear()
                session.task_goal = None
                session.task_context = None
                session.task_materials = None
                session.task_plan_model = DEFAULT_PLAN_MODEL
                session.task_core_mode = DEFAULT_CORE_MODE
                session.task_revision += 1
        if session:
            await self.abandon_tracker_run(session)


store = Store()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    access_mode_startup_check()
    # A loopback follower validates its configuration at startup, so an unusable lane is visible in /api/health
    # immediately. A failure is recorded rather than fatal: health stays answerable and the route fails closed.
    if follow_provider_choice() in ("local", "clef"):
        try:
            store.follower()
        except StageError:
            pass
    sweeper = asyncio.create_task(store.sweep())
    try:
        yield
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except asyncio.CancelledError:
            pass
        await store.close()


app = FastAPI(docs_url="/docs", redoc_url=None, lifespan=lifespan)


@app.exception_handler(ApiFailure)
async def api_error(_request: Request, exc: ApiFailure) -> JSONResponse:
    if exc.detail is None:
        body: dict[str, Any] = {"error": exc.code}
    else:
        # Truncated defensively: it is a diagnostic string, not a payload.
        body = {"error": exc.code, "detail": exc.detail[:200]}
    if exc.reason is not None:
        # Additive: the error code stays exactly what it was, and the stage is added only where it is known.
        body["reason"] = exc.reason.value
    if exc.extra:
        # The structured fields a frozen error body requires (an active run id, the sequence to continue
        # from). Merged after the code so a code can never be overwritten by them.
        body.update(exc.extra)
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
    return JSONResponse(body, status_code=exc.status, headers=headers)


def log_provider_failure(
    code: str,
    stage: InvalidOutputStage | UnavailableStage,
    mode: str,
    started: float,
) -> None:
    """One enum-only line per rejected provider answer.

    Closed-set tokens and a duration are the whole record: the provider's own text, the exception
    message, the goal/context/question, image bytes and every identifier are absent by construction,
    so this line is safe to read out of a service log and is enough to tell a truncated answer from a
    schema, provenance or context refusal, and a network failure from an upstream error status.
    """
    logger.warning(
        "provider call rejected code=%s stage=%s mode=%s elapsed_ms=%d",
        code,
        stage.value,
        mode,
        int((time.monotonic() - started) * 1000),
    )


@app.exception_handler(ValidationError)
async def validation_error(_request: Request, _exc: ValidationError) -> JSONResponse:
    return JSONResponse({"error": "invalid_request"}, status_code=422)


def origin_allowed(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        raise ApiFailure(403, "origin_required")
    host = request.headers.get("host", "")
    if not host or any(c in host for c in "@/\\ \t\r\n"):
        raise ApiFailure(403, "invalid_origin")
    scheme = request.scope.get("scheme", "http")
    if origin == f"{scheme}://{host}":
        return
    raise ApiFailure(403, "invalid_origin")


def secure_cookie(request: Request) -> bool:
    """The intended local deployment is plain-HTTP loopback, so its session cookie is not ``Secure``.

    Every other origin (including a proxy that forwards a public authority) gets ``Secure``. This says
    nothing about whether a browser would accept a ``Secure`` cookie on localhost — it only keeps the
    intended local deployment working without TLS.
    """
    peer = request.client.host if request.client else ""
    return not (is_loopback_address(peer) and is_loopback_authority(request.headers.get("host", "")))


def is_loopback_address(raw: str) -> bool:
    """True for a loopback peer address, including the IPv4-mapped IPv6 form a dual-stack socket reports."""
    try:
        address = ipaddress.ip_address(raw.strip())
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def is_loopback_authority(host_header: str) -> bool:
    """True only for a ``Host`` naming a loopback authority: ``localhost`` or a loopback literal.

    Userinfo, paths, whitespace, and a non-numeric port all disqualify the authority, so a forwarded
    public authority (``aisw.example.com``) can never be read as loopback by accident.
    """
    host = host_header.strip()
    if not host or any(character in host for character in "@/\\ \t\r\n"):
        return False
    if host.startswith("["):
        end = host.find("]")
        if end == -1:
            return False
        literal, rest = host[1:end], host[end + 1:]
        return (not rest or (rest.startswith(":") and rest[1:].isdigit())) and is_loopback_address(literal)
    name, separator, port = host.partition(":")
    if separator and not port.isdigit():
        return False
    return name.lower() == "localhost" or is_loopback_address(name)


def has_forwarding_metadata(request: Request) -> bool:
    """Whether the request carries HTTP forwarding metadata.

    `request.client` may already have been rewritten by proxy-headers middleware (uvicorn trusts a
    loopback proxy by default), so a forwarded request can look like a loopback peer. Forwarding
    metadata is therefore treated as proof that the request did not arrive on a raw loopback socket.
    """
    for name in request.headers.keys():
        lowered = name.lower()
        if lowered == "forwarded" or lowered == "x-real-ip" or lowered.startswith("x-forwarded-"):
            return True
    return False


def follow_provider_choice() -> str | None:
    """``AISW_FOLLOW_PROVIDER`` (default ``local``), or ``None`` when it names no known follower."""
    choice = (os.getenv(FOLLOW_PROVIDER_ENV) or "local").strip().lower()
    return choice if choice in FOLLOW_PROVIDERS else None


def local_only_mode() -> bool:
    """Whether this process was explicitly deployed as a local-only (no access code) deployment."""
    return os.getenv(LOCAL_ONLY_ENV) == "1"


def open_access_mode() -> bool:
    """Whether this process was explicitly deployed in open-access (no code, proxied traffic allowed) mode."""
    return os.getenv(OPEN_ACCESS_ENV) == "1"


def access_mode() -> AccessMode | None:
    """The deployment's session mode, or ``None`` for the refused open + local combination."""
    local, open_access = local_only_mode(), open_access_mode()
    if local and open_access:
        return None
    if local:
        return "local"
    return "open" if open_access else "code"


def access_mode_startup_check() -> None:
    """Log (enum-only, never a value) a refused or overridden access configuration at startup."""
    mode = access_mode()
    if mode is None:
        logger.error("access_mode_conflict: %s and %s are both set; every session is refused (503)",
                     OPEN_ACCESS_ENV, LOCAL_ONLY_ENV)
    elif mode == "open" and os.getenv("DEMO_ACCESS_CODE"):
        logger.warning("access_code_ignored: %s=1 issues sessions without a code; DEMO_ACCESS_CODE is not used",
                       OPEN_ACCESS_ENV)


def enforce_access_mode(request: Request) -> AccessMode:
    """The access gate every session route runs after the origin check.

    A refused configuration (open + local) is ``503 service_unavailable``; local mode refuses non-local
    traffic (``enforce_local_only``); open and code modes accept any request that passed the origin check.
    """
    mode = access_mode()
    if mode is None:
        raise ApiFailure(503, "service_unavailable")
    if mode == "local":
        enforce_local_only(request)
    return mode


def require_access_deployment(request: Request) -> AccessMode:
    """``enforce_access_mode`` plus: a code deployment without a configured code is ``503``."""
    mode = enforce_access_mode(request)
    if mode == "code" and not os.getenv("DEMO_ACCESS_CODE"):
        raise ApiFailure(503, "service_unavailable")
    return mode


def enforce_local_only(request: Request) -> bool:
    """Refuse non-local traffic when local mode is on; report whether this request is served locally.

    Local mode means "the browser reached this process directly over loopback" (a local browser or a raw
    SSH tunnel). A request is served code-lessly only when it carries no forwarding metadata
    (``Forwarded``/``X-Forwarded-*``/``X-Real-IP``), has a loopback peer address, and names a loopback
    ``Host`` authority; anything else is refused.

    This is a narrowing check, not a proof about the network in front of the process: a proxy can rewrite
    ``Host`` and strip forwarding metadata. What actually keeps a code-free process off the network is
    the deployment itself — binding to loopback and reaching it through an authenticated SSH tunnel, as
    the documented deployments do. Never place a code-free instance behind a public relay.
    """
    if not local_only_mode():
        return False
    peer = request.client.host if request.client else ""
    if (
        has_forwarding_metadata(request)
        or not is_loopback_address(peer)
        or not is_loopback_authority(request.headers.get("host", ""))
    ):
        raise ApiFailure(403, "local_only")
    return True


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def no_nonfinite(_value: str) -> Any:
    raise ValueError("nonfinite JSON number")


def content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


async def body_bytes(request: Request, *, max_bytes: int) -> bytes:
    size = 0
    body = bytearray()
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_bytes:
            raise ApiFailure(413, "body_too_large")
        body.extend(chunk)
    return bytes(body)


async def body_model(request: Request, model: type[Any], *, max_bytes: int) -> Any:
    if content_type(request) != "application/json":
        raise ApiFailure(415, "json_required")
    body = await body_bytes(request, max_bytes=max_bytes)
    try:
        obj = json.loads(body, object_pairs_hook=no_duplicate_keys, parse_constant=no_nonfinite)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ApiFailure(422, "invalid_request") from None
    try:
        return model.model_validate(obj)
    except ValidationError:
        raise ApiFailure(422, "invalid_request") from None


def track_frame_query(request: Request) -> TrackFrameQuery:
    """The query string of a binary track frame as ``TrackFrameQuery``: each field exactly once, nothing else."""
    items = request.query_params.multi_items()
    fields: dict[str, Any] = {}
    for key, value in items:
        if key in fields:
            raise ApiFailure(422, "invalid_request")
        fields[key] = value
    raw_seq = fields.get("frame_seq")
    if isinstance(raw_seq, str):
        # Canonical decimal only ("007", "+1", " 1" are refused), so one frame has one spelling.
        if not raw_seq.isascii() or not raw_seq.isdigit() or (len(raw_seq) > 1 and raw_seq[0] == "0"):
            raise ApiFailure(422, "invalid_request")
        fields["frame_seq"] = int(raw_seq)
    try:
        return TrackFrameQuery.model_validate(fields)
    except ValidationError:
        raise ApiFailure(422, "invalid_request") from None


def strip_data_url(encoded: str) -> str:
    if encoded.startswith("data:") and "," in encoded:
        return encoded.split(",", 1)[1].strip()
    return encoded.strip()


def validate_jpeg(encoded: str, *, min_width: int = 320, min_height: int = 240, max_dim: int = 2048,
                  max_pixels: int = 2_073_600, max_bytes: int = MAX_IMAGE) -> str:
    clean = strip_data_url(encoded)
    if len(clean) > (max_bytes + 2) // 3 * 4 + 4:
        raise ApiFailure(422, "image_too_large")
    try:
        image_bytes = base64.b64decode(clean, validate=True)
    except (ValueError, binascii.Error):
        raise ApiFailure(422, "invalid_image") from None
    validate_jpeg_bytes(image_bytes, min_width=min_width, min_height=min_height, max_dim=max_dim,
                        max_pixels=max_pixels, max_bytes=max_bytes)
    return clean


def validate_jpeg_bytes(image_bytes: bytes, *, min_width: int = 320, min_height: int = 240, max_dim: int = 2048,
                        max_pixels: int = 2_073_600, max_bytes: int = MAX_IMAGE) -> None:
    """The decoded-image checks of ``validate_jpeg``, for a JPEG that arrived as raw bytes."""
    if len(image_bytes) > max_bytes:
        raise ApiFailure(422, "image_too_large")
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise ApiFailure(422, "invalid_image")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "JPEG":
                raise ApiFailure(422, "invalid_image")
            w, h = image.size
            if w < min_width or h < min_height or w > max_dim or h > max_dim:
                raise ApiFailure(422, "invalid_image_dimensions")
            if w * h > max_pixels:
                raise ApiFailure(422, "invalid_image_dimensions")
            image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise ApiFailure(422, "invalid_image") from None


async def acquire_provider() -> None:
    if store.provider_slots.locked():
        raise ApiFailure(503, "provider_busy", 1)
    await store.provider_slots.acquire()


# Analysis pacing is per request mode. The manual lane (the startup target selection, one deliberate
# click) is paced at one analysis per 6s and ten per minute.
MANUAL_ANALYSIS_COOLDOWN = 6.0
MANUAL_ANALYSES_PER_MINUTE = 10

#: The anchored guide lane's paid calls (plan, confirm, and a DeepSeek follow) have their own pacing: they
#: are short forced-tool calls dispatched on events, not on a frame cadence. They still share the session's
#: one analysis queue and its 20-per-minute request ceiling, so no lane can spend beyond that budget.
GUIDE_MODE = "guide"
GUIDE_ANALYSIS_COOLDOWN = 1.0
GUIDE_ANALYSES_PER_MINUTE = 20

#: Log mode tokens for the guide routes, so a rejected answer names the route it came from.
GUIDE_PLAN_MODE = "guide_plan"
GUIDE_FOLLOW_MODE = "guide_follow"
GUIDE_CONFIRM_MODE = "guide_confirm"
GUIDE_TALK_MODE = "guide_talk"

#: Log mode token for the startup target-selection call. It is not a guidance analysis (no frame cadence,
#: no history, no goal), so the log line names what it actually was.
SELECT_MODE = "select"


def provider_failure(error: ProviderError, *, mode: str, started: float) -> ApiFailure:
    """The one refusal this API answers for a provider failure, whichever lane hit it.

    Kept in one place so a given failure has exactly one status code, one error code, and one log line
    whether it arrived on the guide lane or the startup-selection lane. Provider text never goes into
    the answer: the stage token is the whole detail.
    """
    if isinstance(error, ProviderRateLimited):
        return ApiFailure(429, "provider_rate_limited", error.retry_after)
    if isinstance(error, ProviderTimeout):
        return ApiFailure(504, "provider_timeout")
    if isinstance(error, ProviderInvalidOutput):
        log_provider_failure("invalid_provider_output", error.stage, mode, started)
        return ApiFailure(502, "invalid_provider_output", reason=error.stage)
    if isinstance(error, ProviderConfigError):
        return ApiFailure(503, "provider_configuration")
    if isinstance(error, ProviderUnavailable):
        log_provider_failure("provider_unavailable", error.stage, mode, started)
        return ApiFailure(502, "provider_unavailable", reason=error.stage)
    # Unreachable today (those five are every ``ProviderError`` subclass), and fail-closed if that changes:
    # an unrecognised provider failure is still an answer the client must not act on.
    return ApiFailure(502, "provider_unavailable")


def enforce_rate(session: Session, *, analysis: bool, mode: str = "manual") -> None:
    now = time.monotonic()
    for queue in (session.calls, session.analyses):
        while queue and queue[0] <= now - 60:
            queue.popleft()
    if len(session.calls) >= 20:
        raise ApiFailure(429, "rate_limited", max(1, int(session.calls[0] + 60 - now) + 1))
    if analysis:
        if mode == GUIDE_MODE:
            cooldown, per_minute = GUIDE_ANALYSIS_COOLDOWN, GUIDE_ANALYSES_PER_MINUTE
        else:
            cooldown, per_minute = MANUAL_ANALYSIS_COOLDOWN, MANUAL_ANALYSES_PER_MINUTE
        if now - session.last_analyze_at < cooldown:
            raise ApiFailure(429, "rate_limited", max(1, int(session.last_analyze_at + cooldown - now) + 1))
        if len(session.analyses) >= per_minute:
            raise ApiFailure(429, "rate_limited", max(1, int(session.analyses[0] + 60 - now) + 1))
        session.last_analyze_at = now
        session.analyses.append(now)
    session.calls.append(now)


def tracker() -> TrackerClient:
    """The tracker transport for a request path: a configuration failure is an unavailable lane.

    The failure names the environment variable that is wrong (never a value), which is what an operator
    needs to fix the deployment; the client sees one closed-set error.
    """
    try:
        return store.tracker()
    except StageConfigError as exc:
        raise ApiFailure(503, "tracker_unavailable", None, str(exc)) from None


async def tracker_readiness() -> tuple[bool, str | None]:
    """Tracker readiness plus its reason, independent of paid provider readiness.

    Read-only and never fatal: a misconfigured or unreachable tracker is reported as not ready, so /health
    stays answerable in a deployment where only the tracker lane exists.
    """
    try:
        return await store.tracker().ready()
    except StageConfigError as exc:
        return False, str(exc)


def plan_models() -> dict[PlanModel, bool]:
    """Configured capability of each selectable upper planning profile, keyed by ``plan_model``.

    ``True`` means that profile's credential is configured; it never proves entitlement or remaining credit
    (only a model call does). ``deepseek:high`` is the basic lane's readiness, ``astra:high`` the OpenAI
    lane's. The two are independent.
    """
    return {
        "deepseek:high": isinstance(store.provider, DeepSeekProvider) and bool(store.provider.api_key),
        "astra:high": bool(store.astra.configured),
    }


def plan_model_ready(plan_model: PlanModel) -> bool:
    """Whether the given upper planning profile can run (configured capability only)."""
    return plan_models()[plan_model]


def plan_ready() -> bool:
    """Aggregate any-ready: at least one upper planning profile is configured. Never a per-choice answer."""
    return any(plan_models().values())


def guide_ready() -> bool:
    """The BASIC guide lane can run: the configured provider is DeepSeek and holds a key.

    Kept at its original meaning (the ``deepseek:high`` profile); the OpenAI lane never makes it true and
    never requires its key.
    """
    return isinstance(store.provider, DeepSeekProvider) and bool(store.provider.api_key)


async def follow_readiness() -> bool:
    """The configured follower can run. Read-only and never fatal, like the tracker readiness."""
    try:
        follower = store.follower()
    except StageError:
        return False
    if isinstance(follower, LOOPBACK_FOLLOWERS):
        return await follower.ready()
    return guide_ready()


@app.get("/api/health", response_model=GuideHealthResponse)
async def health() -> GuideHealthResponse:
    mode = access_mode()
    code_less = mode in ("local", "open")
    tracker_ready, _tracker_error = await tracker_readiness()
    return GuideHealthResponse(
        ready=bool(store.provider.api_key)
        and (code_less or (mode == "code" and bool(os.getenv("DEMO_ACCESS_CODE")))),
        model=store.provider.model,
        provider=store.provider.provider_name,
        access_code_required=not code_less,
        access_mode=mode,
        tracker_ready=tracker_ready,
        guide_ready=guide_ready(),
        plan_ready=plan_ready(),
        plan_models=plan_models(),
        follow_provider=follow_provider_choice(),
        follow_ready=await follow_readiness(),
        build=build_info(),
    )


@app.post("/api/session", response_model=SessionResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": SessionRequest.model_json_schema()}}},
})
async def create_session(request: Request) -> JSONResponse:
    origin_allowed(request)
    mode = enforce_access_mode(request)
    code_less = mode in ("local", "open")
    access_code = os.getenv("DEMO_ACCESS_CODE") or ""
    # A session is the prerequisite for both lanes: the guide needs an upper planning provider (any
    # ``plan_model`` profile's key), tracking needs the local tracker. Any one opens a session;
    # the access/origin checks below are unchanged.
    paid_ready = bool(store.provider.api_key) or plan_ready()
    tracker_ready, _tracker_error = await tracker_readiness()
    if not (paid_ready or tracker_ready) or not (code_less or access_code):
        raise ApiFailure(503, "service_unavailable")
    payload = await body_model(request, SessionRequest, max_bytes=MAX_STANDARD_BODY)
    await store.count_creation(request.client.host if request.client else "unknown")
    if not code_less:
        supplied = payload.access_code or ""
        if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), access_code.encode("utf-8")):
            raise ApiFailure(401, "invalid_access_code")
    token, session = await store.create()
    response = JSONResponse(SessionResponse(session_id=session.id).model_dump(), status_code=200)
    response.set_cookie(COOKIE, token, httponly=True, secure=secure_cookie(request), samesite="strict", path="/")
    return response


@app.post("/api/session/end", response_model=EndSessionResponse)
async def end_session(request: Request) -> JSONResponse:
    origin_allowed(request)
    enforce_access_mode(request)
    token = request.cookies.get(COOKIE)
    if token:
        await store.end(token)
    response = JSONResponse(EndSessionResponse(ended=True).model_dump(), status_code=200)
    response.delete_cookie(COOKIE, path="/")
    return response


@app.post("/api/track/control", response_model=TrackControlResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": TrackControlRequest.model_json_schema()}}},
})
async def track_control(request: Request) -> TrackControlResponse:
    """Start or stop the session's tracking run. No frame, no provider credential, no provider slot."""
    origin_allowed(request)
    enforce_access_mode(request)
    payload = await body_model(request, TrackControlRequest, max_bytes=MAX_STANDARD_BODY)
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")

    if payload.action == "start":
        # No implicit target: a start with an absent or blank one is refused rather than started with a
        # class name nobody chose. The client names the object — the startup analysis's selection, or the
        # user's own words for a box they drew.
        target = (payload.target or "").strip()
        if not target:
            raise ApiFailure(422, "target_required")
        run, retired, opened = session.tracker.apply_start(
            run_id=payload.run_id, start_seq=payload.start_seq, target=target
        )
        # The replaced run's GPU state is released before the new run is registered, so the two never share
        # mutable state. The registry already points at the new run, so this ordering cannot lose the race
        # with a concurrent frame: such a frame is admitted against the NEW run only.
        if retired is not None:
            await release_run(retired, tracker())
        await tracker().start(run.run_id, run.target, fresh=opened)
        # Post-await fence: the session may have ended, or this run may have been stopped or replaced while
        # the service call was in flight. Either way this start must not be reported as good.
        if not session.active or store.sessions.get(token) is not session:
            raise ApiFailure(401, "session_expired")
        if session.tracker.run is not run or not run.active:
            raise ApiFailure(409, "stale_run", extra={"active_run_id": session.tracker.active_run_id})
        return project_control(run)
    run = session.tracker.apply_stop(run_id=payload.run_id)
    # Retired in this application before the service is told: new work cannot join it and a frame still in
    # flight is refused by the post-await fence, whether or not this release succeeds or returns quickly.
    await release_run(run, tracker())
    return project_control(run)


@app.post("/api/track/frame", response_model=TrackFrameResponse, openapi_extra={
    "requestBody": {"required": True, "content": {
        "application/json": {"schema": TrackFrameRequest.model_json_schema()},
        # The same frame as raw bytes; its fields are the query string (`TrackFrameQuery`), no seed_box.
        "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
    }},
    "parameters": [
        {"name": name, "in": "query", "required": False, "schema": schema}
        for name, schema in TrackFrameQuery.model_json_schema()["properties"].items()
    ],
})
async def track_frame(request: Request) -> TrackFrameResponse:
    """One frame of the active run: the tracker answers, this application only admits and forwards it.

    Two equivalent bodies: JSON (``TrackFrameRequest``, base64 image) or ``Content-Type: image/jpeg`` with the
    JPEG bytes as the body and ``TrackFrameQuery`` as the query string. The binary form spares the client a
    base64 encode and a third of the upload per frame; the seeded first frame of a run is always JSON.
    """
    origin_allowed(request)
    enforce_access_mode(request)
    if content_type(request) == "image/jpeg":
        payload = track_frame_query(request)
        image_bytes = await body_bytes(request, max_bytes=MAX_TRACK_BODY)
        # Validated before any run state is touched, so a malformed frame can never advance a run.
        validate_jpeg_bytes(image_bytes)
        # The tracker service's own wire is JSON + base64; encoding here costs the server well under a ms.
        image = base64.b64encode(image_bytes).decode("ascii")
    else:
        payload = await body_model(request, TrackFrameRequest, max_bytes=MAX_TRACK_BODY)
        image = None
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")

    if image is None:
        # Validated before any run state is touched, so a malformed frame can never advance a run.
        image = validate_jpeg(payload.image_base64)

    run = session.tracker.run_for(payload.run_id)
    async with run.inflight:
        # Held only for this run and only for the duration of the step: control never takes it, so a stop or
        # a new start retires state immediately instead of waiting for a frame that is in flight.
        session.tracker.admit_frame(run, frame_seq=payload.frame_seq, has_seed=getattr(payload, "seed_box", None) is not None)
        seed_box = getattr(payload, "seed_box", None)
        seed = seed_box.model_dump() if seed_box is not None else None
        body, elapsed_ms = await tracker().frame(run.run_id, payload.frame_seq, image, seed)
        response = project_frame(
            body,
            run=run,
            frame_id=payload.frame_id,
            frame_seq=payload.frame_seq,
            ingest_age_ms=round(elapsed_ms),
        )
        # Post-await fence, straight-line and therefore atomic: the session may have ended, or this run may
        # have been stopped or replaced while the frame was in flight. A late answer is then discarded and
        # no state is advanced, so old work can never reactivate a retired run.
        if not session.active or store.sessions.get(token) is not session:
            raise ApiFailure(401, "session_expired")
        if session.tracker.run is not run or not run.active:
            raise ApiFailure(409, "stale_run", extra={"active_run_id": session.tracker.active_run_id})
        session.tracker.accepted(run, frame_seq=payload.frame_seq, version=response.version)
        return response


@app.post("/api/track/select", response_model=TrackSelectResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": TrackSelectRequest.model_json_schema()}}},
})
async def track_select(request: Request) -> TrackSelectResponse:
    """One startup analysis: choose what to track, from the frame captured right now.

    The tracking lane's only paid call, and the only route in it that never touches a run: the answer is a
    candidate target (or an honest non-selection) that the client starts a run with. Recurring frames never
    come back here, so this costs one call per Start rather than one per frame.

    Pure means it writes no tracker, task or selection state. It does spend the session's own accounting:
    one manual analysis (cooldown and per-minute ceiling) and one provider slot.
    """
    origin_allowed(request)
    require_access_deployment(request)
    # The startup analysis needs a paid provider. A session can exist without one (the tracker lane opens
    # sessions on its own readiness), so this is checked before the body is parsed and long before any
    # provider work: an unconfigured paid lane is refused locally and no socket is ever opened for it.
    if not store.provider.api_key:
        raise ApiFailure(503, "service_unavailable")
    payload = await body_model(request, TrackSelectRequest, max_bytes=MAX_GUIDANCE_BODY)
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not payload.consent_ai:
        raise ApiFailure(400, "ai_consent_required")
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")

    scene_image = validate_jpeg(payload.scene.image_base64)
    clean_request = payload.model_copy(update={
        "scene": payload.scene.model_copy(update={"image_base64": scene_image}),
    })

    # Overlap guard, taken BEFORE the session lock and released without an await: the lock is held across
    # the provider call, so without this a second Start arriving while the first call still runs would
    # queue, find the manual cooldown already expired when it finally entered, and dispatch a second paid
    # call. Read and written synchronously, so two overlapping selections cannot both pass it.
    if session.select_inflight:
        raise ApiFailure(503, "provider_busy", 1)
    session.select_inflight = True
    try:
        async with session.lock:
            now = time.monotonic()
            if not session.active or session.expires <= now:
                raise ApiFailure(401, "session_expired")
            # One deliberate click is one manual analysis (manual cooldown and per-minute ceiling), inside the
            # session's shared 20-per-minute request ceiling, so no lane can spend beyond that budget.
            enforce_rate(session, analysis=True, mode="manual")

            started = time.monotonic()
            acquired = False
            try:
                await acquire_provider()
                acquired = True
                raw_output, usage = await store.provider.select_target(clean_request)
                try:
                    selection = TargetSelectionToolOutput.model_validate(raw_output)
                except ValidationError as exc:
                    raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA) from exc
                # Post-await fence: a session that ended while the provider was working must not receive an
                # answer it could act on.
                async with store.guard:
                    if (
                        store.sessions.get(token) is not session
                        or not session.active
                        or session.expires <= time.monotonic()
                    ):
                        raise ApiFailure(401, "session_expired")
                return TrackSelectResponse(
                    selection_id=str(uuid.uuid4()),
                    frame_id=payload.scene.frame_id,
                    status=selection.status,
                    target=selection.target,
                    rationale=selection.rationale,
                    provider=store.provider.provider_name,
                    model=store.provider.model,
                    usage=usage,
                )
            except ProviderError as exc:
                raise provider_failure(exc, mode=SELECT_MODE, started=started) from None
            finally:
                if acquired:
                    store.provider_slots.release()
    finally:
        session.select_inflight = False


# --------------------------------------------------------------------------------- anchored guide lane
#
# Seven routes (design docs/redesign-anchored.md §계약): plan (DeepSeek, once per Start), follow (the
# configured follower: a loopback llama.cpp model or DeepSeek), confirm (DeepSeek; the only source of
# ``visually_satisfied``), talk (DeepSeek, the user's own words), the two text-only plan-review routes,
# approve and revert (no provider, no frame: they only re-stamp the stored plan), and the offline plan-material
# import (``POST /api/plan/manual``: local extraction, no provider). Same gates as
# /api/track/select: origin, local-only, access-code deployment,
# consent, session binding, a validated JPEG, the 10 MiB body cap. The paid calls take a provider slot and
# the ``guide`` rate lane; a local follow takes neither (it is not a paid analysis) but is limited to one in
# flight per session. Every model answer is fenced twice: the session, task and plan it was asked about
# must still be current when it returns, or the answer is refused (401 / 409) instead of returned.


def guide_gates(request: Request) -> None:
    origin_allowed(request)
    require_access_deployment(request)


def guide_follower() -> Any:
    """The follower for this request, or ``503 service_unavailable`` naming the variable that is wrong."""
    try:
        follower = store.follower()
    except StageError as exc:
        raise ApiFailure(503, "service_unavailable", None, str(exc)) from None
    if not isinstance(follower, LOOPBACK_FOLLOWERS) and not guide_ready():
        raise ApiFailure(503, "service_unavailable")
    return follower


async def guide_fence(token: str | None, session: Session, *, task_revision: int,
                      plan: StoredPlan | None = None) -> None:
    """Post-await fence: the session is alive, the task unchanged, and (when given) the plan still current."""
    async with store.guard:
        if store.sessions.get(token or "") is not session or not session.active or session.expires <= time.monotonic():
            raise ApiFailure(401, "session_expired")
    if session.task_revision != task_revision:
        raise ApiFailure(409, "task_changed")
    if plan is not None and not session.guide.is_current(plan, task_revision):
        raise stale_plan(session.guide)


def anchors_echo(payload: GuideFollowRequest | GuideConfirmRequest | GuideTalkRequest) -> list[AnchorEcho]:
    return [
        AnchorEcho(anchor_id=anchor.anchor_id, track_id=anchor.track_id, generation=anchor.generation)
        for anchor in payload.anchors
    ]


def elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def log_talk_action(action: str, plan_changed: bool, started: float) -> None:
    """One enum-only line per answered talk: the action token, whether the stored plan changed, a duration.

    The user's words and the model's reply never reach the log (same rule as ``log_provider_failure``).
    """
    logger.info("guide talk answered action=%s plan_changed=%s elapsed_ms=%d", action, plan_changed,
                elapsed_ms(started))


#: ``Accept`` media type that turns /api/guide/plan into a server-sent event stream (see ``guide_plan``).
EVENT_STREAM = "text/event-stream"
#: Seconds between ``: ping`` comments while a streamed plan is still waiting for its terminal event.
STREAM_HEARTBEAT_S = 5.0
#: Streamed plan calls run as tasks the response only listens to, so a client that disconnects never
#: cancels them half way (the JSON path is not cancelled by a disconnect either). Referenced until done.
_plan_tasks: set[asyncio.Task[None]] = set()


def wants_event_stream(request: Request) -> bool:
    """``Accept`` names ``text/event-stream`` (any position, parameters ignored)."""
    return any(
        part.split(";", 1)[0].strip().lower() == EVENT_STREAM
        for part in request.headers.get("accept", "").split(",")
    )


async def guide_plan_answer(
    payload: GuidePlanRequest,
    *,
    token: str | None,
    session: Session,
    goal_clean: str,
    context_clean: str | None,
    scene_image: str,
    on_hint: Callable[[str, str], None] | None = None,
) -> GuidePlanResponse:
    """The plan call itself, shared by the JSON and the streamed route: lock, lane, slot, call, checks, store.

    The caller holds ``plan_inflight``. With ``on_hint`` the provider call is streamed and the hint callback
    receives ``("target", noun)`` and ``("first_say", sentence)`` as soon as each string is closed; the
    answer is still validated, fenced and installed exactly as in the non-streamed call.
    """
    guide = session.guide
    async with session.lock:
        if not session.active or session.expires <= time.monotonic():
            raise ApiFailure(401, "session_expired")
        enforce_rate(session, analysis=True, mode=GUIDE_MODE)
        materials = list(payload.materials) if payload.materials else None
        # A different goal, context, material list, PLAN MODEL, CORE MODE or assisted flag is a new task: the
        # old plan is cleared and ``task_revision`` is bumped, so a changed source — or a switch between
        # ``deepseek:high`` and ``astra:high``, or between the classic, sequential and graph cores — can never
        # inherit the old plan's authority (or its evidence), nor its history.
        if (
            goal_clean != session.task_goal
            or context_clean != session.task_context
            or materials != session.task_materials
            or payload.plan_model != session.task_plan_model
            or payload.core_mode != session.task_core_mode
            or payload.assisted != session.task_assisted
        ):
            session.guide.clear()
            session.task_goal = goal_clean
            session.task_context = context_clean
            session.task_materials = materials
            session.task_plan_model = payload.plan_model
            session.task_core_mode = payload.core_mode
            session.task_assisted = payload.assisted
            session.task_answers = []
            session.task_sources = []
            session.task_revision += 1
        # Clarification changes invalidate execution, but keep public research for the same goal/materials.
        if payload.answers != session.task_answers:
            session.guide.clear()
            session.task_answers = list(payload.answers)
            session.task_revision += 1
        payload._research_sources = list(session.task_sources)
        request_revision = session.task_revision
        # Select the task's own upper planning adapter. Missing credentials never route to the other lane.
        provider = store.upper_provider(payload.plan_model)

        started = time.monotonic()
        acquired = False
        try:
            long_side = 1600 if payload.assisted else 1024
            prepared = await asyncio.to_thread(prepare_frame_b64, scene_image, [], max_long_side=long_side)
            prepared_references = await asyncio.gather(*(
                asyncio.to_thread(prepare_frame_b64, image.image_base64, [], max_long_side=long_side)
                for image in payload.reference_images
            ))
            payload = payload.model_copy(update={"reference_images": [
                image.model_copy(update={"image_base64": prepared_b64})
                for image, prepared_b64 in zip(payload.reference_images, prepared_references, strict=True)
            ]})
            await acquire_provider()
            acquired = True
            if on_hint is None:
                raw_output, usage = await provider.guide_plan(payload, prepared)
            else:
                raw_output, usage = await streamed_plan_call(payload, prepared, on_hint, provider)
            # §15: a step may cite an id from the request's material list. An ungrounded citation (not an
            # object, a blank/non-string or unsupplied id, a version that does not match the material, a quote
            # that is not an excerpt of the material's text or is over its bound, an over-long locator) is
            # DROPPED from the raw answer before validation — logged, never invented — so a bad citation
            # installs the plan without it instead of 502-ing it.
            if payload.plan_model != "astra:high" and isinstance(raw_output.get("research_sources"), list):
                # This adapter has no hosted search. It may cite only the session's verified prior sources.
                known_urls = {source.url for source in payload._research_sources}
                raw_output = {**raw_output, "research_sources": [
                    source for source in raw_output["research_sources"]
                    if isinstance(source, dict) and source.get("url") in known_urls
                ]}
            raw_output, dropped_evidence = drop_ungrounded_evidence(raw_output, materials)
            if dropped_evidence:
                logger.info("guide_plan_evidence_dropped count=%d ids=%s", len(dropped_evidence),
                            ",".join(sorted(set(dropped_evidence))))
            try:
                output = GuidePlanToolOutput.model_validate(raw_output)
            except ValidationError as exc:
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA) from exc
            await guide_fence(token, session, task_revision=request_revision)
            sources = {source.url: source for source in output.research_sources}
            for source in session.task_sources:
                sources.setdefault(source.url, source)
            session.task_sources = list(sources.values())[:6]
            output.research_sources = list(session.task_sources)
            plan = guide.install(
                steps=list(output.steps),
                goal_when=output.goal_when,
                user_goal=goal_clean,
                context=context_clean,
                task_revision=request_revision,
                target=output.selection.target,
                materials=tuple(materials or ()),
                plan_model=payload.plan_model,
                core_mode=payload.core_mode,
                assisted=payload.assisted,
                answers=tuple(payload.answers),
                research_sources=tuple(output.research_sources),
            )
            return GuidePlanResponse(
                **{name: getattr(output, name) for name in GuidePlanToolOutput.model_fields},
                plan_id=plan.plan_id,
                plan_revision=plan.plan_revision,
                core_mode=plan.core_mode,
                frame_id=payload.scene.frame_id,
                provider=provider.provider_name,
                model=provider.model,
                usage=usage,
            )
        except ProviderError as exc:
            raise provider_failure(exc, mode=GUIDE_PLAN_MODE, started=started) from None
        finally:
            if acquired:
                store.provider_slots.release()


async def streamed_plan_call(payload: GuidePlanRequest, prepared: str,
                             on_hint: Callable[[str, str], None],
                             provider: Any) -> tuple[dict[str, Any], Any]:
    """Drain the provider's plan stream, reporting at most one target hint and one first-say hint, in order.

    A target that closes only after the first say has been reported is not reported (the SSE contract
    orders them). ``provider`` is the task's selected upper provider (never a global): the returned arguments
    are its complete, strictly parsed answer.
    """
    scanner = PlanHintScanner()
    sent_target = sent_say = False
    final: GuideStreamFinal | None = None
    async with aclosing(provider.guide_plan_stream(payload, prepared)) as items:
        async for item in items:
            if isinstance(item, GuideStreamFinal):
                final = item
                continue
            scanner.feed(item)
            if not sent_target and not sent_say and scanner.target is not None:
                sent_target = True
                on_hint("target", scanner.target)
            if not sent_say and scanner.first_say is not None:
                sent_say = True
                on_hint("first_say", scanner.first_say)
    if final is None:
        raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
    return final.arguments, final.usage


async def failure_event_body(exc: ApiFailure) -> dict[str, Any]:
    """The exact JSON body the non-streamed route answers for ``exc``, plus its HTTP ``status``."""
    response = await api_error(None, exc)  # type: ignore[arg-type]
    body = json.loads(bytes(response.body))
    body["status"] = exc.status
    return body


def sse_event(kind: str, data: dict[str, Any]) -> bytes:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {kind}\ndata: {text}\n\n".encode()


async def run_streamed_plan(events: asyncio.Queue[tuple[str, dict[str, Any]]], guide: GuideSessionState,
                            **answer_args: Any) -> None:
    """One streamed plan: hints, then exactly one terminal event; ``plan_inflight`` released last."""
    terminal = False
    try:
        try:
            response = await guide_plan_answer(
                **answer_args, on_hint=lambda name, value: events.put_nowait(("partial", {name: value}))
            )
            events.put_nowait(("final", response.model_dump(mode="json")))
            terminal = True
        except ApiFailure as exc:
            events.put_nowait(("error", await failure_event_body(exc)))
            terminal = True
        except Exception:
            logger.exception("guide_plan stream failed")
    finally:
        if not terminal:
            events.put_nowait(("error", {"error": "internal_error", "status": 500}))
        guide.plan_inflight = False


async def plan_event_stream(events: asyncio.Queue[tuple[str, dict[str, Any]]]) -> AsyncIterator[bytes]:
    while True:
        try:
            kind, data = await asyncio.wait_for(events.get(), STREAM_HEARTBEAT_S)
        except TimeoutError:
            yield b": ping\n\n"
            continue
        yield sse_event(kind, data)
        if kind != "partial":
            return


@app.post("/api/guide/plan", response_model=GuidePlanResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": GuidePlanRequest.model_json_schema()}}},
})
async def guide_plan(request: Request) -> Any:
    """Start a guide: one upper-provider call names the object to track and writes a 1-16 step plan.

    The upper adapter is the request's ``plan_model``: ``deepseek:high`` (the default) or ``astra:high``;
    readiness is checked for exactly that choice. The accepted plan (text only) replaces the session's
    previous one under a new ``plan_id`` and the next ``plan_revision``. A different goal, context, material
    list, plan model or core mode is a new task: the old plan is cleared and ``task_revision`` is bumped.
    ``materials``
    (optional, at most four) are reference documents for the plan and every later replan prompt; a step cites
    one with ``evidence``, and a citation the request cannot ground is dropped rather than returned.

    With ``Accept: text/event-stream`` the same call is streamed from the provider and the answer is a
    ``200 text/event-stream``: at most one ``event: partial`` ``{"target"}``, at most one ``event: partial``
    ``{"first_say"}``, then exactly one ``event: final`` (the JSON body below) or ``event: error`` (the
    JSON error body plus ``"status"``); ``: ping`` comments every 5 s. Every check up to the in-flight guard
    runs before the stream starts and answers as plain JSON with its status; everything after it (lock,
    rate lane, provider, validation, fences, store) is the same code as the JSON path.
    """
    guide_gates(request)
    payload = await body_model(request, GuidePlanRequest, max_bytes=MAX_GUIDANCE_BODY)
    # Model-aware, and after the body so ``plan_model`` is known: a start on a profile without its credential
    # is refused here — before the session, the frame or any provider dispatch — and is never silently routed
    # to the other profile. The other lane's own readiness is unchanged.
    if payload.research and payload.plan_model != "astra:high":
        # Research must not silently route DeepSeek credentials or data to a different issuer.
        raise ApiFailure(503, "service_unavailable")
    if not plan_model_ready(payload.plan_model):
        raise ApiFailure(503, "service_unavailable")
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not payload.consent_ai:
        raise ApiFailure(400, "ai_consent_required")
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")
    goal_clean = payload.user_goal.strip()
    if not goal_clean:
        raise ApiFailure(422, "invalid_request")
    context_clean = (payload.context or "").strip() or None
    scene_image = validate_jpeg(payload.scene.image_base64)
    for image in payload.reference_images:
        validate_jpeg(image.image_base64)
    answer_args: dict[str, Any] = {
        "payload": payload, "token": token, "session": session,
        "goal_clean": goal_clean, "context_clean": context_clean, "scene_image": scene_image,
    }
    stream = wants_event_stream(request)

    guide = session.guide
    # Same overlap guard as track_select: taken before the session lock, without an await in between.
    if guide.plan_inflight:
        raise ApiFailure(503, "provider_busy", 1)
    guide.plan_inflight = True
    if not stream:
        try:
            return await guide_plan_answer(**answer_args)
        finally:
            guide.plan_inflight = False
    events: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    try:
        task = asyncio.create_task(run_streamed_plan(events, guide, **answer_args))
    except BaseException:
        guide.plan_inflight = False
        raise
    _plan_tasks.add(task)
    task.add_done_callback(_plan_tasks.discard)
    return StreamingResponse(
        plan_event_stream(events),
        media_type=EVENT_STREAM,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def guide_read_origin(request: Request) -> None:
    """The origin gate for a guide READ (no body, no model call).

    Browsers do not send ``Origin`` on a same-origin ``GET``, so a missing ``Origin`` is accepted only when
    the browser itself says the request is same-origin (``Sec-Fetch-Site: same-origin``). A present
    ``Origin`` is checked exactly as on the POST routes. The session cookie is ``SameSite=Strict`` as well.
    """
    if request.headers.get("origin") is None and request.headers.get("sec-fetch-site", "").lower() == "same-origin":
        return
    origin_allowed(request)


@app.get("/api/guide/plan/current", response_model=GuidePlanCurrentResponse)
async def guide_plan_current(request: Request, session_id: str = "") -> GuidePlanCurrentResponse:
    """The session's current plan (text only), or ``404 no_plan``. Never a provider call.

    The recovery read for a ``409 stale_plan``: a client that missed a plan (a streamed plan it disconnected
    from) or a replan (a confirm answer it dropped) fetches the plan it must name from now on. Same gates as
    the other guide routes — origin (see ``guide_read_origin``), local-only / access-code deployment, the
    session cookie and ``session_id`` (query) bound to it — but no consent check: nothing is sent to a model.
    It reads the session without taking its lock or touching the rate lane, the provider slot or the
    in-flight flags, so it never waits behind a running plan; a plan still being made is not yet current.
    """
    guide_read_origin(request)
    require_access_deployment(request)
    if not (len(session_id) == 36 and session_id.isascii()):  # the contract's UUIDString; compare_digest needs ASCII
        raise ApiFailure(422, "invalid_request")
    session = await store.session(request.cookies.get(COOKIE))
    if not hmac.compare_digest(session.id, session_id):
        raise ApiFailure(401, "session_mismatch")
    plan = session.guide.plan
    if plan is None or plan.task_revision != session.task_revision:
        raise ApiFailure(404, "no_plan")
    return GuidePlanCurrentResponse(
        plan_id=plan.plan_id,
        plan_revision=plan.plan_revision,
        core_mode=plan.core_mode,
        task_revision=plan.task_revision,
        steps=list(plan.steps),
        goal_when=plan.goal_when,
        user_goal=plan.user_goal,
    )


@app.post("/api/plan/manual", response_model=MaterialInput, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": ManualImportRequest.model_json_schema()}}},
})
async def plan_manual(request: Request) -> MaterialInput:
    """Import one reference document into a ``MaterialInput``: local extraction only, no model call.

    The client uploads a file (base64 in the JSON body) and gets the ``MaterialInput`` to include in a later
    ``/api/guide/plan`` ``materials`` list. Extraction is offline and bounded (``plan_manual``), runs off the
    event loop, and fails closed with a closed-set code — a file that does not fit is refused, never truncated.
    Same origin, deployment-access and cookie-session gates as the other guide routes, bound to ``session_id``
    (a different session is ``401 session_mismatch``); there is no consent check and no provider: nothing here
    reaches a model. Nothing is stored: the session keeps the material only from the plan start.
    """
    origin_allowed(request)
    require_access_deployment(request)
    payload = await body_model(request, ManualImportRequest, max_bytes=MANUAL_REQUEST_MAX_BYTES)
    session = await store.session(request.cookies.get(COOKIE))
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")
    return await asyncio.to_thread(extract_manual, payload.filename, payload.content_base64)


@app.post("/api/guide/plan/approve", response_model=GuidePlanCurrentResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": GuidePlanApproveRequest.model_json_schema()}}},
})
async def guide_plan_approve(request: Request) -> GuidePlanCurrentResponse:
    """Adopt the plan the CLIENT reviewed and edited as the session's authoritative plan. No model call.

    ``plan_id``/``plan_revision`` must name the session's current plan, else ``409 stale_plan`` (the usual body):
    the client reviewed a plan that is already gone. On success the adopted plan keeps the ``plan_id``, takes the
    NEXT ``plan_revision`` from the session counter and keeps the session's ``task_revision``, so every fence
    that follows names the adopted pair and the plan the model wrote is ``409`` from then on. A broken §15 graph
    (unknown/self/cyclic ``requires``, duplicate ids, duplicate targets) is ``422 invalid_request`` before any
    state changes, and so is an ``evidence`` the plan's own materials do not ground (an unknown id, a version
    that does not match, a quote that is not an excerpt): the client may keep or drop a validated citation, but
    it may not inject one. Gates are the other guide routes' origin/access/session checks, but there is no
    consent check and no provider: nothing here reaches a model, and no frame is read.
    """
    guide_gates(request)
    payload = await body_model(request, GuidePlanApproveRequest, max_bytes=MAX_STANDARD_BODY)
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")

    guide = session.guide
    plan = guide.require(plan_id=payload.plan_id, plan_revision=payload.plan_revision,
                         task_revision=session.task_revision)
    try:
        check_grounded_evidence(list(payload.steps), list(plan.materials) or None)
    except ValueError:
        raise ApiFailure(422, "invalid_request") from None
    # An omitted field keeps the current plan's own value; a present but blank goal is refused, as on /plan.
    goal_clean = plan.user_goal if payload.user_goal is None else payload.user_goal.strip()
    if not goal_clean:
        raise ApiFailure(422, "invalid_request")
    context_clean = plan.context if payload.context is None else (payload.context.strip() or None)
    # One synchronous call, no await while mutating: a second approve naming the same pair is stale by then.
    adopted = guide.adopt(plan, steps=list(payload.steps), goal_when=payload.goal_when,
                          user_goal=goal_clean, context=context_clean)
    return GuidePlanCurrentResponse(
        plan_id=adopted.plan_id,
        plan_revision=adopted.plan_revision,
        core_mode=adopted.core_mode,
        task_revision=adopted.task_revision,
        steps=list(adopted.steps),
        goal_when=adopted.goal_when,
        user_goal=adopted.user_goal,
    )


@app.post("/api/guide/plan/revert", response_model=GuidePlanCurrentResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": GuidePlanRevertRequest.model_json_schema()}}},
})
async def guide_plan_revert(request: Request) -> GuidePlanCurrentResponse:
    """Undo the last plan change: install the session's most recent previous plan again. No model call.

    ``plan_id``/``plan_revision`` must name the session's current plan, else ``409 stale_plan``. With an empty
    history there is nothing to restore: ``404 no_plan``. On success the restored plan keeps the ``plan_id`` it
    had and takes the NEXT ``plan_revision``, so it is never the pair the client just named (a fence for the
    pre-revert pair is ``409`` from now on) — the revision only ever moves forward.
    """
    guide_gates(request)
    payload = await body_model(request, GuidePlanRevertRequest, max_bytes=MAX_STANDARD_BODY)
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")

    guide = session.guide
    plan = guide.require(plan_id=payload.plan_id, plan_revision=payload.plan_revision,
                         task_revision=session.task_revision)
    if not guide.can_revert:
        raise ApiFailure(404, "no_plan")
    restored = guide.revert(plan)
    return GuidePlanCurrentResponse(
        plan_id=restored.plan_id,
        plan_revision=restored.plan_revision,
        core_mode=restored.core_mode,
        task_revision=restored.task_revision,
        steps=list(restored.steps),
        goal_when=restored.goal_when,
        user_goal=restored.user_goal,
    )


@app.post("/api/guide/follow", response_model=GuideFollowResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": GuideFollowRequest.model_json_schema()}}},
})
async def guide_follow(request: Request) -> GuideFollowResponse:
    """One follow judgement against the current plan. Never a goal status, never an instruction.

    ``step_checks`` must name exactly the plan's checklist: current-to-last for classic, current-only for
    sequential, and the WHOLE plan for graph (earlier/completed steps included, whatever the request's focus).
    Any other id list is ``502 invalid_provider_output``
    (``schema``). A verdict about an anchor that was not sent is ``502 invalid_provider_output``
    (``provenance``). ``needs_reselect`` is set when the follower says an anchor box does NOT contain the target.
    """
    guide_gates(request)
    follower = guide_follower()
    payload = await body_model(request, GuideFollowRequest, max_bytes=MAX_GUIDANCE_BODY)
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not payload.consent_ai:
        raise ApiFailure(400, "ai_consent_required")
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")
    scene_image = validate_jpeg(payload.scene.image_base64)

    guide = session.guide
    plan = guide.require(plan_id=payload.plan_id, plan_revision=payload.plan_revision,
                         task_revision=session.task_revision)
    if payload.current_step not in plan.step_ids:
        raise ApiFailure(422, "unknown_step")
    if guide.follow_inflight:
        raise ApiFailure(503, "follow_busy", 1)
    guide.follow_inflight = True
    try:
        paid = not isinstance(follower, LOOPBACK_FOLLOWERS)
        if paid:
            enforce_rate(session, analysis=True, mode=GUIDE_MODE)
        request_revision = session.task_revision
        started = time.monotonic()
        acquired = False
        try:
            prepared = await asyncio.to_thread(prepare_frame_b64, scene_image, payload.anchors)
            if paid:
                await acquire_provider()
                acquired = True
            raw_output, usage = await follower.guide_follow(payload, plan, prepared)
            try:
                output = GuideFollowToolOutput.model_validate(raw_output)
            except ValidationError as exc:
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA) from exc
            if [check.step_id for check in output.step_checks] != plan.checklist_ids(payload.current_step):
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA)
            sent_anchor_ids = {anchor.anchor_id for anchor in payload.anchors}
            if any(verdict.anchor_id not in sent_anchor_ids for verdict in output.anchor_verdicts):
                raise ProviderInvalidOutput(InvalidOutputStage.PROVENANCE)
            await guide_fence(token, session, task_revision=request_revision, plan=plan)
            return GuideFollowResponse(
                **{name: getattr(output, name) for name in GuideFollowToolOutput.model_fields},
                intent_id=str(uuid.uuid4()),
                frame_id=payload.scene.frame_id,
                intent_seq=payload.intent_seq,
                trigger=payload.trigger,
                task_revision=request_revision,
                plan_revision=plan.plan_revision,
                fence_echo=payload.fence,
                anchors_echo=anchors_echo(payload),
                provider=follower.provider_name,
                model=follower.model,
                usage=usage,
                latency_ms=elapsed_ms(started),
                needs_reselect=any(verdict.matches == "no" for verdict in output.anchor_verdicts),
            )
        except ProviderError as exc:
            raise provider_failure(exc, mode=GUIDE_FOLLOW_MODE, started=started) from None
        finally:
            if acquired:
                store.provider_slots.release()
    finally:
        guide.follow_inflight = False


@app.post("/api/guide/confirm", response_model=GuideConfirmResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": GuideConfirmRequest.model_json_schema()}}},
})
async def guide_confirm(request: Request) -> GuideConfirmResponse:
    """The completion judge (the plan's own upper provider): the only route that can answer ``visually_satisfied``.

    Only the ``replan`` and ``unsure_twice`` triggers offer the model a ``replan`` field; on ``goal_check``
    and ``step_done`` a non-null ``replan`` is ``502 invalid_provider_output`` (``schema``). A ``replan`` is applied to the stored plan only if that plan is still current when the answer returns;
    the response then carries the bumped ``plan_revision`` that follow/confirm must name from now on.

    ``step_done`` names the step it checks (``current_step``, required by the contract); a step outside the
    current plan is ``422 unknown_step`` before any paid call. Under ``graph`` that may be any step of the
    plan — an earlier completed one or a future eligible one — and the response echoes exactly the requested
    id, whatever the client's suggested focus. Its tool REQUIRES ``step_check`` and no other
    trigger's tool has it, so a missing ``step_check`` on ``step_done`` or a non-null one elsewhere is
    ``502 invalid_provider_output`` (``schema``). The response echoes ``step_id = current_step`` on
    ``step_done`` only. ``step_check`` is a step judgement; it never touches ``goal_status``.

    ``follow_checks`` (``replan``/``unsure_twice`` only, by contract) reaches the prompt as the follower's
    recent checklist; an id outside the current plan is ``422 unknown_step`` before any paid call.

    ``replan``/``unsure_twice`` requires ``step_checks`` against the request's plan: current-to-last for
    classic (all steps when current_step is omitted), current-only for sequential. Sequential requires
    current_step before dispatch; this server cannot infer the client's active step. ``graph`` always expects
    the WHOLE plan, so its expected list is the same whether or not the request names one.
    A missing list, wrong id list, or checklist on another trigger is ``502 invalid_provider_output``.
    """
    guide_gates(request)
    payload = await body_model(request, GuideConfirmRequest, max_bytes=MAX_GUIDANCE_BODY)
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not payload.consent_ai:
        raise ApiFailure(400, "ai_consent_required")
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")
    scene_image = validate_jpeg(payload.scene.image_base64)
    before_image = validate_jpeg(payload.before_scene.image_base64) if payload.before_scene is not None else None

    guide = session.guide
    plan = guide.require(plan_id=payload.plan_id, plan_revision=payload.plan_revision,
                         task_revision=session.task_revision)
    # The client cannot override the stored plan's own choice here. A plan made for one profile is judged on
    # that profile; a missing credential refuses this call with 503 rather than sending its data to the other.
    provider = store.upper_provider(plan.plan_model)
    if not plan_model_ready(plan.plan_model):
        raise ApiFailure(503, "service_unavailable")
    if payload.user_goal.strip() != plan.user_goal:
        # The plan was made for another goal: judging this goal against it would mix two tasks.
        raise stale_plan(guide)
    if plan.core_mode == CORE_MODE_SEQUENTIAL and confirm_allows_replan(payload.trigger) and payload.current_step is None:
        raise ApiFailure(422, "invalid_request")
    if payload.current_step is not None and payload.current_step not in plan.step_ids:
        raise ApiFailure(422, "unknown_step")
    if payload.follow_checks is not None and any(c.step_id not in plan.step_ids for c in payload.follow_checks):
        raise ApiFailure(422, "unknown_step")
    checks_step = payload.trigger == "step_done"  # intent.STEP_CHECK_TRIGGER
    infers_done = payload.trigger == "target_left"  # intent.TARGET_LEFT_TRIGGER
    if guide.confirm_inflight:
        raise ApiFailure(503, "provider_busy", 1)
    guide.confirm_inflight = True
    try:
        enforce_rate(session, analysis=True, mode=GUIDE_MODE)
        request_revision = session.task_revision
        started = time.monotonic()
        acquired = False
        try:
            prepared = await asyncio.to_thread(prepare_frame_b64, scene_image, payload.anchors)
            before_prepared = (await asyncio.to_thread(prepare_frame_b64, before_image, [])
                               if before_image is not None else None)
            await acquire_provider()
            acquired = True
            raw_output, usage = await provider.guide_confirm(payload, plan, prepared, before_prepared)
            # A replan prompt renders the plan's own retained materials, so a citation is kept only when it is
            # grounded in one of them; anything else is dropped before validation (logged, never invented).
            raw_output, dropped_evidence = drop_ungrounded_evidence(raw_output, list(plan.materials) or None,
                                                                    path=("replan", "steps"))
            if dropped_evidence:
                logger.info("guide_replan_evidence_dropped route=confirm count=%d", len(dropped_evidence))
            try:
                output = GuideConfirmToolOutput.model_validate(raw_output)
            except ValidationError as exc:
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA) from exc
            if output.replan is not None and not confirm_allows_replan(payload.trigger):
                # The tool offered on this trigger has no replan field: a plan nobody asked for is refused.
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA)
            if (output.step_check is None) == checks_step:
                # step_done's tool requires step_check; every other trigger's tool has no such field.
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA)
            if (output.inferred_done is None) == infers_done:
                # target_left's tool requires inferred_done; every other trigger's tool has no such field.
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA)
            if output.step_checks is None:
                if confirm_allows_replan(payload.trigger):
                    # replan/unsure_twice's tool requires step_checks.
                    raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA)
            elif [check.step_id for check in output.step_checks] != confirm_checklist_ids(payload, plan):
                # Exactly current_step..last step in plan order (same rule as follow); every other trigger's
                # tool has no such field, so its expected list is empty and any value is refused.
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA)
            await guide_fence(token, session, task_revision=request_revision, plan=plan)
            current = plan
            if output.replan is not None:
                current = guide.replan(plan, list(output.replan.steps))
            return GuideConfirmResponse(
                **{name: getattr(output, name) for name in GuideConfirmToolOutput.model_fields},
                step_id=payload.current_step if checks_step else None,
                intent_id=str(uuid.uuid4()),
                frame_id=payload.scene.frame_id,
                intent_seq=payload.intent_seq,
                trigger=payload.trigger,
                task_revision=request_revision,
                plan_revision=current.plan_revision,
                fence_echo=payload.fence,
                anchors_echo=anchors_echo(payload),
                provider=provider.provider_name,
                model=provider.model,
                usage=usage,
                latency_ms=elapsed_ms(started),
            )
        except ProviderError as exc:
            raise provider_failure(exc, mode=GUIDE_CONFIRM_MODE, started=started) from None
        finally:
            if acquired:
                store.provider_slots.release()
    finally:
        guide.confirm_inflight = False


@app.post("/api/guide/talk", response_model=GuideTalkResponse, openapi_extra={
    "requestBody": {"required": True, "content": {"application/json": {"schema": GuideTalkRequest.model_json_schema()}}},
})
async def guide_talk(request: Request) -> GuideTalkResponse:
    """The user's words during a guide (the plan's own upper provider): a short Korean reply and at most one action.

    Gates, rate lane, provider slot and the post-await fence are confirm's. Only ``step_say`` (``guide.restate``
    on ``current_step``) and ``replan`` (``guide.replan``) change the stored plan, and only if that plan is still
    current when the answer returns; the response then carries the bumped ``plan_revision``. ``step_mark``,
    ``go_to`` and ``target`` are the client's to apply. A ``go_to`` that is not a step BEFORE ``current_step``
    is ``502 invalid_provider_output`` (``schema``), like two actions in one answer.
    """
    guide_gates(request)
    payload = await body_model(request, GuideTalkRequest, max_bytes=MAX_GUIDANCE_BODY)
    token = request.cookies.get(COOKIE)
    session = await store.session(token)
    if not payload.consent_ai:
        raise ApiFailure(400, "ai_consent_required")
    if not hmac.compare_digest(session.id, payload.session_id):
        raise ApiFailure(401, "session_mismatch")
    scene_image = validate_jpeg(payload.scene.image_base64)

    guide = session.guide
    plan = guide.require(plan_id=payload.plan_id, plan_revision=payload.plan_revision,
                         task_revision=session.task_revision)
    # Route from the STORED plan's own plan_model, exactly as confirm does: the client cannot switch profiles
    # by calling talk, and a plan never falls back to the other lane when its own credential is unavailable.
    provider = store.upper_provider(plan.plan_model)
    if not plan_model_ready(plan.plan_model):
        raise ApiFailure(503, "service_unavailable")
    if payload.current_step not in plan.step_ids:
        raise ApiFailure(422, "unknown_step")
    if payload.follow_checks is not None and any(c.step_id not in plan.step_ids for c in payload.follow_checks):
        raise ApiFailure(422, "unknown_step")
    if guide.talk_inflight:
        raise ApiFailure(503, "provider_busy", 1)
    guide.talk_inflight = True
    try:
        enforce_rate(session, analysis=True, mode=GUIDE_MODE)
        request_revision = session.task_revision
        started = time.monotonic()
        acquired = False
        try:
            prepared = await asyncio.to_thread(prepare_frame_b64, scene_image, payload.anchors)
            await acquire_provider()
            acquired = True
            raw_output, usage = await provider.guide_talk(payload, plan, prepared)
            # Same rule as confirm's replan: a citation is kept only when the plan's own materials ground it.
            raw_output, dropped_evidence = drop_ungrounded_evidence(raw_output, list(plan.materials) or None,
                                                                    path=("replan", "steps"))
            if dropped_evidence:
                logger.info("guide_replan_evidence_dropped route=talk count=%d", len(dropped_evidence))
            try:
                output = GuideTalkToolOutput.model_validate(raw_output)
            except ValidationError as exc:
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA) from exc
            if output.go_to is not None and not plan.is_before(output.go_to, payload.current_step):
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA)
            if output.replan is not None and not payload.replan_allowed:
                # The tool offered without replan has no such field: a replan nobody may ask for is refused.
                raise ProviderInvalidOutput(InvalidOutputStage.SCHEMA)
            await guide_fence(token, session, task_revision=request_revision, plan=plan)
            current = plan
            if output.replan is not None:
                current = guide.replan(plan, list(output.replan.steps))
            elif output.step_say is not None:
                current = guide.restate(plan, payload.current_step, output.step_say)
            log_talk_action("+".join(output.actions) or "none", current is not plan, started)
            if output.dropped_actions:
                logger.info("talk_action_collapsed kept=%s dropped=%s", ",".join(output.actions) or "none",
                            ",".join(output.dropped_actions))
            return GuideTalkResponse(
                **{name: getattr(output, name) for name in GuideTalkToolOutput.model_fields},
                intent_id=str(uuid.uuid4()),
                frame_id=payload.scene.frame_id,
                intent_seq=payload.intent_seq,
                task_revision=request_revision,
                plan_revision=current.plan_revision,
                fence_echo=payload.fence,
                anchors_echo=anchors_echo(payload),
                provider=provider.provider_name,
                model=provider.model,
                usage=usage,
                latency_ms=elapsed_ms(started),
            )
        except ProviderError as exc:
            raise provider_failure(exc, mode=GUIDE_TALK_MODE, started=started) from None
        finally:
            if acquired:
                store.provider_slots.release()
    finally:
        guide.talk_inflight = False


async def tts_admission(request: Request) -> TtsRequest:
    """``POST /api/tts`` (``tts.py``): the guide routes' origin, access and session gates, then the bounded body."""
    origin_allowed(request)
    enforce_access_mode(request)
    await store.session(request.cookies.get(COOKIE))
    return await body_model(request, TtsRequest, max_bytes=MAX_STANDARD_BODY)


app.include_router(build_tts_router(tts_admission, lambda: store.client))


def bugreport_admission(request: Request) -> None:
    """``/api/debug/bugreport*`` (``bugreport.py``): origin and access gates only."""
    origin_allowed(request)
    enforce_access_mode(request)


app.include_router(build_bugreport_router(bugreport_admission))


def documented_openapi() -> dict[str, Any]:
    if app.openapi_schema is None:
        schema = get_openapi(title="synoptics — visual guidance API", version="0.2.0", routes=app.routes)
        def normalize(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in list(node.items()):
                    if key == "$ref" and isinstance(value, str) and value.startswith("#/$defs/"):
                        node[key] = value.replace("#/$defs/", "#/components/schemas/")
                    else:
                        normalize(value)
            elif isinstance(node, list):
                for item in node:
                    normalize(item)
        normalize(schema)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = documented_openapi

DIST = Path(__file__).resolve().parents[2] / "web/dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")

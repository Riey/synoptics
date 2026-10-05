"""Optional generic loopback voice proxy; no speech model or voice service is bundled.

The browser reads ``speechPolicy.spokenLines`` aloud. With ``AISW_TTS_URL`` set, it asks ``POST /api/tts`` for a
WAV in the operator's fine-tuned voice instead of using its own ``speechSynthesis``; with it unset the lane is
off and the browser voice is the only one. ``GET /api/tts/config`` tells the page which it is.

The endpoint follows the local-stage rule (``loopback.py``): the TTS service runs next to the app and the text
never leaves the machine. The service synthesises one line at a time and caches repeated lines itself, so this
proxy only bounds the text, the wait and the number of requests it lets queue there.

The submission leaves AISW_TTS_URL unset and uses the browser/device voice.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable

import httpx
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.errors import ApiFailure, StageConfigError
from backend.app.loopback import loopback_endpoint

logger = logging.getLogger(__name__)

ENV_TTS_URL = "AISW_TTS_URL"
#: Longer than any line the guide speaks (a step, a talk reply's spoken sentence plus the screen pointer).
MAX_TTS_CHARS = 300
#: One line on the GPU takes well under a second; this also covers waiting behind another line in the service.
TTS_TIMEOUT_S = 20.0
HEALTH_TIMEOUT_S = 2.0
#: Requests this process lets wait on the single-file service at once; more are refused (429), not queued.
MAX_INFLIGHT = 4
#: Far above a 300-character line (about a minute of 24 kHz 16-bit mono, ~3 MB).
MAX_AUDIO_BYTES = 8 * 1024 * 1024


class TtsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text is blank")
        return value


class TtsConfigResponse(BaseModel):
    """``enabled``: ``AISW_TTS_URL`` is set to a usable loopback URL. ``ready``: the service answered its health."""

    enabled: bool
    ready: bool


def tts_endpoint() -> str | None:
    """The configured service URL (no trailing slash), ``None`` when the lane is off; a non-loopback value fails."""
    configured = os.getenv(ENV_TTS_URL, "").strip()
    if not configured:
        return None
    return loopback_endpoint(configured, ENV_TTS_URL).rstrip("/")


def build_tts_router(
    admit: Callable[[Request], Awaitable[TtsRequest]],
    http: Callable[[], httpx.AsyncClient],
) -> APIRouter:
    """The two TTS routes. ``admit`` runs the app's origin/access/session gates and returns the bounded body;
    ``http`` is the app's shared client (kept as a callable so this module never imports ``main``)."""
    router = APIRouter()
    inflight = 0

    @router.get("/api/tts/config", response_model=TtsConfigResponse)
    async def tts_config() -> TtsConfigResponse:
        try:
            base = tts_endpoint()
        except StageConfigError:
            return TtsConfigResponse(enabled=False, ready=False)
        if base is None:
            return TtsConfigResponse(enabled=False, ready=False)
        try:
            response = await http().get(f"{base}/health", timeout=HEALTH_TIMEOUT_S)
            ready = response.status_code == 200 and response.json().get("ready") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            ready = False
        return TtsConfigResponse(enabled=True, ready=ready)

    @router.post("/api/tts", response_class=Response, responses={200: {"content": {"audio/wav": {}}}}, openapi_extra={
        "requestBody": {"required": True, "content": {"application/json": {"schema": TtsRequest.model_json_schema()}}},
    })
    async def tts(request: Request) -> Response:
        nonlocal inflight
        try:
            base = tts_endpoint()
        except StageConfigError as exc:
            raise ApiFailure(503, "tts_unavailable", None, str(exc)) from None
        if base is None:
            raise ApiFailure(503, "tts_disabled", None, f"{ENV_TTS_URL} is not set")
        payload = await admit(request)
        if inflight >= MAX_INFLIGHT:
            raise ApiFailure(429, "tts_busy", 1)
        inflight += 1
        started = time.monotonic()
        outcome = "ok"
        try:
            try:
                response = await http().post(f"{base}/tts", json={"text": payload.text}, timeout=TTS_TIMEOUT_S)
            except httpx.TimeoutException:
                outcome = "timeout"
                raise ApiFailure(504, "tts_timeout") from None
            except httpx.HTTPError:
                outcome = "unreachable"
                raise ApiFailure(503, "tts_unavailable") from None
            if response.status_code == 400:
                outcome = "refused"
                raise ApiFailure(422, "invalid_request")
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if response.status_code != 200 or content_type != "audio/wav" or not 44 < len(response.content) <= MAX_AUDIO_BYTES:
                outcome = f"upstream_{response.status_code}"
                raise ApiFailure(502, "tts_failed")
            return Response(response.content, media_type="audio/wav", headers={"Cache-Control": "no-store"})
        finally:
            inflight -= 1
            # Enum-only, like the provider log: never the text, which can be the user's own words echoed back.
            logger.info("tts outcome=%s chars=%d elapsed_ms=%d", outcome, len(payload.text),
                        int((time.monotonic() - started) * 1000))

    return router

"""Server voice (``/api/tts``, ``/api/tts/config``): off without ``AISW_TTS_URL``, loopback only, the session gates,
the text bound, and every service failure mapped to one closed-set error the page can fall back on.

The TTS service is an in-process fake behind ``httpx.MockTransport`` on the app's shared client; nothing
synthesises audio or opens a socket.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from backend.app.main import store
from backend.app.tts import MAX_TTS_CHARS
from backend.tests.test_guide import ORIGIN, api_client, env, install, open_session  # noqa: F401  (env: fixture)

TTS_URL = "http://127.0.0.1:8060"
WAV = b"RIFF" + b"\0" * 40 + b"\1\0" * 100


class FakeTts:
    def __init__(self, *, status: int = 200, content_type: str = "audio/wav", body: bytes = WAV,
                 error: Exception | None = None, ready: bool = True) -> None:
        self.status, self.content_type, self.body, self.error, self.ready = status, content_type, body, error, ready
        self.texts: list[str] = []
        self.gate: asyncio.Event | None = None

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if self.error is not None:
            raise self.error
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": self.ready})
        assert request.url.path == "/tts"
        self.texts.append(json.loads(request.content)["text"])
        if self.gate is not None:
            await self.gate.wait()
        return httpx.Response(self.status, content=self.body, headers={"content-type": self.content_type})


@pytest.fixture
def tts(env: pytest.MonkeyPatch):  # noqa: F811  (the imported fixture)
    env.setenv("AISW_TTS_URL", TTS_URL)
    install()
    fake = FakeTts()
    env.setattr(store, "client", httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    return fake


async def speak(api: httpx.AsyncClient, headers: dict[str, str], body: Any) -> httpx.Response:
    return await api.post("/api/tts", json=body, headers=headers)


def test_config_reports_off_unset_and_non_loopback(env: pytest.MonkeyPatch) -> None:  # noqa: F811
    async def exercise() -> None:
        async with api_client() as api:
            env.delenv("AISW_TTS_URL", raising=False)
            assert (await api.get("/api/tts/config")).json() == {"enabled": False, "ready": False}
            env.setenv("AISW_TTS_URL", "http://10.0.0.5:8060")
            assert (await api.get("/api/tts/config")).json() == {"enabled": False, "ready": False}
            install()
            headers, _ = await open_session(api)
            refused = await speak(api, headers, {"text": "안녕하세요"})
            assert refused.status_code == 503 and refused.json()["error"] == "tts_unavailable"
            assert "AISW_TTS_URL" in refused.json()["detail"] and "10.0.0.5" not in refused.json()["detail"]
            env.delenv("AISW_TTS_URL")
            off = await speak(api, headers, {"text": "안녕하세요"})
            assert off.status_code == 503 and off.json()["error"] == "tts_disabled"

    asyncio.run(exercise())


def test_config_reports_service_readiness(tts: FakeTts) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            assert (await api.get("/api/tts/config")).json() == {"enabled": True, "ready": True}
            tts.ready = False
            assert (await api.get("/api/tts/config")).json() == {"enabled": True, "ready": False}
            tts.error = httpx.ConnectError("refused")
            assert (await api.get("/api/tts/config")).json() == {"enabled": True, "ready": False}

    asyncio.run(exercise())


def test_speak_returns_the_service_wav(tts: FakeTts) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, _ = await open_session(api)
            response = await speak(api, headers, {"text": "1단계 / 2. 안경을 집으세요."})
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            assert response.headers["cache-control"] == "no-store"
            assert response.content == WAV
            assert tts.texts == ["1단계 / 2. 안경을 집으세요."]

    asyncio.run(exercise())


def test_speak_requires_origin_and_session(tts: FakeTts) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            assert (await speak(api, {}, {"text": "안녕"})).json()["error"] == "origin_required"
            no_session = await speak(api, ORIGIN, {"text": "안녕"})
            assert no_session.status_code == 401 and no_session.json()["error"] == "session_required"
            assert tts.texts == []

    asyncio.run(exercise())


def test_speak_bounds_the_text(tts: FakeTts) -> None:
    async def exercise() -> None:
        async with api_client() as api:
            headers, _ = await open_session(api)
            for bad in ({"text": ""}, {"text": "   "}, {"text": "가" * (MAX_TTS_CHARS + 1)}, {"text": "안녕", "voice": "x"},
                        {"say": "안녕"}):
                response = await speak(api, headers, bad)
                assert response.status_code == 422 and response.json()["error"] == "invalid_request", bad
            assert (await speak(api, headers, {"text": "가" * MAX_TTS_CHARS})).status_code == 200
            assert len(tts.texts) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(("fake", "status", "code"), [
    (FakeTts(error=httpx.ConnectError("refused")), 503, "tts_unavailable"),
    (FakeTts(error=httpx.ReadTimeout("slow")), 504, "tts_timeout"),
    (FakeTts(status=400, content_type="application/json", body=b'{"error":"bad_text"}'), 422, "invalid_request"),
    (FakeTts(status=500, content_type="application/json", body=b'{"error":"synthesis_failed"}'), 502, "tts_failed"),
    (FakeTts(content_type="text/html", body=b"<html></html>"), 502, "tts_failed"),
    (FakeTts(body=b"RIFF"), 502, "tts_failed"),
])
def test_service_failures_map_to_closed_errors(env: pytest.MonkeyPatch, fake: FakeTts, status: int,  # noqa: F811
                                               code: str) -> None:
    env.setenv("AISW_TTS_URL", TTS_URL)
    install()
    env.setattr(store, "client", httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))

    async def exercise() -> None:
        async with api_client() as api:
            headers, _ = await open_session(api)
            response = await speak(api, headers, {"text": "안녕하세요"})
            assert (response.status_code, response.json()["error"]) == (status, code)

    asyncio.run(exercise())


def test_requests_beyond_the_inflight_bound_are_refused(tts: FakeTts) -> None:
    async def exercise() -> None:
        tts.gate = asyncio.Event()
        async with api_client() as api:
            headers, _ = await open_session(api)
            held = [asyncio.create_task(speak(api, headers, {"text": f"문장 {n}"})) for n in range(4)]
            while len(tts.texts) < 4:
                await asyncio.sleep(0)
            refused = await speak(api, headers, {"text": "다섯째"})
            assert refused.status_code == 429 and refused.json()["error"] == "tts_busy"
            assert refused.headers["retry-after"] == "1"
            tts.gate.set()
            assert [r.status_code for r in await asyncio.gather(*held)] == [200] * 4
            assert (await speak(api, headers, {"text": "다섯째"})).status_code == 200

    asyncio.run(exercise())

"""API-level integration tests for session authentication, rate limits, origins, and health."""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

import logging

from backend.app.main import LOCAL_ONLY_ENV, OPEN_ACCESS_ENV, access_mode_startup_check, app, store
from backend.app.provider import ProviderConfigError, get_provider


def test_unicode_access_code_and_readiness(monkeypatch) -> None:
    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        monkeypatch.setenv("DEMO_ACCESS_CODE", "한국어-비밀")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
                headers = {"Origin": "http://127.0.0.1:8000"}
                health = (await api.get("/api/health")).json()
                assert health["ready"] is True
                assert health["model"] == store.provider.model

                # Wrong code rejected
                wrong = await api.post("/api/session", json={"access_code": "다른-비밀"}, headers=headers)
                assert wrong.status_code == 401
                assert wrong.json() == {"error": "invalid_access_code"}

                # Correct code accepted
                correct = await api.post("/api/session", json={"access_code": "한국어-비밀"}, headers=headers)
                assert correct.status_code == 200
                assert "visual_coach_session" in correct.headers["set-cookie"]

                headers["Cookie"] = correct.headers["set-cookie"].split(";", 1)[0]
                ended = await api.post("/api/session/end", headers=headers)
                assert ended.status_code == 200
                assert ended.json() == {"ended": True}

                # Removing access code marks health ready=False
                monkeypatch.delenv("DEMO_ACCESS_CODE")
                assert (await api.get("/api/health")).json()["ready"] is False
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())


def test_proxied_origin_matches_forwarded_host_only(monkeypatch) -> None:
    async def exercise() -> None:
        monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
            forwarded = {"Host": "127.0.0.1:5187", "Origin": "http://127.0.0.1:5187"}
            allowed = await api.post("/api/session", json={"access_code": "local-only"}, headers=forwarded)
            assert allowed.status_code == 503
            assert allowed.json() == {"error": "service_unavailable"}

            # Mismatched origin rejected
            mismatched = await api.post("/api/session", json={"access_code": "local-only"},
                                        headers={"Host": "127.0.0.1:5187", "Origin": "http://evil.com"})
            assert mismatched.status_code == 403
            assert mismatched.json() == {"error": "invalid_origin"}

            # Missing origin rejected
            missing = await api.post("/api/session", json={"access_code": "local-only"},
                                     headers={"Host": "127.0.0.1:5187"})
            assert missing.status_code == 403
            assert missing.json() == {"error": "origin_required"}

    asyncio.run(exercise())


def test_session_creation_rate_limit(monkeypatch) -> None:
    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        monkeypatch.setenv("DEMO_ACCESS_CODE", "test-code")
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
                headers = {"Origin": "http://127.0.0.1:8000"}
                for i in range(5):
                    res = await api.post("/api/session", json={"access_code": "test-code"}, headers=headers)
                    assert res.status_code == 200

                # 6th attempt in 1 minute should be rate limited
                res6 = await api.post("/api/session", json={"access_code": "test-code"}, headers=headers)
                assert res6.status_code == 429
                assert res6.json() == {"error": "rate_limited"}
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())


def test_unsupported_provider_raises_config_error(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER", "invalid_provider_name")
    client = httpx.AsyncClient()
    try:
        with pytest.raises(ProviderConfigError, match="Unsupported PROVIDER configured"):
            get_provider(client)
    finally:
        asyncio.run(client.aclose())


def test_code_protected_deployment_rejects_missing_or_empty_code(monkeypatch) -> None:
    """Without local mode the code is still mandatory: absence is refused, not treated as a bypass."""

    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        monkeypatch.delenv(LOCAL_ONLY_ENV, raising=False)
        monkeypatch.setenv("DEMO_ACCESS_CODE", "test-code")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
                headers = {"Origin": "http://127.0.0.1:8000"}
                for body in ({}, {"access_code": ""}, {"access_code": "wrong-code"}):
                    denied = await api.post("/api/session", json=body, headers=headers)
                    assert denied.status_code == 401
                    assert denied.json() == {"error": "invalid_access_code"}
                    assert "visual_coach_session" not in denied.headers.get("set-cookie", "")

                accepted = await api.post("/api/session", json={"access_code": "test-code"}, headers=headers)
                assert accepted.status_code == 200
                assert (await api.get("/api/health")).json()["access_code_required"] is True
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())


def test_local_only_mode_issues_sessions_without_a_code(monkeypatch) -> None:
    """An explicitly local deployment opens a session for a loopback peer naming a loopback authority."""

    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
        monkeypatch.setenv(LOCAL_ONLY_ENV, "1")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
                health = (await api.get("/api/health")).json()
                assert health["ready"] is True
                assert health["access_code_required"] is False

                headers = {"Origin": "http://127.0.0.1:8000"}
                opened = await api.post("/api/session", json={}, headers=headers)
                assert opened.status_code == 200
                cookie = opened.headers["set-cookie"]
                assert "visual_coach_session" in cookie
                # The intended plain-HTTP local deployment must receive a cookie it can send back.
                assert "Secure" not in cookie

                # The cookie really is the credential: it ends the session it opened.
                ended = await api.post("/api/session/end", headers={**headers, "Cookie": cookie.split(";", 1)[0]})
                assert ended.status_code == 200
                assert ended.json() == {"ended": True}

            # The IPv6 loopback forms are loopback too — a dual-stack localhost must not fall through.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("::1", 55551)),
                                         base_url="http://[::1]:8000") as api:
                dual = await api.post("/api/session", json={}, headers={"Host": "[::1]:8000", "Origin": "http://[::1]:8000"})
                assert dual.status_code == 200
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("::ffff:127.0.0.1", 55552)),
                                         base_url="http://localhost:8000") as api:
                mapped = await api.post("/api/session", json={}, headers={"Host": "localhost:8000", "Origin": "http://localhost:8000"})
                assert mapped.status_code == 200
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())


def test_local_only_mode_refuses_non_loopback_peer_or_authority(monkeypatch) -> None:
    """Local mode trusts one connection only: a loopback peer *and* a loopback authority."""

    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
        monkeypatch.setenv(LOCAL_ONLY_ENV, "1")
        try:
            # A LAN/tunnel peer that is not loopback can never reach the code-less path.
            transport = httpx.ASGITransport(app=app, client=("100.73.106.50", 41234))
            async with httpx.AsyncClient(transport=transport, base_url="http://100.73.106.50:8040") as api:
                headers = {"Host": "100.73.106.50:8040", "Origin": "http://100.73.106.50:8040"}
                denied = await api.post("/api/session", json={}, headers=headers)
                assert denied.status_code == 403
                assert denied.json() == {"error": "local_only"}
                assert (await api.post("/api/session/end", headers=headers)).json() == {"error": "local_only"}

            # A loopback peer that names a public or malformed authority is refused as well: a proxied
            # or forwarded request must not borrow the loopback peer address.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 55553)),
                                         base_url="http://aisw.example.com") as api:
                proxied = {"Host": "aisw.example.com", "Origin": "http://aisw.example.com"}
                forwarded = await api.post("/api/session", json={}, headers=proxied)
                assert forwarded.status_code == 403
                assert forwarded.json() == {"error": "local_only"}

                for authority in ("localhost.evil.com", "127.0.0.1:notaport"):
                    headers = {"Host": authority, "Origin": f"http://{authority}"}
                    refused = await api.post("/api/session", json={}, headers=headers)
                    assert refused.status_code == 403, authority
                    assert refused.json() == {"error": "local_only"}

                # A path in the authority is already refused by the same-origin guard; either way it
                # never reaches the code-less path.
                malformed = {"Host": "127.0.0.1/path", "Origin": "http://127.0.0.1/path"}
                assert (await api.post("/api/session", json={}, headers=malformed)).status_code == 403

                # X-Forwarded-For is never consulted as a trust signal.
                spoofed = {**proxied, "X-Forwarded-For": "127.0.0.1"}
                assert (await api.post("/api/session", json={}, headers=spoofed)).json() == {"error": "local_only"}

            # Forwarding metadata on an otherwise perfect local request is itself the refusal: proxy
            # headers may have rewritten the peer address, so a forwarded request never gets a session.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 55554)),
                                         base_url="http://127.0.0.1:8000") as api:
                local = {"Host": "127.0.0.1:8000", "Origin": "http://127.0.0.1:8000"}
                assert (await api.post("/api/session", json={}, headers=local)).status_code == 200
                for name, value in (
                    ("Forwarded", "for=127.0.0.1"),
                    ("X-Forwarded-For", "127.0.0.1"),
                    ("X-Forwarded-Host", "127.0.0.1:8000"),
                    ("X-Forwarded-Proto", "https"),
                    ("X-Real-IP", "127.0.0.1"),
                ):
                    denied = await api.post("/api/session", json={}, headers={**local, name: value})
                    assert denied.status_code == 403, name
                    assert denied.json() == {"error": "local_only"}
                    assert "visual_coach_session" not in denied.headers.get("set-cookie", "")
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())


def test_local_mode_stays_off_for_other_env_values(monkeypatch) -> None:
    """Only the exact value "1" enables local mode, so a typo keeps the code requirement."""

    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
                headers = {"Origin": "http://127.0.0.1:8000"}
                for value in ("true", "yes", "on", "0", ""):
                    monkeypatch.setenv(LOCAL_ONLY_ENV, value)
                    assert (await api.get("/api/health")).json()["access_code_required"] is True
                    unavailable = await api.post("/api/session", json={}, headers=headers)
                    assert unavailable.status_code == 503, value
                    assert unavailable.json() == {"error": "service_unavailable"}
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())


TAILNET = "live.example.test:8444"


def test_open_access_mode_issues_sessions_without_a_code(monkeypatch) -> None:
    """AISW_OPEN_ACCESS=1: no code at all, a supplied code (right, wrong, empty) is ignored, proxied traffic allowed."""

    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        monkeypatch.delenv(LOCAL_ONLY_ENV, raising=False)
        monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
        monkeypatch.setenv(OPEN_ACCESS_ENV, "1")
        try:
            # The deployed shape: Tailscale Serve -> tunnel -> uvicorn --proxy-headers, so the app sees an https
            # scheme, the public tailnet authority and forwarding metadata from a (rewritten) tailnet peer.
            transport = httpx.ASGITransport(app=app, client=("100.101.102.103", 41000))
            async with httpx.AsyncClient(transport=transport, base_url=f"https://{TAILNET}") as api:
                health = (await api.get("/api/health")).json()
                assert health["ready"] is True
                assert health["access_code_required"] is False
                assert health["access_mode"] == "open"

                proxied = {"Host": TAILNET, "Origin": f"https://{TAILNET}", "X-Forwarded-For": "100.101.102.103",
                           "X-Forwarded-Proto": "https"}
                opened = await api.post("/api/session", json={}, headers=proxied)
                assert opened.status_code == 200
                cookie = opened.headers["set-cookie"]
                assert "visual_coach_session" in cookie
                assert "Secure" in cookie and "HttpOnly" in cookie and "samesite=strict" in cookie.lower()
                session_id = opened.json()["session_id"]

                # The session gates the guide routes exactly as before: no code deployment is required.
                jar = {**proxied, "Cookie": cookie.split(";", 1)[0]}
                current = await api.get("/api/guide/plan/current", params={"session_id": session_id}, headers=jar)
                assert current.status_code == 404
                assert current.json()["error"] == "no_plan"

                for body in ({"access_code": "wrong-code"}, {"access_code": ""}):
                    assert (await api.post("/api/session", json=body, headers=proxied)).status_code == 200

                # The origin check is unchanged.
                bad = await api.post("/api/session", json={}, headers={**proxied, "Origin": "https://evil.example"})
                assert bad.status_code == 403
                assert bad.json() == {"error": "invalid_origin"}
                plain = await api.post("/api/session", json={}, headers={**proxied, "Origin": f"http://{TAILNET}"})
                assert plain.status_code == 403
                missing = await api.post("/api/session", json={}, headers={"Host": TAILNET})
                assert missing.json() == {"error": "origin_required"}

            # A configured code is ignored, not enforced: open wins.
            monkeypatch.setenv("DEMO_ACCESS_CODE", "test-code")
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
                headers = {"Origin": "http://127.0.0.1:8000"}
                assert (await api.get("/api/health")).json()["access_mode"] == "open"
                assert (await api.post("/api/session", json={}, headers=headers)).status_code == 200
                assert (await api.post("/api/session", json={"access_code": "wrong"}, headers=headers)).status_code == 200
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())


def test_open_access_mode_stays_off_for_other_env_values(monkeypatch) -> None:
    """Only the exact value "1" enables open mode; anything else keeps the code requirement (fail-closed)."""

    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        monkeypatch.delenv(LOCAL_ONLY_ENV, raising=False)
        monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
                headers = {"Origin": "http://127.0.0.1:8000"}
                for value in ("true", "yes", "on", "0", "", " 1", "1 ", "01"):
                    monkeypatch.setenv(OPEN_ACCESS_ENV, value)
                    health = (await api.get("/api/health")).json()
                    assert health["access_code_required"] is True, value
                    assert health["access_mode"] == "code", value
                    assert health["ready"] is False, value
                    unavailable = await api.post("/api/session", json={}, headers=headers)
                    assert unavailable.status_code == 503, value
                    assert unavailable.json() == {"error": "service_unavailable"}
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())


def test_open_access_with_local_only_is_refused(monkeypatch, caplog) -> None:
    """Both modes at once is a configuration error: health not ready, every session route 503, logged at startup."""

    async def exercise() -> None:
        store.creations.clear()
        store.sessions.clear()
        prev_key = store.provider.api_key
        store.provider.api_key = "test-only"
        monkeypatch.setenv(OPEN_ACCESS_ENV, "1")
        monkeypatch.setenv(LOCAL_ONLY_ENV, "1")
        monkeypatch.setenv("DEMO_ACCESS_CODE", "test-code")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
                health = (await api.get("/api/health")).json()
                assert health["ready"] is False
                assert health["access_code_required"] is True
                assert health["access_mode"] is None

                headers = {"Origin": "http://127.0.0.1:8000"}
                for body in ({}, {"access_code": "test-code"}):
                    refused = await api.post("/api/session", json=body, headers=headers)
                    assert refused.status_code == 503
                    assert refused.json() == {"error": "service_unavailable"}
                    assert "visual_coach_session" not in refused.headers.get("set-cookie", "")
                assert (await api.post("/api/session/end", headers=headers)).status_code == 503
        finally:
            store.provider.api_key = prev_key

    asyncio.run(exercise())
    with caplog.at_level(logging.WARNING, logger="backend.app.main"):
        access_mode_startup_check()
    assert [record.getMessage().split(":", 1)[0] for record in caplog.records] == ["access_mode_conflict"]
    assert caplog.records[0].levelno == logging.ERROR


def test_open_access_with_a_code_logs_one_enum_line(monkeypatch, caplog) -> None:
    """Open + DEMO_ACCESS_CODE: open wins and startup logs one line that never contains the code."""
    monkeypatch.delenv(LOCAL_ONLY_ENV, raising=False)
    monkeypatch.setenv(OPEN_ACCESS_ENV, "1")
    monkeypatch.setenv("DEMO_ACCESS_CODE", "secret-code-value")
    with caplog.at_level(logging.INFO, logger="backend.app.main"):
        access_mode_startup_check()
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage().startswith("access_code_ignored:")
    assert "secret-code-value" not in caplog.text

    caplog.clear()
    monkeypatch.delenv("DEMO_ACCESS_CODE")
    with caplog.at_level(logging.INFO, logger="backend.app.main"):
        access_mode_startup_check()
    assert caplog.records == []


def test_health_names_the_access_mode(monkeypatch) -> None:
    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as api:
            monkeypatch.delenv(OPEN_ACCESS_ENV, raising=False)
            monkeypatch.delenv(LOCAL_ONLY_ENV, raising=False)
            monkeypatch.setenv("DEMO_ACCESS_CODE", "test-code")
            assert (await api.get("/api/health")).json()["access_mode"] == "code"
            monkeypatch.setenv(LOCAL_ONLY_ENV, "1")
            assert (await api.get("/api/health")).json()["access_mode"] == "local"

    asyncio.run(exercise())

from conftest import create_session, fast_settings

from synoptics_live.contracts import PROTOCOL


def test_health_open_mode(make_client):
    c = make_client()
    body = c.get("/v1/health").json()
    assert body["ready"] is True
    assert body["protocol"] == PROTOCOL
    assert body["access_mode"] == "open" and body["access_code_required"] is False
    assert body["engine"] == "mock"
    # §15: the mock engine serves every profile, and `plan_ready` is their aggregate.
    assert body["plan_models"] == {"deepseek:high": True, "astra:high": True}
    assert body["plan_ready"] is True
    assert body["tts_ready"] is False and body["tts_voice"] is None  # no TTS bridge
    assert set(body["build"]) == {"commit", "time"}


def test_health_code_mode_and_tts(make_client):
    from conftest import FakeTts

    c = make_client(fast_settings(access_code="sesame"), tts=FakeTts())
    body = c.get("/v1/health").json()
    assert body["access_mode"] == "code" and body["access_code_required"] is True
    assert body["tts_ready"] is True and body["tts_voice"] == "fake-voice"


def test_create_open_mode_and_live_url(make_client):
    c = make_client()
    s = create_session(c)
    assert len(s["session_id"]) == 32 and len(s["token"]) >= 32
    assert s["live_url"] == f"ws://testserver/v1/sessions/{s['session_id']}/live"
    assert s["expires_at"] > 1_700_000_000_000
    # behind nginx: prefix stripped, X-Forwarded-Prefix sent
    r = c.post(
        "/v1/sessions",
        json={},
        headers={"X-Forwarded-Proto": "https", "Host": "sangye.kr", "X-Forwarded-Prefix": "/synoptics/api"},
    )
    sid = r.json()["session_id"]
    assert r.json()["live_url"] == f"wss://sangye.kr/synoptics/api/v1/sessions/{sid}/live"
    # behind Tailscale Serve (--set-path): prefix stripped, nothing announced -> LIVE_PATH_PREFIXES by host
    c2 = make_client(fast_settings(path_prefixes={"live.example.test": "/synoptics/live"}))
    r = c2.post("/v1/sessions", json={}, headers={"X-Forwarded-Proto": "https", "Host": "live.example.test"})
    sid = r.json()["session_id"]
    assert r.json()["live_url"] == f"wss://live.example.test/synoptics/live/v1/sessions/{sid}/live"
    r = c2.post("/v1/sessions", json={}, headers={"X-Forwarded-Proto": "https", "Host": "live.example.test:8450"})
    assert r.json()["live_url"].startswith("wss://live.example.test:8450/v1/sessions/")  # legacy port: no prefix


def test_create_code_mode(make_client):
    c = make_client(fast_settings(access_code="sesame"))
    for body in ({}, {"access_code": "wrong"}, {"access_code": ""}):
        r = c.post("/v1/sessions", json=body, headers={"X-Real-IP": f"10.0.0.{len(body)}{len(str(body))}"})
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "invalid_access_code"
    r = c.post("/v1/sessions", json={"access_code": "sesame"}, headers={"X-Real-IP": "10.9.9.9"})
    assert r.status_code == 201


def test_bad_body_is_400(make_client):
    c = make_client()
    r = c.post("/v1/sessions", content=b"{nope", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_rate_limit_per_ip(make_client):
    c = make_client()
    hdr = {"CF-Connecting-IP": "203.0.113.7", "X-Real-IP": "10.1.1.1"}
    for _ in range(5):
        assert c.post("/v1/sessions", json={}, headers=hdr).status_code == 201
    r = c.post("/v1/sessions", json={}, headers=hdr)
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limited"
    assert 1 <= int(r.headers["Retry-After"]) <= 60
    # CF-Connecting-IP wins over X-Real-IP: a different CF IP is a different client
    assert c.post("/v1/sessions", json={}, headers={"CF-Connecting-IP": "203.0.113.8"}).status_code == 201
    # failed attempts count too (brute force on the access code)
    c2 = make_client(fast_settings(access_code="x"))
    for _ in range(5):
        assert c2.post("/v1/sessions", json={"access_code": "bad"}).status_code == 401
    assert c2.post("/v1/sessions", json={"access_code": "x"}).status_code == 429


def test_capacity(make_client):
    c = make_client(max_sessions=2)
    for i in range(2):
        assert c.post("/v1/sessions", json={}, headers={"X-Real-IP": f"10.0.0.{i}"}).status_code == 201
    r = c.post("/v1/sessions", json={}, headers={"X-Real-IP": "10.0.0.9"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "capacity"


def test_delete(make_client):
    c = make_client()
    s = create_session(c)
    url = f"/v1/sessions/{s['session_id']}"
    assert c.delete(url).status_code == 401
    assert c.delete(url, headers={"Authorization": "Bearer nope"}).status_code == 401
    r = c.delete(url, headers={"Authorization": f"Bearer {s['token']}"})
    assert r.status_code == 200 and r.json() == {"ended": True}
    assert c.delete(url, headers={"Authorization": f"Bearer {s['token']}"}).status_code == 404


def test_unknown_engine_is_not_ready(make_client):
    c = make_client(engine="real")
    assert c.get("/v1/health").json()["ready"] is False
    r = c.post("/v1/sessions", json={})
    assert r.status_code == 503 and r.json()["error"]["code"] == "service_unavailable"


def test_idle_expiry(make_client):
    c = make_client(idle_ttl_s=0.05)
    s = create_session(c)
    import time

    time.sleep(0.1)
    c.get("/v1/health")  # sweeps
    r = c.delete(f"/v1/sessions/{s['session_id']}", headers={"Authorization": f"Bearer {s['token']}"})
    assert r.status_code == 404


def test_cors(make_client):
    c = make_client(cors_origins=["https://app.example"])
    r = c.options(
        "/v1/sessions",
        headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert r.headers["access-control-allow-origin"] == "https://app.example"
    r = c.get("/v1/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_demo_page(make_client):
    c = make_client()
    r = c.get("/demo/")
    assert r.status_code == 200 and "<title>" in r.text
    r = c.get("/demo", follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "demo/"

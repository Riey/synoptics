"""LIVE_ENGINE=real end to end through the WebSocket layer, against the fake upstream (real time, no sockets)."""

from __future__ import annotations

import time

from conftest import Live, create_session, fast_settings, frame, open_live
from fake_upstream import FakeUpstream
from starlette.testclient import TestClient

from synoptics_live.app import create_app


def make_real_client(fake: FakeUpstream) -> TestClient:
    settings = fast_settings(engine="real", upstream_url=fake.base_url)
    client = TestClient(create_app(settings, tts=None, upstream_transport=fake.transport()))
    client.__enter__()
    return client


def test_health_reports_upstream_readiness():
    fake = FakeUpstream()
    c = make_real_client(fake)
    try:
        body = c.get("/v1/health").json()
        assert body["engine"] == "real" and body["ready"] is True
        assert body["guide_ready"] and body["follow_ready"] and body["tracker_ready"]
        assert body["plan_models"] == {"deepseek:high": True, "astra:high": True}
        assert body["plan_ready"] is True
    finally:
        c.__exit__(None, None, None)


def test_health_plan_ready_is_the_aggregate_of_the_profile_map():
    """§15: any configured profile keeps `plan_ready`; an empty map makes it false."""
    one = FakeUpstream(plan_models={"deepseek:high": False, "astra:high": True})
    c = make_real_client(one)
    try:
        body = c.get("/v1/health").json()
        assert body["plan_models"] == {"deepseek:high": False, "astra:high": True}
        assert body["plan_ready"] is True
    finally:
        c.__exit__(None, None, None)

    none = FakeUpstream(plan_models={})
    c = make_real_client(none)
    try:
        body = c.get("/v1/health").json()
        assert body["plan_models"] == {"deepseek:high": False, "astra:high": False}
        assert body["plan_ready"] is False and body["guide_ready"] is True
    finally:
        c.__exit__(None, None, None)


def test_real_engine_over_websocket_start_to_tracking_box():
    fake = FakeUpstream()
    c = make_real_client(fake)
    try:
        s = create_session(c)
        with open_live(c, s["session_id"]) as ws:
            live = Live(ws)
            live.hello(s["token"])
            assert live.frame()["state"] == "idle"
            live.send({"type": "start", "goal": "안경을 벗어 주세요", "consent_ai": True})
            hi = live.until_type("capture_hi")
            ws.send_bytes(frame(500, w=1024, h=576, hi_req=hi["req_id"]))
            deadline = time.time() + 5
            box = None
            while time.time() < deadline:
                tr = live.frame()
                if tr["state"] == "tracking":
                    box = tr["box"]
                    break
                time.sleep(0.02)
            assert box == fake.box
            st = live.state
            assert st["phase"] == "running" and st["plan"]["target"] == "eyeglasses"
            says = [m["text"] for m in live.says()]
            assert says[0] == "계획을 세우는 중입니다." and says[1].startswith("1단계 / 3.")
            # the acquired follow arrives; the overlay binds to the tracked anchor
            for _ in range(20):
                live.frame()
                time.sleep(0.02)
                if live.state.get("overlay"):
                    break
            assert live.state["overlay"]["binding"]["run_id"] == tr["run_id"]
            assert fake.calls("/api/guide/follow")
            # §15: even the immediate flow forwards the defaulted profile — and never the retired selector.
            plan = fake.calls("/api/guide/plan")[0]
            assert plan["plan_model"] == "deepseek:high" and "plan_mode" not in plan
            live.send({"type": "stop"})
            live.until(lambda m: m["type"] == "state" and m["phase"] == "idle")
            assert live.frame()["state"] == "idle"

    finally:
        c.__exit__(None, None, None)


def test_start_plan_model_reaches_the_upstream_verbatim():
    """§15: `start.plan_model` is accepted independently of `plan_mode` and sent exactly as given upstream."""
    fake = FakeUpstream()
    c = make_real_client(fake)
    try:
        s = create_session(c)
        with open_live(c, s["session_id"]) as ws:
            live = Live(ws)
            live.hello(s["token"])
            live.send({"type": "start", "goal": "안경을 벗어 주세요", "consent_ai": True,
                       "plan_model": "astra:high"})  # no plan_mode: immediate flow, Astra profile
            hi = live.until_type("capture_hi")
            ws.send_bytes(frame(500, w=1024, h=576, hi_req=hi["req_id"]))
            live.until(lambda m: m["type"] == "state" and m["phase"] == "running")
            plan = fake.calls("/api/guide/plan")[0]
            assert plan["plan_model"] == "astra:high"
            assert "plan_mode" not in plan
            live.send({"type": "stop"})
    finally:
        c.__exit__(None, None, None)


def test_review_run_freezes_the_choice_as_a_separate_axis():
    """§15: with `plan_mode:true` the run still reviews, and the profile rides the plan request."""
    fake = FakeUpstream()
    c = make_real_client(fake)
    try:
        s = create_session(c)
        with open_live(c, s["session_id"]) as ws:
            live = Live(ws)
            live.hello(s["token"])
            live.send({"type": "start", "goal": "안경을 벗어 주세요", "consent_ai": True, "plan_mode": True,
                       "plan_model": "astra:high"})
            hi = live.until_type("capture_hi")
            ws.send_bytes(frame(500, w=1024, h=576, hi_req=hi["req_id"]))
            reviewing = live.until(lambda m: m["type"] == "state" and m["phase"] == "reviewing")
            assert reviewing["plan"]["status"] == "draft"
            plan = fake.calls("/api/guide/plan")[0]
            assert plan["plan_model"] == "astra:high" and "plan_mode" not in plan
            # the frozen choice still drives approve-returned execution (same upstream plan pair)
            live.send({"type": "plan_approve"})
            approved = live.until(lambda m: m["type"] == "state" and m["phase"] == "running")
            assert approved["plan"]["status"] == "approved"
            live.send({"type": "stop"})
    finally:
        c.__exit__(None, None, None)


def test_unknown_plan_model_is_refused_before_any_upstream_plan_call():
    """§15: the profile set is closed at the WS boundary — nothing starts and no plan is requested."""
    fake = FakeUpstream()
    c = make_real_client(fake)
    try:
        s = create_session(c)
        with open_live(c, s["session_id"]) as ws:
            live = Live(ws)
            live.hello(s["token"])
            live.send({"type": "start", "goal": "안경을 벗어 주세요", "consent_ai": True,
                       "plan_model": "gpt-6-astra"})
            err = live.until_type("error")
            assert err["code"] == "invalid_message" and err["retryable"] is False
            assert not fake.calls("/api/guide/plan")
            assert live.state["phase"] == "idle"
    finally:
        c.__exit__(None, None, None)

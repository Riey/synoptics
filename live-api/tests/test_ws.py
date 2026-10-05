import asyncio
import time

import pytest
from conftest import Closed, Live, create_session, frame, jpeg, open_live

from synoptics_live.contracts import PROTOCOL, Limits, StateMsg, TrackMsg

START = {"type": "start", "goal": "물건을 정리하기", "context": "", "consent_ai": True}


def connect(c):
    s = create_session(c)
    return s, open_live(c, s["session_id"])


def is_state(pred):
    return lambda m: m["type"] == "state" and pred(m)


def start_running(L: Live) -> dict:
    L.send(START)
    return L.drive(lambda s: s["phase"] == "running" and s["overlay"] is not None)


# ------------------------------------------------------------------ connection / auth


def test_no_hello_closes_4008(make_client):
    c = make_client()
    _, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        assert L.recv() == {"type": "bye", "reason": "protocol_violation"}
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 4008


def test_bad_token_closes_4001(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.send({"type": "hello", "token": "wrong", "protocol": PROTOCOL, "resume_rev": None})
        assert L.recv()["reason"] == "auth_failed"
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 4001
    with open_live(c, "0" * 32) as ws:  # unknown session
        L = Live(ws)
        L.send({"type": "hello", "token": s["token"], "protocol": PROTOCOL})
        assert L.recv()["reason"] == "auth_failed"
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 4001


def test_unsupported_protocol(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.send({"type": "hello", "token": s["token"], "protocol": "synoptics-live/9"})
        err = L.recv()
        assert err["type"] == "error" and err["code"] == "unsupported_protocol" and err["fatal"] is True
        assert L.recv()["type"] == "bye"
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 4008


def test_origin_check(make_client):
    from starlette.websockets import WebSocketDisconnect

    c = make_client(cors_origins=["https://app.example"])
    s = create_session(c)
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(f"/v1/sessions/{s['session_id']}/live", headers={"Origin": "https://evil.example"}):
            pass
    with c.websocket_connect(
        f"/v1/sessions/{s['session_id']}/live", headers={"Origin": "https://app.example"}
    ) as ws:
        Live(ws).hello(s["token"])


# ------------------------------------------------------------------ happy path


def test_happy_path(make_client):
    c = make_client()
    s, cm = connect(c)
    session = c.app.state.store.sessions[s["session_id"]]
    with cm as ws:
        L = Live(ws)
        ready, st = L.hello(s["token"])
        assert ready["rev"] == 0 and st["rev"] == 0
        assert set(ready["limits"]) == set(Limits.model_fields)
        assert ready["limits"]["stream_long_side"] == 640 and ready["limits"]["resume_grace_ms"] == 500
        StateMsg.model_validate(st)
        assert st["phase"] == "idle" and st["plan"] is None

        L.send({"type": "prefs", "voice_out": True, "tts": "server", "visuals": True})
        t = L.frame()
        assert t["state"] == "idle" and t["seq"] == 0 and t["captured_at"] == 1_759_479_000_000
        assert t["box"] is None

        L.send(START)
        planning = L.until(is_state(lambda m: m["phase"] == "planning"))
        assert planning["pending"]["stage"] == "plan"
        say = L.until_type("say")
        assert say == {"type": "say", "line_id": say["line_id"], "text": "계획을 세우는 중입니다.",
                       "mode": "replace", "audio": "none"}
        hi = L.until_type("capture_hi")
        ws.send_bytes(frame(L.seq, w=1024, h=576, hi_req=hi["req_id"]))
        L.seq += 1

        running = L.drive(lambda m: m["phase"] == "running")
        partials = [m["partial"] for m in L.seen if m["type"] == "state" and m["partial"]]
        assert partials[0] == {"target": "목표 물체", "first_say": None}
        assert partials[-1]["first_say"] == "목표 물체를 손으로 잡으세요"
        assert running["partial"] is None and running["step_index"] == 0
        plan = running["plan"]
        assert plan["revision"] == 1 and [st["id"] for st in plan["steps"]] == ["s1", "s2", "s3"]
        assert plan["steps"][1]["commands"] == [
            {"kind": "action", "anchor": "target", "action": "rotate", "direction": "clockwise"}
        ]
        assert L.seen_or_next(lambda m: m["type"] == "say" and m["text"].startswith("1단계."))["mode"] == "replace"
        assert session.engine.hi_frames_used == 1

        tracks: list[dict] = []
        for _ in range(200):
            tracks.append(L.frame())
            if tracks[-1]["state"] == "tracking" and L.state["overlay"]:
                break
            time.sleep(0.002)
        tr = tracks[-1]
        TrackMsg.model_validate(tr)
        assert tr["box"]["width"] == 0.25 and tr["box"]["height"] == 0.2
        ov = L.state["overlay"]
        assert ov["binding"] == {"anchor_id": "a1", "run_id": tr["run_id"], "track_id": tr["track_id"],
                                 "generation": tr["generation"]}
        assert ov["commands"] == [{"kind": "focus", "pad": 0.15},
                                  {"kind": "action", "action": "grasp", "direction": "none"}]
        assert ov["motion_key"] == f"{tr['run_id']}|1|s1|a1|{tr['track_id']}|{tr['generation']}"
        all_tracks = [m for m in L.seen if m["type"] == "track" and m.get("run_id") == tr["run_id"]]
        assert [m["state"] for m in all_tracks[:5]] == ["acquiring"] * 5

        L.drive(lambda m: m["step_index"] == 1)
        heartbeat = [m for m in L.seen if m["type"] == "state" and m["pending"]
                     and m["pending"]["trigger"] == "heartbeat"]
        assert heartbeat and heartbeat[0]["step_index"] == 0
        assert L.state["pending"] is None and L.state["overlay"]["motion_key"].split("|")[2] == "s2"
        L.seen_or_next(lambda m: m["type"] == "say" and m["text"] == "2단계. 목표 물체를 시계 방향으로 돌리세요")

        L.drive(lambda m: m["completion"] == "confirmed")
        assert any(m["type"] == "state" and m["completion"] == "checking" for m in L.seen)
        done_say = L.seen_or_next(lambda m: m["type"] == "say" and "완료를 확인해" in m["text"])
        assert done_say["mode"] == "append"

        L.send({"type": "confirm_done"})
        final = L.until(is_state(lambda m: m["phase"] == "completed"))
        assert final["completion"] == "user_confirmed" and final["overlay"] is None
        revs = [m["rev"] for m in L.seen if m["type"] == "state"]
        assert revs == sorted(revs) and len(set(revs)) == len(revs)

        L.send({"type": "stop"})
        idle = L.until(is_state(lambda m: m["phase"] == "idle"))
        assert idle["plan"] is None
        assert L.frame()["state"] == "idle"


def test_select_box_reseeds(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        st = start_running(L)
        gen, track = st["overlay"]["binding"]["generation"], st["overlay"]["binding"]["track_id"]
        box = {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.3}
        tr = L.frame(select_box=box)
        assert tr["state"] == "tracking" and tr["generation"] == gen + 1 and tr["track_id"] != track
        assert abs(tr["box"]["x"] - 0.1) < 0.01 and tr["box"]["width"] == 0.2
        b = L.state["overlay"]["binding"]
        assert (b["generation"], b["track_id"]) == (gen + 1, tr["track_id"])
        assert L.state["phase"] == "running"
        # out-of-frame box is refused, guide continues
        ws.send_bytes(frame(L.seq, select_box={"x": 0.9, "y": 0.2, "width": 0.2, "height": 0.3}))
        L.seq += 1
        assert L.until_type("error")["code"] == "frame_invalid"
        assert L.frame()["generation"] == gen + 1


# ------------------------------------------------------------------ talk / follow / replan / gating


def test_talk(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        start_running(L)
        L.send({"type": "talk", "utterance": "뚜껑이 안 열려요"})
        L.send({"type": "talk", "utterance": "또 질문"})
        L.until(is_state(lambda m: m["talk_pending"] is True))
        assert L.until_type("error")["code"] == "talk_busy"
        st = L.until(is_state(lambda m: m["talk"] is not None))
        assert st["talk_pending"] is False
        talk = st["talk"]
        assert talk["utterance"] == "뚜껑이 안 열려요" and talk["reply"].startswith("(모의 응답)")
        assert len(talk["reply"]) <= 400 and len(talk["spoken"]) <= 60
        say = L.until_type("say")
        assert say["text"] == talk["spoken"] and say["mode"] == "replace"

        L.send({"type": "talk", "utterance": "처음부터 다시 알려줘"})
        st = L.until(is_state(lambda m: m["plan"]["revision"] == 2))
        assert st["step_index"] == 0 and st["talk"]["utterance"] == "처음부터 다시 알려줘"
        L.until(lambda m: m["type"] == "say" and m["mode"] == "append" and m["text"].startswith("1단계."))

        L.send({"type": "talk", "utterance": "잡았어요 했어"})
        st = L.until(is_state(lambda m: m["step_index"] == 1))
        assert st["steps_user_done"] == ["s1"]
        L.until(lambda m: m["type"] == "say" and m["text"].startswith("2단계."))


def test_follow_now_notice_autoclears(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        start_running(L)
        L.send({"type": "follow_now"})
        L.until(is_state(lambda m: m["pending"] and m["pending"]["trigger"] == "manual"))
        st = L.until(is_state(lambda m: m["notice"] is not None))
        assert st["notice"] == {"code": "rechecked", "text": "지금 화면을 다시 확인했어요."}
        assert st["pending"] is None
        L.until(is_state(lambda m: m["notice"] is None))  # 8 virtual seconds later


def test_replan_limit(make_client):
    c = make_client()
    s, cm = connect(c)
    gap = 15 / 50 + 0.05
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        start_running(L)

        def replan() -> dict:
            before = L.state["rev"]
            L.send({"type": "replan_now"})
            return L.until(is_state(lambda m: m["rev"] > before))

        assert replan()["plan"]["revision"] == 2
        assert L.until_type("say")["text"].startswith("계획을 다시 짰어요. 1단계.")
        blocked = replan()
        assert blocked["plan"]["revision"] == 2 and "초 뒤에" in blocked["replan_blocked_reason"]
        time.sleep(gap)
        st = replan()
        assert st["plan"]["revision"] == 3 and st["replan_blocked_reason"] is None
        time.sleep(gap)
        assert replan()["plan"]["revision"] == 4
        time.sleep(gap)
        blocked = replan()
        assert blocked["plan"]["revision"] == 4 and "3번" in blocked["replan_blocked_reason"]


def test_consent_and_gating(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        L.send({"type": "start", "goal": "x"})
        assert L.until_type("error")["code"] == "consent_required"
        L.send({**START, "consent_ai": False})
        assert L.until_type("error")["code"] == "consent_required"
        L.send({"type": "talk", "utterance": "안녕", "consent_ai": False})
        assert L.until_type("error")["code"] == "consent_required"
        for kind in ("talk", "follow_now", "replan_now", "confirm_done"):
            L.send({"type": kind, "utterance": "안녕"} if kind == "talk" else {"type": kind})
            err = L.until_type("error")
            assert err["code"] == "not_running" and err["fatal"] is False, kind
        L.send(START)
        L.send({"type": "talk", "utterance": "안녕"})
        L.send({"type": "replan_now"})
        assert L.until_type("error")["code"] == "plan_in_flight"
        assert L.until_type("error")["code"] == "plan_in_flight"
        # invalid fields are a non-fatal error; the socket stays usable
        L.send({"type": "start", "goal": "", "consent_ai": True})
        assert L.until_type("error")["code"] == "invalid_message"
        L.send({"type": "nonsense"})
        assert L.until_type("error")["code"] == "invalid_message"
        assert L.frame()["seq"] >= 0


# ------------------------------------------------------------------ frames


def test_frame_validation(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        bad = [
            b"\x00\x01",                                       # envelope
            frame(0, payload=b"not a jpeg"),                   # FFD8 missing
            frame(0, payload=b"\xff\xd8\xff\xe0garbage"),        # undecodable
            frame(0, w=320, h=180, payload=jpeg(640, 360)),    # w/h mismatch
            frame(0, hi_req="h99"),                            # unknown hi request
        ]
        for data in bad:
            ws.send_bytes(data)
            assert L.until_type("error")["code"] == "frame_invalid"
        big = b"\xff\xd8" + b"\x00" * 130_000
        ws.send_bytes(frame(0, payload=big))
        assert L.until_type("error")["code"] == "frame_too_large"
        assert L.frame()["state"] == "idle"  # still alive
        ws.send_bytes(b"\x00" * (512 * 1024 + 1))
        assert L.until_type("bye")["reason"] == "message_too_large"
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 4009


def test_invalid_json_closes_4008(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        ws.send_text("{not json")
        assert L.until_type("bye")["reason"] == "protocol_violation"
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 4008


def test_credit_violation(make_client):
    c = make_client()
    s, cm = connect(c)
    engine = c.app.state.store.sessions[s["session_id"]].engine
    original = engine.on_frame

    async def slow(header, payload):
        await asyncio.sleep(0.3)
        return await original(header, payload)

    engine.on_frame = slow
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        ws.send_bytes(frame(0))
        ws.send_bytes(frame(1))
        err = L.until_type("error")
        assert err["code"] == "rate_limited" and "seq 1" in err["message"] and err["fatal"] is False
        assert L.until_type("track")["seq"] == 0
        ws.send_bytes(frame(2))
        for i in range(3, 30):
            ws.send_bytes(frame(i))
        assert L.until_type("bye")["reason"] == "protocol_violation"
        assert sum(1 for m in L.seen if m["type"] == "error" and m["code"] == "rate_limited") == 19
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 4008


# ------------------------------------------------------------------ reconnect


def test_second_connection_replaces_first(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws1:
        L1 = Live(ws1)
        L1.hello(s["token"])
        with open_live(c, s["session_id"]) as ws2:
            L2 = Live(ws2)
            L2.hello(s["token"])
            assert L1.until_type("bye")["reason"] == "replaced"
            with pytest.raises(Closed) as e:
                L1.recv()
            assert e.value.code == 4003
            assert L2.frame()["state"] == "idle"


def test_resume_after_reconnect(make_client):
    c = make_client(resume_grace_s=5.0)
    s, cm = connect(c)
    session = c.app.state.store.sessions[s["session_id"]]
    with cm as ws1:
        L1 = Live(ws1)
        L1.hello(s["token"])
        st = start_running(L1)
        rev = L1.state["rev"]
        plan_id = st["plan"]["plan_id"]
    deadline = time.time() + 2
    while session.conn is not None and time.time() < deadline:
        time.sleep(0.01)
    assert session.paused
    with open_live(c, s["session_id"]) as ws2:
        L2 = Live(ws2)
        ready, state = L2.hello(s["token"], resume_rev=rev)
        assert ready["rev"] == state["rev"] >= rev
        assert state["phase"] == "running" and state["plan"]["plan_id"] == plan_id
        assert not session.paused
        assert L2.frame()["state"] == "acquiring"  # re-acquire after reconnect, seq restarts at 0
        assert L2.seen[-1]["seq"] == 0


def test_grace_expiry_then_4002(make_client):
    c = make_client(resume_grace_s=0.2)
    s, cm = connect(c)
    with cm as ws1:
        Live(ws1).hello(s["token"])
    time.sleep(0.4)
    with open_live(c, s["session_id"]) as ws2:
        L = Live(ws2)
        L.send({"type": "hello", "token": s["token"], "protocol": PROTOCOL, "resume_rev": 0})
        assert L.recv() == {"type": "bye", "reason": "session_expired"}
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 4002


def test_delete_while_connected(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        r = c.delete(f"/v1/sessions/{s['session_id']}", headers={"Authorization": f"Bearer {s['token']}"})
        assert r.json() == {"ended": True}
        assert L.until_type("bye")["reason"] == "session_ended"
        with pytest.raises(Closed) as e:
            L.recv()
        assert e.value.code == 1000

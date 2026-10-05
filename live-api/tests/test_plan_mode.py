"""§15 Plan 모드, driven through the real ASGI app and WebSocket (the deployed live engine is ``mock``).

Every rule the mock engine implements is asserted on the wire: the draft is reviewed before anything runs,
the mandatory check cannot be skipped, prerequisites gate progression, the user's pause survives a reconnect,
and the one scripted mid-run proposal is accepted or rejected without changing execution underneath.
"""

import base64
import json
import time

import pytest
from conftest import Closed, Live, create_session, frame, jpeg, open_live

from synoptics_live.contracts import PROTOCOL, StateMsg

START = {"type": "start", "goal": "상자 포장을 도와줘", "context": "", "consent_ai": True}
PLAN_START = {**START, "plan_mode": True}
DRAFT_SAY = "계획 초안을 만들었어요. 검토하고 실행을 눌러 주세요."
REQUIRED_STEP = "s7"
MATERIAL_TEXT = "3.1 상자를 펼친다. 4.1 봉함 전에 내부를 확인한다."


def connect(c):
    s = create_session(c)
    return s, open_live(c, s["session_id"])


def is_state(pred):
    return lambda m: m["type"] == "state" and pred(m)


def reviewing(L: Live) -> dict:
    L.send(PLAN_START)
    return L.drive(lambda st: st["phase"] == "reviewing")


def approved_running(L: Live) -> dict:
    reviewing(L)
    L.send({"type": "plan_approve"})
    return L.drive(lambda st: st["phase"] == "running" and st["plan"]["status"] == "approved")


def plan_ids(state: dict) -> list[str]:
    return [step["id"] for step in state["plan"]["steps"]]


# ------------------------------------------------------------------ draft / review


def test_plan_mode_start_enters_reviewing(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        st = reviewing(L)
        StateMsg.model_validate(st)
        assert st["partial"] is None and st["pending"] is None
        assert st["overlay"] is None and st["paused"] is False
        plan = st["plan"]
        assert plan["status"] == "draft" and plan["revision"] == 1 and plan["approved_revision"] is None
        ids = plan_ids(st)
        assert len(ids) >= 6 and ids == [f"s{i + 1}" for i in range(len(ids))]
        required = [step for step in plan["steps"] if step["required"]]
        assert len(required) == 1
        assert required[0]["say"] == "봉함 전 내부 확인" and required[0]["check"] == "user"
        assert any(step["requires"] for step in plan["steps"])
        assert all(step["targets"] for step in plan["steps"])
        assert all(step["evidence"] is None for step in plan["steps"])  # no material, no invented source
        L.seen_or_next(lambda m: m["type"] == "say" and m["text"] == DRAFT_SAY)

        # §15: a draft is reviewed, never tracked or progressed: every frame is idle and no overlay appears.
        tracks = [L.frame() for _ in range(6)]
        assert {t["state"] for t in tracks} == {"idle"}
        assert all(t["box"] is None and t["run_id"] is None for t in tracks)
        assert L.state["phase"] == "reviewing" and L.state["overlay"] is None
        assert L.state["step_index"] == 0 and L.state["plan"]["revision"] == 1


def test_draft_evidence_only_with_materials(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        L.send({**PLAN_START, "materials": [{"title": "포장 지침", "version": "v3", "text": MATERIAL_TEXT}]})
        st = L.drive(lambda x: x["phase"] == "reviewing")
        plan = st["plan"]
        assert plan["materials"] == [{"id": "m1", "title": "포장 지침", "version": "v3",
                                      "chars": len(MATERIAL_TEXT),
                                      "added_at": plan["materials"][0]["added_at"]}]
        cited = [step["evidence"] for step in plan["steps"] if step["evidence"] is not None]
        assert cited and all(e["material_id"] == "m1" and e["version"] == "v3" and e["locator"] for e in cited)

    # A second session without materials must not carry a single piece of evidence.
    s2, cm2 = connect(c)
    with cm2 as ws2:
        L2 = Live(ws2)
        L2.hello(s2["token"])
        st2 = reviewing(L2)
        assert st2["plan"]["materials"] == []
        assert all(step["evidence"] is None for step in st2["plan"]["steps"])


def test_plain_start_skips_reviewing(make_client):
    """Regression: without plan_mode the guide still runs straight after planning."""
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        L.send(START)
        running = L.drive(lambda st: st["phase"] == "running" and st["overlay"] is not None)
        phases = [m["phase"] for m in L.seen if m["type"] == "state"]
        assert "reviewing" not in phases and "planning" in phases
        assert running["plan"]["status"] == "approved" and running["plan"]["approved_revision"] is None
        assert plan_ids(running) == ["s1", "s2", "s3"]
        assert running["paused"] is False and running["proposal"] is None and running["blocked"] is None


# ------------------------------------------------------------------ start.plan_model (§15)


def test_plan_model_is_a_closed_choice_independent_of_review_mode(make_client):
    """§15: `plan_model` picks the planning profile; `plan_mode` still picks the review boundary."""
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        engine = c.app.state.store.sessions[s["session_id"]].engine
        L.send({**START, "plan_model": "astra:high"})  # no plan_mode: immediate flow, Astra profile
        running = L.drive(lambda st: st["phase"] == "running")
        assert running["plan"]["status"] == "approved"
        assert engine.plan_model == "astra:high" and engine.plan_mode is False


def test_plan_model_defaults_and_does_not_couple_to_review_mode(make_client):
    """§15: an omitted `plan_model` is `deepseek:high`, even when the run reviews before executing."""
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        engine = c.app.state.store.sessions[s["session_id"]].engine
        L.send(PLAN_START)
        assert L.drive(lambda st: st["phase"] == "reviewing")["plan"]["status"] == "draft"
        assert engine.plan_model == "deepseek:high" and engine.plan_mode is True


def test_unknown_plan_model_is_rejected_at_the_start_boundary(make_client):
    """§15: the profile set is closed — an unknown value is an `invalid_message`, never a run."""
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        engine = c.app.state.store.sessions[s["session_id"]].engine
        L.send({**START, "plan_mode": True, "plan_model": "gpt-6-astra"})
        err = L.until_type("error")
        assert err["code"] == "invalid_message" and err["retryable"] is False
        assert engine.plan_model == "deepseek:high" and engine.phase == "idle"


# ------------------------------------------------------------------ start.core_mode (§15.8)


def test_core_mode_is_chosen_before_a_start_and_survives_a_reconnect(make_client):
    """§15.8: `core_mode` is independent of the review boundary and the planning profile, echoed in every
    state, and — being part of the run's identity — retained across a reconnect."""
    c = make_client(resume_grace_s=5.0)
    s, cm = connect(c)
    session = c.app.state.store.sessions[s["session_id"]]
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        L.send({**START, "core_mode": "sequential", "plan_model": "astra:high", "plan_mode": True})
        st = L.drive(lambda m: m["phase"] == "reviewing")
        StateMsg.model_validate(st)
        assert st["core_mode"] == "sequential" and st["plan"]["status"] == "draft"
        engine = session.engine
        assert engine.core_mode == "sequential" and engine.plan_mode is True
        assert engine.plan_model == "astra:high"
        rev = L.state["rev"]
    deadline = time.time() + 2
    while session.conn is not None and time.time() < deadline:
        time.sleep(0.01)
    with open_live(c, s["session_id"]) as ws2:
        L2 = Live(ws2)
        ready, state = L2.hello(s["token"], resume_rev=rev)
        assert state["core_mode"] == "sequential" and state["phase"] == "reviewing"
        assert state["plan"]["status"] == "draft"
        assert all(m.get("core_mode", "sequential") == "sequential"
                   for m in L2.seen if m["type"] == "state")


def test_core_mode_defaults_to_classic_and_an_unknown_value_is_rejected(make_client):
    """§15.8: the core set is closed — an unknown value is an `invalid_message` before any run; omitted is
    `classic`, the core master already executed."""
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        engine = c.app.state.store.sessions[s["session_id"]].engine
        L.send({**START, "core_mode": "quantum"})
        err = L.until_type("error")
        assert err["code"] == "invalid_message" and err["retryable"] is False
        assert engine.phase == "idle"
        L.send(START)
        st = L.drive(lambda m: m["phase"] == "running")
        assert st["core_mode"] == "classic" and engine.core_mode == "classic"


# ------------------------------------------------------------------ draft edits


def test_plan_edit_bumps_draft_revision(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        before_state = reviewing(L)
        before = before_state["plan"]
        L.send({"type": "plan_edit", "edits": [{"step_id": "s1", "say": "상자를 넓게 펼치세요"}]})
        st = L.until(is_state(lambda m: m["plan"]["revision"] == 2))
        assert st["phase"] == "reviewing" and st["plan"]["status"] == "draft"
        assert st["plan"]["approved_revision"] is None and st["plan"]["plan_id"] == before["plan_id"]
        assert plan_ids(st) == plan_ids(before_state)
        assert st["plan"]["steps"][0]["say"] == "상자를 넓게 펼치세요"
        L.until(lambda m: m["type"] == "say" and m["text"] == "계획을 고쳤어요.")


def test_plan_edit_rejects_bad_edits(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        reviewing(L)
        L.send({"type": "plan_edit", "edits": [{"step_id": "s9", "say": "없는 단계"}]})
        err = L.until_type("error")
        assert err["code"] == "invalid_edit" and err["fatal"] is False
        L.send({"type": "plan_edit", "edits": [{"step_id": "s2", "requires": ["s3"]},
                                               {"step_id": "s3", "requires": ["s2"]}]})
        assert L.until_type("error")["code"] == "invalid_edit"
        assert L.state["phase"] == "reviewing" and L.state["plan"]["revision"] == 1
        # the draft is still editable after a rejection
        L.send({"type": "plan_edit", "edits": [{"step_id": "s2", "say": "완충재를 두 겹으로 까세요"}]})
        st = L.until(is_state(lambda m: m["plan"]["revision"] == 2))
        assert st["plan"]["steps"][1]["say"] == "완충재를 두 겹으로 까세요"


# ------------------------------------------------------------------ approval


def test_plan_approve_starts_running(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        rev = reviewing(L)["plan"]["revision"]
        L.send({"type": "plan_approve"})
        st = L.until(is_state(lambda m: m["phase"] == "running"))
        assert st["plan"]["status"] == "approved" and st["plan"]["approved_revision"] == rev
        assert st["step_index"] == 0 and st["blocked"] is None and st["proposal"] is None
        L.seen_or_next(lambda m: m["type"] == "say" and m["text"] == "1단계. 상자를 펼쳐 바닥에 놓으세요")
        states = []
        for _ in range(60):
            states.append(L.frame()["state"])
            if states[-1] == "tracking" and L.state["overlay"] is not None:
                break
            time.sleep(0.002)
        assert states[0] == "acquiring"  # approved execution re-acquires the target
        assert L.state["overlay"]["motion_key"].split("|")[2] == "s1"


def test_plan_control_gating(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        for msg in ({"type": "plan_approve"}, {"type": "plan_discard"},
                    {"type": "plan_edit", "edits": [{"step_id": "s1", "say": "x"}]}):
            L.send(msg)
            assert L.until_type("error")["code"] == "not_reviewing", msg["type"]
        approved_running(L)
        L.send({"type": "plan_edit", "edits": [{"step_id": "s1", "say": "고치기"}]})
        assert L.until_type("error")["code"] == "plan_locked"
        L.send({"type": "plan_approve"})
        assert L.until_type("error")["code"] == "not_reviewing"


def test_plan_discard_resets_to_idle(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        reviewing(L)
        L.send({"type": "plan_discard"})
        st = L.until(is_state(lambda m: m["phase"] == "idle"))
        assert st["plan"] is None and st["overlay"] is None and st["pending"] is None
        assert L.frame()["state"] == "idle"
        L.send({"type": "plan_approve"})
        assert L.until_type("error")["code"] == "not_reviewing"


# ------------------------------------------------------------------ progression / mandatory check


def test_required_step_is_never_skipped_by_the_timer(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        st = L.drive(lambda m: m["blocked"] is not None)
        assert st["step_index"] == 6 and plan_ids(st)[6] == REQUIRED_STEP
        assert st["plan"]["steps"][6]["required"] is True
        assert st["blocked"]["step_id"] == REQUIRED_STEP and "필수" in st["blocked"]["reason"]
        assert st["phase"] == "running" and st["completion"] == "none"
        notice = L.seen_or_next(is_state(lambda m: m["notice"] is not None and m["notice"]["code"] == "required_check"))
        assert notice["notice"]["text"] == st["blocked"]["reason"]
        assert notice["blocked"]["step_id"] == REQUIRED_STEP
        # one more frame's worth of tracking cannot pass it: the timer may not confirm a `check:"user"` step
        L.frame()
        assert L.state["step_index"] == 6 and L.state["blocked"]["step_id"] == REQUIRED_STEP
        assert L.state["completion"] == "none"
        # the user's own confirmation moves it on
        L.send({"type": "step_ack", "step_id": REQUIRED_STEP})
        st = L.until(is_state(lambda m: m["completion"] == "checking"))
        assert st["blocked"] is None and st["step_index"] == 6
        L.until(is_state(lambda m: m["completion"] == "confirmed"))
        L.send({"type": "confirm_done"})
        assert L.until(is_state(lambda m: m["phase"] == "completed"))["completion"] == "user_confirmed"


def test_completion_refused_while_required_step_unfinished(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        st = L.drive(lambda m: m["blocked"] is not None)
        assert st["blocked"]["step_id"] == REQUIRED_STEP
        for _ in range(120):
            L.frame()
        completions = {m["completion"] for m in L.seen if m["type"] == "state"}
        assert "checking" not in completions and "confirmed" not in completions
        assert L.state["phase"] == "running" and L.state["plan"]["status"] == "approved"
        assert L.state["step_index"] == 6 and L.state["blocked"]["step_id"] == REQUIRED_STEP


def test_step_ack_on_visual_step_is_rejected(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        L.send({"type": "step_ack", "step_id": "s1"})
        err = L.until_type("error")
        assert err["code"] == "invalid_edit" and err["fatal"] is False
        L.send({"type": "step_ack", "step_id": "s9"})
        assert L.until_type("error")["code"] == "invalid_edit"
        assert L.state["step_index"] == 0 and L.state["completion"] == "none"


def test_unmet_prerequisite_blocks_the_next_step(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        reviewing(L)
        L.send({"type": "plan_edit", "edits": [{"step_id": "s2", "requires": ["s5"]}]})
        st = L.until(is_state(lambda m: m["plan"]["revision"] == 2))
        assert st["plan"]["steps"][1]["requires"] == ["s5"]
        L.send({"type": "plan_approve"})
        L.until(is_state(lambda m: m["phase"] == "running"))
        st = L.drive(lambda m: m["blocked"] is not None)
        assert st["step_index"] == 0
        assert st["blocked"]["step_id"] == "s2" and st["blocked"]["requires"] == ["s5"]
        for _ in range(20):
            L.frame()
        assert L.state["step_index"] == 0 and L.state["phase"] == "running"
        assert L.state["completion"] == "none" and L.state["blocked"]["requires"] == ["s5"]


# ------------------------------------------------------------------ pause / resume


def test_run_pause_stops_judgement_until_resume(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        L.send({"type": "run_pause"})
        st = L.until(is_state(lambda m: m["paused"] is True))
        assert st["phase"] == "running"
        assert L.frame()["state"] == "idle" and L.frame()["state"] == "idle"
        for msg in ({"type": "follow_now"}, {"type": "step_ack", "step_id": "s1"},
                    {"type": "talk", "utterance": "안녕"}, {"type": "confirm_done"}):
            L.send(msg)
            assert L.until_type("error")["code"] == "not_running", msg["type"]
        L.send({"type": "run_resume"})
        st = L.until(is_state(lambda m: m["paused"] is False))
        assert st["phase"] == "running"
        assert L.frame()["state"] == "acquiring"  # resume re-acquires the target


def test_user_pause_survives_reconnect(make_client):
    c = make_client(resume_grace_s=5.0)
    s, cm = connect(c)
    session = c.app.state.store.sessions[s["session_id"]]
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        L.send({"type": "run_pause"})
        L.until(is_state(lambda m: m["paused"] is True))
        rev = L.state["rev"]
    deadline = time.time() + 2
    while session.conn is not None and time.time() < deadline:
        time.sleep(0.01)
    with open_live(c, s["session_id"]) as ws2:
        L2 = Live(ws2)
        ready, state = L2.hello(s["token"], resume_rev=rev)
        assert state["phase"] == "running" and state["paused"] is True  # a reconnect must not resume the run
        assert L2.frame()["state"] == "idle"
        L2.send({"type": "run_resume"})
        assert L2.until(is_state(lambda m: m["paused"] is False))["phase"] == "running"
        assert L2.frame()["state"] == "acquiring"


# ------------------------------------------------------------------ proposal (§15.6)


def test_proposal_accept_installs_the_change(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        st = L.drive(lambda m: m["proposal"] is not None, max_frames=400)
        proposal, plan = st["proposal"], st["plan"]
        assert proposal["plan_id"] == plan["plan_id"] and proposal["revision"] == plan["revision"] + 1
        assert "봉함" in proposal["reason"]
        assert len(proposal["steps"]) > len(plan["steps"])
        assert st["phase"] == "running"
        # §15.6: nothing moves while the proposal waits for an answer
        for _ in range(20):
            L.frame()
        assert L.state["step_index"] == st["step_index"] and L.state["plan"]["revision"] == 1
        assert L.state["plan"]["plan_id"] == plan["plan_id"] and L.state["completion"] == "none"
        assert L.state["proposal"]["revision"] == 2
        L.send({"type": "proposal_accept"})
        st = L.until(is_state(lambda m: m["proposal"] is None and m["plan"]["revision"] == 2))
        assert st["phase"] == "running" and st["plan"]["status"] == "approved"
        assert st["plan"]["approved_revision"] == 2 and st["blocked"] is None
        assert plan_ids(st) == [f"s{i + 1}" for i in range(8)]
        # the run continues: the mandatory check still needs the user's confirmation
        L.send({"type": "step_ack", "step_id": REQUIRED_STEP})
        assert L.drive(lambda m: m["completion"] == "confirmed")["completion"] == "confirmed"


def test_proposal_reject_keeps_the_installed_plan(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        st = L.drive(lambda m: m["proposal"] is not None, max_frames=400)
        plan = st["plan"]
        L.send({"type": "proposal_reject"})
        st = L.until(is_state(lambda m: m["proposal"] is None))
        assert st["plan"]["plan_id"] == plan["plan_id"] and st["plan"]["revision"] == plan["revision"]
        assert st["plan"]["status"] == "approved" and st["blocked"] is not None
        assert L.state["step_index"] == 6 and L.state["completion"] == "none"
        # execution still moves only through the user's own confirmation
        L.send({"type": "step_ack", "step_id": REQUIRED_STEP})
        assert L.until(is_state(lambda m: m["completion"] == "checking"))["blocked"] is None


def test_proposal_accept_without_proposal(make_client):
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        for kind in ("proposal_accept", "proposal_reject"):
            L.send({"type": kind})
            err = L.until_type("error")
            assert err["code"] == "no_proposal" and err["fatal"] is False, kind
        assert L.state["phase"] == "running" and L.state["proposal"] is None


# ------------------------------------------------------------------ ws 계층이 직접 막는 규칙 (§15.4.5)


def test_confirm_done_with_a_required_step_open_is_refused(make_client):
    """완료 확인은 필수 검사가 다 끝난 뒤에만. 남아 있으면 `error{required_check}`로 거절한다."""
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        approved_running(L)
        L.send({"type": "confirm_done"})
        err = L.until_type("error")
        assert err["code"] == "required_check"
        assert REQUIRED_STEP in err["message"]
        assert L.state["phase"] == "running" and L.state["completion"] == "none"


# ------------------------------------------------------------------ the question round trip (§15)


def test_plan_mode_asks_one_question_then_reviews_the_draft(make_client):
    """§15: a goal phrased as a question gets one easy question first; the answer writes the draft, the run
    keeps its goal, and nothing is tracked until the explicit approval."""
    c = make_client()
    s, cm = connect(c)
    with cm as ws:
        L = Live(ws)
        L.hello(s["token"])
        L.send({**PLAN_START, "goal": "상자 포장을 어떻게 할까?"})
        st = L.drive(lambda x: x["phase"] == "clarifying")
        StateMsg.model_validate(st)
        assert st["plan"] is None and st["clarification"] and st["clarification_id"]
        assert st["research_sources"] == [] and st["overlay"] is None and st["pending"] is None
        cid = st["clarification_id"]
        engine = c.app.state.store.sessions[s["session_id"]].engine
        # a question is not permission to execute: every frame is idle while it waits
        assert {t["state"] for t in [L.frame() for _ in range(4)]} == {"idle"}
        L.seen_or_next(lambda m: m["type"] == "say" and m["text"] == st["clarification"])
        # a question that is not the current one is refused, and the current one still works
        L.send({"type": "plan_answer", "clarification_id": "q-nope", "answer": "갈색 상자"})
        err = L.until_type("error")
        assert err["code"] == "clarification_stale" and err["fatal"] is False
        assert L.state["phase"] == "clarifying" and L.state["clarification_id"] == cid
        L.send({"type": "plan_answer", "clarification_id": cid, "answer": "책상 위 갈색 상자",
                "reference_images": [{"frame_id": "ref-1", "label": "상자 사진",
                                      "image_base64": base64.b64encode(jpeg()).decode("ascii")}]})
        approved_ready = L.drive(lambda x: x["phase"] == "reviewing")
        assert approved_ready["plan"]["status"] == "draft"
        assert approved_ready["plan"]["steps"][0]["details"]  # readable instructions, not just the spoken say
        assert [a.answer for a in engine.answers] == ["책상 위 갈색 상자"]
        assert [r.frame_id for r in engine.reference_images] == ["ref-1"]
        assert "image_base64" not in json.dumps(approved_ready)
        # still nothing runs, even with a valid draft
        assert {t["state"] for t in [L.frame() for _ in range(3)]} == {"idle"}
        L.send({"type": "plan_approve"})
        running = L.drive(lambda x: x["phase"] == "running" and x["overlay"] is not None)
        assert running["plan"]["status"] == "approved" and running["plan"]["steps"][0]["details"]

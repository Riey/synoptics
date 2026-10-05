"""The pure ports in guide_policy.py, checked against the TS rules they cite."""

from __future__ import annotations

from types import SimpleNamespace as NS

from synoptics_live import guide_policy as gp
from synoptics_live.contracts import Box

STEPS = [NS(id="s1", done_when="a", say="one"), NS(id="s2", done_when="b", say="two"),
         NS(id="s3", done_when="c", say="three")]


def checks(**visible):
    return [{"step_id": k, "visible": v} for k, v in visible.items()]


def test_follow_step_rules():
    d = gp.decide_follow_step(STEPS, 0, 0, checks(s1="yes", s2="no", s3="no"), None)
    assert (d.kind, d.from_index, d.to_index, d.step_id) == ("advance", 0, 1, "s1")
    assert gp.decide_follow_step(STEPS, 0, 1, checks(s1="yes"), None).kind == "none"  # screen moved meanwhile
    assert gp.decide_follow_step(STEPS, 0, 0, checks(s1="no", s2="yes", s3="no"), None).kind == "pendingSkip"
    d = gp.decide_follow_step(STEPS, 0, 0, checks(s1="no", s2="yes", s3="no"), checks(s1="no", s2="yes", s3="no"))
    assert (d.kind, d.to_index, d.step_id, d.skipped_ids) == ("skip", 2, "s2", ("s1",))
    assert gp.decide_follow_step(STEPS, 2, 2, checks(s3="yes"), None).kind == "none"  # last step: only goal_check
    assert gp.decide_follow_step(STEPS, 0, 0, checks(s1="unsure", s2="no", s3="no"), None).kind == "none"


def test_confirm_decision_step_and_completion_apart():
    meta = {"from_index": 0, "to_index": 1, "step_id": "s1"}
    ans = {"trigger": "step_done", "step_id": "s1", "step_check": "no",
           "goal_status": {"status": "visually_satisfied"}}
    d = gp.decide_confirm(ans, meta, step_index=1, completion="none", track_lost=False)
    assert d.revert_to == 0 and d.completion == "start_check"
    assert gp.decide_confirm(ans, meta, step_index=1, completion="none", track_lost=False,
                             user_done_ids=["s1"]).revert_to is None
    recheck = gp.decide_confirm({"trigger": "goal_check", "goal_status": {"status": "in_progress"}}, {"recheck": True},
                                step_index=1, completion="checking", track_lost=False)
    assert recheck.completion == "recheck_failed"
    lost = gp.decide_confirm({"trigger": "goal_check", "goal_status": {"status": "in_progress"}}, {},
                             step_index=0, completion="none", track_lost=True)
    assert lost.lost_notice and lost.completion == "keep"


def test_gate_priority_floor_spacing_and_budget():
    cfg = gp.GATE_CONFIG
    s = gp.INITIAL_GATE_STATE
    hb = gp.PendingCall("follow", "heartbeat", "remote")
    manual = gp.PendingCall("follow", "manual", "remote")
    s = gp.enqueue(s, manual)
    assert gp.enqueue(s, hb).follow.pending is manual  # heartbeat never displaces a more important call
    goal = gp.PendingCall("confirm", "goal_check", "remote")
    recheck = gp.PendingCall("confirm", "goal_check", "remote", {"recheck": True})
    s = gp.enqueue(gp.enqueue(s, recheck), goal)
    assert s.confirm.pending is recheck
    d = gp.decide_dispatch(s, "follow", 0, cfg)
    assert d.kind == "dispatch"
    s = gp.mark_dispatched(s, manual, 0, cfg)
    assert gp.decide_dispatch(s, "confirm", 500, cfg).kind == "wait"  # 1.2 s remote spacing
    assert gp.decide_dispatch(s, "confirm", 500, cfg).wait_ms == 700
    s = gp.mark_settled(s, "follow")
    s = gp.enqueue(s, gp.PendingCall("follow", "manual", "remote"))
    assert gp.decide_dispatch(s, "follow", 1500, cfg).wait_ms == 700  # 2.2 s follow floor
    # the rolling budget: 30 DeepSeek-backed calls per 120 s
    full = gp.GateState(remote_times=tuple(float(i * 3000) for i in range(30)), last_remote_at=87000)
    full = gp.enqueue(full, gp.PendingCall("confirm", "goal_check", "remote"))
    d = gp.decide_dispatch(full, "confirm", 90_000, cfg)
    assert d.kind == "capped" and d.limit == "budget" and d.wait_ms == 30_000
    # the per-minute mirror: 20 in 60 s
    minute = gp.GateState(remote_times=tuple(float(i * 2500) for i in range(20)), last_remote_at=47500)
    minute = gp.enqueue(minute, gp.PendingCall("confirm", "goal_check", "remote"))
    assert gp.decide_dispatch(minute, "confirm", 50_000, cfg).limit == "per_minute"
    # local follows are free of the remote budget
    local = gp.enqueue(full, gp.PendingCall("follow", "heartbeat", "local"))
    assert gp.decide_dispatch(local, "follow", 90_000, cfg).kind == "dispatch"
    # a talk waiting holds remote calls
    assert gp.decide_dispatch(gp.enqueue(gp.INITIAL_GATE_STATE, manual), "follow", 0, cfg, True).kind == "busy"


def test_triggers_acquired_moved_lost_heartbeat():
    box = Box(x=0.1, y=0.1, width=0.2, height=0.2)
    obs = gp.TrackObservation("r", "t", 1, "tracking", box)
    st, fired = gp.step_triggers(gp.INITIAL_TRIGGER_STATE, observation=obs, now_ms=0, fence_key="k", accepted=None,
                                 follow_provider="local")
    assert fired == ["acquired"]
    st, fired = gp.step_triggers(st, observation=obs, now_ms=4000, fence_key="k", accepted=None,
                                 follow_provider="local")
    assert fired == ["heartbeat"]
    accepted = gp.AcceptedBox("r", 1, box)
    moved = gp.TrackObservation("r", "t", 1, "tracking", Box(x=0.5, y=0.1, width=0.2, height=0.2))
    st, fired = gp.step_triggers(st, observation=moved, now_ms=4100, fence_key="k", accepted=accepted,
                                 follow_provider="local")
    assert fired == []
    st, fired = gp.step_triggers(st, observation=moved, now_ms=4400, fence_key="k", accepted=accepted,
                                 follow_provider="local")
    assert fired == ["target_moved"]
    lost = gp.TrackObservation("r", "t", 1, "lost", None)
    st, fired = gp.step_triggers(st, observation=lost, now_ms=5000, fence_key="k", accepted=accepted,
                                 follow_provider="local")
    st, fired = gp.step_triggers(st, observation=lost, now_ms=6000, fence_key="k", accepted=accepted,
                                 follow_provider="local")
    assert fired == ["anchor_lost"]


def test_acceptance_bound_text_only_rejected():
    stamp = gp.SentStamp("sess", "p", 1, 5, {"task_epoch": "e", "run_id": "r"},
                         ({"anchor_id": "a1", "track_id": "t", "generation": 1},))
    answer = {"intent_seq": 5, "plan_revision": 1, "fence_echo": {"task_epoch": "e", "run_id": "r"},
              "anchors_echo": [{"anchor_id": "a1", "track_id": "t", "generation": 1}]}
    cur = dict(session_id="sess", task_epoch="e", run_id="r", plan_id="p", plan_revision=1, latest_intent_seq=5,
               anchor={"anchor_id": "a1", "track_id": "t", "generation": 1})
    assert gp.classify_answer(stamp, answer, **cur) == "bound"
    assert gp.classify_answer(stamp, answer, **{**cur, "anchor": {"anchor_id": "a1", "track_id": "t", "generation": 2}}) \
        == "text_only"
    assert gp.classify_answer(stamp, answer, **{**cur, "plan_revision": 2}) == "rejected"
    assert gp.classify_answer(stamp, answer, **{**cur, "latest_intent_seq": 6}) == "rejected"
    assert gp.classify_answer(stamp, {**answer, "plan_revision": 2}, **cur, replan_applied=True) == "bound"


def test_replan_budget_and_talk_gate():
    b = gp.fresh_replan_budget(0)
    assert gp.replan_block(b, 1000) == ("cooldown", 14000)
    assert gp.replan_block(b, 15000)[0] is None
    b = gp.spend_replan(gp.spend_replan(gp.spend_replan(b, 15000), 30000), 45000)
    assert gp.replan_block(b, 99999)[0] == "max"
    assert gp.reserve_replan(b) == (b, False)
    s = gp.mark_talk(gp.INITIAL_GATE_STATE, 0)
    assert gp.decide_talk(running=True, talk_pending=False, gate=s, now_ms=1000) == (None, 1200)  # confirm floor 2.2 s
    assert gp.decide_talk(running=True, talk_pending=True, gate=s, now_ms=1000)[0] == "pending"
    assert gp.is_superseded_stale({"plan_id": "p", "plan_revision": 2}, plan_id="p", plan_revision=2, talk_pending=False)
    assert not gp.is_superseded_stale({"plan_id": "q", "plan_revision": 2}, plan_id="p", plan_revision=2,
                                      talk_pending=False)


def test_speech_lines_by_view_diff():
    planning = gp.SpeechView(phase="planning")
    assert gp.spoken_lines(None, planning) == [(gp.PLANNING_SPEECH, "replace")]
    running = gp.SpeechView(phase="running", step="1단계 / 2. 잡으세요")
    assert gp.spoken_lines(planning, running) == [("1단계 / 2. 잡으세요", "replace")]
    assert gp.spoken_lines(running, running) == []
    noticed = gp.SpeechView(phase="running", step="1단계 / 2. 잡으세요", notice="계획이 다시 짜였습니다.",
                            needs_reselect=True)
    assert gp.spoken_lines(running, noticed) == [(gp.RESELECT_SPEECH, "append"), ("계획이 다시 짜였습니다.", "append")]
    done = gp.SpeechView(phase="running", step="1단계 / 2. 잡으세요", completion="confirmed")
    assert gp.spoken_lines(running, done) == [(gp.COMPLETION_SPEECH, "append")]
    assert gp.talk_speech("긴 답입니다. 둘째 문장.", "긴 답입니다.") == "긴 답입니다. 자세한 건 화면에 있어요."
    assert gp.step_speech("running", STEPS, 1, None) == "2단계 / 3. two"


def test_apply_talk_order_and_marks():
    state = gp.TalkRunState(steps=tuple(STEPS), step_index=1, plan_revision=3, target="glasses", skipped=(),
                            user_done=(), unsure_streak=1, stuck_count=2, last_checks=(), replan_budget=gp.ReplanBudget())
    out = gp.apply_talk(state, "s2", {"step_mark": "done", "target": "case", "plan_revision": 3}, "text_only", 0,
                        lambda s: s)
    assert out.applied and out.state.step_index == 2 and out.state.user_done == ("s2",)
    assert out.effects == (("retarget", "case"),) and out.actions == ("step_mark", "target")
    last = gp.apply_talk(gp.TalkRunState(**{**state.__dict__, "step_index": 2}), "s3",
                         {"step_mark": "done", "plan_revision": 3}, "bound", 0, lambda s: s)
    assert last.state.step_index == 2 and last.effects == (("checkGoal", None),)
    back = gp.apply_talk(state, "s2", {"go_to": "s1", "plan_revision": 3}, "bound", 0, lambda s: s)
    assert back.state.step_index == 0 and back.state.last_checks is None
    assert not gp.apply_talk(state, "s2", {"go_to": "s1", "plan_revision": 3}, "rejected", 0, lambda s: s).applied

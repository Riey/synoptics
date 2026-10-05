"""RealEngine against an in-process fake of the demo backend, with an injected clock (no sockets, no sleeps)."""

from __future__ import annotations

import asyncio
import base64
import io


import httpx
from PIL import Image

from conftest import jpeg
from fake_upstream import FakeUpstream
from synoptics_live.contracts import Box, FrameHeader
from synoptics_live.engine import EngineReject
from synoptics_live.real_engine import RealConfig, RealContext, RealEngine


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeSink:
    def __init__(self) -> None:
        self.states: list = []
        self.says: list[tuple[str, str]] = []
        self.his = 0

    def state(self, state) -> None:
        self.states.append(state)

    def say(self, text: str, mode: str) -> str:
        self.says.append((text, mode))
        return f"L{len(self.says)}"

    def capture_hi(self) -> str:
        self.his += 1
        return f"h{self.his}"

    @property
    def last(self):
        return self.states[-1]


def colored_jpeg(rgb=(200, 40, 40), w=640, h=360) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), rgb).save(buf, "JPEG", quality=80)
    return buf.getvalue()


class H:
    """Harness: one engine, one fake upstream, one clock."""

    def __init__(self, fake: FakeUpstream | None = None):
        self.fake = fake or FakeUpstream()
        self.sink = FakeSink()
        self.clock = FakeClock()
        ctx = RealContext(RealConfig(upstream_url=self.fake.base_url), transport=self.fake.transport())
        self.engine = RealEngine(self.sink, ctx, clock=self.clock, auto_tick=False)
        self.seq = 0

    async def frame(self, payload: bytes | None = None, select_box: dict | None = None, w=640, h=360):
        header = {"t": "frame", "seq": self.seq, "captured_at": 1_759_479_000_000 + self.seq, "w": w, "h": h}
        if select_box is not None:
            header["select_box"] = select_box
        self.seq += 1
        return await self.engine.on_frame(FrameHeader.model_validate(header), payload or jpeg(w, h))

    async def frames(self, n: int, ms: float = 100, payload: bytes | None = None):
        out = []
        for _ in range(n):
            out.append(await self.frame(payload))
            await self.engine.advance(ms)
        return out

    async def start(self, goal="안경을 벗어 주세요", hi: bytes | None = b"") -> None:
        await self.frame()
        await self.engine.on_start(goal, None)
        await self.engine.settle()
        if hi is not None:
            payload = hi or jpeg(1024, 576)
            await self.engine.on_hi_frame(FrameHeader.model_validate(
                {"t": "frame", "seq": 999, "captured_at": 1, "w": 1024, "h": 576, "hi_req": f"h{self.sink.his}"}),
                payload)
        for _ in range(30):
            await self.engine.advance(100)
            if self.sink.last.phase != "planning":
                break

    async def running_and_tracking(self) -> None:
        await self.start()
        assert self.sink.last.phase == "running"
        await self.frames(3)

    def phase(self) -> str:
        return self.sink.last.phase


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------------------------- plan


def test_plan_sse_partials_then_running_and_tracking_starts():
    async def go():
        h = H()
        hi = jpeg(1024, 576)
        await h.start(hi=hi)
        phases = [s.phase for s in h.sink.states]
        assert phases[0] == "planning" and h.phase() == "running"
        partials = [s.partial for s in h.sink.states if s.partial is not None]
        assert partials[0].target == "eyeglasses" and partials[-1].first_say == "안경다리를 양손으로 잡으세요"
        st = h.sink.last
        assert st.plan.target == "eyeglasses" and [s.id for s in st.plan.steps] == ["s1", "s2", "s3"]
        assert st.partial is None and st.pending is None and st.overlay is None  # not tracking yet
        assert h.sink.says[0] == ("계획을 세우는 중입니다.", "replace")
        assert h.sink.says[1] == ("1단계 / 3. 안경다리를 양손으로 잡으세요", "replace")
        plan_req = h.fake.calls("/api/guide/plan")[0]
        assert base64.b64decode(plan_req["scene"]["image_base64"]) == hi  # the capture_hi answer was used
        assert plan_req["user_goal"] == "안경을 벗어 주세요" and plan_req["consent_ai"] is True
        control = h.fake.calls("/api/track/control")[-1]
        assert control["action"] == "start" and control["target"] == "eyeglasses" and control["start_seq"] == 1
        await h.engine.close()
        assert h.fake.calls("/api/session/end")

    run(go())


def test_plan_uses_latest_stream_frame_when_no_hi_frame_within_2s():
    async def go():
        h = H()
        stream = colored_jpeg((10, 200, 10))
        await h.frame(stream)
        await h.engine.on_start("안경을 벗어 주세요", None)
        await h.engine.advance(1900)
        assert not h.fake.calls("/api/guide/plan")
        await h.engine.advance(300)
        assert base64.b64decode(h.fake.calls("/api/guide/plan")[0]["scene"]["image_base64"]) == stream
        assert h.phase() == "running"

    run(go())


def test_plan_clarification_goes_to_state_clarification():
    async def go():
        h = H(FakeUpstream(plan_clarification="안경이 화면에 보이지 않아요. 얼굴을 비춰 주세요."))
        await h.start()
        st = h.sink.last
        assert st.phase == "idle" and st.plan is None and st.clarification.startswith("안경이 화면에")
        assert ("안경이 화면에 보이지 않아요. 얼굴을 비춰 주세요.", "append") in h.sink.says
        assert (await h.frame()).state == "idle"

    run(go())


def test_start_during_planning_is_plan_in_flight():
    async def go():
        h = H()
        await h.frame()
        await h.engine.on_start("a", None)
        try:
            await h.engine.on_start("b", None)
        except EngineReject as rej:
            assert rej.code == "plan_in_flight"
        else:
            raise AssertionError("expected plan_in_flight")

    run(go())


# ---------------------------------------------------------------------------------------------- tracking


def test_track_forwarding_box_and_frame_seq():
    async def go():
        h = H()
        assert (await h.frame()).state == "idle"  # no guide: no upstream call
        assert not h.fake.calls("/api/track/frame")
        await h.start()
        t1, t2 = await h.frame(), await h.frame()
        assert t1.state == "acquiring" and t1.box is None
        assert t2.state == "tracking" and t2.box == Box(**h.fake.box)
        assert t2.run_id == t1.run_id and t2.target == "eyeglasses" and t2.seq == h.seq - 1
        assert t2.captured_at == 1_759_479_000_000 + t2.seq
        assert [c["frame_seq"] for c in h.fake.calls("/api/track/frame")] == [1, 2]

    run(go())


def test_acquired_follow_uses_the_same_frame_as_its_anchor_box_and_binds_overlay():
    async def go():
        h = H()
        await h.start()
        await h.frame()
        img = colored_jpeg((30, 30, 220))
        tr = await h.frame(img)  # first tracking frame -> acquired
        await h.engine.settle()
        follow = h.fake.calls("/api/guide/follow")[0]
        assert follow["trigger"] == "acquired" and follow["current_step"] == "s1"
        assert base64.b64decode(follow["scene"]["image_base64"]) == img  # scene == the pair's capture
        anchor = follow["anchors"][0]
        assert anchor["box"] == tr.box.model_dump() and anchor["run_id"] == tr.run_id == follow["fence"]["run_id"]
        assert anchor["track_id"] == tr.track_id and anchor["state"] == "tracking" and anchor["anchor_id"] == "a1"
        ov = h.sink.last.overlay
        assert ov.binding.run_id == tr.run_id and ov.binding.track_id == tr.track_id
        assert [c.kind for c in ov.commands] == ["focus", "action"] and ov.warn is None

    run(go())


def test_lost_fires_goal_check_after_1s_and_run_is_terminal():
    async def go():
        h = H()
        await h.running_and_tracking()
        n_frames = len(h.fake.calls("/api/track/frame"))
        h.fake.track_script.append(("lost", None))
        tr = await h.frame()
        assert tr.state == "lost" and tr.box is None
        await h.engine.settle()
        assert h.fake.calls("/api/track/control")[-1]["action"] == "stop"
        await h.engine.advance(900)
        assert not [c for c in h.fake.calls("/api/guide/confirm") if c["trigger"] == "goal_check"]
        await h.engine.advance(3000)
        goal = [c for c in h.fake.calls("/api/guide/confirm") if c["trigger"] == "goal_check"]
        assert len(goal) == 1 and goal[0]["anchors"][0]["state"] == "lost" and "box" not in goal[0]["anchors"][0]
        assert (await h.frame()).state == "lost"
        assert len(h.fake.calls("/api/track/frame")) == n_frames + 1  # terminal: no more uploads
        assert h.sink.last.notice.code == "target_lost"

    run(go())


def test_select_box_reseeds_same_run_with_new_generation_and_track():
    async def go():
        h = H()
        await h.running_and_tracking()
        before = await h.frame()
        sel = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.3}
        tr = await h.frame(select_box=sel)
        assert tr.run_id == before.run_id and tr.generation == before.generation + 1
        assert tr.track_id != before.track_id and tr.box == Box(**sel) and tr.target == "user-selected object"
        frame_req = h.fake.calls("/api/track/frame")[-1]
        assert frame_req["seed_box"] == sel and frame_req["frame_seq"] == 1
        await h.engine.advance(100)
        assert h.sink.last.overlay.binding.generation == tr.generation  # acquired re-bound

    run(go())


# ---------------------------------------------------------------------------------------------- follow / confirm


async def _advance_until(h: H, pred, ms=100, limit=200, frames=True):
    for _ in range(limit):
        if pred():
            return
        if frames:
            await h.frame()
        await h.engine.advance(ms)
    raise AssertionError("condition never met")


def test_follow_yes_advances_and_step_done_confirm_rechecks_the_follow_frame():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.append({"visible": {"s1": "yes"}})
        h = H(fake)
        await h.running_and_tracking()
        await h.engine.settle()
        assert h.sink.last.step_index == 1
        assert ("2단계 / 3. 안경을 앞으로 당겨 벗으세요", "replace") in h.sink.says
        await _advance_until(h, lambda: fake.calls("/api/guide/confirm"))
        confirm = fake.calls("/api/guide/confirm")[0]
        follow = fake.calls("/api/guide/follow")[0]
        assert confirm["trigger"] == "step_done" and confirm["current_step"] == "s1"
        assert confirm["scene"]["frame_id"] == follow["scene"]["frame_id"]  # confirmDecision: the follow's frame
        assert confirm["anchors"] == follow["anchors"]
        assert h.sink.last.step_index == 1  # step_check yes: stays

    run(go())


def test_confirm_step_check_no_reverts():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.append({"visible": {"s1": "yes"}})
        fake.confirm_answers.append({"step_check": "no"})
        h = H(fake)
        await h.running_and_tracking()
        await _advance_until(h, lambda: fake.calls("/api/guide/confirm") and h.sink.last.step_index == 0)
        assert h.sink.last.notice.code == "reverted"
        assert h.engine.stats["confirm_reversals"] == 1

    run(go())


def test_skip_needs_a_second_sighting():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.extend([{"visible": {"s2": "yes"}}, {"visible": {"s2": "yes"}}])
        h = H(fake)
        await h.running_and_tracking()
        assert h.sink.last.step_index == 0 and h.engine.stats["follow_pending_skip"] == 1
        await h.engine.on_follow_now()
        await _advance_until(h, lambda: h.sink.last.step_index == 2)
        assert h.sink.last.steps_skipped == ["s1"]

    run(go())


def test_goal_seen_goal_check_then_recheck_then_confirmed_and_confirm_done():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.append({"goal_seen": "yes"})
        fake.confirm_answers.extend([{"status": "visually_satisfied"}, {"status": "visually_satisfied"}])
        h = H(fake)
        await h.running_and_tracking()
        await _advance_until(h, lambda: h.sink.last.completion == "checking")
        first = fake.calls("/api/guide/confirm")[0]
        assert first["trigger"] == "goal_check"
        assert first["scene"]["frame_id"] == fake.calls("/api/guide/follow")[0]["scene"]["frame_id"]
        n = len(fake.calls("/api/guide/confirm"))
        await h.frame()
        await h.engine.advance(1400)
        assert len(fake.calls("/api/guide/confirm")) == n  # not before 1.5 s
        await _advance_until(h, lambda: h.sink.last.completion == "confirmed")
        recheck = fake.calls("/api/guide/confirm")[-1]
        assert recheck["trigger"] == "goal_check" and recheck["scene"]["frame_id"] != first["scene"]["frame_id"]
        st = h.sink.last
        assert st.phase == "running" and st.overlay is None and st.pending is None
        assert ("목표를 달성한 것으로 보입니다. 완료 확인을 눌러 주세요.", "append") in h.sink.says
        calls = len(fake.requests)
        await h.frames(30)  # halted: no follow/confirm any more
        assert not [p for p, _ in fake.requests[calls:] if p.startswith("/api/guide")]
        await h.engine.on_confirm_done()
        assert h.sink.last.phase == "completed" and h.sink.last.completion == "user_confirmed"
        assert h.sink.says[-1] == ("완료를 확인했습니다. 수고하셨습니다.", "replace")

    run(go())


def test_recheck_not_satisfied_resumes_guide():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.append({"goal_seen": "yes"})
        fake.confirm_answers.extend([{"status": "visually_satisfied"}, {"status": "in_progress"}])
        h = H(fake)
        await h.running_and_tracking()
        await _advance_until(h, lambda: len(fake.calls("/api/guide/confirm")) == 2 and h.sink.last.completion == "none")
        assert h.sink.last.notice.code == "recheck_failed"

    run(go())


def test_unsure_twice_asks_replan_and_installs_it():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.extend([{"visible": {"s1": "unsure"}}, {"visible": {"s1": "unsure"}}])
        new_steps = [{"id": "s1", "say": "안경을 한 손으로 잡으세요", "done_when": "한 손이 안경을 잡고 있다",
                      "commands": []}]
        fake.confirm_answers.append({"replan": new_steps})
        h = H(fake)
        await h.running_and_tracking()
        h.clock.advance(15)  # past the 15 s replan gap after the plan
        await h.engine.on_follow_now()
        await _advance_until(h, lambda: len(h.sink.last.plan.steps) == 1)
        confirm = fake.calls("/api/guide/confirm")[0]
        assert confirm["trigger"] == "unsure_twice" and confirm["follow_checks"][0]["visible"] == "unsure"
        st = h.sink.last
        assert st.plan.revision == 2 and st.step_index == 0 and st.notice.code == "replanned"
        assert ("1단계 / 1. 안경을 한 손으로 잡으세요", "replace") in h.sink.says

    run(go())


def test_replan_budget_gap_and_max():
    async def go():
        h = H()
        await h.running_and_tracking()
        await h.engine.on_replan_now()  # within 15 s of the plan: blocked
        assert h.sink.last.replan_blocked_reason.startswith("계획을 방금 짰습니다")
        assert not h.fake.calls("/api/guide/confirm")
        for i in range(3):
            h.clock.advance(16)
            await h.engine.on_replan_now()
            await _advance_until(h, lambda: len(h.fake.calls("/api/guide/confirm")) == i + 1)
        h.clock.advance(16)
        await h.engine.on_replan_now()
        await h.engine.advance(3000)
        assert len(h.fake.calls("/api/guide/confirm")) == 3
        assert "최대 3회" in h.sink.last.replan_blocked_reason

    run(go())


# ---------------------------------------------------------------------------------------------- errors


def test_409_stale_plan_superseded_is_dropped_but_foreign_plan_retires():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.running_and_tracking()
        fake.follow_answers.append(httpx.Response(409, json={"error": "stale_plan", "plan_id": fake.plan_id,
                                                             "plan_revision": fake.plan_revision}))
        await h.engine.on_follow_now()
        await h.frames(15)
        assert h.engine.stats["stale_superseded"] == 1 and h.phase() == "running"
        fake.follow_answers.append(httpx.Response(409, json={"error": "stale_plan", "plan_id": "other",
                                                             "plan_revision": 9}))
        await h.engine.on_follow_now()
        await _advance_until(h, lambda: h.phase() == "error")
        st = h.sink.last
        assert st.error.code == "stale_plan" and st.plan is not None  # keepView
        assert (await h.frame()).state == "idle"

    run(go())


def test_429_retry_after_resends_after_the_wait():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.running_and_tracking()
        n = len(fake.calls("/api/guide/follow"))
        fake.follow_answers.append(httpx.Response(429, json={"error": "rate_limited"}, headers={"Retry-After": "3"}))
        await h.engine.on_follow_now()
        await h.engine.advance(1100)
        assert len(fake.calls("/api/guide/follow")) == n + 1 and h.engine.stats["retry_after"] == 1
        await h.engine.advance(1500)
        assert len(fake.calls("/api/guide/follow")) == n + 1  # still waiting (3 s)
        await h.engine.advance(1600)
        assert len(fake.calls("/api/guide/follow")) >= n + 2
        assert fake.calls("/api/guide/follow")[n + 1]["trigger"] == "manual"

    run(go())


def test_401_retires_with_error_and_next_start_makes_a_new_session():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.running_and_tracking()
        fake.follow_answers.append(httpx.Response(401, json={"error": "session_expired"}))
        await h.engine.on_follow_now()
        await _advance_until(h, lambda: h.phase() == "error")
        assert h.sink.last.error.code == "session_expired"
        sessions = len(fake.calls("/api/session"))
        await h.start()
        assert len(fake.calls("/api/session")) == sessions + 1 and h.phase() == "running"

    run(go())


def test_three_network_failures_retire():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.running_and_tracking()
        for _ in range(3):
            fake.follow_answers.append(httpx.Response(502, json={"error": "invalid_provider_output"}))
        for _ in range(3):
            await h.engine.on_follow_now()
            await h.frames(12)
        assert h.phase() == "error" and h.sink.last.error.code == "provider_error"

    run(go())


# ---------------------------------------------------------------------------------------------- talk


async def _talk(h: H, text: str):
    await h.engine.on_talk(text)
    assert h.sink.last.talk_pending is True
    await _advance_until(h, lambda: not h.sink.last.talk_pending)


def test_talk_step_say_rewrites_sentence_and_bumps_revision():
    async def go():
        fake = FakeUpstream()
        fake.talk_answers.append({"step_say": "장갑 낀 손으로 안경다리를 잡으세요", "reply": "장갑을 끼셨군요. 천천히 잡으세요.",
                                  "spoken": "장갑을 끼셨군요."})
        h = H(fake)
        await h.running_and_tracking()
        await _talk(h, "장갑 끼고 왔어")
        st = h.sink.last
        assert st.plan.steps[0].say == "장갑 낀 손으로 안경다리를 잡으세요" and st.plan.revision == 2
        assert st.talk.reply == "장갑을 끼셨군요. 천천히 잡으세요." and st.talk.spoken == "장갑을 끼셨군요."
        req = fake.calls("/api/guide/talk")[0]
        assert req["utterance"] == "장갑 끼고 왔어" and req["current_step"] == "s1" and req["replan_allowed"] is True
        # §15: the run's planning profile is stored with the plan upstream; talk can never re-pick it.
        assert "plan_model" not in req and "plan_mode" not in req
        assert ("장갑을 끼셨군요. 자세한 건 화면에 있어요.", "replace") in h.sink.says
        assert ("1단계 / 3. 장갑 낀 손으로 안경다리를 잡으세요", "replace") in h.sink.says
        # the next follow names the bumped revision (no 409)
        await h.engine.on_follow_now()
        await h.frames(15)
        assert fake.calls("/api/guide/follow")[-1]["plan_revision"] == 2 and h.phase() == "running"

    run(go())


def test_talk_step_mark_done_advances_and_marks_user_done():
    async def go():
        fake = FakeUpstream()
        fake.talk_answers.append({"step_mark": "done"})
        h = H(fake)
        await h.running_and_tracking()
        await _talk(h, "이미 했어")
        st = h.sink.last
        assert st.step_index == 1 and st.steps_user_done == ["s1"]

    run(go())


def test_talk_go_to_and_previous_exchange():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.append({"visible": {"s1": "yes"}})
        fake.talk_answers.extend([{"reply": "첫 답"}, {"go_to": "s1", "reply": "처음부터 해요"}])
        h = H(fake)
        await h.running_and_tracking()
        await _advance_until(h, lambda: h.sink.last.step_index == 1 and fake.calls("/api/guide/confirm"))
        await _talk(h, "뭐 하면 돼?")
        await _talk(h, "처음으로 돌아가")
        assert h.sink.last.step_index == 0
        second = fake.calls("/api/guide/talk")[1]
        assert second["prev_utterance"] == "뭐 하면 돼?" and second["prev_reply"] == "첫 답"

    run(go())


def test_talk_target_restarts_tracking_with_new_noun():
    async def go():
        fake = FakeUpstream()
        fake.talk_answers.append({"target": "glasses case"})
        h = H(fake)
        await h.running_and_tracking()
        old = await h.frame()
        await _talk(h, "케이스를 봐줘")
        await h.engine.settle()
        control = fake.calls("/api/track/control")
        assert control[-1]["action"] == "start" and control[-1]["target"] == "glasses case"
        tr = await h.frame()
        assert tr.run_id != old.run_id and tr.target == "glasses case"
        assert h.sink.last.plan.target == "glasses case" and h.sink.last.overlay is None  # until re-acquired

    run(go())


def test_talk_replan_spends_a_replan_and_a_busy_talk_is_refused():
    async def go():
        fake = FakeUpstream()
        new_steps = [{"id": "s1", "say": "새 단계", "done_when": "새 상태", "commands": []},
                     {"id": "s2", "say": "새 두번째", "done_when": "두번째 상태", "commands": []}]
        fake.talk_answers.append({"replan": new_steps, "reply": "다시 짰어요"})
        h = H(fake)
        await h.running_and_tracking()
        await h.engine.on_talk("다시 짜줘")
        try:
            await h.engine.on_talk("또")
        except EngineReject as rej:
            assert rej.code == "talk_busy"
        else:
            raise AssertionError("expected talk_busy")
        await _advance_until(h, lambda: not h.sink.last.talk_pending)
        st = h.sink.last
        assert [s.say for s in st.plan.steps] == ["새 단계", "새 두번째"] and st.notice.code == "talk_replan"
        assert h.engine.guide.replan_budget.used == 1 and h.engine.guide.replan_budget.reserved == 0

    run(go())


def test_talk_waits_for_the_confirm_floor_and_holds_remote_calls():
    async def go():
        fake = FakeUpstream(follow_provider="deepseek")
        h = H(fake)
        await h.start()
        await h.engine.on_talk("안녕")  # right after the plan: spacing 1.2 s
        await h.frames(5)
        assert not fake.calls("/api/guide/talk")
        await _advance_until(h, lambda: fake.calls("/api/guide/talk"))
        # the remote follow (acquired) did not take the slot in front of the waiting talk
        follows = [i for i, (p, _) in enumerate(fake.requests) if p == "/api/guide/follow"]
        talk_at = [i for i, (p, _) in enumerate(fake.requests) if p == "/api/guide/talk"][0]
        assert not follows or follows[0] > talk_at

    run(go())


def test_appearance_change_fires_target_changed():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.start()
        red = colored_jpeg((220, 30, 30))
        for _ in range(4):
            await h.frame(red)
            await h.engine.advance(100)
        await h.engine.advance(1100)
        n = len(fake.calls("/api/guide/follow"))
        half = Image.new("RGB", (640, 360), (220, 30, 30))
        half.paste((20, 220, 20), (0, 0, 272, 360))  # a split inside the box (a uniform shift is removed)
        buf = io.BytesIO()
        half.save(buf, "JPEG", quality=80)
        changed = buf.getvalue()
        for _ in range(6):
            await h.frame(changed)
            await h.engine.advance(150)
        triggers = [c["trigger"] for c in fake.calls("/api/guide/follow")[n:]]
        assert "target_changed" in triggers

    run(go())


def test_heartbeat_local_every_4s():
    async def go():
        fake = FakeUpstream(follow_provider="local")
        h = H(fake)
        await h.running_and_tracking()
        n = len(fake.calls("/api/guide/follow"))
        await h.frames(42)  # 4.2 s
        triggers = [c["trigger"] for c in fake.calls("/api/guide/follow")[n:]]
        assert triggers == ["heartbeat"]

    run(go())


def test_pause_releases_tracker_and_resume_starts_fresh_run():
    async def go():
        h = H()
        await h.running_and_tracking()
        old = await h.frame()
        await h.engine.on_pause()
        await h.engine.settle()
        assert h.fake.calls("/api/track/control")[-1]["action"] == "stop"
        await h.engine.on_resume()
        await h.engine.settle()
        tr = await h.frame()
        assert tr.run_id != old.run_id and tr.state == "acquiring"
        assert (await h.frame()).state == "tracking"

    run(go())


def test_coalescing_keeps_one_pending_per_stage():
    async def go():
        fake = FakeUpstream(follow_provider="deepseek")
        h = H(fake)
        await h.running_and_tracking()
        eng = h.engine
        for _ in range(5):
            await eng.on_follow_now()
        gate = eng.gate
        assert gate.follow.pending is not None and gate.follow.pending.trigger == "manual"
        eng._enqueue_follow("heartbeat")  # less important than the pending manual: ignored
        assert eng.gate.follow.pending.trigger == "manual"

    run(go())





def test_tracker_start_failure_is_a_state_error():
    async def go():
        fake = FakeUpstream()
        orig = fake.handle

        async def handle(request):
            if request.url.path == "/api/track/control" and b'"start"' in request.content:
                fake.requests.append((request.url.path, {}))
                return httpx.Response(503, json={"error": "tracker_unavailable"})
            return await orig(request)

        fake.handle = handle  # type: ignore[method-assign]
        h = H(fake)
        await h.start()
        await h.engine.settle()
        st = h.sink.last
        assert st.phase == "error" and st.error.code == "tracker_unavailable" and st.plan is not None
        assert (await h.frame()).state == "idle"

    run(go())


# ---------------------------------------------------------------------------------------------- review regressions


def test_goal_check_after_lost_uses_newer_stream_frames():
    async def go():
        fake = FakeUpstream()
        fake.confirm_answers.extend([{"status": "visually_satisfied"}, {"status": "visually_satisfied"}])
        h = H(fake)
        await h.running_and_tracking()
        fake.track_script.append(("lost", None))
        lost_img = colored_jpeg((90, 90, 90))
        await h.frame(lost_img)
        later = [colored_jpeg((10 * i, 120, 200)) for i in range(1, 40)]
        i = 0
        while h.sink.last.completion != "confirmed" and i < len(later):
            await h.frame(later[i])
            await h.engine.advance(100)
            i += 1
        scenes = [c["scene"]["image_base64"] for c in fake.calls("/api/guide/confirm") if c["trigger"] == "goal_check"]
        assert h.sink.last.completion == "confirmed" and len(scenes) == 2
        assert scenes[0] != scenes[1] and base64.b64decode(scenes[0]) != lost_img

    run(go())


def test_stale_plan_at_a_newer_revision_of_our_plan_is_recovered_not_fatal():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.running_and_tracking()
        fake.plan_revision += 1  # moved upstream by an answer we never saw (same steps: a sentence rewrite)
        orig = fake.handle

        async def handle(request):
            if request.url.path == "/api/guide/plan/current":
                fake.requests.append((request.url.path, {}))
                return httpx.Response(200, json={"plan_id": fake.plan_id, "plan_revision": fake.plan_revision,
                                                 "task_revision": 1, "steps": fake.plan_steps,
                                                 "goal_when": "x", "user_goal": "y"})
            return await orig(request)

        h.engine.up.client._transport = httpx.MockTransport(handle)
        await h.engine.on_follow_now()
        await h.frames(30)
        assert h.phase() == "running" and h.engine.stats["plan_recovery"] == 1
        assert h.sink.last.plan.revision == fake.plan_revision
        await h.engine.on_follow_now()
        await h.frames(15)
        assert fake.calls("/api/guide/follow")[-1]["plan_revision"] == fake.plan_revision and h.phase() == "running"

    run(go())


def test_tracker_http_error_clears_the_observation():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.running_and_tracking()
        orig = fake.handle

        async def handle(request):
            if request.url.path == "/api/track/frame":
                return httpx.Response(503, json={"error": "tracker_unavailable"})
            return await orig(request)

        h.engine.up.client._transport = httpx.MockTransport(handle)
        assert (await h.frame()).state == "unavailable"
        assert h.engine.pair is None
        n = len(fake.calls("/api/guide/follow"))
        await h.frames(60)
        assert len(fake.calls("/api/guide/follow")) == n  # no heartbeat follow on a stale picture

    run(go())


def test_pause_during_planning_holds_the_plan_call():
    async def go():
        h = H()
        await h.frame()
        await h.engine.on_start("안경을 벗어 주세요", None)
        await h.engine.on_pause()
        await h.engine.advance(3000)
        assert not h.fake.calls("/api/guide/plan")
        await h.engine.on_resume()
        await h.engine.advance(300)
        assert h.fake.calls("/api/guide/plan") and h.phase() == "running"
        assert h.fake.calls("/api/track/control")[-1]["action"] == "start"

    run(go())

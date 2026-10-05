"""RealEngine §15 Plan mode against the fake upstream (injected clock, no sockets where it can be helped).

Same harness shape as ``test_real_engine.py``: one engine, one fake upstream, one clock, ``auto_tick=False``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import time

import pytest
from PIL import Image
from conftest import Closed, Live, create_session, fast_settings, frame, jpeg, open_live
from fake_upstream import FakeUpstream
from starlette.testclient import TestClient

from synoptics_live import guide_policy as gp
from synoptics_live.app import create_app
from synoptics_live.contracts import FrameHeader, MaterialInput, PlanEditMsg, ReferenceImage, StepEdit
from synoptics_live.engine import EngineReject
from synoptics_live.real_engine import PROPOSAL_WAIT_REASON, RealConfig, RealContext, RealEngine

MATERIAL_TEXT = "3.1 바닥면은 완충재로 먼저 덮는다. 4.2 봉함 전 상자 안을 확인한다. " * 3
MATERIALS = (MaterialInput(title="포장 지침", version="v3", text=MATERIAL_TEXT),)

#: §15: a photo that is a usable JPEG (1024x1024, ~0.7MB ≤1.5MB) but whose base64 envelope is far over the
#: ordinary 512KB control-message bound — a message that may carry photos gets the larger one. Seeded, so its
#: size does not wobble between runs.
def _big_photo_b64(side: int = 1024, quality: int = 80) -> str:
    buf = io.BytesIO()
    Image.frombytes("RGB", (side, side), random.Random(7).randbytes(side * side * 3)).save(
        buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


BIG_PHOTO_B64 = _big_photo_b64()
PHOTO_B64 = base64.b64encode(jpeg(640, 360)).decode("ascii")


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


class H:
    def __init__(self, fake: FakeUpstream | None = None):
        self.fake = fake or FakeUpstream()
        self.sink = FakeSink()
        self.clock = FakeClock()
        ctx = RealContext(RealConfig(upstream_url=self.fake.base_url), transport=self.fake.transport())
        self.engine = RealEngine(self.sink, ctx, clock=self.clock, auto_tick=False)
        self.seq = 0

    async def frame(self, payload: bytes | None = None, w=640, h=360):
        header = {"t": "frame", "seq": self.seq, "captured_at": 1_759_479_000_000 + self.seq, "w": w, "h": h}
        self.seq += 1
        return await self.engine.on_frame(FrameHeader.model_validate(header), payload or jpeg(w, h))

    async def frames(self, n: int, ms: float = 100, payload: bytes | None = None):
        out = []
        for _ in range(n):
            out.append(await self.frame(payload))
            await self.engine.advance(ms)
        return out

    async def start(self, *, plan_mode: bool = True, plan_model: str = "deepseek:high",
                    core_mode: str = "classic",
                    materials=MATERIALS, context: str | None = "책상 위에서") -> None:
        await self.frame()
        await self.engine.on_start("안경을 벗어 주세요", context, plan_mode=plan_mode,
                                   plan_model=plan_model, core_mode=core_mode, materials=materials)
        await self.engine.settle()
        payload = jpeg(1024, 576)
        await self.engine.on_hi_frame(FrameHeader.model_validate(
            {"t": "frame", "seq": 999, "captured_at": 1, "w": 1024, "h": 576, "hi_req": f"h{self.sink.his}"}), payload)
        for _ in range(30):
            await self.engine.advance(100)
            if self.sink.last.phase != "planning":
                break

    async def running(self) -> None:
        """A plan-mode run that has been approved and is tracking."""
        await self.start()
        assert self.sink.last.phase == "reviewing"
        await self.engine.on_plan_approve()
        assert self.sink.last.phase == "running"
        await self.engine.advance(100)
        await self.frames(3)


async def _advance_until(h: H, pred, ms: float = 100, limit: int = 200, frames: bool = True):
    for _ in range(limit):
        if pred():
            return
        if frames:
            await h.frame()
        await h.engine.advance(ms)
    raise AssertionError("condition never met")


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------------- review (§15.1/§15.2/§15.3)


def test_plan_mode_start_reviews_a_draft_and_tracks_nothing():
    async def go():
        h = H()
        await h.start()
        st = h.sink.last
        assert st.phase == "reviewing" and st.pending is None and st.overlay is None
        plan = st.plan
        assert plan is not None and plan.status == "draft" and plan.approved_revision is None
        assert plan.revision == h.fake.plan_revision == 1 and plan.plan_id == h.fake.plan_id
        assert plan.target == "eyeglasses" and plan.goal_when == "안경이 얼굴에서 벗겨져 있다"
        assert [s.id for s in plan.steps] == ["s1", "s2", "s3"]
        # The upstream's steps are mapped as they are: a field it does not send keeps its contract default.
        s1 = plan.steps[0]
        assert s1.say == "안경다리를 양손으로 잡으세요" and s1.check == "visual" and s1.required is False
        assert s1.requires == [] and s1.targets == [] and s1.evidence is None
        # §15.3: the material's text stays in the engine; the state carries its id, title, version and length.
        assert [(m.id, m.title, m.version, m.chars) for m in plan.materials] == [("m1", "포장 지침", "v3",
                                                                                len(MATERIAL_TEXT))]
        assert plan.materials[0].added_at > 1_700_000_000_000
        # §15.3/§15.10: the material rides the plan prompt as its own field, never inside `context`.
        req = h.fake.calls("/api/guide/plan")[0]
        assert req["context"] == "책상 위에서" and req["user_goal"] == "안경을 벗어 주세요"
        assert req["materials"] == [{"title": "포장 지침", "version": "v3", "text": MATERIAL_TEXT}]
        assert "완충재" not in json.dumps(req["context"], ensure_ascii=False)
        # §15: the plan request carries the profile, never the retired review-mode selector.
        assert req["plan_model"] == "deepseek:high" and "plan_mode" not in req
        # §15.1: no tracking run and no judgement call while reviewing.
        assert not h.fake.calls("/api/track/control") and not h.fake.calls("/api/guide/follow")
        assert (await h.frame()).state == "idle"
        await h.frames(5)
        assert not h.fake.calls("/api/track/control") and not h.fake.calls("/api/guide/follow")
        # §15.9: entering review says the draft line once (and says nothing else while reviewing).
        assert h.sink.says == [(gp.PLANNING_SPEECH, "replace"), (gp.DRAFT_SAY, "replace")]
        # §15.4.1: follow_now / replan_now / talk are not_running before approval.
        for call in (h.engine.on_follow_now(), h.engine.on_replan_now(), h.engine.on_talk("안녕")):
            with pytest.raises(EngineReject) as rej:
                await call
            assert rej.value.code == "not_running"
        await h.engine.close()

    run(go())


def test_each_planning_profile_is_frozen_on_the_run_and_forwarded_upstream():
    """§15: `plan_model` is accepted regardless of review mode and stored on the run's task identity."""
    async def go():
        for choice in ("deepseek:high", "astra:high"):
            fake = FakeUpstream()
            h = H(fake)
            await h.start(plan_mode=True, plan_model=choice)
            assert h.sink.last.phase == "reviewing"
            assert h.engine.guide.plan_model == choice and h.engine.guide.plan_mode is True
            req = fake.calls("/api/guide/plan")[0]
            assert req["plan_model"] == choice and "plan_mode" not in req
            await h.engine.close()

    run(go())


def test_plan_mode_maps_the_upstream_step_fields_and_sends_materials():
    """§15.2/§15.3: what the upstream sends is carried as-is (never defaulted away), and materials reach it."""
    async def go():
        fake = FakeUpstream()
        first = dict(fake.plan_steps[0])
        first.update(check="user", required=True, targets=["상자"],
                     evidence={"material_id": "m1", "version": "v3", "locator": "3.1",
                               "quote": "바닥면은 완충재로 먼저 덮는다."})
        second = dict(fake.plan_steps[1])
        second.update(requires=["s1"])
        fake.plan_steps = [first, second, dict(fake.plan_steps[2])]
        h = H(fake)
        await h.start()
        plan = h.sink.last.plan
        assert plan.steps[0].check == "user" and plan.steps[0].required is True
        assert plan.steps[0].targets == ["상자"]
        evidence = plan.steps[0].evidence
        assert evidence is not None and evidence.material_id == "m1"
        assert evidence.version == "v3" and evidence.locator == "3.1"
        assert evidence.quote == "바닥면은 완충재로 먼저 덮는다."
        assert plan.steps[1].requires == ["s1"]
        # a field the upstream omits still keeps its contract default
        assert plan.steps[2].check == "visual" and plan.steps[2].required is False and plan.steps[2].requires == []
        # §15.3: the material rides its own field into the plan prompt, and never `context`.
        req = h.fake.calls("/api/guide/plan")[0]
        assert req["materials"] == [{"title": "포장 지침", "version": "v3", "text": MATERIAL_TEXT}]
        assert req["context"] == "책상 위에서"
        await h.engine.close()

    run(go())


def test_plan_approve_starts_tracking_and_the_follow_loop():
    async def go():
        h = H()
        await h.start()
        await h.engine.on_plan_approve()
        await h.engine.advance(100)
        st = h.sink.last
        assert st.phase == "running" and st.plan.status == "approved" and st.plan.approved_revision == 2
        assert st.plan.revision == h.fake.plan_revision == 2  # the pair `/approve` returned
        assert h.sink.says == [(gp.PLANNING_SPEECH, "replace"), (gp.DRAFT_SAY, "replace"),
                               ("1단계 / 3. 안경다리를 양손으로 잡으세요", "replace")]
        assert h.fake.calls("/api/track/control")[-1]["action"] == "start"
        assert h.fake.calls("/api/track/control")[-1]["target"] == "eyeglasses"
        t1, t2 = (await h.frame()).state, (await h.frame()).state
        assert (t1, t2) == ("acquiring", "tracking")
        await h.engine.settle()
        follow = h.fake.calls("/api/guide/follow")[0]
        assert follow["trigger"] == "acquired" and follow["current_step"] == "s1"
        assert follow["plan_revision"] == 2 and h.sink.last.overlay is not None
        await h.engine.close()

    run(go())


def test_plan_edit_revises_the_draft_only_and_is_locked_after_approval():
    async def go():
        h = H()
        await h.start()
        upstream_calls = len(h.fake.requests)
        await h.engine.on_plan_edit(PlanEditMsg(type="plan_edit", edits=[StepEdit(step_id="s2", required=True, check="user",
                                                               say="안경을 천천히 당겨 벗으세요")]))
        st = h.sink.last
        s2 = st.plan.steps[1]
        assert s2.required is True and s2.check == "user" and s2.say == "안경을 천천히 당겨 벗으세요"
        assert st.plan.revision == 2 and st.plan.status == "draft" and st.phase == "reviewing"
        assert [s.id for s in st.plan.steps] == ["s1", "s2", "s3"]
        # Nothing upstream, and the revision the upstream knows did not move (§15.5).
        assert len(h.fake.requests) == upstream_calls and h.engine.guide.plan_revision == h.fake.plan_revision == 1
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_edit(PlanEditMsg(type="plan_edit", edits=[StepEdit(step_id="s9", say="없는 단계")]))
        assert rej.value.code == "invalid_edit"
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_edit(PlanEditMsg(type="plan_edit", remove=["s1", "s2", "s3"]))
        assert rej.value.code == "invalid_edit"
        await h.engine.on_plan_approve()
        # §15.5: the approve names the UPSTREAM's draft pair (the client-only edit never moved it).
        approve = h.fake.calls("/api/guide/plan/approve")[0]
        assert approve["plan_revision"] == 1 and approve["steps"][1]["required"] is True
        st = h.sink.last
        assert st.phase == "running" and st.plan.approved_revision == 2 and st.plan.revision == 2
        assert st.plan.steps[1].required is True
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_edit(PlanEditMsg(type="plan_edit", edits=[StepEdit(step_id="s1", say="고치기")]))
        assert rej.value.code == "plan_locked"
        await h.engine.close()

    run(go())


def test_plan_approve_registers_the_plan_and_every_fence_follows_the_returned_pair():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.append({"visible": {"s1": "yes"}})
        h = H(fake)
        await h.start()
        draft = h.sink.last.plan
        await h.engine.on_plan_edit(PlanEditMsg(type="plan_edit",
                                               edits=[StepEdit(step_id="s2", say="안경을 천천히 당겨 벗으세요")]))
        await h.engine.on_plan_approve()
        posted = fake.calls("/api/guide/plan/approve")
        assert len(posted) == 1
        assert posted[0]["plan_id"] == draft.plan_id and posted[0]["plan_revision"] == draft.revision
        assert posted[0]["steps"][1]["say"] == "안경을 천천히 당겨 벗으세요"  # the client's edit, upstream
        assert posted[0]["goal_when"] == draft.goal_when
        pair = (fake.plan_id, fake.plan_revision)
        assert pair == (draft.plan_id, draft.revision + 1)  # the returned pair, not a diverge
        assert h.engine.guide.plan_revision == fake.plan_revision == pair[1]
        assert h.sink.last.plan.plan_id == pair[0] and h.sink.last.plan.revision == pair[1]
        await h.engine.advance(100)
        await _advance_until(h, lambda: fake.calls("/api/guide/confirm"))
        fences = [b for p, b in fake.requests if p in ("/api/guide/follow", "/api/guide/confirm")]
        assert fences and all(f["plan_id"] == pair[0] and f["plan_revision"] == pair[1] for f in fences)
        assert fake.current_steps()[1]["say"] == "안경을 천천히 당겨 벗으세요"  # the upstream stored it too
        await h.engine.close()

    run(go())


def test_plan_approve_on_a_stale_pair_re_reads_current_and_retries_once():
    async def go():
        h = H()
        await h.start()
        h.fake.plan_revision = 7  # the upstream moved on after the draft reached the client
        await h.engine.on_plan_approve()
        assert len(h.fake.calls("/api/guide/plan/current")) == 1
        attempts = h.fake.calls("/api/guide/plan/approve")
        assert [a["plan_revision"] for a in attempts] == [1, 7]
        assert h.engine.guide.plan_revision == h.fake.plan_revision == 8
        st = h.sink.last
        assert st.phase == "running" and st.plan.revision == 8
        assert h.engine.stats["plan_approve:failed:stale_plan"] == 0
        await h.engine.advance(100)
        await _advance_until(h, lambda: h.fake.calls("/api/guide/follow"))
        assert all(b["plan_revision"] == 8 for p, b in h.fake.requests if p == "/api/guide/follow")
        await h.engine.close()

    run(go())


def test_plan_approve_failure_ends_the_run_with_provider_error():
    async def go():
        h = H()
        await h.start()
        h.fake.plan_id = None  # the session has no plan: approve 409s and the current read 404s
        await h.engine.on_plan_approve()
        st = h.sink.last
        assert st.phase == "error" and st.error.code == "provider_error"
        assert st.plan is None
        assert not h.fake.calls("/api/track/control")  # never continued on diverged fences
        assert (await h.frame()).state == "idle"
        await h.engine.close()

    run(go())


def test_plan_discard_returns_to_idle():
    async def go():
        h = H()
        await h.start()
        await h.engine.on_plan_discard()
        st = h.sink.last
        assert st.phase == "idle" and st.plan is None
        assert not h.fake.calls("/api/track/control")
        assert (await h.frame()).state == "idle"
        await h.engine.close()

    run(go())


# ---------------------------------------------------------------------------------- proposals (§15.6)


def test_replan_now_becomes_a_proposal_and_accept_installs_it():
    async def go():
        fake = FakeUpstream()
        new_steps = [{"id": "s1", "say": "새 단계", "done_when": "새 상태", "commands": []},
                     {"id": "s2", "say": "새 두번째", "done_when": "두번째 상태", "commands": []}]
        fake.confirm_answers.append({"replan": new_steps})
        h = H(fake)
        await h.running()
        installed = h.sink.last.plan
        assert installed.revision == 2 and [s.say for s in installed.steps][0] == "안경다리를 양손으로 잡으세요"
        h.clock.advance(16)  # §15.4 keeps the existing per-run replan gap
        await h.engine.on_replan_now()
        await _advance_until(h, lambda: h.sink.last.proposal is not None)
        st = h.sink.last
        assert st.phase == "running" and st.completion == "none"
        assert st.proposal.plan_id == installed.plan_id and st.proposal.revision == 3
        assert st.proposal.reason and [s.say for s in st.proposal.steps] == ["새 단계", "새 두번째"]
        # The installed plan is authoritative until the client answers; the fences follow the upstream.
        assert [s.say for s in st.plan.steps] == [s.say for s in installed.steps] and st.plan.revision == 2
        assert h.engine.guide.plan_revision == fake.plan_revision == 3
        assert st.notice.code == "proposal"
        # §15.6: nothing is judged or advanced while the change waits — but frames and the box keep working.
        before = len(fake.requests)
        for _ in range(30):
            await h.frame()
            await h.engine.advance(100)
        assert set(p for p, _ in fake.requests[before:]) <= {"/api/track/frame"}  # no follow/confirm/replan
        assert len(fake.calls("/api/track/frame")) > 0  # the box is a view, not a decision
        st = h.sink.last
        assert st.proposal is not None and st.plan == installed and st.step_index == 0
        assert (await h.frame()).state == "tracking"
        # the client's judgement requests are refused until it answers
        for call in (h.engine.on_follow_now(), h.engine.on_replan_now(), h.engine.on_talk("다시 해 주세요")):
            with pytest.raises(EngineReject) as rej:
                await call
            assert rej.value.code == "not_running"
            assert rej.value.message == PROPOSAL_WAIT_REASON
        # accept: the change becomes the plan and judgement resumes
        await h.engine.on_proposal(accept=True)
        st = h.sink.last
        assert st.proposal is None and st.plan.revision == 3 and st.plan.approved_revision == 3
        assert [s.say for s in st.plan.steps] == ["새 단계", "새 두번째"]
        assert st.step_index == 0 and st.steps_user_done == [] and st.steps_skipped == []
        assert st.notice.code == "replanned" and st.overlay.commands == []
        n = len(fake.calls("/api/guide/follow"))
        await _advance_until(h, lambda: len(fake.calls("/api/guide/follow")) > n)
        assert fake.calls("/api/guide/follow")[-1]["current_step"] == "s1"
        await h.engine.close()

    run(go())


def test_proposal_reject_without_a_previous_plan_ends_the_run():
    async def go():
        fake = FakeUpstream()
        # No previous plan to restore: the upstream's history is empty, so `/plan/revert` answers 404 and the
        # run cannot be put back — this is the fallback, not the normal path (see the resume test below).
        fake.record_plan_history = False
        fake.confirm_answers.append({"replan": [{"id": "s1", "say": "버릴 단계", "done_when": "x", "commands": []}]})
        h = H(fake)
        await h.running()
        installed = h.sink.last.plan
        h.clock.advance(16)
        await h.engine.on_replan_now()
        await _advance_until(h, lambda: h.sink.last.proposal is not None)
        await h.engine.on_proposal(accept=False)
        await h.engine.settle()
        st = h.sink.last
        assert st.phase == "error" and st.error.code == "plan_changed"
        assert st.proposal is None and st.plan == installed  # what was approved is still shown
        assert h.fake.calls("/api/guide/plan/revert")  # it did try, and the upstream said no
        guides = len(fake.calls("/api/guide/follow")) + len(fake.calls("/api/guide/confirm"))
        for _ in range(20):
            await h.frame()
            await h.engine.advance(100)
        assert len(fake.calls("/api/guide/follow")) + len(fake.calls("/api/guide/confirm")) == guides
        assert (await h.frame()).state == "idle"  # tracking was released with the guide
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_proposal(accept=True)
        assert rej.value.code == "no_proposal"
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_follow_now()
        assert rej.value.code == "not_running"
        await h.engine.close()

    run(go())


def test_proposal_reject_reverts_upstream_and_resumes_judgement():
    async def go():
        fake = FakeUpstream()
        fake.confirm_answers.append({"replan": [{"id": "s1", "say": "버릴 단계", "done_when": "x", "commands": []}]})
        h = H(fake)
        await h.running()
        installed = h.sink.last.plan
        h.clock.advance(16)
        await h.engine.on_replan_now()
        await _advance_until(h, lambda: h.sink.last.proposal is not None)
        # §15.6: the freeze still holds while the change waits — frames only, no judgement.
        before = len(fake.requests)
        for _ in range(30):
            await h.frame()
            await h.engine.advance(100)
        assert set(p for p, _ in fake.requests[before:]) <= {"/api/track/frame"}
        # §15.6 reject: the upstream restores the approved plan under a NEW revision and the run resumes.
        await h.engine.on_proposal(accept=False)
        st = h.sink.last
        assert st.proposal is None and st.phase == "running" and st.error is None
        assert [s.say for s in st.plan.steps] == [s.say for s in installed.steps]
        assert st.plan.revision == fake.plan_revision == 4
        revert = fake.calls("/api/guide/plan/revert")[0]
        assert revert["plan_id"] == installed.plan_id and revert["plan_revision"] == 3
        n = len(fake.calls("/api/guide/follow"))
        await _advance_until(h, lambda: len(fake.calls("/api/guide/follow")) > n)
        follow = fake.calls("/api/guide/follow")[-1]
        assert follow["plan_id"] == installed.plan_id and follow["plan_revision"] == 4  # the restored pair
        assert follow["current_step"] == "s1"
        await h.engine.close()

    run(go())


def test_step_ack_is_recorded_but_moves_nothing_while_a_proposal_waits():
    async def go():
        fake = FakeUpstream()
        user_step = dict(fake.plan_steps[0])
        user_step.update(check="user")
        fake.plan_steps = [user_step, dict(fake.plan_steps[1]), dict(fake.plan_steps[2])]
        fake.confirm_answers.append({"replan": [{"id": "s1", "say": "새 단계", "done_when": "z", "commands": []}]})
        h = H(fake)
        await h.running()
        h.clock.advance(16)
        await h.engine.on_replan_now()
        await _advance_until(h, lambda: h.sink.last.proposal is not None)
        await h.engine.on_step_ack("s1")
        st = h.sink.last
        assert st.steps_user_done == ["s1"] and st.step_index == 0 and st.proposal is not None
        await h.engine.on_proposal(accept=True)
        st = h.sink.last
        assert st.step_index == 0 and st.steps_user_done == [] and [s.say for s in st.plan.steps] == ["새 단계"]
        await h.engine.close()

    run(go())


def test_unsure_twice_replan_is_also_a_proposal():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.extend([{"visible": {"s1": "unsure"}}, {"visible": {"s1": "unsure"}}])
        fake.confirm_answers.append({"replan": [{"id": "s1", "say": "다시 나눈 단계", "done_when": "y", "commands": []}]})
        h = H(fake)
        await h.running()
        h.clock.advance(16)
        await _advance_until(h, lambda: h.sink.last.proposal is not None)
        st = h.sink.last
        assert st.proposal is not None and st.plan.revision == 2
        assert [s.say for s in st.proposal.steps] == ["다시 나눈 단계"]
        assert h.engine.stats["follow_unsure"] >= 2
        await h.engine.close()

    run(go())


# ---------------------------------------------------------------------------------- pause / resume (§15.7)


def test_run_pause_stops_judgement_and_survives_a_reconnect():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.running()
        await _advance_until(h, lambda: fake.calls("/api/guide/follow"))
        await h.engine.on_run_pause()
        await h.engine.settle()
        st = h.sink.last
        assert st.paused is True and st.phase == "running" and st.overlay is None
        assert fake.calls("/api/track/control")[-1]["action"] == "stop"
        assert (await h.frame()).state == "idle"
        calls = len(fake.requests)
        uploads = len(fake.calls("/api/track/frame"))
        for _ in range(40):  # 4 s of frames and ticks: nothing may be dispatched
            await h.frame()
            await h.engine.advance(100)
        assert len(fake.requests) == calls and len(fake.calls("/api/track/frame")) == uploads
        # §15.7: a reconnect lifts the connection pause only — the user's pause stays.
        starts = len([c for c in fake.calls("/api/track/control") if c["action"] == "start"])
        await h.engine.on_pause()
        await h.engine.settle()
        await h.engine.on_resume()
        await h.engine.settle()
        assert h.sink.last.paused is True
        assert (await h.frame()).state == "idle"
        assert len([c for c in fake.calls("/api/track/control") if c["action"] == "start"]) == starts
        await h.engine.on_run_resume()
        await h.engine.settle()
        assert h.sink.last.paused is False
        tr = await h.frame()
        assert tr.state == "acquiring"
        assert (await h.frame()).state == "tracking"
        n = len(fake.calls("/api/guide/follow"))
        await _advance_until(h, lambda: len(fake.calls("/api/guide/follow")) > n)
        await h.engine.close()

    run(go())


# ---------------------------------------------------------------------------------- required steps (§15.4)


def test_required_step_holds_completion_until_the_user_acks_it():
    async def go():
        fake = FakeUpstream()
        required = dict(fake.plan_steps[2])
        required.update(required=True, check="user")
        fake.plan_steps = [dict(fake.plan_steps[0]), dict(fake.plan_steps[1]), required]
        fake.follow_answers.append({"goal_seen": "yes"})
        fake.confirm_answers.append({"status": "visually_satisfied"})
        h = H(fake)
        await h.running()
        await _advance_until(h, lambda: h.sink.last.blocked is not None)
        st = h.sink.last
        assert st.plan.steps[2].required is True and st.plan.steps[2].check == "user"
        assert st.blocked.step_id == "s3" and st.completion == "none"
        assert st.notice.code == "required_check" and "s3" in st.blocked.reason
        assert h.engine.stats["completion_held"] >= 1  # a satisfied goal_check was refused
        # it stays held: another satisfied goal_check cannot confirm either
        fake.follow_answers.append({"goal_seen": "yes"})
        fake.confirm_answers.append({"status": "visually_satisfied"})
        await h.frames(20)
        assert h.sink.last.completion == "none" and h.engine.stats["completion_held"] >= 1
        # the user finishes the required step; only then may completion be confirmed
        await h.engine.on_step_ack("s3")
        st = h.sink.last
        assert st.steps_user_done == ["s3"] and st.blocked is None and st.step_index == 0
        fake.follow_answers.append({"goal_seen": "yes"})
        fake.confirm_answers.extend([{"status": "visually_satisfied"}, {"status": "visually_satisfied"}])
        await _advance_until(h, lambda: h.sink.last.completion == "confirmed")
        st = h.sink.last
        assert st.completion == "confirmed" and st.blocked is None and st.steps_user_done == ["s3"]
        await h.engine.on_confirm_done()
        assert h.sink.last.phase == "completed"
        await h.engine.close()

    run(go())


def test_non_visual_step_is_not_advanced_by_the_frame_and_finishes_on_the_ack():
    async def go():
        fake = FakeUpstream()
        user_step = dict(fake.plan_steps[0])
        user_step.update(check="user")
        fake.plan_steps = [user_step, dict(fake.plan_steps[1]), dict(fake.plan_steps[2])]
        fake.follow_answers.append({"visible": {"s1": "yes"}})  # the frame says the step looks done
        h = H(fake)
        await h.running()
        await _advance_until(h, lambda: h.sink.last.blocked is not None)
        st = h.sink.last
        assert st.step_index == 0 and st.blocked.step_id == "s1" and st.steps_skipped == []
        assert st.notice.code == "required_check" and "화면만으로는" in st.blocked.reason
        assert not fake.calls("/api/guide/confirm")  # nothing moved, so nothing needs re-checking
        await h.engine.on_step_ack("s1")
        st = h.sink.last
        assert st.step_index == 1 and st.steps_user_done == ["s1"] and st.blocked is None
        await _advance_until(h, lambda: [c for c in fake.calls("/api/guide/confirm") if c["trigger"] == "step_done"])
        assert fake.calls("/api/guide/confirm")[-1]["current_step"] == "s1"
        await h.engine.close()

    run(go())


def test_follow_yes_advances_a_visual_step_through_the_gates():
    async def go():
        fake = FakeUpstream()
        fake.follow_answers.append({"visible": {"s1": "yes"}})
        h = H(fake)
        await h.running()
        await _advance_until(h, lambda: h.sink.last.step_index == 1)
        st = h.sink.last
        assert st.blocked is None and st.steps_skipped == [] and st.plan.revision == 2
        await _advance_until(h, lambda: fake.calls("/api/guide/confirm"))
        confirm = fake.calls("/api/guide/confirm")[0]
        follow = fake.calls("/api/guide/follow")[0]
        assert confirm["trigger"] == "step_done" and confirm["current_step"] == "s1"
        assert confirm["scene"]["frame_id"] == follow["scene"]["frame_id"]  # the follow's own frame
        # §15: the profile is frozen on the plan upstream, so later calls never re-pick it.
        assert "plan_model" not in follow and "plan_model" not in confirm
        assert h.sink.last.step_index == 1  # the fake's step_check says yes: the move stands
        await h.engine.close()

    run(go())


def test_step_ack_rejects_a_visual_step_and_an_unknown_one():
    async def go():
        h = H()
        await h.running()
        for step_id in ("s1", "s9"):
            with pytest.raises(EngineReject) as rej:
                await h.engine.on_step_ack(step_id)
            assert rej.value.code == "invalid_edit"
        await h.engine.close()

    run(go())


def test_required_step_is_never_skipped_by_follow_evidence():
    async def go():
        fake = FakeUpstream()
        required = dict(fake.plan_steps[1])
        required.update(required=True)
        fake.plan_steps = [dict(fake.plan_steps[0]), required, dict(fake.plan_steps[2])]
        fake.follow_answers.extend([{"visible": {"s3": "yes"}}, {"visible": {"s3": "yes"}}])
        h = H(fake)
        await h.running()
        await _advance_until(h, lambda: h.sink.last.blocked is not None)
        st = h.sink.last
        assert st.blocked.step_id == "s2" and st.step_index == 0 and st.steps_skipped == []
        assert st.plan.steps[1].required is True
        await h.engine.close()

    run(go())


# ---------------------------------------------------------------------------------- over the websocket


def make_real_client(fake: FakeUpstream) -> TestClient:
    settings = fast_settings(engine="real", upstream_url=fake.base_url)
    client = TestClient(create_app(settings, tts=None, upstream_transport=fake.transport()))
    client.__enter__()
    return client


def test_plan_edit_over_the_websocket_is_locked_after_approval():
    fake = FakeUpstream()
    c = make_real_client(fake)
    try:
        s = create_session(c)
        with open_live(c, s["session_id"]) as ws:
            live = Live(ws)
            live.hello(s["token"])
            live.send({"type": "start", "goal": "안경을 벗어 주세요", "consent_ai": True, "plan_mode": True,
                       "materials": [{"title": "포장 지침", "version": "v3", "text": MATERIAL_TEXT}]})
            hi = live.until_type("capture_hi")
            ws.send_bytes(frame(500, w=1024, h=576, hi_req=hi["req_id"]))
            reviewing = live.until(lambda m: m["type"] == "state" and m["phase"] == "reviewing")
            assert reviewing["plan"]["status"] == "draft" and reviewing["plan"]["approved_revision"] is None
            assert reviewing["plan"]["materials"][0]["chars"] == len(MATERIAL_TEXT)
            assert live.frame()["state"] == "idle"
            live.send({"type": "plan_edit", "edits": [{"step_id": "s2", "required": True, "check": "user"}]})
            edited = live.until(lambda m: m["type"] == "state" and m["plan"]["revision"] == 2)
            assert edited["plan"]["steps"][1]["required"] is True
            assert edited["plan"]["steps"][1]["check"] == "user"
            live.send({"type": "plan_approve"})
            approved = live.until(lambda m: m["type"] == "state" and m["phase"] == "running")
            assert approved["plan"]["status"] == "approved" and approved["plan"]["approved_revision"] == 2
            live.send({"type": "plan_edit", "edits": [{"step_id": "s2", "say": "고쳐 보기"}]})
            err = live.until_type("error")
            assert err["code"] == "plan_locked" and err["retryable"] is False
            live.send({"type": "proposal_accept"})
            err = live.until_type("error")
            assert err["code"] == "no_proposal"
            # the review phase is over: there is no draft left to discard either
            live.send({"type": "plan_discard"})
            err = live.until_type("error")
            assert err["code"] == "not_reviewing"
            live.send({"type": "stop"})
            live.until(lambda m: m["type"] == "state" and m["phase"] == "idle")
    finally:
        c.__exit__(None, None, None)


def test_run_pause_over_the_websocket_survives_the_reconnect():
    fake = FakeUpstream()
    c = make_real_client(fake)
    try:
        s = create_session(c)
        session = c.app.state.store.sessions[s["session_id"]]
        with open_live(c, s["session_id"]) as ws:
            live = Live(ws)
            live.hello(s["token"])
            live.send({"type": "start", "goal": "안경을 벗어 주세요", "consent_ai": True, "plan_mode": True})
            hi = live.until_type("capture_hi")
            ws.send_bytes(frame(500, w=1024, h=576, hi_req=hi["req_id"]))
            live.until(lambda m: m["type"] == "state" and m["phase"] == "reviewing")
            live.send({"type": "plan_approve"})
            live.until(lambda m: m["type"] == "state" and m["phase"] == "running")
            live.send({"type": "run_pause"})
            paused = live.until(lambda m: m["type"] == "state" and m.get("paused") is True)
            assert paused["phase"] == "running"
            assert live.frame()["state"] == "idle"
            rev = paused["rev"]
        deadline = time.time() + 2
        while session.conn is not None and time.time() < deadline:
            time.sleep(0.01)
        assert session.paused  # the socket is gone; the connection pause is on
        with open_live(c, s["session_id"]) as ws2:
            l2 = Live(ws2)
            ready, state = l2.hello(s["token"], resume_rev=rev)
            assert ready["rev"] == state["rev"] >= rev
            # §15.7: reconnecting lifted the connection pause only — the user's pause is still on.
            assert state["phase"] == "running" and state["paused"] is True
            assert not session.paused
            assert l2.frame()["state"] == "idle"
            l2.send({"type": "run_resume"})
            resumed = l2.until(lambda m: m["type"] == "state" and m.get("paused") is False)
            assert resumed["phase"] == "running"
            l2.send({"type": "stop"})
            l2.until(lambda m: m["type"] == "state" and m["phase"] == "idle")
    finally:
        c.__exit__(None, None, None)


# ---------------------------------------------------------------------------------- question turns (§15)

QUESTION = "어떤 대상을 안내할까요? 화면에 보이는 물건 이름을 알려 주세요."
SECOND_QUESTION = "그 물건에서 어떤 부분부터 볼까요?"


async def enter_clarifying(h: H, question: str = QUESTION) -> None:
    """A plan-mode run whose first plan call asks one question instead of writing the plan."""
    h.fake.plan_clarifications.append(question)
    await h.start()
    st = h.sink.last
    assert st.phase == "clarifying" and st.clarification == question and st.clarification_id


async def answer(h: H, text: str, *, clarification_id: str | None = None,
                 reference_images: tuple[ReferenceImage, ...] | None = None) -> None:
    """Answer the current question, then let the run play out its NEXT turn (a draft, or another question)."""
    cid = clarification_id if clarification_id is not None else h.sink.last.clarification_id
    await h.engine.on_plan_answer(cid, text, reference_images=reference_images)
    await h.engine.settle()
    await h.engine.on_hi_frame(FrameHeader.model_validate(
        {"t": "frame", "seq": 997, "captured_at": 1, "w": 1024, "h": 576, "hi_req": f"h{h.sink.his}"}),
        jpeg(1024, 576))
    for _ in range(30):
        await h.engine.advance(100)
        if h.sink.last.phase != "planning":
            break


def test_a_plan_question_is_answered_once_and_the_goal_materials_and_photo_are_kept():
    """The acceptance path: one question, one answer with a photo, then a reviewable draft — never a restart."""
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await enter_clarifying(h)
        st = h.sink.last
        assert st.plan is None and st.overlay is None and st.pending is None
        assert st.clarification_id
        # §15: a question is not permission to execute — no tracker, no judgement, no overlay.
        assert (await h.frame()).state == "idle"
        await h.frames(3)
        assert not fake.calls("/api/track/control") and not fake.calls("/api/guide/follow")
        for call in (h.engine.on_follow_now(), h.engine.on_replan_now(), h.engine.on_talk("안녕")):
            with pytest.raises(EngineReject) as rej:
                await call
            assert rej.value.code == "not_running"
        assert (QUESTION, "append") in h.sink.says  # the question is read aloud
        # one answer and one reference photo; the run keeps its goal, context and materials
        photo = ReferenceImage(frame_id="ref-1", image_base64=PHOTO_B64, label="드라이버 라벨")
        await answer(h, "안경다리입니다", reference_images=(photo,))
        st = h.sink.last
        assert st.phase == "reviewing" and st.plan.status == "draft" and st.plan.approved_revision is None
        assert st.clarification is None and st.clarification_id is None
        calls = fake.calls("/api/guide/plan")
        assert len(calls) == 2
        first, second = calls
        assert first["user_goal"] == second["user_goal"] == "안경을 벗어 주세요"
        assert second["context"] == first["context"] == "책상 위에서"
        assert second["materials"] == first["materials"] == [
            {"title": "포장 지침", "version": "v3", "text": MATERIAL_TEXT}]
        assert "answers" not in first and "reference_images" not in first  # nothing to carry on the first turn
        assert second["answers"] == [{"question": QUESTION, "answer": "안경다리입니다"}]
        assert second["reference_images"] == [
            {"frame_id": "ref-1", "image_base64": PHOTO_B64, "label": "드라이버 라벨"}]
        # §15: review mode asks for assisted planning; DeepSeek has no hosted search tool, so it is never told
        # it has one (Astra is the profile that does — see the sources test below).
        assert second["assisted"] is True and second["research"] is False
        # the photo bytes stay in the run's memory: they are never echoed back in state
        assert "ref-1" not in h.sink.last.model_dump_json()
        # ... and nothing runs until the explicit approval
        assert not fake.calls("/api/track/control") and not fake.calls("/api/guide/follow")
        await h.engine.on_plan_approve()
        await h.engine.advance(100)
        assert h.sink.last.phase == "running" and h.sink.last.plan.status == "approved"
        assert fake.calls("/api/track/control")[-1]["action"] == "start"
        assert fake.calls("/api/track/control")[-1]["target"] == "eyeglasses"
        await h.engine.close()

    run(go())


@pytest.mark.parametrize("status,code", [(503, "service_unavailable"), (401, "session_required")])
def test_failed_answer_keeps_evidence_until_explicit_corrected_resubmission(status, code):
    class FailingOnce(FakeUpstream):
        fail = False

        async def handle(self, request):
            if self.fail and request.url.path == "/api/guide/plan":
                self.fail = False
                return self._err(status, code)
            return await super().handle(request)

    async def go():
        fake = FailingOnce()
        h = H(fake)
        await enter_clarifying(h)
        old_id = h.sink.last.clarification_id
        fake.fail = True
        photo = ReferenceImage(frame_id="ref-retry", image_base64=PHOTO_B64, label="대상")
        await answer(h, "처음 답", reference_images=(photo,))
        failed = h.sink.last
        assert failed.phase == "clarifying" and failed.clarification == QUESTION
        assert failed.clarification_id != old_id and failed.notice.code == "plan_answer_failed"
        await h.frames(10)
        assert len(fake.calls("/api/guide/plan")) == 1  # no automatic provider retry
        assert not fake.calls("/api/track/control") and not fake.calls("/api/guide/follow")
        with pytest.raises(EngineReject) as stale:
            await h.engine.on_plan_answer(old_id, "늦은 답")
        assert stale.value.code == "clarification_stale"
        await answer(h, "고친 답")
        assert h.sink.last.phase == "reviewing" and h.sink.last.plan.status == "draft"
        submitted = fake.calls("/api/guide/plan")[-1]
        assert submitted["user_goal"] == "안경을 벗어 주세요"
        assert submitted["materials"][0]["text"] == MATERIAL_TEXT
        assert submitted["answers"] == [{"question": QUESTION, "answer": "고친 답"}]
        assert submitted["reference_images"][0]["frame_id"] == "ref-retry"
        assert not fake.calls("/api/track/control") and not fake.calls("/api/guide/follow")
        await h.engine.close()

    run(go())


def test_plan_answers_are_fenced_by_the_current_question_id():
    """An answer to a question that has been replaced (or answered) is refused, and the run stays answerable."""
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await enter_clarifying(h)
        first_id = h.sink.last.clarification_id
        fake.plan_clarifications.append(SECOND_QUESTION)
        await answer(h, "첫 답")
        st = h.sink.last
        assert st.phase == "clarifying" and st.clarification == SECOND_QUESTION
        second_id = st.clarification_id
        assert second_id and second_id != first_id
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_answer(first_id, "뒤늦은 답")
        assert rej.value.code == "clarification_stale" and rej.value.retryable is False
        assert h.sink.last.phase == "clarifying" and h.sink.last.clarification_id == second_id
        await answer(h, "둘째 답")
        assert h.sink.last.phase == "reviewing"
        # the finished question cannot be answered again (the id is taken exactly once)
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_answer(second_id, "또 답")
        assert rej.value.code == "no_clarification"
        # both answers were kept, oldest first, with the question each one answered
        assert fake.calls("/api/guide/plan")[-1]["answers"] == [
            {"question": QUESTION, "answer": "첫 답"},
            {"question": SECOND_QUESTION, "answer": "둘째 답"}]
        await h.engine.close()

    run(go())


def test_stop_and_a_new_goal_fence_a_late_question_answer():
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await enter_clarifying(h)
        old_id = h.sink.last.clarification_id
        await h.engine.on_stop()
        assert h.sink.last.phase == "idle" and h.sink.last.clarification is None
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_answer(old_id, "늦은 답")
        assert rej.value.code == "no_clarification"
        # §15: a start IS allowed after a clarification (the run carries no plan in flight), and the previous
        # question still cannot be answered.
        await enter_clarifying(h)
        new_id = h.sink.last.clarification_id
        assert new_id != old_id
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_answer(old_id, "예전 질문의 답")
        assert rej.value.code == "clarification_stale"
        # a new goal replaces the run: the pending question goes with it and its late answer is dropped
        await h.engine.on_start("다른 목표를 도와줘", None, plan_mode=True)
        await h.engine.settle()
        assert h.sink.last.phase == "planning"
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_answer(new_id, "예전 질문의 답")
        assert rej.value.code == "no_clarification"
        await h.engine.on_hi_frame(FrameHeader.model_validate(
            {"t": "frame", "seq": 996, "captured_at": 1, "w": 1024, "h": 576, "hi_req": f"h{h.sink.his}"}),
            jpeg(1024, 576))
        for _ in range(30):
            await h.engine.advance(100)
            if h.sink.last.phase != "planning":
                break
        assert h.sink.last.phase == "reviewing"
        last = fake.calls("/api/guide/plan")[-1]
        assert last["user_goal"] == "다른 목표를 도와줘"
        assert "answers" not in last and "reference_images" not in last  # a fresh run carries neither
        await h.engine.close()

    run(go())


def test_reference_photos_are_checked_before_any_dispatch():
    """Huge or malformed photos are refused at the Live boundary — no provider call sees them."""
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await enter_clarifying(h)
        cid = h.sink.last.clarification_id
        too_big = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 1_500_100).decode("ascii")
        png = io.BytesIO()
        Image.new("RGB", (640, 360), (10, 20, 30)).save(png, "PNG")
        enoent = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(enoent, "JPEG")
        bad = [
            (ReferenceImage(frame_id="r", image_base64="!!!not-base64!!!"), "invalid_image"),
            (ReferenceImage(frame_id="r", image_base64=too_big), "image_too_large"),
            (ReferenceImage(frame_id="r", image_base64=base64.b64encode(png.getvalue()).decode("ascii")),
             "invalid_image"),
            (ReferenceImage(frame_id="r", image_base64=base64.b64encode(enoent.getvalue()).decode("ascii")),
             "invalid_image"),
        ]
        for image, code in bad:
            before = len(fake.requests)
            with pytest.raises(EngineReject) as rej:
                await h.engine.on_plan_answer(cid, "안경", reference_images=(image,))
            assert rej.value.code == code and rej.value.retryable is False
            assert len(fake.requests) == before  # nothing was dispatched
            assert h.sink.last.phase == "clarifying" and h.sink.last.clarification_id == cid
        # a start carrying a bad photo is refused before the run is started at all
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_start("x", None, plan_mode=True, reference_images=(bad[0][0],))
        assert rej.value.code == "invalid_image"
        assert h.sink.last.phase == "clarifying" and h.sink.last.clarification_id == cid
        # the run is still answerable with a usable photo
        await answer(h, "안경", reference_images=(ReferenceImage(frame_id="ok", image_base64=PHOTO_B64),))
        assert h.sink.last.phase == "reviewing"
        assert fake.calls("/api/guide/plan")[-1]["reference_images"][0]["frame_id"] == "ok"
        await h.engine.close()

    run(go())


def test_research_sources_and_step_details_ride_the_review_and_survive_approval():
    async def go():
        fake = FakeUpstream()
        first = dict(fake.plan_steps[0])
        first["details"] = "안경다리를 양손으로 잡고, 렌즈는 건드리지 마세요."
        fake.plan_steps = [first, dict(fake.plan_steps[1]), dict(fake.plan_steps[2])]
        fake.plan_sources = [{"url": "https://example.com/glasses", "title": "안경 관리 안내",
                              "summary": "안경다리를 분리하는 방법"}]
        h = H(fake)
        await h.start(plan_model="astra:high")
        st = h.sink.last
        assert st.phase == "reviewing"
        assert [(s.url, s.title) for s in st.research_sources] == [("https://example.com/glasses", "안경 관리 안내")]
        assert st.plan.research_sources == st.research_sources  # the same facts, also on the Plan
        assert st.plan.steps[0].details == "안경다리를 양손으로 잡고, 렌즈는 건드리지 마세요."
        req = fake.calls("/api/guide/plan")[0]
        assert req["assisted"] is True and req["research"] is True  # the profile with a real search tool
        # a client edit keeps the details and the sources
        await h.engine.on_plan_edit(PlanEditMsg(type="plan_edit", edits=[StepEdit(step_id="s2", say="새 문장")]))
        st = h.sink.last
        assert st.plan.steps[0].details == "안경다리를 양손으로 잡고, 렌즈는 건드리지 마세요."
        assert st.plan.steps[1].say == "새 문장"
        assert [s.url for s in st.plan.research_sources] == ["https://example.com/glasses"]
        # approval carries the readable details upstream and keeps the sources in the client's plan
        await h.engine.on_plan_approve()
        st = h.sink.last
        assert st.phase == "running"
        approve = fake.calls("/api/guide/plan/approve")[0]
        assert approve["steps"][0]["details"] == "안경다리를 양손으로 잡고, 렌즈는 건드리지 마세요."
        assert [s.url for s in st.plan.research_sources] == ["https://example.com/glasses"]
        assert [s.url for s in st.research_sources] == ["https://example.com/glasses"]
        await h.engine.close()

    run(go())


def test_only_well_formed_public_sources_are_kept():
    async def go():
        fake = FakeUpstream()
        fake.plan_sources = [
            {"url": "not-a-url", "title": "제목", "summary": "요약"},
            {"url": "ftp://files.example/glasses", "title": "제목", "summary": "요약"},
            {"url": "https://user:pw@example.com/x", "title": "제목", "summary": "요약"},
            {"url": "https://example.com/glasses", "title": "안경 관리 안내", "summary": "다리 분리 방법"},
        ]
        h = H(fake)
        await h.start()
        st = h.sink.last
        # everything a provider could not have observed stands as an HTTP(S) URL with no credentials — the rest
        # is dropped instead of making the snapshot invalid.
        assert [(s.url, s.title) for s in st.research_sources] == [("https://example.com/glasses", "안경 관리 안내")]
        assert st.plan.research_sources == st.research_sources
        await h.engine.close()

    run(go())


def test_a_run_stops_asking_after_twelve_answered_questions():
    """§15: at most twelve answered questions, and the run never shows a thirteenth it could not accept."""
    async def go():
        fake = FakeUpstream()
        fake.plan_clarifications.extend(f"질문 {n}" for n in range(1, 14))
        h = H(fake)
        await h.start()
        for n in range(1, 13):
            st = h.sink.last
            assert st.phase == "clarifying" and st.clarification == f"질문 {n}"
            await answer(h, f"답 {n}")
        st = h.sink.last
        assert st.phase == "idle" and st.clarification is None and st.clarification_id is None
        assert st.notice is not None and "질문이 너무 많아" in st.notice.text
        calls = fake.calls("/api/guide/plan")
        assert len(calls) == 13 and len(calls[-1]["answers"]) == 12
        assert h.engine.stats["plan_clarification_limit"] == 1
        assert not fake.calls("/api/track/control")
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_plan_answer("q-last", "뒤늦은 답")
        assert rej.value.code == "no_clarification"
        await h.engine.close()

    run(go())


# ---------------------------------------------------------------------------------- question turns over the socket


def test_the_question_survives_a_reconnect_and_the_photo_answer_needs_the_larger_bound():
    fake = FakeUpstream()
    fake.plan_clarifications.append(QUESTION)
    fake.plan_sources = [{"url": "https://example.com/glasses", "title": "안경 관리 안내",
                          "summary": "안경다리를 분리하는 방법"}]
    c = make_real_client(fake)
    try:
        s = create_session(c)
        session = c.app.state.store.sessions[s["session_id"]]
        with open_live(c, s["session_id"]) as ws:
            live = Live(ws)
            live.hello(s["token"])
            live.send({"type": "start", "goal": "안경을 벗어 주세요", "consent_ai": True, "plan_mode": True})
            hi = live.until_type("capture_hi")
            ws.send_bytes(frame(500, w=1024, h=576, hi_req=hi["req_id"]))
            clarifying = live.until(lambda m: m["type"] == "state" and m["phase"] == "clarifying")
            cid = clarifying["clarification_id"]
            assert clarifying["clarification"] == QUESTION and cid
            assert [s["url"] for s in clarifying["research_sources"]] == ["https://example.com/glasses"]
            assert live.frame()["state"] == "idle"
            rev = clarifying["rev"]
        deadline = time.time() + 2
        while session.conn is not None and time.time() < deadline:
            time.sleep(0.01)
        with open_live(c, s["session_id"]) as ws2:
            live2 = Live(ws2)
            ready, state = live2.hello(s["token"], resume_rev=rev)
            # the reconnect snapshot still has the easy question and the id its answer must echo
            assert state["rev"] == ready["rev"] >= rev
            assert state["phase"] == "clarifying" and state["clarification"] == QUESTION
            assert state["clarification_id"] == cid and state["research_sources"] == [
                {"url": "https://example.com/glasses", "title": "안경 관리 안내", "summary": "안경다리를 분리하는 방법"}]
            assert "image_base64" not in json.dumps(state)  # no photo bytes are ever echoed back
            before = len(fake.calls("/api/guide/plan"))
            live2.send({"type": "plan_answer", "clarification_id": cid, "answer": "사진이 너무 많습니다",
                        "reference_images": [{"frame_id": f"extra-{i}", "image_base64": PHOTO_B64}
                                             for i in range(13)]})
            assert live2.until_type("error")["code"] == "invalid_message"
            assert len(fake.calls("/api/guide/plan")) == before
            # All twelve photos survive an answer; the existing aggregate envelope limit still applies.
            photo_answer = {"type": "plan_answer", "clarification_id": cid, "answer": "안경다리입니다",
                            "reference_images": [{"frame_id": f"ref-{i}", "image_base64": BIG_PHOTO_B64 if i < 6 else PHOTO_B64,
                                                  "label": "참고 사진"} for i in range(12)]}
            assert 512 * 1024 < len(json.dumps(photo_answer)) < 10 * 1024 * 1024
            live2.send(photo_answer)
            hi2 = live2.until_type("capture_hi")
            ws2.send_bytes(frame(501, w=1024, h=576, hi_req=hi2["req_id"]))
            reviewing = live2.until(lambda m: m["type"] == "state" and m["phase"] == "reviewing")
            assert reviewing["plan"]["status"] == "draft"
            assert not fake.calls("/api/track/control") and not fake.calls("/api/guide/follow")
            posted = fake.calls("/api/guide/plan")[-1]
            assert posted["user_goal"] == "안경을 벗어 주세요"
            assert posted["answers"] == [{"question": QUESTION, "answer": "안경다리입니다"}]
            assert [r["frame_id"] for r in posted["reference_images"]] == [f"ref-{i}" for i in range(12)]
            live2.send({"type": "plan_approve"})
            live2.until(lambda m: m["type"] == "state" and m["phase"] == "running")
            assert fake.calls("/api/track/control")[-1]["action"] == "start"
            live2.send({"type": "stop"})
            live2.until(lambda m: m["type"] == "state" and m["phase"] == "idle")
    finally:
        c.__exit__(None, None, None)


def test_the_larger_bound_is_only_for_messages_that_may_carry_photos():
    fake = FakeUpstream()
    c = make_real_client(fake)
    try:
        s = create_session(c)
        with open_live(c, s["session_id"]) as ws:
            live = Live(ws)
            live.hello(s["token"])
            # `plan_edit` cannot carry a photo, so it keeps the ordinary bound even in review mode.
            live.send({"type": "plan_edit", "edits": [], "junk": "x" * (512 * 1024)})
            assert live.until_type("bye")["reason"] == "message_too_large"
            with pytest.raises(Closed) as e:
                live.recv()
            assert e.value.code == 4009
    finally:
        c.__exit__(None, None, None)

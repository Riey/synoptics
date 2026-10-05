"""§15.8 guide-core choice (``classic`` | ``sequential``) against the fake upstream.

The harness (``H``, the fake clock/sink and ``_advance_until``) is shared with ``test_real_plan_mode.py``: one
engine, one fake upstream, one injected clock, ``auto_tick=False``. Every assertion here is about an actual
state transition of the real engine — a follower nomination, the upper's own verdict, the fences that still
hold, and what survives a stale/replayed answer.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fake_upstream import FakeUpstream
from test_real_plan_mode import H, _advance_until, run

from synoptics_live.engine import EngineReject


def yes_current(body: dict) -> dict:
    """A follower answer that says the current step looks done (and nothing about any other step)."""
    return {"visible": {body["current_step"]: "yes"}}


def yes_only(step_id: str):
    def answer(body: dict) -> dict:
        return {"visible": {body["current_step"]: "yes"}} if body["current_step"] == step_id else {}
    return answer


# ------------------------------------------------------------------------- start / selection (§15.8)


@pytest.mark.parametrize("core", ["classic", "sequential"])
@pytest.mark.parametrize("plan_mode", [False, True])
@pytest.mark.parametrize("plan_model", ["deepseek:high", "astra:high"])
def test_core_mode_is_frozen_on_the_run_and_forwarded_upstream(core, plan_mode, plan_model):
    """The core is an axis of its own: any core × review boundary × planning profile is accepted, stored on the
    run's identity, echoed in ``state``, and sent to the upstream — which stores it with the plan."""
    async def go():
        fake = FakeUpstream()
        h = H(fake)
        await h.start(plan_mode=plan_mode, plan_model=plan_model, core_mode=core)
        engine_run = h.engine.guide
        assert engine_run.core_mode == core and engine_run.plan_mode is plan_mode
        assert engine_run.plan_model == plan_model
        req = fake.calls("/api/guide/plan")[0]
        assert req["core_mode"] == core and req["plan_model"] == plan_model and "plan_mode" not in req
        assert h.sink.last.core_mode == core
        assert h.sink.last.phase == ("reviewing" if plan_mode else "running")
        # Classic without review keeps master's immediate flow (no server-side plan model, no §15.4 fences);
        # a reviewed run and either sequential run carries the plan model the fences are computed from.
        assert (h.engine.plan is None) == (core == "classic" and not plan_mode)
        if h.engine.plan is not None:
            assert h.engine.plan.status == ("draft" if plan_mode else "approved")
        await h.engine.close()

    run(go())


# ------------------------------------------------------------------------- nomination → commit (§15.8)


def test_sequential_nominates_then_commits_exactly_one_step_on_the_upper_verdict():
    """A follower yes only nominates; the upper's own ``step_done`` yes on that same activation commits it."""
    async def go():
        fake = FakeUpstream()
        fake.follow_fn = yes_only("s1")
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: h.engine.guide.seq_candidate is not None)
        engine_run = h.engine.guide
        st = h.sink.last
        # nominated only: the screen has NOT moved and nothing is confirmed yet.
        assert engine_run.step_index == 0 and st.step_index == 0
        assert engine_run.confirmed_steps == () and engine_run.seq_candidate is not None
        assert engine_run.seq_candidate.step_id == "s1" and engine_run.seq_candidate.activation == 0
        assert st.notice is not None and st.notice.code == "step_candidate"
        # the checklist the engine kept is the current step alone (never the future plan)
        assert [c["step_id"] for c in engine_run.last_checks] == ["s1"]
        await _advance_until(h, lambda: engine_run.step_index == 1)
        assert engine_run.confirmed_steps and engine_run.confirmed_steps[0].step_id == "s1"
        assert engine_run.confirmed_steps[0].activation == 0 and engine_run.seq_activation == 1
        assert engine_run.confirmed_steps[0].run_id and engine_run.confirmed_steps[0].frame_id
        assert engine_run.confirmed_steps[0].plan_revision == engine_run.plan_revision
        assert engine_run.seq_candidate is None and h.sink.last.step_index == 1
        confirm = fake.calls("/api/guide/confirm")[0]
        follow = fake.calls("/api/guide/follow")[0]
        assert confirm["trigger"] == "step_done" and confirm["current_step"] == "s1"
        assert confirm["scene"]["frame_id"] == follow["scene"]["frame_id"]  # the follower's own frame
        assert h.sink.last.steps_skipped == []  # sequential never skips a step
        await h.engine.close()

    run(go())


def test_sequential_holds_on_a_non_yes_upper_verdict_and_a_fresh_nomination_can_retry():
    """``no``/``unsure`` on the upper verdict moves nothing and confirms nothing; a fresh follower yes retries."""
    async def go():
        fake = FakeUpstream()
        fake.follow_fn = yes_only("s1")
        fake.confirm_answers.append({"step_check": "no"})
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: len(fake.calls("/api/guide/confirm")) == 1)
        await _advance_until(h, lambda: h.sink.last.notice is not None and h.sink.last.notice.code == "step_held")
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        # the same step can be nominated again, and this time the upper agrees
        await _advance_until(h, lambda: engine_run.step_index == 1)
        assert [e.step_id for e in engine_run.confirmed_steps] == ["s1"]
        assert len(fake.calls("/api/guide/confirm")) >= 2
        assert all(c["current_step"] == "s1" for c in fake.calls("/api/guide/confirm")[:2])
        await h.engine.close()

    run(go())


def test_sequential_never_moves_on_a_verdict_about_another_step():
    """The backend scopes the checklist to the current step; a verdict about a later step is discarded here too
    (a classic plan reaching a sequential run can never skip the screen forward)."""
    async def go():
        fake = FakeUpstream()
        fake.follow_fn = lambda body: {"checks": [{"step_id": "s3", "visible": "yes"}]}
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: bool(fake.calls("/api/guide/follow")))
        await h.frames(10)
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        assert engine_run.seq_candidate is None and engine_run.skipped == ()
        assert not fake.calls("/api/guide/confirm")  # nothing was nominated, so nothing is re-checked
        assert engine_run.last_checks == ()  # the other step's verdict was discarded
        await h.engine.close()

    run(go())


def test_sequential_commit_does_not_pass_an_unmet_requirement_of_the_next_step():
    """§15.4.3: the step a commit is about to enter may still require others — the run holds and says why."""
    async def go():
        fake = FakeUpstream()
        second = dict(fake.plan_steps[1])
        second["requires"] = ["s3"]  # s3 is only done once the plan has moved past it
        fake.plan_steps = [dict(fake.plan_steps[0]), second, dict(fake.plan_steps[2])]
        fake.follow_fn = yes_only("s1")
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: h.sink.last.blocked is not None)
        engine_run = h.engine.guide
        st = h.sink.last
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        assert st.blocked.step_id == "s2" and st.blocked.requires == ["s3"]
        assert st.notice is not None and st.notice.code == "required_check"
        await h.engine.close()

    run(go())


def test_sequential_user_step_waits_for_its_own_ack_and_records_no_evidence():
    """A ``check:"user"`` step is never finished by the frame or by the upper's picture verdict."""
    async def go():
        fake = FakeUpstream()
        first = dict(fake.plan_steps[0])
        first["check"] = "user"
        fake.plan_steps = [first, dict(fake.plan_steps[1]), dict(fake.plan_steps[2])]
        fake.follow_fn = yes_only("s1")
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: h.sink.last.blocked is not None)
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        assert h.sink.last.blocked.step_id == "s1" and "화면만으로는" in h.sink.last.blocked.reason
        await h.engine.on_step_ack("s1")
        assert h.sink.last.step_index == 1 and h.sink.last.steps_user_done == ["s1"]
        assert engine_run.confirmed_steps == ()  # the ack advanced the step; the frame confirmed nothing
        await h.engine.close()

    run(go())


def test_sequential_required_visual_step_is_not_finished_by_the_frame():
    """Master's guard stands: a ``required`` step only moves on an explicit user record, and a visual one
    rejects ``step_ack`` — so a required visual step can never be finished by a picture."""
    async def go():
        fake = FakeUpstream()
        first = dict(fake.plan_steps[0])
        first["required"] = True
        fake.plan_steps = [first, dict(fake.plan_steps[1]), dict(fake.plan_steps[2])]
        fake.follow_fn = yes_only("s1")
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: h.sink.last.blocked is not None)
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        assert h.sink.last.blocked.step_id == "s1"
        with pytest.raises(EngineReject):
            await h.engine.on_step_ack("s1")
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        await h.engine.close()

    run(go())


def test_sequential_last_step_confirms_without_an_out_of_range_move_and_lets_the_goal_complete():
    """The final step's own upper yes records evidence in place (there is nowhere to move to), and the goal
    confirmation then completes the run — without pretending a user/measure acknowledgement happened."""
    async def go():
        fake = FakeUpstream()
        fake.plan_steps = [dict(fake.plan_steps[0])]
        fake.follow_fn = yes_current
        fake.confirm_answers.extend([{"status": "visually_satisfied"}, {"status": "visually_satisfied"}])
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: bool(h.engine.guide.confirmed_steps))
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and len(h.engine.plan.steps) == 1
        assert [e.step_id for e in engine_run.confirmed_steps] == ["s1"]
        assert h.sink.last.steps_user_done == []  # nothing was faked
        await _advance_until(h, lambda: h.sink.last.completion == "confirmed")
        assert h.sink.last.completion == "confirmed" and h.sink.last.step_index == 0
        assert [c["trigger"] for c in fake.calls("/api/guide/confirm")][:2] == ["step_done", "goal_check"]
        await h.engine.on_confirm_done()
        assert h.sink.last.phase == "completed" and h.sink.last.core_mode == "sequential"
        await h.engine.close()

    run(go())


def test_sequential_stale_candidate_cannot_move_a_newly_activated_step():
    """An upper verdict that arrives after the run has moved on (here: the user finished the step while the
    answer was still open) commits nothing — the step that is current now is never moved by it."""
    async def go():
        fake = FakeUpstream()
        first = dict(fake.plan_steps[0])
        first["check"] = "user"
        fake.plan_steps = [first, dict(fake.plan_steps[1]), dict(fake.plan_steps[2])]
        fake.follow_fn = yes_only("s1")
        gate = asyncio.Event()

        async def hold(body: dict) -> None:
            if body["trigger"] == "step_done" and not gate.is_set():
                await gate.wait()

        fake.confirm_hold = hold
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: len(fake.calls("/api/guide/confirm")) == 1)
        engine_run = h.engine.guide
        assert engine_run.seq_candidate is not None and engine_run.step_index == 0
        # the user finishes s1 while the upper's step_done verdict is still open
        await h.engine.on_step_ack("s1")
        assert h.sink.last.step_index == 1 and h.sink.last.steps_user_done == ["s1"]
        assert engine_run.seq_activation == 1
        gate.set()
        await h.engine.advance(200)
        await h.frames(10)
        assert engine_run.step_index == 1 and h.sink.last.step_index == 1
        assert engine_run.confirmed_steps == ()
        await h.engine.close()

    run(go())


def test_sequential_upper_call_error_holds_and_the_retry_can_still_commit():
    """A retryable upper failure moves nothing and keeps the nomination; the retry's own yes then commits."""
    async def go():
        fake = FakeUpstream()
        fake.follow_fn = yes_only("s1")
        fake.confirm_answers.append(httpx.Response(503, json={"error": "provider_busy"},
                                                   headers={"Retry-After": "1"}))
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: h.engine.guide.seq_candidate is not None)
        await _advance_until(h, lambda: len(fake.calls("/api/guide/confirm")) == 1)
        await h.engine.advance(200)
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()  # the error holds
        assert engine_run.seq_candidate is not None
        await h.engine.advance(2_000)  # past the retry wait
        await _advance_until(h, lambda: engine_run.step_index == 1)
        assert [e.step_id for e in engine_run.confirmed_steps] == ["s1"]
        await h.engine.close()

    run(go())


def test_sequential_rejected_verdict_drops_the_nomination_without_moving():
    """An answer the fence no longer binds to (here: the anchor drifted under the call) is discarded: the step
    stays put, nothing is confirmed, and a fresh follower yes may nominate again."""
    async def go():
        fake = FakeUpstream()
        fake.follow_fn = yes_only("s1")
        fake.confirm_echo_drift = True
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: h.engine.guide.seq_candidate is not None)
        await _advance_until(h, lambda: bool(fake.calls("/api/guide/confirm")))
        await h.frames(10)
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        assert fake.calls("/api/guide/confirm")[0]["trigger"] == "step_done"
        # with the fence intact again the same step commits normally
        fake.confirm_echo_drift = False
        await _advance_until(h, lambda: engine_run.step_index == 1)
        assert [e.step_id for e in engine_run.confirmed_steps] == ["s1"]
        await h.engine.close()

    run(go())


def test_sequential_runs_the_same_progression_after_review_approval():
    """The core is independent of the review boundary: after ``plan_approve`` the sequential progression is
    the same (nominate on the frame, commit on the upper's own verdict, one step at a time)."""
    async def go():
        fake = FakeUpstream()
        fake.follow_fn = yes_only("s1")
        h = H(fake)
        await h.start(plan_mode=True, core_mode="sequential")
        assert h.sink.last.phase == "reviewing" and h.sink.last.plan.status == "draft"
        await h.engine.on_plan_approve()
        assert h.sink.last.phase == "running" and h.sink.last.core_mode == "sequential"
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        await h.frames(3)
        await _advance_until(h, lambda: engine_run.seq_candidate is not None)
        assert engine_run.step_index == 0
        activation = engine_run.seq_candidate.activation
        assert engine_run.seq_activation == activation
        await _advance_until(h, lambda: engine_run.step_index == 1)
        assert [e.step_id for e in engine_run.confirmed_steps] == ["s1"]
        assert engine_run.confirmed_steps[0].activation == activation
        assert engine_run.seq_activation == activation + 1 and engine_run.skipped == ()
        assert h.sink.last.plan.approved_revision == h.sink.last.plan.revision
        await h.engine.close()

    run(go())


def test_sequential_replan_confirm_names_the_current_step_and_classic_stays_unnamed():
    """§15.8: a sequential replan/unsure confirm names the current step (the upstream refuses to guess one and
    its sequential checklist is that step alone). Classic keeps master's payload exactly."""
    async def go():
        for core, expected in (("sequential", "s1"), ("classic", None)):
            fake = FakeUpstream()
            h = H(fake)
            await h.start(plan_mode=False, core_mode=core)
            await h.frames(3)
            h.clock.advance(16)  # §15.4's per-run replan gap
            await h.engine.on_replan_now()
            await _advance_until(
                h, lambda: [c for c in fake.calls("/api/guide/confirm") if c["trigger"] == "replan"])
            replan = [c for c in fake.calls("/api/guide/confirm") if c["trigger"] == "replan"][0]
            run_step = h.engine.guide.steps[h.engine.guide.step_index].id
            assert replan.get("current_step") == expected
            assert run_step == "s1"
            await h.engine.close()

    run(go())


def test_sequential_follow_answer_from_an_older_activation_cannot_nominate():
    """s1 → s2 → back to s1 while the follower's answer is open: the activation was captured at DISPATCH, so the
    old answer can neither nominate nor commit under the activation that is current now."""
    async def go():
        fake = FakeUpstream()
        first, second = dict(fake.plan_steps[0]), dict(fake.plan_steps[1])
        first["check"] = second["check"] = "user"
        fake.plan_steps = [first, second, dict(fake.plan_steps[2])]
        gate = asyncio.Event()

        async def hold(body: dict) -> None:
            if not gate.is_set():
                await gate.wait()

        fake.follow_hold = hold
        fake.follow_fn = yes_only("s1")
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: bool(fake.calls("/api/guide/follow")))
        assert h.engine.guide.seq_activation == 0 and h.engine.guide.seq_candidate is None
        # the user finishes s1 and s2, then talks the run back to s1 — three activations later
        await h.engine.on_step_ack("s1")
        await h.engine.on_step_ack("s2")
        assert h.sink.last.step_index == 2
        fake.talk_answers.append({"go_to": "s1", "reply": "처음부터 해요"})
        await h.engine.on_talk("처음으로 돌아가")
        await _advance_until(h, lambda: h.sink.last.step_index == 0)
        engine_run = h.engine.guide
        assert engine_run.seq_activation == 3
        gate.set()  # the s1 answer from activation 0 arrives now
        await h.engine.advance(200)
        await h.frames(5)
        assert engine_run.step_index == 0 and engine_run.seq_candidate is None
        assert engine_run.confirmed_steps == ()
        await h.engine.close()

    run(go())


def test_sequential_yes_without_the_step_echo_holds():
    """The upper's yes must name the step it is about: a ``yes`` with no ``step_id`` echo is not a verdict about
    this step, so nothing moves until a conforming answer arrives."""
    async def go():
        fake = FakeUpstream()
        fake.follow_fn = yes_only("s1")
        fake.confirm_omit_step_echo = True
        h = H(fake)
        await h.start(plan_mode=False, core_mode="sequential")
        await _advance_until(h, lambda: bool(fake.calls("/api/guide/confirm")))
        await h.frames(10)
        engine_run = h.engine.guide
        assert engine_run.step_index == 0 and engine_run.confirmed_steps == ()
        fake.confirm_omit_step_echo = False  # a conforming answer now commits the same activation
        await _advance_until(h, lambda: engine_run.step_index == 1)
        assert [e.step_id for e in engine_run.confirmed_steps] == ["s1"]
        await h.engine.close()

    run(go())


def test_classic_core_keeps_the_whole_remaining_plan_checklist_and_advances_on_a_later_yes():
    """Regression: with ``classic`` (the default) nothing changed — a later step's yes still moves the screen
    after its second sighting, exactly as master did."""
    async def go():
        fake = FakeUpstream()  # a classic plan: the checklist covers s1..s3
        fake.follow_answers.extend([{"visible": {"s3": "yes"}}, {"visible": {"s3": "yes"}}])
        h = H(fake)
        await h.start(plan_mode=False, core_mode="classic")
        await _advance_until(h, lambda: h.sink.last.step_index > 0)
        assert h.sink.last.steps_skipped == ["s1", "s2"]
        assert h.engine.guide.core_mode == "classic" and h.engine.guide.confirmed_steps == ()
        assert h.engine.plan is None  # classic without review is still the immediate flow
        await h.engine.close()

    run(go())


def test_confirmed_current_step_satisfies_the_next_steps_prerequisite():
    async def go():
        fake = FakeUpstream()
        fake.plan_steps[1] = {**fake.plan_steps[1], "requires": ["s1"]}
        fake.follow_fn = yes_only("s1")
        h = H(fake)
        try:
            await h.start(plan_mode=False, core_mode="sequential")
            await _advance_until(h, lambda: h.engine.guide.step_index == 1)
            assert [entry.step_id for entry in h.engine.guide.confirmed_steps] == ["s1"]
            assert h.sink.last.steps_user_done == []
        finally:
            await h.engine.close()
    run(go())


def test_slow_confirmation_keeps_one_candidate_and_final_step_commits_once():
    async def go():
        fake = FakeUpstream()
        fake.plan_steps = fake.plan_steps[:1]
        fake.follow_fn = yes_current
        confirm_gate = asyncio.Event()
        fake.confirm_hold = lambda body: confirm_gate.wait()
        h = H(fake)
        try:
            await h.start(plan_mode=False, core_mode="sequential")
            await _advance_until(h, lambda: bool(fake.calls("/api/guide/confirm")))
            candidate = h.engine.guide.seq_candidate
            await h.frames(100)
            assert h.engine.guide.seq_candidate is candidate
            confirm_gate.set()
            await _advance_until(h, lambda: bool(h.engine.guide.confirmed_steps))
            await h.frames(100)
            assert h.engine.guide.step_index == 0
            assert len(h.engine.guide.confirmed_steps) == 1
            assert len(fake.calls("/api/guide/confirm")) == 1
        finally:
            confirm_gate.set()
            await h.engine.close()
    run(go())

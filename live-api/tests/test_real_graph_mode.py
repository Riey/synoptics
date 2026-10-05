"""§15.8 graph core (``core_mode="graph"``) against the fake upstream.

Same harness shape as ``test_real_plan_mode.py``/``test_real_core_mode.py``: one real engine, one fake
upstream, one injected clock, ``auto_tick=False``. The graph core is exercised where it differs from the other
two cores — the follower is read about every plan step, a visual ``yes`` only nominates, the upper's own
same-node ``step_done`` commits, independent nodes may complete in either order, a fresh ``no`` revokes a
revocable claim and rechecks its dependents, and ``step_index`` is only a suggested focus.

The plan below is deliberately NOT a chain: ``s2``/``s5``/``s7``/``s8`` stand alone, ``s3``─▶``s4`` depend on
``s1``, and ``s6`` (a manual, required check) depends on ``s5``.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import Live, create_session, frame, jpeg, open_live
from fake_upstream import FakeUpstream
from test_real_plan_mode import H, _advance_until, make_real_client, run

from synoptics_live import plan_policy as pp
from synoptics_live.contracts import (
    CORE_GRAPH,
    CORE_SEQUENTIAL,
    MAX_REQUIRES,
    MAX_STEPS,
    FrameHeader,
    GuideStep,
    Plan,
    PlanEditMsg,
    StepAdd,
    StepEdit,
)
from synoptics_live.engine import EngineReject


def step(step_id: str, *, requires=(), check: str = "visual", required: bool = False,
         kind: str = "state", goal_required: bool | None = None) -> dict:
    """One wire step: no commands (the graph ledger does not need an overlay to be exercisable).

    ``required`` is the explicit-user gate (only the user's own record finishes it). ``goal_required`` is sent
    only when it is given, so a fixture that does not mention it keeps exercising the contract default: a node
    is part of the ORIGINAL GOAL unless the plan says otherwise (§15.4.8).
    """
    out = {"id": step_id, "say": f"{step_id} 안내", "done_when": f"{step_id} 상태", "commands": [],
           "requires": list(requires), "check": check, "required": required, "condition_kind": kind}
    if goal_required is not None:
        out["goal_required"] = goal_required
    return out


#            s1 ─▶ s3 ─▶ s4
#  s2 (free)            s5 (free) ─▶ s6 (user, required)
#                       s7 (required visual, free)     s8 (event, free)
GRAPH_STEPS = [
    step("s1"),
    step("s2"),
    step("s3", requires=["s1"]),
    step("s4", requires=["s3"]),
    step("s5"),
    step("s6", check="user", required=True, requires=["s5"]),
    step("s7", required=True),
    step("s8", kind="event"),
]

COMPLETION_STEPS = [step("s1"), step("s2", check="user", required=True, requires=["s1"])]

#  s1 (safety precondition: required, but NOT part of the goal) ─▶ s2 (goal) ─▶ s3 (goal) ─▶ s4 (optional)
#  s2/s3 are visual goal nodes (a committed observation finishes them); s1 is a mandatory PROCEDURAL check that
#  only the user's own record can finish, and s4 is an optional flourish nobody needs.
GOAL_STEPS = [
    step("s1", required=True, goal_required=False),
    step("s2", requires=["s1"]),
    step("s3", requires=["s2"]),
    step("s4", requires=["s3"], required=False, goal_required=False),
]

#  s1 (mandatory safety check, required, NOT a goal step) ─▶ s2 (goal) ─▶ s3 (mandatory user check)
#  ``s3``'s only DIRECT prerequisite is ``s2``, which observation may commit before ``s1`` is verified.
ACK_STEPS = [
    step("s1", required=True, goal_required=False),
    step("s2", requires=["s1"]),
    step("s3", requires=["s2"], check="user", required=True),
]


def make_fake(steps: list | None = None) -> FakeUpstream:
    fake = FakeUpstream()
    fake.plan_steps = [dict(s) for s in (GRAPH_STEPS if steps is None else steps)]
    return fake


class Verdicts:
    """The follower's per-step reading. Every step defaults to ``unsure`` — never a silent revocation."""

    def __init__(self, steps: list):
        self.visible = {s["id"]: "unsure" for s in steps}
        self.goal_seen = False

    def set(self, **verdicts: str) -> "Verdicts":
        for step_id, verdict in verdicts.items():
            self.visible[step_id] = verdict
        return self

    def answer(self, body: dict) -> dict:
        return {"visible": dict(self.visible), "goal_seen": "yes" if self.goal_seen else "no"}


async def graph_run(steps: list | None = None, verdicts: Verdicts | None = None):
    """A started graph run: ``(h, verdicts)`` with the fake's follower wired to ``verdicts``."""
    fake = make_fake(steps)
    v = verdicts or Verdicts(fake.plan_steps)
    h = H(fake)
    fake.follow_fn = v.answer
    await h.start(plan_mode=False, core_mode=CORE_GRAPH)
    return h, v


async def drive(h: H, pred, *, limit: int = 400, ms: float = 100) -> None:
    await _advance_until(h, pred, ms=ms, limit=limit)


def _graph_plan() -> Plan:
    return Plan(plan_id="p", revision=1, target="목표", goal_when="목표 상태",
                steps=[GuideStep.model_validate(s) for s in GRAPH_STEPS])


# --------------------------------------------------------------------------- start / contract (§15.8)


def test_graph_core_is_accepted_frozen_and_forwarded_upstream():
    """``graph`` joins the closed core set: it is accepted, stored on the run, sent on the plan request, and
    echoed in every state — with an explicit ledger instead of master's index-derived progress."""
    async def go():
        h, _ = await graph_run()
        run_ = h.engine.guide
        assert run_.core_mode == CORE_GRAPH and run_.plan_mode is False
        req = h.fake.calls("/api/guide/plan")[0]
        assert req["core_mode"] == CORE_GRAPH and "plan_mode" not in req
        st = h.sink.last
        assert st.core_mode == CORE_GRAPH and st.phase == "running"
        assert h.engine.plan is not None and h.engine.plan.status == "approved"
        # nothing is done yet; the enabled set is deterministic plan order, and the focus is its first node.
        assert st.steps_done == [] and st.step_statuses == {} and st.steps_skipped == []
        assert st.steps_ready == ["s1", "s2", "s5", "s7", "s8"] and st.step_index == 0
        await h.engine.close()

    run(go())


def test_other_cores_keep_an_empty_graph_ledger():
    """Classic and sequential are untouched: they carry no graph ledger at all."""
    async def go():
        for core in ("classic", CORE_SEQUENTIAL):
            h = H(make_fake())
            await h.start(plan_mode=False, core_mode=core)
            st = h.sink.last
            assert st.steps_done == [] and st.steps_ready == [] and st.step_statuses == {}
            assert h.engine.guide.graph_done == () and h.engine.guide.graph_status == {}
            await h.engine.close()

    run(go())


# --------------------------------------------------------------------------- independent order (§15.8)


def test_graph_commits_a_free_node_before_an_earlier_one():
    """``s2`` has no prerequisite, so a follower ``yes`` on it plus the upper's own ``step_done`` commits it
    while ``s1`` (earlier, still open) is untouched — index movement is never completion."""
    async def go():
        h, v = await graph_run()
        v.set(s2="yes")
        await drive(h, lambda: "s2" in h.engine.guide.graph_done)
        run_ = h.engine.guide
        st = h.sink.last
        assert run_.graph_done == ("s2",)
        assert st.steps_done == ["s2"] and st.step_statuses["s2"] == "yes"
        assert st.step_index == 0 and "s1" not in st.steps_done and st.steps_skipped == []
        # the upper was asked about the node the follower nominated, never about the suggested focus
        confirm = [c for c in h.fake.calls("/api/guide/confirm") if c["trigger"] == "step_done"]
        assert confirm and all(c["current_step"] == "s2" for c in confirm)
        # and a later s1 commits independently: both are done, in plan order
        v.set(s1="yes")
        await drive(h, lambda: "s1" in h.engine.guide.graph_done)
        assert h.sink.last.steps_done == ["s1", "s2"]
        ready = h.sink.last.steps_ready
        assert ready == ["s3", "s5", "s7", "s8"]   # s3 became enabled; s6 still waits on s5
        await h.engine.close()

    run(go())


def test_a_visual_yes_only_nominates_until_the_upper_confirms_it():
    """The follower's ``yes`` alone moves nothing; the upper's own ``step_done`` yes on that same node is the
    commit, and a ``no`` verdict there holds without confirming anything."""
    async def go():
        fake = make_fake([step("s1")])
        v = Verdicts(fake.plan_steps)
        h = H(fake)
        release = asyncio.Event()

        async def hold(body: dict) -> None:
            if body["trigger"] == "step_done":
                await release.wait()

        fake.confirm_hold = hold
        fake.follow_fn = v.answer
        await h.start(plan_mode=False, core_mode=CORE_GRAPH)
        v.set(s1="yes")
        await drive(h, lambda: bool(h.fake.calls("/api/guide/confirm")), limit=200)
        # nominated only: the node is not done, but the upper has been asked about exactly it
        assert h.engine.guide.graph_candidate is not None
        assert h.engine.guide.graph_done == () and h.sink.last.steps_done == []
        assert h.sink.last.notice is not None and h.sink.last.notice.code == "step_candidate"
        assert [c["current_step"] for c in h.fake.calls("/api/guide/confirm")] == ["s1"]
        # the upper disagrees: nothing is confirmed and the nomination is released
        fake.confirm_answers.append({"step_check": "no"})
        release.set()
        await drive(h, lambda: h.engine.stats["graph_held"] >= 1)
        assert h.engine.guide.graph_done == () and h.engine.guide.graph_candidate is None
        assert h.sink.last.notice is not None and h.sink.last.notice.code == "step_held"
        # a fresh follower yes re-nominates it, and this time the upper agrees
        await drive(h, lambda: "s1" in h.engine.guide.graph_done)
        assert h.sink.last.steps_done == ["s1"]
        await h.engine.close()

    run(go())


# --------------------------------------------------------------------------- revocation / recheck (§15.8)


def test_a_fresh_no_revokes_a_state_claim_and_is_recorded_as_history():
    async def go():
        h, v = await graph_run()
        v.set(s5="yes")
        await drive(h, lambda: "s5" in h.engine.guide.graph_done)
        v.set(s5="no")
        await drive(h, lambda: h.engine.guide.graph_status.get("s5") == "no")
        run_ = h.engine.guide
        assert "s5" not in run_.graph_done
        assert h.sink.last.steps_done == [] and h.sink.last.step_statuses["s5"] == "no"
        # a revocation is not amnesia: both observations are still on record
        assert any(e.step_id == "s5" and e.status == "yes" for e in run_.graph_history)
        assert any(e.step_id == "s5" and e.status == "no" for e in run_.graph_history)
        assert h.engine.stats["graph_revoked"] >= 1
        await h.engine.close()

    run(go())


def test_an_event_nodes_occurrence_survives_a_later_non_occurrence():
    async def go():
        h, v = await graph_run()
        v.set(s8="yes")
        await drive(h, lambda: "s8" in h.engine.guide.graph_done)
        v.set(s8="no")
        await drive(h, lambda: h.engine.guide.graph_status.get("s8") == "no")
        run_ = h.engine.guide
        # the occurrence is historical: the fresh reading is recorded, nothing is revoked
        assert "s8" in run_.graph_done
        assert h.sink.last.steps_done == ["s8"] and h.sink.last.step_statuses["s8"] == "no"
        assert h.engine.stats["graph_event_kept"] >= 1 and h.engine.stats["graph_revoked"] == 0
        await h.engine.close()

    run(go())


def test_revoking_a_prerequisite_rechecks_dependents_and_leaves_unrelated_nodes_done():
    """A verified occurrence is not still-applicable authorization: ``s4`` is an ``event`` node, but the state
    its chain rested on is gone, so it goes to recheck (``unsure``, never ``no``). ``s5`` is unrelated and stays
    done, and the whole chain is re-committable once the follower sees it again."""
    async def go():
        h, v = await graph_run()
        v.set(s1="yes")
        await drive(h, lambda: "s1" in h.engine.guide.graph_done)
        v.set(s1="yes", s3="yes")
        await drive(h, lambda: "s3" in h.engine.guide.graph_done)
        v.set(s1="yes", s3="yes", s4="yes")
        await drive(h, lambda: "s4" in h.engine.guide.graph_done)
        v.set(s1="yes", s3="yes", s4="yes", s5="yes")
        await drive(h, lambda: "s5" in h.engine.guide.graph_done)
        assert set(h.engine.guide.graph_done) == {"s1", "s3", "s4", "s5"}

        v.set(s1="no", s3="unsure", s4="unsure")
        await drive(h, lambda: "s3" not in h.engine.guide.graph_done
                     and h.engine.guide.graph_status.get("s1") == "no")
        run_ = h.engine.guide
        assert run_.graph_status["s1"] == "no" and "s1" not in run_.graph_done
        assert "s3" not in run_.graph_done and run_.graph_status["s3"] == "unsure"
        assert "s4" not in run_.graph_done and run_.graph_status["s4"] == "unsure"
        assert h.sink.last.steps_done == ["s5"] and h.sink.last.step_statuses["s5"] == "yes"
        assert h.engine.stats["graph_recheck"] >= 2 and h.engine.stats["graph_revoked"] >= 1

        v.set(s1="yes", s3="yes")
        await drive(h, lambda: "s3" in h.engine.guide.graph_done)
        assert h.sink.last.steps_done == ["s1", "s3", "s5"]
        await h.engine.close()

    run(go())


# --------------------------------------------------------------------------- explicit-user authority


def test_a_model_verdict_never_un_does_the_users_own_record():
    """§15.4.4, both ways: the frame never manufactures the user's ack, and it never un-does one either. A
    follower ``no`` on an acked node is recorded as a reading; the completion stands (as the sequential core
    already treats an acknowledged step)."""
    async def go():
        h, v = await graph_run()
        await h.engine.on_step_ack("s7")
        v.set(s7="no")
        await drive(h, lambda: h.engine.guide.graph_status.get("s7") == "no")
        run_ = h.engine.guide
        assert "s7" in run_.graph_done                       # the user's own record stands
        assert h.sink.last.steps_done == ["s7"] and h.sink.last.steps_user_done == ["s7"]
        assert h.engine.stats["graph_user_kept"] >= 1 and h.engine.stats["graph_revoked"] == 0
        assert any(e.step_id == "s7" and e.status == "no" for e in run_.graph_history)
        await h.engine.close()

    run(go())


def test_a_dependent_manual_ack_is_refused_and_a_required_step_needs_the_user():
    async def go():
        h, v = await graph_run()
        # s6 is the user's own check AND required, and s5 (its prerequisite) is not done yet
        await h.engine.on_step_ack("s6")
        assert h.engine.blocked is not None and h.engine.blocked.step_id == "s6"
        assert h.engine.blocked.requires == ["s5"]
        assert "s6" not in h.engine.guide.graph_done and h.sink.last.steps_user_done == []
        # a plain visual step is never a user ack, in any core
        with pytest.raises(EngineReject) as rej:
            await h.engine.on_step_ack("s1")
        assert rej.value.code == "invalid_edit"
        # s7 is a required VISUAL step: the frame never finishes it...
        v.set(s7="yes")
        await drive(h, lambda: bool(h.fake.calls("/api/guide/follow")), limit=100)
        await h.frames(40)
        assert not [c for c in h.fake.calls("/api/guide/confirm") if c["trigger"] == "step_done"]
        assert "s7" not in h.engine.guide.graph_done and h.engine.stats["graph_nominated"] == 0
        # ...only the user's own record does
        await h.engine.on_step_ack("s7")
        assert "s7" in h.engine.guide.graph_done
        assert h.sink.last.steps_done == ["s7"] and h.sink.last.step_statuses["s7"] == "yes"
        # and once its prerequisite is committed, s6 can be acknowledged too
        v.set(s7="yes", s5="yes")
        await drive(h, lambda: "s5" in h.engine.guide.graph_done)
        await h.engine.on_step_ack("s6")
        assert set(h.engine.guide.graph_done) == {"s5", "s6", "s7"}
        assert h.sink.last.steps_done == ["s5", "s6", "s7"]
        await h.engine.close()

    run(go())


def test_a_visual_verdict_on_a_manual_step_is_a_reading_not_an_authorization():
    async def go():
        h, v = await graph_run()
        v.set(s1="yes", s2="yes")   # s2 is `check:"user"` + `required`
        await drive(h, lambda: "s1" in h.engine.guide.graph_done)
        run_ = h.engine.guide
        assert run_.graph_status["s2"] == "yes"          # the reading is recorded...
        assert "s2" not in run_.graph_done               # ...and never becomes an authorization
        assert h.sink.last.steps_done == ["s1"]
        assert h.engine.stats["graph_nominated"] == 1     # only the visual s1 was ever nominated
        await h.engine.close()

    run(go())


# --------------------------------------------------------------------------- staleness / identity


def test_a_fresh_negative_beats_an_older_pending_upper_yes():
    async def go():
        fake = make_fake()
        v = Verdicts(fake.plan_steps)
        h = H(fake)
        release = asyncio.Event()

        async def hold(body: dict) -> None:
            if body["trigger"] == "step_done":
                await release.wait()

        fake.confirm_hold = hold
        fake.follow_fn = v.answer
        await h.start(plan_mode=False, core_mode=CORE_GRAPH)
        v.set(s5="yes")
        await drive(h, lambda: bool(fake.calls("/api/guide/confirm")), limit=200)
        assert h.engine.guide.graph_candidate is not None
        # a fresher capture says the node is not there after all, while the upper's yes is still in flight
        v.set(s5="no")
        await drive(h, lambda: h.engine.guide.graph_status.get("s5") == "no", limit=300)
        release.set()
        await h.frames(20)
        assert "s5" not in h.engine.guide.graph_done and h.sink.last.steps_done == []
        assert h.engine.guide.graph_candidate is None
        assert h.engine.stats["graph_stale_ignored"] >= 1
        await h.engine.close()

    run(go())


def test_reselect_keeps_history_but_not_old_object_authorization():
    """First acquisition keeps a pre-track ack; explicit reselection lacks same-object proof."""
    async def go():
        h, v = await graph_run()
        assert h.engine.guide.binding is None
        await h.engine.on_step_ack("s7")
        assert "s7" in h.engine.guide.graph_done
        v.set(s5="yes")
        await drive(h, lambda: "s5" in h.engine.guide.graph_done)
        assert h.engine.guide.binding is not None            # tracking bound it while the ack survived
        facts = dict(h.engine.guide.graph_facts)
        assert "s7" in h.sink.last.steps_user_done
        v.set(s5="unsure")                                   # the follower must not re-assert it afterwards
        header = FrameHeader.model_validate({
            "t": "frame", "seq": h.seq, "captured_at": 1_759_479_000_000 + h.seq, "w": 640, "h": 360,
            "select_box": {"x": 0.3, "y": 0.2, "width": 0.25, "height": 0.2}})
        h.seq += 1
        await h.engine.on_frame(header, jpeg(640, 360))
        await h.engine.advance(100)
        run_ = h.engine.guide
        assert h.sink.last.steps_done == [] and h.sink.last.steps_user_done == []
        assert run_.graph_status.get("s5", "unsure") == "unsure"
        assert all(fact in run_.graph_history for fact in facts.values())
        # A later revision of the same plan cannot resurrect the invalidated object's confirmations.
        current = {**h.fake._current_plan(), "plan_revision": h.fake.plan_revision + 1}
        assert h.engine._pin_upstream_plan(run_, current)
        assert h.engine.snapshot().steps_done == []
        await h.engine.close()

    run(go())


def test_talk_marks_are_authority_and_a_pointer_move_finishes_nothing():
    """An explicit user mark is authority in both directions; a ``go_to`` is a pointer move, so in graph mode it
    changes no completion (the suggestion is deterministic)."""
    async def go():
        # a standing "no" on a node nobody has acted on keeps the follower's readings from being uniformly
        # unsure (which would make the core propose a replan and hold the talk behind it).
        v = Verdicts([dict(s) for s in GRAPH_STEPS]).set(s5="no")
        h, _ = await graph_run(verdicts=v)
        assert h.engine.guide.step_index == 0
        await drive(h, lambda: h.engine.guide.binding is not None)   # a talk needs a tracked scene
        assert h.engine.guide.step_index == 0
        async def talk(text: str) -> None:
            await h.engine.on_talk(text)
            await drive(h, lambda: not h.engine.guide.talk_pending)   # a talk holds the lane until it answers

        h.fake.talk_answers.append({"step_mark": "done"})   # "이거 했어" — applies to the suggested focus, s1
        await talk("이거 했어요")
        assert set(h.engine.guide.graph_done) == {"s1"}
        assert h.sink.last.steps_user_done == ["s1"] and h.sink.last.steps_done == ["s1"]
        assert h.engine.guide.step_index == 1               # s2 is the first safe-action-eligible node
        # a mark applies to the run's own suggested focus — s2 now, so the next mark lands there
        # a pointer move finishes nothing, whatever index it names
        h.fake.talk_answers.append({"go_to": "s7"})
        await talk("7단계로 가줘")
        assert set(h.engine.guide.graph_done) == {"s1"}
        assert h.engine.guide.step_index == 1               # a go_to never moves the deterministic suggestion
        assert "s7" not in h.sink.last.steps_done
        # the user can say a node is not done: the reading is recorded, and unrelated completions stand
        h.fake.talk_answers.append({"step_mark": "not_done"})
        await talk("아직 안 했어요")
        run_ = h.engine.guide
        assert run_.graph_status["s2"] == "no" and set(run_.graph_done) == {"s1"}
        await h.engine.close()

    run(go())


# --------------------------------------------------------------------------- completion (§15.4.5)


def test_goal_completion_waits_for_required_authorization_then_confirms():
    async def go():
        fake = make_fake(COMPLETION_STEPS)
        v = Verdicts(fake.plan_steps)
        h = H(fake)
        fake.follow_fn = v.answer
        await h.start(plan_mode=False, core_mode=CORE_GRAPH)
        v.set(s1="yes", s2="yes")
        await drive(h, lambda: "s1" in h.engine.guide.graph_done)
        # the picture looks finished, but the required check has not been authorized by the user
        v.goal_seen = True
        fake.confirm_answers.append({"status": "visually_satisfied"})
        await drive(h, lambda: h.engine.blocked is not None)
        assert h.engine.guide.completion == "none" and h.sink.last.completion == "none"
        assert h.engine.blocked.step_id == "s2"
        assert h.sink.last.notice is not None and h.sink.last.notice.code == "required_check"
        assert h.engine.stats["completion_held"] >= 1     # a satisfied goal_check was refused
        # the user's own record satisfies it, and only then can the goal be checked
        await h.engine.on_step_ack("s2")
        assert h.sink.last.steps_done == ["s1", "s2"]
        fake.confirm_answers.append({"status": "visually_satisfied"})
        await drive(h, lambda: h.engine.guide.completion == "checking")
        assert h.sink.last.completion == "checking" and h.sink.last.phase == "running"
        fake.confirm_answers.append({"status": "visually_satisfied"})
        await drive(h, lambda: h.engine.guide.completion == "confirmed")
        assert h.sink.last.completion == "confirmed" and h.sink.last.phase == "running"
        await h.engine.on_confirm_done()
        assert h.sink.last.phase == "completed"
        assert h.sink.last.steps_done == ["s1", "s2"]
        await h.engine.close()

    run(go())


# --------------------------------------------- §15.4.8 goal steps vs safety preconditions (round 2)


def test_a_goal_node_is_observable_while_its_mandatory_safety_check_is_unverified():
    """Reading a goal state is not permission to perform it: ``s2`` may be observed and committed while ``s1``
    (mandatory, but NOT part of the goal) is unverified. The downstream node is never suggested as safe, and the
    outstanding check is reported beside the goal instead of blocking it."""
    async def go():
        h, v = await graph_run(GOAL_STEPS)
        v.set(s2="yes")
        await drive(h, lambda: "s2" in h.engine.guide.graph_done)
        st = h.sink.last
        assert st.steps_done == ["s2"] and st.step_statuses["s2"] == "yes"
        assert st.steps_ready == ["s1"]                  # the mandatory check is the safe next action, not s3
        assert st.pending_required_checks == ["s1"]
        assert st.step_index == 0                        # the suggestion points at the unverified check
        assert "s3" not in st.steps_ready and st.steps_skipped == []
        # the safety check can only be finished by the user's own record, and then s3 is the safe next action
        await h.engine.on_step_ack("s1")
        assert h.sink.last.steps_done == ["s1", "s2"]
        assert h.sink.last.steps_ready == ["s3"] and h.sink.last.pending_required_checks == []
        await h.engine.close()

    run(go())


def test_a_goal_required_prerequisite_gap_blocks_observation_and_commit():
    """A ``goal_required`` prerequisite gap is not a reading licence: ``s3`` requires the goal node ``s2``, so a
    follower ``yes`` on it nominates nothing while ``s2`` is undone."""
    async def go():
        h, v = await graph_run(GOAL_STEPS)
        v.set(s3="yes")
        await drive(h, lambda: h.engine.stats["follow"] >= 2)
        assert h.engine.guide.graph_done == () and h.engine.guide.graph_candidate is None
        assert h.engine.stats["graph_nominated"] == 0
        assert not [c for c in h.fake.calls("/api/guide/confirm") if c["trigger"] == "step_done"]
        assert h.sink.last.steps_done == [] and h.sink.last.pending_required_checks == ["s1"]
        await h.engine.close()

    run(go())


def test_the_goal_confirms_with_its_mandatory_check_pending_and_an_optional_step_undone():
    """The ORIGINAL GOAL is what ``goal_required`` nodes define: with ``s2``/``s3`` committed it confirms even
    though ``s1`` (mandatory, non-goal) is unverified, the optional ``s4`` is never needed, and the outstanding
    check is frozen into the completed snapshot instead of being treated as the goal unmet."""
    async def go():
        h, v = await graph_run(GOAL_STEPS)
        v.set(s2="yes")
        await drive(h, lambda: "s2" in h.engine.guide.graph_done)
        v.set(s2="yes", s3="yes")
        await drive(h, lambda: "s3" in h.engine.guide.graph_done)
        assert h.engine.blocked is None and h.sink.last.pending_required_checks == ["s1"]
        v.goal_seen = True
        h.fake.confirm_answers.append({"status": "visually_satisfied"})
        await drive(h, lambda: h.engine.guide.completion == "checking")
        h.fake.confirm_answers.append({"status": "visually_satisfied"})
        await drive(h, lambda: h.engine.guide.completion == "confirmed")
        st = h.sink.last
        assert st.steps_done == ["s2", "s3"] and "s4" not in st.steps_done
        assert st.pending_required_checks == ["s1"]
        await h.engine.on_confirm_done()
        assert h.sink.last.phase == "completed"
        assert h.sink.last.steps_done == ["s2", "s3"]
        assert h.sink.last.pending_required_checks == ["s1"]   # frozen with the confirmation
        await h.engine.close()

    run(go())


def test_a_manual_ack_cannot_bypass_a_mandatory_ancestor_hidden_behind_an_observed_node():
    """§15.4.8: ``s3``'s DIRECT prerequisite ``s2`` is already done — and was committed by OBSERVATION while the
    mandatory check ``s1`` is still unverified. The ack is still refused: an intermediate node must never hide a
    mandatory ancestor. The check itself stays ackable, so nothing deadlocks."""
    async def go():
        h, v = await graph_run(ACK_STEPS)
        # a standing "yes" on the safety check keeps the follower's readings from being uniformly unsure (which
        # would have the core propose a replan and hold every ack behind it); a required node is never nominated.
        v.set(s1="yes", s2="yes")
        await drive(h, lambda: "s2" in h.engine.guide.graph_done)     # observation through the non-goal gap
        assert h.sink.last.steps_done == ["s2"] and "s1" not in h.engine.guide.graph_done
        await h.engine.on_step_ack("s3")
        assert h.engine.blocked is not None and h.engine.blocked.step_id == "s3"
        assert h.engine.blocked.requires == ["s1"]
        assert "s3" not in h.engine.guide.graph_done and h.sink.last.steps_user_done == []
        assert h.sink.last.steps_ready == ["s1"]        # the safety check is the safe next action
        await h.engine.on_step_ack("s1")                # …and it is the user's own to close
        assert h.sink.last.steps_done == ["s1", "s2"] and h.engine.blocked is None
        await h.engine.on_step_ack("s3")
        assert h.sink.last.steps_done == ["s1", "s2", "s3"]
        await h.engine.close()

    run(go())


def test_a_closed_ledger_asks_nothing_while_observation_continues():
    """§15.4.8 emission: once every node is done the suggestion falls back to the last node, but no action
    command and no spoken instruction may be produced — while the follow/confirm loop keeps observing."""
    async def go():
        h, v = await graph_run(COMPLETION_STEPS)
        v.set(s1="yes")
        await drive(h, lambda: "s1" in h.engine.guide.graph_done)
        await h.engine.on_step_ack("s2")
        assert set(h.engine.guide.graph_done) == {"s1", "s2"}
        assert h.engine.guide.step_index == 1            # the fallback index still points somewhere real
        says_before, follows_before = len(h.sink.says), h.engine.stats["follow"]
        await drive(h, lambda: h.engine.stats["follow"] >= follows_before + 2)
        st = h.sink.last
        assert st.phase == "running" and st.steps_done == ["s1", "s2"]
        assert st.overlay is not None and st.overlay.commands == []     # tracked, but nothing asked
        assert [line for line, _ in h.sink.says[says_before:] if line.startswith(("1단계", "2단계"))] == []
        # the observation channel is untouched: a fresh reading still lands
        v.set(s1="no")
        await drive(h, lambda: "s1" not in h.engine.guide.graph_done)
        assert h.engine.guide.graph_status["s1"] == "no"
        await h.engine.close()

    run(go())


def test_slow_upper_confirmation_does_not_replace_a_newer_occluded_reading():
    async def go():
        fake = make_fake([step("s1"), step("s2")])
        v = Verdicts(fake.plan_steps).set(s2="no")
        h = H(fake)
        release = asyncio.Event()

        async def hold(body):
            if body["trigger"] == "step_done":
                await release.wait()

        fake.confirm_hold, fake.follow_fn = hold, v.answer
        await h.start(plan_mode=False, core_mode=CORE_GRAPH)
        v.set(s1="yes")
        await drive(h, lambda: h.engine.guide.graph_candidate is not None)
        v.set(s1="unsure")
        await drive(h, lambda: h.engine.guide.graph_status.get("s1") == "unsure")
        release.set()
        await drive(h, lambda: "s1" in h.sink.last.steps_done)
        assert h.sink.last.step_statuses["s1"] == "unsure"
        fact = h.engine.guide.graph_facts["s1"]
        assert fact.seq < h.engine.guide.graph_evidence["s1"].seq
        assert fact.status == "yes" and fact.accepted_via == "upper.bound"
        v.set(s1="no")
        await drive(h, lambda: "s1" not in h.sink.last.steps_done)
        assert fact in h.engine.guide.graph_history
        await h.engine.close()

    run(go())


@pytest.mark.parametrize("accept", [True, False])
def test_replan_maps_confirmed_conditions_without_forging_new_observations(accept):
    async def go():
        steps = [step("s1"), step("s2", requires=["s1"], check="user", required=True), step("s3")]
        v = Verdicts(steps).set(s1="yes", s3="no")
        h, _ = await graph_run(steps, v)
        await drive(h, lambda: "s1" in h.sink.last.steps_done)
        await h.engine.on_step_ack("s2")
        original_facts = dict(h.engine.guide.graph_facts)
        v.set(s1="unsure")
        await drive(h, lambda: h.sink.last.step_statuses.get("s1") == "unsure")
        h.clock.advance(16)
        renamed = [{**steps[0], "id": "s4", "say": "같은 조건의 새 안내"},
                   {**steps[1], "id": "s5", "requires": ["s4"]}, {**steps[2], "id": "s6"}]
        h.fake.confirm_answers.append({"replan": renamed})
        await h.engine.on_replan_now()
        await drive(h, lambda: h.sink.last.proposal is not None)
        assert h.sink.last.steps_done == ["s1", "s2"]
        await h.engine.on_proposal(accept=accept)
        expected = ["s4", "s5"] if accept else ["s1", "s2"]
        assert h.sink.last.steps_done == expected
        assert h.sink.last.steps_user_done == expected[1:]
        assert h.sink.last.step_statuses == {}  # no frame has observed the newly installed revision
        for old, new in zip(("s1", "s2"), expected):
            assert h.engine.guide.graph_facts[new] is original_facts[old]
        v.visible = {expected[0]: "unsure", expected[1]: "unsure", "s6" if accept else "s3": "no"}
        await drive(h, lambda: h.sink.last.step_statuses.get(expected[0]) == "unsure")
        assert h.sink.last.steps_done == expected
        # Current counterevidence revokes the condition and its dependent manual authorization, not history.
        v.visible[expected[0]] = "no"
        await drive(h, lambda: not h.sink.last.steps_done)
        assert h.sink.last.steps_user_done == []
        assert all(fact in h.engine.guide.graph_history for fact in original_facts.values())
        await h.engine.close()

    run(go())


@pytest.mark.parametrize("change", [
    {"done_when": "다른 상태"}, {"check": "measure"}, {"required": True}, {"goal_required": False},
    {"condition_kind": "event"}, {"targets": ["다른 부품"]}, {"requires": ["s5"]},
])
def test_changed_condition_rechecks_only_its_dependency_subgraph(change):
    previous = _graph_plan()
    steps = [s.model_copy(update=change) if s.id == "s1" else s for s in previous.steps]
    current = previous.model_copy(update={"revision": 2, "steps": steps})
    assert pp.graph_fact_mapping(previous, current) == {sid: sid for sid in ("s2", "s5", "s6", "s7", "s8")}


def test_ambiguous_identical_parts_cannot_transfer_their_dependent_approval():
    steps = [GuideStep.model_validate(step("s1")),
             GuideStep.model_validate({**step("s1"), "id": "s2"}),
             GuideStep.model_validate(step("s3", requires=["s1"], check="user")),
             GuideStep.model_validate(step("s4"))]
    previous = Plan(plan_id="p", revision=1, target="목표", goal_when="목표 상태", steps=steps)
    current = previous.model_copy(update={"revision": 2})
    assert pp.graph_fact_mapping(previous, current) == {"s4": "s4"}
    assert pp.graph_fact_mapping(previous, current.model_copy(update={"target": "다른 대상"})) == {}
    assert pp.graph_fact_mapping(previous, current.model_copy(update={"goal_when": "다른 목표"})) == {}


def test_explicit_user_correction_retracts_an_event_and_dependent_authorization():
    async def go():
        steps = [step("s1", kind="event", check="user"), step("s2", requires=["s1"], check="user"), step("s3")]
        h, v = await graph_run(steps, Verdicts(steps).set(s3="no"))
        await h.engine.on_step_ack("s1")
        await h.engine.on_step_ack("s2")
        facts = tuple(h.engine.guide.graph_facts.values())
        # This correction is different from a later frame merely not showing a past event.
        h.engine._graph_undo(h.engine.guide, "s1")
        snapshot = h.engine.snapshot()
        assert snapshot.steps_done == [] and snapshot.steps_user_done == []
        assert snapshot.step_statuses["s1"] == "no"
        assert all(fact in h.engine.guide.graph_history for fact in facts)
        await h.engine.close()

    run(go())


def test_graph_helpers_select_enabled_nodes_in_plan_order():
    plan = _graph_plan()
    steps = list(plan.steps)
    assert pp.graph_dependents(steps, "s1") == ("s3", "s4")
    assert pp.graph_dependents(steps, "s5") == ("s6",)
    assert pp.graph_dependents(steps, "s7") == ()
    assert pp.graph_ready_ids(steps, ()) == ("s1", "s2", "s5", "s7", "s8")
    assert pp.graph_ready_ids(steps, ("s2",)) == ("s1", "s5", "s7", "s8")
    assert pp.graph_ready_ids(steps, ("s1", "s2")) == ("s3", "s5", "s7", "s8")
    assert pp.graph_ready_ids(steps, ("s1", "s2", "s5")) == ("s3", "s6", "s7", "s8")
    assert pp.graph_focus_index(steps, ()) == 0
    assert pp.graph_focus_index(steps, ("s1", "s2")) == 2                 # s3 is the first enabled node
    assert pp.graph_focus_index(steps, ("s1", "s2", "s5")) == 2           # plan order still wins over s6
    assert pp.graph_focus_index(steps, tuple(step.id for step in steps)) == len(steps) - 1
    # §15.8/§15.4.8: what keeps the goal unachieved is ``goal_required``. Every node of GRAPH_STEPS is a goal
    # step, so the pending set is simply everything undone and completion needs all of them.
    all_ids = [s.id for s in plan.steps]
    assert pp.graph_completion_pending(plan, ()) == all_ids
    assert pp.graph_completion_ready(plan, ("s6",)) is False
    assert pp.graph_completion_ready(plan, tuple(all_ids)) is True
    # a MIXED plan holds only on its goal nodes: the mandatory safety check does not keep the goal unachieved.
    mixed = Plan(plan_id="p", revision=1, target="목표", goal_when="목표 상태",
                 steps=[GuideStep.model_validate(s) for s in GOAL_STEPS])
    assert pp.graph_completion_pending(mixed, ("s2", "s3")) == []
    assert pp.graph_completion_pending(mixed, ("s3",)) == ["s2"]
    assert pp.graph_ack_index(plan, "s7") == 6                             # required visual
    assert pp.graph_ack_index(plan, "s6") == 5                             # manual
    with pytest.raises(pp.PlanEditError):
        pp.graph_ack_index(plan, "s1")                                     # plain visual
    blocker = pp.graph_prerequisite_block(plan, "s3", ())
    assert blocker is not None and blocker.requires == ["s1"]
    assert pp.graph_prerequisite_block(plan, "s3", ("s1",)) is None


def test_condition_kind_is_case_folded_defaulted_and_editable():
    base = {"say": "단계", "done_when": "상태", "commands": []}
    assert GuideStep.model_validate({**base, "id": "s1"}).condition_kind == "state"
    assert GuideStep.model_validate({**base, "id": "s1", "condition_kind": "EVENT"}).condition_kind == "event"
    with pytest.raises(ValueError):
        GuideStep.model_validate({**base, "id": "s1", "condition_kind": "sometime"})

    plan = Plan(plan_id="p", revision=1, status="draft", target="목표", goal_when="목표 상태",
                steps=[GuideStep.model_validate({**base, "id": "s1"})])
    edited = pp.apply_plan_edit(plan, PlanEditMsg(
        type="plan_edit", edits=[StepEdit(step_id="s1", condition_kind="event")]))
    assert edited.steps[0].condition_kind == "event" and plan.steps[0].condition_kind == "state"
    added = pp.apply_plan_edit(plan, PlanEditMsg(
        type="plan_edit", add=[StepAdd(say="새 단계", done_when="상태", condition_kind="EVENT")]))
    assert added.steps[1].condition_kind == "event"
    assert pp.apply_plan_edit(plan, PlanEditMsg(
        type="plan_edit", add=[StepAdd(say="새 단계", done_when="상태")])).steps[1].condition_kind == "state"

    # §15.4.8: the same shape carries goal_required — accepted on the wire, and switchable on a draft step or
    # on a new one. The contract default (a node is a goal step unless the plan says otherwise) is exercised
    # end-to-end by the fixtures that leave the field out.
    assert GuideStep.model_validate({**base, "id": "s1", "goal_required": False}).goal_required is False
    relaxed = pp.apply_plan_edit(plan, PlanEditMsg(
        type="plan_edit", edits=[StepEdit(step_id="s1", goal_required=False)]))
    assert relaxed.steps[0].goal_required is False and plan.steps[0].goal_required is True
    added_safety = pp.apply_plan_edit(plan, PlanEditMsg(
        type="plan_edit", add=[StepAdd(say="새 단계", done_when="상태", goal_required=False)]))
    assert added_safety.steps[1].goal_required is False


def test_a_node_may_name_every_other_step_as_a_prerequisite():
    assert MAX_REQUIRES == MAX_STEPS - 1 == 15
    steps = [step(f"s{i}") for i in range(1, 17)]
    steps[15]["requires"] = [f"s{i}" for i in range(1, 16)]
    plan = Plan(plan_id="p", revision=1, target="목표", goal_when="목표 상태",
                steps=[GuideStep.model_validate(s) for s in steps])
    assert len(plan.steps[15].requires) == 15
    with pytest.raises(ValueError):
        GuideStep.model_validate({**step("s16"), "requires": [f"s{i}" for i in range(1, 17)]})


def test_the_graph_ledger_must_name_plan_steps():
    from synoptics_live.contracts import GuideState

    plan = _graph_plan()
    GuideState(phase="running", core_mode=CORE_GRAPH, plan=plan, steps_done=["s1"], steps_ready=["s2"],
               step_statuses={"s1": "yes"}, pending_required_checks=["s7"])
    with pytest.raises(ValueError):
        GuideState(phase="running", core_mode=CORE_GRAPH, plan=plan, steps_done=["s9"])
    with pytest.raises(ValueError):
        GuideState(phase="running", core_mode=CORE_GRAPH, plan=plan, step_statuses={"s9": "yes"})
    with pytest.raises(ValueError):
        GuideState(phase="running", core_mode=CORE_GRAPH, plan=plan, pending_required_checks=["s9"])
    with pytest.raises(ValueError):
        GuideState(phase="idle", steps_done=["s1"])          # a ledger without a plan is not a state
    with pytest.raises(ValueError):
        GuideState(phase="idle", pending_required_checks=["s1"])


def test_websocket_completion_uses_goal_nodes_not_pending_safety_checks():
    """The transport gate must agree with the engine: physical completion is not procedural approval."""
    fake = make_fake([step("s1", check="user", required=True, goal_required=False),
                      step("s2", requires=["s1"])])
    fake.follow_fn = lambda body: {"visible": {"s1": "unsure", "s2": "yes"}, "goal_seen": "yes"}
    fake.confirm_answers.extend([{"status": "visually_satisfied"} for _ in range(4)])
    client = make_real_client(fake)
    try:
        session = create_session(client)
        with open_live(client, session["session_id"]) as ws:
            live = Live(ws)
            live.hello(session["token"])
            live.send({"type": "start", "goal": "조립 상태 확인", "consent_ai": True,
                       "plan_mode": True, "core_mode": "graph"})
            hi = live.until_type("capture_hi")
            ws.send_bytes(frame(500, w=1024, h=576, hi_req=hi["req_id"]))
            live.until(lambda m: m["type"] == "state" and m["phase"] == "reviewing")
            live.send({"type": "plan_approve"})
            live.until(lambda m: m["type"] == "state" and m["phase"] == "running")
            # Before the goal node is confirmed, even an explicit final acknowledgment is refused.
            live.send({"type": "confirm_done"})
            assert live.until_type("error")["code"] == "required_check"
            confirmed = live.drive(lambda s: s["completion"] == "confirmed", max_frames=800, pause_s=.02)
            assert confirmed["steps_done"] == ["s2"]
            assert confirmed["pending_required_checks"] == ["s1"]
            live.send({"type": "confirm_done"})
            result = live.until(lambda m: m["type"] == "error" or
                                (m["type"] == "state" and m["phase"] == "completed"))
            assert result["type"] == "state", result
            assert result["completion"] == "user_confirmed"
            assert result["pending_required_checks"] == ["s1"]
            assert result["steps_user_done"] == []  # no forged safety acknowledgment
    finally:
        client.__exit__(None, None, None)

"""§15.4 규칙 단위 테스트 — wire를 거치지 않고 plan_policy만 직접 잰다.

mock 엔진과 실제 엔진이 같은 규칙을 쓰므로, 규칙 자체를 여기서 고정한다. I/O도 시계도 없다.
"""

from __future__ import annotations

import pytest

from synoptics_live.contracts import (
    Blocked,
    GuideStep,
    MaterialInput,
    Plan,
    PlanEditMsg,
    StepAdd,
    StepEdit,
    overlay_command_from_step,
)
from synoptics_live import plan_policy as policy
from synoptics_live.plan_policy import PlanEditError

FOCUS = {"kind": "focus", "anchor": "target", "pad": 0.15}


def step(step_id: str, *, required: bool = False, check: str = "visual", requires: list[str] | None = None,
         say: str | None = None) -> GuideStep:
    return GuideStep.model_validate({
        "id": step_id,
        "say": say or f"{step_id} 단계를 하세요",
        "done_when": f"{step_id} 상태가 되었다",
        "commands": [FOCUS],
        "check": check,
        "required": required,
        "requires": requires or [],
    })


def plan(steps: list[GuideStep], *, status: str = "draft", revision: int = 1,
         materials: list[dict] | None = None) -> Plan:
    return Plan.model_validate({
        "plan_id": "p1",
        "revision": revision,
        "status": status,
        "approved_revision": revision if status == "approved" else None,
        "target": "상자",
        "goal_when": "상자가 봉함되어 있다",
        "materials": materials or [],
        "steps": [s.model_dump() for s in steps],
    })


# ------------------------------------------------------------------ 자료

def test_materials_get_ids_and_chars_but_the_text_stays_out_of_the_state() -> None:
    inputs = (MaterialInput(title="포장 지침", version="v3", text="가" * 10),
              MaterialInput(title="사내 규격", text="나" * 3))
    materials = policy.materials_from_inputs(inputs, now_ms=1700)
    assert [m.id for m in materials] == ["m1", "m2"]
    assert [m.chars for m in materials] == [10, 3]
    assert materials[0].version == "v3" and materials[1].version is None
    assert "text" not in materials[0].model_dump()
    assert policy.material_texts(inputs) == {"m1": "가" * 10, "m2": "나" * 3}


# ------------------------------------------------------------------ 진행 판정

def test_done_never_counts_a_skipped_step_and_user_done_counts_anywhere() -> None:
    steps = [step("s1"), step("s2"), step("s3")]
    plan_ = plan(steps)
    assert policy.done_step_ids(plan_, step_index=2, skipped=["s1"], user_done=[]) == {"s2"}
    assert policy.done_step_ids(plan_, step_index=0, skipped=[], user_done=["s3"]) == {"s3"}


def test_completion_needs_every_required_step_even_after_it_was_skipped() -> None:
    steps = [step("s1"), step("s2", required=True), step("s3")]
    plan_ = plan(steps)
    assert not policy.completion_ready(plan_, step_index=3, skipped=["s2"], user_done=[])
    assert policy.required_pending(plan_, step_index=3, skipped=["s2"], user_done=[]) == ["s2"]
    assert policy.completion_ready(plan_, step_index=3, skipped=["s2"], user_done=["s2"])


def test_a_prerequisite_that_is_not_done_blocks_the_step() -> None:
    steps = [step("s1"), step("s2", requires=["s1"])]
    plan_ = plan(steps)
    blocked = policy.prerequisite_block(plan_, 1, step_index=0, skipped=[], user_done=[])
    assert blocked is not None and blocked.step_id == "s2" and blocked.requires == ["s1"]
    assert policy.prerequisite_block(plan_, 1, step_index=1, skipped=[], user_done=[]) is None
    assert policy.prerequisite_block(plan_, 1, step_index=0, skipped=[], user_done=["s1"]) is None


def test_a_required_step_is_never_skipped_and_a_user_check_step_needs_the_user() -> None:
    steps = [step("s1"), step("s2", required=True), step("s3"), step("s4", check="user"), step("s5")]
    plan_ = plan(steps)
    required = policy.skip_block(plan_, 2, step_index=0, skipped=[], user_done=[])
    assert required is not None and required.step_id == "s2" and required.reason == policy.REQUIRED_HOLD_REASON
    user = policy.skip_block(plan_, 4, step_index=2, skipped=[], user_done=[])
    assert user is not None and user.step_id == "s4" and user.reason == policy.USER_CHECK_HOLD_REASON
    assert policy.skip_block(plan_, 4, step_index=2, skipped=[], user_done=["s4"]) is None
    assert policy.skip_block(plan_, 2, step_index=0, skipped=[], user_done=["s2"]) is None
    assert policy.skip_block(plan_, 2, step_index=0, skipped=[], user_done=[]) is not None


def test_advance_to_passes_visual_steps_as_skipped_and_holds_on_everything_else() -> None:
    steps = [step("s1"), step("s2"), step("s3", required=True), step("s4")]
    plan_ = plan(steps)
    jump = policy.advance_to(plan_, 2, step_index=0, skipped=[], user_done=[])
    assert jump.moved and jump.index == 2 and jump.skipped == ("s1", "s2") and jump.blocked is None
    held = policy.advance_to(plan_, 3, step_index=0, skipped=[], user_done=[])
    assert not held.moved and held.index == 0 and held.blocked is not None
    assert held.blocked.step_id == "s3"
    back = policy.advance_to(plan_, 0, step_index=1, skipped=[], user_done=[])
    assert not back.moved and back.index == 1 and back.blocked is None


def test_advance_by_one_holds_a_required_or_user_step_until_it_is_acknowledged() -> None:
    steps = [step("s1", required=True), step("s2")]
    plan_ = plan(steps)
    held = policy.advance_by_one(plan_, step_index=0, skipped=[], user_done=[])
    assert not held.moved and held.blocked is not None and held.blocked.reason == policy.REQUIRED_HOLD_REASON
    acked = policy.advance_by_one(plan_, step_index=0, skipped=[], user_done=[], user_ack=True)
    assert acked.moved and acked.index == 1
    user_steps = plan([step("s1", check="measure"), step("s2")])
    held_user = policy.advance_by_one(user_steps, step_index=0, skipped=[], user_done=[])
    assert not held_user.moved and held_user.blocked is not None
    assert held_user.blocked.reason == policy.USER_CHECK_HOLD_REASON
    assert policy.advance_by_one(user_steps, step_index=0, skipped=[], user_done=["s1"]).moved
    last = policy.advance_by_one(plan([step("s1")]), step_index=0, skipped=[], user_done=[])
    assert last.moved and last.index == 0  # 마지막 단계: 완료 판정으로 넘어간다


def test_step_ack_applies_only_to_user_or_measure_steps() -> None:
    plan_ = plan([step("s1"), step("s2", check="user"), step("s3", check="measure")])
    assert policy.ack_index(plan_, "s2") == 1
    assert policy.ack_index(plan_, "s3") == 2
    with pytest.raises(PlanEditError):
        policy.ack_index(plan_, "s1")
    with pytest.raises(PlanEditError):
        policy.ack_index(plan_, "s9")


# ------------------------------------------------------------------ 수정 (§15.5)

def test_edit_patches_a_step_and_bumps_the_draft_revision() -> None:
    plan_ = plan([step("s1"), step("s2")])
    edited = policy.apply_plan_edit(plan_, PlanEditMsg(type="plan_edit", edits=[
        StepEdit(step_id="s2", say="봉함 전에 안을 보여 주세요", required=True, check="user",
                 targets=["상자 내부"], requires=["s1"]),
    ]))
    assert edited.revision == 2 and edited.status == "draft" and edited.approved_revision is None
    assert edited.plan_id == plan_.plan_id
    assert edited.steps[0].say == plan_.steps[0].say            # 손대지 않은 단계는 그대로
    assert edited.steps[1].required and edited.steps[1].check == "user"
    assert edited.steps[1].targets == ["상자 내부"] and edited.steps[1].requires == ["s1"]
    assert plan_.steps[1].required is False                     # 원본은 불변


def test_details_survive_other_edits_but_explicit_null_clears_them() -> None:
    original = plan([step("s1").model_copy(update={"details": "양쪽 끝을 서로 다른 구멍에 끼우세요."})])
    renamed = policy.apply_plan_edit(original, PlanEditMsg(type="plan_edit", edits=[
        StepEdit(step_id="s1", say="부품을 끼우세요"),
    ]))
    assert renamed.steps[0].details == original.steps[0].details
    cleared = policy.apply_plan_edit(renamed, PlanEditMsg.model_validate({
        "type": "plan_edit", "edits": [{"step_id": "s1", "details": None}],
    }))
    assert cleared.steps[0].details is None
    assert original.steps[0].details == "양쪽 끝을 서로 다른 구멍에 끼우세요."


def test_edit_can_add_remove_and_reorder() -> None:
    plan_ = plan([step("s1"), step("s2")])
    added = policy.apply_plan_edit(plan_, PlanEditMsg(type="plan_edit", add=[
        StepAdd(say="테이프로 봉하세요", done_when="테이프가 붙어 있다", requires=["s2"]),
    ]))
    assert [s.id for s in added.steps] == ["s1", "s2", "s3"]
    removed = policy.apply_plan_edit(plan_, PlanEditMsg(type="plan_edit", remove=["s2"]))
    assert [s.id for s in removed.steps] == ["s1"]
    reordered = policy.apply_plan_edit(plan_, PlanEditMsg(type="plan_edit", order=["s2", "s1"]))
    assert [s.id for s in reordered.steps] == ["s2", "s1"]
    renamed = policy.apply_plan_edit(plan_, PlanEditMsg(type="plan_edit", goal_when="상자가 테이프로 봉해져 있다"))
    assert renamed.goal_when == "상자가 테이프로 봉해져 있다"


@pytest.mark.parametrize("message", [
    PlanEditMsg(type="plan_edit", edits=[StepEdit(step_id="s9", required=True)]),
    PlanEditMsg(type="plan_edit", remove=["s1", "s2"]),
    PlanEditMsg(type="plan_edit", order=["s1"]),
    PlanEditMsg(type="plan_edit", add=[StepAdd(say="쓰레기를 버리세요", done_when="쓰레기통이 비었다")] * 15),
])
def test_bad_edits_are_refused(message: PlanEditMsg) -> None:
    with pytest.raises(PlanEditError):
        policy.apply_plan_edit(plan([step("s1"), step("s2")]), message)


def test_removing_a_step_another_one_requires_is_refused() -> None:
    plan_ = plan([step("s1"), step("s2", requires=["s1"])])
    with pytest.raises(PlanEditError) as excinfo:
        policy.apply_plan_edit(plan_, PlanEditMsg(type="plan_edit", remove=["s1"]))
    assert "선행 조건" in str(excinfo.value)


def test_a_cyclic_or_unknown_requirement_is_refused() -> None:
    plan_ = plan([step("s1"), step("s2")])
    with pytest.raises(PlanEditError):
        policy.apply_plan_edit(plan_, PlanEditMsg(type="plan_edit", edits=[
            StepEdit(step_id="s1", requires=["s2"]), StepEdit(step_id="s2", requires=["s1"])]))
    with pytest.raises(PlanEditError):
        policy.apply_plan_edit(plan_, PlanEditMsg(type="plan_edit", edits=[StepEdit(step_id="s1", requires=["s7"])]))


def test_an_approved_plan_cannot_be_edited() -> None:
    with pytest.raises(PlanEditError):
        policy.apply_plan_edit(plan([step("s1")], status="approved"),
                               PlanEditMsg(type="plan_edit", edits=[StepEdit(step_id="s1", required=True)]))


# ------------------------------------------------------------------ 승인·변경안 (§15.5·§15.6)

def test_approve_pins_this_revision() -> None:
    draft = plan([step("s1")], revision=3)
    approved = policy.approve_plan(draft)
    assert approved.status == "approved" and approved.approved_revision == 3 and approved.revision == 3
    assert draft.status == "draft"


def test_an_accepted_proposal_becomes_the_next_approved_revision_keeping_identity() -> None:
    installed = plan([step("s1"), step("s2")], status="approved", revision=2,
                     materials=[{"id": "m1", "title": "지침", "version": "v3", "chars": 10, "added_at": 1}])
    proposal = policy.make_proposal(installed, reason="내부 확인을 먼저 하도록 단계를 나눴어요",
                                    steps=[step("s1"), step("s2", required=True), step("s3")])
    assert proposal.revision == 3 and proposal.plan_id == installed.plan_id
    adopted = policy.install_proposal(installed, proposal)
    assert (adopted.revision, adopted.approved_revision, adopted.status) == (3, 3, "approved")
    assert adopted.plan_id == "p1" and adopted.target == "상자" and adopted.materials == installed.materials
    assert [s.id for s in adopted.steps] == ["s1", "s2", "s3"] and adopted.steps[1].required
    assert installed.revision == 2                              # 설치 전 계획은 그대로


def test_blocked_is_a_wire_model_the_client_can_render() -> None:
    blocked = Blocked(step_id="s2", requires=["s1"], reason="s1 단계가 끝나야 해요.")
    assert blocked.model_dump()["requires"] == ["s1"]


@pytest.mark.parametrize("action", ["align", "connect", "disconnect", "bend"])
def test_circuit_assembly_actions_are_in_the_closed_set_and_direction_free(action: str) -> None:
    """§15.4: the Live action vocabulary carries the circuit-assembly verbs, each with ``direction: none`` only."""
    def with_direction(direction: str) -> GuideStep:
        return GuideStep.model_validate({
            "id": "s1",
            "say": "부품을 이어 붙인다",
            "done_when": "부품이 붙어 있다",
            "commands": [{"kind": "action", "anchor": "target", "action": action, "direction": direction}],
        })

    built = with_direction("none")
    assert built.commands[0].action == action
    assert overlay_command_from_step(built.commands[0]).action == action   # the overlay path takes it too
    with pytest.raises(ValueError):
        with_direction("up")


def test_an_unknown_action_verb_is_refused() -> None:
    with pytest.raises(ValueError):
        GuideStep.model_validate({
            "id": "s1",
            "say": "부품을 용접한다",
            "done_when": "용접되어 있다",
            "commands": [{"kind": "action", "anchor": "target", "action": "weld", "direction": "none"}],
        })

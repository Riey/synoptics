"""Plan 모드 규칙 (스펙 §15.4) — 순수 함수만. 엔진이 여기에 물어보고, 판단은 여기서만 나온다.

이 모듈이 지키는 것:

* 승인 경계: 계획은 검토(`draft`)와 승인(`approved`)이 분리되고, 승인은 그 ``revision``을 고정한다.
* 필수 검사(`required`)는 **어떤 경우에도** 건너뛸 수 없다.
* 선행 조건(`requires`)이 끝나기 전에는 그 단계로 갈 수 없다.
* `check:"user"`/`"measure"` 단계는 화면 판정만으로 끝나지 않는다(`step_ack`이 필요하다).
* 완료는 필수 단계가 모두 끝났을 때만. ``skipped``는 done이 아니다.

I/O도 시계도 없다. 그래서 mock 엔진과 실제 엔진이 같은 규칙을 쓰는 것을 테스트로 고정할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import (
    DEFAULT_CONDITION_KIND,
    MAX_STEPS,
    Blocked,
    CheckMethod,
    GuideStep,
    Material,
    MaterialId,
    MaterialInput,
    Plan,
    PlanEditMsg,
    Proposal,
    StepAdd,
    StepEdit,
    StepTarget,
)

#: 같은 단계를 다시 확인하기 전에 서버가 하는 말 (엔진이 say로 내보낸다).
REQUIRED_HOLD_REASON = "필수 검사라 건너뛸 수 없어요. 이 단계를 먼저 끝내 주세요."
PREREQ_HOLD_REASON = "앞 단계가 끝나야 할 수 있어요."
USER_CHECK_HOLD_REASON = "화면만으로는 끝난 것으로 보지 않아요. 확인되면 알려 주세요."


class PlanEditError(Exception):
    """§15.5가 거절하는 수정. ws 계층이 ``invalid_edit``로 바꾼다."""


# ------------------------------------------------------------------ 자료 (§15.3)

def materials_from_inputs(inputs: Iterable[MaterialInput], *, now_ms: int) -> tuple[Material, ...]:
    """``m1``…``mN`` in order. The text itself stays in the engine; the state only carries ``chars``."""
    out: list[Material] = []
    for index, item in enumerate(inputs, start=1):
        if index > 12:  # MaterialId is m1..m12; the ws layer already caps the count
            break
        out.append(
            Material(
                id=f"m{index}",
                title=item.title,
                version=item.version,
                chars=len(item.text),
                added_at=now_ms,
            )
        )
    return tuple(out)


def material_texts(inputs: Iterable[MaterialInput]) -> dict[MaterialId, str]:
    return {f"m{index}": item.text for index, item in enumerate(inputs, start=1)}


# ------------------------------------------------------------------ 진행 상태 (§15.4)

def _index_of(steps: list[GuideStep], step_id: str) -> int | None:
    for index, step in enumerate(steps):
        if step.id == step_id:
            return index
    return None


def step_of(plan: Plan, step_id: str) -> GuideStep:
    index = _index_of(list(plan.steps), step_id)
    if index is None:
        raise PlanEditError(f"계획에 없는 단계입니다: {step_id}")
    return plan.steps[index]


def done_step_ids(plan: Plan, *, step_index: int, skipped: Iterable[str], user_done: Iterable[str]) -> set[str]:
    """What counts as finished: steps behind ``step_index`` that were not skipped, plus user-done ones."""
    skipped_ids = set(skipped)
    ids = {step.id for step in plan.steps}
    done = {step.id for step in plan.steps[:step_index] if step.id not in skipped_ids}
    done |= {step_id for step_id in user_done if step_id in ids}
    return done


def required_pending(plan: Plan, *, step_index: int, skipped: Iterable[str], user_done: Iterable[str]) -> list[str]:
    """Required steps that are not done yet (§15.4.5). Completion is refused while this is non-empty."""
    done = done_step_ids(plan, step_index=step_index, skipped=skipped, user_done=user_done)
    return [step.id for step in plan.steps if step.required and step.id not in done]


def completion_ready(plan: Plan, *, step_index: int, skipped: Iterable[str], user_done: Iterable[str]) -> bool:
    return not required_pending(plan, step_index=step_index, skipped=skipped, user_done=user_done)


def prerequisite_block(plan: Plan, index: int, *, step_index: int, skipped: Iterable[str],
                       user_done: Iterable[str]) -> Blocked | None:
    """Why the step at ``index`` cannot become current: a `requires` that is not done yet."""
    step = plan.steps[index]
    done = done_step_ids(plan, step_index=step_index, skipped=skipped, user_done=user_done)
    missing = [need for need in step.requires if need not in done]
    if not missing:
        return None
    return Blocked(step_id=step.id, requires=missing,
                   reason=f"{', '.join(missing)} 단계가 끝나야 '{step.say}'를 할 수 있어요.")


def skip_block(plan: Plan, index: int, *, step_index: int, skipped: Iterable[str],
               user_done: Iterable[str]) -> Blocked | None:
    """Why the steps between ``step_index`` and ``index`` cannot be passed over (§15.4.2/§15.4.4).

    A ``required`` step — or a step that only the user can finish and has not — is never skipped. The
    server holds on it instead of pretending it happened.
    """
    done = done_step_ids(plan, step_index=step_index, skipped=skipped, user_done=user_done)
    for step in plan.steps[step_index:index]:
        if step.id in done:
            continue
        if step.required:
            return Blocked(step_id=step.id, requires=list(step.requires), reason=REQUIRED_HOLD_REASON)
        if step.check != "visual":
            return Blocked(step_id=step.id, requires=list(step.requires), reason=USER_CHECK_HOLD_REASON)
    return None


@dataclass(frozen=True, slots=True)
class Advance:
    """The outcome of "the screen shows step ``target`` as done" or "the user says the current step is done"."""

    index: int
    skipped: tuple[str, ...]
    blocked: Blocked | None
    moved: bool

    @property
    def held(self) -> bool:
        return self.blocked is not None


def advance_to(plan: Plan, target_index: int, *, step_index: int, skipped: Iterable[str],
               user_done: Iterable[str]) -> Advance:
    """Move the current step to ``target_index`` if §15.4 allows it; otherwise hold and say why."""
    skipped_ids = tuple(skipped)
    if target_index <= step_index:
        return Advance(index=step_index, skipped=skipped_ids, blocked=None, moved=False)
    if target_index >= len(plan.steps):
        raise PlanEditError(f"계획에 없는 단계 위치입니다: {target_index}")
    prereq = prerequisite_block(plan, target_index, step_index=step_index, skipped=skipped_ids,
                                user_done=user_done)
    if prereq is not None:
        return Advance(index=step_index, skipped=skipped_ids, blocked=prereq, moved=False)
    jump = skip_block(plan, target_index, step_index=step_index, skipped=skipped_ids, user_done=user_done)
    if jump is not None:
        return Advance(index=step_index, skipped=skipped_ids, blocked=jump, moved=False)
    done = done_step_ids(plan, step_index=step_index, skipped=skipped_ids, user_done=user_done)
    passed = tuple(step.id for step in plan.steps[step_index:target_index] if step.id not in done)
    return Advance(index=target_index, skipped=skipped_ids + passed, blocked=None, moved=True)


def advance_by_one(plan: Plan, *, step_index: int, skipped: Iterable[str], user_done: Iterable[str],
                   user_ack: bool = False) -> Advance:
    """The current step is finished: step forward. A required/non-visual step only moves on an explicit ack."""
    step = plan.steps[step_index]
    skipped_ids = tuple(skipped)
    if step.required or step.check != "visual":
        if not user_ack and step.id not in set(user_done):
            reason = REQUIRED_HOLD_REASON if step.required else USER_CHECK_HOLD_REASON
            return Advance(index=step_index, skipped=skipped_ids,
                           blocked=Blocked(step_id=step.id, requires=list(step.requires), reason=reason),
                           moved=False)
    if step_index + 1 >= len(plan.steps):
        return Advance(index=step_index, skipped=skipped_ids, blocked=None, moved=True)
    return Advance(index=step_index + 1, skipped=skipped_ids, blocked=None, moved=True)


def ack_index(plan: Plan, step_id: str) -> int:
    """Where a ``step_ack`` applies. Only ``check:"user"``/``"measure"`` steps accept one (§15.4.4)."""
    index = _index_of(list(plan.steps), step_id)
    if index is None:
        raise PlanEditError(f"계획에 없는 단계입니다: {step_id}")
    step = plan.steps[index]
    if step.check == "visual":
        raise PlanEditError(f"'{step.say}'는 화면 확인으로 끝나는 단계입니다. 사용자 확인을 받지 않습니다.")
    return index


# ------------------------------------------------------------------ 그래프 코어 (§15.8)

def step_ids(steps: list[GuideStep]) -> tuple[str, ...]:
    return tuple(step.id for step in steps)


def graph_fact_mapping(previous: Plan, current: Plan) -> dict[str, str]:
    """Exact, unambiguous old→new condition bindings, not a step-id or prose-similarity match.

    Tracking continuity is the caller's responsibility. A changed goal/target invalidates all applicability;
    a changed predicate, authority, subject or dependency invalidates only that node and its dependents.
    Presentation text/commands and step numbering do not define a completion condition.
    """
    if (previous.plan_id, previous.target, previous.goal_when) != \
            (current.plan_id, current.target, current.goal_when):
        return {}

    def signatures(plan: Plan) -> dict[str, tuple]:
        by_id = {step.id: step for step in plan.steps}
        keys: dict[str, tuple] = {}

        def key(step_id: str) -> tuple:
            if step_id not in keys:
                step = by_id[step_id]
                keys[step_id] = (
                    step.done_when, step.check, step.required, step.goal_required, step.condition_kind,
                    tuple(sorted(step.targets)), tuple(sorted(key(need) for need in step.requires)),
                )
            return keys[step_id]

        for step in plan.steps:
            key(step.id)
        return keys

    old_keys, new_keys = signatures(previous), signatures(current)
    old_groups: dict[tuple, list[str]] = {}
    new_groups: dict[tuple, list[str]] = {}
    for step_id, key in old_keys.items():
        old_groups.setdefault(key, []).append(step_id)
    for step_id, key in new_keys.items():
        new_groups.setdefault(key, []).append(step_id)
    mapping = {ids[0]: new_groups[key][0] for key, ids in old_groups.items()
               if len(ids) == 1 and len(new_groups.get(key, ())) == 1}
    new_steps = {step.id: step for step in current.steps}
    # A unique dependent cannot resolve ambiguous prerequisites by guessing which identical item was meant.
    while True:
        invalid = [step.id for step in previous.steps if step.id in mapping
                   and (any(need not in mapping for need in step.requires)
                        or {mapping[need] for need in step.requires}
                        != set(new_steps[mapping[step.id]].requires))]
        if not invalid:
            return mapping
        for step_id in invalid:
            del mapping[step_id]


def graph_dependents(steps: list[GuideStep], step_id: str) -> tuple[str, ...]:
    """Every node that transitively requires ``step_id`` — the ones whose applicability it underpins.

    Returned in plan order (deterministic, for the snapshot). The revoked node itself is not included.
    """
    children: dict[str, list[str]] = {}
    for step in steps:
        for need in step.requires:
            children.setdefault(need, []).append(step.id)
    seen = {step_id}
    stack = [step_id]
    out: list[str] = []
    while stack:
        for child in children.get(stack.pop(), ()):
            if child in seen:
                continue
            seen.add(child)
            out.append(child)
            stack.append(child)
    order = {step.id: index for index, step in enumerate(steps)}
    return tuple(sorted(out, key=lambda sid: order[sid]))


def graph_ancestors(steps: list[GuideStep], step_id: str) -> tuple[str, ...]:
    """Every node ``step_id`` transitively requires (its prerequisite closure), in plan order."""
    by_id = {step.id: step for step in steps}
    seen: set[str] = set()
    stack = [step_id]
    while stack:
        step = by_id.get(stack.pop())
        if step is None:
            continue
        for need in step.requires:
            if need not in seen:
                seen.add(need)
                stack.append(need)
    order = {step.id: index for index, step in enumerate(steps)}
    return tuple(sorted(seen, key=lambda sid: order.get(sid, len(steps))))


def graph_goal_gap(steps: list[GuideStep], done: Iterable[str], step_id: str) -> tuple[str, ...]:
    """Unresolved ``goal_required`` prerequisites of ``step_id`` — the whole goal chain, not just direct
    ``requires`` — in plan order. Reading a goal state is licensed only once its goal chain is done."""
    by_id = {step.id: step for step in steps}
    done_ids = set(done)
    return tuple(need for need in graph_ancestors(steps, step_id)
                 if by_id[need].goal_required and need not in done_ids)


def graph_mandatory_gap(steps: list[GuideStep], done: Iterable[str], step_id: str) -> tuple[str, ...]:
    """Unresolved ``required`` prerequisites of ``step_id`` — every MANDATORY check up its chain — in plan order.

    A ``required`` node is a mandatory check, so no node behind it may be acted on or recorded done while it is
    unverified, even when the node's DIRECT ``requires`` happen to be satisfied: an intermediate node committed
    by observation must never hide a mandatory ancestor behind its own ``requires``.
    """
    by_id = {step.id: step for step in steps}
    done_ids = set(done)
    return tuple(need for need in graph_ancestors(steps, step_id)
                 if by_id[need].required and need not in done_ids)


def graph_ready_ids(steps: list[GuideStep], done: Iterable[str]) -> tuple[str, ...]:
    """SAFE-ACTION eligibility (§15.4.8): every prerequisite is done AND no unresolved MANDATORY ancestor.

    A node whose completion was only OBSERVED — which §15.4.8 allows before a non-goal (safety/precondition)
    prerequisite is verified — is not thereby safe to perform, so an unresolved ``required`` ancestor keeps it
    out of this set: an already-observed downstream node never bypasses a mandatory procedural step.
    Optional/supporting ancestors (``required`` false) never block.
    """
    done_ids = set(done)
    out: list[str] = []
    for step in steps:
        if step.id in done_ids:
            continue
        if any(need not in done_ids for need in step.requires):
            continue
        if graph_mandatory_gap(steps, done_ids, step.id):
            continue
        out.append(step.id)
    return tuple(out)


def graph_focus_index(steps: list[GuideStep], done: Iterable[str], *, suggested: int = 0) -> int:
    """The suggested focus for the overlay — a hint only, never completion and never a permission.

    The first ACTION-READY node in plan order; otherwise the first node that is at least READABLE (its goal
    chain is done), otherwise the first open node, otherwise the last one. The overlay's commands and the spoken
    instruction are suppressed whenever the node at this index is not action-ready (§15.4.8), so these fallbacks
    can only ever point at something to look at, never at something to do.
    """
    ready = graph_ready_ids(steps, done)
    if ready:
        return _index_of(list(steps), ready[0]) or 0
    done_ids = set(done)
    for step in steps:
        if step.id not in done_ids and not graph_goal_gap(steps, done_ids, step.id):
            return _index_of(list(steps), step.id) or 0
    for index, step in enumerate(steps):
        if step.id not in done_ids:
            return index
    return len(steps) - 1


def graph_ack_index(plan: Plan, step_id: str) -> int:
    """Where a ``step_ack`` applies in graph mode: a ``user``/``measure`` step, or a ``required`` one.

    A required visual node is still finished only by the user's own record (§15.4.4): the frame never supplies
    the authorization a mandatory check needs, and in graph mode the user can therefore ack it directly.
    """
    index = _index_of(list(plan.steps), step_id)
    if index is None:
        raise PlanEditError(f"계획에 없는 단계입니다: {step_id}")
    step = plan.steps[index]
    if step.required or step.check != "visual":
        return index
    raise PlanEditError(f"'{step.say}'는 화면 확인으로 끝나는 단계입니다. 사용자 확인을 받지 않습니다.")


def graph_prerequisite_block(plan: Plan, step_id: str, done: Iterable[str]) -> Blocked | None:
    """Why the user's own ``step_ack`` cannot be recorded yet.

    Explicit authority still respects the procedure it belongs to: a DIRECT ``requires`` that is not done, and —
    exactly like action readiness — an unresolved MANDATORY (``required``) ancestor anywhere up the chain. An
    intermediate node committed by observation must never hide a mandatory ancestor behind its own ``requires``.
    """
    step = step_of(plan, step_id)
    done_ids = set(done)
    direct = [need for need in step.requires if need not in done_ids]
    mandatory = list(graph_mandatory_gap(list(plan.steps), done_ids, step_id))
    missing = list(dict.fromkeys((*direct, *mandatory)))
    if not missing:
        return None
    if direct:
        reason = f"{', '.join(missing)} 단계가 끝나야 '{step.say}'를 할 수 있어요."
    else:
        reason = f"{', '.join(mandatory)} 필수 검사가 끝나야 '{step.say}'를 할 수 있어요."
    return Blocked(step_id=step.id, requires=missing, reason=reason)


def graph_observation_block(plan: Plan, done: Iterable[str], step_id: str) -> Blocked | None:
    """Why a node cannot be OBSERVED/committed yet: an unresolved ``goal_required`` prerequisite anywhere up its
    chain — a goal node reached through an intermediate node is still behind its whole goal chain.

    Non-goal prerequisites (a safety precondition, an optional step) may still be unverified — the follower is
    read about the goal state anyway, and the upper's own verdict may commit it. Reading a state is not
    permission to perform it: those guards are ``graph_ready_ids`` and ``graph_prerequisite_block``.
    """
    step = step_of(plan, step_id)
    missing = list(graph_goal_gap(list(plan.steps), done, step_id))
    if not missing:
        return None
    return Blocked(step_id=step.id, requires=missing,
                   reason=f"{', '.join(missing)} 단계가 끝나야 '{step.say}' 완료를 확인할 수 있어요.")


def graph_observation_ready(plan: Plan, done: Iterable[str], step_id: str) -> bool:
    """``step_id`` is not done and every GOAL-REQUIRED prerequisite of it is done (§15.4.8)."""
    done_ids = set(done)
    return step_id not in done_ids and graph_observation_block(plan, done_ids, step_id) is None


def graph_completion_pending(plan: Plan, done: Iterable[str]) -> list[str]:
    """Nodes that still keep the ORIGINAL GOAL unachieved: ``goal_required`` and not explicitly done.

    A ``required`` safety/precondition node (``goal_required`` false) never appears here: an unverified
    unplugging confirmation does not make an already-built circuit unachieved (§15.4.8). Completion still needs
    the upper's positive goal confirmation as well.
    """
    done_ids = set(done)
    return [step.id for step in plan.steps if step.goal_required and step.id not in done_ids]


def graph_completion_ready(plan: Plan, done: Iterable[str]) -> bool:
    return not graph_completion_pending(plan, done)


def graph_pending_required_checks(plan: Plan, done: Iterable[str]) -> list[str]:
    """Mandatory procedural/user checks still unverified — reported beside the goal, never as its condition."""
    done_ids = set(done)
    return [step.id for step in plan.steps if step.required and step.id not in done_ids]


# ------------------------------------------------------------------ 수정 (§15.5)

def _next_free_id(steps: list[GuideStep]) -> str:
    used = {step.id for step in steps}
    for index in range(1, MAX_STEPS + 1):
        candidate = f"s{index}"
        if candidate not in used:
            return candidate
    raise PlanEditError(f"단계는 최대 {MAX_STEPS}개까지입니다.")


def _apply_edits(steps: list[GuideStep], edits: Iterable[StepEdit]) -> list[GuideStep]:
    out = list(steps)
    for edit in edits:
        index = _index_of(out, edit.step_id)
        if index is None:
            raise PlanEditError(f"계획에 없는 단계입니다: {edit.step_id}")
        fields = edit.model_dump(exclude_none=True, exclude={"step_id"})
        if "details" in edit.model_fields_set:
            fields["details"] = edit.details
        out[index] = out[index].model_copy(update=fields)
    return out


def _apply_removals(steps: list[GuideStep], remove: Iterable[str]) -> list[GuideStep]:
    doomed = set(remove)
    if len(doomed) >= len(steps):
        raise PlanEditError("모든 단계를 지울 수는 없습니다. 최소 한 단계는 남겨야 합니다.")
    for step in steps:
        if step.id in doomed:
            continue
        blockers = [need for need in step.requires if need in doomed]
        if blockers:
            raise PlanEditError(f"{step.id} 단계가 {', '.join(blockers)}를 선행 조건으로 씁니다.")
    return [step for step in steps if step.id not in doomed]


def _apply_order(steps: list[GuideStep], order: list[str]) -> list[GuideStep]:
    if sorted(order) != sorted(step.id for step in steps):
        raise PlanEditError("order는 계획의 모든 단계 id를 한 번씩 담아야 합니다.")
    by_id = {step.id: step for step in steps}
    return [by_id[step_id] for step_id in order]


def _step_from_add(steps: list[GuideStep], add: StepAdd) -> GuideStep:
    step_id = _next_free_id(steps)
    return GuideStep(
        id=step_id,
        say=add.say,
        details=add.details,
        done_when=add.done_when,
        commands=[],
        check=add.check or "visual",
        required=bool(add.required),
        requires=list(add.requires or ()),
        targets=list(add.targets or ()),
        condition_kind=add.condition_kind or DEFAULT_CONDITION_KIND,
        goal_required=True if add.goal_required is None else add.goal_required,
    )


def apply_plan_edit(plan: Plan, msg: PlanEditMsg) -> Plan:
    """A new draft revision of ``plan`` with §15.5's edit applied. Raises :class:`PlanEditError`."""
    if plan.status != "draft":
        raise PlanEditError("승인된 계획은 고칠 수 없습니다. 새로 시작하거나 초안을 다시 만드세요.")
    steps = _apply_edits(list(plan.steps), msg.edits or ())
    for add in msg.add or ():
        if len(steps) >= MAX_STEPS:
            raise PlanEditError(f"단계는 최대 {MAX_STEPS}개까지입니다.")
        steps.append(_step_from_add(steps, add))
    if msg.remove:
        steps = _apply_removals(steps, msg.remove)
    if msg.order:
        steps = _apply_order(steps, msg.order)
    if not steps:
        raise PlanEditError("단계가 없으면 계획이 될 수 없습니다.")
    try:
        return Plan(
            plan_id=plan.plan_id,
            revision=plan.revision + 1,
            status="draft",
            approved_revision=None,
            target=plan.target,
            goal_when=msg.goal_when or plan.goal_when,
            materials=list(plan.materials),
            steps=steps,
            research_sources=list(plan.research_sources),
        )
    except ValueError as exc:  # pydantic: unknown/cyclic requires, duplicate ids, over-long text
        raise PlanEditError(str(exc).splitlines()[-2].strip() if "\n" in str(exc) else str(exc)) from None


def approve_plan(plan: Plan) -> Plan:
    """Pin execution to this revision (§15.5). The approved plan can no longer be edited."""
    return plan.model_copy(update={"status": "approved", "approved_revision": plan.revision})


def install_proposal(plan: Plan, proposal: Proposal) -> Plan:
    """§15.6: an accepted change becomes the new approved plan (revision+1), keeping identity and materials."""
    return Plan(
        plan_id=plan.plan_id,
        revision=plan.revision + 1,
        status="approved",
        approved_revision=plan.revision + 1,
        target=plan.target,
        goal_when=proposal.goal_when,
        materials=list(plan.materials),
        steps=list(proposal.steps),
        research_sources=list(plan.research_sources),
    )


def make_proposal(plan: Plan, *, reason: str, steps: list[GuideStep], goal_when: str | None = None) -> Proposal:
    return Proposal(plan_id=plan.plan_id, revision=plan.revision + 1, reason=reason,
                    goal_when=goal_when or plan.goal_when, steps=steps)


__all__ = [
    "Advance", "PREREQ_HOLD_REASON", "PlanEditError", "REQUIRED_HOLD_REASON", "USER_CHECK_HOLD_REASON",
    "ack_index", "advance_by_one", "advance_to", "apply_plan_edit", "approve_plan", "completion_ready",
    "done_step_ids", "graph_ack_index", "graph_ancestors", "graph_completion_pending", "graph_completion_ready",
    "graph_dependents", "graph_fact_mapping", "graph_focus_index", "graph_goal_gap", "graph_mandatory_gap",
    "graph_observation_block", "graph_observation_ready",
    "graph_pending_required_checks", "graph_prerequisite_block", "graph_ready_ids", "install_proposal",
    "make_proposal", "material_texts", "materials_from_inputs", "prerequisite_block", "required_pending",
    "skip_block", "step_ids", "step_of",
]

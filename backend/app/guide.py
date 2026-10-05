"""Per-session plan store for the anchored guide lane: identity, revision, and the fences that use them.

What a session keeps is TEXT ONLY: the plan's steps (Korean sentences, visible conditions and role-anchored
commands), the goal condition, the target noun the plan selected, the task text it was planned for and, when
the request supplied them, the §15 reference materials the plan cites. No image, no coordinate and no model
answer other than the accepted plan is ever stored, so nothing here can leak a frame or replay a position.

Revision rules:

* ``plan_revision`` comes from a per-session counter that only goes up — a new plan, an applied replan, a talk
  restate (one step's sentence rewritten), an approve (the client's reviewed/edited plan adopted) and a revert
  (the previous plan restored) each take the next value, and clearing the plan does not reset it. A stale
  revision can therefore never be mistaken for the current one, even after the plan was cleared and re-made.
* A plan is bound to the session's ``task_revision`` at the time it was made. A goal or context change
  (on any route) bumps ``task_revision`` and clears the plan, the same way the guidance history is cleared.
* Every follow/confirm names ``plan_id`` + ``plan_revision``; anything other than the current pair is a
  ``409 stale_plan`` before any model work, and again after it (the post-await fence), so an answer computed
  against a replaced plan is never returned as current.
* The session keeps a bounded history (``maxlen`` 2) of the plans it replaced, so ``revert`` can install the
  most recent previous plan again. A restored plan keeps the ``plan_id`` it had but is re-stamped with the NEXT
  ``plan_revision``, never the revision it carried before: every fence only ever moves forward, and a late
  answer about the pre-revert pair is still refused as stale.

No lock is held across an await anywhere in this module: each mutation is one synchronous call.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections import deque
from dataclasses import dataclass, field

from backend.app.errors import ApiFailure
from backend.app.guide_contracts import (
    CORE_MODE_GRAPH,
    CORE_MODE_SEQUENTIAL,
    DEFAULT_CORE_MODE,
    DEFAULT_PLAN_MODEL,
    CoreMode,
    GuideStep,
    MaterialInput,
    PlanAnswer,
    PlanModel,
    ResearchSource,
)


@dataclass(frozen=True, slots=True)
class StoredPlan:
    """The accepted plan, as text. Immutable: a replan produces a new ``StoredPlan``."""

    plan_id: str
    plan_revision: int
    steps: tuple[GuideStep, ...]
    goal_when: str
    user_goal: str
    context: str | None
    task_revision: int
    #: The plan's ``selection.target`` (the detector noun it was made for); the talk prompt names it.
    target: str | None = None
    #: §15 reference documents the plan was made with, kept for its task so a later replan prompt can render the
    #: same source and its evidence can be grounded. Never echoed by ``/plan/current``; carried unchanged by
    #: ``adopt``/``replan``/``restate`` and restored with the plan by ``revert``.
    materials: tuple[MaterialInput, ...] = ()
    #: Which upper planning profile wrote this plan (``deepseek:high`` or ``astra:high``). It is part of the
    #: plan's identity — confirm/talk route from it, so the client can never switch profiles mid-plan — and
    #: every revision (``adopt``/``replan``/``restate``/``revert``) carries it unchanged.
    plan_model: PlanModel = DEFAULT_PLAN_MODEL
    #: The guide core this plan runs under (``classic``, ``sequential`` or ``graph``). Part of the plan's
    #: identity — it decides the checklist a follow answer must cover (``classic``: remaining; ``sequential``:
    #: the current step alone; ``graph``: the whole plan, earlier steps included) — and every revision carries
    #: it unchanged: a plan can never silently change core mid-run, and a core change under the same
    #: goal/context is a new task (see ``main.Session``).
    core_mode: CoreMode = DEFAULT_CORE_MODE
    assisted: bool = False
    answers: tuple[PlanAnswer, ...] = ()
    research_sources: tuple[ResearchSource, ...] = ()

    @property
    def step_ids(self) -> frozenset[str]:
        return frozenset(step.id for step in self.steps)

    def checklist_ids(self, current_step: str) -> list[str]:
        """The step ids a follow checklist covers, in plan order.

        ``classic``: ``current_step`` through the last step. ``sequential``: the current step ALONE — every
        follower (Clef, local, paid) is asked about the current step only, so a future step's verdict can
        never arrive and can never move the screen. ``graph``: EVERY step of the plan — earlier and already
        completed ones included — regardless of ``current_step``, which is the client's suggested focus only.
        A ``current_step`` outside the plan is no checklist at all for the cores that scope by it; ``graph``
        never narrows or truncates, so it is unaffected.
        """
        ids = [step.id for step in self.steps]
        if self.core_mode == CORE_MODE_GRAPH:
            return ids
        if current_step not in ids:
            return []
        if self.core_mode == CORE_MODE_SEQUENTIAL:
            return [current_step]
        return ids[ids.index(current_step):]

    def is_before(self, step_id: str, current_step: str) -> bool:
        """``step_id`` is a step of this plan strictly before ``current_step`` (also a step of it)."""
        ids = [step.id for step in self.steps]
        return step_id in ids and current_step in ids and ids.index(step_id) < ids.index(current_step)


@dataclass(slots=True)
class GuideSessionState:
    """One session's guide state: the current plan (or none) and the in-flight admission flags.

    The flags are read and written without an await in between, so two overlapping calls of the same kind
    cannot both pass them; they are transient bookkeeping, never state an answer is derived from.
    """

    plan: StoredPlan | None = None
    revision_counter: int = 0
    #: The plans this session replaced, most recent last (bounded: one session can never grow without limit).
    #: ``revert`` installs ``history[-1]`` again under a NEW revision. Cleared with the plan, so a revert can
    #: never revive a plan from a task the session has since left.
    history: deque[StoredPlan] = field(default_factory=lambda: deque(maxlen=2))
    plan_inflight: bool = False
    follow_inflight: bool = False
    confirm_inflight: bool = False
    talk_inflight: bool = False

    def clear(self) -> None:
        """Forget the plan and its history (task change, session end). The revision counter is kept on purpose."""
        self.plan = None
        self.history.clear()

    def _next_revision(self) -> int:
        self.revision_counter += 1
        return self.revision_counter

    def _replace(self, plan: StoredPlan) -> None:
        """Make ``plan`` current, remembering the plan it replaced (bounded history)."""
        if self.plan is not None:
            self.history.append(self.plan)
        self.plan = plan

    def install(
        self,
        *,
        steps: list[GuideStep],
        goal_when: str,
        user_goal: str,
        context: str | None,
        task_revision: int,
        target: str | None = None,
        materials: tuple[MaterialInput, ...] = (),
        plan_model: PlanModel = DEFAULT_PLAN_MODEL,
        core_mode: CoreMode = DEFAULT_CORE_MODE,
        assisted: bool = False,
        answers: tuple[PlanAnswer, ...] = (),
        research_sources: tuple[ResearchSource, ...] = (),
    ) -> StoredPlan:
        """Replace the current plan with a new one (new ``plan_id``, next revision)."""
        plan = StoredPlan(
            plan_id=str(uuid.uuid4()),
            plan_revision=self._next_revision(),
            steps=tuple(steps),
            goal_when=goal_when,
            user_goal=user_goal,
            context=context,
            task_revision=task_revision,
            target=target,
            materials=materials,
            plan_model=plan_model,
            core_mode=core_mode,
            assisted=assisted,
            answers=answers,
            research_sources=research_sources,
        )
        self._replace(plan)
        return plan

    def _revise(self, plan: StoredPlan, steps: list[GuideStep]) -> StoredPlan:
        """``plan`` (which must still be current) with new steps: same ``plan_id``, next revision."""
        if self.plan is not plan:
            raise stale_plan(self)
        updated = dataclasses.replace(plan, plan_revision=self._next_revision(), steps=tuple(steps))
        self._replace(updated)
        return updated

    def adopt(
        self,
        plan: StoredPlan,
        *,
        steps: list[GuideStep],
        goal_when: str,
        user_goal: str,
        context: str | None,
    ) -> StoredPlan:
        """The client's reviewed plan replaces ``plan``: same ``plan_id``, next revision (``/plan/approve``).

        ``plan`` must still be current (the caller resolved it with ``require``). No provider call is involved;
        the caller passes ``plan``'s own ``user_goal``/``context`` when the request omitted them.
        """
        if self.plan is not plan:
            raise stale_plan(self)
        adopted = dataclasses.replace(
            plan,
            plan_revision=self._next_revision(),
            steps=tuple(steps),
            goal_when=goal_when,
            user_goal=user_goal,
            context=context,
        )
        self._replace(adopted)
        return adopted

    def revert(self, plan: StoredPlan) -> StoredPlan:
        """Install the most recent previous plan again, under the NEXT revision (``/plan/revert``).

        ``plan`` must still be current (the caller resolved it with ``require``) and the history must not be
        empty (an empty history is the caller's ``404 no_plan``). The restored plan keeps the ``plan_id`` it had
        and gets a fresh ``plan_revision``, so no fence ever moves backwards.
        """
        if self.plan is not plan:
            raise stale_plan(self)
        restored = dataclasses.replace(self.history[-1], plan_revision=self._next_revision())
        self._replace(restored)
        return restored

    @property
    def can_revert(self) -> bool:
        return bool(self.history)

    def replan(self, plan: StoredPlan, steps: list[GuideStep]) -> StoredPlan:
        """Swap the steps of ``plan`` (which must still be current); same ``plan_id``, next revision."""
        return self._revise(plan, steps)

    def restate(self, plan: StoredPlan, step_id: str, say: str) -> StoredPlan:
        """Rewrite ONE step's ``say`` (a talk ``step_say``); every other step and field is kept.

        ``say`` is already bounded by the talk contract (``StepSay``). Same ``plan_id``, next revision, so a
        follow/confirm still naming the old revision is ``409 stale_plan`` and the next prompt sees the new say.
        """
        if self.plan is not plan:
            raise stale_plan(self)
        if step_id not in plan.step_ids:
            raise ValueError(f"unknown step {step_id!r}")
        steps = [step.model_copy(update={"say": say}) if step.id == step_id else step for step in plan.steps]
        return self._revise(plan, steps)

    def require(self, *, plan_id: str, plan_revision: int, task_revision: int) -> StoredPlan:
        """The current plan if the request names it exactly and it belongs to the current task, else 409."""
        plan = self.plan
        if (
            plan is None
            or plan.plan_id != plan_id
            or plan.plan_revision != plan_revision
            or plan.task_revision != task_revision
        ):
            raise stale_plan(self)
        return plan

    def is_current(self, plan: StoredPlan, task_revision: int) -> bool:
        return self.plan is plan and plan.task_revision == task_revision


def stale_plan(state: GuideSessionState) -> ApiFailure:
    """``409 stale_plan`` naming the plan the client should continue from (or ``None`` when there is none)."""
    current = state.plan
    return ApiFailure(
        409,
        "stale_plan",
        extra={
            "plan_id": current.plan_id if current is not None else None,
            "plan_revision": current.plan_revision if current is not None else None,
        },
    )

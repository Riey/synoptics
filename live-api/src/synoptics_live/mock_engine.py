"""Deterministic scripted engine: no model, no tracker, no image analysis.

It exists so a web client can be built against the real protocol before the real engine is wired in. Every
"judgement" is a timer or a keyword. Durations are virtual seconds divided by ``MockConfig.speed`` (env
``LIVE_MOCK_SPEED``) so tests can run the whole script in well under a second.
"""

from __future__ import annotations

import asyncio
import math
import secrets
import time
from dataclasses import dataclass
from typing import Callable

from . import plan_policy
from .contracts import (
    CORE_GRAPH,
    ActionAnchorCommand,
    Blocked,
    Box,
    DEFAULT_CORE_MODE,
    DEFAULT_PLAN_MODEL,
    Evidence,
    FocusAnchorCommand,
    FrameHeader,
    GuideState,
    GuideStep,
    LabelAnchorCommand,
    MaterialInput,
    Notice,
    Overlay,
    OverlayBinding,
    Partial,
    Pending,
    Plan,
    PlanAnswer,
    PlanEditMsg,
    Prefs,
    Proposal,
    ReferenceImage,
    TalkResult,
    TrackMsg,
    TARGET_ROLE,
    TALK_REPLY_MAX,
    overlay_command_from_step,
    reference_images_problem,
)
from .engine import EngineReject, EngineSink

TARGET = "목표 물체"
PACKAGE_TARGET = "포장 상자"
ANCHOR_ID = "a1"
ACQUIRE_FRAMES = 5
MAX_REPLANS = 3
REPLAN_GAP_S = 15.0
NOTICE_TTL_S = 8.0
#: Virtual seconds of tracking a single frame can contribute (frames stop -> progress stops).
MAX_FRAME_DT_S = 0.5
#: §15: how many times the timer may re-hold the same step before the mock proposes a change.
HOLDS_BEFORE_PROPOSAL = 2
DRAFT_SAY = "계획 초안을 만들었어요. 검토하고 실행을 눌러 주세요."
EDIT_SAY = "계획을 고쳤어요."
PROPOSAL_REASON = "봉함 전 확인을 먼저 하도록 단계를 나눴어요."
PACKAGE_GOAL_WHEN = "상자를 봉함할 준비가 되었다"
#: §15: the mock has no model, so its one question is scripted too. It asks it only for a goal that is phrased
#: as a question (a concrete target is missing), which is what a real planner's clarification looks like.
CLARIFY_QUESTION = "무엇을 기준으로 안내할까요? 화면에 보이는 대상이나 제품 이름을 알려 주세요."


@dataclass
class MockConfig:
    speed: float = 1.0
    step_s: float = 8.0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _generic_steps() -> list[GuideStep]:
    t = TARGET_ROLE
    return [
        GuideStep(
            id="s1",
            say="목표 물체를 손으로 잡으세요",
            details="손바닥을 물체 옆면에 붙이고, 엄지와 네 손가락으로 감싸 쥐세요.",
            done_when="손이 목표 물체를 잡고 있다",
            commands=[
                FocusAnchorCommand(kind="focus", anchor=t, pad=0.15),
                ActionAnchorCommand(kind="action", anchor=t, action="grasp", direction="none"),
            ],
        ),
        GuideStep(
            id="s2",
            say="목표 물체를 시계 방향으로 돌리세요",
            done_when="목표 물체가 돌아간 상태다",
            commands=[ActionAnchorCommand(kind="action", anchor=t, action="rotate", direction="clockwise")],
        ),
        GuideStep(
            id="s3",
            say="목표 물체를 제자리에 내려놓으세요",
            done_when="목표 물체가 내려놓여 있다",
            commands=[
                LabelAnchorCommand(kind="label", anchor=t, text="여기에 놓기"),
                ActionAnchorCommand(kind="action", anchor=t, action="place", direction="none"),
            ],
        ),
    ]


def _packaging_steps(inputs: tuple[MaterialInput, ...] = (), *, with_seal: bool = False) -> list[GuideStep]:
    """§15.9's box-packaging draft. ``with_seal`` adds the sealing step the mid-run proposal splits off.

    Every §15 field is exercised: one ``required`` inspection (``s7``), one ``check:"user"`` step, ``requires``
    chains, logical ``targets``.  ``evidence`` is only attached when the client sent a reference document, and
    it always cites ``m1`` — the server never invents a source.
    """
    t = TARGET_ROLE

    def ev(locator: str) -> Evidence | None:
        if not inputs:
            return None
        return Evidence(material_id="m1", version=inputs[0].version, locator=locator)

    steps = [
        GuideStep(
            id="s1",
            say="상자를 펼쳐 바닥에 놓으세요",
            details="테이프를 떼고 상자를 완전히 펼친 뒤, 접힌 선을 따라 바닥을 눌러 평평하게 만드세요.",
            done_when="상자가 펼쳐져 바닥에 있다",
            commands=[
                FocusAnchorCommand(kind="focus", anchor=t, pad=0.15),
                ActionAnchorCommand(kind="action", anchor=t, action="place", direction="none"),
            ],
            targets=["상자"],
            evidence=ev("3.1 상자 준비"),
        ),
        GuideStep(
            id="s2",
            say="완충재를 상자 바닥에 까세요",
            done_when="완충재가 바닥을 덮고 있다",
            commands=[
                LabelAnchorCommand(kind="label", anchor=t, text="완충재 깔기"),
                ActionAnchorCommand(kind="action", anchor=t, action="place", direction="none"),
            ],
            targets=["상자", "완충재"],
            evidence=ev("3.2 완충재 깔기"),
        ),
        GuideStep(
            id="s3",
            say="제품을 완충재 위에 올려놓으세요",
            done_when="제품이 완충재 위에 있다",
            commands=[
                FocusAnchorCommand(kind="focus", anchor=t, pad=0.15),
                ActionAnchorCommand(kind="action", anchor=t, action="place", direction="none"),
            ],
            requires=["s2"],
            targets=["제품"],
            evidence=ev("3.3 제품 배치"),
        ),
        GuideStep(
            id="s4",
            say="남은 빈 곳을 완충재로 채우세요",
            done_when="빈 곳이 완충재로 채워져 있다",
            commands=[
                FocusAnchorCommand(kind="focus", anchor=t, pad=0.15),
                ActionAnchorCommand(kind="action", anchor=t, action="place", direction="none"),
            ],
            requires=["s3"],
            targets=["완충재"],
            evidence=ev("3.4 빈 곳 채우기"),
        ),
        GuideStep(
            id="s5",
            say="받는 분 주소 라벨을 붙이세요",
            done_when="라벨이 상자에 붙어 있다",
            commands=[
                FocusAnchorCommand(kind="focus", anchor=t, pad=0.15),
                ActionAnchorCommand(kind="action", anchor=t, action="attach", direction="none"),
            ],
            targets=["라벨"],
            evidence=ev("2.1 라벨 부착"),
        ),
        GuideStep(
            id="s6",
            say="완충재 위에 덮개를 올리세요",
            done_when="덮개가 올려져 있다",
            commands=[
                FocusAnchorCommand(kind="focus", anchor=t, pad=0.15),
                ActionAnchorCommand(kind="action", anchor=t, action="place", direction="none"),
            ],
            requires=["s5"],
            targets=["상자"],
            evidence=ev("3.5 덮개 올리기"),
        ),
        GuideStep(
            id="s7",
            say="봉함 전 내부 확인",
            done_when="내부가 확인되었다",
            commands=[FocusAnchorCommand(kind="focus", anchor=t, pad=0.15)],
            check="user",
            required=True,
            requires=["s6"],
            targets=["상자", "제품"],
            evidence=ev("4.1 봉함 전 점검"),
        ),
    ]
    if with_seal:
        steps.append(
            GuideStep(
                id="s8",
                say="테이프로 상자를 봉함하세요",
                done_when="테이프가 상자에 붙어 있다",
                commands=[
                    LabelAnchorCommand(kind="label", anchor=t, text="테이프 붙이기"),
                    ActionAnchorCommand(kind="action", anchor=t, action="press", direction="none"),
                ],
                requires=["s7"],
                targets=["테이프"],
                evidence=ev("4.2 봉함"),
            )
        )
    return steps


class MockEngine:
    def __init__(self, sink: EngineSink, config: MockConfig | None = None, clock: Callable[[], float] = time.monotonic):
        self.sink = sink
        self.cfg = config or MockConfig()
        self._clock = clock
        self._t0 = clock()
        self._tasks: set[asyncio.Task] = set()
        self._resumed = asyncio.Event()
        self._resumed.set()
        self.prefs = Prefs()
        self.hi_frames_used = 0
        self._track_counter = 0
        self._reset_guide()

    # ------------------------------------------------------------ time and tasks

    def vnow(self) -> float:
        return (self._clock() - self._t0) * self.cfg.speed

    async def _sleep(self, virtual_s: float) -> None:
        await asyncio.sleep(max(virtual_s, 0) / self.cfg.speed)
        await self._resumed.wait()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _cancel_tasks(self) -> None:
        current = asyncio.current_task()
        for task in list(self._tasks):
            if task is not current:
                task.cancel()

    # ------------------------------------------------------------ state

    def _reset_guide(self) -> None:
        self.phase = "idle"
        self.plan: Plan | None = None
        self.partial: Partial | None = None
        self.step_index = 0
        self.steps_user_done: list[str] = []
        self.steps_skipped: list[str] = []
        self.pending: Pending | None = None
        self.notice: Notice | None = None
        self._notice_token: object | None = None
        self.completion = "none"
        # §15 Plan 모드
        self.plan_mode = False
        self.plan_model = DEFAULT_PLAN_MODEL
        #: §15.8: the guide core this run was started with (recorded and echoed; the mock's canned progression
        #: is unchanged — only the real engine implements the sequential/graph execution policy). The graph core
        #: still gets the plan-policy gates (it carries an approved plan), and its ledger is reported from the
        #: mock's own linear progress.
        self.core_mode = DEFAULT_CORE_MODE
        self.materials_inputs: tuple[MaterialInput, ...] = ()
        #: §15: the clarification round trip (one question at a time) and the run's reference photos/answers.
        self.clarification: str | None = None
        self.clarification_id: str | None = None
        self.clarification_asked = False
        self.answers: list[PlanAnswer] = []
        self.reference_images: tuple[ReferenceImage, ...] = ()
        self.paused = False
        self.blocked: Blocked | None = None
        self.proposal: Proposal | None = None
        self._hold_step: str | None = None
        self._hold_streak = 0
        self._proposed = False
        self.replan_blocked_reason: str | None = None
        self.talk: TalkResult | None = None
        self.talk_pending = False
        self.needs_reselect = False
        self.replans_used = 0
        self.last_replan_v = -math.inf
        self.tracked_s = 0.0
        self.hi_req: str | None = None
        self._hi_event = asyncio.Event()
        self._confirm_task: asyncio.Task | None = None
        # tracking
        self.run_id: str | None = None
        self.track_id: str | None = None
        self.generation = 0
        self.frames_in_gen = 0
        self.bound = False
        self.seed_box: Box | None = None
        self.seed_v = 0.0
        self.last_frame_v: float | None = None

    def _new_run(self) -> None:
        self.run_id = f"r-{secrets.token_hex(3)}"
        self._track_counter += 1
        self.track_id = f"t{self._track_counter}"
        self.generation = 1
        self.frames_in_gen = 0
        self.bound = False
        self.seed_box = None
        self.last_frame_v = None

    def _target(self) -> str | None:
        if self.plan is not None:
            return self.plan.target
        if self.partial is not None:
            return self.partial.target
        return None

    def _graph_action_allowed(self) -> bool:
        """§15.4.8, mirrored from the real engine: in graph mode only an ACTION-READY node may carry an action
        command. The mock's plans have no ``requires``, so the only node that can fail this is an already-done
        one (the suggestion falls back to it once the ledger is closed)."""
        if self.core_mode != CORE_GRAPH or self.plan is None:
            return True
        if not (0 <= self.step_index < len(self.plan.steps)):
            return False
        return self.plan.steps[self.step_index].id in set(
            plan_policy.graph_ready_ids(list(self.plan.steps), self._graph_done()))

    def _overlay(self) -> Overlay | None:
        if self.phase != "running" or self.plan is None or not self.bound or self.run_id is None:
            return None
        step = self.plan.steps[self.step_index]
        commands = ([overlay_command_from_step(c) for c in step.commands]
                    if self._graph_action_allowed() else [])
        return Overlay(
            binding=OverlayBinding(
                anchor_id=ANCHOR_ID, run_id=self.run_id, track_id=self.track_id, generation=self.generation
            ),
            commands=commands,
            warn="needs_reselect" if self.needs_reselect else None,
            motion_key=f"{self.run_id}|{self.plan.revision}|{step.id}|{ANCHOR_ID}|{self.track_id}|{self.generation}",
        )

    def _uses_plan(self) -> bool:
        """§15.8: the mock drives its canned progression through :mod:`plan_policy` exactly when it carries a
        plan — plan review mode, or the graph core (which executes immediately from an approved plan)."""
        return self.plan_mode or self.core_mode == CORE_GRAPH

    def _graph_done(self, *, include_current: bool = False) -> set[str]:
        """The mock's linear progress as a done set. Its plans have no ``requires``, so this is exact."""
        if self.plan is None:
            return set()
        return plan_policy.done_step_ids(self.plan, step_index=self.step_index + (1 if include_current else 0),
                                         skipped=self.steps_skipped, user_done=self.steps_user_done)

    def snapshot(self) -> GuideState:
        plan = self.plan
        steps_done: list[str] = []
        steps_ready: list[str] = []
        step_statuses: dict[str, str] = {}
        pending_required: list[str] = []
        if plan is not None and self.core_mode == CORE_GRAPH:
            done = self._graph_done()
            steps_done = [step.id for step in plan.steps if step.id in done]
            steps_ready = list(plan_policy.graph_ready_ids(list(plan.steps), done))
            step_statuses = {step.id: "yes" for step in plan.steps if step.id in done}
            pending_required = plan_policy.graph_pending_required_checks(plan, done)
        elif plan is not None and self._uses_plan():
            pending_required = plan_policy.required_pending(plan, step_index=self.step_index,
                                                            skipped=self.steps_skipped,
                                                            user_done=self.steps_user_done)
        return GuideState(
            phase=self.phase,
            core_mode=self.core_mode,
            plan=self.plan,
            partial=self.partial,
            clarification=self.clarification,
            clarification_id=self.clarification_id,
            step_index=self.step_index,
            steps_skipped=list(self.steps_skipped),
            steps_user_done=list(self.steps_user_done),
            steps_done=steps_done,
            steps_ready=steps_ready,
            step_statuses=step_statuses,
            pending_required_checks=pending_required,
            pending=self.pending,
            notice=self.notice,
            completion=self.completion,
            needs_reselect=self.needs_reselect,
            replan_blocked_reason=self.replan_blocked_reason,
            talk=self.talk,
            talk_pending=self.talk_pending,
            overlay=self._overlay(),
            paused=self.paused,
            proposal=self.proposal,
            blocked=self.blocked,
        )

    def _emit(self) -> None:
        self.sink.state(self.snapshot())

    def _set_notice(self, code: str, text: str) -> None:
        self.notice = Notice(code=code, text=text)
        token = object()
        self._notice_token = token

        async def clear() -> None:
            await self._sleep(NOTICE_TTL_S)
            if self._notice_token is token:
                self.notice = None
                self._notice_token = None
                self._emit()

        self._spawn(clear())

    def _clear_notice(self) -> None:
        self.notice = None
        self._notice_token = None

    def _build_plan(self, revision: int) -> Plan:
        return Plan(
            plan_id=secrets.token_hex(6),
            revision=revision,
            target=TARGET,
            goal_when="목표 물체가 원하는 상태가 되었다",
            steps=_generic_steps(),
        )

    def _step_line(self) -> str:
        assert self.plan is not None
        return f"{self.step_index + 1}단계. {self.plan.steps[self.step_index].say}"

    # ------------------------------------------------------------ guide flow

    async def on_start(self, goal: str, context: str | None, *, plan_mode: bool = False,
                       plan_model: str = DEFAULT_PLAN_MODEL, core_mode: str = DEFAULT_CORE_MODE,
                       materials: tuple[MaterialInput, ...] = (),
                       reference_images: tuple[ReferenceImage, ...] = ()) -> None:
        problem = reference_images_problem(reference_images)
        if problem is not None:
            raise EngineReject(problem[0], problem[1], retryable=False)
        self._cancel_tasks()
        self._reset_guide()
        self.goal, self.context = goal, context
        self.plan_mode = bool(plan_mode)
        self.plan_model = str(plan_model)
        self.core_mode = str(core_mode)
        self.materials_inputs = tuple(materials)
        self.reference_images = tuple(reference_images)
        self._new_run()
        self.phase = "planning"
        self.pending = Pending(stage="plan", trigger="start", since=_now_ms())
        self._emit()
        self.sink.say("계획을 세우는 중입니다.", "replace")
        self.hi_req = self.sink.capture_hi()
        self._spawn(self._plan_flow())

    def _asks_a_question(self) -> bool:
        """§15: the mock's deterministic stand-in for a planner that needs one detail: a goal phrased as a
        question ("…어떻게 할까?") has no concrete target to anchor, so it asks once before writing the plan."""
        return "?" in self.goal

    async def on_plan_answer(self, clarification_id: str, answer: str,
                             reference_images: tuple[ReferenceImage, ...] | None = None) -> None:
        """§15: the answer to the mock's one question. The run (goal, materials, photos) is not restarted."""
        if not self.plan_mode or self.phase != "clarifying" or self.clarification_id is None:
            raise EngineReject("no_clarification", "지금은 답할 질문이 없습니다", retryable=False)
        if clarification_id != self.clarification_id:
            raise EngineReject("clarification_stale", "이미 지난 질문입니다. 화면에 보이는 질문에 답해 주세요.",
                               retryable=False)
        if reference_images is not None:
            problem = reference_images_problem(reference_images)
            if problem is not None:
                raise EngineReject(problem[0], problem[1], retryable=False)
            self.reference_images = tuple(reference_images)
        self.answers.append(PlanAnswer(question=self.clarification, answer=answer))
        self.clarification = None
        self.clarification_id = None
        self.phase = "planning"
        self.pending = Pending(stage="plan", trigger="start", since=_now_ms())
        self._emit()
        # The next turn looks at a FRESH frame; the answer itself is kept for the draft.
        self._hi_event.clear()
        self.hi_req = self.sink.capture_hi()
        self._spawn(self._plan_flow())

    async def _plan_flow(self) -> None:
        start = self.vnow()
        steps = _packaging_steps(self.materials_inputs) if self.plan_mode else _generic_steps()
        await self._sleep(1.2)
        self.partial = Partial(target=TARGET)
        self._emit()
        await self._sleep(0.3)
        self.partial = Partial(target=TARGET, first_say=steps[0].say)
        self._emit()
        # Use the high-resolution frame if it arrives within 2 s of start; otherwise carry on without it.
        remaining = 2.0 - (self.vnow() - start)
        if remaining > 0 and not self._hi_event.is_set():
            try:
                await asyncio.wait_for(self._hi_event.wait(), remaining / self.cfg.speed)
            except asyncio.TimeoutError:
                pass
        if self._hi_event.is_set():
            self.hi_frames_used += 1
        await self._sleep(2.4 - (self.vnow() - start))
        if self.plan_mode and not self.clarification_asked and self._asks_a_question():
            # §15: ask ONE question and wait for ``plan_answer``; the draft is written on the next turn, with
            # the same goal, materials and reference photos.
            self.clarification_asked = True
            self.phase = "clarifying"
            self.clarification = CLARIFY_QUESTION
            self.clarification_id = f"q-{secrets.token_hex(4)}"
            self.partial = None
            self.pending = None
            self.tracked_s = 0.0
            self._emit()
            self.sink.say(CLARIFY_QUESTION, "replace")
            return
        if self.plan_mode:
            # §15.1: the draft is reviewed before anything runs. No tracking, no progression yet.
            self.plan = Plan(
                plan_id=secrets.token_hex(6),
                revision=1,
                status="draft",
                approved_revision=None,
                target=PACKAGE_TARGET,
                goal_when=PACKAGE_GOAL_WHEN,
                materials=list(plan_policy.materials_from_inputs(self.materials_inputs, now_ms=_now_ms())),
                steps=steps,
            )
            self.phase = "reviewing"
            self.partial = None
            self.pending = None
            self.step_index = 0
            self.tracked_s = 0.0
            self._emit()
            self.sink.say(DRAFT_SAY, "replace")
            return
        self.plan = self._build_plan(revision=1)
        self.phase = "running"
        self.partial = None
        self.pending = None
        self.step_index = 0
        self.tracked_s = 0.0
        self._emit()
        self.sink.say(self._step_line(), "replace")

    def _advance(self) -> str | None:
        """Move to the next step (or into completion checking) and emit. Returns the step line to say, if any."""
        assert self.plan is not None
        self.tracked_s = 0.0
        self.pending = None
        self._clear_notice()
        if self.step_index + 1 < len(self.plan.steps):
            self.step_index += 1
            self._emit()
            return self._step_line()
        self.completion = "checking"
        self.pending = Pending(stage="confirm", trigger="goal_check", since=_now_ms())
        self._emit()
        self._confirm_task = self._spawn(self._confirm_flow())
        return None

    async def _confirm_flow(self) -> None:
        await self._sleep(1.5)
        if self.completion != "checking":
            return
        self.completion = "confirmed"
        self.pending = None
        self._emit()
        self.sink.say("작업이 끝난 것 같아요. 완료를 확인해 주세요.", "append")

    # ------------------------------------------------------------ §15 Plan 모드

    def _reacquire(self) -> None:
        """Drop the tracker binding: the next frames re-acquire the target (reconnect/resume path)."""
        self.frames_in_gen = 0
        self.seed_box = None
        self.bound = False
        self.last_frame_v = None
        self._resumed.set()

    def _clear_hold(self) -> None:
        self.blocked = None
        self._hold_step = None
        self._hold_streak = 0
        self._clear_notice()

    def _hold(self, blocked: Blocked) -> None:
        """Hold on ``blocked.step_id``: one notice per new hold, and one scripted proposal after two holds."""
        self.blocked = blocked
        if self._hold_step == blocked.step_id:
            self._hold_streak += 1
        else:
            self._hold_step = blocked.step_id
            self._hold_streak = 1
            if blocked.reason == plan_policy.REQUIRED_HOLD_REASON:
                code = "required_check"
            elif blocked.reason == plan_policy.USER_CHECK_HOLD_REASON:
                code = "user_check"
            else:
                code = "prerequisite"
            self._set_notice(code, blocked.reason)
        step = self.plan.steps[self.step_index] if self.plan is not None else None
        if (step is not None and step.required and self._hold_streak > HOLDS_BEFORE_PROPOSAL
                and not self._proposed and self.proposal is None):
            self._proposed = True
            self.proposal = plan_policy.make_proposal(
                self.plan, reason=PROPOSAL_REASON,
                steps=_packaging_steps(self.materials_inputs, with_seal=True),
            )
        self._emit()

    def _enter_step(self, index: int) -> str:
        assert self.plan is not None
        self.step_index = index
        self.tracked_s = 0.0
        self.pending = None
        self._clear_hold()
        self._emit()
        return self._step_line()

    def _complete_or_hold(self) -> None:
        """§15.4.5: completion is refused while a required step is unfinished."""
        assert self.plan is not None
        if self.core_mode == CORE_GRAPH:
            done = self._graph_done(include_current=True)
            if not plan_policy.graph_completion_ready(self.plan, done):
                pending = plan_policy.graph_completion_pending(self.plan, done)
                step = self.plan.steps[self.step_index]
                self._hold(Blocked(step_id=pending[0] if pending else step.id, requires=list(step.requires),
                                   reason=plan_policy.REQUIRED_HOLD_REASON))
                return
        else:
            kwargs = {"step_index": self.step_index + 1, "skipped": self.steps_skipped,
                      "user_done": self.steps_user_done}
            if not plan_policy.completion_ready(self.plan, **kwargs):
                pending = plan_policy.required_pending(self.plan, **kwargs)
                step = self.plan.steps[self.step_index]
                self._hold(Blocked(step_id=pending[0] if pending else step.id, requires=list(step.requires),
                                   reason=plan_policy.REQUIRED_HOLD_REASON))
                return
        self.completion = "checking"
        self.pending = Pending(stage="confirm", trigger="goal_check", since=_now_ms())
        self._clear_hold()
        self._emit()
        self._confirm_task = self._spawn(self._confirm_flow())

    def _policy_advance(self, *, user_ack: bool) -> str | None:
        """Progression of a plan-mode run: every decision comes from :mod:`plan_policy`."""
        assert self.plan is not None
        self.tracked_s = 0.0
        adv = plan_policy.advance_by_one(self.plan, step_index=self.step_index, skipped=self.steps_skipped,
                                         user_done=self.steps_user_done, user_ack=user_ack)
        if adv.held:
            self._hold(adv.blocked)
            return None
        if self.step_index + 1 >= len(self.plan.steps):
            self._complete_or_hold()
            return None
        prereq = plan_policy.prerequisite_block(self.plan, self.step_index + 1,
                                                step_index=self.step_index + 1, skipped=self.steps_skipped,
                                                user_done=self.steps_user_done)
        if prereq is not None:
            self._hold(prereq)
            return None
        return self._enter_step(self.step_index + 1)

    def _plan_tick(self) -> str | None:
        """The timer wants to close the current step. Nothing moves while paused or while a proposal waits."""
        if self.paused:
            return None
        if self.proposal is not None:
            self.tracked_s = 0.0
            return None
        return self._policy_advance(user_ack=False)

    async def on_plan_edit(self, edit: PlanEditMsg) -> None:
        if self.phase != "reviewing" or self.plan is None:
            raise EngineReject("not_reviewing", "검토 중인 계획이 없습니다", retryable=False)
        try:
            self.plan = plan_policy.apply_plan_edit(self.plan, edit)
        except plan_policy.PlanEditError as exc:
            raise EngineReject("invalid_edit", str(exc), retryable=False) from None
        self._emit()
        self.sink.say(EDIT_SAY, "replace")

    async def on_plan_approve(self) -> None:
        if self.phase != "reviewing" or self.plan is None:
            raise EngineReject("not_reviewing", "검토 중인 계획이 없습니다", retryable=False)
        self.plan = plan_policy.approve_plan(self.plan)
        self.phase = "running"
        self.step_index = 0
        self.steps_user_done = []
        self.steps_skipped = []
        self.tracked_s = 0.0
        self.completion = "none"
        self.pending = None
        self.paused = False
        self.proposal = None
        self._hold_step, self._hold_streak, self._proposed = None, 0, False
        self._clear_hold()
        self._reacquire()
        self._emit()
        self.sink.say(self._step_line(), "replace")

    async def on_plan_discard(self) -> None:
        if self.phase != "reviewing":
            raise EngineReject("not_reviewing", "검토 중인 계획이 없습니다", retryable=False)
        self._cancel_tasks()
        self._reset_guide()
        self._emit()

    async def on_run_pause(self) -> None:
        if self.phase != "running":
            raise EngineReject("not_running", "진행 중인 가이드가 없습니다")
        self.paused = True
        self._emit()

    async def on_run_resume(self) -> None:
        if self.phase != "running":
            raise EngineReject("not_running", "진행 중인 가이드가 없습니다")
        self.paused = False
        self._reacquire()
        self._emit()

    async def on_step_ack(self, step_id: str) -> None:
        if self.phase != "running" or self.plan is None:
            raise EngineReject("not_running", "진행 중인 가이드가 없습니다")
        try:
            # §15.8: in graph mode a ``required`` step (visual or not) is finished by the user's own record.
            index = (plan_policy.graph_ack_index if self.core_mode == CORE_GRAPH
                     else plan_policy.ack_index)(self.plan, step_id)
        except plan_policy.PlanEditError as exc:
            raise EngineReject("invalid_edit", str(exc), retryable=False) from None
        if self.core_mode == CORE_GRAPH:
            blocker = plan_policy.graph_prerequisite_block(self.plan, step_id,
                                                           self._graph_done(include_current=True))
            if blocker is not None:
                self._hold(blocker)
                self._emit()
                return
        if step_id not in self.steps_user_done:
            self.steps_user_done.append(step_id)
        if index != self.step_index or self.paused or self.proposal is not None:
            self._emit()  # the user's own confirmation is recorded even when it is not the current step
            return
        line = self._policy_advance(user_ack=True)
        if line:
            self.sink.say(line, "replace")

    async def on_proposal(self, *, accept: bool) -> None:
        if self.proposal is None:
            raise EngineReject("no_proposal", "제안된 변경안이 없습니다", retryable=False)
        proposal = self.proposal
        self.proposal = None
        if accept:
            assert self.plan is not None
            self.plan = plan_policy.install_proposal(self.plan, proposal)
            self.step_index = min(self.step_index, len(self.plan.steps) - 1)
            self._clear_hold()
            self._emit()
            self.sink.say(f"제안을 반영했어요. {self._step_line()}", "replace")
            return
        self._emit()  # rejected: keep holding on the installed plan

    # ------------------------------------------------------------ replanning

    def _try_replan(self) -> bool:
        now = self.vnow()
        if self.proposal is not None:
            # §15.6: nothing changes while a proposal waits for the client's answer.
            self.replan_blocked_reason = "제안된 변경안을 먼저 확인해 주세요."
            return False
        if self.replans_used >= MAX_REPLANS:
            self.replan_blocked_reason = f"이번 가이드에서는 계획을 {MAX_REPLANS}번까지만 다시 짤 수 있어요."
            return False
        gap = now - self.last_replan_v
        if gap < REPLAN_GAP_S:
            wait = max(1, math.ceil(REPLAN_GAP_S - gap))
            self.replan_blocked_reason = f"계획을 방금 다시 짰어요. {wait}초 뒤에 다시 시도해 주세요."
            return False
        assert self.plan is not None
        self.replans_used += 1
        self.last_replan_v = now
        if self._confirm_task is not None:
            self._confirm_task.cancel()
            self._confirm_task = None
        self.plan = self._build_plan(revision=self.plan.revision + 1)
        self.step_index = 0
        self.steps_user_done = []
        self.steps_skipped = []
        self.completion = "none"
        self.tracked_s = 0.0
        self.pending = None
        self.proposal = None
        self._hold_step, self._hold_streak, self._proposed = None, 0, False
        self._clear_hold()
        self.replan_blocked_reason = None
        return True

    async def on_stop(self) -> None:
        was_idle = self.phase == "idle"
        self._cancel_tasks()
        self._reset_guide()
        if not was_idle:
            self._emit()

    async def on_talk(self, utterance: str) -> None:
        if self.phase != "running":
            raise EngineReject("not_running", "진행 중인 가이드가 없습니다")
        if self.talk_pending:
            raise EngineReject("talk_busy", "이전 질문에 답하는 중입니다")
        self.talk_pending = True
        self.pending = Pending(stage="talk", trigger="user", since=_now_ms())
        self._emit()
        self._spawn(self._talk_flow(utterance))

    async def _talk_flow(self, utterance: str) -> None:
        await self._sleep(1.0)
        if self.phase != "running" or self.plan is None:
            self.talk_pending = False
            self._emit()
            return
        reply = f"(모의 응답) \"{utterance}\"라고 하셨어요. 실제 엔진이 연결되면 화면을 보고 답합니다."
        spoken = "(모의 응답) 말씀 잘 들었어요."
        follow_line: str | None = None
        self.talk_pending = False
        if self.pending is not None and self.pending.stage == "talk":
            self.pending = None
        if "다시" in utterance:
            if self._try_replan():
                spoken = "알겠어요. 계획을 다시 짰어요."
                follow_line = self._step_line()
            else:
                spoken = "지금은 계획을 다시 짤 수 없어요."
        elif "했어" in utterance or "완료" in utterance:
            step_id = self.plan.steps[self.step_index].id
            if step_id not in self.steps_user_done and self.completion == "none":
                self.steps_user_done.append(step_id)
                spoken = "좋아요. 다음으로 넘어갈게요."
                self.talk = TalkResult(utterance=utterance, reply=reply[:TALK_REPLY_MAX], spoken=spoken, at=_now_ms())
                follow_line = (self._policy_advance(user_ack=True) if self._uses_plan()
                               else self._advance())  # emits
                self.sink.say(spoken, "replace")
                if follow_line:
                    self.sink.say(follow_line, "append")
                return
        self.talk = TalkResult(utterance=utterance, reply=reply[:TALK_REPLY_MAX], spoken=spoken, at=_now_ms())
        self._emit()
        self.sink.say(spoken, "replace")
        if follow_line:
            self.sink.say(follow_line, "append")

    async def on_follow_now(self) -> None:
        if self.phase != "running":
            raise EngineReject("not_running", "진행 중인 가이드가 없습니다")
        self.pending = Pending(stage="follow", trigger="manual", since=_now_ms())
        self._emit()

        async def recheck() -> None:
            await self._sleep(0.8)
            if self.phase != "running":
                return
            if self.pending is not None and self.pending.stage == "follow":
                self.pending = None
            self._set_notice("rechecked", "지금 화면을 다시 확인했어요.")
            self._emit()

        self._spawn(recheck())

    async def on_replan_now(self) -> None:
        if self.phase != "running":
            raise EngineReject("not_running", "진행 중인 가이드가 없습니다")
        ok = self._try_replan()
        self._emit()
        if ok:
            self.sink.say(f"계획을 다시 짰어요. {self._step_line()}", "replace")

    async def on_confirm_done(self) -> None:
        if self.completion != "confirmed":
            raise EngineReject("not_running", "아직 완료 확인 단계가 아닙니다")
        self.completion = "user_confirmed"
        self.phase = "completed"
        self.pending = None
        self._clear_notice()
        self._emit()
        self.sink.say("완료를 확인했어요. 수고하셨어요.", "replace")

    async def on_prefs(self, prefs: Prefs) -> None:
        self.prefs = prefs

    # ------------------------------------------------------------ frames

    def _box(self, now_v: float) -> Box:
        if self.seed_box is not None:
            s = self.seed_box
            w, h = s.width, s.height
            cx0, cy0 = s.x + w / 2, s.y + h / 2
            ang = (now_v - self.seed_v) * 0.6
            cx, cy = cx0 + 0.03 * (math.cos(ang) - 1), cy0 + 0.03 * math.sin(ang)
        else:
            w, h = 0.25, 0.2
            ang = now_v * 0.6
            cx, cy = 0.5 + 0.05 * math.cos(ang), 0.5 + 0.05 * math.sin(ang)
        w, h = round(w, 4), round(h, 4)
        x = min(max(round(cx - w / 2, 4), 0.0), round(1 - w, 4))
        y = min(max(round(cy - h / 2, 4), 0.0), round(1 - h, 4))
        return Box(x=x, y=y, width=w, height=h)

    async def on_frame(self, header: FrameHeader, jpeg: bytes) -> TrackMsg:
        base = {"seq": header.seq, "captured_at": header.captured_at}
        if self.phase == "idle" or self.run_id is None:
            return TrackMsg(state="idle", **base)
        if self.phase == "reviewing" or self.phase == "clarifying" or self.paused:
            # §15: reviewing shows a draft and a clarifying run waits for an answer; both answer frames with
            # idle. A paused run judges nothing.
            return TrackMsg(state="idle", **base)
        now_v = self.vnow()
        changed = False
        if header.select_box is not None:
            self.generation += 1
            self._track_counter += 1
            self.track_id = f"t{self._track_counter}"
            self.seed_box = header.select_box
            self.seed_v = now_v
            self.frames_in_gen = 0
            self.bound = False
            self.needs_reselect = False
            changed = True
        self.frames_in_gen += 1
        target = self._target()
        ids = {"run_id": self.run_id, "track_id": self.track_id, "generation": self.generation, "target": target}
        tracking = self.seed_box is not None or (target is not None and self.frames_in_gen > ACQUIRE_FRAMES)
        if not tracking:
            self.last_frame_v = now_v
            if changed:
                self._emit()
            return TrackMsg(state="acquiring", **base, **ids)
        if not self.bound:
            self.bound = True
            changed = True
        msg = TrackMsg(state="tracking", box=self._box(now_v), **base, **ids)
        line: str | None = None
        if self.phase == "running" and self.completion == "none" and not self.talk_pending and self._resumed.is_set():
            if self.last_frame_v is not None:
                self.tracked_s += min(max(now_v - self.last_frame_v, 0.0), MAX_FRAME_DT_S)
            if self.tracked_s >= self.cfg.step_s:
                line = (self._plan_tick() if self._uses_plan() else self._advance())  # emits
                changed = False
            elif self.tracked_s >= self.cfg.step_s - 1.0 and self.pending is None:
                self.pending = Pending(stage="follow", trigger="heartbeat", since=_now_ms())
                changed = True
        self.last_frame_v = now_v
        if changed:
            self._emit()
        if line:
            self.sink.say(line, "replace")
        return msg

    async def on_hi_frame(self, header: FrameHeader, jpeg: bytes) -> None:
        if header.hi_req is not None and header.hi_req == self.hi_req:
            self._hi_event.set()

    # ------------------------------------------------------------ connection lifecycle

    async def on_pause(self) -> None:
        self._resumed.clear()

    async def on_resume(self) -> None:
        # The tracker re-acquires the plan's target after a reconnect: a few frames of "acquiring" again.
        # A user's run_pause is deliberately NOT cleared here (§15.7): it only ends on run_resume.
        self._reacquire()

    async def close(self) -> None:
        self._cancel_tasks()
        self._resumed.set()

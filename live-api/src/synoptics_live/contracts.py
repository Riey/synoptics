"""Wire contracts of synoptics-live/0.1-draft (spec: ~/Projects/aisw/api-spec/live-api.md).

Field names and bounds follow the demo's contracts (aisw-hybrid-talk backend/app/*_contracts.py): Box is
0..1 normalised with x+width<=1, plan steps are s1..s16 with say/done_when <= 60 characters and at most two
commands of which at most one is an action, and the 21 action types carry the same direction rules.

Server -> client models are strict and ``extra="forbid"`` (we write them). Client -> server models ignore
unknown fields so an older server does not break a newer client; their known fields are still validated.
"""

from __future__ import annotations

import base64
import binascii
import io
from typing import Annotated, Iterable, Literal, Union
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError
from pydantic import (BaseModel, ConfigDict, Field, StrictBool, StrictInt, TypeAdapter, field_validator,
                      model_validator)

PROTOCOL = "synoptics-live/0.1-draft"

# ---------------------------------------------------------------- bounds (same as the demo contracts)

ANCHOR_LABEL_MAX = 40
STEP_SAY_MAX = 60
VISUAL_CONDITION_MAX = 60
GOAL_MAX = 300
CONTEXT_MAX = 1000
TALK_UTTERANCE_MAX = 200
TALK_REPLY_MAX = 400
TALK_SPOKEN_MAX = 60
#: §15: leave room for independently verifiable repeated actions. Ids are ``s1``..``s16``.
MAX_STEPS = 16
#: §15.4.3: with 16 steps and an explicit dependency graph, a node may name every other node as a prerequisite.
MAX_REQUIRES = MAX_STEPS - 1
MAX_STEP_TARGETS = 3
MATERIALS_MAX = 4
MATERIAL_TITLE_MAX = 80
MATERIAL_VERSION_MAX = 32
MATERIAL_TEXT_MAX = 4000
EVIDENCE_LOCATOR_MAX = 80
EVIDENCE_QUOTE_MAX = 120
PROPOSAL_REASON_MAX = 200

# ---------------------------------------------------------------- bounds (§15 novice plan assistance)

#: §15: one actionable question at a time. A run accepts at most this many answered questions, then stops
#: asking (the upstream's ``answers`` list is capped at the same number).
MAX_CLARIFICATION_ANSWERS = 12
CLARIFICATION_MAX = 300
CLARIFICATION_ID_MAX = 64
ANSWER_MAX = 1000
#: §15.2: readable novice instructions *under* ``say`` (which stays speech-sized). Never hidden reasoning.
DETAILS_MAX = 600
#: §15: public facts a planner actually observed with a research tool — never an invented citation.
RESEARCH_SOURCES_MAX = 6
RESEARCH_URL_MAX = 2048
RESEARCH_TITLE_MAX = 160
RESEARCH_SUMMARY_MAX = 1200
#: §15: reference photos (a close-up, a label) are separate evidence, never the current scene. The bounds
#: mirror the demo backend's ``validate_jpeg`` so Live refuses a photo the upstream would refuse anyway.
REFERENCE_IMAGES_MAX = 12
REFERENCE_IMAGE_BYTES_MAX = 1_500_000
REFERENCE_IMAGE_MIN_WIDTH = 320
REFERENCE_IMAGE_MIN_HEIGHT = 240
REFERENCE_IMAGE_MAX_SIDE = 2048
REFERENCE_IMAGE_MAX_PIXELS = 2_073_600
#: The base64-inflated envelope those photos may arrive in (the demo backend's guidance body cap). Only a
#: message that may carry reference photos is allowed to exceed the ordinary control-message bound.
REFERENCE_MESSAGE_BYTES_MAX = 10 * 1024 * 1024


class WireModel(BaseModel):
    """Server-written messages: strict, closed."""

    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)


class ClientModel(BaseModel):
    """Client-written messages: known fields strict, unknown fields ignored (forward compatible)."""

    model_config = ConfigDict(strict=True, extra="ignore", validate_default=True)


Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveUnit = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
NonNegInt = Annotated[StrictInt, Field(ge=0)]
EpochMs = Annotated[StrictInt, Field(ge=0)]


class Box(WireModel):
    x: Unit
    y: Unit
    width: PositiveUnit
    height: PositiveUnit

    @model_validator(mode="after")
    def inside_frame(self) -> Box:
        if self.x + self.width > 1 + 1e-9 or self.y + self.height > 1 + 1e-9:
            raise ValueError("box extends beyond the image")
        return self


# ---------------------------------------------------------------- limits (§4.1)


class Limits(WireModel):
    stream_long_side: StrictInt = 640
    stream_jpeg_quality: float = 0.7
    stream_max_bytes: StrictInt = 120_000
    hi_long_side: StrictInt = 1024
    hi_max_bytes: StrictInt = 500_000
    max_fps: StrictInt = 15
    box_max_age_ms: StrictInt = 500
    tts_wait_ms: StrictInt = 1500
    resume_grace_ms: StrictInt = 30_000
    goal_max_chars: StrictInt = GOAL_MAX
    context_max_chars: StrictInt = CONTEXT_MAX
    utterance_max_chars: StrictInt = TALK_UTTERANCE_MAX
    #: §15.3/§15.2: Plan 모드 상한. 클라이언트는 단계 id와 개수를 이 값에 맞춰 검증하세요.
    plan_steps_max: StrictInt = MAX_STEPS
    materials_max: StrictInt = MATERIALS_MAX
    material_text_max_chars: StrictInt = MATERIAL_TEXT_MAX
    #: §15: 한 번에 보낼 수 있는 참조 사진 수(각 JPEG ≤1.5MB). 현재 장면이 아니라 별도의 참고 증거입니다.
    reference_images_max: StrictInt = REFERENCE_IMAGES_MAX


# ---------------------------------------------------------------- plan / commands (§6.2, §8)

TARGET_ROLE = "target"
AnchorId = Annotated[str, Field(min_length=1, max_length=16)]
AnchorLabel = Annotated[str, Field(min_length=1, max_length=ANCHOR_LABEL_MAX)]
StepId = Annotated[str, Field(pattern=r"^s(?:[1-9]|1[0-6])$")]
StepSay = Annotated[str, Field(min_length=1, max_length=STEP_SAY_MAX)]
VisualCondition = Annotated[str, Field(min_length=1, max_length=VISUAL_CONDITION_MAX)]
Pad = Annotated[float, Field(ge=0, le=0.5, allow_inf_nan=False)]
#: §15.2: a step's logical subject ("상자", "완충재"). A label, never a coordinate.
StepTarget = Annotated[str, Field(min_length=1, max_length=ANCHOR_LABEL_MAX)]
MaterialId = Annotated[str, Field(pattern=r"^m(?:[1-9]|1[0-2])$")]
#: §15.4: what decides a step is finished. ``visual`` (the frame), ``user``/``measure`` (the user says so).
CheckMethod = Literal["visual", "user", "measure"]
PlanStatus = Literal["draft", "approved"]
#: §15: the closed set of upstream planning profiles. Both run high reasoning under the same provider deadline;
#: the choice describes configured capability, not a quota/credit guarantee. Changing it starts a fresh run.
PLAN_MODELS = ("deepseek:high", "astra:high")
DEFAULT_PLAN_MODEL = "deepseek:high"
PlanModel = Literal["deepseek:high", "astra:high"]
#: §15.8: the closed set of guide cores — how the follower is asked and how the screen may move. ``classic``
#: asks a follow call about every remaining step (a later step's ``yes`` may move the screen); ``sequential``
#: scopes every follower answer to the current step, so only an upper ``step_done`` on that same step moves it;
#: ``graph`` asks about EVERY plan step (earlier/done ones included) and keeps an explicit per-node ledger — a
#: visual ``yes`` only nominates an eligible node and the upper ``step_done`` on that same node commits it, so
#: independent nodes may complete in either order. Independent of ``plan_mode`` (the review boundary) and
#: ``plan_model`` (which upper profile writes the plan). Fixed for a run: it joins the plan's identity upstream,
#: and a different choice is a different ``start``.
CORE_MODES = ("classic", "sequential", "graph")
#: The members by name, so the engines never spell the strings themselves.
CORE_CLASSIC, CORE_SEQUENTIAL, CORE_GRAPH = CORE_MODES
DEFAULT_CORE_MODE = CORE_CLASSIC
CoreMode = Literal["classic", "sequential", "graph"]

#: §15.4.6: how a step's ``done_when`` predicate is read once it is satisfied. ``state`` (default): the latest
#: applicable reading may revoke an earlier ``yes`` (the condition can stop holding). ``event``: a verified
#: occurrence is historical — a later frame not showing it does not un-happen it. Historical occurrence is not
#: still-applicable authorization: revoking a prerequisite still puts every dependent node into recheck.
CONDITION_KINDS = ("state", "event")
DEFAULT_CONDITION_KIND = "state"
ConditionKind = Literal["state", "event"]
#: §15.8 graph: the follower's current reading of one node.
StepStatus = Literal["yes", "no", "unsure"]


def fold_condition_kind(value: object) -> object:
    """``state``/``event`` are accepted case-folded on the wire; anything else is left for the Literal to reject."""
    return value.strip().lower() if isinstance(value, str) else value

ActionType = Literal[
    "press", "grasp", "move", "rotate", "open", "close", "insert", "remove",
    "attach", "fold", "place", "fit", "screw", "hold", "flip", "pull", "push",
    # circuit-assembly actions: direction-free (``none`` only), like everything outside the side/rotate/screw sets.
    "align", "connect", "disconnect", "bend",
]
ActionDirection = Literal[
    "up", "down", "left", "right", "clockwise", "counterclockwise", "tighten", "loosen", "none",
]
SIDE_DIRECTION_ACTIONS = {"move", "pull", "push"}
MOVE_DIRECTIONS = {"up", "down", "left", "right"}
ROTATE_DIRECTIONS = {"clockwise", "counterclockwise"}
SCREW_DIRECTIONS = {"tighten", "loosen"}


def check_direction(action: str, direction: str) -> None:
    if action in SIDE_DIRECTION_ACTIONS:
        ok = direction in MOVE_DIRECTIONS
    elif action == "rotate":
        ok = direction in ROTATE_DIRECTIONS
    elif action == "screw":
        ok = direction in SCREW_DIRECTIONS
    else:
        ok = direction == "none"
    if not ok:
        raise ValueError(f"direction '{direction}' is not allowed for action '{action}'")


class FocusAnchorCommand(WireModel):
    kind: Literal["focus"]
    anchor: AnchorId
    pad: Pad = 0.15


class LabelAnchorCommand(WireModel):
    kind: Literal["label"]
    anchor: AnchorId
    text: AnchorLabel


class ActionAnchorCommand(WireModel):
    kind: Literal["action"]
    anchor: AnchorId
    action: ActionType
    direction: ActionDirection

    @model_validator(mode="after")
    def direction_matches_action(self) -> ActionAnchorCommand:
        check_direction(self.action, self.direction)
        return self


AnchoredCommand = Annotated[
    Union[FocusAnchorCommand, LabelAnchorCommand, ActionAnchorCommand], Field(discriminator="kind")
]


def _at_most_one_action(commands: list) -> None:
    if sum(1 for c in commands if c.kind == "action") > 1:
        raise ValueError("at most one action command per step")


class Evidence(WireModel):
    """§15.2: which material (and where in it) a step's instruction came from. Never invented by the server."""

    material_id: MaterialId
    version: Annotated[str, Field(min_length=1, max_length=MATERIAL_VERSION_MAX)] | None = None
    locator: Annotated[str, Field(min_length=1, max_length=EVIDENCE_LOCATOR_MAX)] | None = None
    quote: Annotated[str, Field(min_length=1, max_length=EVIDENCE_QUOTE_MAX)] | None = None


class GuideStep(WireModel):
    id: StepId
    say: StepSay
    #: §15.2: readable novice instructions under the spoken ``say`` (never hidden model reasoning). Preserved
    #: by edits/replans/approval; the client shows it as the step's detail text.
    details: Annotated[str, Field(min_length=1, max_length=DETAILS_MAX)] | None = None
    done_when: VisualCondition
    commands: Annotated[list[AnchoredCommand], Field(max_length=2)]
    #: §15.4: ``user``/``measure`` steps finish only through ``step_ack``, never on the frame alone.
    check: CheckMethod = "visual"
    #: §15.4.2: a required step (a mandatory inspection point) can never be skipped.
    required: StrictBool = False
    #: §15.4.3: steps that must be done before this one can become current.
    requires: Annotated[list[StepId], Field(max_length=MAX_REQUIRES)] = Field(default_factory=list)
    #: §15.2: logical subjects of this step, for the work list. Not the tracked anchor.
    targets: Annotated[list[StepTarget], Field(max_length=MAX_STEP_TARGETS)] = Field(default_factory=list)
    #: §15.4.6: ``state`` (a condition that currently holds; the latest reading may revoke it) or ``event``
    #: (an occurrence that happened; a later frame not showing it does not un-happen it).
    condition_kind: ConditionKind = DEFAULT_CONDITION_KIND
    #: §15.4.8: whether this step's completion is part of the ORIGINAL goal the user asked for. ``required``
    #: stays the mandatory procedural/user-check flag (it gates safe action and is what an unverified check is
    #: reported as); ``goal_required`` decides whether an unfinished step keeps the goal unachieved:
    #: ``True/True`` a mandatory goal step · ``False/True`` a safety precondition or gate (blocking the action,
    #: never reported as the goal being unachieved) · ``False/False`` optional/supporting.
    goal_required: StrictBool = True
    evidence: Evidence | None = None

    @field_validator("condition_kind", mode="before")
    @classmethod
    def _fold_condition_kind(cls, value: object) -> object:
        return fold_condition_kind(value)

    @model_validator(mode="after")
    def role_commands_only(self) -> GuideStep:
        for command in self.commands:
            if command.anchor != TARGET_ROLE:
                raise ValueError("plan commands must refer to the role 'target'")
        _at_most_one_action(self.commands)
        if len(set(self.requires)) != len(self.requires):
            raise ValueError("duplicate step ids in requires")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate logical targets")
        return self


class Material(WireModel):
    """§15.3: a reference document the plan was built from. The extracted text itself is not sent back."""

    id: MaterialId
    title: Annotated[str, Field(min_length=1, max_length=MATERIAL_TITLE_MAX)]
    version: Annotated[str, Field(min_length=1, max_length=MATERIAL_VERSION_MAX)] | None = None
    chars: NonNegInt
    added_at: EpochMs


class ResearchSource(WireModel):
    """§15: one public fact a research tool actually returned. Live keeps only well-formed public URLs —
    arbitrary model-generated links are never treated as verified evidence."""

    url: Annotated[str, Field(min_length=1, max_length=RESEARCH_URL_MAX)]
    title: Annotated[str, Field(min_length=1, max_length=RESEARCH_TITLE_MAX)]
    summary: Annotated[str, Field(min_length=1, max_length=RESEARCH_SUMMARY_MAX)]

    @model_validator(mode="after")
    def public_url(self) -> ResearchSource:
        try:
            parsed = urlsplit(self.url)
            valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
            valid = valid and parsed.username is None and parsed.password is None
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("research sources require a credential-free HTTP(S) URL")
        return self


class PlanAnswer(WireModel):
    """§15: one answered clarification, retained as the user's own evidence. It never changes the goal."""

    question: Annotated[str, Field(min_length=1, max_length=CLARIFICATION_MAX)]
    answer: Annotated[str, Field(min_length=1, max_length=ANSWER_MAX)]


def check_step_graph(steps: list[GuideStep]) -> None:
    """Ids unique, ``requires`` resolvable, acyclic (§15.2). Raises ValueError on the first violation."""
    ids = [step.id for step in steps]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate step ids")
    known = set(ids)
    by_id = {step.id: step for step in steps}
    for step in steps:
        for need in step.requires:
            if need not in known:
                raise ValueError(f"step {step.id} requires unknown step {need}")
            if need == step.id:
                raise ValueError(f"step {step.id} requires itself")
    state: dict[str, int] = {}

    def visit(node: str, chain: tuple[str, ...]) -> None:
        if state.get(node) == 1:
            raise ValueError("requires cycle: " + " -> ".join((*chain, node)))
        if state.get(node) == 2:
            return
        state[node] = 1
        for need in by_id[node].requires:
            visit(need, (*chain, node))
        state[node] = 2

    for step in steps:
        visit(step.id, ())


class Plan(WireModel):
    plan_id: Annotated[str, Field(min_length=1, max_length=64)]
    revision: Annotated[StrictInt, Field(ge=1)]
    #: §15.5: ``draft`` while the user reviews; ``approved`` once execution is pinned to ``approved_revision``.
    #: Immediate guidance (§11) is simply an approved plan with no pinned revision.
    status: PlanStatus = "approved"
    approved_revision: Annotated[StrictInt, Field(ge=1)] | None = None
    target: AnchorLabel
    goal_when: VisualCondition
    materials: Annotated[list[Material], Field(max_length=MATERIALS_MAX)] = Field(default_factory=list)
    steps: Annotated[list[GuideStep], Field(min_length=1, max_length=MAX_STEPS)]
    #: §15: the public facts the plan was written from (also mirrored in ``state.research_sources``). They are
    #: evidence the planner read — never a permission to execute and never the user's own goal.
    research_sources: Annotated[list[ResearchSource], Field(max_length=RESEARCH_SOURCES_MAX)] = Field(
        default_factory=list)

    @model_validator(mode="after")
    def consistent(self) -> Plan:
        check_step_graph(self.steps)
        if self.approved_revision is not None and self.approved_revision > self.revision:
            raise ValueError("approved_revision cannot be ahead of revision")
        if self.status == "approved" and self.approved_revision not in (None, self.revision):
            raise ValueError("an approved plan is pinned to its own revision")
        if self.status == "draft" and self.approved_revision is not None:
            raise ValueError("a draft cannot be approved")
        known = {material.id for material in self.materials}
        for step in self.steps:
            if step.evidence is not None and step.evidence.material_id not in known:
                raise ValueError(f"step {step.id} cites an unknown material")
        return self


class Proposal(WireModel):
    """§15.6: a change the server wants to make mid-run. The installed plan stays authoritative until the
    client answers ``proposal_accept`` (or ``proposal_reject``)."""

    plan_id: Annotated[str, Field(min_length=1, max_length=64)]
    revision: Annotated[StrictInt, Field(ge=1)]
    reason: Annotated[str, Field(min_length=1, max_length=PROPOSAL_REASON_MAX)]
    goal_when: VisualCondition
    steps: Annotated[list[GuideStep], Field(min_length=1, max_length=MAX_STEPS)]

    @model_validator(mode="after")
    def consistent(self) -> Proposal:
        check_step_graph(self.steps)
        return self


class Blocked(WireModel):
    """§15.4: why the run cannot move on right now. The client shows the reason; the server keeps waiting."""

    step_id: StepId
    requires: Annotated[list[StepId], Field(max_length=MAX_REQUIRES)] = Field(default_factory=list)
    reason: Annotated[str, Field(min_length=1, max_length=200)]


class FocusOverlay(WireModel):
    kind: Literal["focus"]
    pad: Pad = 0.15


class LabelOverlay(WireModel):
    kind: Literal["label"]
    text: AnchorLabel


class ActionOverlay(WireModel):
    kind: Literal["action"]
    action: ActionType
    direction: ActionDirection

    @model_validator(mode="after")
    def direction_matches_action(self) -> ActionOverlay:
        check_direction(self.action, self.direction)
        return self


OverlayCommand = Annotated[Union[FocusOverlay, LabelOverlay, ActionOverlay], Field(discriminator="kind")]


def overlay_command_from_step(command: FocusAnchorCommand | LabelAnchorCommand | ActionAnchorCommand):
    """A plan command with its ``anchor`` role removed (the overlay binding names the anchor instead)."""
    data = command.model_dump(exclude={"anchor"})
    return {"focus": FocusOverlay, "label": LabelOverlay, "action": ActionOverlay}[command.kind](**data)


class OverlayBinding(WireModel):
    anchor_id: AnchorId
    run_id: Annotated[str, Field(min_length=1, max_length=64)]
    track_id: Annotated[str, Field(min_length=1, max_length=64)]
    generation: NonNegInt


class Overlay(WireModel):
    binding: OverlayBinding
    commands: Annotated[list[OverlayCommand], Field(max_length=2)]
    warn: Literal["needs_reselect"] | None = None
    motion_key: Annotated[str, Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def one_action(self) -> Overlay:
        _at_most_one_action(self.commands)
        return self


# ---------------------------------------------------------------- state snapshot (§6.2)

#: §15: ``clarifying`` is the plan-mode question turn — the run waits for one answer (``plan_answer``) before
#: it plans again; nothing is tracked, judged or executed while it is current.
Phase = Literal["idle", "planning", "clarifying", "reviewing", "running", "completed", "error"]
Completion = Literal["none", "checking", "confirmed", "user_confirmed"]
PendingStage = Literal["plan", "follow", "confirm", "talk"]


class Partial(WireModel):
    target: AnchorLabel | None = None
    first_say: StepSay | None = None


class Pending(WireModel):
    stage: PendingStage
    trigger: Annotated[str, Field(min_length=1, max_length=32)]
    since: EpochMs


class Notice(WireModel):
    code: Annotated[str, Field(min_length=1, max_length=64)]
    text: Annotated[str, Field(min_length=1, max_length=200)]


class BudgetNotice(WireModel):
    text: Annotated[str, Field(min_length=1, max_length=200)]
    until: EpochMs


class TalkResult(WireModel):
    utterance: Annotated[str, Field(min_length=1, max_length=TALK_UTTERANCE_MAX)]
    reply: Annotated[str, Field(min_length=1, max_length=TALK_REPLY_MAX)]
    spoken: Annotated[str, Field(min_length=1, max_length=TALK_SPOKEN_MAX)]
    at: EpochMs


class StateError(WireModel):
    code: Annotated[str, Field(min_length=1, max_length=64)]
    message: Annotated[str, Field(min_length=1, max_length=300)]
    retryable: StrictBool


class GuideState(WireModel):
    """Everything in a ``state`` message except ``type`` and ``rev`` (the server owns ``rev``)."""

    phase: Phase = "idle"
    #: §15.8: the guide core this run executes (``classic`` default). Part of the run's identity: it is chosen
    #: before ``start`` and never changes mid-run, so a reconnecting client knows which follow semantics apply.
    core_mode: CoreMode = DEFAULT_CORE_MODE
    plan: Plan | None = None
    partial: Partial | None = None
    clarification: Annotated[str, Field(min_length=1, max_length=CLARIFICATION_MAX)] | None = None
    #: §15: paired with ``clarification`` while ``phase == "clarifying"`` — the id ``plan_answer`` must echo.
    #: It is null in every other phase (the basic flow's idle clarification notice has no answer to fence).
    clarification_id: Annotated[str, Field(min_length=1, max_length=CLARIFICATION_ID_MAX)] | None = None
    #: §15: the public facts the plan (or the pending clarification) was written from, bounded and text-only.
    research_sources: Annotated[list[ResearchSource], Field(max_length=RESEARCH_SOURCES_MAX)] = Field(
        default_factory=list)
    step_index: NonNegInt = 0
    steps_skipped: list[StepId] = Field(default_factory=list)
    steps_user_done: list[StepId] = Field(default_factory=list)
    #: §15.8 graph: nodes whose completion is explicitly recorded — a visual commit the upper confirmed, or the
    #: user's own ``step_ack``. Never derived from ``step_index``: index movement is not completion.
    steps_done: list[StepId] = Field(default_factory=list)
    #: §15.8 graph: enabled nodes — every prerequisite is done and the node itself is not (deterministic order).
    steps_ready: list[StepId] = Field(default_factory=list)
    #: §15.8 graph: the follower's current reading of each node it has looked at. ``unsure`` never asserts absence.
    step_statuses: dict[StepId, StepStatus] = Field(default_factory=dict)
    #: §15.4.8: mandatory checks (``required``) that are still unverified. They never keep the goal unachieved
    #: when they are not ``goal_required``; the client shows them beside the goal as outstanding safety/gate
    #: checks. Live while running, frozen at the goal confirmation.
    pending_required_checks: list[StepId] = Field(default_factory=list)
    pending: Pending | None = None
    basis_age_s: Annotated[float, Field(ge=0)] | None = None
    notice: Notice | None = None
    budget_notice: BudgetNotice | None = None
    completion: Completion = "none"
    needs_reselect: StrictBool = False
    replan_blocked_reason: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    talk: TalkResult | None = None
    talk_pending: StrictBool = False
    talk_blocked_reason: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    overlay: Overlay | None = None
    #: §15.7: the user paused the run. Judgement and progression stop; a reconnect does not clear it.
    paused: StrictBool = False
    #: §15.6: a change proposed while running; the installed plan stays authoritative until it is answered.
    proposal: Proposal | None = None
    #: §15.4: why the run cannot move on (required check, unmet prerequisite).
    blocked: Blocked | None = None
    error: StateError | None = None

    @model_validator(mode="after")
    def consistent(self) -> GuideState:
        if (self.error is not None) != (self.phase == "error"):
            raise ValueError("error is present if and only if phase == 'error'")
        if self.plan is not None and self.step_index >= len(self.plan.steps):
            raise ValueError("step_index out of range")
        if self.proposal is not None and self.phase != "running":
            raise ValueError("a proposal exists only while running")
        if self.paused and self.phase != "running":
            raise ValueError("paused is a running state")
        if self.plan is not None:
            if self.plan.status == "draft" and self.phase not in ("reviewing",):
                raise ValueError("a draft plan exists only while reviewing")
            if self.phase == "reviewing" and self.plan.status != "draft":
                raise ValueError("reviewing shows a draft")
            known = {step.id for step in self.plan.steps}
            for label, ids in (("steps_done", self.steps_done), ("steps_ready", self.steps_ready),
                               ("pending_required_checks", self.pending_required_checks)):
                if len(set(ids)) != len(ids) or any(step_id not in known for step_id in ids):
                    raise ValueError(f"{label} must name plan steps at most once")
            if any(step_id not in known for step_id in self.step_statuses):
                raise ValueError("step_statuses names a step outside the plan")
        elif self.steps_done or self.steps_ready or self.step_statuses or self.pending_required_checks:
            raise ValueError("the graph ledger exists only with a plan")
        if self.clarification_id is not None and self.phase != "clarifying":
            raise ValueError("clarification_id exists only while clarifying")
        if self.phase == "clarifying":
            if self.clarification is None or self.clarification_id is None:
                raise ValueError("clarifying needs the question and the id its answer must echo")
            if self.plan is not None:
                raise ValueError("a clarifying run has no plan yet")
        return self


class StateMsg(GuideState):
    type: Literal["state"] = "state"
    rev: NonNegInt


# ---------------------------------------------------------------- server -> client text messages

TrackState = Literal["idle", "acquiring", "tracking", "occluded", "lost", "unavailable"]


class TrackMsg(WireModel):
    type: Literal["track"] = "track"
    seq: NonNegInt
    captured_at: EpochMs
    state: TrackState
    run_id: str | None = None
    track_id: str | None = None
    generation: NonNegInt | None = None
    target: str | None = None
    box: Box | None = None

    @model_validator(mode="after")
    def box_only_while_tracking(self) -> TrackMsg:
        if (self.box is not None) != (self.state == "tracking"):
            raise ValueError("box must be present if and only if state == 'tracking'")
        return self


class ReadyMsg(WireModel):
    type: Literal["ready"] = "ready"
    rev: NonNegInt
    server_time: EpochMs
    limits: Limits


class SayMsg(WireModel):
    type: Literal["say"] = "say"
    line_id: Annotated[str, Field(min_length=1, max_length=32)]
    text: Annotated[str, Field(min_length=1, max_length=TALK_REPLY_MAX)]
    mode: Literal["replace", "append"]
    audio: Literal["pending", "none"]


class CaptureHiMsg(WireModel):
    type: Literal["capture_hi"] = "capture_hi"
    req_id: Annotated[str, Field(min_length=1, max_length=32)]


ErrorCode = Literal[
    "invalid_access_code", "rate_limited", "capacity", "consent_required", "not_running",
    "plan_in_flight", "talk_busy", "talk_budget", "frame_invalid", "frame_too_large",
    "tracker_unavailable", "provider_error", "unsupported_protocol", "invalid_message",
    # §15 (Plan 모드)
    "not_reviewing", "plan_locked", "invalid_edit", "required_check", "no_proposal", "plan_changed",
    # §15 (질문 왕복): 확인할 질문이 없거나, 지난 질문 id로 답했거나, 참조 사진이 계약 밖일 때.
    "no_clarification", "clarification_stale", "image_too_large", "invalid_image",
]


class ErrorMsg(WireModel):
    type: Literal["error"] = "error"
    code: ErrorCode
    message: Annotated[str, Field(min_length=1, max_length=300)]
    retryable: StrictBool
    retry_after_ms: NonNegInt | None = None
    fatal: StrictBool = False


ByeReason = Literal[
    "session_expired", "session_ended", "replaced", "auth_failed", "protocol_violation",
    "unsupported_protocol", "message_too_large", "server_restart",
]


class ByeMsg(WireModel):
    type: Literal["bye"] = "bye"
    reason: ByeReason


# ---------------------------------------------------------------- binary headers (§5.1, §6.4)


class FrameHeader(ClientModel):
    t: Literal["frame"]
    seq: NonNegInt
    captured_at: EpochMs
    w: Annotated[StrictInt, Field(gt=0, le=8192)]
    h: Annotated[StrictInt, Field(gt=0, le=8192)]
    hi_req: Annotated[str, Field(min_length=1, max_length=32)] | None = None
    select_box: Box | None = None


class AudioHeader(WireModel):
    t: Literal["audio"] = "audio"
    line_id: Annotated[str, Field(min_length=1, max_length=32)]
    mime: Annotated[str, Field(min_length=1, max_length=64)]
    duration_ms: NonNegInt


# ---------------------------------------------------------------- client -> server text messages (§5.2)


class HelloMsg(ClientModel):
    type: Literal["hello"]
    token: Annotated[str, Field(min_length=1, max_length=256)]
    protocol: Annotated[str, Field(min_length=1, max_length=64)]
    resume_rev: NonNegInt | None = None
    audio_accept: Annotated[list[Annotated[str, Field(max_length=64)]], Field(max_length=8)] | None = None


class MaterialInput(ClientModel):
    """§15.3: extracted text of a reference document. The client extracts; the server never parses files."""

    title: Annotated[str, Field(min_length=1, max_length=MATERIAL_TITLE_MAX)]
    version: Annotated[str, Field(min_length=1, max_length=MATERIAL_VERSION_MAX)] | None = None
    text: Annotated[str, Field(min_length=1, max_length=MATERIAL_TEXT_MAX)]


class ReferenceImage(ClientModel):
    """§15: one reference photo (a close-up, a label, a drawing) the user supplies as *separate* evidence.

    It is never the current scene: the plan's scene stays the camera frame. ``label`` names what it shows. The
    bytes are validated (JPEG, ≤1.5MB, the demo backend's dimension bounds) before any provider call, kept only
    in the current run's memory, and never logged, persisted or echoed back in ``state``.
    """

    frame_id: Annotated[str, Field(min_length=1, max_length=64)]
    image_base64: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1, max_length=MATERIAL_TITLE_MAX)] | None = None


def reference_images_problem(images: Iterable[ReferenceImage]) -> tuple[str, str] | None:
    """``(error_code, message)`` for the first unusable reference photo; ``None`` when every one is usable.

    Mirrors the demo backend's ``validate_jpeg`` so Live refuses a photo before any paid dispatch instead of
    passing it on to be refused (or to be stored) there. Never reads anything but the image itself.
    """
    for index, image in enumerate(images, start=1):
        where = f"{index}번째 참조 사진"
        encoded = image.image_base64.strip()
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1].strip()
        if len(encoded) > (REFERENCE_IMAGE_BYTES_MAX + 2) // 3 * 4 + 4:
            return "image_too_large", f"{where}이 너무 큽니다(JPEG 1.5MB 이하)."
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return "invalid_image", f"{where}을(를) 읽지 못했습니다. JPEG로 다시 보내주세요."
        if len(data) > REFERENCE_IMAGE_BYTES_MAX:
            return "image_too_large", f"{where}이 너무 큽니다(JPEG 1.5MB 이하)."
        if not data.startswith(b"\xff\xd8\xff"):
            return "invalid_image", f"{where}이 JPEG가 아닙니다."
        try:
            with Image.open(io.BytesIO(data)) as opened:
                if opened.format != "JPEG":
                    return "invalid_image", f"{where}이 JPEG가 아닙니다."
                width, height = opened.size
                opened.load()
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            return "invalid_image", f"{where}을(를) 읽지 못했습니다. JPEG로 다시 보내주세요."
        if (width < REFERENCE_IMAGE_MIN_WIDTH or height < REFERENCE_IMAGE_MIN_HEIGHT
                or width > REFERENCE_IMAGE_MAX_SIDE or height > REFERENCE_IMAGE_MAX_SIDE
                or width * height > REFERENCE_IMAGE_MAX_PIXELS):
            return "invalid_image", f"{where} 크기가 맞지 않습니다({width}x{height})."
    return None


class StartMsg(ClientModel):
    type: Literal["start"]
    goal: Annotated[str, Field(min_length=1, max_length=GOAL_MAX)]
    context: Annotated[str, Field(max_length=CONTEXT_MAX)] | None = None
    consent_ai: StrictBool | None = None
    #: §15: true = 계획을 만들고 검토·승인한 뒤 실행. 생략하면 지금까지의 즉시 안내 흐름.
    plan_mode: StrictBool | None = None
    #: §15: 이 run의 계획·확인·재계획·대화에 쓸 상위 planning profile. 생략하면 기본 ``deepseek:high``.
    #: ``plan_mode``(검토 경계)와 독립이며, 실행 중에는 바꿀 수 없습니다 — 바꾸려면 새 ``start``.
    plan_model: PlanModel = DEFAULT_PLAN_MODEL
    #: §15.8: 이 run의 안내 코어. ``classic``(기본)은 남은 단계 전체를 추종자에게 묻고, ``sequential``은
    #: 현재 단계 하나만, ``graph``는 이전·완료 단계까지 포함한 모든 단계를 묻고 노드별 원장으로 전진합니다.
    #: ``plan_mode``·``plan_model``과 독립이고 실행 중에는 바꿀 수 없습니다.
    core_mode: CoreMode = DEFAULT_CORE_MODE
    materials: Annotated[list[MaterialInput], Field(max_length=MATERIALS_MAX)] | None = None
    #: §15: 참조 사진(별도 증거). 생략하면 없음. 각 JPEG ≤1.5MB, 최대 12장. 현재 장면은 여전히 카메라 프레임입니다.
    reference_images: Annotated[list[ReferenceImage], Field(max_length=REFERENCE_IMAGES_MAX)] | None = None


class StepEdit(ClientModel):
    """§15.5: a patch to one draft step. Omitted fields keep their value; ``commands`` are never edited."""

    step_id: StepId
    say: StepSay | None = None
    details: Annotated[str, Field(min_length=1, max_length=DETAILS_MAX)] | None = None
    done_when: VisualCondition | None = None
    check: CheckMethod | None = None
    required: StrictBool | None = None
    requires: Annotated[list[StepId], Field(max_length=MAX_REQUIRES)] | None = None
    targets: Annotated[list[StepTarget], Field(max_length=MAX_STEP_TARGETS)] | None = None
    #: §15.4.6: switch a draft step between a revocable ``state`` and a historical ``event``.
    condition_kind: ConditionKind | None = None
    #: §15.4.8: switch a draft step between a goal step, a safety precondition, and an optional step.
    goal_required: StrictBool | None = None

    @field_validator("condition_kind", mode="before")
    @classmethod
    def _fold_condition_kind(cls, value: object) -> object:
        return fold_condition_kind(value)


class StepAdd(ClientModel):
    """§15.5: a new draft step. The server assigns the id and returns it in the next ``state``."""

    say: StepSay
    details: Annotated[str, Field(min_length=1, max_length=DETAILS_MAX)] | None = None
    done_when: VisualCondition
    check: CheckMethod | None = None
    required: StrictBool | None = None
    requires: Annotated[list[StepId], Field(max_length=MAX_REQUIRES)] | None = None
    targets: Annotated[list[StepTarget], Field(max_length=MAX_STEP_TARGETS)] | None = None
    condition_kind: ConditionKind | None = None
    goal_required: StrictBool | None = None

    @field_validator("condition_kind", mode="before")
    @classmethod
    def _fold_condition_kind(cls, value: object) -> object:
        return fold_condition_kind(value)


class PlanEditMsg(ClientModel):
    type: Literal["plan_edit"]
    edits: Annotated[list[StepEdit], Field(max_length=MAX_STEPS)] | None = None
    add: Annotated[list[StepAdd], Field(max_length=MAX_STEPS)] | None = None
    remove: Annotated[list[StepId], Field(max_length=MAX_STEPS)] | None = None
    order: Annotated[list[StepId], Field(max_length=MAX_STEPS)] | None = None
    goal_when: VisualCondition | None = None


class PlanApproveMsg(ClientModel):
    type: Literal["plan_approve"]


class PlanDiscardMsg(ClientModel):
    type: Literal["plan_discard"]


class PlanAnswerMsg(ClientModel):
    """§15: one answer to the single question a clarifying run asked.

    ``clarification_id`` must be the id of the question that is CURRENT for this run; it is what makes a late
    or duplicated answer from an earlier question harmless. ``reference_images`` — when the field is present at
    all, even empty — REPLACES the run's whole reference set; omitting it keeps what the run already has.
    """

    type: Literal["plan_answer"]
    clarification_id: Annotated[str, Field(min_length=1, max_length=CLARIFICATION_ID_MAX)]
    answer: Annotated[str, Field(min_length=1, max_length=ANSWER_MAX)]
    reference_images: Annotated[list[ReferenceImage], Field(max_length=REFERENCE_IMAGES_MAX)] | None = None


class RunPauseMsg(ClientModel):
    type: Literal["run_pause"]


class RunResumeMsg(ClientModel):
    type: Literal["run_resume"]


class StepAckMsg(ClientModel):
    """§15.4.4: the user's own completion of a ``check:"user"``/``"measure"`` step."""

    type: Literal["step_ack"]
    step_id: StepId


class ProposalAcceptMsg(ClientModel):
    type: Literal["proposal_accept"]


class ProposalRejectMsg(ClientModel):
    type: Literal["proposal_reject"]


class StopMsg(ClientModel):
    type: Literal["stop"]


class TalkMsg(ClientModel):
    type: Literal["talk"]
    utterance: Annotated[str, Field(min_length=1, max_length=TALK_UTTERANCE_MAX)]
    #: Not in the spec's talk row; when present and not true the talk is refused (consent_required).
    consent_ai: StrictBool | None = None


class FollowNowMsg(ClientModel):
    type: Literal["follow_now"]


class ReplanNowMsg(ClientModel):
    type: Literal["replan_now"]


class ConfirmDoneMsg(ClientModel):
    type: Literal["confirm_done"]


class PrefsMsg(ClientModel):
    """Every field optional: an omitted field keeps its current value."""

    type: Literal["prefs"]
    voice_out: StrictBool | None = None
    tts: Literal["server", "browser"] | None = None
    visuals: StrictBool | None = None


class PlayedMsg(ClientModel):
    type: Literal["played"]
    line_id: Annotated[str, Field(min_length=1, max_length=32)]
    via: Literal["server", "browser", "skipped"]


ClientMsg = Annotated[
    Union[
        HelloMsg, StartMsg, StopMsg, TalkMsg, FollowNowMsg, ReplanNowMsg, ConfirmDoneMsg, PrefsMsg, PlayedMsg,
        # §15 (Plan 모드)
        PlanEditMsg, PlanApproveMsg, PlanDiscardMsg, RunPauseMsg, RunResumeMsg, StepAckMsg,
        ProposalAcceptMsg, ProposalRejectMsg, PlanAnswerMsg,
    ],
    Field(discriminator="type"),
]
CLIENT_MSG = TypeAdapter(ClientMsg)
CLIENT_TYPES = frozenset({
    "hello", "start", "stop", "talk", "follow_now", "replan_now", "confirm_done", "prefs", "played",
    "plan_edit", "plan_approve", "plan_discard", "run_pause", "run_resume", "step_ack",
    "proposal_accept", "proposal_reject", "plan_answer",
})


class Prefs(WireModel):
    voice_out: StrictBool = True
    tts: Literal["server", "browser"] = "server"
    visuals: StrictBool = True


# ---------------------------------------------------------------- REST (§3)


class BuildInfo(WireModel):
    commit: str
    time: str


class HealthResponse(WireModel):
    ready: StrictBool
    protocol: str = PROTOCOL
    access_mode: Literal["code", "local", "open"]
    access_code_required: StrictBool
    guide_ready: StrictBool
    #: §15: aggregate of ``plan_models`` — any configured profile makes a run possible.
    plan_ready: StrictBool = False
    #: §15: per-profile configured capability (keys are exactly ``PLAN_MODELS``). Not a quota statement.
    plan_models: dict[PlanModel, StrictBool]
    follow_ready: StrictBool
    tracker_ready: StrictBool
    tts_ready: StrictBool
    tts_voice: str | None
    build: BuildInfo
    engine: str


class CreateSessionRequest(ClientModel):
    access_code: Annotated[str, Field(max_length=256)] | None = None


class CreateSessionResponse(WireModel):
    session_id: str
    token: str
    expires_at: EpochMs
    live_url: str


class EndedResponse(WireModel):
    ended: StrictBool = True


class RestError(WireModel):
    """REST error body (not fixed by the spec): ``{"error": {"code", "message"}}``."""

    class Body(WireModel):
        code: str
        message: str

    error: Body

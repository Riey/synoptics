"""Anchored guide contract: plan once, follow cheaply, confirm completion with the large model.

Six routes, four tools, one rule each (design ``docs/redesign-anchored.md`` §계약; the plan-review fields are
§15, shared with the ``synoptics-live`` protocol server):

* ``POST /api/guide/plan``    — the large model, once per Start. It names the object to track (an English
  detector noun, exactly like ``/api/track/select``) and writes a short plan: 1-16 steps, each with the Korean
  sentence the screen shows (``say``), at most two anchored commands, a visible ``done_when`` condition, and
  the optional §15 review fields (``condition_kind``/``check``/``required``/``requires``/``targets``/``evidence``).
  Commands refer to the ROLE ``"target"`` because no track exists yet; the client binds the role to a real
  anchor once the tracker has acquired the object.
* ``POST /api/guide/plan/approve`` — the plan the CLIENT reviewed and edited becomes the session's authoritative
  plan. No provider call, no image, no consent: the client already holds the plan (it may have edited it), and
  the route only re-validates and re-stamps it under the next ``plan_revision``.
* ``POST /api/guide/plan/revert``  — undo the last plan change: install the session's most recent previous plan
  (bounded history) under the next ``plan_revision``, so no fence ever moves backwards.
* ``POST /api/guide/follow``  — the follower (a local small model or the large one). It writes NO
  instructions and NO goal status: it only answers a checklist — which steps it covers depends on the plan's
  core (``classic``: the current step to the last; ``sequential``: the current step alone; ``graph``: every
  step of the plan) — whether that step's ``done_when`` is visible now, and whether each drawn anchor box
  still contains the target. Which step the screen shows is the client's decision, made from that checklist.
* ``POST /api/guide/confirm`` — the large model again. The ONLY route that can answer ``visually_satisfied``,
  and (with talk) the only route that may rewrite the remaining steps (``replan``).
* ``POST /api/guide/talk``    — the large model again, on the user's own words during a guide. It answers in
  short Korean (``reply``) and takes at most ONE action: rewrite the current step's sentence, name a more
  specific target, mark the current step done/skipped, go back to an earlier step, or replace the remaining
  steps. Only the sentence rewrite and the replan change the stored plan; the rest belong to the client.

Nothing in these models carries a coordinate written by a model: positions come from the tracker
(``AnchorRef.box``), and the models refer to anchors by id only. Every model is a strict ``WireModel``
(``extra="forbid"``), so a follower answer that tries to carry a ``goal_status`` is refused as a schema
violation rather than ignored.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, PrivateAttr, StrictBool, model_validator, ModelWrapValidatorHandler

from backend.app.api_contracts import (
    HealthResponse,
    NonnegativeInt,
    ProviderName,
    SceneInput,
    Usage,
    UUIDString,
    VisualGoalStatus,
)
from backend.app.tracking_contracts import (
    ModelName,
    RunId,
    TargetSelectionToolOutput,
    TrackId,
    TrackStateKind,
)
from backend.app.visual_contracts import Box, FrameId, WireModel


#: The role a plan's commands refer to before any track exists. It is never a valid anchor id, so a role
#: and an anchor can never be confused once the client has bound one to the other.
TARGET_ROLE = "target"

#: The selectable upper planning profiles. Each names the issuer-specific adapter of one configured
#: deployment capability plus its reasoning profile: ``deepseek:high`` is the basic DeepSeek provider at HIGH
#: (thinking), ``astra:high`` is the OpenAI Responses adapter at HIGH. A choice describes configured
#: capability only, never a quota: whether a plan can actually run is reported separately (``plan_models``).
#: The client picks one before a Start and cannot change it mid-plan; confirm/talk route from the plan's own
#: stored choice.
PlanModel = Literal["deepseek:high", "astra:high"]
PLAN_MODELS: tuple[PlanModel, ...] = ("deepseek:high", "astra:high")
DEFAULT_PLAN_MODEL: PlanModel = "deepseek:high"

#: The selectable guide-core profiles: how the follower is asked and how the client may move the screen.
#:
#: * ``classic`` — the whole-plan checklist. A follow answer covers every step from the current one to the
#:   last, and the client may advance or skip from a later step's ``yes``.
#: * ``sequential`` — current step only. Every follower (Clef, local, paid) is asked about the CURRENT step
#:   alone, so a future step's verdict can never move the screen; the client advances at most one step per
#:   confirmed observation.
#: * ``graph`` — the whole plan, every frame. The follower is asked about EVERY step of the plan, earlier and
#:   already-completed ones included; the client's ``current_step`` is only its suggested focus and never the
#:   question set or a completion. A node is committed by the upper's matching ``step_done`` verdict, never by
#:   position: a step's ``state`` claim may be revoked by a fresh ``no`` (its affected descendants then recheck
#:   as ``unsure``), while an ``event`` occurrence stays historical even when a later frame no longer shows it.
#:
#: All profiles share one follower model, one tool schema, one mode and one threshold: only WHAT the
#: follower is asked about, and what the client may do with the answer, differ. The client picks one before a
#: Start and cannot change it mid-plan — confirm/talk and the client route from the plan's own stored choice.
#: There is no alias for it and no legacy field: an unknown value is refused as an unknown field.
CoreMode = Literal["classic", "sequential", "graph"]
CORE_MODES: tuple[CoreMode, ...] = ("classic", "sequential", "graph")
DEFAULT_CORE_MODE: CoreMode = "classic"
#: The non-default cores, named so the branches that implement them (``guide.StoredPlan.checklist_ids``,
#: ``intent`` follower prompts, ``clef_follow``) cannot drift. ``graph`` never narrows to the current step and
#: never truncates the remaining plan; ``sequential`` does both.
CORE_MODE_SEQUENTIAL: CoreMode = "sequential"
CORE_MODE_GRAPH: CoreMode = "graph"

#: Length bounds of the model-written plan text. Over-long text from a model is CUT to the bound before
#: validation (``_clip``), not refused: a 133-character ``goal_when`` failed a whole glass plan as 502 schema
#: (2026-10-01 E2E). No ellipsis is added. Absent or empty fields are still refused.
ANCHOR_LABEL_MAX = 40
STEP_SAY_MAX = 60
VISUAL_CONDITION_MAX = 60

#: The user-utterance channel (``/api/guide/talk``). The user's words are REFUSED over their bound (the client
#: caps its input at the same length); the model's ``reply`` is cut to its bound like plan text. The reply was
#: 100 characters until 2026-10-02, too short for real troubleshooting (a cause plus 2-3 alternative techniques).
TALK_UTTERANCE_MAX = 200
TALK_REPLY_MAX = 400
#: What the voice reads of a talk answer (operator, 2026-10-02: a 400-character reply is far too long to listen
#: to). One sentence; the screen keeps the whole ``reply``.
TALK_SPOKEN_MAX = 60

AnchorId = Annotated[str, Field(min_length=1, max_length=16)]
AnchorLabel = Annotated[str, Field(min_length=1, max_length=ANCHOR_LABEL_MAX)]
#: A plan has at most sixteen independently verifiable steps.
MAX_STEPS = 16
#: Plan step ids: ``s1`` .. ``s16``.
StepId = Annotated[str, Field(pattern=r"^s(?:[1-9]|1[0-6])$")]
StepSay = Annotated[str, Field(min_length=1, max_length=STEP_SAY_MAX)]
VisualCondition = Annotated[str, Field(min_length=1, max_length=VISUAL_CONDITION_MAX)]

#: §15 plan-review fields. ``StepTarget`` is a step's logical subject (a part, a box, a tool — a label, never a
#: coordinate); ``MaterialId`` names one of the request's ``materials`` (``m1`` .. ``m12``, in request order);
#: ``CheckMethod`` is what decides a step is finished (``visual`` the frame, ``user``/``measure`` the user says
#: so). Caps match the ``synoptics-live`` protocol server's own contract.
STEP_TARGET_MAX = 40
#: A step may list every other step of the plan as a direct prerequisite (``MAX_STEPS - 1``). A smaller cap
#: would force a real dependency graph into an artificial chain: a step that truly needs the whole board
#: assembled (an all-components inspection) has no honest way to say so with three ids.
MAX_REQUIRES = MAX_STEPS - 1
MAX_STEP_TARGETS = 3
MAX_MATERIALS = 4
MATERIAL_TITLE_MAX = 80
MATERIAL_VERSION_MAX = 32
MATERIAL_TEXT_MAX = 4000
EVIDENCE_LOCATOR_MAX = 80
EVIDENCE_QUOTE_MAX = 120

StepTarget = Annotated[str, Field(min_length=1, max_length=STEP_TARGET_MAX)]
MaterialId = Annotated[str, Field(pattern=r"^m(?:[1-9]|1[0-2])$")]
CheckMethod = Literal["visual", "user", "measure"]
#: What a step's ``done_when`` describes, and therefore how a later frame may treat it.
#:
#: * ``state`` (the default) — a condition that holds while it is visible: the latest applicable verdict
#:   decides, so a fresh ``no`` can revoke an earlier ``yes``.
#: * ``event`` — an occurrence that was verified once. A later frame that no longer shows its trace cannot
#:   undo it, so the verified occurrence stays historical.
#:
#: The distinction is metadata the CLIENT's graph state uses (revocation and recheck); it never changes what
#: the follower is asked to look at, and no verdict here is a completion by itself.
ConditionKind = Literal["state", "event"]
CONDITION_KINDS: tuple[ConditionKind, ...] = ("state", "event")
DEFAULT_CONDITION_KIND: ConditionKind = "state"


def _clip(data: object, limits: dict[str, int]) -> object:
    """A copy of a raw model object with each named string field cut to its bound (other values untouched)."""
    if not isinstance(data, dict):
        return data
    clipped = dict(data)
    for name, limit in limits.items():
        value = clipped.get(name)
        if isinstance(value, str) and len(value) > limit:
            clipped[name] = value[:limit]
    return clipped


def fold_whitespace(value: str) -> str:
    """Text with every whitespace run (line breaks too) folded to one space, for quote matching.

    A model copies an excerpt across a wrapped line or with a stray double space; folding both sides lets the
    excerpt still be recognised as coming from the same material, while a kept ``quote`` is stored folded.
    """
    return " ".join(value.split())


ClarificationPrompt = Annotated[str, Field(min_length=1, max_length=300)]
TaskEpoch = Annotated[str, Field(min_length=1, max_length=64)]
Pad = Annotated[float, Field(ge=0, le=0.5, allow_inf_nan=False)]

GuideEvidenceKind = Literal["observed_scene", "uncertain_view"]
FollowTrigger = Literal["acquired", "target_moved", "target_changed", "anchor_lost", "manual", "heartbeat"]
ConfirmTrigger = Literal["goal_check", "step_done", "unsure_twice", "replan", "target_left"]
#: Which frame edge the tracked box was nearest when the tracker lost it (``none``: not at an edge).
ExitEdge = Literal["left", "right", "top", "bottom", "none"]
Tristate = Literal["yes", "no", "unsure"]
FollowProviderName = Literal["local", "clef", "deepseek"]


class AnchorRef(WireModel):
    """The client's registry snapshot of one anchor, captured with the request frame. Not server authority.

    ``box`` is the tracker's box on the SAME capture as ``scene`` and exists iff ``state == "tracking"``
    (the tracking contract's own rule). The server draws it on its copy of the frame together with
    ``anchor_id`` (Set-of-Mark) so the model can refer to the object by id without ever writing a position.
    """

    anchor_id: AnchorId
    role: Literal["target"]
    label: AnchorLabel
    run_id: RunId
    track_id: TrackId
    generation: NonnegativeInt
    state: TrackStateKind
    box: Box | None = None

    @model_validator(mode="after")
    def box_only_while_tracking(self) -> AnchorRef:
        if self.anchor_id == TARGET_ROLE:
            raise ValueError("an anchor id cannot be the role name 'target'")
        if (self.box is not None) != (self.state == "tracking"):
            raise ValueError("box must be present if and only if state == 'tracking'")
        return self


class FocusAnchorCommand(WireModel):
    """Emphasis ring around the anchor's live box; ``pad`` is a fraction of the box, not a position."""

    kind: Literal["focus"]
    anchor: AnchorId
    pad: Pad = 0.15


class LabelAnchorCommand(WireModel):
    """A short callout attached above the anchor's live box (screen-space offset, drawn by the client)."""

    kind: Literal["label"]
    anchor: AnchorId
    text: AnchorLabel

    @model_validator(mode="before")
    @classmethod
    def clip_text(cls, data: object) -> object:
        return _clip(data, {"text": ANCHOR_LABEL_MAX})


#: Action glyph vocabulary. The last four are assembly actions added for component/wire work (circuit assembly).
#: Each is a distinct physical action, not an alias: ``align`` lines a lead or part up by orientation,
#: ``connect`` mates a wire/connector (``attach`` stays surface adhesion such as tape or a sticker),
#: ``disconnect`` separates a mated connector or power plug (``remove`` lifts a whole part out), ``bend`` shapes
#: a component lead (``fold`` folds a sheet/board). All four carry direction ``'none'``: ``ActionDirection``'s
#: side/rotate/screw sets are the only exceptions, so the validator below already requires ``'none'`` for them and
#: there is no per-action direction list to add.
#: The glyph is SCHEMATIC: it marks the whole target and is never evidence of a specific pin, polarity or
#: voltage, so nothing about it may be read as electrical detail (see ``intent.STEP_COMMANDS_RULE``).
ActionType = Literal[
    "press", "grasp", "move", "rotate", "open", "close", "insert", "remove",
    "attach", "fold", "place", "fit", "screw", "hold", "flip", "pull", "push",
    "align", "connect", "disconnect", "bend",
]
ActionDirection = Literal[
    "up", "down", "left", "right", "clockwise", "counterclockwise", "tighten", "loosen", "none",
]

MOVE_DIRECTIONS: set[ActionDirection] = {"up", "down", "left", "right"}
ROTATE_DIRECTIONS: set[ActionDirection] = {"clockwise", "counterclockwise"}
SCREW_DIRECTIONS: set[ActionDirection] = {"tighten", "loosen"}
#: Actions whose direction is a displayed-image side (move relocates the object; pull/push are a force on it).
SIDE_DIRECTION_ACTIONS: set[str] = {"move", "pull", "push"}


class ActionAnchorCommand(WireModel):
    """Action glyph attached to the anchor's live box (drawn by the client). Marks the whole target, not subpart."""

    kind: Literal["action"]
    anchor: AnchorId
    action: ActionType
    direction: ActionDirection

    @model_validator(mode="after")
    def direction_matches_action(self) -> ActionAnchorCommand:
        if self.action in SIDE_DIRECTION_ACTIONS:
            if self.direction not in MOVE_DIRECTIONS:
                raise ValueError(f"action '{self.action}' requires direction 'up', 'down', 'left', or 'right'")
        elif self.action == "rotate":
            if self.direction not in ROTATE_DIRECTIONS:
                raise ValueError("action 'rotate' requires direction 'clockwise' or 'counterclockwise'")
        elif self.action == "screw":
            if self.direction not in SCREW_DIRECTIONS:
                raise ValueError("action 'screw' requires direction 'tighten' or 'loosen'")
        else:
            if self.direction != "none":
                raise ValueError(f"action '{self.action}' requires direction 'none'")
        return self


AnchoredCommand = FocusAnchorCommand | LabelAnchorCommand | ActionAnchorCommand


class Evidence(WireModel):
    """§15.2: which material (and where in it) a step's instruction came from.

    ``material_id`` names one of the request's ``materials`` (``m1`` .. in request order). The route knows that
    list and DROPS an evidence it cannot ground — an id not among the materials, a ``version`` that does not
    equal the material's own, a ``quote`` longer than ``EVIDENCE_QUOTE_MAX`` or not a whitespace-normalised
    excerpt of the material's text, or a field outside its bound (the drop is logged; nothing is invented) — so
    no fabricated or malformed citation ever reaches a client. A quote is never silently cut to fit: an
    over-long quote is ungrounded and dropped, and a kept quote is stored whitespace-folded. A blank optional
    string is absent.
    """

    material_id: MaterialId
    version: Annotated[str, Field(min_length=1, max_length=MATERIAL_VERSION_MAX)] | None = None
    locator: Annotated[str, Field(min_length=1, max_length=EVIDENCE_LOCATOR_MAX)] | None = None
    quote: Annotated[str, Field(min_length=1, max_length=EVIDENCE_QUOTE_MAX)] | None = None

    @model_validator(mode="before")
    @classmethod
    def fold_and_blank(cls, data: object) -> object:
        """Fold whitespace in the three optional texts; a blank one is absent. Over-long text is left for the
        drop pass (model answers) or the field bound (client bodies) to refuse — never cut silently."""
        if not isinstance(data, dict):
            return data
        folded = dict(data)
        for name in ("version", "locator", "quote"):
            value = folded.get(name)
            if isinstance(value, str):
                folded[name] = fold_whitespace(value) or None
        return folded


class MaterialInput(WireModel):
    """§15.3: one reference document a plan may cite.

    ``text`` is the extracted plain text; it is rendered into a prompt as a labelled block (``m1``, ``m2``, ...
    in request order) that a step's ``evidence.material_id`` can point at. A plan keeps the materials it was
    made with for the life of its task (so a confirm/talk replan stays grounded in the same source) but they are
    never echoed by ``/plan/current`` and never written to a log.
    """

    title: Annotated[str, Field(min_length=1, max_length=MATERIAL_TITLE_MAX)]
    version: Annotated[str, Field(min_length=1, max_length=MATERIAL_VERSION_MAX)] | None = None
    text: Annotated[str, Field(min_length=1, max_length=MATERIAL_TEXT_MAX)]


class PlanAnswer(WireModel):
    """One answered clarification, retained as user evidence without replacing the goal."""

    question: Annotated[str, Field(min_length=1, max_length=300)]
    answer: Annotated[str, Field(min_length=1, max_length=1000)]


class ResearchSource(WireModel):
    """A public source actually returned by a research tool, not an invented citation."""

    url: Annotated[str, Field(min_length=1, max_length=2048)]
    title: Annotated[str, Field(min_length=1, max_length=160)]
    summary: Annotated[str, Field(min_length=1, max_length=1200)]

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


class GuideStep(WireModel):
    """One plan step: what the screen says, what to draw on the target, and the visible end condition.

    Commands refer to the role ``"target"`` only. The step is plain text plus that role reference, which is
    why the server can keep a plan in the session without keeping an image or a coordinate.

    A plan (or a confirm replan) is written before any anchor id exists, so ``"target"`` is the only thing a
    step command can refer to: whatever ``anchor`` value the model wrote (DeepSeek wrote ``"a1"`` on 3/3
    increase_temp plans, 2026-10-01) is rewritten to the role BEFORE validation instead of failing the whole
    plan. Nothing else in the command is touched, and a command without an ``anchor`` is still refused. The
    follow/confirm request anchors (``AnchorRef``) are a different model and are validated as before.

    The optional §15 review fields are defaulted, so a caller that does not use them is unchanged:

    * ``condition_kind`` is what ``done_when`` describes — ``state`` (default) while it holds, ``event`` once
      verified. Metadata for the client's graph state; it changes nothing about what the follower looks at.
    * ``check`` is what decides the step is finished (``visual`` on the frame; ``user``/``measure`` only when
      the user says so — a check the camera cannot make).
    * ``required`` marks a mandatory inspection point that must never be skipped.
    * ``goal_required`` says whose condition this is: the USER'S GOAL (the default) or only safety,
      preparation, teardown or optional support around it. It is orthogonal to ``required``: a safety
      precondition such as unplugging the power before touching the wiring is ``goal_required=False`` WITH
      ``required=True`` (the user must still confirm it, but its state is not evidence the goal was reached),
      an optional supporting step is ``goal_required=False, required=False``, and only ``goal_required=True``
      steps are goal evidence — so the client never counts a procedure step toward goal completion and never
      revokes a completed goal when a precondition later stops holding.
    * ``requires`` names steps of the SAME plan that must be finished before this one is reachable; the plan
      graph (unique ids, resolvable ``requires``, no self-reference, no cycle) is checked plan-wide.
    * ``targets`` are the step's logical subjects (parts, boxes, tools) — labels, never coordinates.
    * ``evidence`` cites the material the instruction came from, when the request supplied materials.
    """

    id: StepId
    say: StepSay
    details: Annotated[str, Field(min_length=1, max_length=600)] | None = None
    commands: Annotated[list[AnchoredCommand], Field(max_length=2)]
    done_when: VisualCondition
    condition_kind: ConditionKind = DEFAULT_CONDITION_KIND
    check: CheckMethod = "visual"
    required: bool = False
    goal_required: bool = True
    requires: Annotated[list[StepId], Field(max_length=MAX_REQUIRES)] = Field(default_factory=list)
    targets: Annotated[list[StepTarget], Field(max_length=MAX_STEP_TARGETS)] = Field(default_factory=list)
    evidence: Evidence | None = None

    @model_validator(mode="before")
    @classmethod
    def normalise_raw_step(cls, data: object) -> object:
        data = _clip(data, {"say": STEP_SAY_MAX, "done_when": VISUAL_CONDITION_MAX})
        if not isinstance(data, dict):
            return data
        # ``condition_kind`` is a closed enum: fold the case a model wrote ("State"/"EVENT") rather than failing
        # a whole plan on it. An absent or unreadable value still defaults to ``state`` at the field.
        if isinstance(data.get("condition_kind"), str):
            data = {**data, "condition_kind": data["condition_kind"].strip().lower()}
        commands = data.get("commands")
        if isinstance(commands, list):
            data = {**data, "commands": [
                {**command, "anchor": TARGET_ROLE} if isinstance(command, dict) and "anchor" in command else command
                for command in commands
            ]}
        targets = data.get("targets")
        if isinstance(targets, list):
            data = {**data, "targets": [
                target[:STEP_TARGET_MAX] if isinstance(target, str) and len(target) > STEP_TARGET_MAX else target
                for target in targets
            ]}
        evidence = data.get("evidence")
        if isinstance(evidence, dict) and not str(evidence.get("material_id") or "").strip():
            # A citation with no material id is not a citation (the model was given no material to name).
            data = {key: value for key, value in data.items() if key != "evidence"}
        return data

    @model_validator(mode="after")
    def role_commands_only(self) -> GuideStep:
        for command in self.commands:
            if command.anchor != TARGET_ROLE:
                raise ValueError("plan commands must refer to the role 'target', never to an anchor id")
        actions = [command for command in self.commands if command.kind == "action"]
        if len(actions) > 1:
            raise ValueError("at most one action command per step")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError("duplicate step ids in requires")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate logical targets")
        return self


#: A plan and a replan share the same bound.
PlanSteps = Annotated[list[GuideStep], Field(max_length=MAX_STEPS)]


def _check_step_graph(steps: list[GuideStep]) -> None:
    """§15 plan-level graph rules: unique ids, resolvable ``requires``, no self-reference, no cycle.

    Raises ``ValueError`` on the first violation, which fails closed at the caller's own status: a model answer
    is ``502 invalid_provider_output`` and a client body is ``422 invalid_request``.
    """
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


def _steps_at(answer: object, path: tuple[str, ...]) -> object:
    """The value ``path`` names inside a raw model answer, or ``None`` when any link is missing."""
    node = answer
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _with_steps(answer: dict, path: tuple[str, ...], steps: object) -> dict:
    """``answer`` with ``steps`` placed at ``path`` (a copy: the provider's own dict is never mutated)."""
    if len(path) == 1:
        return {**answer, path[0]: steps}
    child = answer.get(path[0])
    if not isinstance(child, dict):
        return answer
    return {**answer, path[0]: _with_steps(child, path[1:], steps)}


def _material_index(materials: list[MaterialInput] | None) -> dict[str, MaterialInput]:
    """The request's materials by their prompt label (``m1`` .. in request order)."""
    return {f"m{index}": material for index, material in enumerate(materials or [], 1)}


def evidence_is_grounded(evidence: object, index: dict[str, MaterialInput]) -> bool:
    """Whether a RAW evidence object cites a supplied material and matches that material's own text.

    Grounded means: ``material_id`` names a supplied material; a non-blank ``version`` equals the material's
    ``version`` (a material with no version cannot be cited with one); a non-blank ``quote`` is at most
    ``EVIDENCE_QUOTE_MAX`` characters and, whitespace-normalised, is a substring of the material's text; and a
    non-blank ``locator`` fits its bound. Nothing else (the shape of the object, the exact keys) is checked here;
    model validation still fails closed on a malformed citation.
    """
    if not isinstance(evidence, dict):
        return False
    material_id = evidence.get("material_id")
    if not isinstance(material_id, str) or not material_id.strip() or material_id not in index:
        return False
    material = index[material_id]
    version = evidence.get("version")
    if isinstance(version, str) and version.strip():
        if material.version is None or fold_whitespace(version) != fold_whitespace(material.version):
            return False
    locator = evidence.get("locator")
    if isinstance(locator, str) and len(fold_whitespace(locator)) > EVIDENCE_LOCATOR_MAX:
        return False
    quote = evidence.get("quote")
    if isinstance(quote, str) and quote.strip():
        folded = fold_whitespace(quote)
        if len(folded) > EVIDENCE_QUOTE_MAX or folded not in fold_whitespace(material.text):
            return False
    return True


def drop_ungrounded_evidence(answer: object, materials: list[MaterialInput] | None, *,
                             path: tuple[str, ...] = ("steps",)) -> tuple[object, list[str]]:
    """A copy of a RAW model answer whose steps lose every ``evidence`` the request cannot ground.

    Runs on the answer's own dict, BEFORE model validation, so a citation can be dropped instead of failing the
    whole answer: an ``evidence`` is removed as a whole unless ``evidence_is_grounded`` accepts it — its
    ``material_id`` is not a supplied one, its ``version`` does not match the material's, its ``quote`` is not a
    whitespace-normalised excerpt of the material (including a quote longer than ``EVIDENCE_QUOTE_MAX``, which
    is never silently cut to fit), its ``locator`` is over-long, or it is not an object at all. A ``None``
    evidence is left alone (it is already absent).

    ``path`` names the step list inside ``answer`` (a plan's ``("steps",)``, a confirm/talk replan's
    ``("replan", "steps")``). Returns ``(answer, dropped)`` with the offending ids in step order — the caller
    logs it. A material is never invented and an ungrounded citation never reaches a client; with no materials
    in the request ``materials`` is empty, so every citation is dropped.
    """
    steps = _steps_at(answer, path)
    if not isinstance(answer, dict) or not isinstance(steps, list):
        return answer, []
    index = _material_index(materials)
    dropped: list[str] = []
    filtered: list[object] = []
    for step in steps:
        if not isinstance(step, dict) or "evidence" not in step:
            filtered.append(step)
            continue
        evidence = step["evidence"]
        if evidence is None or evidence_is_grounded(evidence, index):
            filtered.append(step)
            continue
        material_id = evidence.get("material_id") if isinstance(evidence, dict) else None
        dropped.append(material_id if isinstance(material_id, str) else "")
        filtered.append({key: value for key, value in step.items() if key != "evidence"})
    if not dropped:
        return answer, []
    return _with_steps(answer, path, filtered), dropped


def check_grounded_evidence(steps: list[GuideStep], materials: list[MaterialInput] | None) -> None:
    """Client-supplied steps must carry only evidence the plan's materials ground, else ``ValueError``.

    The approve route turns this into ``422 invalid_request``: a client may not inject a citation the server
    cannot ground. Evidence that IS grounded is kept exactly as sent (its validated metadata is preserved).
    """
    index = _material_index(materials)
    for step in steps:
        if step.evidence is not None and not evidence_is_grounded(step.evidence.model_dump(), index):
            raise ValueError(f"step {step.id} carries evidence the plan's materials do not ground")


def _check_clarification(evidence_kind: str, needs_clarification: bool, prompt: str | None) -> None:
    """``uncertain_view`` => ``needs_clarification`` <=> a non-blank clarification prompt."""
    if evidence_kind == "uncertain_view" and not needs_clarification:
        raise ValueError("uncertain_view evidence requires needs_clarification=True")
    has_prompt = prompt is not None and prompt.strip() != ""
    if prompt is not None and not has_prompt:
        raise ValueError("a blank clarification prompt is not a prompt")
    if needs_clarification != has_prompt:
        raise ValueError("clarification_prompt is present if and only if needs_clarification")


#: The plan fields written after ``goal_when`` (the only ones a leaked tail may carry).
_PLAN_TAIL_FIELDS = frozenset({"evidence_kind", "needs_clarification", "clarification_prompt"})


def _recover_leaked_tail(data: dict) -> dict:
    """A copy of a raw plan whose ``goal_when`` swallowed the rest of the object, split back; else ``data``."""
    value = data.get("goal_when")
    if not isinstance(value, str) or '"' not in value:
        return data
    try:
        parsed = json.loads('{"goal_when": "' + value)
    except ValueError:
        return data
    if not isinstance(parsed, dict) or not isinstance(parsed.get("goal_when"), str):
        return data
    extra = set(parsed) - {"goal_when"}
    if not extra or not extra <= _PLAN_TAIL_FIELDS:
        return data
    repaired = {**data, "goal_when": parsed["goal_when"]}
    for name in extra:
        repaired.setdefault(name, parsed[name])
    return repaired


class GuidePlanToolOutput(WireModel):
    """The large model's one plan for this task, from the frame captured at Start.

    Relational rules (fail closed):

    * ``selection`` is the ``/api/track/select`` answer shape: a target exists iff ``status == "selected"``.
    * A non-selection (``no_target``/``uncertain``) has no steps: there is nothing to anchor them to.
    * ``uncertain_view`` requires ``needs_clarification``; while clarification is needed no step carries a
      command (steps may then be empty).
    * Otherwise a selected target has 1-``MAX_STEPS`` steps with unique ids and a valid ``requires`` graph.
    """

    selection: TargetSelectionToolOutput
    steps: PlanSteps
    goal_when: VisualCondition
    evidence_kind: GuideEvidenceKind
    needs_clarification: StrictBool
    clarification_prompt: ClarificationPrompt | None = None
    research_sources: Annotated[list[ResearchSource], Field(max_length=6)] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def repair_trailing_fields(cls, data: object) -> object:
        """Recover, default and clip the fields a model wrote last (glass E2E, 2026-10-01).

        1. DeepSeek once wrote the trailing fields INSIDE ``goal_when`` as escaped JSON
           (``'…있음", "evidence_kind": "observed_scene", "needs_clarification": false, …}'``). When
           ``goal_when`` parses as the continuation of the object and every extra key is a trailing plan field,
           ``goal_when`` keeps only its own text and each extra key fills the field only if it is absent.
        2. A plan that has a selection and at least one step observed the scene: a still missing
           ``evidence_kind`` is ``observed_scene`` and a missing ``needs_clarification`` is ``False``. Without
           steps or without a selection both stay required (502 schema).
        3. ``goal_when`` is cut to its bound.
        """
        if not isinstance(data, dict):
            return data
        data = _recover_leaked_tail(data)
        if isinstance(data.get("selection"), dict) and isinstance(data.get("steps"), list) and data["steps"]:
            data.setdefault("evidence_kind", "observed_scene")
            data.setdefault("needs_clarification", False)
        return _clip(data, {"goal_when": VISUAL_CONDITION_MAX})

    @model_validator(mode="after")
    def consistent_plan(self) -> GuidePlanToolOutput:
        _check_step_graph(self.steps)
        _check_clarification(self.evidence_kind, self.needs_clarification, self.clarification_prompt)
        if self.selection.status != "selected" and self.steps:
            raise ValueError("a non-selection carries no steps")
        if self.needs_clarification:
            if any(step.commands for step in self.steps):
                raise ValueError("no commands while clarification is needed")
        elif self.selection.status == "selected" and not self.steps:
            raise ValueError("a selected target without clarification needs 1-16 steps")
        return self


class GuidePlanRequest(WireModel):
    """Start a guide: the task text and the frame captured right now. A new plan replaces the old one.

    ``materials`` are reference documents: each is rendered into a prompt as a labelled block (``m1`` .. in
    request order) that a step's ``evidence.material_id`` may cite. They join the session's task identity (a
    changed list under the same goal/context is a new task) and are kept with the plan for its task, so a
    confirm/talk replan stays grounded in the same source; they are never echoed by ``/plan/current`` and never
    logged.

    ``plan_model`` (one of ``deepseek:high`` / ``astra:high``, default ``deepseek:high``) selects which upper
    planning profile writes the plan: the basic DeepSeek provider at HIGH, or the OpenAI Responses adapter at
    HIGH. The two profiles share one intent contract, one reasoning+output cap and one deadline; only the
    issuer adapter differs, and a missing credential is a clean refusal — never a fallback to the other lane.
    The choice joins the session's task identity too — changing it under the same goal/context is a new task,
    so a plan made for one choice can never be inherited or revised by the other. There is no alias for it and
    no legacy ``plan_mode``/``mode``/``manual`` field (any of those is an unknown field and refused). Once a
    plan is accepted, confirm/talk route from that plan's own stored ``plan_model``, never from the client.

    ``core_mode`` (``classic`` by default, ``sequential`` or ``graph``) selects the guide core: how the follower
    is asked and how the client may move the screen. ``classic`` is the whole-plan checklist (a follow answer
    covers every step from the current one to the last). ``sequential`` scopes every follower answer to the
    CURRENT step, so a later step's verdict can never move the screen. ``graph`` asks about EVERY step of the
    plan, earlier and already-completed ones included, and treats the client's suggested focus as a hint only: a
    step is committed by the upper's own ``step_done`` verdict, and an explicit ``condition_kind`` says whether
    a step's claim can be revoked (``state``) or is a historical occurrence (``event``). All cores share one
    follower model, mode and threshold; only the question set differs. The choice joins the session's task
    identity too — changing it under the same goal/context is a new task, so a plan made for one core can never
    be inherited, revised or reverted back into by the other. There is no alias for it. It is independent of
    ``plan_model`` (which upper profile writes the plan) and of any review flag the client applies.
    """

    session_id: UUIDString
    consent_ai: StrictBool
    scene: SceneInput
    user_goal: Annotated[str, Field(min_length=1, max_length=300)]
    context: Annotated[str, Field(max_length=1000)] | None = None
    materials: Annotated[list[MaterialInput], Field(max_length=MAX_MATERIALS)] | None = None
    plan_model: PlanModel = DEFAULT_PLAN_MODEL
    core_mode: CoreMode = DEFAULT_CORE_MODE
    assisted: StrictBool = False
    research: StrictBool = False
    answers: Annotated[list[PlanAnswer], Field(max_length=12)] = Field(default_factory=list)
    reference_images: Annotated[list[SceneInput], Field(max_length=12)] = Field(default_factory=list)
    _research_sources: list[ResearchSource] = PrivateAttr(default_factory=list)


class GuidePlanResponse(GuidePlanToolOutput):
    """The plan plus its server identity. ``plan_revision`` is per session and never goes backwards.

    ``core_mode`` echoes the guide core the plan was built for; it is fixed for the plan's whole task, so a
    client never has to guess which follow semantics apply.
    """

    plan_id: UUIDString
    plan_revision: NonnegativeInt
    core_mode: CoreMode = DEFAULT_CORE_MODE
    frame_id: FrameId
    provider: ProviderName
    model: ModelName | None = None
    usage: Usage | None = None


class GuidePlanApproveRequest(WireModel):
    """``POST /api/guide/plan/approve``: the client-reviewed plan becomes the session's authoritative plan.

    The client holds a plan it may have edited (changed a sentence or the order, added or removed a step);
    approving it pins execution to exactly those steps, so a later fence names the adopted pair, not the plan
    the model wrote. ``plan_id``/``plan_revision`` must name the session's CURRENT plan, else ``409 stale_plan``
    with the usual body. No provider is called, no frame is read and no consent is asked: nothing here reaches a
    model.

    ``steps`` (1-``MAX_STEPS``) are re-validated exactly like a model answer's: the §15 requires graph (unique
    ids, resolvable and acyclic ``requires``) is fail-closed as ``422 invalid_request``. ``user_goal``/``context``
    keep the adopted plan's own values when absent, so a client that only edits steps need not resend them.
    """

    session_id: UUIDString
    plan_id: Annotated[str, Field(min_length=1, max_length=64)]
    plan_revision: NonnegativeInt
    steps: Annotated[list[GuideStep], Field(min_length=1, max_length=MAX_STEPS)]
    goal_when: VisualCondition
    user_goal: Annotated[str, Field(min_length=1, max_length=300)] | None = None
    context: Annotated[str, Field(max_length=1000)] | None = None

    @model_validator(mode="after")
    def valid_graph(self) -> GuidePlanApproveRequest:
        _check_step_graph(self.steps)
        return self


class GuidePlanRevertRequest(WireModel):
    """``POST /api/guide/plan/revert``: undo the last plan change (an approve, a replan or a new plan).

    ``plan_id``/``plan_revision`` must name the session's CURRENT plan, else ``409 stale_plan``. On success the
    session's most recent previous plan (bounded history; see ``guide.GuideSessionState``) becomes current again
    under the NEXT ``plan_revision``, so every fence keeps moving forward. An empty history is ``404 no_plan``.
    """

    session_id: UUIDString
    plan_id: Annotated[str, Field(min_length=1, max_length=64)]
    plan_revision: NonnegativeInt


class GuideFence(WireModel):
    """The client's own fence for one intent call, echoed back verbatim so a late answer is recognisable."""

    task_epoch: TaskEpoch
    run_id: RunId


class AnchorEcho(WireModel):
    """Which anchor identity an answer was computed against (a generation change makes it text-only)."""

    anchor_id: AnchorId
    track_id: TrackId
    generation: NonnegativeInt


def _check_anchors(anchors: list[AnchorRef], fence: GuideFence) -> None:
    ids = [anchor.anchor_id for anchor in anchors]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate anchor ids")
    if any(anchor.run_id != fence.run_id for anchor in anchors):
        raise ValueError("every anchor must belong to the fenced run")


class GuideFollowRequest(WireModel):
    """One follow judgement about the frame captured right now, against the current plan revision."""

    session_id: UUIDString
    consent_ai: StrictBool
    scene: SceneInput
    plan_id: UUIDString
    plan_revision: NonnegativeInt
    #: The step the client's screen shows now. It scopes the checklist for ``classic`` (this step to the last)
    #: and ``sequential`` (this step alone); under ``graph`` it is only the suggested focus — the checklist is
    #: the WHOLE plan, earlier and completed steps included — and never a completion.
    current_step: StepId
    anchors: Annotated[list[AnchorRef], Field(max_length=1)]
    trigger: FollowTrigger
    intent_seq: NonnegativeInt
    fence: GuideFence

    @model_validator(mode="after")
    def consistent_anchors(self) -> GuideFollowRequest:
        _check_anchors(self.anchors, self.fence)
        return self


class AnchorVerdict(WireModel):
    """Does the box drawn with this id still contain the plan's target object?"""

    anchor_id: AnchorId
    matches: Tristate


class StepCheck(WireModel):
    """Is this step's ``done_when`` (a visible result state) visible in the frame right now?

    A state observation only: it says nothing about order, about the action, or about which step is current.
    ``unsure`` when the thing ``done_when`` talks about is out of view, covered or blurred: absence is
    neither ``yes`` nor ``no``.
    """

    step_id: StepId
    visible: Tristate


StepChecks = Annotated[list[StepCheck], Field(min_length=1, max_length=MAX_STEPS)]


def _check_unique_steps(checks: list[StepCheck]) -> None:
    ids = [check.step_id for check in checks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate step ids in step checks")


class GuideFollowToolOutput(WireModel):
    """The follower's whole answer. No instruction text, no goal status, no coordinates, no free text.

    ``step_checks`` has one entry per step from the request's ``current_step`` to the plan's last step, in plan
    order (the WHOLE plan under ``graph``, earlier steps included, whatever the request's focus); the route
    refuses any other id list as a schema violation (the contract alone cannot know the plan).
    A ``yes`` and ``goal_seen == "yes"`` are TRIGGERS for the client's step decision and for a confirm call,
    never completion. There is deliberately no ``note``: in the 1a/1e spikes it was where answers got truncated
    (DeepSeek: every capped follow was cut inside it) and where the local model broke its length bound.
    """

    step_checks: StepChecks
    anchor_verdicts: Annotated[list[AnchorVerdict], Field(max_length=1)]
    goal_seen: Tristate

    @model_validator(mode="after")
    def unique_ids(self) -> GuideFollowToolOutput:
        _check_unique_steps(self.step_checks)
        ids = [verdict.anchor_id for verdict in self.anchor_verdicts]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate anchor verdicts")
        return self


class GuideFollowResponse(GuideFollowToolOutput):
    """The follower's answer plus everything the client needs to accept it as bound, text-only or stale.

    ``needs_reselect`` is ``True`` exactly when some verdict is ``"no"``. This response has no
    ``goal_status`` field at all: completion is never decided on this route.
    """

    intent_id: UUIDString
    frame_id: FrameId
    intent_seq: NonnegativeInt
    trigger: FollowTrigger
    task_revision: NonnegativeInt
    plan_revision: NonnegativeInt
    fence_echo: GuideFence
    anchors_echo: Annotated[list[AnchorEcho], Field(max_length=1)]
    provider: ProviderName
    model: ModelName | None = None
    usage: Usage | None = None
    latency_ms: NonnegativeInt
    needs_reselect: StrictBool

    @model_validator(mode="after")
    def reselect_iff_rejected(self) -> GuideFollowResponse:
        if self.needs_reselect != any(v.matches == "no" for v in self.anchor_verdicts):
            raise ValueError("needs_reselect must be true exactly when some verdict is 'no'")
        return self


class GuideReplan(WireModel):
    """Replacement steps for the current plan (same step shape, role-anchored commands only)."""

    steps: Annotated[list[GuideStep], Field(min_length=1, max_length=MAX_STEPS)]

    @model_validator(mode="after")
    def unique_steps(self) -> GuideReplan:
        _check_step_graph(self.steps)
        return self


class GuideConfirmRequest(WireModel):
    """Ask the large model whether the goal is visibly done (or the plan must change) on a fresh frame."""

    session_id: UUIDString
    consent_ai: StrictBool
    scene: SceneInput
    plan_id: UUIDString
    plan_revision: NonnegativeInt
    user_goal: Annotated[str, Field(min_length=1, max_length=300)]
    anchors: Annotated[list[AnchorRef], Field(max_length=1)]
    trigger: ConfirmTrigger
    intent_seq: NonnegativeInt
    fence: GuideFence
    #: The step to verify on ``step_done`` (required): under ``graph`` any step of the plan may be named —
    #: an earlier completed one or a future eligible one — and the answer binds to exactly that id. For
    #: ``replan``/``unsure_twice``, the step currently shown: required by the route under sequential; classic
    #: narrows its checklist to it when present, and graph takes the whole plan either way.
    #: A supplied id must belong to the plan, else ``422 unknown_step``.
    current_step: StepId | None = None
    #: The ``step_checks`` of the client's last accepted follow, sent only with a replan-capable trigger
    #: (``replan``/``unsure_twice``) so the judge sees where the follower got stuck. Context for the prompt,
    #: never evidence: the judge decides from the frame. Every id must be a step of the current plan, or
    #: ``422 unknown_step``.
    follow_checks: Annotated[list[StepCheck], Field(min_length=1, max_length=MAX_STEPS)] | None = None
    #: ``target_left`` only (REQUIRED there): the first frame of the tracking run (the task's starting state), so the judge
    #: can infer completion from before/after when the target itself has left the frame.
    before_scene: SceneInput | None = None
    #: ``target_left`` only: the frame edge the target's last tracked box was at.
    exit_edge: ExitEdge | None = None

    @model_validator(mode="after")
    def consistent_anchors(self) -> GuideConfirmRequest:
        _check_anchors(self.anchors, self.fence)
        if self.trigger == "step_done" and self.current_step is None:
            raise ValueError("a step_done confirm names the step it checks (current_step)")
        if (self.before_scene is None) == (self.trigger == "target_left"):
            raise ValueError("before_scene accompanies exactly the target_left confirm")
        if self.exit_edge is not None and self.trigger != "target_left":
            raise ValueError("exit_edge only accompanies a target_left confirm")
        if self.follow_checks is not None:
            if self.trigger not in {"replan", "unsure_twice"}:  # intent.REPLAN_TRIGGERS
                raise ValueError("follow_checks only accompany a replan or unsure_twice confirm")
            _check_unique_steps(self.follow_checks)
        return self


class GuideConfirmToolOutput(WireModel):
    """The completion judge's answer: the only place ``visually_satisfied`` can come from.

    Same relational rules as the guidance lane (``validate_guidance_context``): a goal is always present
    here, so ``not_applicable`` is refused; ``visually_satisfied`` is refused under ``uncertain_view`` or
    ``needs_clarification``. A satisfied goal has nothing to replan, and an unclear view is no basis for one.
    """

    goal_status: VisualGoalStatus
    evidence_kind: GuideEvidenceKind
    replan: GuideReplan | None = None
    needs_clarification: StrictBool
    clarification_prompt: ClarificationPrompt | None = None
    #: Only on a ``step_done`` confirm (where the route requires it): is the named step's ``done_when``
    #: visibly satisfied in THIS frame. A step judgement, never goal completion; ``goal_status`` is unchanged.
    #: Any other trigger's tool has no such field, and a value there is refused by the route.
    step_check: Tristate | None = None
    #: Only on a ``target_left`` confirm (where the route requires it): the target has left the frame, and the
    #: before/after frames show the goal reached anyway ('yes'), not reached ('no'), or cannot tell ('unsure').
    #: An inference, separate from ``goal_status`` (which still judges only what this frame shows).
    inferred_done: Tristate | None = None
    #: The replan/unsure_twice checklist against the request's plan, before any replacement: current-only
    #: under sequential, current-to-last under classic (whole plan if current_step is absent), and the WHOLE
    #: plan under graph (a suggested focus never narrows it).
    #: Other id lists and checklists on other triggers are refused by the route.
    step_checks: StepChecks | None = None

    @model_validator(mode="after")
    def consistent_confirm(self) -> GuideConfirmToolOutput:
        _check_clarification(self.evidence_kind, self.needs_clarification, self.clarification_prompt)
        if self.step_checks is not None:
            _check_unique_steps(self.step_checks)
        status = self.goal_status.status
        if status == "not_applicable":
            raise ValueError("a guide always has a goal, so goal_status cannot be not_applicable")
        if status == "visually_satisfied":
            if self.needs_clarification:
                raise ValueError("cannot claim goal is visually satisfied when scene needs clarification")
            if self.evidence_kind == "uncertain_view":
                raise ValueError("cannot claim goal is visually satisfied when evidence is uncertain_view")
            if self.replan is not None:
                raise ValueError("a satisfied goal has nothing to replan")
        if self.needs_clarification and self.replan is not None:
            raise ValueError("no replan from a view that needs clarification")
        return self


class GuideConfirmResponse(GuideConfirmToolOutput):
    """The judge's answer plus the same acceptance envelope as follow.

    ``plan_revision`` is the revision AFTER this call: bumped when ``replan`` was applied to the stored plan.
    ``step_id`` echoes the request's ``current_step`` on a ``step_done`` confirm, the one trigger that
    carries ``step_check``; both are present together or not at all. Under ``graph`` that is the step the
    upper was asked about — an earlier completed step or a future eligible one — and never a different
    "suggested focus": the id is the request's, echoed verbatim.
    """

    step_id: StepId | None = None
    intent_id: UUIDString
    frame_id: FrameId
    intent_seq: NonnegativeInt
    trigger: ConfirmTrigger
    task_revision: NonnegativeInt
    plan_revision: NonnegativeInt
    fence_echo: GuideFence
    anchors_echo: Annotated[list[AnchorEcho], Field(max_length=1)]
    provider: ProviderName
    model: ModelName | None = None
    usage: Usage | None = None
    latency_ms: NonnegativeInt

    @model_validator(mode="after")
    def step_check_names_its_step(self) -> GuideConfirmResponse:
        if (self.step_check is None) != (self.step_id is None):
            raise ValueError("step_check and step_id are present together or not at all")
        return self


TalkUtterance = Annotated[str, Field(min_length=1, max_length=TALK_UTTERANCE_MAX)]
TalkReply = Annotated[str, Field(min_length=1, max_length=TALK_REPLY_MAX)]
TalkSpoken = Annotated[str, Field(min_length=1, max_length=TALK_SPOKEN_MAX)]


def _first_sentence(text: str) -> str:
    """The reply's first sentence (or line), cut to the spoken bound: the voice line when the model wrote none."""
    for piece in re.split(r"(?<=[.!?。])\s+|\n+", text):
        if piece.strip():
            return piece.strip()[:TALK_SPOKEN_MAX]
    return ""
#: A detector noun the talk route may switch the tracker to. The plan's ``selection.target`` rule (an ENGLISH
#: lowercase common noun phrase of at most 40 characters) is CHECKED here, because this answer reaches the
#: tracker with no selection call in between. Case is folded before validation; Korean, punctuation other
#: than an apostrophe or hyphen, or a leading digit is refused (``502 invalid_provider_output`` schema).
TalkTarget = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9 '-]*$")]
StepMark = Literal["done", "skipped"]
TalkAction = Literal["none", "step_say", "target", "step_mark", "go_to", "replan"]

#: The talk answer's action fields, in the order ``GuideTalkToolOutput.action`` reports them.
TALK_ACTION_FIELDS: tuple[str, ...] = ("step_say", "target", "step_mark", "go_to", "replan")
#: The order ``GuideTalkToolOutput.actions`` reports the kept actions in: the user's own confirmation first, then
#: moving back, rewriting the plan, retargeting, rewording.
TALK_ACTION_PRECEDENCE: tuple[str, ...] = ("step_mark", "go_to", "replan", "target", "step_say")
#: The actions that move the guide's position or replace its steps: at most one survives (in precedence order).
#: ``target`` and ``step_say`` may come with it (operator, 2026-10-02: advice must reach both the step sentence and
#: the overlay — "장갑 끼고 왔어" rewrites the step for the gloved hand AND moves the overlay onto it), except that a
#: ``replan`` drops a ``step_say`` (it rewrites the sentences itself). Nothing here is a 502 any more.
TALK_STATE_ACTIONS: tuple[str, ...] = ("step_mark", "go_to", "replan")


def _one_line(value: object) -> object:
    """User text with outer whitespace removed and every inner whitespace run (line breaks too) folded to one
    space, so the user's words always stay on the one prompt line they are quoted on."""
    return " ".join(value.split()) if isinstance(value, str) else value


class GuideTalkRequest(WireModel):
    """What the user just said during a guide, with the frame captured right now.

    ``prev_utterance``/``prev_reply`` are the previous exchange (both or neither): the server keeps no
    conversation. ``follow_checks`` is the client's last accepted follow checklist, prompt context only; every
    id must be a step of the current plan, or ``422 unknown_step`` (checked by the route, like confirm).
    ``replan_allowed`` is ``False`` once the client's run has spent its replans: the tool is then offered without
    ``replan`` and a replan in the answer is a schema violation.
    """

    session_id: UUIDString
    consent_ai: StrictBool
    scene: SceneInput
    plan_id: UUIDString
    plan_revision: NonnegativeInt
    current_step: StepId
    anchors: Annotated[list[AnchorRef], Field(max_length=1)]
    utterance: TalkUtterance
    prev_utterance: TalkUtterance | None = None
    prev_reply: TalkReply | None = None
    follow_checks: Annotated[list[StepCheck], Field(min_length=1, max_length=MAX_STEPS)] | None = None
    replan_allowed: StrictBool = True
    intent_seq: NonnegativeInt
    fence: GuideFence

    @model_validator(mode="before")
    @classmethod
    def fold_whitespace(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        names = ("utterance", "prev_utterance", "prev_reply")
        return {**data, **{name: _one_line(data[name]) for name in names if name in data}}

    @model_validator(mode="after")
    def consistent_talk(self) -> GuideTalkRequest:
        _check_anchors(self.anchors, self.fence)
        if (self.prev_utterance is None) != (self.prev_reply is None):
            raise ValueError("prev_utterance and prev_reply are one exchange: both or neither")
        if self.follow_checks is not None:
            _check_unique_steps(self.follow_checks)
        return self


class GuideTalkToolOutput(WireModel):
    """The guide's answer to the user's words: a short Korean ``reply`` and at most ONE action.

    * ``step_say`` — the current step's sentence, rewritten (the plan ``say`` bound).
    * ``target`` — a more specific detector noun; the client restarts tracking with it (no paid call).
    * ``step_mark`` — the user says the current step is done (``done``) or wants it skipped (``skipped``).
    * ``go_to`` — an EARLIER step to return to; the route refuses the current or a later one.
    * ``replan`` — replacement steps, the same shape as a confirm replan.

    ``spoken`` is the one short sentence the voice reads (the screen shows ``reply``); a missing or blank one is the
    reply's first sentence.

    ``user_says_done`` is a classification of the user's WORDS alone (not of the image): do they say the current
    step is already done, or that they want to move on? When it is true the user's word is the answer (design
    guide-talk §2, "사용자 확인"): the action becomes ``step_mark`` — ``done``, or ``skipped`` if the model said
    so — and every other action is dropped, so a model that judged the image otherwise cannot refuse it
    (glass e2e 2026-10-02: "이미 했어" was argued with 2/2 when ``step_mark`` was the model's own choice).

    Actions combine: at most one of ``TALK_STATE_ACTIONS`` (the first in precedence), plus ``target`` and
    ``step_say`` (a ``replan`` drops ``step_say``). Whatever had to go is named in ``dropped_actions`` (the route
    logs it); ``actions`` lists what was kept. Over-long ``reply``/``step_say`` are cut
    (``_clip``) as plan text is. A blank reply or a target outside ``TalkTarget`` is a schema violation.
    """

    user_says_done: StrictBool
    reply: TalkReply
    spoken: TalkSpoken
    step_say: StepSay | None = None
    target: TalkTarget | None = None
    step_mark: StepMark | None = None
    go_to: StepId | None = None
    replan: GuideReplan | None = None

    _dropped: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="wrap")
    @classmethod
    def repair_and_collapse(cls, data: object, handler: ModelWrapValidatorHandler[GuideTalkToolOutput]) -> GuideTalkToolOutput:
        """Clip, fold the target, default ``user_says_done``, apply it, then keep one action (see the class)."""
        data = _clip(data, {"reply": TALK_REPLY_MAX, "spoken": TALK_SPOKEN_MAX, "step_say": STEP_SAY_MAX})
        dropped: tuple[str, ...] = ()
        if isinstance(data, dict):
            # The voice line is required by the tool; a model that left it out (or blank) gets the reply's first
            # sentence, rather than a failed answer or the whole reply read aloud.
            spoken = data.get("spoken")
            if (not isinstance(spoken, str) or not spoken.strip()) and isinstance(data.get("reply"), str):
                data = {**data, "spoken": _first_sentence(data["reply"]) or None}
            if isinstance(data.get("target"), str):
                data = {**data, "target": data["target"].strip().lower()}
            # Required by the tool, but deepseek-flash left it out in 2/12 bench answers (2026-10-02): absent means
            # the words were not classified as "done", which is not a reason to fail the whole answer.
            data = {"user_says_done": False, **data}
            present = [name for name in TALK_ACTION_PRECEDENCE if data.get(name) is not None]
            if data.get("user_says_done") is True:
                mark = "skipped" if data.get("step_mark") == "skipped" else "done"
                data = {**data, **{name: None for name in TALK_ACTION_FIELDS}, "step_mark": mark}
            else:
                states = [name for name in TALK_STATE_ACTIONS if data.get(name) is not None]
                drop = set(states[1:])
                if states[:1] == ["replan"] and data.get("step_say") is not None:
                    drop.add("step_say")
                data = {**data, **{name: None for name in drop}}
            dropped = tuple(name for name in TALK_ACTION_FIELDS if name in present and data.get(name) is None)
        model = handler(data)
        model._dropped = dropped
        return model

    @model_validator(mode="after")
    def one_action(self) -> GuideTalkToolOutput:
        if not self.reply.strip():
            raise ValueError("a blank reply is not a reply")
        if sum(getattr(self, name) is not None for name in TALK_STATE_ACTIONS) > 1:
            raise ValueError("a talk answer takes at most one of step_mark, go_to and replan")
        if self.replan is not None and self.step_say is not None:
            raise ValueError("a replan rewrites the sentences itself: no step_say beside it")
        return self

    @property
    def dropped_actions(self) -> tuple[str, ...]:
        """The actions the model also took and the collapse dropped (``TALK_ACTION_FIELDS`` order)."""
        return self._dropped

    @property
    def actions(self) -> tuple[str, ...]:
        """The actions this answer takes, in ``TALK_ACTION_PRECEDENCE`` order (empty: it only replies)."""
        return tuple(name for name in TALK_ACTION_PRECEDENCE if getattr(self, name) is not None)

    @property
    def action(self) -> TalkAction:
        """The first of ``actions`` (``none`` for an answer that only replies)."""
        return (self.actions or ("none",))[0]  # type: ignore[return-value]


class GuideTalkResponse(GuideTalkToolOutput):
    """The talk answer plus the same acceptance envelope as confirm (there is no trigger).

    ``plan_revision`` is the revision AFTER this call: bumped when ``step_say`` (``guide.restate``) or
    ``replan`` (``guide.replan``) was applied to the stored plan, unchanged for every other action, because
    the step position (``step_mark``/``go_to``) and the tracker (``target``) belong to the client.
    """

    intent_id: UUIDString
    frame_id: FrameId
    intent_seq: NonnegativeInt
    task_revision: NonnegativeInt
    plan_revision: NonnegativeInt
    fence_echo: GuideFence
    anchors_echo: Annotated[list[AnchorEcho], Field(max_length=1)]
    provider: ProviderName
    model: ModelName | None = None
    usage: Usage | None = None
    latency_ms: NonnegativeInt


class GuidePlanCurrentResponse(WireModel):
    """``GET /api/guide/plan/current``: the session's current plan as stored (text only), no provider call.

    The recovery read after a ``409 stale_plan`` (for example a streamed plan whose client disconnected, or a
    replan applied by a confirm the client did not see): the client continues from exactly this
    ``plan_id``/``plan_revision``. ``task_revision`` is the task the plan was made for (always the session's
    current one: a task change clears the plan). ``steps`` may be empty only for a plan that selected no
    target or asked for clarification, as on ``POST /api/guide/plan``. ``core_mode`` is the guide core the
    plan was built for (fixed for its task), so a recovering client knows which follow semantics apply.
    """

    plan_id: UUIDString
    plan_revision: NonnegativeInt
    core_mode: CoreMode = DEFAULT_CORE_MODE
    task_revision: NonnegativeInt
    steps: PlanSteps
    goal_when: VisualCondition
    user_goal: Annotated[str, Field(min_length=1, max_length=300)]

    @model_validator(mode="after")
    def unique_steps(self) -> GuidePlanCurrentResponse:
        _check_step_graph(self.steps)
        return self


class BuildInfo(WireModel):
    """Which build answers (``backend/app/build.py``): project version, short commit, build time (ISO)."""

    version: Annotated[str, Field(min_length=1, max_length=40)]
    commit: Annotated[str, Field(min_length=1, max_length=40)]
    built_at: Annotated[str, Field(min_length=1, max_length=40)]


class GuideHealthResponse(HealthResponse):
    """``/api/health`` with the guide lane's readiness, each reported independently.

    * ``guide_ready`` — the BASIC guide lane can run: the configured provider is DeepSeek and has a key. This
      is exactly the ``deepseek:high`` profile's capability. Global readiness is unchanged by the OpenAI lane:
      it never requires an OpenAI key.
    * ``plan_models`` — per-choice configured capability, ``{'deepseek:high': bool, 'astra:high': bool}``. A
      ``True`` means that profile's credential is configured; it does NOT prove the key's model entitlement or
      remaining credit. The two entries are independent.
    * ``plan_ready`` — aggregate any-ready: at least one ``plan_models`` entry is ``True``. A session may be
      opened on it alone.
    * ``follow_provider`` — ``AISW_FOLLOW_PROVIDER`` (``None`` when it names no known follower).
    * ``follow_ready`` — the follower can run: the local or Clef endpoint answers its health probe (cached
      2 s), or, for ``deepseek``, the same condition as ``guide_ready``.
    """

    guide_ready: bool = False
    plan_ready: bool = False
    plan_models: dict[PlanModel, bool] = Field(
        default_factory=lambda: {model: False for model in PLAN_MODELS})
    follow_provider: FollowProviderName | None = None
    follow_ready: bool = False
    #: Which build answers, shown on the demo page (older servers omit it).
    build: BuildInfo | None = None


for _model in (
    AnchorRef, FocusAnchorCommand, LabelAnchorCommand, ActionAnchorCommand, Evidence, MaterialInput, GuideStep,
    GuidePlanToolOutput, GuidePlanRequest, GuidePlanResponse, GuidePlanApproveRequest, GuidePlanRevertRequest,
    GuideFence, AnchorEcho, GuideFollowRequest, AnchorVerdict, StepCheck,
    GuideFollowToolOutput, GuideFollowResponse, GuideReplan, GuideConfirmRequest, GuideConfirmToolOutput,
    GuideConfirmResponse, GuideTalkRequest, GuideTalkToolOutput, GuideTalkResponse, GuidePlanCurrentResponse,
    BuildInfo, GuideHealthResponse,
):
    _model.model_rebuild()
del _model

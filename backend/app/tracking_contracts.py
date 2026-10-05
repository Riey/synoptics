"""Standalone object-tracking HTTP contract (frozen v4).

Object tracking is **not** part of the guide contract: it has its own routes, its own run identity
(``run_id``/``start_seq``/``frame_seq``) and, for control/frame (and the user's own drawn-box path), no
paid-provider dependency. The guide routes never read tracker state.

Frozen semantics summary (see the contract document for the full text):

* Three routes: ``POST /api/track/control`` (start/stop), ``POST /api/track/frame`` (hot path), and
  ``POST /api/track/select`` (the one-shot startup analysis that chooses the target).
* Control carries **no** image/frame/seed field; ``seed_box`` belongs to the first frame of a run.
* ``start_seq`` is a client counter that is strictly increasing per application session; a delayed old
  start can never replace a newer run. ``frame_seq`` is monotonically increasing **within** a run.
* ``box`` is present **iff** ``state == "tracking"``; ``generation`` is local to a run and a server-side
  automatic re-acquire issues a new ``track_id`` with **no** identity guarantee.
* ``confidence`` is an **uncalibrated** model score, not a probability.
* ``ingest_age_ms`` is a server-side diagnostic; display freshness is the client's own capture->now clock.
* **Only ``select`` needs a paid provider.** Control, frame, and a run started from the user's drawn box
  work with no provider at all. ``select`` is also **pure**: it never reads or writes a tracker run and
  retains **no** selection result, so no later request can be answered from a remembered choice. What it
  does spend is the session's own accounting (one manual analysis and one provider slot); the
  in-flight admission flag it takes is transient bookkeeping for the duration of the call, not state.
* A target is the object detector's **text prompt** and not a human label: the selection prompt asks for a
  short **English** noun (the constraint the old UI met by requiring the user to type English), while
  human-facing strings (``rationale``) are Korean. This is a prompt convention, not an API restriction —
  nothing here validates a language.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAliasType

from pydantic import Field, StrictBool, StrictInt, model_validator

from backend.app.api_contracts import (
    ImageBase64,
    NonnegativeInt,
    ProviderName,
    SceneInput,
    Usage,
    UUIDString,
)
from backend.app.visual_contracts import Box, FrameId, Unit, WireModel


RunId = Annotated[str, Field(min_length=1, max_length=64)]
TrackId = Annotated[str, Field(min_length=1, max_length=64)]
#: The tracker service's free-text target, which becomes the object detector's text prompt. The selection
#: prompt asks for a short English noun; the field itself is unconstrained free text.
TargetText = Annotated[str, Field(min_length=1, max_length=40)]
ModelName = Annotated[str, Field(min_length=1, max_length=120)]

TrackActionKind = Literal["start", "stop"]
TrackStateKind = Literal["acquiring", "tracking", "occluded", "lost", "unavailable"]
TrackTransitionKind = Literal[
    "none", "acquired", "tracking", "occluded", "lost", "reacquired", "ambiguous"
]
TrackSourceKind = Literal["grounder", "user"]

#: The neutral target metadata a client sends for a run it starts from the user's own drawn box.
#: It is a provenance label, not a guess at the object's class: on that path the box, not this text,
#: drives acquisition, so the value is never used as a grounder prompt. It exists so the run still
#: carries a truthful target instead of an invented class name (the old implicit ``"eyeglasses."``).
MANUAL_TARGET = "user-selected object"

#: The provider's closed-set answer for one target-selection call.
#:
#: Declared as a named alias (``TypeAliasType``) on purpose: pydantic exports it as its own schema
#: definition, so the generated TypeScript can name the type ``TargetSelectionStatus`` instead of an
#: order-dependent inline name, and both the provider's answer and the wire response share one definition.
TargetSelectionStatus = TypeAliasType(
    "TargetSelectionStatus",
    Literal["selected", "no_target", "uncertain"],
)
SelectionRationale = Annotated[str, Field(min_length=1, max_length=300)]


class TrackControlRequest(WireModel):
    """Start or stop a tracking run.

    ``action="start"`` requires ``start_seq`` (strictly increasing per application session) and a
    non-blank ``target``. ``action="stop"`` names the ``run_id`` it intends to stop and carries neither.
    Image/frame/seed fields are never accepted here.

    ``target`` stays optional on the wire only so a stop can omit it: the application refuses a start
    with an absent or blank target (``422 target_required``) rather than substituting a default class
    name. The client supplies the text — the object the startup analysis selected, or the user's own
    words for a box they drew (see ``MANUAL_TARGET``).
    """

    session_id: UUIDString
    run_id: RunId
    action: TrackActionKind
    start_seq: NonnegativeInt | None = None
    target: TargetText | None = None

    @model_validator(mode="after")
    def action_fields(self) -> TrackControlRequest:
        if self.action == "start":
            if self.start_seq is None:
                raise ValueError("start requires start_seq")
        elif self.start_seq is not None or self.target is not None:
            raise ValueError("stop takes no start_seq or target")
        return self


class TrackControlResponse(WireModel):
    """Control acknowledgement: no frame, no box, no fabricated result."""

    run_id: RunId
    active: StrictBool
    target: TargetText


class TrackFrameRequest(WireModel):
    """One captured frame of the active run.

    ``seed_box`` is the user's manual selection on that captured image (normalised 0..1 in the uploaded
    image) and is accepted on the **first frame of the run only**; the frame still receives an ordinary
    frame response.
    """

    session_id: UUIDString
    run_id: RunId
    frame_id: FrameId
    frame_seq: NonnegativeInt
    image_base64: ImageBase64
    seed_box: Box | None = None


class TrackFrameQuery(WireModel):
    """The same frame as ``TrackFrameRequest``, uploaded as raw JPEG bytes instead of JSON + base64.

    ``POST /api/track/frame`` with ``Content-Type: image/jpeg``: the body is the JPEG itself and these
    fields are the query string. It carries no ``seed_box`` -- a run's seeded first frame is always sent as
    JSON. A server that predates this form answers ``415 json_required``, which a client takes as "send JSON".
    """

    session_id: UUIDString
    run_id: RunId
    frame_id: FrameId
    frame_seq: NonnegativeInt


class TrackFrameResponse(WireModel):
    """Track state for exactly the frame named by ``frame_id``/``frame_seq``."""

    run_id: RunId
    target: TargetText
    track_id: TrackId
    generation: NonnegativeInt
    state: TrackStateKind
    transition: TrackTransitionKind
    box: Box | None = None
    source: TrackSourceKind | None = None
    frame_id: FrameId
    frame_seq: NonnegativeInt
    version: NonnegativeInt
    confidence: Unit | None = None
    ingest_age_ms: NonnegativeInt

    @model_validator(mode="after")
    def box_only_while_tracking(self) -> TrackFrameResponse:
        if (self.box is not None) != (self.state == "tracking"):
            raise ValueError("box must be present if and only if state == 'tracking'")
        return self


def _check_selection(status: str, target: str | None, rationale: str | None) -> None:
    """The one target-provenance gate, shared by the provider's answer and the wire response.

    A target exists exactly when something was selected, and a blank string counts as nothing: a
    whitespace-only target or rationale is treated as missing rather than as content. This is what keeps
    a contradictory or empty provider answer from becoming a fabricated target downstream — the caller
    refuses the answer instead of renumbering it.

    Nothing here checks the target's language: the wording that asks for an English detector prompt lives
    in the provider request (``TARGET_SYSTEM_PROMPT``), and this gate cannot verify a language anyway.
    """
    has_target = target is not None and target.strip() != ""
    if target is not None and not has_target:
        raise ValueError("a blank target is not a target")
    if has_target != (status == "selected"):
        raise ValueError("target must be present if and only if status == 'selected'")
    has_rationale = rationale is not None and rationale.strip() != ""
    if rationale is not None and not has_rationale:
        raise ValueError("a blank rationale is not a rationale")
    if status != "selected" and not has_rationale:
        raise ValueError("a non-selection must state its rationale")


class TargetSelectionToolOutput(WireModel):
    """The provider's structured answer for one startup target-selection call.

    ``status`` is the honest closed set: ``selected`` names one object, ``no_target`` says the scene holds
    no plausible object to track, ``uncertain`` says the frame cannot support the decision (blurred,
    occluded, ambiguous). The latter two never carry a target, and they must say why.

    ``target`` is what the tracker's detector will be prompted with, so it is a short **English** noun
    (``laptop``, ``screwdriver``) rather than a translated or descriptive phrase; ``rationale`` is the
    human-facing Korean sentence. The two are never mixed, and no translation step sits between them.
    """

    status: TargetSelectionStatus
    target: TargetText | None = None
    rationale: SelectionRationale | None = None

    @model_validator(mode="after")
    def target_iff_selected(self) -> TargetSelectionToolOutput:
        _check_selection(self.status, self.target, self.rationale)
        return self


class TrackSelectRequest(WireModel):
    """One startup analysis: choose what the tracker should track, from the frame captured right now.

    ``scene.frame_id`` is the frame this analysis is about, and the client reuses that exact frame as the
    tracking run's first frame. ``user_goal``/``context`` are the task text the user is working under, so
    the choice is about *their* object rather than the most salient thing in the image.
    """

    session_id: UUIDString
    consent_ai: bool
    scene: SceneInput
    user_goal: Annotated[str, Field(min_length=1, max_length=300)] | None = None
    context: Annotated[str, Field(min_length=1, max_length=1000)] | None = None


class TrackSelectResponse(WireModel):
    """One startup analysis answer: a candidate target (or an honest non-selection) and what it cost.

    A non-selection is a successful answer, not an error: ``status`` says which one it is and ``rationale``
    says why. ``target`` is a short **English** noun for the tracker's detector, never a location — this
    analysis does not claim to have located anything, and the tracker's own acquisition accuracy is
    unchanged by it. A client may show that noun as-is (it is the label the detector is working from);
    ``rationale`` is the human-facing Korean sentence to show a person.
    ``usage``/``provider``/``model`` are the one call's own accounting, reported for every outcome
    including a non-selection, so a consumed call is never silently uncounted.
    """

    selection_id: UUIDString
    frame_id: FrameId
    status: TargetSelectionStatus
    target: TargetText | None = None
    rationale: SelectionRationale | None = None
    provider: ProviderName
    model: ModelName | None = None
    usage: Usage | None = None

    @model_validator(mode="after")
    def target_iff_selected(self) -> TrackSelectResponse:
        _check_selection(self.status, self.target, self.rationale)
        return self

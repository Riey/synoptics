"""Application HTTP/session contract: the shared envelopes (health, session, scene, usage) of the API.

This layer composes the domain-neutral visual contract (``visual_contracts``) with the concrete
request/response envelopes the browser talks to. It is generated for the web app only; the shared
``@visual-coach/visual-tools`` package deliberately does not export it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictInt

from backend.app.visual_contracts import FrameId, WireModel


NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
UUIDString = Annotated[str, Field(min_length=36, max_length=36, pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", json_schema_extra={"format": "uuid"})]
ImageBase64 = Annotated[str, Field(min_length=1, max_length=2_500_000)]
AssetLabel = Annotated[str, Field(min_length=1, max_length=100)]
GoalStatusKind = Literal["visually_satisfied", "in_progress", "uncertain", "not_applicable"]
#: The deployment's session mode: ``code`` (``DEMO_ACCESS_CODE``), ``local`` (``AISW_LOCAL_ONLY=1``, loopback
#: only) or ``open`` (``AISW_OPEN_ACCESS=1``, no code; the network in front of it is the boundary).
AccessMode = Literal["code", "local", "open"]


class VisualGoalStatus(WireModel):
    """Whole-goal visual assessment for the application/user, distinct from step-level advice.

    - visually_satisfied: Current scene clearly and directly shows the user goal has been accomplished.
    - in_progress: Goal is active and work visibly remains to be done.
    - uncertain: Visual evidence in the current scene is insufficient, blurry, occluded, or missing targets.
    - not_applicable: No explicit user goal was supplied.
    """

    status: GoalStatusKind
    rationale: Annotated[str, Field(min_length=1, max_length=600)]


class Usage(WireModel):
    """Provider token accounting for one call. Belongs to the app/provider layer, not the renderer SDK."""

    input_tokens: NonnegativeInt | None = None
    output_tokens: NonnegativeInt | None = None
    total_tokens: NonnegativeInt | None = None
    cached_input_tokens: NonnegativeInt | None = None
    thought_tokens: NonnegativeInt | None = None
    tool_use_tokens: NonnegativeInt | None = None


ProviderName = Annotated[str, Field(min_length=1, max_length=40)]


class HealthResponse(WireModel):
    """Readiness probe.

    ``access_code_required`` reports the deployment's session authentication mode, never the
    calling browser's identity: it is ``False`` only for an explicit local-only deployment
    (``AISW_LOCAL_ONLY=1``, loopback peer + loopback authority) or an explicit open-access deployment
    (``AISW_OPEN_ACCESS=1``), where the UI must not ask for a code at all. The client is not allowed
    to infer this from its own hostname.

    ``access_mode`` names that mode (``code`` / ``local`` / ``open``). It is ``None`` only for the
    refused configuration (both ``AISW_OPEN_ACCESS`` and ``AISW_LOCAL_ONLY`` set), where ``ready`` is
    false, ``access_code_required`` is true and every session route answers ``503``.

    ``tracker_ready`` reports the standalone object tracker independently of the paid provider:
    either readiness may be true without the other, and neither is derived from the other.
    """

    ready: bool
    model: str
    provider: str = "gemini"
    access_code_required: bool
    access_mode: AccessMode | None = None
    tracker_ready: bool = False


class SessionRequest(WireModel):
    """Session request.

    ``access_code`` is optional on the wire. An explicit local-only or open-access deployment issues
    sessions without a code (any supplied value is ignored), so the browser omits the field entirely;
    every other deployment requires the configured ``DEMO_ACCESS_CODE`` and rejects a missing, empty,
    or wrong value.
    """

    access_code: str | None = None


class SessionResponse(WireModel):
    session_id: UUIDString


class EndSessionResponse(WireModel):
    ended: bool


class SceneInput(WireModel):
    """The single current frame the advice is drawn against."""

    frame_id: FrameId
    image_base64: ImageBase64
    label: AssetLabel | None = None


VisualGoalStatus.model_rebuild()

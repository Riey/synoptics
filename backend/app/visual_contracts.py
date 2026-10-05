"""Domain-neutral visual guidance contract: scene primitives, advice, and tool-schema helpers.

Nothing in this module knows about any particular domain (PC assembly, paper folding, drawing...).
An agent supplies the goal, the how-to knowledge, and the meaning of the scene; this contract only
carries the visual primitives, the provenance references, and the structured advice envelope.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


PositiveUnit = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class WireModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)


class Point(WireModel):
    x: Unit
    y: Unit


class Box(WireModel):
    x: Unit
    y: Unit
    width: PositiveUnit
    height: PositiveUnit

    @model_validator(mode="after")
    def inside_frame(self) -> Box:
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("box extends beyond the image")
        return self


CommandId = Annotated[str, Field(min_length=1, max_length=64)]
CommandLabel = Annotated[str, Field(min_length=1, max_length=60)]
FrameId = Annotated[str, Field(min_length=1, max_length=64)]
ReferenceId = Annotated[str, Field(min_length=1, max_length=64)]


class FocusCommand(WireModel):
    id: CommandId
    kind: Literal["focus"]
    box: Box
    label: CommandLabel


class ArrowCommand(WireModel):
    id: CommandId
    kind: Literal["arrow"]
    from_: Point = Field(alias="from")
    to: Point
    label: CommandLabel


class PathCommand(WireModel):
    id: CommandId
    kind: Literal["path"]
    points: Annotated[list[Point], Field(min_length=2, max_length=64)]
    label: CommandLabel


class GestureCommand(WireModel):
    id: CommandId
    kind: Literal["gesture"]
    points: Annotated[list[Point], Field(min_length=2, max_length=64)]
    duration_ms: Annotated[StrictInt, Field(ge=1000, le=5000)]
    label: CommandLabel


class HintCommand(WireModel):
    id: CommandId
    kind: Literal["hint"]
    text: Annotated[str, Field(min_length=1, max_length=200)]
    reference: Literal["none", "line", "circle", "hatching"]


VisualCommand = Union[FocusCommand, ArrowCommand, PathCommand, GestureCommand, HintCommand]


EvidenceKind = Literal["reference", "observed_scene", "general_inference", "uncertain_view"]


class GuidelineStep(WireModel):
    """An optional, agent-authored step. Steps are content, never a mandatory state machine."""

    step_id: Annotated[str, Field(min_length=1, max_length=64)]
    title: Annotated[str, Field(min_length=1, max_length=120)]
    detail: Annotated[str, Field(min_length=1, max_length=400)]


class GuidanceAdvice(WireModel):
    """The single advice envelope produced by an agent/provider and returned to the browser."""

    explanation: Annotated[str, Field(min_length=1, max_length=600)]
    observation: Annotated[str, Field(min_length=1, max_length=600)]
    steps: Annotated[list[GuidelineStep], Field(max_length=20)]
    evidence_kind: EvidenceKind
    references: Annotated[list[ReferenceId], Field(max_length=10)]
    commands: Annotated[list[VisualCommand], Field(max_length=3)]
    needs_clarification: bool
    clarification_prompt: Annotated[str, Field(min_length=1, max_length=300)] | None = None
    ready_to_advance: bool
    warnings: Annotated[list[Annotated[str, Field(min_length=1, max_length=200)]], Field(max_length=5)]

    @model_validator(mode="after")
    def consistent_advice(self) -> GuidanceAdvice:
        if self.evidence_kind == "uncertain_view" and not self.needs_clarification:
            raise ValueError("uncertain_view evidence requires needs_clarification=True")
        if self.needs_clarification:
            if self.ready_to_advance:
                raise ValueError("cannot advise advancing while clarification is needed")
            if not self.clarification_prompt or not self.clarification_prompt.strip():
                raise ValueError("clarification_prompt is required when clarification is needed")
            if any(cmd.kind != "hint" for cmd in self.commands):
                raise ValueError("only hint commands are allowed when clarification is needed")
        if not self.needs_clarification and self.clarification_prompt is not None:
            raise ValueError("clarification_prompt must be null when clarification is not needed")
        if self.evidence_kind == "reference" and not self.references:
            raise ValueError("reference evidence requires at least one cited reference")
        if len(set(self.references)) != len(self.references):
            raise ValueError("duplicate references")
        if len({cmd.id for cmd in self.commands}) != len(self.commands):
            raise ValueError("duplicate command ids")
        if sum(cmd.kind == "gesture" for cmd in self.commands) > 1:
            raise ValueError("at most one gesture command allowed")
        return self


class VisualGuidance(WireModel):
    """The framed envelope consumed by every renderer entrypoint: which guidance, which frame, what advice."""

    guidance_id: FrameId
    frame_id: FrameId
    advice: GuidanceAdvice


class FramedRenderReport(WireModel):
    """What the renderer did with a VisualGuidance payload.

    This is a *client-reported* observation: it records what the local renderer actually did on the
    caller's machine. It is untrusted input to the server — never a completion proof, a security
    assertion, or evidence that a human verified anything.

    ``frame_id`` is the frame the guidance **targeted** (the report's identity), so a report about an
    older frame stays identifiable. In a live-camera flow the report a client forwards usually names the
    *previous* capture, not the one being analysed now: it is then history about superseded guidance,
    and it is never evidence about the current scene. Only when the payload is too malformed to identify
    at all does the renderer fall back to the frame it was displaying. ``guidance_id`` is null in that
    same case.
    """

    guidance_id: FrameId | None
    frame_id: FrameId
    status: Literal["rendered", "rejected", "stale"]
    command_ids: Annotated[list[CommandId], Field(max_length=3)]
    reason: Annotated[str, Field(min_length=1, max_length=200)] | None = None


def clean_tool_schema(model: type[BaseModel], name: str, description: str) -> dict[str, Any]:
    """Generate a standard function tool schema directly from a canonical WireModel."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_key = node["$ref"].split("/")[-1]
                target = defs.get(ref_key, {})
                return resolve(target)
            return {k: resolve(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": resolve(schema),
        },
    }


def gemini_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Unwrap OpenAI/DeepSeek nested tool schema to Gemini Interactions flat tool format."""
    fn = schema["function"]
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn["description"],
        "parameters": fn["parameters"],
    }

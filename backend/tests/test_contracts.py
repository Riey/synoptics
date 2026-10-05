"""Unit tests for the domain-neutral visual contract and the tool schema shape."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.api_contracts import VisualGoalStatus
from backend.app.visual_contracts import (
    Box,
    FocusCommand,
    FramedRenderReport,
    GuidanceAdvice,
    VisualGuidance,
    clean_tool_schema,
)


def advice(**overrides: object) -> GuidanceAdvice:
    payload: dict[str, object] = {
        "explanation": "설명",
        "observation": "관찰",
        "steps": [],
        "evidence_kind": "observed_scene",
        "references": [],
        "commands": [],
        "needs_clarification": False,
        "clarification_prompt": None,
        "ready_to_advance": False,
        "warnings": [],
    }
    payload.update(overrides)
    return GuidanceAdvice.model_validate(payload)


def test_box_must_stay_inside_the_frame_with_finite_positive_extent() -> None:
    for bad in (
        {"x": 0.5, "y": 0.5, "width": 0.6, "height": 0.1},
        {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.5},
        {"x": float("nan"), "y": 0.0, "width": 0.5, "height": 0.5},
        {"x": 0.0, "y": float("inf"), "width": 0.5, "height": 0.5},
        {"x": -0.1, "y": 0.0, "width": 0.5, "height": 0.5},
    ):
        with pytest.raises(ValidationError):
            Box.model_validate(bad)
    assert Box.model_validate({"x": 0.9, "y": 0.9, "width": 0.1, "height": 0.1}).x == 0.9


def test_advice_relational_rules_are_fail_closed() -> None:
    hint = {"id": "h1", "kind": "hint", "text": "각도를 바꿔보세요", "reference": "none"}
    focus = {"id": "f1", "kind": "focus", "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}, "label": "대상"}

    with pytest.raises(ValidationError, match="uncertain_view"):
        advice(evidence_kind="uncertain_view")

    with pytest.raises(ValidationError, match="cannot advise advancing"):
        advice(needs_clarification=True, clarification_prompt="각도를 바꿔주세요", ready_to_advance=True)

    with pytest.raises(ValidationError, match="clarification_prompt is required"):
        advice(needs_clarification=True)

    with pytest.raises(ValidationError, match="only hint commands"):
        advice(needs_clarification=True, clarification_prompt="각도를 바꿔주세요", commands=[focus])

    with pytest.raises(ValidationError, match="must be null"):
        advice(clarification_prompt="필요 없는데 있음")

    with pytest.raises(ValidationError, match="reference evidence requires"):
        advice(evidence_kind="reference")

    with pytest.raises(ValidationError, match="duplicate command ids"):
        advice(commands=[focus, {**focus, "label": "다른 라벨"}])

    with pytest.raises(ValidationError, match="at most one gesture"):
        advice(commands=[
            {"id": "g1", "kind": "gesture", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}], "duration_ms": 1000, "label": "a"},
            {"id": "g2", "kind": "gesture", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}], "duration_ms": 1000, "label": "b"},
        ])

    # Mixed support is allowed: an observed-scene answer may also cite a supplied reference.
    assert advice(references=["manual-1"]).references == ["manual-1"]
    assert advice(needs_clarification=True, clarification_prompt="각도를 바꿔주세요", commands=[hint]).needs_clarification


def test_framed_report_and_envelope_shapes() -> None:
    report = FramedRenderReport(guidance_id=None, frame_id="frame-1", status="rejected", command_ids=[], reason="x")
    assert report.guidance_id is None
    with pytest.raises(ValidationError):
        FramedRenderReport(guidance_id="g1", frame_id="frame-1", status="rendered", command_ids=[], reason="x" * 201)

    with pytest.raises(ValidationError):
        VisualGuidance.model_validate({"guidance_id": "g1", "frame_id": "frame-1"})


def test_tool_schema_is_inlined_and_commands_are_optional() -> None:
    schema = clean_tool_schema(GuidanceAdvice, name="guidance_advice", description="advice")
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "guidance_advice"

    def assert_no_refs(node: object) -> None:
        if isinstance(node, dict):
            assert "$ref" not in node
            for value in node.values():
                assert_no_refs(value)
        elif isinstance(node, list):
            for item in node:
                assert_no_refs(item)

    assert_no_refs(schema)

    parameters = schema["function"]["parameters"]
    assert parameters["properties"]["commands"]["maxItems"] == 3
    assert "steps" in parameters["required"]
    assert "usage" not in parameters["properties"]


def test_imported_focus_command_aliases_wire_key() -> None:
    command = FocusCommand.model_validate(
        {"id": "f1", "kind": "focus", "box": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}, "label": "전체"}
    )
    assert command.model_dump(by_alias=True)["box"]["width"] == 1.0


def test_goal_status_contract() -> None:
    status = VisualGoalStatus.model_validate({"status": "visually_satisfied", "rationale": "작업이 완료되었습니다."})
    assert status.status == "visually_satisfied"

    with pytest.raises(ValidationError):
        VisualGoalStatus.model_validate({"status": "invalid_status", "rationale": "r"})

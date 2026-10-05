"""Export the canonical Pydantic contracts as two stable JSON Schema bundles.

* visual bundle -> ``packages/visual-tools/src/visual.schema.json``
  The domain-neutral visual contract any agent may author against (shared package).
* app bundle    -> ``web/src/generated/api.schema.json``
  This application's HTTP/session contract (web only; never exported by the shared package).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic.json_schema import models_json_schema

from backend.app import api_contracts, guide_contracts, plan_manual, tracking_contracts, visual_contracts


VISUAL_MODELS = (
    visual_contracts.Point,
    visual_contracts.Box,
    visual_contracts.FocusCommand,
    visual_contracts.ArrowCommand,
    visual_contracts.PathCommand,
    visual_contracts.GestureCommand,
    visual_contracts.HintCommand,
    visual_contracts.GuidelineStep,
    visual_contracts.GuidanceAdvice,
    visual_contracts.VisualGuidance,
    visual_contracts.FramedRenderReport,
)

API_MODELS = (
    api_contracts.HealthResponse,
    api_contracts.SessionRequest,
    api_contracts.SessionResponse,
    api_contracts.EndSessionResponse,
    api_contracts.SceneInput,
    api_contracts.Usage,
    api_contracts.VisualGoalStatus,
    # Standalone tracking contract (frozen v4): its own routes.
    tracking_contracts.TrackControlRequest,
    tracking_contracts.TrackControlResponse,
    tracking_contracts.TrackFrameRequest,
    tracking_contracts.TrackFrameQuery,
    tracking_contracts.TrackFrameResponse,
    # The startup target-selection route: the tracking lane's only paid call.
    tracking_contracts.TrackSelectRequest,
    tracking_contracts.TrackSelectResponse,
    # Anchored guide lane: plan (DeepSeek) -> follow (local or DeepSeek) -> confirm (DeepSeek).
    guide_contracts.AnchorRef,
    guide_contracts.FocusAnchorCommand,
    guide_contracts.LabelAnchorCommand,
    guide_contracts.ActionAnchorCommand,
    guide_contracts.GuideStep,
    guide_contracts.GuideFence,
    guide_contracts.AnchorEcho,
    guide_contracts.AnchorVerdict,
    guide_contracts.GuideReplan,
    guide_contracts.GuidePlanRequest,
    guide_contracts.GuidePlanResponse,
    guide_contracts.GuideFollowRequest,
    guide_contracts.GuideFollowResponse,
    guide_contracts.GuideConfirmRequest,
    guide_contracts.GuideConfirmResponse,
    # User utterances during a guide: one reply, at most one action (DeepSeek).
    guide_contracts.GuideTalkRequest,
    guide_contracts.GuideTalkToolOutput,
    guide_contracts.GuideTalkResponse,
    guide_contracts.GuidePlanCurrentResponse,
    # Plan review (§15): the client-reviewed plan adopted, or the previous plan restored.
    guide_contracts.GuidePlanApproveRequest,
    guide_contracts.GuidePlanRevertRequest,
    guide_contracts.Evidence,
    guide_contracts.MaterialInput,
    # The offline plan-material import: a file in, a MaterialInput out (no model call).
    plan_manual.ManualImportRequest,
    guide_contracts.BuildInfo,
    guide_contracts.GuideHealthResponse,
)


def write_bundle(models: tuple[type[Any], ...], title: str, target: Path) -> None:
    _, generated = models_json_schema(
        [(model, "validation") for model in models],
        ref_template="#/$defs/{model}",
    )
    definitions = generated["$defs"]
    expected = {model.__name__ for model in models}
    if not expected <= set(definitions):
        raise RuntimeError(f"{title}: missing definitions {expected - set(definitions)}")
    output = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "properties": {name: {"$ref": f"#/$defs/{name}"} for name in sorted(expected)},
        "additionalProperties": False,
        "$defs": definitions,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{target} ({len(definitions)} definitions)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    write_bundle(
        VISUAL_MODELS,
        "VisualContracts",
        root / "packages/visual-tools/src/visual.schema.json",
    )
    write_bundle(
        API_MODELS,
        "GuidanceApiContracts",
        root / "web/src/generated/api.schema.json",
    )


if __name__ == "__main__":
    main()

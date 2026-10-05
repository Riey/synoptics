"""Talk reply bench: one ``/api/guide/talk`` provider call per (case, variant, repeat), no browser, no tracker.

Each call is the production request (``intent.talk_tool``/``talk_prompt``, ``prepare_frame_b64``, the DeepSeek
provider's own payload) on a fixed frame and a fixed plan, so variants differ only in model and thinking.
Prints one JSON line per call (latency, finish reason, usage incl. reasoning tokens, action, reply) and appends
it to ``--out``. PAID: every call is a DeepSeek request.

Usage:
  DEEPSEEK_API_KEY_FILE=~/.config/synoptics/API_KEY.txt uv run python tools/e2e/talk_bench.py \
      --frames <dir with glass.jpg, put-airpod.jpg> --out talk_bench.jsonl --reps 2 \
      --variants flash-off flash-think pro-off
  (--probe-cap N: one flash-think call with max_tokens=N, to see whether the cap counts reasoning tokens)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from backend.app import intent
from backend.app.guide import GuideSessionState, StoredPlan
from backend.app.guide_contracts import GuideStep, GuideTalkRequest, GuideTalkToolOutput
from backend.app.provider import DeepSeekProvider, UPPER_MAX_TOKENS, _guide_tool_arguments

VARIANTS = {
    "flash-off": ("deepseek-flash", False),
    "flash-think": ("deepseek-flash", True),
    "pro-off": ("deepseek-v4-pro", False),
}


def step(step_id: str, say: str, done_when: str) -> GuideStep:
    return GuideStep.model_validate({"id": step_id, "say": say, "commands": [], "done_when": done_when})


#: (name, frame file, goal, target, steps, current step, anchors, utterance, expected action)
CASES: list[dict[str, Any]] = [
    {
        "name": "bottle-stuck", "frame": "put-airpod.jpg", "goal": "페트병 뚜껑 열기", "target": "plastic bottle",
        "steps": [step("s1", "병 몸통을 한 손으로 꼭 잡으세요", "한 손이 병 몸통을 감싸 쥔 모습이 보임"),
                  step("s2", "뚜껑을 왼쪽으로 돌려 여세요", "뚜껑이 병에서 분리되어 손에 들려 있음")],
        "goal_when": "뚜껑이 열려 병과 분리되어 있음", "current": "s2", "anchors": [],
        "utterance": "힘껏 돌리는데 열리지 않아", "expected": "step_say",
    },
    {
        "name": "glass-where", "frame": "glass.jpg", "goal": "안경 벗기", "target": "eyeglasses",
        "steps": [step("s1", "양손으로 안경 다리를 잡으세요", "양손이 안경 다리를 잡고 있는 모습이 보임"),
                  step("s2", "안경을 얼굴에서 앞으로 빼세요", "안경이 얼굴에서 떨어져 손에 들려 보임")],
        "goal_when": "얼굴에 안경이 없고 안경이 손에 들려 있음", "current": "s2",
        "anchors": [{"anchor_id": "a1", "role": "target", "label": "eyeglasses", "run_id": "run-1",
                     "track_id": "t-1", "generation": 0, "state": "tracking",
                     "box": {"x": 0.44, "y": 0.5, "width": 0.25, "height": 0.13}}],
        "utterance": "어디를 말하는지 모르겠어", "expected": "target",
    },
    {
        "name": "glass-easier", "frame": "glass.jpg", "goal": "안경 벗기", "target": "eyeglasses",
        "steps": [step("s1", "양손으로 안경 다리를 잡으세요", "양손이 안경 다리를 잡고 있는 모습이 보임"),
                  step("s2", "안경을 얼굴에서 앞으로 빼세요", "안경이 얼굴에서 떨어져 손에 들려 보임")],
        "goal_when": "얼굴에 안경이 없고 안경이 손에 들려 있음", "current": "s2",
        "anchors": [{"anchor_id": "a1", "role": "target", "label": "eyeglasses", "run_id": "run-1",
                     "track_id": "t-1", "generation": 0, "state": "tracking",
                     "box": {"x": 0.44, "y": 0.5, "width": 0.25, "height": 0.13}}],
        "utterance": "더 쉽게 설명해줘", "expected": "step_say",
    },
]


def case_plan(case: dict[str, Any]) -> StoredPlan:
    return GuideSessionState().install(steps=case["steps"], goal_when=case["goal_when"], user_goal=case["goal"],
                                       context=None, task_revision=1, target=case["target"])


def case_request(case: dict[str, Any], plan: StoredPlan, image_b64: str) -> GuideTalkRequest:
    return GuideTalkRequest.model_validate({
        "session_id": str(uuid.uuid4()), "consent_ai": True,
        "scene": {"frame_id": f"bench-{case['name']}", "image_base64": image_b64},
        "plan_id": plan.plan_id, "plan_revision": plan.plan_revision, "current_step": case["current"],
        "anchors": case["anchors"], "utterance": case["utterance"], "intent_seq": 1,
        "fence": {"task_epoch": "bench", "run_id": "run-1"},
    })


async def one_call(provider: DeepSeekProvider, case: dict[str, Any], frames: Path, variant: str,
                   rep: int, cap: int | None = None) -> dict[str, Any]:
    model, thinking = VARIANTS[variant]
    plan = case_plan(case)
    raw = base64.b64encode((frames / case["frame"]).read_bytes()).decode("ascii")
    request = case_request(case, plan, raw)
    prepared = intent.prepare_frame_b64(raw, request.anchors)
    system_prompt, tool_schema = intent.talk_tool(request.replan_allowed)
    max_tokens = cap if cap is not None else UPPER_MAX_TOKENS
    payload = provider._guide_payload(
        system_prompt=system_prompt, text=intent.talk_prompt(request, plan), frame_id=request.scene.frame_id,
        image_b64=prepared, tool_schema=tool_schema, tool_name=intent.TALK_TOOL_NAME, max_tokens=max_tokens,
        thinking=thinking,
    )
    payload["model"] = model
    record: dict[str, Any] = {"case": case["name"], "variant": variant, "rep": rep, "max_tokens": max_tokens,
                              "expected": case["expected"]}
    started = time.monotonic()
    try:
        response = await provider._post_chat(payload, timeout_seconds=60.0)
        record["latency_ms"] = round((time.monotonic() - started) * 1000)
        choice = (response.get("choices") or [{}])[0]
        record["finish_reason"] = choice.get("finish_reason")
        usage = response.get("usage") or {}
        record["usage"] = {"completion": usage.get("completion_tokens"), "prompt": usage.get("prompt_tokens"),
                           "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        output = GuideTalkToolOutput.model_validate(_guide_tool_arguments(response, intent.TALK_TOOL_NAME))
        record.update(action=output.action, hit=output.action == case["expected"], reply=output.reply,
                      reply_len=len(output.reply), step_say=output.step_say, target=output.target,
                      user_says_done=output.user_says_done)
    except Exception as exc:  # noqa: BLE001 - a bench records every failure kind
        record.setdefault("latency_ms", round((time.monotonic() - started) * 1000))
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--cases", nargs="+", default=[case["name"] for case in CASES])
    parser.add_argument("--probe-cap", type=int, default=None)
    args = parser.parse_args()
    async with httpx.AsyncClient() as client:
        provider = DeepSeekProvider(client)
        jobs: list[tuple[dict[str, Any], str, int, int | None]] = []
        if args.probe_cap is not None:
            jobs.append((CASES[0], "flash-think", 0, args.probe_cap))
        else:
            for rep in range(1, args.reps + 1):
                for case in CASES:
                    if case["name"] in args.cases:
                        jobs.extend((case, variant, rep, None) for variant in args.variants)
        with args.out.open("a", encoding="utf-8") as out:
            for case, variant, rep, cap in jobs:  # sequential: latency is not shared with another call
                record = await one_call(provider, case, args.frames, variant, rep, cap)
                line = json.dumps(record, ensure_ascii=False)
                print(line, flush=True)
                out.write(line + "\n")


if __name__ == "__main__":
    asyncio.run(main())

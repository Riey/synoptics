"""A clarification changes execution authority, not the user's goal or verified research."""
from __future__ import annotations

import asyncio
import io
from copy import deepcopy
from PIL import Image

from backend.app.intent import prepare_frame

from backend.tests.test_guide import (
    PLAN, api_client, approve_body, env, install,
    open_session, plan, post_approve, unpace,
)
from backend.tests.test_plan_astra import astra

SOURCE = {"url": "https://example.com/manual", "title": "제조사 설명서", "summary": "해당 부품의 방향 표시는 아랫면 기준입니다."}
QUESTION = "라벨을 읽을 수 있도록 부품 윗면 사진 한 장을 보내 주세요."


def test_answer_replaces_execution_authority_but_keeps_verified_research(env, astra):
    async def exercise():
        install()
        astra.answers["guide_plan"] = {
            **deepcopy(PLAN), "steps": [], "needs_clarification": True,
            "clarification_prompt": QUESTION, "research_sources": [SOURCE],
        }
        async with api_client() as api:
            headers, sid = await open_session(api)
            first = await plan(api, headers, sid, plan_model="astra:high", assisted=True, research=True)
            assert first.status_code == 200, first.text
            assert first.json()["clarification_prompt"] == QUESTION
            unpace(sid)
            astra.answers["guide_plan"] = deepcopy(PLAN)
            next_plan = await plan(api, headers, sid, plan_model="astra:high", assisted=True,
                                   answers=[{"question": QUESTION, "answer": "사진의 라벨은 ABC입니다."}])
            assert next_plan.status_code == 200, next_plan.text
            assert next_plan.json()["research_sources"] == [SOURCE]
            stale_approval = await post_approve(api, headers, approve_body(sid, first.json(), PLAN["steps"]))
            assert stale_approval.status_code == 409
            unpace(sid)
            changed = await plan(api, headers, sid, user_goal="다른 작업", plan_model="astra:high", assisted=True)
            assert changed.status_code == 200, changed.text
            assert changed.json()["research_sources"] == []
    asyncio.run(exercise())


def test_invalid_reference_never_reaches_paid_provider(env, astra):
    async def exercise():
        install()
        async with api_client() as api:
            headers, sid = await open_session(api)
            response = await plan(api, headers, sid, plan_model="astra:high", assisted=True,
                                  reference_images=[{"frame_id": "detail", "image_base64": "not-a-jpeg"}])
            assert response.status_code == 422
            assert astra.calls == []
    asyncio.run(exercise())


def test_research_never_silently_falls_back_to_another_issuer(env, astra):
    async def exercise():
        basic = install()
        async with api_client() as api:
            headers, sid = await open_session(api)
            response = await plan(api, headers, sid, plan_model="deepseek:high", assisted=True, research=True)
            assert response.status_code == 503
            assert basic.tools() == [] and astra.calls == []
    asyncio.run(exercise())


def test_phone_reference_orientation_matches_the_displayed_photo():
    original = Image.new("RGB", (200, 80), "blue")
    original.paste("red", (0, 0, 100, 80))
    exif = Image.Exif()
    exif[274] = 6  # Phone viewer rotates this JPEG clockwise.
    encoded = io.BytesIO()
    original.save(encoded, format="JPEG", exif=exif)
    shown_to_model = Image.open(io.BytesIO(prepare_frame(encoded.getvalue(), [])))
    assert shown_to_model.size == (80, 200)
    assert shown_to_model.getpixel((40, 40))[0] > 200
    assert shown_to_model.getpixel((40, 160))[2] > 200

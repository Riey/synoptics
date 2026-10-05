"""Plan-model upper-lane routing: the OpenAI lane and the basic paid lane never mix.

Same discipline as ``test_guide.py``: the real ASGI application, with the BASIC lane a real
``DeepSeekProvider`` behind an ``httpx.MockTransport`` (prompt, envelope parsing and usage accounting are the
production ones) and the PLAN lane a contract-shaped fake standing in for ``store.astra``. The OpenAI
provider's own transport is exercised by ``test_astra.py``; what is under test here is only *which* lane a
route dispatches to, with *which* model name, and that a missing or unusable Plan config is a clean refusal —
never a silent fallback to the basic provider.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from backend.app.api_contracts import Usage
from backend.app.guide_contracts import GuidePlanRequest
from backend.app.main import store
from backend.app.provider import GuideStreamFinal, ProviderConfigError, ProviderUnavailable, UnavailableStage
from backend.tests.test_guide import (
    GOAL,
    PLAN,
    api_client,
    approve_body,
    confirm,
    confirm_body,
    env,  # noqa: F401  (fixture)
    install,
    make_test_jpeg,
    open_session,
    plan,
    post_approve,
    post_revert,
    revert_body,
    review_step,
    unpace,
)
from backend.tests.test_guide_stream import parse_sse
from backend.tests.test_guide_talk import TALK, talk, talk_body

SSE = {"Accept": "text/event-stream"}
USAGE = Usage(input_tokens=7, output_tokens=3, total_tokens=10)

CONFIRM = {"goal_status": {"status": "in_progress", "rationale": "컵이 아직 화면 왼쪽에 있습니다."},
           "evidence_kind": "observed_scene", "replan": None, "needs_clarification": False}


class FakeAstra:
    """A ``store.astra`` stand-in with the exact Plan-provider surface (see ``astra.AstraProvider``).

    Records every call by method name, answers canned forced-tool arguments, and reports ``configured`` as a
    property (like the real one) so the model-aware readiness probe is exercised for real.
    """

    provider_name = "openai"
    model = "gpt-6-astra"

    def __init__(self, answers: dict[str, Any] | None = None, *, configured: bool = True) -> None:
        self.answers = {"guide_plan": PLAN, "guide_confirm": CONFIRM, "guide_talk": TALK, **(answers or {})}
        #: Tool name -> ProviderError to raise from that method (failure injection for the no-fallback tests).
        self.failures: dict[str, Exception] = {}
        self._configured = configured
        self.calls: list[str] = []
        self.requests: list[GuidePlanRequest] = []

    @property
    def configured(self) -> bool:
        return self._configured

    def _answer(self, tool: str) -> dict[str, Any]:
        return dict(self.answers[tool])

    def _maybe_fail(self, tool: str) -> None:
        if tool in self.failures:
            raise self.failures[tool]

    async def guide_plan(self, request: GuidePlanRequest, image_b64: str) -> tuple[dict[str, Any], Usage]:
        self.calls.append("guide_plan")
        self.requests.append(request)
        self._maybe_fail("guide_plan")
        return self._answer("guide_plan"), USAGE

    async def guide_plan_stream(self, request: GuidePlanRequest, image_b64: str):  # async generator
        self.calls.append("guide_plan_stream")
        self.requests.append(request)
        self._maybe_fail("guide_plan_stream")
        text = json.dumps(self._answer("guide_plan"), ensure_ascii=False)
        for start in range(0, len(text), 8):
            yield text[start:start + 8]
        yield GuideStreamFinal(self._answer("guide_plan"), USAGE)

    async def guide_confirm(self, request: Any, plan: Any, image_b64: str,
                            before_b64: str | None = None) -> tuple[dict[str, Any], Usage]:
        self.calls.append("guide_confirm")
        self._maybe_fail("guide_confirm")
        return self._answer("guide_confirm"), USAGE

    async def guide_talk(self, request: Any, plan: Any, image_b64: str) -> tuple[dict[str, Any], Usage]:
        self.calls.append("guide_talk")
        self._maybe_fail("guide_talk")
        return self._answer("guide_talk"), USAGE


@pytest.fixture
def astra(env) -> FakeAstra:  # noqa: F811
    """Install a fake Plan provider on the store for the test and restore the real one after."""
    fake = FakeAstra()
    previous = store.astra
    store.astra = fake
    try:
        yield fake
    finally:
        store.astra = previous


def plan_body(session_id: str, **extra: Any) -> dict[str, Any]:
    return {"session_id": session_id, "consent_ai": True,
            "scene": {"frame_id": "cam-1", "image_base64": make_test_jpeg()},
            "user_goal": GOAL, "context": "작업 공간 정리", **extra}


# --------------------------------------------------------------------------------- request strictness


def test_plan_model_is_a_closed_choice_and_plan_mode_is_obsolete(env) -> None:  # noqa: F811
    """``plan_model`` takes only a known choice: an unknown value, a legacy ``plan_mode`` boolean or a
    ``mode``/``manual`` alias is refused as ``422`` before any provider is dispatched. Both choices and the
    absent default are accepted."""
    async def exercise() -> None:
        stub = install()
        async with api_client() as api:
            headers, session_id = await open_session(api)
            for extra in ({"plan_model": "opus:max"}, {"plan_model": "deepseek"}, {"plan_model": True},
                          {"plan_mode": True}, {"plan_mode": False}, {"mode": True}, {"manual": True}):
                response = await api.post("/api/guide/plan", json=plan_body(session_id, **extra), headers=headers)
                assert response.status_code == 422, (extra, response.text)
                assert response.json() == {"error": "invalid_request"}
            assert stub.tools() == []  # nothing reached any provider

    asyncio.run(exercise())


# --------------------------------------------------------------------------------- lane selection


def test_basic_and_plan_sessions_route_to_their_own_lane_concurrently(env, astra) -> None:
    """Two independent sessions in flight: basic calls DeepSeek, Plan calls OpenAI, each with
    its own model name in the JSON response — and neither call lands on the other's provider."""
    async def exercise() -> None:
        stub = install()
        async with api_client() as api:
            basic_headers, basic_sid = await open_session(api)
            plan_headers, plan_sid = await open_session(api)
            basic, planned = await asyncio.gather(
                plan(api, basic_headers, basic_sid),
                plan(api, plan_headers, plan_sid, plan_model="astra:high"),
            )
            assert basic.status_code == 200, basic.text
            assert (basic.json()["provider"], basic.json()["model"]) == ("deepseek", "deepseek-flash")
            assert planned.status_code == 200, planned.text
            assert (planned.json()["provider"], planned.json()["model"]) == ("openai", "gpt-6-astra")
            assert stub.tools() == ["guide_plan"]
            assert astra.calls == ["guide_plan"]

    asyncio.run(exercise())


def test_astra_plan_accepts_zero_materials(env, astra) -> None:
    """An ``astra:high`` start with no reference documents reaches the OpenAI lane without materials and
    installs a plan (no synthesized empty list, no refusal)."""
    async def exercise() -> None:
        install()
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await plan(api, headers, session_id, plan_model="astra:high")
            assert response.status_code == 200, response.text
            assert response.json()["provider"] == "openai"
            assert astra.requests[0].materials is None

    asyncio.run(exercise())


def test_streamed_plan_uses_the_plan_lane_and_names_its_model(env, astra) -> None:
    """With ``Accept: text/event-stream`` the Plan lane's stream is drained (not the basic one) and the
    terminal event carries the OpenAI provider/model."""
    async def exercise() -> None:
        stub = install()
        async with api_client() as api:
            headers, session_id = await open_session(api)
            response = await api.post("/api/guide/plan", json=plan_body(session_id, plan_model="astra:high"),
                                      headers={**headers, **SSE})
            unpace(session_id)
            assert response.status_code == 200, response.text
            events, _pings = parse_sse(response.text)
            kinds = [kind for kind, _data in events]
            assert kinds[-1] == "final"
            final = events[-1][1]
            assert (final["provider"], final["model"]) == ("openai", "gpt-6-astra")
            partials = {name for kind, data in events if kind == "partial" for name in data}
            assert partials == {"target", "first_say"}
            assert astra.calls == ["guide_plan_stream"] and stub.tools() == []

    asyncio.run(exercise())


# --------------------------------------------------------------------------------- inherited route


def test_approve_revert_and_talk_keep_the_plan_lane(env, astra) -> None:
    """An ``astra:high`` plan stays on the OpenAI lane across approve → confirm → revert → talk: every upper
    call after Start routes from the stored plan's own ``plan_model``, and the basic lane is never called."""
    async def exercise() -> None:
        stub = install()
        async with api_client() as api:
            headers, session_id = await open_session(api)
            planned = (await plan(api, headers, session_id, plan_model="astra:high")).json()
            edited = [review_step(1, say="뚜껑을 여세요"), review_step(2, say="덮개를 닫으세요")]
            approved = await post_approve(api, headers, approve_body(session_id, planned, edited))
            assert approved.status_code == 200, approved.text
            adopted = approved.json()
            confirmed = await confirm(api, headers, confirm_body(session_id, adopted, trigger="goal_check"))
            reverted = (await post_revert(api, headers, revert_body(session_id, adopted))).json()
            talked = await talk(api, headers, talk_body(session_id, reverted))

            assert confirmed.status_code == 200, confirmed.text
            assert (confirmed.json()["provider"], confirmed.json()["model"]) == ("openai", "gpt-6-astra")
            assert talked.status_code == 200, talked.text
            assert (talked.json()["provider"], talked.json()["model"]) == ("openai", "gpt-6-astra")
            assert astra.calls == ["guide_plan", "guide_confirm", "guide_talk"]
            assert stub.tools() == []

    asyncio.run(exercise())


def test_replan_revision_keeps_the_plan_lane(env, astra) -> None:
    """A talk replan stays on the Plan lane and bumps the stored revision; the next confirm does too."""
    async def exercise() -> None:
        new_steps = [review_step(1, say="새 단계를 하세요")]
        stub = install()
        astra.answers["guide_talk"] = {**TALK, "replan": {"steps": new_steps}}
        async with api_client() as api:
            headers, session_id = await open_session(api)
            planned = (await plan(api, headers, session_id, plan_model="astra:high")).json()
            replanned = await talk(api, headers, talk_body(session_id, planned))
            assert replanned.status_code == 200, replanned.text
            data = replanned.json()
            assert data["provider"] == "openai" and data["replan"]["steps"][0]["say"] == "새 단계를 하세요"
            assert data["plan_revision"] == planned["plan_revision"] + 1

            next_call = await confirm(api, headers, confirm_body(
                session_id, {**planned, "plan_revision": data["plan_revision"]}, trigger="goal_check"))
            assert next_call.status_code == 200, next_call.text
            assert next_call.json()["provider"] == "openai"
            assert stub.tools() == []  # the basic lane never saw the plan or its revision

    asyncio.run(exercise())


# --------------------------------------------------------------------------------- model switch


def test_switching_plan_model_invalidates_the_old_plan_and_its_history(env, astra) -> None:
    """A plan_model change under the same goal/context is a new task: the ``deepseek:high`` plan and its
    history are gone, its pair is stale on every upper route, and revert cannot revive it."""
    async def exercise() -> None:
        stub = install()
        async with api_client() as api:
            headers, session_id = await open_session(api)
            basic = (await plan(api, headers, session_id)).json()
            planned = (await plan(api, headers, session_id, plan_model="astra:high")).json()
            assert planned["provider"] == "openai"

            old_talk = await talk(api, headers, talk_body(session_id, basic))
            assert old_talk.status_code == 409 and old_talk.json()["error"] == "stale_plan"
            # History was cleared with the plan, so the new plan has nothing to revert to.
            no_history = await post_revert(api, headers, revert_body(session_id, planned))
            assert no_history.status_code == 404 and no_history.json() == {"error": "no_plan"}
            assert stub.tools() == ["guide_plan"]  # the basic lane ran exactly once

    asyncio.run(exercise())


# --------------------------------------------------------------------------------- no fallback


def test_unconfigured_api_key_blocks_plan_without_touching_the_basic_lane(env, astra) -> None:
    """A missing Plan key causes a clean ``503`` before dispatch, and the basic
    provider is never called as a fallback. The basic lane itself is unaffected."""
    async def exercise() -> None:
        stub = install()
        astra._configured = False
        async with api_client() as api:
            headers, session_id = await open_session(api)
            refused = await plan(api, headers, session_id, plan_model="astra:high")
            assert refused.status_code == 503 and refused.json() == {"error": "service_unavailable"}
            assert astra.calls == [] and stub.tools() == []
            # The basic lane still works in the same deployment.
            basic = await plan(api, headers, session_id)
            assert basic.status_code == 200 and basic.json()["provider"] == "deepseek"

    asyncio.run(exercise())


def test_plan_provider_config_failure_is_not_papered_over_by_deepseek(env, astra) -> None:
    """An API auth failure is reported as itself, never by re-running the basic lane."""
    async def exercise() -> None:
        stub = install()
        astra.failures["guide_plan"] = ProviderConfigError("OpenAI API credentials are not accepted")
        async with api_client() as api:
            headers, session_id = await open_session(api)
            refused = await plan(api, headers, session_id, plan_model="astra:high")
            assert refused.status_code == 503 and refused.json() == {"error": "provider_configuration"}
            assert stub.tools() == []

    asyncio.run(exercise())


def test_plan_transport_failure_is_reported_as_provider_unavailable(env, astra) -> None:
    async def exercise() -> None:
        stub = install()
        astra.failures["guide_plan"] = ProviderUnavailable(UnavailableStage.TRANSPORT)
        async with api_client() as api:
            headers, session_id = await open_session(api)
            refused = await plan(api, headers, session_id, plan_model="astra:high")
            assert refused.status_code == 502
            assert refused.json() == {"error": "provider_unavailable", "reason": "transport"}
            assert stub.tools() == []

    asyncio.run(exercise())


# --------------------------------------------------------------------------------- readiness


def test_openai_only_deployment_opens_a_session_and_plans(env, astra) -> None:
    """An OpenAI key opens sessions and serves ``astra:high`` with no basic key; ``deepseek:high`` stays closed
    and ``guide_ready`` never starts requiring an OpenAI key. ``plan_models`` names each choice, and
    ``plan_ready`` is their aggregate."""
    async def exercise() -> None:
        stub = install()
        stub.provider.api_key = ""
        async with api_client() as api:
            health = (await api.get("/api/health")).json()
            assert health["guide_ready"] is False  # the basic lane is not ready
            assert health["plan_models"] == {"deepseek:high": False, "astra:high": True}
            assert health["plan_ready"] is True  # aggregate any-ready

            headers, session_id = await open_session(api)  # Astra alone is enough to open a session
            assert (await plan(api, headers, session_id, plan_model="astra:high")).status_code == 200
            assert astra.calls == ["guide_plan"]

            basic = await plan(api, headers, session_id)  # deepseek:high is still refused, never routed to Astra
            assert basic.status_code == 503 and basic.json() == {"error": "service_unavailable"}
            assert astra.calls == ["guide_plan"]  # the OpenAI lane did not serve the deepseek request

    asyncio.run(exercise())

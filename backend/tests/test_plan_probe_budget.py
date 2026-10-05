"""The plan-probe's append-only budget ledger plus its fake-transport provider lanes.

In-memory HTTP only (``httpx.MockTransport``); no real provider, no key and no research tool is reached. What
is under test is the ledger's append-only reservation/settlement arithmetic (refusal at the attempt cap and the
USD cap, retention and blocking when accounting is missing, survival of a crashed reservation, thread-safe
locking, fail-closed corruption handling), the conservative cost model (per-call search fees, no silent zeros),
and that the probe never spends or invents a plan outside those boundaries.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # the probe lives under tools/, outside the installed application package
    sys.path.insert(0, str(_ROOT))

from backend.app.astra import AstraProvider  # noqa: E402
from backend.app.guide_contracts import GuidePlanRequest, ResearchSource  # noqa: E402
from backend.app.provider import DeepSeekProvider  # noqa: E402
from backend.tests.test_guide import PLAN, make_test_jpeg  # noqa: E402
from tools.e2e.plan_usability_probe import (  # noqa: E402
    ASTRA_PRICE,
    MAX_TOOL_RESULT_CHARS,
    BudgetLimits,
    BudgetRefused,
    LedgerError,
    ProbeConfigError,
    PLAN_PREPARE_LONG_SIDE,
    ProbeLedger,
    ResearchObservation,
    astra_cost_estimate,
    prepare_plan_images,
    probe_astra,
    probe_deepseek,
    run_plan_probe,
)

KEY = "test-deepseek-key"
OPENAI_KEY = "test-openai-key"
REASONING = "private reasoning that must never be serialized"
SEARCH_URL = "https://www.meanwell.com/upload/pdf/LDD-L-SPEC.pdf"
INVENTED_URL = "https://example.invalid/invented-datasheet.pdf"
PAGE_URL = "https://www.meanwell.com/upload/pdf/LDD-L-SPEC.pdf#page=3"


def plan_request() -> GuidePlanRequest:
    return GuidePlanRequest.model_validate({
        "session_id": "11111111-1111-4111-8111-111111111111", "consent_ai": True,
        "plan_model": "deepseek:high",
        "scene": {"frame_id": "probe-scene", "image_base64": make_test_jpeg()},
        "reference_images": [{"frame_id": "ref-pin-side", "image_base64": make_test_jpeg(color="green")}],
        "answers": [{"question": "사용할 LED는?", "answer": "500 mA 파워 LED입니다."}],
        "user_goal": "LED 회로를 배선하고 시험 점등하기", "context": "초보자 안내", "assisted": True,
        "research": True,
    })


def limits(**overrides: Any) -> dict[str, BudgetLimits]:
    return {"p": BudgetLimits(**{"max_attempts": 10, "max_usd": 10.0, "reserve_usd": 0.5, **overrides})}


# ------------------------------------------------------------------------------------------------------------------
# Ledger boundaries
# ------------------------------------------------------------------------------------------------------------------


def test_attempt_cap_refuses_the_next_reservation(tmp_path: Path) -> None:
    ledger = ProbeLedger(tmp_path / "ledger.jsonl", limits(max_attempts=2, max_usd=100.0, reserve_usd=1.0))
    first = ledger.reserve("p")
    second = ledger.reserve("p")
    with pytest.raises(BudgetRefused):
        ledger.reserve("p")
    ledger.settle(first, 0.25)
    ledger.settle(second, 0.25)
    assert ledger.spent_usd("p") == 0.5
    assert ledger.attempts("p") == 2


def test_usd_cap_counts_the_settled_cost(tmp_path: Path) -> None:
    ledger = ProbeLedger(tmp_path / "ledger.jsonl", limits(max_usd=1.0, reserve_usd=0.4))
    reservation = ledger.reserve("p")
    ledger.settle(reservation, 0.9)
    with pytest.raises(BudgetRefused):
        ledger.reserve("p")
    assert ledger.spent_usd("p") == 0.9


def test_unknown_usage_retains_the_reservation_and_blocks_the_provider(tmp_path: Path) -> None:
    ledger = ProbeLedger(tmp_path / "ledger.jsonl", limits(reserve_usd=0.7))
    reservation = ledger.reserve("p")
    ledger.settle(reservation, None, note="transport error")
    snapshot = ledger.snapshot()["providers"]["p"]
    assert snapshot["blocked"] is True
    assert snapshot["retained_usd"] == pytest.approx(0.7)
    assert snapshot["spent_usd"] == pytest.approx(0.7)
    with pytest.raises(BudgetRefused):
        ledger.reserve("p")
    ledger.unblock("p", reason="checked by hand")
    assert ledger.reserve("p").provider == "p"


def test_a_crashed_reservation_still_counts_after_the_ledger_is_reopened(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    crashed = ProbeLedger(path, limits(max_usd=1.0, reserve_usd=0.6))
    crashed.reserve("p")  # never settled: the probe died mid-attempt
    reopened = ProbeLedger(path, limits(max_usd=1.0, reserve_usd=0.6))
    snapshot = reopened.snapshot()["providers"]["p"]
    assert snapshot["outstanding_usd"] == pytest.approx(0.6)
    assert snapshot["spent_usd"] == pytest.approx(0.6)
    with pytest.raises(BudgetRefused):
        reopened.reserve("p")


def test_release_drops_an_attempt_that_never_opened_a_socket(tmp_path: Path) -> None:
    ledger = ProbeLedger(tmp_path / "ledger.jsonl", limits(reserve_usd=0.6))
    reservation = ledger.reserve("p")
    ledger.release(reservation)
    assert ledger.spent_usd("p") == 0.0
    assert ledger.reserve("p").provider == "p"


def test_ledger_is_append_only_and_replays_in_order(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = ProbeLedger(path, limits())
    reservation = ledger.reserve("p")
    after_reserve = path.read_text(encoding="utf-8")
    ledger.settle(reservation, 0.25, known_usd=0.20, basis="estimated")
    after_settle = path.read_text(encoding="utf-8")
    assert after_settle.startswith(after_reserve), "a later mutation must never rewrite an earlier line"
    events = [json.loads(line) for line in after_settle.splitlines()]
    assert [event.get("op") or event.get("kind") for event in events] == ["header", "reserve", "settle"]
    assert [event["seq"] for event in events[1:]] == [1, 2]
    snapshot = ledger.snapshot()["providers"]["p"]
    assert snapshot["settled_usd"] == pytest.approx(0.25)
    assert snapshot["settled_known_usd"] == pytest.approx(0.20)
    assert snapshot["estimated_settlements"] == 1


def test_a_corrupt_ledger_log_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"version": 1, "kind": "header"}\n{"op": "reserve", oops\n', encoding="utf-8")
    ledger = ProbeLedger(path, limits())
    with pytest.raises(LedgerError):
        ledger.snapshot()


def test_a_ledger_that_vanishes_after_initialization_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = ProbeLedger(path, limits())
    path.unlink()
    with pytest.raises(LedgerError):
        ledger.snapshot()


def test_concurrent_reservations_cannot_oversubscribe(tmp_path: Path) -> None:
    ledger = ProbeLedger(tmp_path / "ledger.jsonl", limits(max_usd=1.0, reserve_usd=0.6))
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def worker() -> None:
        barrier.wait()
        try:
            ledger.reserve("p")
            outcomes.append("ok")
        except BudgetRefused:
            outcomes.append("refused")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["ok", "refused"]


def test_reconciled_total_preserves_history_and_replaces_provider_caps(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    provider_limits = {
        name: BudgetLimits(max_attempts=1, max_usd=0.8, reserve_usd=0.4) for name in ("a", "b")
    }
    ledger = ProbeLedger(path, provider_limits)
    old = ledger.reserve("a")
    ledger.settle(old, None)
    ledger.unblock("a", reason="operator checked the completed attempt")
    old_log = path.read_bytes()
    ledger.reconcile_total(reported_usd=0.2, max_usd=1.0, reason="operator-reported cumulative spend")
    assert path.read_bytes().startswith(old_log)
    ledger = ProbeLedger(path, provider_limits)
    first = ledger.reserve("a")
    second = ledger.reserve("b")
    with pytest.raises(BudgetRefused):
        ledger.reserve("b")
    ledger.settle(first, 0.1)
    ledger.settle(second, 0.1)
    third = ledger.reserve("a")
    ledger.release(third)
    snapshot = ledger.snapshot()
    assert snapshot["providers"]["a"]["attempts"] == 3
    assert snapshot["providers"]["a"]["retained_usd"] == pytest.approx(0.4)
    assert snapshot["total_budget"]["spent_usd"] == pytest.approx(0.4)
    assert snapshot["total_budget"]["max_attempts"] is None
    unknown = ledger.reserve("a")
    ledger.settle(unknown, None)
    assert ledger.snapshot()["total_budget"]["spent_usd"] == pytest.approx(0.8)
    with pytest.raises(BudgetRefused):
        ledger.reserve("a")


def test_reconciliation_cannot_erase_an_in_flight_attempt(tmp_path: Path) -> None:
    ledger = ProbeLedger(tmp_path / "ledger.jsonl", limits())
    ledger.reserve("p")
    with pytest.raises(LedgerError):
        ledger.reconcile_total(reported_usd=0.0, max_usd=8.0, reason="operator report")


# ------------------------------------------------------------------------------------------------------------------
# Cost arithmetic: per-call search fees, conservative upper settlements, no silent zeros.
# ------------------------------------------------------------------------------------------------------------------


def test_search_fees_are_per_call_not_per_token() -> None:
    assert ASTRA_PRICE.search_cost_usd(2) == pytest.approx(0.02)
    assert ASTRA_PRICE.token_cost_usd(input_tokens=0, cached_input_tokens=0, cache_write_tokens=0,
                                      output_tokens=0) == 0.0


def test_astra_cost_charges_unreported_cache_writes_as_an_upper_bound() -> None:
    class _Usage:
        input_tokens = 2000
        cached_input_tokens = 400
        output_tokens = 500

    known = astra_cost_estimate(_Usage(), search_calls=1, search_observed=True)
    assert known is not None
    assert known.basis == "estimated"  # cache-write tokens are never reported by the Responses envelope
    assert known.settle_usd == pytest.approx(0.0004 + 0.02 + 0.025 + 0.01)
    assert known.known_usd == pytest.approx(0.016 + 0.0004 + 0.025 + 0.01)
    assert known.settle_usd > known.known_usd

    uncounted = astra_cost_estimate(_Usage(), search_calls=0, search_observed=False)
    assert uncounted is not None
    assert uncounted.settle_usd == pytest.approx(0.0004 + 0.02 + 0.025 + 0.02), \
        "an uncountable search is charged at the bounded maximum, never zero"


def test_astra_cost_refuses_to_guess_without_usage() -> None:
    assert astra_cost_estimate(None, search_calls=0, search_observed=True) is None


# ------------------------------------------------------------------------------------------------------------------
# Fake-transport provider lanes
# ------------------------------------------------------------------------------------------------------------------


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def chat_message(tool_calls: list[dict[str, Any]], *, reasoning: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": tool_calls}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500,
                      "prompt_cache_hit_tokens": 200, "prompt_cache_miss_tokens": 1000,
                      "completion_tokens_details": {"reasoning_tokens": 120}}}


def deepseek_transport(requests: list[dict[str, Any]],
                       plan_arguments: dict[str, Any] | None = None,
                       first_call: dict[str, Any] | None = None) -> httpx.MockTransport:
    """Round 1 asks ``first_call`` (a ``search_web`` call by default); round 2 answers with ``guide_plan``."""
    answers = plan_arguments or {**PLAN, "research_sources": [
        {"url": SEARCH_URL, "title": "LDD-L datasheet", "summary": "pin configuration"},
        {"url": INVENTED_URL, "title": "invented", "summary": "never observed"}]}
    opener = first_call or tool_call("c1", "search_web", {"query": "MEAN WELL LDD-500L pinout"})

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({"auth": request.headers.get("authorization"), "body": body})
        if len(requests) == 1:
            return httpx.Response(200, json=chat_message([opener], reasoning=REASONING))
        return httpx.Response(200, json=chat_message(
            [tool_call("c2", "guide_plan", answers)], reasoning=REASONING))

    return httpx.MockTransport(handler)


async def research(name: str, arguments: dict[str, Any]) -> ResearchObservation:
    if name == "search_web":
        assert arguments.get("query")
        return ResearchObservation(
            text="MEAN WELL LDD-500L: 500 mA; pin 6 = +Vout.",
            sources=(ResearchSource(url=SEARCH_URL, title="LDD-L datasheet", summary="pin configuration"),))
    return ResearchObservation(text="page text")


def test_deepseek_lane_refuses_a_missing_research_callback_before_any_request(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=deepseek_transport(requests)) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            with pytest.raises(ProbeConfigError):
                await probe_deepseek(request=plan_request(), prepared_scene=make_test_jpeg(), research=None,
                                     ledger=ProbeLedger(tmp_path / "ledger.jsonl"), provider=provider,
                                     client=client, allow_network=True)

    asyncio.run(exercise())
    assert requests == []


def test_deepseek_lane_runs_the_real_tool_loop_and_records_the_plan(tmp_path: Path) -> None:
    run = tmp_path / "run"
    requests: list[dict[str, Any]] = []

    async def exercise() -> Any:
        async with httpx.AsyncClient(transport=deepseek_transport(requests)) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            return await run_plan_probe(run_dir=run, request=plan_request(), research=research,
                                        providers=("deepseek",), allow_network=True, deepseek_provider=provider)

    report = asyncio.run(exercise())
    result = report["results"]["deepseek"]
    assert result["ok"] and result["outcome"] == "plan"
    assert result["has_final_plan"] and result["has_steps"] and result["needs_clarification"] is False
    assert [step["id"] for step in result["final_plan"]["steps"]] == ["s1", "s2"]
    # Only a URL the tool really returned is accepted as evidence; a model-written URL is discarded.
    assert [source["url"] for source in result["research_sources"]] == [SEARCH_URL]
    assert result["discarded_source_urls"] == [INVENTED_URL]
    assert result["provenance_verified"] is True
    assert result["observed_source_urls"] == [SEARCH_URL]
    # Every round is one accounted attempt with the provider's own usage and cost.
    assert [attempt["outcome"] for attempt in result["attempts"]] == ["ok", "ok"]
    assert result["attempts"][0]["usage"]["prompt_cache_miss_tokens"] == 1000
    assert result["cost_basis"] == "known"  # both cache buckets were reported
    assert result["total_cost_usd"] == pytest.approx(2 * 0.0006612)  # two rounds, same reported usage
    assert result["total_cost_usd"] == result["total_known_cost_usd"]
    assert result["latency_s"] >= 0
    # The tool conversation: all three tools, a high-thinking profile, and the reasoning continued to its issuer.
    first, second = requests[0]["body"], requests[1]["body"]
    assert requests[0]["auth"] == f"Bearer {KEY}"
    assert [tool["function"]["name"] for tool in first["tools"]] == ["search_web", "read_url", "guide_plan"]
    search_tool = next(tool for tool in first["tools"] if tool["function"]["name"] == "search_web")
    assert "METADATA ONLY" in search_tool["function"]["description"]  # never the search provider's answer
    assert first["thinking"] == {"type": "enabled"} and first["reasoning_effort"] == "high"
    assert first["tool_choice"] == "auto"
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-2].get("reasoning_content") == REASONING
    # Reference photos are labelled separately; the current frame is last.
    texts = [part["text"] for part in first["messages"][1]["content"] if part["type"] == "text"]
    assert any("별도 참고 사진 1" in text for text in texts)
    assert texts[-1].startswith("[현재 장면 사진]")
    assert [record["name"] for record in result["tool_calls"]] == ["search_web"]
    assert result["tool_calls"][0]["arguments"] == {"query": "MEAN WELL LDD-500L pinout"}
    assert result["tool_calls"][0]["round"] == 1
    # Both lanes are prepared at the production assisted size, and the reference photo was downscaled.
    assert report["request"]["prepare_long_side"] == 1600
    assert report["request"]["scene_prepared_bytes"] > 0
    assert report["request"]["reference_frames"] == ["ref-pin-side"]
    reference_prepared_bytes = report["request"]["reference_prepared_bytes"]
    assert len(reference_prepared_bytes) == 1 and reference_prepared_bytes[0] > 0
    # The DeepSeek lane received the same PREPARED reference photo the report accounted for.
    image_parts = [part for part in first["messages"][1]["content"] if part["type"] == "image_url"]
    assert len(image_parts) == 2  # the reference photo, then the current scene
    reference_payload = image_parts[0]["image_url"]["url"].split(",", 1)[1]
    assert (len(reference_payload) * 3) // 4 == reference_prepared_bytes[0]
    assert report["request"]["answers"] == [{"question": "사용할 LED는?", "answer": "500 mA 파워 LED입니다."}]

    # Artifacts: the plan and its accounting, never hidden reasoning and never a credential.
    before = (run / "attempts.jsonl").read_text(encoding="utf-8")
    assert len(before.splitlines()) == 2
    ledger_before = (run / "ledger.jsonl").read_text(encoding="utf-8")
    for name in ("report.json", "attempts.jsonl", "ledger.jsonl"):
        text = (run / name).read_text(encoding="utf-8")
        assert REASONING not in text
        assert KEY not in text
    # An attempts artifact is appended, never overwritten; the ledger log is never rewritten either.
    async def again() -> Any:
        async with httpx.AsyncClient(transport=deepseek_transport([])) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            return await run_plan_probe(run_dir=run, request=plan_request(), research=research,
                                        providers=("deepseek",), allow_network=True, deepseek_provider=provider)

    asyncio.run(again())
    after = (run / "attempts.jsonl").read_text(encoding="utf-8")
    assert after.startswith(before) and len(after.splitlines()) == 4
    assert (run / "ledger.jsonl").read_text(encoding="utf-8").startswith(ledger_before)


def test_prepare_plan_images_uses_the_production_assisted_long_side() -> None:
    """Both lanes are handed the production assisted preparation (1600), not one lane's raw originals."""
    import io

    from PIL import Image

    def size(base64_text: str) -> tuple[int, int]:
        with Image.open(io.BytesIO(base64.b64decode(base64_text))) as image:
            return image.size

    request = plan_request()
    request = request.model_copy(update={
        "scene": request.scene.model_copy(update={"image_base64": make_test_jpeg(width=2400, height=1600)}),
        "reference_images": [request.reference_images[0].model_copy(
            update={"image_base64": make_test_jpeg(width=1600, height=2400, color="green")})],
    })
    scene, prepared = prepare_plan_images(request)
    assert PLAN_PREPARE_LONG_SIDE == 1600
    assert max(size(scene)) == 1600
    assert max(size(prepared.reference_images[0].image_base64)) == 1600
    assert max(size(request.scene.image_base64)) == 2400  # the caller's request is not mutated
    assert max(size(request.reference_images[0].image_base64)) == 2400


def test_a_requested_read_url_is_not_evidence_by_itself(tmp_path: Path) -> None:
    """Asking to read a page proves nothing about what came back: only the observation's own sources ground."""
    run = tmp_path / "run"
    requests: list[dict[str, Any]] = []
    plan = {**PLAN, "research_sources": [{"url": PAGE_URL, "title": "page", "summary": "cited"}]}

    async def blind(name: str, arguments: dict[str, Any]) -> ResearchObservation:
        return ResearchObservation(text="nothing useful came back")  # no sources observed

    async def exercise() -> Any:
        transport = deepseek_transport(requests, plan_arguments=plan,
                                       first_call=tool_call("c1", "read_url", {"url": PAGE_URL}))
        async with httpx.AsyncClient(transport=transport) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            return await probe_deepseek(request=plan_request(), prepared_scene=make_test_jpeg(), research=blind,
                                        ledger=ProbeLedger(run / "ledger.jsonl"), provider=provider,
                                        client=client, allow_network=True)

    result = asyncio.run(exercise())
    assert result.research_sources == []
    assert result.discarded_source_urls == [PAGE_URL]
    assert result.observed_source_urls == []


def test_a_cancelled_attempt_retains_its_reservation_and_blocks_the_provider(tmp_path: Path) -> None:
    run = tmp_path / "run"
    ledger = ProbeLedger(run / "ledger.jsonl", {"deepseek": BudgetLimits(128, 10.0, 0.5)})

    def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async def exercise() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            try:
                await probe_deepseek(request=plan_request(), prepared_scene=make_test_jpeg(), research=research,
                                     ledger=ledger, provider=provider, client=client, allow_network=True)
            except asyncio.CancelledError:
                return "cancelled"
        return "returned"

    assert asyncio.run(exercise()) == "cancelled"
    snapshot = ledger.snapshot()["providers"]["deepseek"]
    assert snapshot["attempts"] == 1
    assert snapshot["blocked"] is True
    assert snapshot["retained_usd"] == pytest.approx(0.5)
    with pytest.raises(BudgetRefused):
        ledger.reserve("deepseek")


def test_provider_failures_are_reported_as_a_type_and_stage_only(tmp_path: Path) -> None:
    run = tmp_path / "run"
    marker = "SECRET-TRANSPORT-MESSAGE-MARKER"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(marker)

    async def exercise() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            return await run_plan_probe(run_dir=run, request=plan_request(), research=research,
                                        providers=("deepseek",), allow_network=True, deepseek_provider=provider)

    report = asyncio.run(exercise())
    result = report["results"]["deepseek"]
    assert result["outcome"] == "provider_error"
    assert result["error_type"] == "ProviderUnavailable"
    assert result["error_stage"] == "transport"
    assert result["has_final_plan"] is False
    assert result["total_cost_usd"] is None
    assert result["cost_basis"] == "unknown"
    assert report["ledger"]["providers"]["deepseek"]["retained_usd"] > 0
    assert marker not in json.dumps(result, ensure_ascii=False)
    assert marker not in (run / "report.json").read_text(encoding="utf-8")


def test_tool_results_are_bounded_and_the_bound_is_recorded(tmp_path: Path) -> None:
    run = tmp_path / "run"
    requests: list[dict[str, Any]] = []

    async def huge(name: str, arguments: dict[str, Any]) -> ResearchObservation:
        return ResearchObservation(text="x" * (MAX_TOOL_RESULT_CHARS * 3))

    async def exercise() -> Any:
        async with httpx.AsyncClient(transport=deepseek_transport(requests)) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            return await probe_deepseek(request=plan_request(), prepared_scene=make_test_jpeg(),
                                        research=huge, ledger=ProbeLedger(run / "ledger.jsonl"),
                                        provider=provider, client=client, allow_network=True)

    result = asyncio.run(exercise())
    record = result.tool_calls[0]
    original = MAX_TOOL_RESULT_CHARS * 3
    assert record["truncated"] is True
    assert record["original_chars"] == original
    assert record["result_chars"] <= MAX_TOOL_RESULT_CHARS + 128
    tool_content = requests[1]["body"]["messages"][-1]["content"]
    assert len(tool_content) == record["result_chars"]
    assert f"the tool returned {original} characters" in tool_content  # the marker names the true length


def test_a_real_sized_manufacturer_read_is_not_truncated(tmp_path: Path) -> None:
    """The trial's real read_url extraction (~12k characters; the pin table starts around 6.9k) must survive
    intact — a 6000-character cap would have kept the chart pages and cut the pinout."""
    run = tmp_path / "run"
    requests: list[dict[str, Any]] = []
    headline = "PIN CONFIGURATION\n"
    page = headline + "x" * (12015 - len(headline))

    async def page_reader(name: str, arguments: dict[str, Any]) -> ResearchObservation:
        return ResearchObservation(text=page)

    async def exercise() -> Any:
        transport = deepseek_transport(requests, first_call=tool_call("c1", "read_url", {"url": PAGE_URL}))
        async with httpx.AsyncClient(transport=transport) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            return await probe_deepseek(request=plan_request(), prepared_scene=make_test_jpeg(),
                                        research=page_reader, ledger=ProbeLedger(run / "ledger.jsonl"),
                                        provider=provider, client=client, allow_network=True)

    result = asyncio.run(exercise())
    record = result.tool_calls[0]
    assert MAX_TOOL_RESULT_CHARS >= 12015
    assert record["truncated"] is False
    assert record["original_chars"] == 12015 and record["result_chars"] == 12015
    assert requests[1]["body"]["messages"][-1]["content"] == page


def test_astra_lane_uses_the_production_stream_and_counts_real_searches(tmp_path: Path) -> None:
    run = tmp_path / "run"
    calls: list[dict[str, Any]] = []
    search_item = {"id": "ws_1", "type": "web_search_call", "status": "completed",
                   "action": {"type": "search", "query": "LDD-500L pinout",
                              "sources": [{"url": SEARCH_URL, "title": "LDD-L datasheet"}]}}

    def stream(arguments: dict[str, Any]) -> str:
        item = {"id": "fc_1", "call_id": "call_1", "type": "function_call", "name": "guide_plan",
                "status": "completed", "arguments": json.dumps(arguments, ensure_ascii=False)}
        events = [
            {"type": "response.output_item.added", "item": search_item},
            {"type": "response.output_item.done", "item": search_item},
            {"type": "response.output_item.added", "item": {**item, "arguments": "", "status": "in_progress"}},
            {"type": "response.reasoning_summary_text.delta", "delta": "astra private reasoning"},
            {"type": "response.function_call_arguments.delta", "item_id": item["id"], "delta": item["arguments"]},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"status": "completed", "output": [item],
             "usage": {"input_tokens": 2000, "output_tokens": 500, "total_tokens": 2500,
                       "input_tokens_details": {"cached_tokens": 400}}}},
        ]
        return "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"auth": request.headers.get("authorization"), "body": json.loads(request.content)})
        return httpx.Response(200, content=stream({**PLAN, "research_sources": [
            {"url": SEARCH_URL, "title": "LDD-L datasheet", "summary": "adapter observed"}]}),
            headers={"content-type": "text/event-stream"})

    async def exercise() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = AstraProvider(client, api_key=OPENAI_KEY)
            return await probe_astra(request=plan_request(), prepared_scene=make_test_jpeg(),
                                     ledger=ProbeLedger(run / "ledger.jsonl"), provider=provider, client=client,
                                     allow_network=True)

    result = asyncio.run(exercise())
    assert result.ok and result.outcome == "plan"
    assert result.model == "gpt-6-astra"
    assert result.native_research_requested is True
    # The real web_search_call item was observed, so the search fee is charged once (not the bounded maximum).
    assert result.observed_search_calls == 1
    assert result.attempts[0].search_calls == 1
    assert result.provenance_verified is True
    assert [source["url"] for source in result.research_sources] == [SEARCH_URL]
    assert result.attempts[0].usage["cached_input_tokens"] == 400
    assert result.cost_basis == "estimated"  # cache-write tokens are unobservable; the bound carries them
    assert result.total_cost_usd == pytest.approx(0.0004 + 0.025 + 0.02 + 0.01)
    assert result.total_known_cost_usd == pytest.approx(0.016 + 0.0004 + 0.025 + 0.01)
    assert calls[0]["auth"] == f"Bearer {OPENAI_KEY}"
    assert calls[0]["body"]["reasoning"] == {"effort": "high"}
    tools = calls[0]["body"]["tools"]
    assert tools[0] == {"type": "web_search"}
    assert [tool["name"] for tool in tools[1:]] == ["guide_plan"]
    assert calls[0]["body"]["max_tool_calls"] == 2
    assert "astra private reasoning" not in json.dumps(result.to_json(), ensure_ascii=False)


@pytest.mark.parametrize("terminal", ["response.completed", "response.incomplete"])
def test_rejected_astra_answer_still_settles_reported_usage(tmp_path: Path, terminal: str) -> None:
    ledger = ProbeLedger(tmp_path / "ledger.jsonl")
    events = [
        {"type": "response.output_item.done", "item": {
            "id": "ws_1", "type": "web_search_call", "status": "completed",
            "action": {"type": "search", "sources": [{"url": SEARCH_URL}]}}},
        {"type": terminal, "response": {
            "status": "completed" if terminal.endswith("completed") else "incomplete",
            "output": [], "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 2000, "output_tokens": 500, "total_tokens": 2500,
                      "input_tokens_details": {"cached_tokens": 400}}}},
    ]
    stream = "".join("data: " + json.dumps(event) + "\n\n" for event in events)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=stream,
                                            headers={"content-type": "text/event-stream"}))) as client:
            return await probe_astra(request=plan_request(), prepared_scene=make_test_jpeg(),
                                     ledger=ledger, provider=AstraProvider(client, api_key=OPENAI_KEY),
                                     allow_network=True)

    result = asyncio.run(exercise())
    assert not result.ok and result.outcome == "provider_error"
    assert result.total_cost_usd == pytest.approx(0.0554)
    assert result.observed_source_urls == [SEARCH_URL]
    account = ledger.snapshot()["providers"]["astra"]
    assert account["retained_usd"] == 0 and not account["blocked"]
    assert account["settled_usd"] == pytest.approx(result.total_cost_usd)


def test_astra_lane_refuses_a_missing_credential_before_any_request(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must never run
        calls.append({})
        raise AssertionError("no request may be sent without a credential")

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = AstraProvider(client)  # conftest points OPENAI_API_KEY_FILE at an absent file
            with pytest.raises(ProbeConfigError):
                await probe_astra(request=plan_request(), prepared_scene=make_test_jpeg(),
                                  ledger=ProbeLedger(tmp_path / "ledger.jsonl"), provider=provider,
                                  client=client, allow_network=True)

    asyncio.run(exercise())
    assert calls == []


def test_lane_functions_refuse_until_the_network_release(tmp_path: Path) -> None:
    ledger = ProbeLedger(tmp_path / "ledger.jsonl")

    async def exercise() -> None:
        with pytest.raises(ProbeConfigError):
            await probe_astra(request=plan_request(), prepared_scene=make_test_jpeg(), ledger=ledger)
        with pytest.raises(ProbeConfigError):
            await probe_deepseek(request=plan_request(), prepared_scene=make_test_jpeg(), ledger=ledger)

    asyncio.run(exercise())
    assert ledger.snapshot()["providers"] == {}


def test_entrypoint_refuses_until_the_network_release(tmp_path: Path) -> None:
    async def exercise() -> None:
        with pytest.raises(ProbeConfigError):
            await run_plan_probe(run_dir=tmp_path / "run", request=plan_request(), allow_network=False)

    asyncio.run(exercise())
    assert not (tmp_path / "run").exists()


def test_entrypoint_records_each_lane_independently(tmp_path: Path) -> None:
    """A lane that cannot run is recorded as its own failure; the other lane still completes."""
    run = tmp_path / "run"
    requests: list[dict[str, Any]] = []

    async def exercise() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_unused)) as astra_client:
            async with httpx.AsyncClient(transport=deepseek_transport(requests)) as deepseek_client:
                return await run_plan_probe(
                    run_dir=run, request=plan_request(), research=research, allow_network=True,
                    astra_provider=AstraProvider(astra_client),  # unconfigured in the test environment
                    deepseek_provider=DeepSeekProvider(deepseek_client, api_key=KEY))

    report = asyncio.run(exercise())
    assert report["results"]["astra"]["outcome"] == "config_error"
    assert report["results"]["astra"]["error_type"] == "ProbeConfigError"
    assert report["results"]["deepseek"]["ok"] is True
    assert report["ledger"]["providers"]["deepseek"]["attempts"] == 2
    assert "astra" not in report["ledger"]["providers"]


def handler_unused(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
    raise AssertionError("the unconfigured Astra lane must not open a socket")


def test_attempt_cap_ends_the_loop_without_inventing_a_plan(tmp_path: Path) -> None:
    run = tmp_path / "run"
    requests: list[dict[str, Any]] = []
    budgets = {"deepseek": BudgetLimits(max_attempts=1, max_usd=10.0, reserve_usd=0.06)}

    async def exercise() -> Any:
        async with httpx.AsyncClient(transport=deepseek_transport(requests)) as client:
            provider = DeepSeekProvider(client, api_key=KEY)
            return await run_plan_probe(run_dir=run, request=plan_request(), research=research,
                                        providers=("deepseek",), allow_network=True, budgets=budgets,
                                        deepseek_provider=provider)

    report = asyncio.run(exercise())
    result = report["results"]["deepseek"]
    assert result["outcome"] == "budget_refused"
    assert result["has_final_plan"] is False and result["final_plan"] is None
    assert len(result["attempts"]) == 1
    assert len(requests) == 1

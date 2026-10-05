"""The contextual temporal record (v2): scoping parity, observed-fact/continuity boundaries and hash binding.

The model's own yes/no/unsure output is not this module's behavior — ``contextual_clef_record`` only builds and
validates the student input. These tests pin that input's fail-closed boundaries (what a context may and may not
claim) and the identity hashes the review export and the fine-tuning labels carry.
"""

from __future__ import annotations

import pytest

from backend.app import clef_follow

from backend.app.temporal_follow import (
    MAX_CONTINUITY_INTERVALS,
    MAX_OBSERVED_FACTS,
    MAX_TEXT,
    TEMPORAL_VERSION,
    TemporalContextError,
    contextual_clef_record,
    student_record_sha256,
    temporal_context_sha256,
    validate_temporal_context,
)

PLAN_STEPS = [
    {"id": "s1", "say": "전원을 분리하세요.", "done_when": "어댑터가 분리되어 있다.", "check": "user",
     "required": True, "requires": []},
    {"id": "s2", "say": "초록 단자대를 끼우세요.", "done_when": "초록 단자대가 자리에 있다.", "check": "visual",
     "requires": ["s1"]},
    {"id": "s3", "say": "LED를 눌러 넣으세요.", "done_when": "LED가 단자대에 삽입되어 있다.", "check": "visual",
     "requires": ["s2"]},
]
USER_GOAL = "초록 단자대와 LED를 장착한다."
GOAL_WHEN = "초록 단자대와 LED가 계획 위치에 장착되어 있다."
#: The questions asked for ``current_step='s2'`` (classic scoping: the current step to the last, then goal).
KEYS = ("s2", "s3", "goal")


def _context(*, now: float = 10.0, facts=(), continuity=()) -> dict:
    return {"version": TEMPORAL_VERSION, "recording_id": "elk6q6as", "plan_revision": "led-plan-v3-atomic-1",
            "current_timestamp_s": now, "observed_facts": list(facts), "continuity": list(continuity)}


def _fact(**over) -> dict:
    fact = {"id": "f1", "key": "s2", "predicate": "초록 단자대가 계획의 장착 위치에 꽂혀 있다.", "verdict": "yes",
            "observed_at_s": 4.0, "source": "astra_generated", "source_ref": "video:00:04#frame",
            "evidence": "4초 프레임에서 단자대가 자리에 꽂혀 보인다."}
    fact.update(over)
    return fact


def _entry(**over) -> dict:
    entry = {"fact_id": "f1", "from_s": 4.0, "through_s": 9.0, "status": "observed_sequence",
             "source_ref": "video:00:04-00:09"}
    entry.update(over)
    return entry


def _record(context: dict | None = None, *, current_step: str = "s2", steps=PLAN_STEPS) -> dict:
    return contextual_clef_record(user_goal=USER_GOAL, steps=steps, goal_when=GOAL_WHEN, current_step=current_step,
                                  temporal_context=context if context is not None else _context())


def _validate(context: dict, keys=KEYS) -> None:
    validate_temporal_context(context, question_keys=keys)


def test_question_scope_starts_at_current_step() -> None:
    record = _record()
    assert list(record["questions"]) == ["s2", "s3", "goal"]


def test_a_step_missing_its_visible_condition_is_refused() -> None:
    with pytest.raises(ValueError, match="done_when"):
        _record(steps=[{"id": "s1", "say": "전원을 분리하세요."}])


@pytest.mark.parametrize("at", [10.0, 12.5])
def test_a_fact_at_or_after_the_current_photo_is_refused(at: float) -> None:
    with pytest.raises(TemporalContextError, match="not strictly before"):
        _validate(_context(facts=[_fact(observed_at_s=at)]))


def test_continuity_interval_is_capped_at_the_current_photo() -> None:
    _validate(_context(facts=[_fact()], continuity=[_entry(through_s=10.0)]))
    with pytest.raises(TemporalContextError, match="after current_timestamp_s"):
        _validate(_context(facts=[_fact()], continuity=[_entry(through_s=10.5)]))


def test_continuity_interval_must_start_at_or_after_its_fact() -> None:
    with pytest.raises(TemporalContextError, match="precedes its fact"):
        _validate(_context(facts=[_fact(observed_at_s=4.0)], continuity=[_entry(from_s=3.0)]))


def test_continuity_interval_must_link_an_observed_fact() -> None:
    with pytest.raises(TemporalContextError, match="links no observed fact"):
        _validate(_context(continuity=[_entry()]))


def test_continuity_interval_must_not_run_backwards() -> None:
    with pytest.raises(TemporalContextError, match="is after through_s"):
        _validate(_context(facts=[_fact()], continuity=[_entry(from_s=8.0, through_s=5.0)]))


def test_an_unknown_continuity_status_is_refused() -> None:
    # The v1 status is not silently accepted as a v2 persistence interval.
    with pytest.raises(TemporalContextError, match="status"):
        _validate(_context(facts=[_fact()], continuity=[_entry(status="confirmed_unchanged")]))


def test_an_unsure_is_not_an_observed_fact() -> None:
    with pytest.raises(TemporalContextError, match="verdict"):
        _validate(_context(facts=[_fact(verdict="unsure")]))


def test_an_unknown_fact_source_is_refused() -> None:
    with pytest.raises(TemporalContextError, match="source"):
        _validate(_context(facts=[_fact(source="unreviewed_model")]))
    with pytest.raises(TemporalContextError, match="source"):
        _validate(_context(facts=[_fact(source="verified_observation")]))


def test_keys_outside_the_asked_questions_are_refused() -> None:
    # s1 is earlier than the current step, so this frame's questions do not ask it.
    with pytest.raises(TemporalContextError, match="outside the asked question keys"):
        _validate(_context(facts=[_fact(key="s1")]))
    with pytest.raises(TemporalContextError, match="outside the asked question keys"):
        _validate(_context(facts=[_fact(key="s9")]))


def test_duplicate_fact_ids_are_refused() -> None:
    with pytest.raises(TemporalContextError, match="duplicate"):
        _validate(_context(facts=[_fact(), _fact(key="s3")]))


def test_a_second_active_fact_for_one_key_is_refused() -> None:
    with pytest.raises(TemporalContextError, match="at most one active fact per key"):
        _validate(_context(facts=[_fact(), _fact(id="f2", verdict="no")]))


def test_the_context_and_entry_key_sets_are_exact() -> None:
    extra = _context()
    extra["note"] = "not part of the contract"
    with pytest.raises(TemporalContextError, match="keys must be exactly"):
        _validate(extra)
    missing = _context()
    del missing["continuity"]
    with pytest.raises(TemporalContextError, match="keys must be exactly"):
        _validate(missing)
    with pytest.raises(TemporalContextError, match="keys must be exactly"):
        _validate(_context(facts=[{**_fact(), "confidence": 0.9}]))
    with pytest.raises(TemporalContextError, match="keys must be exactly"):
        _validate(_context(facts=[{k: v for k, v in _fact().items() if k != "evidence"}]))


def test_the_old_v1_context_is_refused() -> None:
    with pytest.raises(TemporalContextError, match="temporal-state-v2"):
        _validate({**_context(), "version": "temporal-state-v1"})


def test_the_context_is_size_capped() -> None:
    keys = tuple(f"k{n}" for n in range(MAX_OBSERVED_FACTS + 1))
    over_facts = [_fact(id=f"f{n}", key=f"k{n}") for n in range(MAX_OBSERVED_FACTS + 1)]
    with pytest.raises(TemporalContextError, match="over the cap"):
        _validate(_context(facts=over_facts), keys=keys)
    over_continuity = [_entry() for _ in range(MAX_CONTINUITY_INTERVALS + 1)]
    with pytest.raises(TemporalContextError, match="over the cap"):
        _validate(_context(facts=[_fact()], continuity=over_continuity))
    with pytest.raises(TemporalContextError, match="longer than the cap"):
        _validate(_context(facts=[_fact(predicate="가" * (MAX_TEXT + 1))]))


def test_the_factory_refuses_an_unusable_context() -> None:
    with pytest.raises(TemporalContextError):
        _record(_context(facts=[_fact(key="s1")]))


def test_the_context_hash_is_canonical_and_content_sensitive() -> None:
    context = _context(facts=[_fact()], continuity=[_entry()])
    reordered = {key: context[key] for key in reversed(list(context))}
    assert temporal_context_sha256(context) == temporal_context_sha256(reordered)
    assert temporal_context_sha256(context) != temporal_context_sha256(
        _context(facts=[_fact(verdict="no")], continuity=[_entry()]))


def test_the_record_hash_preserves_question_order() -> None:
    record = _record()
    reordered = {**record, "questions": {key: record["questions"][key]
                                         for key in reversed(list(record["questions"]))}}
    assert list(reordered["questions"]) != list(record["questions"])
    assert student_record_sha256(record) != student_record_sha256(reordered)
    assert student_record_sha256(record) != student_record_sha256({**record, "questions": {}})
    assert student_record_sha256(record) == student_record_sha256(_record())


def test_observation_rule_changes_do_not_reinterpret_frozen_temporal_facts(monkeypatch) -> None:
    context = _context(facts=[_fact()], continuity=[_entry(through_s=10.0)])
    before = _record(context)
    before_hash = student_record_sha256(before)
    monkeypatch.setattr(clef_follow, "CHOICE_RULES", "A different single-frame observation policy")
    monkeypatch.setattr(clef_follow, "STATE_CRITERIA",
                        {key: f"Changed observation criterion: {key}" for key in ("yes", "no", "unsure")})
    after = _record(context)
    assert student_record_sha256(after) == before_hash
    assert after == before

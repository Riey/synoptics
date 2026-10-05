"""Pure guide policies, ported from the demo browser client (aisw-hybrid-talk ``web/src``, commit 73ec3a8).

Every function here is a faithful port of one TS module; the file:line of the rule is cited next to it.
Times are milliseconds on the engine's own clock (the TS used ``performance.now()``/``Date.now()``).
Nothing here does I/O; ``real_engine.py`` drives it.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

from PIL import Image

INF = math.inf

# ====================================================================== triggers.ts (triggers + dispatch gate)


@dataclass(frozen=True)
class TriggerConfig:
    """triggers.ts:42-49 DEFAULT_TRIGGER_CONFIG."""

    moved_fraction: float = 0.2
    area_ratio: float = 1.5
    moved_sustain_ms: float = 300
    lost_ms: float = 1000
    heartbeat_local_ms: float = 4000
    heartbeat_remote_ms: float = 8000


DEFAULT_TRIGGER_CONFIG = TriggerConfig()


@dataclass(frozen=True)
class TrackObservation:
    run_id: str
    track_id: str
    generation: int
    state: str
    box: Any  # contracts.Box | None


@dataclass(frozen=True, eq=False)
class AcceptedBox:
    """The box of the last accepted answer (triggers.ts:61). Compared by identity, like the TS reference."""

    run_id: str
    generation: int
    box: Any


@dataclass(frozen=True)
class TriggerState:
    run_id: str | None = None
    fence_key: str | None = None
    acquired_keys: tuple[str, ...] = ()
    moved_since: float | None = None
    moved_fired: bool = False
    moved_reference: AcceptedBox | None = None
    lost_since: float | None = None
    lost_fired: bool = False
    heartbeat_from: float | None = None


INITIAL_TRIGGER_STATE = TriggerState()


def _centre(box) -> tuple[float, float]:
    return box.x + box.width / 2, box.y + box.height / 2


def has_moved(box, reference, config: TriggerConfig = DEFAULT_TRIGGER_CONFIG, frame_aspect: float = 1.0) -> bool:
    """triggers.ts:116-132: centre shift in frame-WIDTH units, or area ratio."""
    ax, ay = _centre(box)
    bx, by = _centre(reference)
    aspect = frame_aspect if math.isfinite(frame_aspect) and frame_aspect > 0 else 1.0
    if math.hypot(ax - bx, (ay - by) * aspect) > config.moved_fraction:
        return True
    area_a = box.width * box.height
    area_b = reference.width * reference.height
    if area_a <= 0 or area_b <= 0:
        return area_a != area_b
    ratio = area_a / area_b if area_a > area_b else area_b / area_a
    return ratio > config.area_ratio


def step_triggers(
    previous: TriggerState,
    *,
    observation: TrackObservation | None,
    now_ms: float,
    fence_key: str,
    accepted: AcceptedBox | None,
    follow_provider: str | None,
    frame_aspect: float = 1.0,
    config: TriggerConfig = DEFAULT_TRIGGER_CONFIG,
) -> tuple[TriggerState, list[str]]:
    """triggers.ts:134-223 ``stepTriggers``."""
    obs = observation
    now = now_ms
    fired: list[str] = []
    state = previous
    # triggers.ts:143-152 a new run or a fence change cancels every timer; acquisition keys are per run.
    run_id = obs.run_id if obs else None
    if run_id != state.run_id or fence_key != state.fence_key:
        state = replace(
            INITIAL_TRIGGER_STATE,
            run_id=run_id,
            fence_key=fence_key,
            acquired_keys=state.acquired_keys if (run_id is not None and run_id == state.run_id) else (),
        )
    tracking = obs is not None and obs.state == "tracking" and obs.box is not None

    # triggers.ts:156-163 acquired: once per (run, generation).
    if obs and tracking:
        key = f"{obs.run_id}|{obs.generation}"
        if key not in state.acquired_keys:
            state = replace(state, acquired_keys=state.acquired_keys + (key,))
            fired.append("acquired")

    # triggers.ts:165-178 anchor_lost: lost sustained; any other state cancels, re-entry re-arms.
    if obs is not None and obs.state == "lost":
        if state.lost_since is None:
            state = replace(state, lost_since=now, lost_fired=False)
        if not state.lost_fired and now - state.lost_since >= config.lost_ms:
            state = replace(state, lost_fired=True)
            fired.append("anchor_lost")
    else:
        state = replace(state, lost_since=None, lost_fired=False)

    # triggers.ts:180-204 target_moved: against the last accepted answer's box on the same anchor identity.
    reference = accepted
    if reference is not state.moved_reference:
        state = replace(state, moved_reference=reference, moved_since=None, moved_fired=False)
    if (
        obs
        and tracking
        and reference is not None
        and reference.run_id == obs.run_id
        and reference.generation == obs.generation
        and has_moved(obs.box, reference.box, config, frame_aspect)
    ):
        if state.moved_since is None:
            state = replace(state, moved_since=now)
        if not state.moved_fired and now - state.moved_since >= config.moved_sustain_ms:
            state = replace(state, moved_fired=True)
            fired.append("target_moved")
    else:
        state = replace(state, moved_since=None, moved_fired=False)

    # triggers.ts:206-220 heartbeat: while tracking, from tracking start / the last follow dispatch.
    if tracking:
        if state.heartbeat_from is None:
            state = replace(state, heartbeat_from=now)
        elif not fired:
            interval = config.heartbeat_local_ms if follow_provider in ("local", "clef") else config.heartbeat_remote_ms
            if now - state.heartbeat_from >= interval:
                state = replace(state, heartbeat_from=now)
                fired.append("heartbeat")
    else:
        state = replace(state, heartbeat_from=None)
    return state, fired


def note_stage_dispatch(state: TriggerState, stage: str, now_ms: float) -> TriggerState:
    """triggers.ts:226-237: only a FOLLOW restarts the heartbeat interval."""
    if stage != "follow" or state.heartbeat_from is None:
        return state
    return replace(state, heartbeat_from=now_ms)


@dataclass(eq=False)
class PendingCall:
    """Mutable on purpose: a dispatched confirm drops its held follow frame in place (useIntentLoop.ts:850)."""

    kind: Literal["follow", "confirm"]
    trigger: str
    lane: Literal["local", "remote"]
    meta: dict | None = None


@dataclass(frozen=True)
class GateConfig:
    """useIntentLoop.ts:163-174 GATE_CONFIG (+ dispatchFloor.ts:422 LIVE_MIN_DISPATCH_INTERVAL_MS = 2200)."""

    follow_local_floor_ms: float = 1000
    follow_remote_floor_ms: float = 2200
    confirm_floor_ms: float = 2200
    remote_spacing_ms: float = 1200
    remote_budget: int = 30
    remote_window_ms: float = 120_000
    remote_per_minute: int = 20


GATE_CONFIG = GateConfig()


@dataclass(frozen=True)
class StageSlot:
    in_flight: PendingCall | None = None
    pending: PendingCall | None = None
    last_at: float = -INF


@dataclass(frozen=True)
class GateState:
    follow: StageSlot = StageSlot()
    confirm: StageSlot = StageSlot()
    plan_in_flight: bool = False
    last_remote_at: float = -INF
    remote_times: tuple[float, ...] = ()

    def slot(self, stage: str) -> StageSlot:
        return self.follow if stage == "follow" else self.confirm


INITIAL_GATE_STATE = GateState()
GATE_STAGES = ("follow", "confirm")


def call_priority(call: PendingCall) -> int:
    """triggers.ts:300-307: recheck 4 > step_done/replan/unsure_twice 3 > goal_check 2; follow: heartbeat 1, else 2."""
    if call.kind == "confirm":
        if call.meta and call.meta.get("recheck"):
            return 4
        if call.trigger == "goal_check":
            return 2
        return 3
    return 1 if call.trigger == "heartbeat" else 2


def _with_slot(state: GateState, stage: str, slot: StageSlot) -> GateState:
    return replace(state, follow=slot) if stage == "follow" else replace(state, confirm=slot)


def enqueue(state: GateState, call: PendingCall) -> GateState:
    """triggers.ts:314-318: one pending slot per stage, newest wins unless a more important call holds it."""
    slot = state.slot(call.kind)
    if slot.pending is not None and call_priority(slot.pending) > call_priority(call):
        return state
    return _with_slot(state, call.kind, replace(slot, pending=call))


@dataclass(frozen=True)
class GateDecision:
    kind: Literal["idle", "busy", "wait", "capped", "dispatch"]
    wait_ms: float = 0
    limit: str | None = None
    call: PendingCall | None = None


def _recent(times: Sequence[float], now_ms: float, window_ms: float) -> list[float]:
    return [at for at in times if at > now_ms - window_ms]


def _stage_floor(call: PendingCall, config: GateConfig) -> float:
    if call.kind == "confirm":
        return config.confirm_floor_ms
    return config.follow_local_floor_ms if call.lane == "local" else config.follow_remote_floor_ms


def decide_dispatch(state: GateState, stage: str, now_ms: float, config: GateConfig = GATE_CONFIG,
                    remote_held: bool = False) -> GateDecision:
    """triggers.ts:343-369 ``decideDispatch``."""
    slot = state.slot(stage)
    call = slot.pending
    if call is None:
        return GateDecision("idle")
    if slot.in_flight is not None or state.plan_in_flight:
        return GateDecision("busy")
    if remote_held and call.lane == "remote":
        return GateDecision("busy")
    if call.lane == "remote":
        in_budget = _recent(state.remote_times, now_ms, config.remote_window_ms)
        if len(in_budget) >= config.remote_budget:
            return GateDecision("capped", max(1, in_budget[0] + config.remote_window_ms - now_ms), "budget")
        in_minute = _recent(state.remote_times, now_ms, 60_000)
        if len(in_minute) >= config.remote_per_minute:
            return GateDecision("capped", max(1, in_minute[0] + 60_000 - now_ms), "per_minute")
    wait = slot.last_at + _stage_floor(call, config) - now_ms
    if call.lane == "remote":
        wait = max(wait, state.last_remote_at + config.remote_spacing_ms - now_ms)
    if wait > 0:
        return GateDecision("wait", wait)
    return GateDecision("dispatch", call=call)


def _note_remote(state: GateState, now_ms: float, config: GateConfig | None) -> GateState:
    keep = max(config.remote_window_ms, 60_000) if config else INF
    return replace(state, last_remote_at=now_ms, remote_times=tuple(_recent(state.remote_times, now_ms, keep)) + (now_ms,))


def mark_dispatched(state: GateState, call: PendingCall, now_ms: float, config: GateConfig | None = GATE_CONFIG) -> GateState:
    """triggers.ts:376-385."""
    slot = state.slot(call.kind)
    nxt = _with_slot(state, call.kind, StageSlot(in_flight=call, pending=None if slot.pending is call else slot.pending,
                                                 last_at=now_ms))
    if call.lane == "remote":
        nxt = _note_remote(nxt, now_ms, config)
    return nxt


def mark_settled(state: GateState, stage: str) -> GateState:
    return _with_slot(state, stage, replace(state.slot(stage), in_flight=None))


def mark_plan(state: GateState, in_flight: bool, now_ms: float, config: GateConfig | None = GATE_CONFIG) -> GateState:
    """triggers.ts:392-395: the plan is exclusive and counts as a DeepSeek-backed call."""
    nxt = replace(state, plan_in_flight=in_flight)
    return _note_remote(nxt, now_ms, config) if in_flight else nxt


def mark_talk(state: GateState, now_ms: float, config: GateConfig | None = GATE_CONFIG) -> GateState:
    """triggers.ts:401-403: a talk is DeepSeek-backed and restarts the confirm floor (without the confirm slot)."""
    return _note_remote(_with_slot(state, "confirm", replace(state.confirm, last_at=now_ms)), now_ms, config)


def remote_calls_in_window(state: GateState, now_ms: float, window_ms: float) -> int:
    return len(_recent(state.remote_times, now_ms, window_ms))


# ====================================================================== followStep.ts


def _last_yes_index(steps, checks, from_index: int) -> int:
    """followStep.ts:36-44."""
    ids = [s.id for s in steps]
    last = -1
    for check in checks:
        if check.get("visible") != "yes":
            continue
        index = ids.index(check["step_id"]) if check.get("step_id") in ids else -1
        if index >= from_index and index > last:
            last = index
    return last


def current_step_visible(steps, asked_index: int, checks) -> str | None:
    """followStep.ts:47-51."""
    if not (0 <= asked_index < len(steps)):
        return None
    asked = steps[asked_index].id
    for check in checks:
        if check.get("step_id") == asked:
            return check.get("visible")
    return None


@dataclass(frozen=True)
class FollowStepDecision:
    kind: Literal["none", "pendingSkip", "advance", "skip"]
    from_index: int = 0
    to_index: int = 0
    step_id: str = ""
    skipped_ids: tuple[str, ...] = ()
    last_yes_index: int = -1


def decide_follow_step(steps, asked_index: int, current_index: int, checks, previous_checks) -> FollowStepDecision:
    """followStep.ts:53-77: asked `yes` advances; a later `yes` skips only on its second sighting."""
    if not (0 <= asked_index < len(steps)) or current_index != asked_index:
        return FollowStepDecision("none")
    last_yes = _last_yes_index(steps, checks, asked_index)
    if last_yes < 0:
        return FollowStepDecision("none")
    target = min(last_yes + 1, len(steps) - 1)
    if last_yes == asked_index:
        if target == asked_index:
            return FollowStepDecision("none")
        return FollowStepDecision("advance", asked_index, target, steps[asked_index].id)
    seen_before = previous_checks is not None and _last_yes_index(steps, previous_checks, last_yes) >= last_yes
    if not seen_before:
        return FollowStepDecision("pendingSkip", last_yes_index=last_yes)
    visible = {c.get("step_id"): c.get("visible") for c in checks}
    skipped = tuple(s.id for s in steps[asked_index:last_yes] if visible.get(s.id) != "yes")
    return FollowStepDecision("skip", asked_index, target, steps[last_yes].id, skipped)


# ====================================================================== planProgress.ts


def add_skipped(skipped: Sequence[str], ids: Sequence[str]) -> list[str]:
    """planProgress.ts:355-357 (order-preserving set union)."""
    return list(dict.fromkeys([*skipped, *ids]))


def skipped_before(steps, skipped: Sequence[str], index: int) -> list[str]:
    """planProgress.ts:360-363."""
    before = {s.id for s in steps[: max(0, index)]}
    return [i for i in skipped if i in before]


# ====================================================================== confirmDecision.ts


def confirm_frame_source(trigger: str, meta: dict, *, plan_id, plan_revision, run_id) -> tuple[str, Any]:
    """confirmDecision.ts:137-148: a follow-raised step_done/goal_check re-uses the follow's own frame."""
    frame = meta.get("follow_frame")
    if frame is None or meta.get("recheck") or trigger not in ("step_done", "goal_check"):
        return "fresh", None
    if frame.plan_id != plan_id or frame.plan_revision != plan_revision or frame.run_id != run_id:
        return "drop", None
    return "follow", frame


@dataclass(frozen=True)
class ConfirmDecision:
    completion: Literal["start_check", "confirmed", "recheck_failed", "keep"]
    revert_to: int | None
    lost_notice: bool


def decide_confirm(answer: dict, meta: dict, *, step_index: int, completion: str, track_lost: bool,
                   user_done_ids: Sequence[str] = ()) -> ConfirmDecision:
    """confirmDecision.ts:175-210: step and completion decided apart."""
    satisfied = answer.get("goal_status", {}).get("status") == "visually_satisfied"
    if meta.get("recheck"):
        return ConfirmDecision("confirmed" if satisfied and completion == "checking" else "recheck_failed", None, False)
    comp = "start_check" if satisfied and completion == "none" else "keep"
    revert_to = None
    fi, ti, sid = meta.get("from_index"), meta.get("to_index"), meta.get("step_id")
    if (
        answer.get("trigger") == "step_done"
        and isinstance(fi, int)
        and isinstance(ti, int)
        and isinstance(sid, str)
        and ti != fi
        and answer.get("step_id") == sid
    ):
        if answer.get("step_check") == "no" and step_index == ti and sid not in user_done_ids:
            revert_to = fi
    return ConfirmDecision(comp, revert_to, answer.get("trigger") == "goal_check" and not satisfied and track_lost)


def retry_recheck_after_stale(meta: dict | None, completion: str) -> bool:
    """confirmDecision.ts:217-219."""
    return bool(meta and meta.get("recheck")) and completion == "checking"


# ====================================================================== replanPolicy.ts

STUCK_TARGET_CHANGED_FOLLOWS = 3
REPLAN_MAX_PER_RUN = 3
REPLAN_MIN_GAP_MS = 15_000


def next_stuck_count(count: int, *, trigger: str, step_changed: bool) -> tuple[int, bool]:
    """replanPolicy.ts:243-251."""
    if step_changed:
        return 0, False
    if trigger != "target_changed":
        return count, False
    nxt = count + 1
    return (0, True) if nxt >= STUCK_TARGET_CHANGED_FOLLOWS else (nxt, False)


@dataclass(frozen=True)
class ReplanBudget:
    used: int = 0
    reserved: int = 0
    last_plan_at: float = 0.0


def fresh_replan_budget(plan_at: float) -> ReplanBudget:
    return ReplanBudget(0, 0, plan_at)


def replan_block(budget: ReplanBudget, now_ms: float) -> tuple[str | None, float]:
    """replanPolicy.ts:269-273."""
    if budget.used + budget.reserved >= REPLAN_MAX_PER_RUN:
        return "max", 0
    wait = budget.last_plan_at + REPLAN_MIN_GAP_MS - now_ms
    return ("cooldown", wait) if wait > 0 else (None, 0)


def spend_replan(budget: ReplanBudget, now_ms: float) -> ReplanBudget:
    return replace(budget, used=budget.used + 1, last_plan_at=now_ms)


def reserve_replan(budget: ReplanBudget) -> tuple[ReplanBudget, bool]:
    """replanPolicy.ts:281-285."""
    if budget.used + budget.reserved >= REPLAN_MAX_PER_RUN:
        return budget, False
    return replace(budget, reserved=budget.reserved + 1), True


def release_replan(budget: ReplanBudget) -> ReplanBudget:
    return replace(budget, reserved=max(0, budget.reserved - 1))


def replan_dispatch(budget: ReplanBudget, asked_epoch: int | None, current_epoch: int) -> str:
    """replanPolicy.ts:296-299."""
    if asked_epoch is not None and asked_epoch != current_epoch:
        return "drop"
    return "drop" if budget.used + budget.reserved > REPLAN_MAX_PER_RUN else "send"


def note_plan_installed(budget: ReplanBudget, now_ms: float) -> ReplanBudget:
    return replace(budget, last_plan_at=max(budget.last_plan_at, now_ms))


def describe_replan_block(block: str | None) -> str | None:
    """replanPolicy.ts:307-311."""
    if block == "max":
        return f"이번 가이드에서는 계획을 더 다시 짤 수 없습니다(최대 {REPLAN_MAX_PER_RUN}회)."
    if block == "cooldown":
        return f"계획을 방금 짰습니다. {REPLAN_MIN_GAP_MS // 1000}초가 지나면 다시 짤 수 있습니다."
    return None


# ====================================================================== talkPolicy.ts


def decide_talk(*, running: bool, talk_pending: bool, gate: GateState, now_ms: float,
                config: GateConfig = GATE_CONFIG) -> tuple[str | None, float]:
    """talkPolicy.ts:29-39: one at a time, shared budget; a floor/spacing only delays."""
    if not running:
        return "not_running", 0
    if talk_pending:
        return "pending", 0
    if remote_calls_in_window(gate, now_ms, config.remote_window_ms) >= config.remote_budget:
        return "budget", 0
    if remote_calls_in_window(gate, now_ms, 60_000) >= config.remote_per_minute:
        return "per_minute", 0
    wait = max(0, gate.confirm.last_at + config.confirm_floor_ms - now_ms,
               gate.last_remote_at + config.remote_spacing_ms - now_ms)
    return None, wait


def describe_talk_block(block: str | None, config: GateConfig = GATE_CONFIG) -> str | None:
    """talkPolicy.ts:41-54."""
    if block == "not_running":
        return "가이드가 진행 중일 때만 말할 수 있습니다."
    if block == "pending":
        return "앞의 말에 대한 답을 기다리는 중입니다."
    if block == "budget":
        return (f"DeepSeek 호출 한도({round(config.remote_window_ms / 60_000)}분에 {config.remote_budget}회)에 "
                "도달했습니다. 잠시 후 다시 말해 주세요.")
    if block == "per_minute":
        return f"서버 호출 한도(1분에 {config.remote_per_minute}회)에 도달했습니다. 잠시 후 다시 말해 주세요."
    return None


def describe_talk_error(status: int | None, code: str | None = None, reason: str | None = None) -> str:
    """talkPolicy.ts:57-67 (``status is None``: no HTTP answer at all)."""
    if status is None:
        return "안내 서버에 연결하지 못했습니다. 다시 말해 주세요."
    if status == 409:
        return "그사이 계획이 바뀌어 답을 적용하지 못했습니다. 다시 말해 주세요."
    if status == 429 or code == "provider_busy":
        return "안내 서버가 바쁩니다. 잠시 후 다시 말해 주세요."
    if code == "service_unavailable":
        return "대화 경로를 사용할 수 없습니다. DeepSeek 설정을 확인해주세요."
    if code == "provider_timeout":
        return "답이 늦어 취소됐습니다. 다시 말해 주세요."
    if code == "invalid_provider_output":
        return f"답을 이해하지 못했습니다{f'({reason})' if reason else ''}. 다른 말로 다시 말해 주세요."
    return f"말을 보내지 못했습니다({status} {code})."


def is_superseded_stale(body: dict, *, plan_id, plan_revision, talk_pending: bool) -> bool:
    """talkPolicy.ts:69-77: a 409 stale_plan naming our plan at our (or a talk's newer) revision is moot."""
    if not isinstance(body.get("plan_id"), str) or body.get("plan_id") != plan_id:
        return False
    rev = body.get("plan_revision")
    if not isinstance(rev, int) or isinstance(rev, bool) or plan_revision is None:
        return False
    if rev == plan_revision:
        return True
    return talk_pending and rev > plan_revision


def talk_send_step(accepted: dict, *, plan_id, steps) -> str | None:
    """talkPolicy.ts:85-92."""
    if plan_id != accepted["plan_id"]:
        return None
    for step in steps:
        if step.id == accepted["step_id"]:
            return step.id if step.done_when == accepted["done_when"] else None
    return None


# ====================================================================== acceptance.ts


@dataclass(frozen=True)
class SentStamp:
    session_id: str
    plan_id: str
    plan_revision: int
    intent_seq: int
    fence: dict  # {task_epoch, run_id}
    anchors: tuple[dict, ...]  # AnchorEcho dicts


def _same_anchors(a: Sequence[dict], b: Sequence[dict]) -> bool:
    if len(a) != len(b):
        return False
    return all(x.get("anchor_id") == y.get("anchor_id") and x.get("track_id") == y.get("track_id")
               and x.get("generation") == y.get("generation") for x, y in zip(a, b))


def classify_answer(sent: SentStamp, answer: dict, *, session_id, task_epoch, run_id, plan_id, plan_revision,
                    latest_intent_seq: int, anchor: dict | None, replan_applied: bool = False) -> str:
    """acceptance.ts:166-203: bound / text_only / rejected."""
    if answer.get("intent_seq") != sent.intent_seq:
        return "rejected"
    fe = answer.get("fence_echo") or {}
    if fe.get("task_epoch") != sent.fence["task_epoch"] or fe.get("run_id") != sent.fence["run_id"]:
        return "rejected"
    if not _same_anchors(answer.get("anchors_echo") or [], sent.anchors):
        return "rejected"
    if session_id is None or session_id != sent.session_id:
        return "rejected"
    if task_epoch is None or task_epoch != sent.fence["task_epoch"]:
        return "rejected"
    if run_id is None or run_id != sent.fence["run_id"]:
        return "rejected"
    if plan_id is None or plan_id != sent.plan_id:
        return "rejected"
    if plan_revision is None or plan_revision != sent.plan_revision:
        return "rejected"
    rev = answer.get("plan_revision")
    revision_ok = (isinstance(rev, int) and rev > sent.plan_revision) if replan_applied else rev == sent.plan_revision
    if not revision_ok:
        return "rejected"
    if sent.intent_seq != latest_intent_seq:
        return "rejected"
    if not sent.anchors:
        return "bound" if anchor is None else "text_only"
    s = sent.anchors[0]
    if anchor and anchor["anchor_id"] == s["anchor_id"] and anchor["track_id"] == s["track_id"] \
            and anchor["generation"] == s["generation"]:
        return "bound"
    return "text_only"


def capture_age_seconds(captured_at_ms: float, now_ms: float) -> int:
    """acceptance.ts:206-209."""
    if not (math.isfinite(captured_at_ms) and math.isfinite(now_ms)):
        return 0
    return max(0, round((now_ms - captured_at_ms) / 1000))


# ====================================================================== applyTalk.ts

TALK_REPLAN_NOTICE = "말씀대로 계획을 다시 짰습니다. 새 1단계부터 안내합니다."
_TALK_ACTION_ORDER = ("step_mark", "go_to", "replan", "target", "step_say")


def talk_actions(answer: dict) -> list[str]:
    return [name for name in _TALK_ACTION_ORDER if answer.get(name)]


def talk_changes_plan(answer: dict) -> bool:
    """applyTalk.ts:300-302."""
    return bool(answer.get("step_say") or answer.get("replan"))


@dataclass(frozen=True)
class TalkRunState:
    steps: tuple  # GuideStep models
    step_index: int
    plan_revision: int
    target: str
    skipped: tuple[str, ...]
    user_done: tuple[str, ...]
    unsure_streak: int
    stuck_count: int
    last_checks: tuple | None
    replan_budget: ReplanBudget


@dataclass(frozen=True)
class TalkOutcome:
    state: TalkRunState
    effects: tuple[tuple[str, str | None], ...]  # ("retarget", noun) | ("checkGoal", None) | ("notice", text)
    actions: tuple[str, ...]
    applied: bool


def _move_to(state: TalkRunState, index: int) -> TalkRunState:
    return replace(state, step_index=index, unsure_streak=0, stuck_count=0, last_checks=None)


def _apply_mark(state: TalkRunState, sent_step_id: str, mark: str, effects: list) -> TalkRunState:
    """applyTalk.ts:309-322."""
    index = state.step_index
    if not (0 <= index < len(state.steps)) or state.steps[index].id != sent_step_id:
        return replace(state, user_done=tuple(add_skipped(state.user_done, [sent_step_id]))) if mark == "done" else state
    last = index >= len(state.steps) - 1
    if mark == "done":
        marked = replace(state, user_done=tuple(add_skipped(state.user_done, [sent_step_id])),
                         skipped=tuple(i for i in state.skipped if i != sent_step_id))
        if last:
            effects.append(("checkGoal", None))
        return marked if last else _move_to(marked, index + 1)
    marked = replace(state, skipped=tuple(add_skipped(state.skipped, [sent_step_id])),
                     user_done=tuple(i for i in state.user_done if i != sent_step_id))
    return marked if last else _move_to(marked, index + 1)


def _apply_go_to(state: TalkRunState, step_id: str) -> TalkRunState:
    """applyTalk.ts:325-332."""
    ids = [s.id for s in state.steps]
    index = ids.index(step_id) if step_id in ids else -1
    if index < 0 or index >= state.step_index:
        return state
    return _move_to(replace(state, skipped=tuple(skipped_before(state.steps, state.skipped, index)),
                            user_done=tuple(skipped_before(state.steps, state.user_done, index))), index)


def apply_talk(state: TalkRunState, sent_step_id: str, answer: dict, acceptance: str, now_ms: float,
               parse_steps) -> TalkOutcome:
    """applyTalk.ts:339-367: plan text, then position, then target. ``parse_steps`` turns wire steps into models."""
    actions = tuple(talk_actions(answer))
    if acceptance == "rejected":
        return TalkOutcome(state, (), actions, False)
    effects: list = []
    nxt = state
    if answer.get("replan"):
        budget = note_plan_installed(replace(nxt.replan_budget, used=nxt.replan_budget.used + 1), now_ms)
        nxt = replace(_move_to(nxt, 0), steps=tuple(parse_steps(answer["replan"]["steps"])),
                      plan_revision=answer["plan_revision"], skipped=(), user_done=(), replan_budget=budget)
        effects.append(("notice", TALK_REPLAN_NOTICE))
    elif answer.get("step_say"):
        say = answer["step_say"]
        nxt = replace(nxt, steps=tuple(s.model_copy(update={"say": say}) if s.id == sent_step_id else s
                                       for s in nxt.steps), plan_revision=answer["plan_revision"])
    if answer.get("step_mark"):
        nxt = _apply_mark(nxt, sent_step_id, answer["step_mark"], effects)
    elif answer.get("go_to"):
        nxt = _apply_go_to(nxt, answer["go_to"])
    if answer.get("target"):
        nxt = replace(nxt, target=answer["target"])
        effects.append(("retarget", answer["target"]))
    return TalkOutcome(nxt, tuple(effects), actions, True)


# ====================================================================== noticePolicy.ts

NOTICE_TTL_MS = 8000


def notice_expired(stamp_step: int, stamp_at: float, step_index: int, now_ms: float) -> bool:
    """noticePolicy.ts:383-385: gone on a step change or after 8 s."""
    return step_index != stamp_step or now_ms - stamp_at >= NOTICE_TTL_MS


# ====================================================================== anchorAppearance.ts (+ anchorSampler.ts)


@dataclass(frozen=True)
class AppearanceConfig:
    """anchorAppearance.ts:56-65 (threshold 0.06 measured 2026-10-01)."""

    grid: int = 24
    pad: float = 0.15
    sample_ms: float = 250
    threshold: float = 0.06
    sustain: int = 2
    patch_cells: int = 6


DEFAULT_APPEARANCE_CONFIG = AppearanceConfig()


def appearance_crop(box, width: int, height: int, pad: float) -> tuple[int, int, int, int] | None:
    """anchorAppearance.ts:80-94: the padded box in source pixels (x, y, w, h), or None when < 2 px."""
    if box.width <= 0 or box.height <= 0 or width <= 0 or height <= 0:
        return None
    p = pad if math.isfinite(pad) and pad > 0 else 0.0
    clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    left = clamp(box.x - box.width * p) * width
    top = clamp(box.y - box.height * p) * height
    right = clamp(box.x + box.width * (1 + p)) * width
    bottom = clamp(box.y + box.height * (1 + p)) * height
    x, y = math.floor(left), math.floor(top)
    w = min(width, math.ceil(right)) - x
    h = min(height, math.ceil(bottom)) - y
    if w < 2 or h < 2:
        return None
    return x, y, w, h


def sample_anchor_grid(jpeg: bytes, box, config: AppearanceConfig = DEFAULT_APPEARANCE_CONFIG) -> list[float] | None:
    """anchorSampler.ts: crop to the padded box and area-average to grid×grid BT.601 luma (0..1).

    The browser used a 2×-per-step canvas ladder; Pillow's BOX filter is a single exact area average, which the
    ladder approximates. ``convert("L")`` is ITU-R 601-2 luma, the same weights as ``lumaFromRgba``.
    """
    try:
        with Image.open(io.BytesIO(jpeg)) as img:
            crop = appearance_crop(box, img.width, img.height, config.pad)
            if crop is None:
                return None
            x, y, w, h = crop
            small = img.convert("L").crop((x, y, x + w, y + h)).resize((config.grid, config.grid), Image.BOX)
            return [v / 255 for v in small.tobytes()]
    except Exception:
        return None


@dataclass(frozen=True)
class AppearanceScore:
    mean: float
    patch: float
    shift: float


def compare_appearance(reference: Sequence[float], current: Sequence[float], patch_cells: int = 6) -> AppearanceScore:
    """anchorAppearance.ts:133-171: mean residual after removing the brightness shift."""
    n = len(reference)
    if n == 0 or n != len(current):
        return AppearanceScore(1, 1, 0)
    shift = (sum(current) - sum(reference)) / n
    residual = [abs(c - r - shift) for r, c in zip(reference, current)]
    side = round(math.sqrt(n))
    patch = 0.0
    if side * side == n and 0 < patch_cells <= side:
        blocks = side // patch_cells
        for by in range(blocks):
            for bx in range(blocks):
                s = 0.0
                for yy in range(patch_cells):
                    row = (by * patch_cells + yy) * side + bx * patch_cells
                    s += sum(residual[row: row + patch_cells])
                patch = max(patch, s / (patch_cells * patch_cells))
    return AppearanceScore(sum(residual) / n, patch, shift)


@dataclass(frozen=True, eq=False)
class AppearanceReference:
    run_id: str
    track_id: str
    generation: int
    grid: tuple[float, ...]


@dataclass(frozen=True)
class AppearanceState:
    reference: AppearanceReference | None = None
    streak: int = 0
    fired: bool = False
    last_sample_at: float | None = None
    last_score: AppearanceScore | None = None


INITIAL_APPEARANCE_STATE = AppearanceState()


def appearance_sample_due(state: AppearanceState, now_ms: float, config: AppearanceConfig = DEFAULT_APPEARANCE_CONFIG) -> bool:
    return state.last_sample_at is None or now_ms - state.last_sample_at >= config.sample_ms


def step_appearance(previous: AppearanceState, *, now_ms: float, anchor: tuple[str, str, int] | None,
                    grid: Sequence[float] | None, reference: AppearanceReference | None,
                    config: AppearanceConfig = DEFAULT_APPEARANCE_CONFIG) -> tuple[AppearanceState, bool]:
    """anchorAppearance.ts:226-249: fires after ``sustain`` samples above threshold, once per excursion."""
    state = replace(previous, last_sample_at=now_ms)
    if reference is not state.reference:
        state = replace(state, reference=reference, streak=0, fired=False, last_score=None)
    ref = state.reference
    if ref is None or anchor is None or grid is None or (ref.run_id, ref.track_id, ref.generation) != anchor:
        return replace(state, streak=0, last_score=None), False
    score = compare_appearance(ref.grid, grid, config.patch_cells)
    if score.mean <= config.threshold:
        return replace(state, streak=0, fired=False, last_score=score), False
    streak = state.streak + 1
    if not state.fired and streak >= config.sustain:
        return replace(state, streak=streak, fired=True, last_score=score), True
    return replace(state, streak=streak, last_score=score), False


# ====================================================================== speechPolicy.ts

COMPLETION_SPEECH = "목표를 달성한 것으로 보입니다. 완료 확인을 눌러 주세요."
USER_CONFIRMED_SPEECH = "완료를 확인했습니다. 수고하셨습니다."
RESELECT_SPEECH = "대상이 다릅니다. 대상을 다시 지정하세요."
PLANNING_SPEECH = "계획을 세우는 중입니다."
#: §15.1/§15.9: the draft is ready to review (the mock engine says the same line as ``DRAFT_SAY``).
DRAFT_SAY = "계획 초안을 만들었어요. 검토하고 실행을 눌러 주세요."
TALK_DETAIL_SPEECH = "자세한 건 화면에 있어요."


def talk_speech(reply: str, spoken: str) -> str:
    """speechPolicy.ts:256-259."""
    s = spoken.strip() or reply.strip()
    return s if reply.strip() == s else f"{s} {TALK_DETAIL_SPEECH}"


@dataclass(frozen=True)
class SpeechView:
    """The parts of the view speechPolicy reads (speechPolicy.ts:262-305)."""

    phase: str = "idle"
    step: str | None = None  # stepSpeech(view)
    talk_at: int | None = None
    talk_text: str | None = None
    completion: str = "none"
    needs_reselect: bool = False
    notice: str | None = None
    error: str | None = None


def step_speech(phase: str, steps, step_index: int, partial_first_say: str | None) -> str | None:
    """speechPolicy.ts:262-268."""
    if phase not in ("running", "planning"):
        return None
    say = steps[step_index].say if steps and 0 <= step_index < len(steps) else partial_first_say
    if not say:
        return None
    total = f" / {len(steps)}" if steps else ""
    return f"{step_index + 1}단계{total}. {say}"


def spoken_lines(previous: SpeechView | None, nxt: SpeechView) -> list[tuple[str, str]]:
    """speechPolicy.ts:271-305: (text, "replace"|"append") for the transition previous -> next."""
    lines: list[tuple[str, str]] = []
    step = nxt.step
    if step is not None and nxt.phase == "running" and (previous is None or previous.step != step
                                                       or previous.phase != "running"):
        lines.append((step, "replace"))
    if nxt.phase == "planning" and (previous is None or previous.phase != "planning"):
        lines.append((PLANNING_SPEECH, "replace"))
    if nxt.phase == "reviewing" and (previous is None or previous.phase != "reviewing"):
        lines.append((DRAFT_SAY, "replace"))
    if nxt.talk_at is not None and nxt.talk_text and nxt.talk_at != (previous.talk_at if previous else None):
        lines.append((nxt.talk_text, "replace"))
    if nxt.completion == "confirmed" and (previous is None or previous.completion != "confirmed"):
        lines.append((COMPLETION_SPEECH, "append"))
    if nxt.completion == "user_confirmed" and (previous is None or previous.completion != "user_confirmed"):
        lines.append((USER_CONFIRMED_SPEECH, "replace"))
    if nxt.needs_reselect and not (previous and previous.needs_reselect):
        lines.append((RESELECT_SPEECH, "append"))
    if nxt.notice and nxt.notice != (previous.notice if previous else None):
        lines.append((nxt.notice, "append"))
    if nxt.error and nxt.error != (previous.error if previous else None):
        lines.append((nxt.error, "append"))
    return lines



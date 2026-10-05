"""The real engine: a server-side port of the demo browser's guide orchestration, driving the demo backend.

Source of truth: aisw-hybrid-talk ``web/src`` at commit 73ec3a8 — ``intent/useIntentLoop.ts`` (the loop),
``tracking/useObjectTrack.ts`` (the tracking run) and the pure modules ported in ``guide_policy.py``. The
upstream is the demo backend's HTTP API (``upstream.py``); it is not modified.

What the browser did and where it now happens:

* Camera capture → the client's stream frames. Each accepted stream frame of the current tracking run is
  forwarded to ``/api/track/frame`` and answered with a ``track`` message (``on_frame``).
* ``liveTrackStore`` → ``self.pair``: the newest tracker answer TOGETHER with the JPEG it was computed on.
* Follow / confirm / talk scenes: the browser captured the video "now" and attached the store snapshot read in
  the same synchronous turn (anchorProject.ts:387-405, useIntentLoop.ts:900-920). Here the scene is the JPEG
  of ``self.pair`` and the AnchorRef is that same pair's tracker answer, so the anchor box and the scene are the
  same capture by construction. ``_capture_scene`` is the one place that chooses the scene; switching to
  high-resolution frames later means issuing ``capture_hi`` there and pairing the answer's box with it.
* The plan's frame is the ``capture_hi`` answer if it arrives within 2 s of ``start``, else the latest stream frame.
* ``setView`` + React render → ``self.view`` + ``_emit()`` (full ``state`` snapshot when it changed, plus the
  ``say`` lines ``speechPolicy.spokenLines`` derives from the view transition).
* ``setTimeout`` / the 100 ms interval → a timer heap driven by ``_ticker`` (or by ``advance()`` in tests, with
  an injected clock).

Simplified because the server is now the only authority: no camera/mirror/consent fences (the WS layer gates
consent), no React commit throttling, no debug trace beyond ``self.stats``. The acceptance rule
(bound / text_only / rejected) is kept exactly: an answer computed against an older plan revision, run or
anchor generation is dropped or downgraded as in acceptance.ts.

§15 Plan mode rides on the same loop: the plan call is unchanged, but its answer becomes a ``draft`` Plan and
the run stops in ``phase:"reviewing"`` (no tracking run, no judgement, ``overlay`` null) until
``plan_approve``. From then on the loop is the one above, with the §15.4 gates of ``plan_policy`` consulted
wherever the loop used to move itself forward, and every mid-run plan change (``_install_replan``'s four
callers) published as an approvable ``state.proposal`` instead of an installed plan. ``self.plan`` is the plan
the client sees; ``run.plan_revision`` stays the revision the upstream knows, because it is what every fence
echo is judged against.

Plan mode is real, not a scripted mock: the draft is registered upstream by ``POST /api/guide/plan/approve``
(with any client edits) and that endpoint's returned ``plan_id``/``plan_revision`` become the run's
authoritative pair, so ``run.plan_revision`` and the revision every fence echoes never diverge. A change the
client rejects is undone by ``POST /api/guide/plan/revert``: the upstream restores its previous plan under a
NEW revision (so fences only ever move forward) and the run resumes where it was. Only when the upstream
cannot be put back (no previous plan, a stale pair, a transport failure) does the run end with
``plan_changed``. While a proposal waits for the client, judgement and progression are frozen the way the user
pause freezes them (frames and the box keep working).

§15.8 adds two alternative guide cores beside ``classic``. ``sequential`` asks the follower about the current
step alone, so only an upper ``step_done`` on that same step moves the run. ``graph`` asks about EVERY plan step
on every capture and keeps an explicit per-node ledger: a visual ``yes`` only nominates an eligible node, the
upper's own same-node ``step_done`` commits it, independent nodes may complete in either order, a fresh ``no``
revokes a revocable (``state``) claim and sends its dependents to recheck, and ``step_index`` is only the
deterministic suggested focus — never completion.
"""

from __future__ import annotations

import asyncio
import base64
import heapq
import logging
import math
import secrets
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import httpx

from . import guide_policy as gp
from . import plan_policy as pp
from .contracts import (
    CLARIFICATION_MAX,
    CORE_GRAPH,
    CORE_SEQUENTIAL,
    MAX_CLARIFICATION_ANSWERS,
    MAX_STEPS,
    RESEARCH_SOURCES_MAX,
    Blocked,
    Box,
    BudgetNotice,
    DEFAULT_CORE_MODE,
    DEFAULT_PLAN_MODEL,
    FrameHeader,
    GuideState,
    GuideStep,
    MaterialInput,
    Notice,
    Overlay,
    OverlayBinding,
    Partial,
    Pending,
    Plan,
    PlanAnswer,
    PlanEditMsg,
    Prefs,
    Proposal,
    ReferenceImage,
    ResearchSource,
    StateError,
    StepStatus,
    TalkResult,
    TrackMsg,
    overlay_command_from_step,
    reference_images_problem,
)
from .engine import EngineReject, EngineSink
from .upstream import UpstreamClient, UpstreamError, UpstreamHealth

log = logging.getLogger("synoptics_live.real_engine")

TICK_MS = 100  # useIntentLoop.ts:153
COMPLETION_RECHECK_MS = 1500  # useIntentLoop.ts:151
MAX_CONSECUTIVE_FAILURES = 3  # useIntentLoop.ts:157
LIVE_BOX_FRESHNESS_MS = 300  # liveTrackStore.ts:271
HI_FRAME_WAIT_MS = 2000  # spec §6.5
FIRST_FRAME_WAIT_MS = 5000
NO_FRAME_CONFIRM_RETRY_MS = 500
PLAN_BUSY_RETRIES = 5
PRIMARY_ANCHOR_ID = "a1"  # anchorProject.ts:29
MANUAL_SELECTION_TARGET = "user-selected object"  # useObjectTrack.ts:45
SCENE_LABEL = "카메라 현재 화면"
FENCE_KEY = "live"
#: §15.8: how many observations the graph's append-only history keeps. The ledger itself stays exact; the
#: history is a readable record, so a very long run trims its oldest entries instead of growing without bound.
GRAPH_HISTORY_MAX = 1024
#: §15: the only planning profile with a real hosted research tool. The DeepSeek profile has none, so Live
#: asks for research only under this profile (and only in review mode).
RESEARCH_PLAN_MODEL = "astra:high"


def _epoch_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(8)}"


@dataclass
class RealConfig:
    upstream_url: str
    access_code: str | None = None
    #: "stream": follow/confirm/talk use the newest stream frame that has a tracker answer (see module doc).
    scene_source: str = "stream"
    call_timeout_s: float = 60.0
    track_timeout_s: float = 5.0


class RealContext:
    """Per-app shared state of the real engine: config, the cached upstream health, the transport (tests)."""

    def __init__(self, cfg: RealConfig, transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.transport = transport
        self.health = UpstreamHealth(cfg.upstream_url, transport=transport, clock=clock)

    def client(self) -> UpstreamClient:
        return UpstreamClient(self.cfg.upstream_url, access_code=self.cfg.access_code, transport=self.transport,
                              call_timeout_s=self.cfg.call_timeout_s, track_timeout_s=self.cfg.track_timeout_s)


# ------------------------------------------------------------------------------------------ state holders


@dataclass(eq=False)
class TrackPair:
    """liveTrackStore.ts:275-287 snapshot + the exact JPEG it describes (wire identities, engine-clock time)."""

    run_id: str
    track_id: str
    generation: int
    state: str
    box: Box | None
    target: str | None
    jpeg: bytes
    width: int
    height: int
    at_ms: float
    grid: list[float] | None = None
    grid_done: bool = False


@dataclass(eq=False)
class LiveRun:
    """One wire tracking run. A ``select_box`` re-seeds it with a NEW upstream run under the same wire ``run_id``
    (spec §14: same run_id, generation+1, new track_id); every other (re)start is a new LiveRun."""

    run_id: str
    target: str
    up_index: int = 1
    up_run_id: str | None = None
    ready: bool = False
    terminal: str | None = None
    frame_seq: int = 0
    last_version: int = 0
    gen_base: int = 0
    up_gen_first: int | None = None
    last_track_id: str | None = None
    last_generation: int | None = None
    last_state: str | None = None


@dataclass(eq=False)
class FollowFrame:
    """confirmDecision.ts:100-111."""

    frame_id: str
    jpeg: bytes
    captured_at: float
    anchor: dict | None
    run_id: str
    plan_id: str
    plan_revision: int


@dataclass(frozen=True)
class Binding:
    run_id: str
    track_id: str
    generation: int


@dataclass(frozen=True)
class SequentialEvidence:
    """§15.8 sequential core: one follower nomination / upper confirmation, bound to everything that made it.

    A follower ``yes`` only NOMINATES (``GuideRun.seq_candidate``); the upper ``step_done`` answer about the
    SAME activation of the SAME step of the SAME plan revision then commits it (``GuideRun.confirmed_steps``)
    and moves the run exactly one step. The binding is what stops a replayed frame, a late answer, or a verdict
    about a step that has since become current from moving the screen.
    """

    task_epoch: str
    plan_id: str
    plan_revision: int
    step_id: str
    step_index: int
    #: Every time the current step changes, ``GuideRun.seq_activation`` grows: a candidate is only ever about
    #: the activation it was made in.
    activation: int
    run_id: str
    track_id: str
    generation: int
    frame_id: str
    captured_at: float
    #: Engine-clock time the upper ``step_done`` yes committed it (0.0 while it is only a candidate).
    confirmed_at: float = 0.0


@dataclass(frozen=True)
class GraphEvidence:
    """§15.8 graph core: one per-node observation, bound to everything that made it.

    ``seq`` orders captures inside the run (monotone): a reading from a later capture revokes one from an
    earlier capture, and an earlier capture can never restore — or take back — a fact a later one settled. The
    task/plan/entity binding is what stops a replayed frame, a late answer, a verdict about another plan
    revision, or an anchor that has moved from touching the ledger.
    """

    task_epoch: str
    plan_id: str
    plan_revision: int
    step_id: str
    #: Monotone capture order inside the run (bumped once per applied follower capture, and once per ack).
    seq: int
    #: ``visual`` (a follower capture) or ``user`` (the user's own ``step_ack``).
    source: str
    status: StepStatus
    frame_id: str
    captured_at: float
    run_id: str
    track_id: str
    generation: int
    #: Engine-clock time a commit was applied for this evidence (0.0 while it is only a nomination).
    committed_at: float = 0.0
    #: Actual confirmation response provenance; never inferred from the selected planning profile.
    provider: str | None = None
    model: str | None = None
    accepted_via: str | None = None


@dataclass(eq=False)
class GuideRun:
    """useIntentLoop.ts:281-327."""

    id: int
    task_epoch: str
    goal: str
    context: str | None
    started_at: float
    session_id: str | None = None
    #: §15: the run reviews a draft plan before executing (``self.plan`` is the plan the client sees).
    plan_mode: bool = False
    #: §15: the upstream planning profile this run's plan — and its confirm/replan/talk — is bound to. The
    #: engine forwards it on the plan request and the upstream stores it with the plan; it never changes
    #: inside a run (a different choice is a different ``start``, i.e. a new task authority).
    plan_model: str = DEFAULT_PLAN_MODEL
    #: §15.8: which guide core this run executes (``classic`` default, ``sequential``, or ``graph``).
    #: Independent of ``plan_mode`` and ``plan_model``: the engine forwards it on the plan request, the upstream
    #: stores it with the plan (so the follower is asked about the current step alone, or about every step), and
    #: it never changes inside a run.
    core_mode: str = DEFAULT_CORE_MODE
    #: §15.8 sequential: the current step's activation. ``_set_step``/replans grow it; a candidate or verdict
    #: from an earlier activation can never move the step that is current now.
    seq_activation: int = 0
    #: §15.8 sequential: the follower's nomination awaiting the upper's own ``step_done`` verdict.
    seq_candidate: SequentialEvidence | None = None
    #: §15.8 sequential: what the upper has confirmed, in order. Retained across scene changes; reset when the
    #: plan id/revision moves (a verdict about another revision is never carried blindly onto this one).
    confirmed_steps: tuple = ()
    confirmed_frame_ids: set[str] = field(default_factory=set)
    #: §15.8 graph: the follower's current reading of each node it has looked at (``yes``/``no``/``unsure``).
    #: A ``no`` is the only thing that revokes; ``unsure`` never asserts physical absence.
    graph_status: dict = field(default_factory=dict)
    #: §15.8 graph: the explicit completion ledger (ordered by plan position in the snapshot). A node is here
    #: only if a visual commit was confirmed by the upper's same-node ``step_done`` yes, or the user acked it.
    #: Never derived from ``step_index``.
    graph_done: tuple = ()
    #: §15.8 graph: immutable accepted evidence, separate from the latest reading. Keys are CURRENT step ids;
    #: values keep their ORIGINAL plan/frame/step/provider binding even after an exact replan mapping.
    #: Identity invalidation removes applicability, not the historical user/upper confirmation.
    graph_facts: dict[str, GraphEvidence] = field(default_factory=dict)
    #: §15.8 graph: the latest applied evidence per node (the freshness/identity binding every result must beat).
    graph_evidence: dict = field(default_factory=dict)
    #: §15.8 graph: append-only historical record of observations and acks. Retained when applicability is
    #: revoked — a revocation is not amnesia, and historical occurrence is not still-applicable authorization.
    graph_history: tuple = ()
    #: §15.8 graph: the follower's nomination awaiting the upper's own ``step_done`` verdict on that node.
    graph_candidate: GraphEvidence | None = None
    #: §15.8 graph: the monotone capture counter (see ``GraphEvidence.seq``).
    graph_seq: int = 0
    #: §15.4.8: the mandatory checks that were still unverified at the moment the goal was confirmed, frozen
    #: there so a completed scenario can show the goal plus its outstanding safety/gate checks. ``None`` while
    #: the run is still going (the snapshot then reports the live set).
    pending_required: tuple | None = None
    #: §15.3: the extracted material texts the draft was built from. The wire only ever carries their ``chars``.
    materials: tuple = ()
    #: §15: reference photos the user supplied (a close-up, a label) — separate evidence, never the current
    #: scene. They live only in this run's memory: validated before a plan call, forwarded to the upstream,
    #: and never logged, persisted or echoed back in ``state``. A supplied ``plan_answer`` list replaces them.
    reference_images: tuple = ()
    #: §15: the answered questions of this run's clarification turn, oldest first (at most
    #: ``MAX_CLARIFICATION_ANSWERS``). They ride every later plan request, so the planner keeps the context
    #: without the goal being rewritten.
    answers: tuple = ()
    #: §15: the CURRENT question and the id its answer must echo (both None outside ``phase:"clarifying"``).
    clarification: str | None = None
    clarification_id: str | None = None
    retrying_answer: bool = False
    #: §15: public facts the planner reported; they ride ``state.research_sources`` and the client's Plan.
    research_sources: tuple = ()
    plan_id: str | None = None
    plan_revision: int | None = None
    steps: tuple = ()
    goal_when: str = ""
    target: str = ""
    step_index: int = 0
    intent_seq: int = 0
    latest_seq: dict = field(default_factory=lambda: {"follow": 0, "confirm": 0, "talk": 0})
    unsure_streak: int = 0
    last_checks: tuple | None = None
    skipped: tuple = ()
    user_done: tuple = ()
    talk_pending: bool = False
    talk_waiting: bool = False
    last_talk: tuple[str, str] | None = None
    stuck_count: int = 0
    replan_budget: gp.ReplanBudget = field(default_factory=gp.ReplanBudget)
    replan_epoch: int = 0
    consecutive_failures: int = 0
    completion: str = "none"
    halted: bool = False
    binding: Binding | None = None
    accepted: gp.AcceptedBox | None = None
    appearance_ref: gp.AppearanceReference | None = None
    warn: bool = False
    first_box_seen: bool = False
    aborted: bool = False
    tasks: set = field(default_factory=set)


@dataclass
class View:
    """useIntentLoop.ts:190-227 IntentView, in wire terms."""

    phase: str = "idle"
    #: §15.8: the core of the run this view belongs to (echoed in every state, including after it ends).
    core_mode: str = DEFAULT_CORE_MODE
    plan_id: str | None = None
    plan_revision: int | None = None
    steps: tuple = ()
    goal_when: str = ""
    target: str = ""
    step_index: int = 0
    skipped: tuple = ()
    user_done: tuple = ()
    partial_target: str | None = None
    partial_first_say: str | None = None
    clarification: str | None = None
    clarification_id: str | None = None
    research_sources: tuple = ()
    pending: Pending | None = None
    basis_age_s: float | None = None
    notice: Notice | None = None
    budget_notice: BudgetNotice | None = None
    completion: str = "none"
    needs_reselect: bool = False
    replan_blocked_reason: str | None = None
    talk: TalkResult | None = None
    talk_pending: bool = False
    talk_blocked_reason: str | None = None
    error: StateError | None = None


def _clip(text: str | None, n: int) -> str | None:
    if text is None:
        return None
    text = text.strip()
    return text[:n] if text else None


def anchor_label(primary: str | None, fallback: str | None) -> str:
    """anchorProject.ts:378-381."""
    raw = (primary or "").strip() or (fallback or "").strip() or "object"
    return raw[:40]


def build_anchor_ref(pair: TrackPair, label: str) -> dict:
    """anchorProject.ts:387-405: the box goes on the wire only while tracking."""
    ref: dict[str, Any] = {
        "anchor_id": PRIMARY_ANCHOR_ID, "role": "target", "label": label[:40] or "object",
        "run_id": pair.run_id, "track_id": pair.track_id, "generation": pair.generation, "state": pair.state,
    }
    if pair.state == "tracking" and pair.box is not None:
        ref["box"] = pair.box.model_dump()
    return ref


def echo_of(ref: dict | None) -> tuple[dict, ...]:
    """anchorProject.ts:407-409."""
    if ref is None:
        return ()
    return ({"anchor_id": ref["anchor_id"], "track_id": ref["track_id"], "generation": ref["generation"]},)


def decide_track_error(status: int, code: str) -> str:
    """trackFence.ts:430-439."""
    if status == 409:
        if code in ("stale_run", "not_active_run", "seed_not_first_frame"):
            return "retire_run"
        if code == "stale_frame":
            return "drop_frame"
        if code == "stale_start":
            return "lost_start_race"
    if status in (503, 504):
        return "unavailable"
    return "unknown"


def describe_plan_error(exc: UpstreamError) -> str:
    """useIntentLoop.ts:1612-1630."""
    code = exc.code
    if exc.status == 0:
        return "안내 서버에 연결하지 못했습니다. 잠시 후 다시 시도해주세요."
    if code == "service_unavailable":
        return "계획 서버를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요."
    if code == "provider_busy":
        return "이전 계획 요청이 아직 진행 중입니다. 잠시 후 다시 시도해주세요."
    if code == "rate_limited":
        return "요청이 너무 잦습니다. 잠시 후 다시 시도해주세요."
    if code == "ai_consent_required":
        return "AI 동의가 필요합니다."
    if code == "invalid_provider_output":
        return f"모델 응답이 계약을 통과하지 못했습니다{f'({exc.reason})' if exc.reason else ''}. 다시 시도해주세요."
    if code in ("session_expired", "session_mismatch"):
        return "세션이 만료되었습니다. 다시 시도해주세요."
    return f"계획을 받지 못했습니다({exc.status} {code})."


FOLLOW_TRIGGER_TEXT = {  # useIntentLoop.ts:342-355 (display; kept for pending.trigger readers)
    "acquired": "대상 잡힘", "target_moved": "대상 이동", "target_changed": "대상 변화", "anchor_lost": "대상 놓침",
    "manual": "직접 요청", "heartbeat": "주기 확인", "goal_check": "완료 확인", "step_done": "단계 확인",
    "unsure_twice": "판단 불확실", "replan": "계획 재검토", "start": "가이드 시작", "talk": "사용자 말",
}

#: §15.6: why the change is proposed. Used unless the upstream's own ``replan.reason`` arrives with it.
PROPOSAL_REASON = {
    "unsure_twice": "판단이 계속 불확실해 계획을 다시 나눴어요.",
    "replan": "대상이 자주 바뀌어 계획을 다시 짰어요.",
    "goal_check": "완료 확인에서 계획을 손봐야 했어요.",
    "step_done": "단계 확인에서 계획을 손봐야 했어요.",
}
DEFAULT_PROPOSAL_REASON = "화면과 계획이 맞지 않아 계획을 바꾸려 합니다."
TALK_PROPOSAL_REASON = "말씀하신 대로 계획을 다시 짰어요."
#: §15.6: while a change waits for the client's answer, judgement and progression are frozen.
PROPOSAL_WAIT_REASON = "제안된 변경안을 먼저 확인해 주세요."
#: §15: while a question waits for the user's answer, nothing else may be asked or started.
CLARIFY_WAIT_REASON = "질문에 먼저 답해 주세요."


def _parse_steps(raw: list) -> tuple[GuideStep, ...]:
    steps = tuple(GuideStep.model_validate(s) for s in raw)
    if not (1 <= len(steps) <= MAX_STEPS):
        raise ValueError(f"plan must have 1-{MAX_STEPS} steps")
    return steps


def _plan_pair(current: dict) -> tuple[str, int] | None:
    """``(plan_id, plan_revision)`` from a ``plan/current`` shape, or None when the pair is not usable."""
    pid, rev = current.get("plan_id"), current.get("plan_revision")
    if not isinstance(pid, str) or not pid or not isinstance(rev, int) or isinstance(rev, bool) or rev < 1:
        return None
    return pid, rev


def _parse_research_sources(raw: object) -> tuple[ResearchSource, ...]:
    """The plan answer's ``research_sources`` as live models, dropping anything malformed.

    Live is not the research authority (the upstream keeps only URLs its own tool actually returned), but it is
    the wire authority: a source that is not a well-formed public HTTP(S) URL is not evidence and is dropped
    rather than allowed to make the whole snapshot invalid. Bounded by the same maximum as the plan itself.
    """
    if not isinstance(raw, list):
        return ()
    out: list[ResearchSource] = []
    for item in raw:
        if not isinstance(item, dict) or len(out) >= RESEARCH_SOURCES_MAX:
            continue
        try:
            out.append(ResearchSource.model_validate(item))
        except ValueError:
            continue
    return tuple(out)


class RealEngine:
    def __init__(self, sink: EngineSink, ctx: RealContext, *, clock: Callable[[], float] = time.monotonic,
                 auto_tick: bool = True):
        self.sink = sink
        self.ctx = ctx
        self._clock = clock
        self._auto_tick = auto_tick
        self.up: UpstreamClient | None = None
        self.follow_provider: str | None = None
        self.view = View()
        self.guide: GuideRun | None = None
        #: §15: the plan the client sees (draft while reviewing, then the approved one). None = immediate (§11).
        self.plan: Plan | None = None
        #: §15.6: a change the server wants to make; the installed plan stays authoritative until answered.
        self.proposal: Proposal | None = None
        #: §15.4: why the run cannot move on right now (required check / unmet prerequisite).
        self.blocked: Blocked | None = None
        #: §15.7 user pause. Distinct from ``self.paused`` (the connection pause): a reconnect never clears it.
        self.user_paused = False
        self.gate = gp.INITIAL_GATE_STATE
        self.trig = gp.INITIAL_TRIGGER_STATE
        self.appearance = gp.INITIAL_APPEARANCE_STATE
        self.pending_views: dict[str, Pending] = {}
        self.budget_capped = False
        self.talk_block: str | None = None
        self.notice_stamp: tuple[int, float] | None = None
        self.gate_timer: tuple[int, float] | None = None
        self.track: LiveRun | None = None
        self.pair: TrackPair | None = None
        self.latest_frame: tuple[bytes, int, int, float] | None = None
        self.paused = False
        self._resumed = asyncio.Event()
        self._resumed.set()
        self.prefs = Prefs()
        self.stats: Counter = Counter()
        self.milestones: dict[str, float] = {}
        self.step_log: list[tuple[float, int, str]] = []
        self._run_counter = 0
        self._timers: list[tuple[float, int, Callable[[], None]]] = []
        self._timer_seq = 0
        self._cancelled_timers: set[int] = set()
        self._sleepers: set[asyncio.Future] = set()
        self._tasks: set[asyncio.Task] = set()
        self._ticker: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._hi_req: str | None = None
        self._hi_future: asyncio.Future | None = None
        self._frame_future: asyncio.Future | None = None
        self._last_state: str | None = None
        self._speech: gp.SpeechView | None = None
        self._closed = False

    # ================================================================================== clock, timers, tasks

    def now(self) -> float:
        return self._clock() * 1000.0

    def _later(self, fn: Callable[[], None], ms: float) -> int:
        self._timer_seq += 1
        heapq.heappush(self._timers, (self.now() + max(0.0, ms), self._timer_seq, fn))
        if self._wake is not None:
            self._wake.set()
        return self._timer_seq

    def _clear_timer(self, handle: int) -> None:
        self._cancelled_timers.add(handle)

    def _clear_timers(self) -> None:
        """useIntentLoop.ts:432-436 (pending sleeps are cancelled with them)."""
        self._timers.clear()
        self._cancelled_timers.clear()
        self.gate_timer = None
        for fut in list(self._sleepers):
            if not fut.done():
                fut.cancel()
        self._sleepers.clear()

    async def _sleep(self, ms: float) -> None:
        fut = asyncio.get_running_loop().create_future()
        self._sleepers.add(fut)
        self._later(lambda: fut.done() or fut.set_result(None), ms)
        try:
            await fut
        finally:
            self._sleepers.discard(fut)

    def _run_due_timers(self) -> None:
        now = self.now()
        while self._timers and self._timers[0][0] <= now:
            _, handle, fn = heapq.heappop(self._timers)
            if handle in self._cancelled_timers:
                self._cancelled_timers.discard(handle)
                continue
            try:
                fn()
            except Exception:
                log.exception("timer failed")

    def _spawn(self, coro, run: GuideRun | None = None) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if run is not None:
            run.tasks.add(task)
            task.add_done_callback(run.tasks.discard)
        return task

    def _ensure_ticker(self) -> None:
        if not self._auto_tick or self._ticker is not None or self._closed:
            return
        self._wake = asyncio.Event()
        self._ticker = asyncio.get_running_loop().create_task(self._tick_loop())

    async def _tick_loop(self) -> None:
        assert self._wake is not None
        while True:
            next_due = self._timers[0][0] if self._timers else math.inf
            wait_ms = min(TICK_MS, max(1.0, next_due - self.now()))
            try:
                await asyncio.wait_for(self._wake.wait(), wait_ms / 1000)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                self.tick_all()
            except Exception:
                log.exception("tick failed")

    def tick_all(self) -> None:
        """One pass: due timers, the loop's tick, then one emit."""
        self._run_due_timers()
        self._tick()
        self._emit()

    async def advance(self, ms: float, settle_rounds: int = 60) -> None:
        """Tests only (``auto_tick=False`` + an injected clock): run timers/tick after the clock moved ``ms``."""
        step = TICK_MS
        remaining = ms
        while True:
            chunk = min(step, remaining)
            if hasattr(self._clock, "advance"):
                self._clock.advance(chunk / 1000)  # type: ignore[attr-defined]
            self.tick_all()
            await self.settle(settle_rounds)
            remaining -= chunk
            if remaining <= 0:
                break

    async def settle(self, rounds: int = 60) -> None:
        for _ in range(rounds):
            await asyncio.sleep(0)

    # ================================================================================== view + emit

    def _upstream(self) -> UpstreamClient:
        if self.up is None:
            self.up = self.ctx.client()
        return self.up

    def _set_notice(self, code: str, text: str) -> None:
        """Stamped with the step it was raised on (useIntentLoop.ts:1581-1584)."""
        self.view.notice = Notice(code=code, text=text[:200])
        self.notice_stamp = (self.view.step_index, self.now())

    def _plan_model(self) -> Plan | None:
        if self.plan is not None:
            return self.plan  # §15: the plan the client sees (draft, approved, or edited draft)
        v = self.view
        if v.plan_id is None or v.plan_revision is None or not v.steps:
            return None
        return Plan(plan_id=v.plan_id[:64], revision=max(1, v.plan_revision), target=anchor_label(v.target, None),
                    goal_when=_clip(v.goal_when, 60) or "목표 상태", steps=list(v.steps),
                    research_sources=list(v.research_sources))

    def _graph_action_allowed(self, run: GuideRun | None) -> bool:
        """§15.4.8: may the node at the suggested focus carry an ACTION — an overlay command, a spoken
        instruction?

        Every other core decides that with its own gates (a hold already blocks its own dispatch). In graph mode
        only an ACTION-READY node qualifies: an unresolved prerequisite, an unresolved MANDATORY ancestor or an
        unverified safety precondition of the focused node means the run may still READ it — the follow/confirm
        loop is untouched — but must not tell the user to act on it.
        """
        if run is None or run.core_mode != CORE_GRAPH or self.plan is None:
            return True
        if not (0 <= run.step_index < len(run.steps)):
            return False
        return run.steps[run.step_index].id in set(pp.graph_ready_ids(list(self.plan.steps), run.graph_done))

    def _overlay(self) -> Overlay | None:
        """Spec §8 from the current step's commands (anchorProject.ts:126-150 bindStepCommands, role → a1)."""
        run = self.guide
        v = self.view
        if v.phase != "running" or run is None or run.halted or run.binding is None or run.plan_id is None:
            return None
        if self.user_paused:
            return None  # §15.7: tracking is released, so there is no anchor left to draw on
        if not (0 <= run.step_index < len(run.steps)):
            return None
        step = run.steps[run.step_index]
        b = run.binding
        # §15.4.8: a node the run may not act on yet carries no action command. The binding and the motion key
        # stay, so the client keeps drawing what is tracked, but nothing is asked of the user.
        commands = ([overlay_command_from_step(c) for c in step.commands]
                    if self._graph_action_allowed(run) else [])
        return Overlay(
            binding=OverlayBinding(anchor_id=PRIMARY_ANCHOR_ID, run_id=b.run_id, track_id=b.track_id,
                                   generation=b.generation),
            commands=commands,
            warn="needs_reselect" if run.warn else None,
            # actionMotion.ts:1028-1046 motionKey parts.
            motion_key=f"{run.id}|{run.plan_revision}|{step.id}|{PRIMARY_ANCHOR_ID}|{b.run_id}|{b.track_id}|{b.generation}"[:200],
        )

    def snapshot(self) -> GuideState:
        v = self.view
        plan = self._plan_model()
        partial = None
        if v.phase == "planning" and (v.partial_target or v.partial_first_say):
            partial = Partial(target=_clip(v.partial_target, 40), first_say=_clip(v.partial_first_say, 60))
        step_index = v.step_index if plan is not None and v.step_index < len(plan.steps) else 0
        # §15.8 graph: the ledger is the run's explicit per-node state, in plan order (deterministic snapshots).
        # ``classic``/``sequential`` leave these empty — their own snapshot is unchanged.
        steps_done: list[str] = []
        steps_ready: list[str] = []
        step_statuses: dict[str, StepStatus] = {}
        pending_required: list[str] = []
        run = self.guide
        if run is not None and plan is not None:
            if run.core_mode == CORE_GRAPH:
                done_ids = set(run.graph_done)
                steps_done = [step.id for step in plan.steps if step.id in done_ids]
                steps_ready = list(pp.graph_ready_ids(list(plan.steps), run.graph_done))
                step_statuses = {step.id: run.graph_status[step.id] for step in plan.steps
                                 if step.id in run.graph_status}
                # §15.4.8: mandatory checks still unverified — live while running, frozen once the goal was
                # confirmed. Never part of the goal's own condition.
                pending_required = list(run.pending_required) if run.pending_required is not None else \
                    pp.graph_pending_required_checks(plan, run.graph_done)
            elif self._gated(run):
                pending_required = pp.required_pending(plan, step_index=run.step_index, skipped=run.skipped,
                                                       user_done=run.user_done)
        return GuideState(
            phase=v.phase, core_mode=v.core_mode, plan=plan, partial=partial,
            clarification=_clip(v.clarification, CLARIFICATION_MAX),
            clarification_id=v.clarification_id,
            research_sources=list(v.research_sources),
            step_index=step_index, steps_skipped=list(v.skipped), steps_user_done=list(v.user_done),
            steps_done=steps_done, steps_ready=steps_ready, step_statuses=step_statuses,
            pending_required_checks=pending_required,
            pending=v.pending, basis_age_s=v.basis_age_s, notice=v.notice, budget_notice=v.budget_notice,
            completion=v.completion, needs_reselect=v.needs_reselect,
            replan_blocked_reason=_clip(v.replan_blocked_reason, 200), talk=v.talk, talk_pending=v.talk_pending,
            talk_blocked_reason=_clip(v.talk_blocked_reason, 200), overlay=self._overlay(), error=v.error,
            # §15.7: the user's pause is a running state; it is never the connection pause (self.paused).
            paused=self.user_paused and v.phase == "running", proposal=self.proposal, blocked=self.blocked,
        )

    def _speech_view(self) -> gp.SpeechView:
        v = self.view
        talk = v.talk
        # §15.4.8: no instruction for a node the run may not act on yet (talk replies, notices, the completion
        # line and errors still speak — only the step instruction is withheld).
        instruction = (gp.step_speech(v.phase, v.steps, v.step_index, v.partial_first_say)
                       if self._graph_action_allowed(self.guide) else None)
        return gp.SpeechView(
            phase=v.phase,
            step=instruction,
            talk_at=talk.at if talk else None,
            talk_text=gp.talk_speech(talk.reply, talk.spoken) if talk else None,
            completion=v.completion,
            needs_reselect=v.needs_reselect,
            notice=v.notice.text if v.notice else v.clarification,
            error=v.error.message if v.error else None,
        )

    def _emit(self) -> None:
        if self._closed:
            return
        try:
            state = self.snapshot()
        except Exception:
            log.exception("state snapshot invalid; not sent")
            return
        text = state.model_dump_json()
        if text != self._last_state:
            self._last_state = text
            self.sink.state(state)
        speech = self._speech_view()
        lines = gp.spoken_lines(self._speech, speech)
        self._speech = speech
        for line, mode in lines:
            self.sink.say(line[:400], mode)  # type: ignore[arg-type]

    def _set_pending_view(self, stage: str, pending: Pending | None) -> None:
        """useIntentLoop.ts:438-450: the follow wins the single chip."""
        if pending is not None:
            self.pending_views[stage] = pending
        else:
            self.pending_views.pop(stage, None)
        self.view.pending = self.pending_views.get("follow") or self.pending_views.get("confirm")

    # ================================================================================== §15 plan mode

    def _reviewing(self) -> bool:
        """§15.1: the draft is being reviewed. No tracking run, no judgement, ``overlay`` null."""
        return self.view.phase == "reviewing"

    def _clarifying(self) -> bool:
        """§15: the run is waiting for one answer (``plan_answer``). Nothing is tracked, judged or executed."""
        return self.view.phase == "clarifying"

    def _planning(self) -> bool:
        """§15: a plan is being written (first turn, or the turn after an answer). There is nothing to judge:
        the run may already carry the previous turn's ``plan_id``, so this is not implied by it being None."""
        return self.view.phase == "planning"

    def _judging_suspended(self) -> bool:
        """True while nothing may be dispatched: the connection pause, the user pause (§15.7), planning, plan
        review (§15.1), the question turn (§15), or a change waiting for the client's answer (§15.6)."""
        return self.paused or self.user_paused or self._planning() or self._reviewing() \
            or self._clarifying() or self.proposal is not None

    def _enqueue_blocked(self) -> bool:
        """Planning, plan review, the question turn, the user's pause and a pending proposal take no new work at
        all (the connection pause only defers it)."""
        return self.user_paused or self._planning() or self._reviewing() or self._clarifying() \
            or self.proposal is not None

    def _set_blocked(self, blocked: Blocked | None) -> None:
        """§15.4.2/§15.4.3: the run holds and the client is told why (once, when it starts holding)."""
        self.blocked = blocked
        if blocked is not None:
            self._set_notice("required_check", blocked.reason)

    def _hold(self, run: GuideRun, blocked: Blocked | None) -> None:
        if blocked is None:
            return
        self.stats["plan_hold"] += 1
        self._set_blocked(blocked)
        self._emit()

    def _uses_plan(self, run: GuideRun) -> bool:
        """§15/§15.8: the run keeps a server-side plan model (the §15.4 gates are computed from it)."""
        return run.plan_mode or run.core_mode in (CORE_SEQUENTIAL, CORE_GRAPH)

    def _gated(self, run: GuideRun) -> bool:
        """§15.4 fences apply to a reviewed run, and — §15.8 — to the sequential/graph core's own run too.
        Classic without ``plan_mode`` is the immediate flow and stays exactly as it was."""
        return self._uses_plan(run)

    def _completion_allowed(self, run: GuideRun) -> bool:
        """§15.4.5: every required step must be done. A skipped or pending one is not done.

        §15.8 graph: "done" is the explicit ledger (a confirmed commit or the user's ack) — never the suggested
        focus index. What keeps the goal unachieved is ``goal_required``: a mandatory NON-goal check (a safety
        precondition such as an unplugged adapter) does not make an already-satisfied goal unachieved, it is
        reported through ``pending_required_checks`` instead (§15.4.8).
        """
        if not self._gated(run) or self.plan is None:
            return True
        if run.core_mode == CORE_GRAPH:
            return pp.graph_completion_ready(self.plan, run.graph_done)
        return pp.completion_ready(self.plan, step_index=run.step_index, skipped=run.skipped,
                                   user_done=run.user_done)

    def _hold_completion(self, run: GuideRun) -> None:
        """§15.4.5: the picture looks finished but a required check is still empty — stay and say so.

        In graph mode only the ``goal_required`` nodes can hold completion (an unverified safety check is shown
        beside the goal, not as the goal unmet).
        """
        plan = self.plan
        if plan is None:
            return
        self.stats["completion_held"] += 1
        if run.core_mode == CORE_GRAPH:
            pending = pp.graph_completion_pending(plan, run.graph_done)
        else:
            pending = pp.required_pending(plan, step_index=run.step_index, skipped=run.skipped,
                                          user_done=run.user_done)
        step_id = pending[0] if pending else plan.steps[min(run.step_index, len(plan.steps) - 1)].id
        self._set_blocked(Blocked(step_id=step_id, requires=[],
                                  reason=f"필수 검사 {'·'.join(pending)} 단계가 남아 완료로 넘어가지 않습니다."))

    def _clear_resolved_block(self, run: GuideRun) -> None:
        """A hold the answer resolved: its step is done, its prerequisites are, or completion is free again."""
        blocked = self.blocked
        if blocked is None or self.plan is None:
            return
        if run.core_mode == CORE_GRAPH:
            done = set(run.graph_done)
        else:
            done = pp.done_step_ids(self.plan, step_index=run.step_index, skipped=run.skipped,
                                    user_done=run.user_done)
        prereqs_met = bool(blocked.requires) and all(need in done for need in blocked.requires)
        if blocked.step_id in done or prereqs_met or self._completion_allowed(run):
            self._set_blocked(None)

    # ==================================================================== §15.8 sequential core

    def _reset_sequential(self, run: GuideRun, *, moved: bool = False, replanned: bool = False) -> None:
        """Drop a pending nomination whenever the ground under it changes, and the ledger when so does the plan.

        ``moved``: the current step changed (an explicit user done/talk reposition, or a committed step), so a
        candidate/verdict from the previous activation can never apply. ``replanned``: the plan id/revision
        moved, so confirmed evidence about the old revision is not carried blindly onto this one. Evidence is
        otherwise RETAINED (a scene change or a lost anchor does not unset what the upper already confirmed).
        """
        if run.core_mode != CORE_SEQUENTIAL:
            return
        run.seq_candidate = None
        if moved:
            run.seq_activation += 1
        if replanned:
            run.confirmed_steps = ()

    def _candidate_for(self, run: GuideRun, frame: FollowFrame, index: int, step_id: str,
                       activation: int) -> SequentialEvidence:
        anchor = frame.anchor or {}
        return SequentialEvidence(
            task_epoch=run.task_epoch, plan_id=run.plan_id or "", plan_revision=run.plan_revision or 0,
            step_id=step_id, step_index=index, activation=activation,
            run_id=str(anchor.get("run_id") or frame.run_id),
            track_id=str(anchor.get("track_id") or ""),
            generation=int(anchor.get("generation") or 0),
            frame_id=frame.frame_id, captured_at=frame.captured_at,
        )

    def _nominate_sequential(self, run: GuideRun, index: int, step_id: str, frame: FollowFrame,
                             activation: int) -> SequentialEvidence:
        """A follower ``yes`` about the CURRENT step only nominates; the screen does not move yet (§15.8)."""
        candidate = self._candidate_for(run, frame, index, step_id, activation)
        run.seq_candidate = candidate
        self.stats["sequential_nominated"] += 1
        self._set_notice("step_candidate", "이 단계가 끝난 것으로 보입니다. 위 확인 결과를 기다리는 중입니다.")
        return candidate

    def _settle_sequential(self, run: GuideRun, candidate: SequentialEvidence, answer: dict) -> None:
        """The upper's own ``step_done`` verdict on the nominated activation: exactly one step, or a hold."""
        index = candidate.step_index
        step = run.steps[index] if 0 <= index < len(run.steps) else None
        if run.core_mode != CORE_SEQUENTIAL or run.seq_candidate is not candidate:
            return
        if candidate.activation != run.seq_activation or step is None or step.id != candidate.step_id \
                or candidate.plan_id != run.plan_id or candidate.plan_revision != run.plan_revision \
                or candidate.task_epoch != run.task_epoch or index != run.step_index \
                or candidate.frame_id in run.confirmed_frame_ids:
            # A replayed frame, a late answer, or a step/revision that moved under it: hold, never carry it.
            run.seq_candidate = None
            self.stats["sequential_stale_ignored"] += 1
            return
        if answer.get("step_check") != "yes" or answer.get("step_id") != candidate.step_id:
            # no/unsure/other: nothing moves. A fresh follower yes may nominate this step again.
            run.seq_candidate = None
            self.stats["sequential_held"] += 1
            self._set_notice("step_held", "화면 확인에서 이 단계가 끝난 것으로 보이지 않아 그대로 기다립니다.")
            return
        plan = self.plan
        nxt = index + 1
        if plan is not None and nxt < len(plan.steps):
            # §15.4.3: the step we are about to enter may require others; a sequential commit never passes one.
            blocker = pp.prerequisite_block(plan, nxt, step_index=run.step_index, skipped=run.skipped,
                                            user_done=(*run.user_done, *(e.step_id for e in run.confirmed_steps),
                                                       candidate.step_id))
            if blocker is not None:
                run.seq_candidate = None
                self._hold(run, blocker)
                return
        if not self._advance_plan(run, None):
            # §15.4.2/§15.4.4: a required or manual check still holds — nothing is confirmed, nothing moves.
            run.seq_candidate = None
            self.stats["sequential_blocked"] += 1
            return
        run.seq_candidate = None
        run.confirmed_steps = (*run.confirmed_steps, replace(candidate, confirmed_at=self.now()))
        run.confirmed_frame_ids.add(candidate.frame_id)
        self.stats["sequential_confirmed"] += 1
        if self.view.notice is not None and self.view.notice.code == "step_candidate":
            self.view.notice = None
            self.notice_stamp = None

    # ==================================================================== §15.8 graph core

    def _reset_graph(self, run: GuideRun, *, replanned: bool = False, identity_changed: bool = False,
                     previous_plan: Plan | None = None) -> None:
        """Invalidate pending/current readings, retaining only explicitly mappable confirmed facts.

        History is never rewritten as a new observation. Reselection lacks same-object proof, so both visual
        and user confirmations lose current applicability; their original records remain in history.
        Automatic first tracking acquisition is not a reselection and does not call this invalidation.
        """
        if run.core_mode != CORE_GRAPH:
            return
        run.graph_candidate = None
        if not replanned and not identity_changed:
            return
        mapping = pp.graph_fact_mapping(previous_plan, self.plan) \
            if replanned and not identity_changed and previous_plan is not None and self.plan is not None else {}
        run.graph_facts = {mapping[sid]: fact for sid, fact in run.graph_facts.items() if sid in mapping}
        run.graph_done = tuple(run.graph_facts)
        run.graph_status = {}
        run.graph_evidence = {}
        run.pending_required = None
        run.user_done = tuple(sid for sid, fact in run.graph_facts.items() if fact.source == "user")
        self.view.user_done = run.user_done
        self._graph_focus(run)

    def _init_graph(self, run: GuideRun) -> None:
        """A fresh graph run: nothing is done, nothing has been read, and the focus is its first enabled node."""
        if run.core_mode != CORE_GRAPH:
            return
        run.graph_status = {}
        run.graph_done = ()
        run.graph_facts = {}
        run.graph_evidence = {}
        run.graph_history = ()
        run.graph_candidate = None
        run.graph_seq = 0
        run.pending_required = None
        self._graph_focus(run)

    def _graph_focus(self, run: GuideRun) -> None:
        """The deterministic suggested focus — a hint for the overlay, never completion (§15.8)."""
        if run.core_mode != CORE_GRAPH or not run.steps:
            return
        run.step_index = pp.graph_focus_index(list(run.steps), run.graph_done, suggested=run.step_index)
        self.view.step_index = run.step_index

    def _graph_remember(self, run: GuideRun, evidence: GraphEvidence) -> None:
        run.graph_evidence[evidence.step_id] = evidence
        run.graph_history = (*run.graph_history, evidence)[-GRAPH_HISTORY_MAX:]

    def _graph_observe(self, run: GuideRun, checks, *, frame_id: str, captured_at: float,
                       anchor: dict) -> int:
        """Apply one follower capture's per-node readings, then propagate any revocation.

        ``seq`` is the capture order inside the run: only a reading NEWER than the node's latest applied one is
        applied, so a late answer never restores a fact a later capture settled. ``no`` is the only thing that
        revokes — an ``event`` node's own occurrence is historical and stands; a ``state`` node's claim stops
        applying and every dependent node goes to recheck (``unsure``), never to ``no``, and unrelated nodes are
        left exactly as they were.
        """
        plan = self.plan
        if plan is None:
            return run.graph_seq
        seq = run.graph_seq + 1
        run.graph_seq = seq
        statuses = {c["step_id"]: c.get("visible") for c in checks if isinstance(c.get("step_id"), str)}
        revoked: list[str] = []
        for step in plan.steps:
            status = statuses.get(step.id)
            if status not in ("yes", "no", "unsure"):
                continue
            evidence = GraphEvidence(
                task_epoch=run.task_epoch, plan_id=plan.plan_id, plan_revision=plan.revision, step_id=step.id,
                seq=seq, source="visual", status=status, frame_id=frame_id, captured_at=captured_at,
                run_id=str(anchor.get("run_id") or ""), track_id=str(anchor.get("track_id") or ""),
                generation=int(anchor.get("generation") or 0),
            )
            self._graph_remember(run, evidence)
            run.graph_status[step.id] = status
            if status == "no":
                candidate = run.graph_candidate
                if candidate is not None and candidate.step_id == step.id:
                    # a fresh negative supersedes the pending nomination: the older yes may no longer commit
                    run.graph_candidate = None
                if step.id in run.graph_done:
                    if step.condition_kind == "event":
                        self.stats["graph_event_kept"] += 1
                    elif run.graph_facts[step.id].source == "user":
                        # §15.4.4: a model verdict never un-does the user's own record, exactly as the
                        # sequential core treats an acknowledged step. The reading is recorded above; the
                        # completion stands.
                        self.stats["graph_user_kept"] += 1
                    else:
                        revoked.append(step.id)
        for step_id in revoked:
            self._graph_revoke(run, step_id)
        self._graph_focus(run)
        return seq

    def _graph_revoke(self, run: GuideRun, step_id: str) -> None:
        """A fresh ``no`` on a ``state`` node: its claim stops applying and its dependents go to recheck.

        Historical occurrence is not still-applicable authorization: a dependent (inspection or manual approval
        included) is never kept merely because its own ``condition_kind`` says ``event`` — it must be rechecked.
        """
        plan = self.plan
        if plan is None or step_id not in {step.id for step in plan.steps}:
            return
        self.stats["graph_revoked"] += 1
        run.graph_done = tuple(sid for sid in run.graph_done if sid != step_id)
        run.graph_facts.pop(step_id, None)
        dependents = pp.graph_dependents(list(plan.steps), step_id)
        for dependent in dependents:
            if dependent in run.graph_done:
                run.graph_done = tuple(sid for sid in run.graph_done if sid != dependent)
            run.graph_facts.pop(dependent, None)
            if run.graph_status.get(dependent) != "no":
                run.graph_status[dependent] = "unsure"
            self.stats["graph_recheck"] += 1
        run.user_done = tuple(sid for sid in run.user_done if sid in run.graph_facts)
        self.view.user_done = run.user_done
        candidate = run.graph_candidate
        if candidate is not None and (candidate.step_id == step_id or candidate.step_id in dependents):
            run.graph_candidate = None

    def _graph_undo(self, run: GuideRun, step_id: str) -> None:
        """The user's own "that is not done" (talk ``step_mark: not_done``), with explicit-user authority.

        Recorded as a ``user`` observation so it outranks any nomination still in flight. Unlike a later frame
        not showing an event, this explicitly retracts the assertion, including an incorrectly recorded event.
        """
        plan = self.plan
        if plan is None or step_id not in {step.id for step in plan.steps}:
            return
        run.graph_seq += 1
        self._graph_remember(run, GraphEvidence(
            task_epoch=run.task_epoch, plan_id=plan.plan_id, plan_revision=plan.revision, step_id=step_id,
            seq=run.graph_seq, source="user", status="no", frame_id="", captured_at=0.0,
            run_id="", track_id="", generation=0, committed_at=self.now(), accepted_via="talk.step_mark",
        ))
        run.graph_status[step_id] = "no"
        candidate = run.graph_candidate
        if candidate is not None and candidate.step_id == step_id:
            run.graph_candidate = None
        if step_id in run.graph_done:
            self._graph_revoke(run, step_id)
        self._graph_focus(run)

    def _graph_ack(self, run: GuideRun, step_id: str, *, accepted_via: str = "step_ack") -> bool:
        """The user's own completion of a graph node (``user``/``measure``, or a ``required`` one).

        Explicit authority, so no frame is needed — and a frame can never manufacture it. Prerequisites still
        gate it: a dependent approval is not recorded before what it depends on is done.
        """
        plan = self.plan
        if plan is None:
            return False
        blocker = pp.graph_prerequisite_block(plan, step_id, run.graph_done)
        if blocker is not None:
            self._hold(run, blocker)
            return False
        if step_id in run.graph_done:
            return False
        run.graph_seq += 1
        evidence = GraphEvidence(
            task_epoch=run.task_epoch, plan_id=plan.plan_id, plan_revision=plan.revision, step_id=step_id,
            seq=run.graph_seq, source="user", status="yes", frame_id="", captured_at=0.0,
            run_id=run.binding.run_id if run.binding else "",
            track_id=run.binding.track_id if run.binding else "",
            generation=run.binding.generation if run.binding else 0,
            committed_at=self.now(), accepted_via=accepted_via,
        )
        self._graph_remember(run, evidence)
        run.graph_done = (*run.graph_done, step_id)
        run.graph_facts[step_id] = evidence
        self.stats["graph_acked"] += 1
        self._graph_focus(run)
        self._set_blocked(None)
        return True

    def _graph_nominate(self, run: GuideRun, follow_frame: FollowFrame, seq: int) -> GraphEvidence | None:
        """A fresh follower ``yes`` on one ELIGIBLE visual node nominates it; the upper's own ``step_done``
        verdict on that same node then commits it (§15.8).

        Deterministic: the first eligible node in plan order whose reading belongs to this capture. A
        ``required``/``user``/``measure`` node is never nominated — only the user's own record finishes those.
        """
        plan = self.plan
        if plan is None or run.graph_candidate is not None:
            return None
        done = set(run.graph_done)
        for step in plan.steps:
            if step.id in done or step.required or step.check != "visual":
                continue
            if not pp.graph_observation_ready(plan, done, step.id):
                # §15.4.8: only GOAL-required prerequisite gaps block an observation; the action guards that a
                # non-goal (safety/optional) gap raises live in ``steps_ready`` instead.
                continue
            evidence = run.graph_evidence.get(step.id)
            if evidence is None or evidence.seq != seq or evidence.status != "yes":
                continue
            run.graph_candidate = evidence
            self.stats["graph_nominated"] += 1
            self._set_notice("step_candidate", "이 단계가 끝난 것으로 보입니다. 위 확인 결과를 기다리는 중입니다.")
            self._enqueue_confirm("step_done", {"step_id": step.id, "follow_frame": follow_frame,
                                                "candidate": evidence})
            return evidence
        return None

    def _graph_commit(self, run: GuideRun, candidate: GraphEvidence, answer: dict) -> bool:
        """The upper's own ``step_done`` yes on the SAME node the follower nominated commits it.

        Everything is re-checked here: it is still the pending nomination, the plan binding still holds, no
        newer capture has said ``no`` for the node (an older pending ``yes`` never overwrites a fresh negative),
        the response echoes the node, and its prerequisites are still done.
        """
        plan = self.plan
        if plan is None or run.core_mode != CORE_GRAPH:
            return False
        if run.graph_candidate is not candidate:
            self.stats["graph_stale_ignored"] += 1
            return False
        run.graph_candidate = None
        current = run.graph_evidence.get(candidate.step_id)
        if not self._is_current(run) or candidate.plan_id != plan.plan_id \
                or candidate.plan_revision != plan.revision \
                or (current is not None and current.seq > candidate.seq and current.status == "no"):
            self.stats["graph_stale_ignored"] += 1
            return False
        if answer.get("step_check") != "yes" or answer.get("step_id") != candidate.step_id:
            # no/unsure/another node: nothing moves. A fresh follower yes may nominate this node again.
            self.stats["graph_held"] += 1
            self._set_notice("step_held", "화면 확인에서 이 단계가 끝난 것으로 보이지 않아 그대로 기다립니다.")
            return False
        if candidate.step_id in run.graph_done:
            return False
        blocker = pp.graph_observation_block(plan, run.graph_done, candidate.step_id)
        if blocker is not None:
            # §15.4.8: observation eligibility — a GOAL-required prerequisite gap blocks; a non-goal one
            # (safety/optional) does not, because reading a state is not permission to perform the step.
            self._hold(run, blocker)
            return False
        run.graph_done = (*run.graph_done, candidate.step_id)
        fact = replace(candidate, committed_at=self.now(), provider=answer.get("provider"),
                       model=answer.get("model"), accepted_via="upper.bound")
        run.graph_facts[candidate.step_id] = fact
        run.graph_history = (*run.graph_history, fact)[-GRAPH_HISTORY_MAX:]
        # A slower upper confirmation must not overwrite a newer ``unsure`` reading with its older ``yes``.
        self.stats["graph_committed"] += 1
        self._graph_focus(run)
        if self.view.notice is not None and self.view.notice.code == "step_candidate":
            self.view.notice = None
            self.notice_stamp = None
        self._clear_resolved_block(run)
        return True

    def _graph_follow(self, run: GuideRun, checks, *, trigger: str, goal_seen: bool, needs_reselect: bool,
                      follow_frame: FollowFrame) -> None:
        """The graph core's whole answer to one follower capture.

        Every plan step is read (earlier/done ones included), revoked claims propagate, the suggested focus
        moves to the first enabled node, and a fresh yes on one eligible visual node is nominated. The common
        fences stay: ``needs_reselect`` voids visual applicability, and the replan triggers still fire — an
        enabled visual set that stays ``unsure``, and a stuck target.
        """
        plan = self.plan
        if plan is None or not run.steps:
            return
        done_before = run.graph_done
        if needs_reselect:
            # The anchor identity is no longer supported; past confirmations are history, not authorization.
            self._reset_graph(run, identity_changed=True)
            self.stats["graph_reselect"] += 1
        else:
            seq = self._graph_observe(run, checks, frame_id=follow_frame.frame_id,
                                      captured_at=follow_frame.captured_at, anchor=follow_frame.anchor)
            if self._graph_nominate(run, follow_frame, seq) is not None:
                return  # the confirm slot now holds the nomination; a goal check would displace it
        # The classic replan trigger, made focus-independent: it is the ENABLED VISUAL nodes that must be
        # unclear for the run to look stuck. A manual/required node has no visual reading to be unsure about,
        # and a node that is merely suggested is not something to replan around.
        visual_ids = {step.id for step in run.steps if step.check == "visual"}
        readings = [run.graph_status[sid] for sid in pp.graph_ready_ids(list(run.steps), run.graph_done)
                    if sid in visual_ids and sid in run.graph_status]
        if readings and all(reading == "unsure" for reading in readings):
            self.stats["follow_unsure"] += 1
            run.unsure_streak += 1
            if run.unsure_streak >= 2:
                run.unsure_streak = 0
                if self._request_replan(run, "unsure_twice", checks):
                    return
        else:
            run.unsure_streak = 0
            if "yes" in readings:
                self.stats["follow_done"] += 1
        count, stuck = gp.next_stuck_count(run.stuck_count, trigger=trigger,
                                           step_changed=run.graph_done != done_before)
        run.stuck_count = count
        if stuck and self._request_replan(run, "replan", checks):
            self.stats["stuck_replan"] += 1
            return
        if goal_seen and run.completion == "none":
            self._enqueue_confirm("goal_check", {"follow_frame": follow_frame})

    def _drop_candidate(self, run: GuideRun, candidate) -> None:
        """A nomination the answer could not bind is dropped: nothing moves, a fresh reading may re-nominate."""
        if isinstance(candidate, SequentialEvidence) and run.seq_candidate is candidate:
            run.seq_candidate = None
            self.stats["sequential_stale_ignored"] += 1
        elif isinstance(candidate, GraphEvidence) and run.graph_candidate is candidate:
            run.graph_candidate = None
            self.stats["graph_stale_ignored"] += 1

    def _advance_plan(self, run: GuideRun, target_index: int | None) -> bool:
        """§15.4: move forward through the gates. ``None`` = one step (its evidence says it is done).

        Returns True when the position moved; otherwise the run holds and ``state.blocked`` says why. The
        server's own ``user_done`` record is never pruned: a recorded ``step_ack`` stays done (§15.4.7).
        """
        plan = self.plan
        if plan is None:
            return False
        if target_index is None:
            adv = pp.advance_by_one(plan, step_index=run.step_index, skipped=run.skipped, user_done=run.user_done)
        else:
            adv = pp.advance_to(plan, target_index, step_index=run.step_index, skipped=run.skipped,
                                user_done=run.user_done)
        if not adv.moved:
            self._hold(run, adv.blocked)
            return False
        if adv.index != run.step_index:
            self._set_step(run, adv.index, list(adv.skipped), keep_user_done=True)
        else:  # the last step is finished: there is nowhere to move to
            run.unsure_streak = 0
            run.stuck_count = 0
            self.view.basis_age_s = None
            self._set_blocked(None)
        return True

    def _proposal_base(self, revision: int) -> Plan | None:
        """The installed plan as of the revision just before ``revision`` (they differ after a rejection).

        The copy is only a revision carrier for ``plan_policy``: it is never emitted and never becomes the
        plan the client sees (that one is always built by ``install_proposal``).
        """
        plan = self.plan
        if plan is None or revision < 2:
            return None
        if plan.revision == revision - 1:
            return plan
        return plan.model_copy(update={"revision": revision - 1})

    def _propose_replan(self, run: GuideRun, steps: list, *, reason: str, goal_when: str | None = None,
                        revision: int | None = None) -> bool:
        """§15.6: publish an approvable change instead of replacing the plan the user approved.

        The installed plan stays authoritative; ``run.plan_revision`` does follow the upstream, because every
        fence echo is judged against it and the upstream has already moved to the proposed revision.
        """
        rev = revision if revision is not None else (run.plan_revision or 0) + 1
        base = self._proposal_base(rev)
        if base is None:
            return False
        try:
            proposal = pp.make_proposal(base, reason=reason[:200], steps=list(steps),
                                        goal_when=_clip(goal_when, 60) or base.goal_when)
        except ValueError as exc:  # unknown/cyclic requires, duplicate ids, over-long text
            log.warning("proposal steps did not validate; not proposed: %s", exc)
            return False
        self.proposal = proposal
        run.plan_revision = rev
        run.replan_budget = gp.note_plan_installed(run.replan_budget, self.now())
        run.replan_epoch += 1
        self.stats["proposal"] += 1
        self._set_notice("proposal", "바꿀 계획을 변경안으로 만들었어요. 확인해 주세요.")
        self._refresh_replan_block(run)
        return True

    def _install_proposal(self, run: GuideRun) -> bool:
        """§15.6 accept: install the revision; graph facts survive only exact condition/dependency mapping."""
        proposal = self.proposal
        base = self._proposal_base(proposal.revision) if proposal is not None else None
        if proposal is None or base is None:
            return False
        installed = pp.install_proposal(base, proposal)
        self.plan = installed
        run.steps = tuple(installed.steps)
        run.plan_revision = installed.revision
        run.step_index = 0
        run.unsure_streak = 0
        run.last_checks = None
        run.skipped = ()
        run.user_done = ()
        run.stuck_count = 0
        v = self.view
        v.steps, v.plan_revision, v.step_index, v.skipped, v.user_done = (tuple(installed.steps), installed.revision,
                                                                         0, (), ())
        v.goal_when, v.target = installed.goal_when, installed.target
        if run.completion == "checking":  # that check was about the plan that just changed
            run.completion = "none"
            v.completion = "none"
        self._set_blocked(None)
        self._set_notice("replanned", "계획이 다시 짜였습니다. 남은 조건부터 안내합니다." if run.core_mode == CORE_GRAPH
                         else "계획이 다시 짜였습니다. 새 1단계부터 안내합니다.")
        self.stats["proposal_accepted"] += 1
        self.step_log.append((self.now() - run.started_at, 0, "replan"))
        self._reset_sequential(run, moved=True, replanned=True)
        self._reset_graph(run, replanned=True, previous_plan=base)
        self._refresh_replan_block(run)
        return True

    def _require_running(self, run: GuideRun | None, *, allow_paused: bool = False,
                         allow_waiting: bool = False) -> GuideRun:
        """The §15.4.1 / §15.6 / §15.7 gates a control message must pass before it may touch the run."""
        if run is None or run.halted or run.plan_id is None or self._planning() or self._reviewing() \
                or self._clarifying() \
                or (self.user_paused and not allow_paused) \
                or (self.proposal is not None and not allow_waiting):
            reason = (PROPOSAL_WAIT_REASON if self.proposal is not None
                      else CLARIFY_WAIT_REASON if self._clarifying()
                      else "진행 중인 가이드가 없습니다")
            raise EngineReject("not_running", reason)
        return run

    def _schedule_recheck(self, run: GuideRun) -> None:
        """The §6.2 completion re-check. A §15.6/§15.7 freeze re-arms it instead of dropping the check."""

        def fire() -> None:
            if not self._is_current(run) or run.completion != "checking":
                return
            if self._enqueue_blocked():
                self._schedule_recheck(run)
                return
            self._enqueue_confirm("goal_check", {"recheck": True})

        self._later(fire, COMPLETION_RECHECK_MS)

    def _reconcile_talk(self, run: GuideRun, answer: dict, s: gp.TalkRunState, *, changed: bool) -> gp.TalkRunState:
        """§15.4/§15.6 in the talk path (``runTalk`` applies its answer to the run in one step).

        A plan-changing answer (`replan`/`step_say`) is proposed, never installed; a position change still has
        to pass the §15.4 gates, and a step the user acknowledged is never un-done by a model verdict.
        """
        plan = self.plan
        if plan is None:
            return s
        ids = {step.id for step in plan.steps}
        user_done = tuple(dict.fromkeys((*run.user_done, *(i for i in s.user_done if i in ids))))
        if run.core_mode == CORE_GRAPH:
            # §15.8: an explicit user mark IS authority — a "done" is the user's own record (an ack), a
            # "not_done" retracts a claim. A ``go_to``/skip is a pointer move and never finishes a
            # node, so only the ledger changes here; the suggested focus stays deterministic.
            undone = [step_id for step_id in s.skipped if step_id in ids]
            for step_id in user_done:
                if step_id not in run.user_done:
                    self._graph_ack(run, step_id, accepted_via="talk.step_mark")
            for step_id in undone:
                self._graph_undo(run, step_id)
            recorded = tuple(sid for sid, fact in run.graph_facts.items() if fact.source == "user")
            if changed:
                parsed = self._answer_replan(answer)
                steps = parsed[0] if parsed else list(s.steps)
                goal = parsed[1] if parsed else None
                reason = (parsed[2] if parsed else None) or TALK_PROPOSAL_REASON
                self._propose_replan(run, steps, revision=s.plan_revision, goal_when=goal, reason=reason)
            return replace(s, steps=tuple(run.steps), step_index=run.step_index, skipped=tuple(run.skipped),
                           user_done=recorded, plan_revision=run.plan_revision or s.plan_revision,
                           last_checks=run.last_checks)
        if changed:
            parsed = self._answer_replan(answer)
            steps = parsed[0] if parsed else list(s.steps)
            goal = parsed[1] if parsed else None
            reason = (parsed[2] if parsed else None) or TALK_PROPOSAL_REASON
            self._propose_replan(run, steps, revision=s.plan_revision, goal_when=goal, reason=reason)
            # The installed plan keeps its steps and its position until the proposal is answered.
            return replace(s, steps=tuple(run.steps), step_index=run.step_index, skipped=tuple(run.skipped),
                           user_done=user_done, plan_revision=run.plan_revision or s.plan_revision,
                           last_checks=run.last_checks)
        protected = {step.id for step in plan.steps if step.required or step.check != "visual"}
        skipped = tuple(i for i in s.skipped if i not in protected)
        if s.step_index > run.step_index:
            self._advance_plan(run, s.step_index)
            return replace(s, step_index=run.step_index, skipped=tuple(run.skipped), user_done=user_done)
        return replace(s, skipped=skipped, user_done=user_done)

    # ================================================================================== retire

    def _is_current(self, run: GuideRun) -> bool:
        return self.guide is run and not run.aborted

    def _retire(self, *, error: tuple[str, str] | None = None, notice: tuple[str, str] | None = None,
                keep_view: bool = False, clarification: str | None = None) -> None:
        """useIntentLoop.ts:483-517. ``error`` = (code, message). Tracking stops with the guide (server choice)."""
        run = self.guide
        self.guide = None
        if run is not None:
            run.aborted = True
            current = asyncio.current_task()
            for task in list(run.tasks):
                if task is not current:
                    task.cancel()
        self._clear_timers()
        self.gate = gp.INITIAL_GATE_STATE
        self.trig = gp.INITIAL_TRIGGER_STATE
        self.appearance = gp.INITIAL_APPEARANCE_STATE
        self.pending_views = {}
        self.budget_capped = False
        self.talk_block = None
        self._hi_req = None
        # §15: a draft only exists while reviewing, so it never survives a retire; the approved plan is kept
        # with the view when the caller asked for it (``phase:"error"`` still shows what was being executed).
        if not keep_view or (self.plan is not None and self.plan.status == "draft"):
            self.plan = None
        self.proposal = None
        self.blocked = None
        self.user_paused = False
        phase = "error" if error else "idle"
        err = StateError(code=error[0][:64], message=error[1][:300], retryable=True) if error else None
        if keep_view:
            v = self.view
            v.pending = None
            v.budget_notice = None
            v.talk_pending = False
            v.talk_blocked_reason = None
            v.phase = phase
            v.error = err
            v.notice = None
            v.partial_target = v.partial_first_say = None
            # §15: whatever was being asked is over; a late answer has no id to echo any more.
            v.clarification = v.clarification_id = None
        else:
            self.view = View(phase=phase, core_mode=self.view.core_mode, error=err, clarification=clarification)
        if notice:
            self._set_notice(*notice)
        self._stop_tracking()

    # ================================================================================== tracking runs

    def _start_live_run(self, target: str) -> None:
        """useObjectTrack.ts:549-640 beginRun (startWithTarget): a new run; the live snapshot is cleared."""
        self._stop_tracking()
        if self.paused or self.user_paused:
            return  # on_resume / run_resume starts the run on the guide's (current) target
        run = LiveRun(run_id=f"r-{secrets.token_hex(4)}", target=target[:40])
        self.track = run
        self.pair = None
        self._spawn(self._live_start(run))

    async def _live_start(self, run: LiveRun) -> None:
        up = self._upstream()
        index = run.up_index
        up_run_id = f"{run.run_id}.{index}"
        try:
            if up.session_id is None:
                health = await self.ctx.health.get()
                await up.ensure_session(bool((health or {}).get("access_code_required")))
            await up.track_start(up_run_id, run.target)
        except UpstreamError as exc:
            if self.track is run and run.up_index == index:
                log.warning("track start failed: %s", exc)
                run.terminal = "unavailable"
                self.stats["track_start_failed"] += 1
                # Nothing can progress without a tracker: spec §10 tracker_unavailable -> state.error.
                if self.guide is not None and not self.guide.halted:
                    self._retire(error=("tracker_unavailable",
                                        "추적기를 시작하지 못해 가이드를 멈췄습니다. 잠시 후 다시 시작해주세요."),
                                 keep_view=True)
                self._emit()
            return
        if self.track is not run or run.up_index != index:
            self._spawn(up.track_stop(up_run_id))
            return
        run.up_run_id = up_run_id
        run.ready = True

    def _stop_tracking(self) -> None:
        run = self.track
        self.track = None
        self.pair = None
        if run is not None and run.up_run_id is not None and self.up is not None and run.terminal is None:
            self._spawn(self.up.track_stop(run.up_run_id))

    async def _reseed(self, run: LiveRun) -> None:
        """A user-drawn box (spec §5.1): a new upstream run under the same wire run_id (reseed, useObjectTrack.ts:806)."""
        old = run.up_run_id
        if old is not None and run.terminal is None and self.up is not None:
            self._spawn(self.up.track_stop(old))
        run.gen_base = (run.last_generation + 1) if run.last_generation is not None else 0
        run.up_index += 1
        run.up_run_id = None
        run.ready = False
        run.terminal = None
        run.frame_seq = 0
        run.last_version = 0
        run.up_gen_first = None
        run.target = MANUAL_SELECTION_TARGET
        self.pair = None  # useObjectTrack.ts:560 liveTrack.clear()
        await self._live_start(run)

    def _track_msg(self, header: FrameHeader, run: LiveRun | None, state: str) -> TrackMsg:
        base = {"seq": header.seq, "captured_at": header.captured_at}
        if run is None:
            return TrackMsg(state="idle", **base)
        return TrackMsg(state=state, run_id=run.run_id, track_id=run.last_track_id, generation=run.last_generation,
                        target=run.target, **base)

    async def on_frame(self, header: FrameHeader, jpeg: bytes) -> TrackMsg:
        recv = self.now()
        self.latest_frame = (jpeg, header.w, header.h, recv)
        if self._frame_future is not None and not self._frame_future.done():
            self._frame_future.set_result(None)
        guide = self.guide
        if header.select_box is not None and guide is not None and guide.plan_id is not None and not guide.halted \
                and not self._planning() and not self._reviewing() and not self._clarifying() \
                and not self.user_paused:
            self.stats["select_box"] += 1
            # Engine-owned: the WS layer may cancel this frame's task (a newer frame after 2 s), which must not
            # strand a half-started reseed; the start and its seed frame complete either way.
            return await asyncio.shield(self._spawn(self._select(header, jpeg, recv)))
        run = self.track
        if run is None:
            return TrackMsg(state="idle", seq=header.seq, captured_at=header.captured_at)
        if run.terminal is not None:
            return self._track_msg(header, run, run.terminal)
        if not run.ready:
            return self._track_msg(header, run, "acquiring")
        return await self._forward(header, jpeg, run, recv)

    async def _select(self, header: FrameHeader, jpeg: bytes, recv: float) -> TrackMsg:
        run = self.track
        if run is None:
            run = LiveRun(run_id=f"r-{secrets.token_hex(4)}", target=MANUAL_SELECTION_TARGET)
            self.track = run
            await self._live_start(run)
        else:
            await self._reseed(run)
        if self.guide is not None:
            # §15.8: a re-selected target starts a new anchor world — a nomination (and any verdict on it) from
            # the previous one no longer applies. Historical user confirmations are not new-object approvals.
            self._reset_sequential(self.guide, moved=True)
            self._reset_graph(self.guide, identity_changed=True)
        return await self._forward(header, jpeg, run, recv, seed=header.select_box)

    async def _forward(self, header: FrameHeader, jpeg: bytes, run: LiveRun, recv: float,
                       seed: Box | None = None) -> TrackMsg:
        """useObjectTrack.ts:380-494 dispatchFrame + 496-543 runLoop's error/terminal handling."""
        if not run.ready or run.terminal is not None or run.up_run_id is None:
            return self._track_msg(header, run, run.terminal or "acquiring")
        up = self._upstream()
        up_run_id = run.up_run_id
        run.frame_seq += 1
        frame_seq = run.frame_seq
        t0 = self.now()
        try:
            ans = await up.track_frame(up_run_id, _new_id("trk"), frame_seq, base64.b64encode(jpeg).decode("ascii"),
                                       seed.model_dump() if seed is not None else None)
        except UpstreamError as exc:
            if self.track is not run or run.up_run_id != up_run_id:
                return self._track_msg(header, self.track, "acquiring") if self.track else self._track_msg(header, None, "idle")
            action = decide_track_error(exc.status, exc.code)
            self.stats[f"track_error:{exc.code}"] += 1
            if action == "drop_frame":
                # No answer for this frame: report the run without a box (fail closed, never a stale box).
                last = run.last_state if run.last_state not in (None, "tracking") else "acquiring"
                return self._track_msg(header, run, last)
            # retire_run / unavailable / lost_start_race / unknown: the run cannot continue honestly.
            log.warning("track frame failed (%s): retiring run %s", exc, run.run_id)
            run.terminal = "unavailable"
            run.last_state = "unavailable"
            self.pair = None  # useObjectTrack.ts:519-527 clearVisible: no stale observation may drive the guide
            self._spawn(up.track_stop(up_run_id))
            self._set_notice("tracker_unavailable", "추적기에 연결할 수 없습니다. 대상을 직접 지정하거나 다시 시작해주세요.")
            self._emit()
            return self._track_msg(header, run, "unavailable")
        self.stats["track_frames"] += 1
        self.stats["track_rtt_ms_sum"] += int(self.now() - t0)
        if self.track is not run or run.up_run_id != up_run_id:
            cur = self.track
            return self._track_msg(header, cur, (cur.terminal or "acquiring")) if cur else self._track_msg(header, None, "idle")
        # trackFence.ts:392-403 decideUpdate: this run, this frame, a newer version.
        version = ans.get("version")
        if ans.get("run_id") != up_run_id or ans.get("frame_seq") != frame_seq or not isinstance(version, int) \
                or version <= run.last_version:
            self.stats["track_fence_dropped"] += 1
            last = run.last_state if run.last_state not in (None, "tracking") else "acquiring"
            return self._track_msg(header, run, last)
        run.last_version = version
        state = ans.get("state")
        up_gen = int(ans.get("generation", 0))
        if run.up_gen_first is None:
            run.up_gen_first = up_gen
        generation = run.gen_base + (up_gen - run.up_gen_first) if run.up_index > 1 else up_gen
        track_id = f"t{run.up_index}-{ans.get('track_id')}"[:64]
        box = None
        if state == "tracking" and isinstance(ans.get("box"), dict):
            try:
                box = Box.model_validate(ans["box"])
            except Exception:
                state = "acquiring"
        elif state == "tracking":
            state = "acquiring"
        run.last_track_id, run.last_generation, run.last_state = track_id, generation, state
        self.pair = TrackPair(run_id=run.run_id, track_id=track_id, generation=generation, state=state, box=box,
                              target=run.target, jpeg=jpeg, width=header.w, height=header.h, at_ms=recv)
        if state in ("lost", "unavailable"):
            # trackFence.ts:420-422: terminal; stop uploading, keep the snapshot (useObjectTrack.ts:530-536).
            run.terminal = state
            self._spawn(up.track_stop(up_run_id))
        if box is not None and "start_to_first_box_ms" not in self.milestones and self.guide is not None:
            self.milestones["start_to_first_box_ms"] = self.now() - self.guide.started_at
        self._tick()  # liveTrack.subscribe(tick): every accepted frame drives the loop
        self._emit()
        return TrackMsg(seq=header.seq, captured_at=header.captured_at, state=state, run_id=run.run_id,
                        track_id=track_id, generation=generation, target=run.target, box=box)

    async def on_hi_frame(self, header: FrameHeader, jpeg: bytes) -> None:
        fut = self._hi_future
        if header.hi_req is not None and header.hi_req == self._hi_req and fut is not None and not fut.done():
            fut.set_result(jpeg)

    # ================================================================================== the tick

    def _grid(self, pair: TrackPair) -> list[float] | None:
        if not pair.grid_done:
            pair.grid_done = True
            pair.grid = gp.sample_anchor_grid(pair.jpeg, pair.box) if pair.box is not None else None
        return pair.grid

    def _tick(self) -> None:
        """useIntentLoop.ts:1081-1161."""
        run = self.guide
        if self._judging_suspended() or run is None or run.halted or run.plan_id is None:
            return
        snap = self.pair
        now = self.now()
        obs = gp.TrackObservation(snap.run_id, snap.track_id, snap.generation, snap.state, snap.box) if snap else None
        aspect = snap.height / snap.width if snap and snap.width > 0 else 1.0
        self.trig, fired = gp.step_triggers(self.trig, observation=obs, now_ms=now, fence_key=FENCE_KEY,
                                            accepted=run.accepted, follow_provider=self.follow_provider,
                                            frame_aspect=aspect)
        block, _ = gp.decide_talk(running=True, talk_pending=run.talk_pending, gate=self.gate, now_ms=now)
        reason = gp.describe_talk_block(block)
        if reason != self.talk_block:
            self.talk_block = reason
            self.view.talk_blocked_reason = reason
        if self.view.notice is not None and self.notice_stamp is not None:
            if gp.notice_expired(self.notice_stamp[0], self.notice_stamp[1], run.step_index, now):
                self.view.notice = None
                self.notice_stamp = None
        drawable = snap is not None and snap.state == "tracking" and snap.box is not None \
            and now - snap.at_ms <= LIVE_BOX_FRESHNESS_MS
        if drawable and not run.first_box_seen:
            run.first_box_seen = True
        appearance_fired = False
        if gp.appearance_sample_due(self.appearance, now):
            grid = self._grid(snap) if drawable and snap is not None else None
            anchor = (snap.run_id, snap.track_id, snap.generation) if drawable and snap is not None else None
            self.appearance, appearance_fired = gp.step_appearance(
                self.appearance, now_ms=now, anchor=anchor, grid=grid, reference=run.appearance_ref)
        for kind in fired:
            if kind == "acquired":
                # Bind the role to the anchor the store reports now; a fresh verdict is pending.
                run.binding = Binding(snap.run_id, snap.track_id, snap.generation) if snap and snap.box else None
                run.warn = False
                run.accepted = None
                run.appearance_ref = None
                self.view.needs_reselect = False
                self._enqueue_follow("acquired")
            elif kind == "anchor_lost":
                self._enqueue_confirm("goal_check")
            else:
                self._enqueue_follow(kind)
        if appearance_fired:
            self._enqueue_follow("target_changed")

    # ================================================================================== dispatch gate

    def _follow_lane(self) -> str:
        return "local" if self.follow_provider in ("local", "clef") else "remote"

    def _enqueue_call(self, call: gp.PendingCall) -> None:
        run = self.guide
        if run is None or run.halted or run.plan_id is None or self._enqueue_blocked():
            # §15.8: the call never reaches the gate (no run to attach it to, or something outranks it), so a
            # nomination it carried is not waiting on anything.
            self._forget_dropped_confirm(call)
            return
        prior = self.gate.slot(call.kind).pending
        self.gate = gp.enqueue(self.gate, call)
        if call.kind == "confirm":
            landed = self.gate.slot(call.kind).pending
            # §15.8: one pending slot per stage (newest wins) — whichever confirm did not land was replaced
            # before it was ever sent, and its nomination must stop being shown as waiting.
            self._forget_dropped_confirm(call if landed is not call else None)
            self._forget_dropped_confirm(prior if landed is not prior else None)
        self._pump()

    def _forget_dropped_confirm(self, call: gp.PendingCall | None) -> None:
        """§15.8: a confirm that was never queued/sent (or was never re-armed after an error) cannot still be
        waiting for a step's confirmation — clear the nomination it was carrying."""
        run = self.guide
        if call is None or run is None or run.core_mode not in (CORE_SEQUENTIAL, CORE_GRAPH):
            return
        candidate = (call.meta or {}).get("candidate")
        if isinstance(candidate, SequentialEvidence) and run.seq_candidate is candidate:
            run.seq_candidate = None
            self.stats["sequential_dropped"] += 1
        elif isinstance(candidate, GraphEvidence) and run.graph_candidate is candidate:
            run.graph_candidate = None
            self.stats["graph_dropped"] += 1

    def _enqueue_follow(self, trigger: str) -> None:
        self._enqueue_call(gp.PendingCall("follow", trigger, self._follow_lane()))  # type: ignore[arg-type]

    def _enqueue_confirm(self, trigger: str, meta: dict | None = None) -> None:
        self._enqueue_call(gp.PendingCall("confirm", trigger, "remote", meta))

    def _wake_pump_in(self, wait_ms: float) -> None:
        """useIntentLoop.ts:1028-1041: an earlier wake replaces a later one."""
        due = self.now() + wait_ms
        current = self.gate_timer
        if current is not None and current[1] <= due:
            return
        if current is not None:
            self._clear_timer(current[0])
        holder: dict[str, int] = {}

        def fire() -> None:
            if self.gate_timer is not None and self.gate_timer[0] == holder.get("h"):
                self.gate_timer = None
            self._pump()

        holder["h"] = self._later(fire, wait_ms)
        self.gate_timer = (holder["h"], due)

    def _pump(self) -> None:
        """useIntentLoop.ts:1043-1077."""
        run = self.guide
        if run is None or run.halted or self._judging_suspended():
            return
        soonest = math.inf
        capped: tuple[float, str] | None = None
        for stage in gp.GATE_STAGES:
            now = self.now()
            d = gp.decide_dispatch(self.gate, stage, now, gp.GATE_CONFIG, run.talk_waiting)
            if d.kind == "dispatch" and d.call is not None:
                self.gate = gp.mark_dispatched(self.gate, d.call, now, gp.GATE_CONFIG)
                self._spawn(self._run_call(run, d.call), run)
            elif d.kind == "wait":
                soonest = min(soonest, d.wait_ms)
            elif d.kind == "capped":
                soonest = min(soonest, d.wait_ms)
                if capped is None or d.wait_ms < capped[0]:
                    capped = (d.wait_ms, d.limit or "budget")
        if capped is not None:
            seconds = math.ceil(capped[0] / 1000)
            cfg = gp.GATE_CONFIG
            text = (f"DeepSeek 호출 한도({round(cfg.remote_window_ms / 60_000)}분에 {cfg.remote_budget}회)에 도달했습니다. "
                    f"{seconds}초 뒤에 이어서 확인합니다." if capped[1] == "budget" else
                    f"서버 호출 한도(1분에 {cfg.remote_per_minute}회)에 도달했습니다. {seconds}초 뒤에 이어서 확인합니다.")
            if not self.budget_capped:
                self.budget_capped = True
                self.stats["budget_capped"] += 1
            self.view.budget_notice = BudgetNotice(text=text, until=_epoch_ms() + seconds * 1000)
        elif self.budget_capped:
            self.budget_capped = False
            self.view.budget_notice = None
        if math.isfinite(soonest):
            self._wake_pump_in(max(1.0, soonest))

    # ================================================================================== one call

    def _capture_scene(self, run: GuideRun, stage: str) -> dict | None:
        """The scene for a follow/confirm/talk: the newest stream frame WITH its own tracker answer.

        Browser: capture now + the store snapshot read in the same turn (useIntentLoop.ts:900-920). Here the pair
        IS one capture, so ``anchors[0].box`` is drawn on exactly the image the model sees (anchorProject.ts).
        To use a high-resolution scene later: request ``capture_hi`` here and send its JPEG with the box of the
        pair answered for it (the box is normalised, so it transfers to the larger image of the same capture).
        """
        snap = self.pair
        if snap is None or (self.track is not None and snap.run_id != self.track.run_id):
            return None
        anchor = build_anchor_ref(snap, anchor_label(run.target, snap.target))
        jpeg, captured_at = snap.jpeg, snap.at_ms
        if snap.box is None and self.latest_frame is not None and self.latest_frame[3] > snap.at_ms:
            # A boxless answer (acquiring / occluded / a terminal lost run that no longer uploads) pins no
            # geometry to its image, so the scene is the newest stream frame: a completion re-check after `lost`
            # must look at a fresh picture (useIntentLoop.ts:772-778), as the browser's fresh capture did.
            jpeg, captured_at = self.latest_frame[0], self.latest_frame[3]
        sent_box = gp.AcceptedBox(snap.run_id, snap.generation, snap.box) \
            if snap.state == "tracking" and snap.box is not None else None
        appearance = None
        if stage == "follow" and sent_box is not None:
            grid = self._grid(snap)
            if grid is not None:
                appearance = gp.AppearanceReference(snap.run_id, snap.track_id, snap.generation, tuple(grid))
        return {"frame_id": _new_id("guide"), "jpeg": jpeg, "captured_at": captured_at, "anchor": anchor,
                "sent_box": sent_box, "appearance": appearance, "run_id": snap.run_id}

    def _classify(self, run: GuideRun, stage: str, stamp: gp.SentStamp, answer: dict, replan: bool) -> str:
        """useIntentLoop.ts:549-571."""
        snap = self.pair
        return gp.classify_answer(
            stamp, answer,
            session_id=self.up.session_id if self.up else None,
            task_epoch=run.task_epoch if self._is_current(run) else None,
            run_id=snap.run_id if snap else None,
            plan_id=run.plan_id, plan_revision=run.plan_revision,
            latest_intent_seq=run.latest_seq[stage],
            anchor={"anchor_id": PRIMARY_ANCHOR_ID, "track_id": snap.track_id, "generation": snap.generation}
            if snap else None,
            replan_applied=replan,
        )

    async def _run_call(self, run: GuideRun, call: gp.PendingCall) -> None:
        """useIntentLoop.ts:840-1025 runCall."""
        stage = call.kind
        meta = dict(call.meta or {})
        held: FollowFrame | None = meta.get("follow_frame")
        if call.meta and held is not None:
            call.meta = {**meta, "follow_frame": None}
        try:
            step_index = run.step_index
            if not (0 <= step_index < len(run.steps)) or run.plan_id is None or run.plan_revision is None \
                    or run.session_id is None or self.up is None:
                return
            step = run.steps[step_index]
            snap = self.pair
            source, frame = ("fresh", None)
            if stage == "confirm":
                source, frame = gp.confirm_frame_source(call.trigger, meta, plan_id=run.plan_id,
                                                        plan_revision=run.plan_revision,
                                                        run_id=snap.run_id if snap else None)
            if source == "drop":
                held = None
                self.stats["confirm_follow_frame_dropped"] += 1
                return
            if stage == "confirm" and call.trigger in ("replan", "unsure_twice") and \
                    gp.replan_dispatch(run.replan_budget, meta.get("replan_epoch"), run.replan_epoch) == "drop":
                run.replan_budget = replace(run.replan_budget, used=max(0, run.replan_budget.used - 1))
                self.stats["replan_limited"] += 1
                self._refresh_replan_block(run)
                return
            sent_box = None
            sent_appearance = None
            if source == "follow" and frame is not None:
                frame_id, jpeg, captured_at, anchor, fence_run_id = (frame.frame_id, frame.jpeg, frame.captured_at,
                                                                     frame.anchor, frame.run_id)
            else:
                scene = self._capture_scene(run, stage)
                if scene is None:
                    # No tracked frame yet for the current run (the browser returned here too). A confirm is
                    # retried shortly instead of dropped, so a completion re-check cannot be lost (deviation).
                    if stage == "confirm":
                        self.stats["confirm_no_frame_retry"] += 1
                        self._later(lambda: (self._is_current(run) and not run.halted
                                             and self.gate.slot("confirm").pending is None
                                             and self._enqueue_call(call)), NO_FRAME_CONFIRM_RETRY_MS)
                    return
                frame_id, jpeg, captured_at, anchor = scene["frame_id"], scene["jpeg"], scene["captured_at"], scene["anchor"]
                fence_run_id = scene["run_id"]
                sent_box, sent_appearance = scene["sent_box"], scene["appearance"]
            run.intent_seq += 1
            run.latest_seq[stage] = run.intent_seq
            stamp = gp.SentStamp(run.session_id, run.plan_id, run.plan_revision, run.intent_seq,
                                 {"task_epoch": run.task_epoch, "run_id": fence_run_id}, echo_of(anchor))
            scene_payload = {"frame_id": frame_id, "image_base64": base64.b64encode(jpeg).decode("ascii"),
                             "label": SCENE_LABEL}
            self._set_pending_view(stage, Pending(stage=stage, trigger=call.trigger, since=_epoch_ms()))  # type: ignore[arg-type]
            self.trig = gp.note_stage_dispatch(self.trig, stage, self.now())
            self.stats[f"{stage}:{call.trigger}"] += 1
            self.stats[stage] += 1
            if source == "follow":
                self.stats["confirm_follow_frame"] += 1
            self._emit()
            common = {"session_id": run.session_id, "consent_ai": True, "scene": scene_payload,
                      "plan_id": run.plan_id, "plan_revision": run.plan_revision,
                      "anchors": [anchor] if anchor else [], "intent_seq": stamp.intent_seq, "fence": stamp.fence}
            if stage == "follow":
                # §15.8: the activation this question is about is fixed HERE, before the await. An answer that
                # arrives after the run moved — even if it came back to the same step — is about a different
                # activation and can neither nominate nor commit.
                asked_activation = run.seq_activation
                answer = await self.up.follow({**common, "current_step": step.id, "trigger": call.trigger})
                if not self._is_current(run):
                    self.stats["follow:late"] += 1
                    return
                run.consecutive_failures = 0
                acceptance = self._classify(run, "follow", stamp, answer, False)
                self.stats[f"follow:{acceptance}"] += 1
                follow_frame = FollowFrame(frame_id, jpeg, captured_at, anchor, fence_run_id, stamp.plan_id,
                                           stamp.plan_revision)
                self._apply_follow(run, answer, acceptance, sent_box, sent_appearance, step_index, captured_at,
                                   follow_frame, asked_activation)
            else:
                payload = {**common, "user_goal": run.goal, "trigger": call.trigger}
                if call.trigger == "step_done" and meta.get("step_id"):
                    payload["current_step"] = meta["step_id"]
                elif call.trigger in ("replan", "unsure_twice") and run.core_mode in (CORE_SEQUENTIAL, CORE_GRAPH):
                    # §15.8: the upstream's checklist for sequential starts at (and only covers) the current
                    # step, and it refuses to guess one — so a sequential replan/unsure confirm names it. A
                    # retry or recovery re-derives it from the current step at dispatch time. The graph core
                    # names the suggested focus the same way (its checklist always covers the whole plan).
                    payload["current_step"] = step.id
                if call.trigger in ("replan", "unsure_twice") and meta.get("follow_checks"):
                    payload["follow_checks"] = list(meta["follow_checks"])
                answer = await self.up.confirm(payload)
                held = None
                if not self._is_current(run):
                    self.stats["confirm:late"] += 1
                    return
                run.consecutive_failures = 0
                acceptance = self._classify(run, "confirm", stamp, answer, bool(answer.get("replan")))
                self.stats[f"confirm:{acceptance}"] += 1
                self._apply_confirm(run, answer, acceptance, meta, captured_at)
        except asyncio.CancelledError:
            raise
        except UpstreamError as exc:
            if self._is_current(run):
                self.stats[f"{stage}:failed:{exc.code}"] += 1
                retry = gp.PendingCall(call.kind, call.trigger, call.lane, {**meta, "follow_frame": held}) \
                    if held is not None else call
                self._handle_call_error(run, retry, exc)
        except Exception:
            log.exception("%s call failed", stage)
            if self._is_current(run):
                self._handle_call_error(run, call, UpstreamError(0, "internal"))
        finally:
            held = None
            if self.guide is run:
                self.gate = gp.mark_settled(self.gate, stage)
                self._set_pending_view(stage, None)
                self._pump()
            self._emit()

    def _handle_call_error(self, run: GuideRun, call: gp.PendingCall, exc: UpstreamError) -> None:
        """useIntentLoop.ts:786-838."""
        if exc.status == 0:
            run.consecutive_failures += 1
            if run.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self._forget_dropped_confirm(call)
                self._retire(error=("upstream_unreachable", "안내 서버에 연결하지 못해 가이드를 멈췄습니다."), keep_view=True)
            return
        body = exc.body
        if exc.status == 401:
            if self.up is not None:
                self.up.invalidate_session()
            self._forget_dropped_confirm(call)
            self._retire(error=("session_expired", "세션이 만료되었습니다. 가이드를 다시 시작해주세요."), keep_view=True)
            return
        if exc.status == 409:
            if exc.code == "stale_plan" and gp.is_superseded_stale(body, plan_id=run.plan_id,
                                                                   plan_revision=run.plan_revision,
                                                                   talk_pending=run.talk_pending):
                # A talk or replan moved the plan on while this call was out: its question is moot.
                self.stats["stale_superseded"] += 1
                self._forget_dropped_confirm(call)
                if call.kind == "confirm" and gp.retry_recheck_after_stale(call.meta, run.completion):
                    self._later(lambda: (self._is_current(run) and run.completion == "checking"
                                         and self._enqueue_confirm("goal_check", {"recheck": True})),
                                COMPLETION_RECHECK_MS)
                return
            rev = body.get("plan_revision")
            if exc.code == "stale_plan" and body.get("plan_id") == run.plan_id and isinstance(rev, int) \
                    and run.plan_revision is not None and rev > run.plan_revision:
                # Our plan moved on upstream without us seeing the answer that moved it (e.g. a talk/replan answer
                # dropped while disconnected): read it back (GET /api/guide/plan/current, the backend's recovery
                # read for 409 stale_plan) instead of retiring the guide.
                self.stats["plan_recovery"] += 1
                self._forget_dropped_confirm(call)
                self._spawn(self._recover_plan(run), run)
                return
            if exc.code == "task_changed":
                message = "작업 내용이 바뀌어 가이드를 멈췄습니다. 다시 시작해주세요."
            elif isinstance(body.get("plan_id"), str) and body.get("plan_id") == run.plan_id:
                message = "서버의 계획이 갱신되었습니다. 가이드를 다시 시작해주세요."
            else:
                message = "다른 계획이 시작되어 이 가이드를 멈췄습니다."
            self._forget_dropped_confirm(call)
            self._retire(error=(exc.code, message), keep_view=True)
            return
        if exc.status == 503 and exc.code == "service_unavailable":
            self._forget_dropped_confirm(call)
            self._retire(error=("service_unavailable", "안내 서버의 가이드(추종) 경로를 사용할 수 없어 가이드를 멈췄습니다."),
                         keep_view=True)
            return
        if exc.status in (503, 429):
            wait = exc.retry_after_ms if exc.retry_after_ms is not None else 1000
            self.stats["retry_after"] += 1
            self._later(lambda: (self._is_current(run) and not run.halted
                                 and self.gate.slot(call.kind).pending is None and self._enqueue_call(call)), wait)
            return
        run.consecutive_failures += 1
        if run.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._forget_dropped_confirm(call)
            self._retire(error=("provider_error", f"안내 응답이 연속으로 실패해 가이드를 멈췄습니다({exc.code})."),
                         keep_view=True)

    async def _recover_plan(self, run: GuideRun) -> None:
        if self.up is None:
            return
        try:
            current = await self.up.plan_current()
        except UpstreamError as exc:
            if self._is_current(run):
                self._retire(error=(exc.code, "서버의 계획이 갱신되었습니다. 가이드를 다시 시작해주세요."), keep_view=True)
                self._emit()
            return
        rev = current.get("plan_revision")
        if not self._is_current(run) or current.get("plan_id") != run.plan_id or not isinstance(rev, int) \
                or run.plan_revision is None or rev <= run.plan_revision:
            return
        try:
            steps = _parse_steps(current.get("steps") or [])
        except Exception:
            self._retire(error=("stale_plan", "서버의 계획이 갱신되었습니다. 가이드를 다시 시작해주세요."), keep_view=True)
            self._emit()
            return
        same_shape = [(s.id, s.done_when) for s in steps] == [(s.id, s.done_when) for s in run.steps]
        if self._gated(run):
            # §15.6: a plan the server moved on its own is still a change the user must approve.
            self._propose_replan(run, list(steps), revision=rev, goal_when=current.get("goal_when"),
                                 reason="서버의 계획이 갱신되어 변경안으로 올렸어요.")
        elif same_shape:  # a sentence rewrite: same steps, same position
            run.steps, run.plan_revision = steps, rev
            self.view.steps, self.view.plan_revision = steps, rev
            self._reset_sequential(run, moved=True, replanned=True)
            self._reset_graph(run, replanned=True)
        else:  # a replan: as installReplan
            self._install_replan(run, {"replan": {"steps": [s.model_dump() for s in steps]}, "plan_revision": rev})
        self._pump()
        self._emit()

    # ================================================================================== answers

    def _set_step(self, run: GuideRun, index: int, skipped: list[str] | None = None,
                  *, keep_user_done: bool = False) -> None:
        """useIntentLoop.ts:596-604."""
        if skipped is None:
            skipped = gp.skipped_before(run.steps, run.skipped, index)
        run.step_index = index
        run.skipped = tuple(skipped)
        if not keep_user_done:
            run.user_done = tuple(gp.skipped_before(run.steps, run.user_done, index))
        run.stuck_count = 0
        run.unsure_streak = 0
        self.blocked = None  # §15.4: a block is about the position, so moving on clears it
        v = self.view
        v.step_index, v.skipped, v.user_done, v.basis_age_s = index, run.skipped, run.user_done, None
        self.step_log.append((self.now() - run.started_at, index, "step"))
        # §15.8: a new activation begins here — a candidate (or a verdict) from the old one can never apply.
        self._reset_sequential(run, moved=True)

    def _refresh_replan_block(self, run: GuideRun) -> None:
        """useIntentLoop.ts:607-612."""
        if not self._is_current(run):
            return
        block, wait = gp.replan_block(run.replan_budget, self.now())
        self.view.replan_blocked_reason = gp.describe_replan_block(block)
        if block == "cooldown":
            self._later(lambda: self._refresh_replan_block(run), wait + 20)

    def _request_replan(self, run: GuideRun, trigger: str, checks) -> bool:
        """useIntentLoop.ts:615-625. §15.6: while a change waits for an answer, nothing asks for another."""
        if self.proposal is not None:
            self.view.replan_blocked_reason = PROPOSAL_WAIT_REASON
            return False
        now = self.now()
        if gp.replan_block(run.replan_budget, now)[0] is not None:
            self.stats["replan_limited"] += 1
            return False
        run.replan_budget = gp.spend_replan(run.replan_budget, now)
        meta: dict[str, Any] = {"replan_epoch": run.replan_epoch}
        if checks:
            meta["follow_checks"] = list(checks)
        self._enqueue_confirm(trigger, meta)
        self._refresh_replan_block(run)
        return True

    def _install_replan(self, run: GuideRun, answer: dict) -> bool:
        """useIntentLoop.ts:627-649."""
        rp = answer.get("replan")
        rev = answer.get("plan_revision")
        if not rp or run.plan_revision is None or not isinstance(rev, int) or rev <= run.plan_revision:
            return False
        try:
            steps = _parse_steps(rp.get("steps") or [])
        except Exception:
            log.warning("replan steps did not validate; ignored")
            return False
        run.steps = steps
        run.plan_revision = rev
        run.step_index = 0
        run.unsure_streak = 0
        run.last_checks = None
        run.skipped = ()
        run.user_done = ()
        run.stuck_count = 0
        run.replan_budget = gp.note_plan_installed(run.replan_budget, self.now())
        run.replan_epoch += 1
        v = self.view
        v.steps, v.plan_revision, v.step_index, v.skipped, v.user_done = steps, rev, 0, (), ()
        self._set_notice("replanned", "계획이 다시 짜였습니다. 새 1단계부터 안내합니다.")
        self.stats["replan_installed"] += 1
        self.step_log.append((self.now() - run.started_at, 0, "replan"))
        self._reset_sequential(run, moved=True, replanned=True)
        self._reset_graph(run, replanned=True)
        self._refresh_replan_block(run)
        return True

    @staticmethod
    def _answer_replan(answer: dict) -> tuple[list, str | None, str | None] | None:
        """The ``replan`` an upstream answer carries: ``(steps, goal_when, reason)``, or None when it has none."""
        rp = answer.get("replan")
        if not isinstance(rp, dict) or not rp.get("steps"):
            return None
        try:
            steps = _parse_steps(rp.get("steps") or [])
        except Exception:
            log.warning("replan steps did not validate; ignored")
            return None
        goal = rp.get("goal_when") if isinstance(rp.get("goal_when"), str) else None
        reason = rp.get("reason") if isinstance(rp.get("reason"), str) else None
        return list(steps), goal, reason

    def _propose_answer_replan(self, run: GuideRun, answer: dict) -> bool:
        """§15.6: the answer's replan becomes an approvable proposal (the installed plan stays authoritative)."""
        parsed = self._answer_replan(answer)
        if parsed is None:
            return False
        steps, goal, reason = parsed
        revision = answer.get("plan_revision")
        return self._propose_replan(run, steps,
                                    revision=revision if isinstance(revision, int)
                                    and not isinstance(revision, bool) else None,
                                    goal_when=goal,
                                    reason=reason or PROPOSAL_REASON.get(str(answer.get("trigger") or ""),
                                                                        DEFAULT_PROPOSAL_REASON))

    def _apply_follow(self, run: GuideRun, answer: dict, acceptance: str, sent_box, sent_appearance,
                      asked_index: int, captured_at: float, follow_frame: FollowFrame,
                      asked_activation: int) -> None:
        """useIntentLoop.ts:651-719."""
        if acceptance == "rejected":
            return
        if acceptance == "text_only":
            self.view.basis_age_s = float(gp.capture_age_seconds(captured_at, self.now()))
            return
        if sent_box is not None:
            run.accepted = sent_box
        run.appearance_ref = sent_appearance
        needs_reselect = bool(answer.get("needs_reselect"))
        run.warn = needs_reselect
        self.view.needs_reselect = needs_reselect
        self.view.basis_age_s = None

        checks = tuple(c for c in answer.get("step_checks") or [] if isinstance(c, dict))
        if run.core_mode == CORE_GRAPH:
            # §15.8: the backend answers about EVERY plan step (done ones included); the whole reading is the
            # core's own path — no index comparison, no pointer-based move.
            run.last_checks = checks
            self._graph_follow(run, checks, trigger=answer.get("trigger", ""),
                               goal_seen=answer.get("goal_seen") == "yes", needs_reselect=needs_reselect,
                               follow_frame=follow_frame)
            return
        if run.core_mode == CORE_SEQUENTIAL and 0 <= asked_index < len(run.steps):
            # §15.8: the backend asks the follower about the current step alone (``checklist_ids``); should a
            # stale or other-step verdict arrive anyway, it can never move the screen or pass a step here.
            asked_id = run.steps[asked_index].id
            checks = tuple(c for c in checks if c.get("step_id") == asked_id)
        previous = run.last_checks
        run.last_checks = checks
        step_before = run.step_index
        current = gp.current_step_visible(run.steps, asked_index, checks)
        replan_asked = False
        if current == "yes":
            self.stats["follow_done"] += 1
        if current == "unsure":
            self.stats["follow_unsure"] += 1
            run.unsure_streak += 1
            if run.unsure_streak >= 2:
                run.unsure_streak = 0
                replan_asked = True
                self._request_replan(run, "unsure_twice", checks)
        else:
            run.unsure_streak = 0
        decision = gp.decide_follow_step(run.steps, asked_index, run.step_index, checks, previous)
        if run.core_mode == CORE_SEQUENTIAL:
            # §15.8: only the CURRENT step's own verdict — on the activation it was ASKED about (§ the value
            # captured before the await), with the anchor still accepted — nominates. The upper's own
            # ``step_done`` verdict then commits exactly one step; a verdict about another step, a replayed
            # frame, an anchor that needs re-selecting, or an activation that has since ended moves nothing.
            if current == "yes" and not needs_reselect and asked_activation == run.seq_activation \
                    and run.step_index == asked_index and 0 <= asked_index < len(run.steps):
                step_id = run.steps[asked_index].id
                already_confirmed = any(e.activation == asked_activation and e.step_id == step_id
                                        for e in run.confirmed_steps)
                if run.seq_candidate is None and not already_confirmed \
                        and follow_frame.frame_id not in run.confirmed_frame_ids:
                    candidate = self._nominate_sequential(run, asked_index, step_id, follow_frame, asked_activation)
                    self._enqueue_confirm("step_done", {
                        "step_id": step_id, "follow_frame": follow_frame, "candidate": candidate})
            elif current == "yes" and decision.kind in ("advance", "skip"):
                # The follower said this step looks done, but on stale ground (another activation, a moved
                # index, or a bad anchor): discard it — never apply a verdict from a world that is gone.
                self.stats["sequential_stale_ignored"] += 1
        elif decision.kind == "pendingSkip":
            self.stats["follow_pending_skip"] += 1
        elif decision.kind in ("advance", "skip"):
            if decision.kind == "skip":
                self.stats["follow_skip"] += 1
            if run.plan_mode:
                # §15.4: the frame never finishes a required or `check:"user"` step, and never passes one.
                target = None if decision.kind == "advance" else decision.to_index
                if not self._advance_plan(run, target):
                    return
            elif decision.kind == "skip":
                self._set_step(run, decision.to_index, gp.add_skipped(run.skipped, decision.skipped_ids))
            else:
                self._set_step(run, decision.to_index)
            # The large model re-checks the step the move relied on, ON THE FRAME THE FOLLOWER JUDGED.
            self._enqueue_confirm("step_done", {"from_index": decision.from_index, "to_index": run.step_index,
                                                "step_id": decision.step_id, "follow_frame": follow_frame})
        count, stuck = gp.next_stuck_count(run.stuck_count, trigger=answer.get("trigger", ""),
                                           step_changed=run.step_index != step_before)
        run.stuck_count = count
        if stuck and not replan_asked and self._request_replan(run, "replan", checks):
            self.stats["stuck_replan"] += 1
        if decision.kind in ("advance", "skip"):
            return
        if answer.get("goal_seen") == "yes" and run.completion == "none":
            self._enqueue_confirm("goal_check", {"follow_frame": follow_frame})

    def _apply_confirm(self, run: GuideRun, answer: dict, acceptance: str, meta: dict, captured_at: float) -> None:
        """useIntentLoop.ts:721-784."""
        candidate = meta.get("candidate")
        same_task_plan = (answer.get("fence_echo") or {}).get("task_epoch") == run.task_epoch \
            and self._is_current(run) and run.plan_id is not None
        if same_task_plan and acceptance != "rejected":
            if self._gated(run):
                # §15.6: a mid-run change is proposed, not installed.
                self._propose_answer_replan(run, answer)
            else:
                self._install_replan(run, answer)
        if acceptance != "bound":
            if isinstance(candidate, (SequentialEvidence, GraphEvidence)):
                # nomination is dropped and nothing moves; a fresh follower yes may nominate again.
                self._drop_candidate(run, candidate)
            if meta.get("recheck") and run.completion == "checking":
                run.completion = "none"
                self.view.completion = "none"
                self._set_notice("recheck_other_frame", "완료 재확인이 다른 화면 기준이라 확정하지 않았습니다.")
            if acceptance == "text_only":
                self.view.basis_age_s = float(gp.capture_age_seconds(captured_at, self.now()))
            return
        if isinstance(candidate, SequentialEvidence):
            # §15.8: this IS the sequential step's commit or hold; the completion logic below still runs, so an
            # original goal can become checkable without pretending a user/measure/required ack happened.
            self._settle_sequential(run, candidate, answer)
        elif isinstance(candidate, GraphEvidence):
            # §15.8 graph: the upper's own ``step_done`` yes on the nominated node commits it; the completion
            # logic below still runs, on the same classic goal predicate.
            self._graph_commit(run, candidate, answer)
        if answer.get("needs_clarification") and answer.get("clarification_prompt"):
            self._set_notice("clarification", str(answer["clarification_prompt"]))
        snap = self.pair
        decision = gp.decide_confirm(answer, meta, step_index=run.step_index, completion=run.completion,
                                     track_lost=snap is not None and snap.state == "lost",
                                     user_done_ids=run.user_done)
        if decision.completion in ("start_check", "confirmed") and not self._completion_allowed(run):
            # §15.4.5: the picture looks finished but a required check is still empty — stay instead of
            # pretending it happened. A completion check in flight is withdrawn, so nothing is left hanging.
            self.stats["completion_held"] += 1
            if decision.completion == "confirmed":
                run.completion = "none"
                self.view.completion = "none"
            self._hold_completion(run)
            return
        if decision.completion == "confirmed":
            run.completion = "confirmed"
            run.halted = True
            # §15.4.8: freeze what was still unverified at the moment the goal was confirmed, so a completed
            # scenario can show the goal plus its outstanding mandatory checks (never as the goal unmet).
            if run.core_mode == CORE_GRAPH and self.plan is not None:
                run.pending_required = tuple(pp.graph_pending_required_checks(self.plan, run.graph_done))
            self._clear_timers()
            self.gate = gp.INITIAL_GATE_STATE
            # The goal is reached: a pending change is moot (and a proposal may not outlive `running`).
            self.proposal = None
            v = self.view
            # Wire convention (spec §6.2/§11): phase stays "running" until confirm_done; loops are halted.
            v.completion, v.pending, v.notice = "confirmed", None, None
            self.pending_views = {}
            self.milestones.setdefault("start_to_confirmed_ms", self.now() - run.started_at)
            self.step_log.append((self.now() - run.started_at, run.step_index, "confirmed"))
            return
        if decision.completion == "recheck_failed":
            run.completion = "none"
            self.view.completion = "none"
            self._set_notice("recheck_failed", "재확인에서 완료로 보이지 않았습니다. 안내를 계속합니다.")
            return
        if decision.revert_to is not None:
            self.stats["confirm_reversals"] += 1
            self._set_step(run, decision.revert_to)
            self._set_notice("reverted", "확인해 보니 이전 단계가 아직 끝나지 않았습니다. 이전 단계로 돌아갑니다.")
        if decision.completion == "start_check":
            run.completion = "checking"
            self.view.completion = "checking"
            self.milestones.setdefault("start_to_checking_ms", self.now() - run.started_at)
            self._schedule_recheck(run)
        if decision.lost_notice:
            self._set_notice("target_lost",
                             "대상을 놓쳤습니다. 화면에 대상을 다시 보여주거나 “대상 직접 지정”으로 다시 지정하세요.")

    # ================================================================================== engine protocol: guide

    async def on_start(self, goal: str, context: str | None, *, plan_mode: bool = False,
                       plan_model: str = DEFAULT_PLAN_MODEL, core_mode: str = DEFAULT_CORE_MODE,
                       materials: tuple[MaterialInput, ...] = (),
                       reference_images: tuple[ReferenceImage, ...] = ()) -> None:
        """useIntentLoop.ts:1175-1324 start(). §15: ``plan_mode`` stops at ``reviewing`` until approval;
        ``plan_model`` picks the upstream planning profile the upstream stores with the plan; ``core_mode``
        picks the guide core (``classic`` whole-remaining checklist vs ``sequential`` current step only vs
        ``graph`` every step with an explicit per-node ledger).

        ``reference_images`` are the user's *separate* reference photos (a close-up, a label) — never the
        current scene, which stays the camera frame. They are validated here, before any provider call, and
        kept only in the new run's memory. A start after a clarification is allowed (the run is replaced), but
        a late ``plan_answer`` of the old run is refused by its id."""
        problem = reference_images_problem(reference_images)
        if problem is not None:
            raise EngineReject(problem[0], problem[1], retryable=False)
        if self.guide is not None and self.guide.plan_id is None:
            raise EngineReject("plan_in_flight", "계획을 세우는 중입니다")
        self._ensure_ticker()
        self._retire()
        self._run_counter += 1
        now = self.now()
        run = GuideRun(id=self._run_counter, task_epoch=_new_id("task"), goal=goal.strip(),
                       context=(context or "").strip() or None, started_at=now, plan_mode=bool(plan_mode),
                       plan_model=str(plan_model), core_mode=str(core_mode), materials=tuple(materials),
                       reference_images=tuple(reference_images),
                       replan_budget=gp.fresh_replan_budget(now))
        self.guide = run
        self.milestones = {}
        self.step_log = []
        self.view = View(phase="planning", core_mode=run.core_mode,
                         pending=Pending(stage="plan", trigger="start", since=_epoch_ms()))
        self._emit()
        self._request_hi()
        self._spawn(self._plan_flow(run), run)

    def _request_hi(self) -> None:
        """Ask for one high-resolution frame, the way ``start`` does (§6.5; ``_plan_scene`` waits 2 s)."""
        loop = asyncio.get_running_loop()
        self._hi_future = loop.create_future()
        self._hi_req = self.sink.capture_hi()
        fut = self._hi_future
        self._later(lambda: fut.done() or fut.set_result(None), HI_FRAME_WAIT_MS)

    async def _plan_scene(self, run: GuideRun) -> bytes | None:
        """The high-resolution answer if it arrives within 2 s, else the latest stream frame (spec §6.5)."""
        fut = self._hi_future
        if fut is not None:
            jpeg = await fut
            if jpeg is not None:
                self.stats["plan_hi_frame"] += 1
                return jpeg
        if self.latest_frame is None:
            self._frame_future = asyncio.get_running_loop().create_future()
            f = self._frame_future
            self._later(lambda: f.done() or f.set_result(None), FIRST_FRAME_WAIT_MS)
            await f
        if self.latest_frame is None:
            return None
        self.stats["plan_stream_frame"] += 1
        return self.latest_frame[0]

    async def _plan_flow(self, run: GuideRun, *, answer_turn: bool = False) -> None:
        """One plan request for ``run``: the first turn after ``start``, or the turn after an answer.

        Both turns dispatch the same body; an answer turn reuses the upstream session the question came from,
        so the upstream keeps the task (goal, context, materials, references, prior answers) without anything
        being restarted. ``answer_turn`` only skips the one-time health/session setup.
        """
        up = self._upstream()
        try:
            if not answer_turn or run.session_id is None:
                health = await self.ctx.health.get()
                self.follow_provider = (health or {}).get("follow_provider")
                run.session_id = await up.ensure_session(bool((health or {}).get("access_code_required")))
                if not self._is_current(run):
                    return
            jpeg = await self._plan_scene(run)
            if not self._is_current(run):
                return
            if jpeg is None:
                self._plan_failed(run, answer_turn, "no_frame",
                                  "카메라 화면이 없습니다. 카메라를 켠 뒤 다시 보내 주세요.")
                return

            def on_partial(name: str, value: str) -> None:
                if not self._is_current(run):
                    return
                if name == "target":
                    self.view.partial_target = value
                    self.milestones.setdefault("start_to_partial_target_ms", self.now() - run.started_at)
                else:
                    self.view.partial_first_say = value
                    self.milestones.setdefault("start_to_first_say_ms", self.now() - run.started_at)
                self._emit()

            payload = self._plan_payload(run, jpeg)
            if self.paused:  # spec §9: nothing is dispatched while the client is away
                await self._resumed.wait()
                if not self._is_current(run):
                    return
            self.gate = gp.mark_plan(self.gate, True, self.now())
            self.stats["plan"] += 1
            plan = await self._call_plan(run, payload, on_partial)
            if self.guide is run:
                self.gate = gp.mark_plan(self.gate, False, self.now())
            if plan is None or not self._is_current(run):
                return
            self.milestones["start_to_final_plan_ms"] = self.now() - run.started_at
            self._finish_plan_turn(run, plan)
        except asyncio.CancelledError:
            raise
        except UpstreamError as exc:
            if self.guide is run:
                self.gate = gp.mark_plan(self.gate, False, self.now())
            if not self._is_current(run):
                return
            self.stats[f"plan:failed:{exc.code}"] += 1
            if exc.status == 401:
                up.invalidate_session()
                run.session_id = None
            self._plan_failed(run, answer_turn, exc.code or "provider_error", describe_plan_error(exc))
        except Exception as exc:
            log.exception("plan flow failed")
            if self._is_current(run):
                self._plan_failed(run, answer_turn, "provider_error",
                                  f"계획을 받지 못했습니다({type(exc).__name__}).")

    def _plan_failed(self, run: GuideRun, answer_turn: bool, code: str, text: str) -> None:
        self.gate = gp.mark_plan(self.gate, False, self.now())
        if answer_turn and run.answers:
            # Preserve the task and accepted evidence. Only an explicit resubmission can retry;
            # a fresh question id fences delayed duplicates from the failed request.
            run.retrying_answer = True
            self._set_notice("plan_answer_failed", text)
            self._enter_clarifying(run, {}, run.answers[-1].question)
        else:
            self._retire(error=(code, text))
            self._emit()

    def _plan_payload(self, run: GuideRun, jpeg: bytes) -> dict:
        """The plan request body: this run's goal/context/materials plus everything §15 adds."""
        payload = {
            "session_id": run.session_id, "consent_ai": True,
            "scene": {"frame_id": _new_id("guide"), "image_base64": base64.b64encode(jpeg).decode("ascii"),
                      "label": SCENE_LABEL},
            "user_goal": run.goal, "context": run.context,
        }
        # §15: the run's planning profile rides every plan request (review or immediate), and the upstream
        # stores it with the plan so confirm/talk/replan cannot silently use another profile. There is no
        # ``plan_mode`` selector upstream; review-before-execution stays Live-side policy.
        payload["plan_model"] = run.plan_model
        # §15.8: the guide core rides the plan request too. The upstream stores it with the plan (part of
        # the task identity) so a sequential plan always gets the current-step-only follow checklist.
        payload["core_mode"] = run.core_mode
        # §15: novice assistance IS the review boundary here. ``research`` asks for real researched facts —
        # only the profile with a hosted search tool can do that, so the DeepSeek profile is never told it has
        # one (a claim it could not honour). Both flags stay constant for the whole run: the upstream treats a
        # change as a new task.
        payload["assisted"] = run.plan_mode
        payload["research"] = run.plan_mode and run.plan_model == RESEARCH_PLAN_MODEL
        if run.materials:
            # §15.3: the extracted material texts ride the plan PROMPT as their own field (never `context`).
            payload["materials"] = [m.model_dump(exclude_none=True) for m in run.materials]
        if run.answers:
            # §15: the user's own answers are evidence for the planner. The goal itself is never rewritten.
            payload["answers"] = [a.model_dump() for a in run.answers]
        if run.reference_images:
            # §15: the user's reference photos are separate evidence, never the scene, and they ride every
            # later request of the run so the planner keeps them. They are never logged or echoed to the client.
            payload["reference_images"] = [r.model_dump(exclude_none=True) for r in run.reference_images]
        return payload

    async def _call_plan(self, run: GuideRun, payload: dict, on_partial) -> dict | None:
        """The plan call with the server-side busy retry. ``None`` when the run was superseded while waiting."""
        up = self._upstream()
        attempt = 0
        while True:
            try:
                return await up.plan(payload, on_partial)
            except UpstreamError as exc:
                # Server-side retry of a busy plan lane (an abandoned plan still finishing upstream).
                if exc.status == 503 and exc.code == "provider_busy" and attempt < PLAN_BUSY_RETRIES:
                    attempt += 1
                    self.stats["plan_busy_retry"] += 1
                    await self._sleep(exc.retry_after_ms or 1000)
                    if not self._is_current(run):
                        return None
                    continue
                raise

    def _finish_plan_turn(self, run: GuideRun, plan: dict) -> None:
        """What one plan answer means: ask the run's next question, install the plan, or end with the reason."""
        run.retrying_answer = False
        selection = plan.get("selection") or {}
        target = (selection.get("target") or "").strip() if selection.get("status") == "selected" else ""
        prompt = plan.get("clarification_prompt")
        if run.plan_mode and plan.get("needs_clarification") and isinstance(prompt, str) and prompt.strip():
            # §15: review mode turns the question into a two-turn exchange instead of ending the run; the other
            # mode keeps the existing "stop and say why" behaviour.
            if len(run.answers) >= MAX_CLARIFICATION_ANSWERS:
                self.stats["plan_clarification_limit"] += 1
                self._retire(notice=("clarification_limit",
                                     "질문이 너무 많아 계획을 세우지 못했습니다. 목표를 조금 더 구체적으로 적어 다시 시작해 주세요."))
                self._emit()
                return
            self._enter_clarifying(run, plan, prompt.strip())
            return
        if not target or plan.get("needs_clarification") or not plan.get("steps"):
            why = prompt or selection.get("rationale") or \
                "이 화면에서 다룰 대상을 고르지 못했습니다. 대상을 화면에 보여주고 다시 시작해주세요."
            self.stats["plan_clarification"] += 1
            self._retire(clarification=str(why)[:CLARIFICATION_MAX])
            self._emit()
            return
        steps = _parse_steps(plan["steps"])
        run.plan_id = plan["plan_id"]
        run.plan_revision = plan["plan_revision"]
        run.steps = steps
        run.goal_when = plan.get("goal_when") or ""
        run.target = target
        # §15: the public facts the planner reported (already only well-formed public URLs) stay on the run so
        # the same sources reach the reviewed Plan, and a later replan keeps them. An answer that reports none
        # never wipes what this task already read (the upstream accumulates them per task).
        sources = _parse_research_sources(plan.get("research_sources"))
        if sources:
            run.research_sources = sources
        v = self.view
        v.plan_id, v.plan_revision, v.steps, v.goal_when, v.target = (run.plan_id, run.plan_revision, steps,
                                                                      run.goal_when, target)
        v.step_index, v.skipped, v.partial_target, v.partial_first_say = 0, (), None, None
        v.research_sources = run.research_sources
        self.milestones.setdefault("start_to_first_say_ms", self.now() - run.started_at)
        if self._uses_plan(run):
            # §15.1/§15.8: the plan the client sees. A reviewed run shows a draft; the sequential/graph core
            # executes immediately but still needs the plan model for its required/prerequisite gates.
            self.plan = Plan(plan_id=run.plan_id[:64], revision=max(1, run.plan_revision),
                             status="draft" if run.plan_mode else "approved",
                             approved_revision=None if run.plan_mode else max(1, run.plan_revision),
                             target=anchor_label(target, None),
                             goal_when=_clip(run.goal_when, 60) or "목표 상태",
                             materials=list(pp.materials_from_inputs(run.materials, now_ms=_epoch_ms())),
                             steps=list(steps),
                             research_sources=list(run.research_sources))
        if run.plan_mode:
            # §15.1 execution boundary: the draft is reviewed before anything is tracked or judged. The
            # upstream's steps are mapped as they are — a field it does not send is left at its default.
            v.phase, v.pending = "reviewing", None
            self.milestones["start_to_reviewing_ms"] = self.now() - run.started_at
            self.stats["plan_mode"] += 1
            self._emit()
            return
        run.replan_budget = gp.fresh_replan_budget(self.now())
        self.milestones["start_to_running_ms"] = self.now() - run.started_at
        v.phase, v.pending = "running", None
        self.step_log.append((self.now() - run.started_at, 0, "running"))
        self._init_graph(run)
        self._refresh_replan_block(run)
        self._start_live_run(target)
        self._emit()

    def _enter_clarifying(self, run: GuideRun, plan: dict, question: str) -> None:
        """§15: the planner needs one answer before it can write the plan — keep the run and ask it."""
        run.clarification = question[:CLARIFICATION_MAX]
        run.clarification_id = _new_id("q")
        # The upstream registers a plan record even for a question, and it accumulates the public sources it
        # has read for this task. Keeping both is what makes the answer continue THIS task; a fresh id per
        # question is what makes an answer to a question that has been replaced harmless.
        pair = _plan_pair(plan)
        if pair is not None:
            run.plan_id, run.plan_revision = pair
        sources = _parse_research_sources(plan.get("research_sources"))
        if sources:
            run.research_sources = sources
        v = self.view
        v.phase = "clarifying"
        v.clarification = run.clarification
        v.clarification_id = run.clarification_id
        v.research_sources = run.research_sources
        v.pending = None
        v.partial_target = v.partial_first_say = None
        self.milestones.setdefault("start_to_clarifying_ms", self.now() - run.started_at)
        self.stats["plan_clarifying"] += 1
        self._emit()

    async def on_plan_answer(self, clarification_id: str, answer: str,
                             reference_images: tuple[ReferenceImage, ...] | None = None) -> None:
        """§15: the answer to the run's one question, then one more plan turn.

        A supplied ``reference_images`` tuple REPLACES the run's reference set; ``None`` keeps it. Everything up
        to the phase change is synchronous, so a duplicate answer — or one racing a newer question — cannot be
        taken twice: the id it had to echo is gone the moment the answer is accepted.
        """
        run = self.guide
        if run is None or not run.plan_mode or run.clarification_id is None or not self._clarifying():
            raise EngineReject("no_clarification", "지금은 답할 질문이 없습니다", retryable=False)
        if clarification_id != run.clarification_id:
            raise EngineReject("clarification_stale", "이미 지난 질문입니다. 화면에 보이는 질문에 답해 주세요.",
                               retryable=False)
        if reference_images is not None:
            problem = reference_images_problem(reference_images)
            if problem is not None:
                raise EngineReject(problem[0], problem[1], retryable=False)
            run.reference_images = tuple(reference_images)
        prior = run.answers[:-1] if run.retrying_answer else run.answers
        run.answers = (*prior, PlanAnswer(question=run.clarification, answer=answer))
        run.clarification = None
        run.clarification_id = None
        v = self.view
        v.phase, v.pending = "planning", Pending(stage="plan", trigger="start", since=_epoch_ms())
        v.clarification = v.clarification_id = None
        v.partial_target = v.partial_first_say = None
        v.notice = None
        self.stats["plan_answer"] += 1
        self._emit()
        # The next turn reads a FRESH current frame; the run itself (goal, context, materials, references,
        # answers, upstream session, plan profile) is untouched, so the goal is never restarted.
        self._request_hi()
        self._spawn(self._plan_flow(run, answer_turn=True), run)

    async def on_stop(self) -> None:
        self._retire()
        self._emit()

    # ---------------------------------------------------------------------------- §15: review, approve, run

    async def on_plan_edit(self, edit: PlanEditMsg) -> None:
        """§15.5: edit the draft. Nothing upstream, no tracker, and no revision the upstream knows is touched."""
        plan = self.plan
        if plan is None:
            raise EngineReject("not_reviewing", "검토 중인 계획이 없습니다", retryable=False)
        if plan.status != "draft":
            raise EngineReject("plan_locked", "승인된 계획은 고칠 수 없습니다. 새로 시작하거나 초안을 다시 만드세요.",
                               retryable=False)
        try:
            self.plan = pp.apply_plan_edit(plan, edit)
        except pp.PlanEditError as exc:
            raise EngineReject("invalid_edit", str(exc), retryable=False) from None
        v = self.view
        v.steps, v.goal_when, v.plan_revision = tuple(self.plan.steps), self.plan.goal_when, self.plan.revision
        v.step_index = min(v.step_index, len(self.plan.steps) - 1)
        self.stats["plan_edit"] += 1
        self._emit()

    async def on_plan_approve(self) -> None:
        """§15.5: register the reviewed plan upstream, pin execution to the pair it returns, then run.

        The upstream is what every fence is judged against, so the pair ``/api/guide/plan/approve`` returns
        becomes the run's authoritative ``plan_id``/``plan_revision`` — the approved steps never execute under a
        revision the upstream has not adopted.
        """
        run = self.guide
        if run is None or self.plan is None or self.plan.status != "draft" or not self._reviewing() \
                or run.plan_id is None or run.plan_revision is None:
            raise EngineReject("not_reviewing", "검토 중인 계획이 없습니다", retryable=False)
        self.plan = pp.approve_plan(self.plan)
        body: dict[str, Any] = {
            "session_id": run.session_id, "plan_id": run.plan_id, "plan_revision": run.plan_revision,
            "steps": [s.model_dump() for s in self.plan.steps], "goal_when": self.plan.goal_when,
            "user_goal": run.goal, "context": run.context,
        }
        current = await self._approve_upstream(run, body)
        if current is None or not self._is_current(run) or not self._pin_upstream_plan(run, current):
            return  # retired (provider_error) or replaced meanwhile: never continue on diverged fences
        run.step_index = 0
        run.skipped = ()
        run.user_done = ()
        run.stuck_count = 0
        run.unsure_streak = 0
        run.last_checks = None
        run.replan_budget = gp.fresh_replan_budget(self.now())
        self._set_blocked(None)
        v = self.view
        v.phase, v.pending = "running", None
        v.goal_when, v.target = self.plan.goal_when, self.plan.target
        v.steps, v.step_index, v.skipped, v.user_done = tuple(self.plan.steps), 0, (), ()
        self.milestones["start_to_running_ms"] = self.now() - run.started_at
        self.step_log.append((self.now() - run.started_at, 0, "running"))
        self.stats["plan_approved"] += 1
        self._init_graph(run)
        self._refresh_replan_block(run)
        self._start_live_run(run.target)
        self._emit()

    async def _approve_upstream(self, run: GuideRun, body: dict) -> dict | None:
        """POST the approved plan; a ``409 stale_plan`` re-reads ``/plan/current`` and retries exactly once.

        ``None`` means the run was retired — the failure was surfaced as ``provider_error``. A returned dict is
        the upstream's new current plan, whose pair becomes the run's authoritative one.
        """
        up = self._upstream()
        try:
            return await up.plan_approve(body)
        except UpstreamError as exc:
            if exc.status != 409 or exc.code != "stale_plan":
                self._fail_approve(run, exc)
                return None
        try:
            current = await up.plan_current()
        except UpstreamError as exc:
            self._fail_approve(run, exc)
            return None
        if not self._is_current(run):
            return None
        pair = _plan_pair(current)
        if pair is None:
            self._fail_approve(run, UpstreamError(502, "invalid_response"))
            return None
        try:
            return await up.plan_approve({**body, "plan_id": pair[0], "plan_revision": pair[1]})
        except UpstreamError as exc:
            self._fail_approve(run, exc)
            return None

    def _fail_approve(self, run: GuideRun, exc: UpstreamError) -> None:
        """The approved plan could not be registered: surface it and end the run (no diverged fences)."""
        if not self._is_current(run):
            return
        self.stats[f"plan_approve:failed:{exc.code}"] += 1
        self._retire(error=("provider_error", f"승인한 계획을 서버에 반영하지 못했습니다({exc.code}). 다시 시작해 주세요."))
        self._emit()

    def _pin_upstream_plan(self, run: GuideRun, current: dict) -> bool:
        """Adopt the ``plan/current`` shape an approve/revert returned: its pair, steps and goal condition."""
        pair = _plan_pair(current)
        raw_steps = current.get("steps")
        if pair is None or not isinstance(raw_steps, list):
            return False
        try:
            steps = _parse_steps(raw_steps)
        except Exception:
            log.warning("upstream returned plan steps that did not validate; keeping the local plan")
            return False
        pid, rev = pair
        goal_when = current.get("goal_when") if isinstance(current.get("goal_when"), str) else run.goal_when
        run.plan_id, run.plan_revision, run.steps, run.goal_when = pid, rev, steps, goal_when
        plan = self.plan
        if plan is not None:
            self.plan = plan.model_copy(update={
                "plan_id": pid[:64], "revision": max(1, rev),
                "approved_revision": max(1, rev) if plan.status == "approved" else plan.approved_revision,
                "steps": list(steps), "goal_when": goal_when or plan.goal_when,
            })
        v = self.view
        v.plan_id, v.plan_revision, v.steps, v.goal_when = pid, max(1, rev), tuple(steps), goal_when
        # Pending responses never cross revisions. Exact confirmed conditions may retain applicability.
        self._reset_sequential(run, moved=True, replanned=True)
        self._reset_graph(run, replanned=True, previous_plan=plan)
        return True

    async def on_plan_discard(self) -> None:
        """§15.5: 초안 버리기 — the same teardown ``stop`` uses."""
        if self.plan is None or self.plan.status != "draft":
            raise EngineReject("not_reviewing", "검토 중인 계획이 없습니다", retryable=False)
        self.stats["plan_discarded"] += 1
        self._retire()
        self._emit()

    async def on_run_pause(self) -> None:
        """§15.7: judgement and progression stop, tracking is released, and only ``run_resume`` lifts it."""
        run = self._require_running(self.guide, allow_waiting=True)
        self.user_paused = True
        self.stats["run_paused"] += 1
        # §15.8: nothing is being judged while paused, so no nomination is still waiting on an answer.
        self._reset_sequential(run, moved=True)
        self._reset_graph(run)
        self._stop_tracking()
        self._emit()

    async def on_run_resume(self) -> None:
        """§15.7: resume from the newest frame (the first frames are ``acquiring`` again)."""
        run = self._require_running(self.guide, allow_paused=True, allow_waiting=True)
        self.user_paused = False
        self.stats["run_resumed"] += 1
        self._start_live_run(run.target)
        self._pump()
        self._emit()

    async def on_step_ack(self, step_id: str) -> None:
        """§15.4.4: the user's own completion of a ``user``/``measure`` step — or, in graph mode, a ``required``
        one, which a frame must never finish."""
        run = self._require_running(self.guide, allow_waiting=True)
        plan = self.plan
        if plan is None:
            raise EngineReject("not_running", "진행 중인 가이드가 없습니다")
        if run.core_mode == CORE_GRAPH:
            try:
                pp.graph_ack_index(plan, step_id)
            except pp.PlanEditError as exc:
                raise EngineReject("invalid_edit", str(exc), retryable=False) from None
            self.stats["step_ack"] += 1
            if self.proposal is not None:
                # §15.6: the confirmation is recorded, but nothing moves while the change waits for an answer.
                accepted = True
            else:
                accepted = self._graph_ack(run, step_id)
            if accepted:
                run.user_done = tuple(dict.fromkeys((*run.user_done, step_id)))
                self.view.user_done = run.user_done
            self._emit()
            return
        try:
            index = pp.ack_index(plan, step_id)
        except pp.PlanEditError as exc:
            raise EngineReject("invalid_edit", str(exc), retryable=False) from None
        run.user_done = tuple(dict.fromkeys((*run.user_done, step_id)))
        self.view.user_done = run.user_done
        self.stats["step_ack"] += 1
        if self.proposal is not None:
            # §15.6: the confirmation is recorded, but nothing moves while the change waits for an answer.
            self._emit()
            return
        if index == run.step_index:
            # The ack is the only thing that finishes this step; the gates still decide whether it may move.
            if self._advance_plan(run, None):
                self._enqueue_confirm("step_done", {"from_index": index, "to_index": run.step_index, "step_id": step_id})
        else:
            self._clear_resolved_block(run)
        self._emit()

    async def on_proposal(self, *, accept: bool) -> None:
        """§15.6: install the proposed change (revision+1), or put the upstream back and resume.

        Rejecting is no longer the end of the run: the upstream restores its previous plan under a NEW revision
        (``/api/guide/plan/revert``) and judgement continues where the run left off. Only when it cannot be put
        back — no previous plan, a stale pair, a transport failure — is the run ended with ``plan_changed``.
        """
        run = self.guide
        if run is None or self.proposal is None:
            raise EngineReject("no_proposal", "제안된 변경안이 없습니다", retryable=False)
        if accept:
            if not self._install_proposal(run):
                raise EngineReject("no_proposal", "제안된 변경안이 없습니다", retryable=False)
            self.proposal = None
            self._pump()
            self._emit()
            return
        if run.plan_id is None or run.plan_revision is None:
            self._reject_without_revert(run)
            return
        try:
            current = await self._upstream().plan_revert({
                "session_id": run.session_id, "plan_id": run.plan_id, "plan_revision": run.plan_revision})
        except UpstreamError:
            self._reject_without_revert(run)
            return
        if not self._is_current(run):
            return
        if not self._pin_upstream_plan(run, current):
            self._reject_without_revert(run)
            return
        run.step_index = min(run.step_index, len(run.steps) - 1)
        self.view.step_index = run.step_index
        if run.core_mode == CORE_GRAPH:
            self._graph_focus(run)
        self.stats["proposal_rejected"] += 1
        self.stats["proposal_reverted"] += 1
        self.proposal = None
        self._set_blocked(None)
        self._set_notice("reverted", "변경안을 받아들이지 않았어요. 원래 계획대로 이어서 안내합니다.")
        self._refresh_replan_block(run)
        self._pump()
        self._emit()

    def _reject_without_revert(self, run: GuideRun) -> None:
        """§15.6: the upstream cannot be put back, so the approved procedure can no longer be executed."""
        if not self._is_current(run):
            return
        self.stats["proposal_rejected"] += 1
        self._retire(error=("plan_changed", "승인한 절차가 상위에서 바뀌어 되돌릴 수 없습니다. 새로 시작해 주세요."),
                     keep_view=True)
        self._emit()

    async def on_follow_now(self) -> None:
        """useIntentLoop.ts:1330-1334."""
        self._require_running(self.guide)
        self._enqueue_follow("manual")
        self._emit()

    async def on_replan_now(self) -> None:
        """useIntentLoop.ts:1337-1342 (a blocked replan only shows ``replan_blocked_reason``)."""
        run = self._require_running(self.guide)
        self._request_replan(run, "replan", run.last_checks)
        self._emit()

    async def on_confirm_done(self) -> None:
        """useIntentLoop.ts:1344-1350; then the guide is over (tracking stops)."""
        run = self.guide
        if run is None or run.completion != "confirmed" or run.plan_id is None:
            raise EngineReject("not_running", "아직 완료 확인 단계가 아닙니다")
        run.completion = "user_confirmed"
        self.view.completion = "user_confirmed"
        self.view.phase = "completed"
        self.view.notice = None
        self._stop_tracking()
        self._emit()

    async def on_prefs(self, prefs: Prefs) -> None:
        self.prefs = prefs

    # ================================================================================== talk

    async def on_talk(self, utterance: str) -> None:
        """useIntentLoop.ts:1553-1578 talk()."""
        run = self.guide
        text = utterance.strip()[:200]
        if not text:
            raise EngineReject("invalid_message", "빈 말은 보낼 수 없습니다", retryable=False)
        if run is not None and (self._planning() or self._reviewing() or self._clarifying() or self.user_paused
                                or self.proposal is not None):
            reason = (PROPOSAL_WAIT_REASON if self.proposal is not None
                      else "검토가 끝난 뒤에 말할 수 있습니다." if self._reviewing()
                      else CLARIFY_WAIT_REASON if self._clarifying()
                      else "계획을 세우는 중입니다. 잠시 후 다시 말씀해 주세요." if self._planning()
                      else "일시정지 중입니다. 재개한 뒤 말씀해 주세요.")
            self.talk_block = reason
            self.view.talk_blocked_reason = reason
            self._emit()
            raise EngineReject("not_running", reason)
        block, wait = gp.decide_talk(running=bool(run and not run.halted and run.plan_id is not None),
                                     talk_pending=bool(run and run.talk_pending), gate=self.gate, now_ms=self.now())
        if run is None or block is not None:
            reason = gp.describe_talk_block(block or "not_running") or "지금은 말할 수 없습니다."
            self.talk_block = reason
            self.view.talk_blocked_reason = reason
            self._emit()
            code = {"pending": "talk_busy", "budget": "talk_budget", "per_minute": "talk_budget"}.get(block or "",
                                                                                                   "not_running")
            raise EngineReject(code, reason)
        if not (0 <= run.step_index < len(run.steps)) or run.plan_id is None:
            raise EngineReject("not_running", "진행 중인 가이드가 없습니다")
        step = run.steps[run.step_index]
        # runTalk's synchronous prefix (useIntentLoop.ts:1386-1393): pending and holding the remote lane.
        run.talk_pending = True
        run.talk_waiting = True
        self.view.talk_pending = True
        self._emit()
        self._spawn(self._run_talk(run, text, wait, {"plan_id": run.plan_id, "step_id": step.id,
                                                    "done_when": step.done_when}), run)

    def _talk_state(self, run: GuideRun) -> gp.TalkRunState:
        return gp.TalkRunState(steps=tuple(run.steps), step_index=run.step_index,
                               plan_revision=run.plan_revision or 0, target=run.target, skipped=tuple(run.skipped),
                               user_done=tuple(run.user_done), unsure_streak=run.unsure_streak,
                               stuck_count=run.stuck_count, last_checks=run.last_checks,
                               replan_budget=run.replan_budget)

    async def _run_talk(self, run: GuideRun, text: str, first_wait_ms: float, accepted: dict) -> None:
        """useIntentLoop.ts:1381-1547 runTalk."""
        holds_replan = False
        try:
            if first_wait_ms > 0:
                await self._sleep(first_wait_ms)
                if not self._is_current(run) or run.halted:
                    return
                again, _ = gp.decide_talk(running=True, talk_pending=False, gate=self.gate, now_ms=self.now())
                if again is not None:
                    self._set_notice("talk_failed", gp.describe_talk_block(again) or "지금은 말할 수 없습니다.")
                    return
            scene = self._capture_scene(run, "talk")
            if scene is None or run.session_id is None or self.up is None:
                self._set_notice("talk_failed", "대상 추적이 시작된 뒤에 말할 수 있습니다.")
                return
            if not self._is_current(run) or run.halted or run.plan_id is None or run.plan_revision is None:
                return
            sent_step_id = gp.talk_send_step(accepted, plan_id=run.plan_id, steps=run.steps)
            if sent_step_id is None:
                self._set_notice("talk_failed", "말하는 사이 계획이 바뀌어 보내지 않았습니다. 다시 말해 주세요.")
                return
            run.replan_budget, holds_replan = gp.reserve_replan(run.replan_budget)
            self._refresh_replan_block(run)
            self.gate = gp.mark_talk(self.gate, self.now())
            run.talk_waiting = False
            self._pump()
            run.intent_seq += 1
            run.latest_seq["talk"] = run.intent_seq
            anchor = scene["anchor"]
            stamp = gp.SentStamp(run.session_id, run.plan_id, run.plan_revision, run.intent_seq,
                                 {"task_epoch": run.task_epoch, "run_id": scene["run_id"]}, echo_of(anchor))
            payload: dict[str, Any] = {
                "session_id": run.session_id, "consent_ai": True,
                "scene": {"frame_id": scene["frame_id"], "image_base64": base64.b64encode(scene["jpeg"]).decode("ascii"),
                          "label": SCENE_LABEL},
                "plan_id": run.plan_id, "plan_revision": run.plan_revision, "current_step": sent_step_id,
                "anchors": [anchor] if anchor else [], "utterance": text, "replan_allowed": holds_replan,
                "intent_seq": stamp.intent_seq, "fence": stamp.fence,
            }
            if run.last_talk:
                payload["prev_utterance"], payload["prev_reply"] = run.last_talk
            if run.last_checks:
                payload["follow_checks"] = list(run.last_checks)
            self.stats["talk"] += 1
            self._emit()
            answer = await self.up.talk(payload)
            if not self._is_current(run):
                return
            acceptance = self._classify(run, "talk", stamp, answer, gp.talk_changes_plan(answer))
            self.stats[f"talk:{acceptance}"] += 1
            outcome = gp.apply_talk(self._talk_state(run), sent_step_id, answer, acceptance, self.now(), _parse_steps)
            if not outcome.applied:
                self.stats["talk_rejected"] += 1
                self._set_notice("talk_failed", "답이 지금 화면·계획과 맞지 않아 적용하지 않았습니다. 다시 말해 주세요.")
                return
            for action in outcome.actions:
                self.stats[f"talk_action:{action}"] += 1
            step_before = run.step_index
            plan_rev_before = run.plan_revision
            s = outcome.state
            if self._gated(run):
                s = self._reconcile_talk(run, answer, s,
                                         changed="replan" in outcome.actions or "step_say" in outcome.actions)
            run.steps, run.step_index, run.plan_revision, run.target = s.steps, s.step_index, s.plan_revision, s.target
            run.skipped, run.user_done, run.unsure_streak, run.stuck_count = s.skipped, s.user_done, s.unsure_streak, s.stuck_count
            run.last_checks, run.replan_budget = s.last_checks, s.replan_budget
            if run.core_mode == CORE_SEQUENTIAL and (run.step_index != step_before
                                                     or run.plan_revision != plan_rev_before
                                                     or "step_mark" in outcome.actions
                                                     or "go_to" in outcome.actions or "target" in outcome.actions):
                # §15.8: an explicit user mark, a go_to, or a plan revision moves the ground under any pending
                # nomination: hold it, and never carry confirmed evidence across a revision.
                self._reset_sequential(run, moved=True,
                                       replanned=run.plan_revision != plan_rev_before)
            elif run.core_mode == CORE_GRAPH and "target" in outcome.actions:
                # Retargeting has no same-entity proof. Retain confirmation history, not its applicability.
                self._reset_graph(run, identity_changed=True)
            run.last_talk = (text, str(answer.get("reply", ""))[:400])
            notice: str | None = None
            for kind, value in outcome.effects:
                if kind == "retarget" and value:
                    # The new run binds on its own `acquired`; until then nothing is drawn.
                    run.binding = None
                    run.accepted = None
                    run.appearance_ref = None
                    run.warn = False
                    self._start_live_run(value)
                elif kind == "checkGoal":
                    if run.completion == "none":
                        self._enqueue_confirm("goal_check")
                elif kind == "notice" and value:
                    notice = value
            v = self.view
            v.steps, v.plan_revision, v.target = run.steps, run.plan_revision, run.target
            v.step_index, v.skipped, v.user_done = run.step_index, run.skipped, run.user_done
            if "target" in outcome.actions:
                v.needs_reselect = False
            reply = (str(answer.get("reply") or "").strip() or "…")[:400]
            spoken = (str(answer.get("spoken") or "").strip() or reply)[:60]
            v.talk = TalkResult(utterance=text[:200], reply=reply, spoken=spoken, at=_epoch_ms())
            if run.step_index != step_before:
                v.basis_age_s = None
                self.step_log.append((self.now() - run.started_at, run.step_index, "talk"))
            if notice:
                self._set_notice("talk_replan", notice)
            if "replan" in outcome.actions and not run.plan_mode:
                run.replan_epoch += 1
        except asyncio.CancelledError:
            raise
        except UpstreamError as exc:
            if not self._is_current(run):
                return
            self.stats[f"talk:failed:{exc.code}"] += 1
            if exc.status == 401:
                if self.up is not None:
                    self.up.invalidate_session()
                self._retire(error=("session_expired", "세션이 만료되었습니다. 가이드를 다시 시작해주세요."), keep_view=True)
                return
            if exc.status == 409 and exc.code == "task_changed":
                self._retire(error=("task_changed", "작업 내용이 바뀌어 가이드를 멈췄습니다. 다시 시작해주세요."), keep_view=True)
                return
            self._set_notice("talk_failed", gp.describe_talk_error(exc.status or None, exc.code, exc.reason))
        except Exception:
            log.exception("talk failed")
            if self._is_current(run):
                self._set_notice("talk_failed", gp.describe_talk_error(None))
        finally:
            if holds_replan:
                run.replan_budget = gp.release_replan(run.replan_budget)
                self._refresh_replan_block(run)
            if self.guide is run:
                run.talk_pending = False
                self.view.talk_pending = False
                if run.talk_waiting:
                    run.talk_waiting = False
                    self._pump()
            self._emit()

    # ================================================================================== lifecycle

    async def on_pause(self) -> None:
        """Disconnected: nothing is sent upstream while no frames can arrive; the tracker run is released."""
        self.paused = True
        self._resumed.clear()
        if self.guide is not None:
            # §15.8: no frames can arrive while disconnected, so a nomination has no pending answer to wait for.
            self._reset_sequential(self.guide, moved=True)
        self._stop_tracking()
        self._emit()

    async def on_resume(self) -> None:
        """Spec §9: tracking restarts on the plan's target (a fresh run; acquiring for the first frames)."""
        self.paused = False
        self._resumed.set()
        run = self.guide
        # §15.7: a reconnect lifts the connection pause only — a pause the user set survives it.
        if run is not None and not run.halted and run.plan_id is not None and self.view.phase == "running" \
                and not self.user_paused:
            self._start_live_run(run.target)
            self._pump()
        self._emit()

    async def close(self) -> None:
        self._closed = True
        self._retire()
        if self._ticker is not None:
            self._ticker.cancel()
            self._ticker = None
        current = asyncio.current_task()
        for task in list(self._tasks):
            if task is not current:
                task.cancel()
        if self.up is not None:
            try:
                await asyncio.wait_for(self.up.end_session(), 5.0)
            except (asyncio.TimeoutError, Exception):
                pass
            await self.up.aclose()
            self.up = None


__all__ = ["RealConfig", "RealContext", "RealEngine"]

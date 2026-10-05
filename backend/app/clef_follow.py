"""Clef follower: the ``guide_follow`` judgement from a Clef decision server on this machine (loopback only).

Clef answers closed questions about one image directly, as probabilities, instead of writing text: one
``POST /v1/systemone`` carries a ``state`` (the plan as text) and one question per verdict the follow tool
output needs, and returns one answer per question. Nothing is generated, so there is no grammar, no envelope
of prose to refuse and no ceiling to hit; the answers are expanded here to the ``GuideFollowToolOutput`` shape
and the route checks them exactly like the other followers' (same ids, same provenance, same 502/504 stages).

Questions (``ClefFollower.payload``), one per key:

* each checklist step: "지금 카메라 화면에서 이 상태가 보이는가? {done_when}" -> ``step_checks[].visible``.
  Under the default ``classic`` core this is ``current_step`` .. last (and the state lists the whole plan);
  under ``sequential`` it is the current step ALONE and the state lists only that step, so the follower can
  never answer about a future step; under ``graph`` it is EVERY step of the plan, earlier and already
  completed ones included, and the state lists the whole plan, so the client's focus never narrows a question;
* ``goal``: the same question about ``user_goal`` -> ``goal_seen``. ``choice`` mode asks ``goal_when`` under
  ``classic`` and ``graph``; ``sequential`` always asks the user's own words, never a generated ``goal_when``;
* the anchor id, only while it is tracking (its box is drawn on the frame by ``prepare_frame_b64``): "그려진
  상자 안의 물체가 '{label}'인가?" -> ``anchor_verdicts[].matches``. A box-less anchor gets no verdict, which
  the contract allows.

Modes:

* ``noul`` (default): each question is a ``noul`` (yes probability ``p``); the verdict is ``yes`` when
  ``p >= threshold`` (default 0.5), otherwise ``no``. This mode never answers ``unsure``. Measured 2026-10-03
  with ``tools/e2e/follower_bench.py``'s clips, frames, tracks and fixed plans (Clef 27B BF16 split over two
  RTX 5090, 72 frames, threshold not tuned): step verdicts 131/138, false step ``yes`` 0/66 (no truth) and
  0/15 (absent), goal false ``yes`` 1/39, call p50/p95 389/419 ms (server median 377 ms).
* ``choice``: each question is a ``choice`` among ``yes``/``no``/``unsure`` with one criterion each, and the
  verdict is Clef's choice. Principles v2.1 clarify the v2 single-frame policy's visibility predicates
  (``CHOICE_RULES`` in ``state["rules"]`` plus ``current_step``): yes = the state itself is
  visible, no = visible counter-evidence (even when the place is out of frame), unsure = nothing visible decides;
  never infer from step order, absence is not completion. ``clef_record`` builds the state and questions and is
  the one builder both this follower and ``tools/clef_ft/build_data.py`` (the fine-tuning set) use, so training
  and new serving records share one builder. Frozen v2 review files remain historical inputs; they are not
  relabelled or re-encoded automatically. The v2.1 clarification is a rule candidate, not a measured accuracy
  improvement; its evidence must come from real model A/B calls rather than prompt-string assertions.

Configuration (environment only; values are never logged):

* ``AISW_FOLLOW_PROVIDER=clef`` selects it. Read by the application, not here.
* ``AISW_FOLLOW_CLEF_URL`` — required: the full ``/v1/systemone`` URL, loopback host only (``loopback.py``),
  e.g. ``http://127.0.0.1:8085/v1/systemone``. Redirects are never followed. Readiness is ``GET /health``
  on the same authority.
* ``AISW_FOLLOW_CLEF_MODEL`` — optional, default ``clef``: the model name requested and reported.
* ``AISW_FOLLOW_CLEF_MODE`` — ``noul`` (default) or ``choice``.
* ``AISW_FOLLOW_CLEF_THRESHOLD`` — optional, default ``0.5``; ``noul`` mode only, in (0, 1].
* ``AISW_FOLLOW_CLEF_YES_MIN`` — optional, ``choice`` mode only, in (0, 1]: answer ``yes`` only when its
  probability is at least this, otherwise the likelier of ``no``/``unsure``. Unset = Clef's own choice. The
  clef-ft 2026-10-04 adapter was measured with 0.5 (test 119 rows: false current-step ``yes`` 12/66 -> 8/66).
* ``AISW_FOLLOW_CLEF_TIMEOUT_S`` — optional, default ``10``; positive number.

Missing or unusable configuration raises ``StageConfigError`` when the follower is built (at startup when
``AISW_FOLLOW_PROVIDER=clef``); it never falls back to another follower. One call per request, no retries.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx

from backend.app.api_contracts import Usage
from backend.app.errors import StageConfigError
from backend.app.guide_contracts import CORE_MODE_GRAPH, CORE_MODE_SEQUENTIAL, CORE_MODES, DEFAULT_CORE_MODE
from backend.app.local_follow import _health_url, _positive_float, post_loopback_json
from backend.app.loopback import loopback_endpoint
from backend.app.provider import InvalidOutputStage, ProviderInvalidOutput

if TYPE_CHECKING:
    from backend.app.guide import StoredPlan
    from backend.app.guide_contracts import GuideFollowRequest


ENV_URL = "AISW_FOLLOW_CLEF_URL"
ENV_MODEL = "AISW_FOLLOW_CLEF_MODEL"
ENV_MODE = "AISW_FOLLOW_CLEF_MODE"
ENV_THRESHOLD = "AISW_FOLLOW_CLEF_THRESHOLD"
ENV_TIMEOUT_S = "AISW_FOLLOW_CLEF_TIMEOUT_S"
ENV_YES_MIN = "AISW_FOLLOW_CLEF_YES_MIN"

MODE_NOUL = "noul"
MODE_CHOICE = "choice"
MODES = (MODE_NOUL, MODE_CHOICE)
DEFAULT_MODE = MODE_NOUL
DEFAULT_MODEL = "clef"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TIMEOUT_S = 10.0

#: Readiness probe: cached like the local follower's, so /api/health polling cannot hammer the server.
HEALTH_TTL_S = 2.0
HEALTH_TIMEOUT_S = 2.0

CLEF_PROVIDER_NAME = "clef"

#: The goal question's key. Step ids are ``s1``..``s5``, so it never collides with one.
GOAL_KEY = "goal"
#: The anchor question's key when the anchor id itself is a step id or ``goal``.
ANCHOR_FALLBACK_KEY = "anchor"

CHOICE_VALUES = ("yes", "no", "unsure")
#: Single-frame observation principles v2.1: distinguish a visibility predicate's readable evidence from
#: mere object presence. Frozen v2 review records/labels are not rewritten by this change.
STATE_CRITERIA = {
    "yes": "보여서 충족: 이 상태 자체가 지금 화면에서 보인다. 손이 일부를 가려도 보이는 부분이 이 상태를 보여 주면 해당한다. "
           "식별·판독이 조건이면 필요한 영역·문자를 실제로 구분할 수 있어야 한다.",
    "no": "반대 근거가 보임: 보이는 것이 이 상태의 미완료·반대 상태를 입증한다. 장소가 화면 밖이어도 명확한 반증은 해당한다. "
          "작음·흐림·가림으로 식별하지 못한 것 자체는 반증이 아니다.",
    "unsure": "판단할 근거 없음: 상태가 화면 밖·가려짐·너무 작음·흐림으로 식별되지 않고 반증도 보이지 않는다. "
              "문자·번호 판독이 조건인데 읽을 수 없으면 해당한다.",
}
#: ``choice`` mode ``state["rules"]``: single-frame principles v2.1; shared by serving and new observation rows.
CHOICE_RULES = (
    "카메라 화면 한 장을 계획과 맞춰 판정한다. 단계 질문은 그 단계의 done_when(눈에 보이는 결과 상태)이, goal 질문은 "
    "goal_when이 지금 이 화면에서 보이는지 묻는다. 아래 원칙을 순서대로 적용한다.\n"
    "- yes(보여서 충족): 그 상태 자체가 보인다. 손이 일부를 가려도 보이는 부분이 그 상태를 보여 주면 yes다.\n"
    "- no(반대 근거가 보임): 보이는 무언가가 그 상태에 아직 이르지 않았음을 보여 준다. 그 상태가 일어날 장소가 화면 밖이어도 "
    "마찬가지다(예: done_when이 '접시에 토마토 여섯 조각이 있다'일 때 접시는 화면 밖이지만 토마토가 아직 통째로 손에 있거나 "
    "조각이 아직 도마 위에 있다).\n"
    "- unsure(판단할 근거 없음): 그 상태일 수도 아닐 수도 있는데 이를 정해 줄 것이 보이지 않는다(화면 밖, 가려짐, 너무 작음, "
    "흐림이고 다른 보이는 단서도 없다).\n"
    "- 가시성·식별 자체가 조건('영역이 보인다', '행 문자·번호를 읽을 수 있다')이면 필요한 영역·문자를 지금 화면에서 "
    "구분할 수 있을 때만 yes다. 물체가 있다는 것, 매뉴얼의 배치, 이전에 읽혔다는 사실로 대신하지 않는다. "
    "작음·흐림·가림으로 구분할 수 없으면 unsure이며, 식별 가능한 반대 내용·상태가 보일 때만 no다. "
    "일반 상태 술어도 미식별 자체를 미완료의 반증으로 삼지 않는다.\n"
    "- 계획의 단계 순서만으로 추론하지 않는다('s1이 안 됐으니 s2도 아니다'는 금지).\n"
    "- 무언가가 없다는 것만으로 yes라고 하지 않는다(부재는 완료가 아니다).\n"
    "- 거짓 yes가 가장 해로운 오류다."
)
#: ``choice`` mode ``state["rules"]`` for the ``sequential`` core: the same principles plus the one difference
#: this core introduces — the checklist is the current step alone, and the goal question asks the user's own
#: words rather than the generated ``goal_when``.
CHOICE_SEQUENTIAL_RULES = (
    CHOICE_RULES
    + "\n- 이 실행 모드에서는 지금 안내 중인 현재 단계 하나만 판정한다(다른 단계는 질문에 없다). goal 질문도 "
      "goal_when이 아니라 사용자 원문의 목표를 묻는다."
)


def anchor_criteria(label: str) -> dict[str, str]:
    """``choice`` mode criteria for the anchor question."""
    return {
        "yes": f"그려진 상자 안의 물체가 '{label}'이다.",
        "no": f"그려진 상자 안의 물체가 '{label}'이 아니다.",
        "unsure": "상자 안이 가려졌거나 너무 작거나 흐려서 판단할 수 없다.",
    }


def anchor_key(anchor_id: str, step_ids: list[str]) -> str:
    return ANCHOR_FALLBACK_KEY if anchor_id in step_ids or anchor_id == GOAL_KEY else anchor_id


class ClefAnchor(NamedTuple):
    """What ``clef_record`` needs from the first follow anchor."""

    anchor_id: str
    label: str
    state: str
    boxed: bool


def _question(mode: str, instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    if mode == MODE_CHOICE:
        return {"type": "choice", "instructions": instructions, "criteria": criteria}
    return {"type": "noul", "instructions": instructions}


def _user_goal_question(user_goal: str) -> str:
    """The goal question asked of the user's OWN words (never a generated ``goal_when``)."""
    return (
        f"지금 카메라 화면에서 사용자 원문의 목표가 달성됐는가? {user_goal} "
        "계획·goal_when은 참고일 뿐, 요청하지 않은 내려놓기·정리·보관·위치 조건은 무시한다. "
        "남은 단계가 있어도 원래 목표가 보이면 달성이다. 사용자가 명시한 모든 결과는 필요하다. "
        "제거 목표는 떼어 낸 자리가 보이면 판단하되, 대상이 사라졌다는 사실만으로 달성을 추측하지 않는다.")


def clef_record(*, user_goal: str, steps: Sequence[Mapping[str, str]], goal_when: str, current_step: str,
                anchor: ClefAnchor | None, mode: str, core_mode: str = DEFAULT_CORE_MODE) -> dict[str, Any]:
    """The ``state`` and ``questions`` of one Clef follow call (no model, no image).

    ``steps`` are the plan steps in order, each with ``id``, ``say`` and ``done_when``. Under the default
    ``classic`` core questions cover ``current_step`` through the last step (``StoredPlan.checklist_ids``) and
    the state lists the whole plan, then ``goal``, then the anchor when its box is drawn. Under
    ``sequential`` both the questions AND the plan in the state are scoped to the current step alone, so the
    follower is never shown — and can never answer about — a future step; its goal question asks the user's
    own words rather than the generated ``goal_when``. Under ``graph`` the questions and the state cover the
    WHOLE plan (earlier, already-completed steps included; the request's focus never narrows them) and the goal
    question keeps the classic ``goal_when`` wording, so a graph run judges completion against the same
    predicate as classic. ``choice`` mode adds the principles v2.1 rules and the
    current step to the state. Shared with the fine-tuning set builder (``tools/clef_ft/build_data.py``);
    change both sides by changing only this.
    """
    if mode not in MODES:
        raise ValueError(f"unknown Clef mode {mode!r}")
    if core_mode not in CORE_MODES:
        raise ValueError(f"unknown Clef core mode {core_mode!r}")
    ids = [step["id"] for step in steps]
    if core_mode == CORE_MODE_GRAPH:
        # Graph asks about the whole plan whatever the focus: an earlier step's verdict is what lets the client
        # revoke a state claim or keep a verified event, so it can never be dropped from the question set.
        scoped_ids = ids
    elif current_step not in ids:
        scoped_ids = []
    elif core_mode == CORE_MODE_SEQUENTIAL:
        scoped_ids = [current_step]
    else:
        scoped_ids = ids[ids.index(current_step):]
    scoped_steps = ([step for step in steps if step["id"] == current_step] if core_mode == CORE_MODE_SEQUENTIAL
                    else steps)
    done_when = {step["id"]: step["done_when"] for step in steps}
    if anchor is not None and anchor.boxed:
        tracker = f"이미지에 그려진 상자는 추적기가 '{anchor.label}'(으)로 따라가는 대상이다."
    else:
        # The app sends no anchor while a track has never acquired, so no anchor reads as ``acquiring``.
        tracker = f"추적기 상태: {anchor.state if anchor is not None else 'acquiring'} — 이미지에 상자가 없다."
    plan = [{"id": step["id"], "instruction": step["say"], "done_when": step["done_when"]}
            for step in scoped_steps]
    if mode == MODE_CHOICE and core_mode != CORE_MODE_SEQUENTIAL:
        # The fine-tuned wording (clef-ft 2026-10-04): goal_when in the state, the goal question about goal_when.
        # ``graph`` takes this branch on purpose: its goal predicate is the classic one, so switching a plan to
        # graph never changes how completion is judged.
        state: dict[str, Any] = {"rules": CHOICE_RULES, "user_goal": user_goal, "plan": plan, "goal_when": goal_when,
                                 "tracker": tracker, "current_step": current_step}
        goal_question = f"지금 카메라 화면에서 목표 상태가 보이는가? {goal_when}"
    elif mode == MODE_CHOICE:
        # Sequential: no generated goal_when in the state and the goal question asks the user's own words, so
        # the follower never judges completion against an obligation the user did not ask for.
        state = {"rules": CHOICE_SEQUENTIAL_RULES, "user_goal": user_goal, "plan": plan,
                 "tracker": tracker, "current_step": current_step}
        goal_question = _user_goal_question(user_goal)
    else:
        # noul: the goal is the user's own words; generated obligations in goal_when are ignored (92ae0f6).
        state = {"user_goal": user_goal, "plan": plan, "tracker": tracker}
        goal_question = _user_goal_question(user_goal)
    questions = {
        step_id: _question(mode, f"지금 카메라 화면에서 이 상태가 보이는가? {done_when[step_id]}", STATE_CRITERIA)
        for step_id in scoped_ids
    }
    questions[GOAL_KEY] = _question(mode, goal_question, STATE_CRITERIA)
    if anchor is not None and anchor.boxed:
        questions[anchor_key(anchor.anchor_id, scoped_ids)] = _question(
            mode, f"그려진 상자 안의 물체가 '{anchor.label}'인가?", anchor_criteria(anchor.label))
    return {"state": state, "questions": questions}


class ClefFollower:
    """One ``/v1/systemone`` call per follow request against a loopback Clef server; no retries."""

    provider_name = CLEF_PROVIDER_NAME
    #: Not a paid analysis: the route keeps it out of the paid rate lane and the provider slots.
    paid = False

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        model: str = DEFAULT_MODEL,
        mode: str = DEFAULT_MODE,
        threshold: float = DEFAULT_THRESHOLD,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        yes_min: float | None = None,
    ) -> None:
        if mode not in MODES:
            raise StageConfigError(f"{ENV_MODE} must be one of {', '.join(repr(m) for m in MODES)}")
        if not 0 < threshold <= 1:
            raise StageConfigError(f"{ENV_THRESHOLD} must be in (0, 1]")
        if yes_min is not None and not 0 < yes_min <= 1:
            raise StageConfigError(f"{ENV_YES_MIN} must be in (0, 1]")
        self.client = client
        self.url = loopback_endpoint(url, ENV_URL)
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.yes_min = yes_min
        self._probe: tuple[float, bool] = (0.0, False)

    def payload(self, request: GuideFollowRequest, plan: StoredPlan, image_b64: str) -> dict[str, Any]:
        anchor = request.anchors[0] if request.anchors else None
        record = clef_record(
            user_goal=plan.user_goal,
            steps=[{"id": step.id, "say": step.say, "done_when": step.done_when} for step in plan.steps],
            goal_when=plan.goal_when,
            current_step=request.current_step,
            anchor=None if anchor is None else ClefAnchor(anchor.anchor_id, anchor.label, anchor.state,
                                                          anchor.box is not None),
            mode=self.mode,
            core_mode=plan.core_mode,
        )
        return {
            "model": self.model,
            "state": record["state"],
            "images": [f"data:image/jpeg;base64,{image_b64}"],
            "questions": record["questions"],
        }

    async def guide_follow(self, request: GuideFollowRequest, plan: StoredPlan,
                           image_b64: str) -> tuple[dict[str, Any], Usage | None]:
        payload = self.payload(request, plan, image_b64)
        body = await post_loopback_json(self.client, self.url, payload, self.timeout_s, "clef follower")
        verdicts = self._verdicts(body, payload["questions"])
        step_ids = plan.checklist_ids(request.current_step)
        tracked = [anchor for anchor in request.anchors if anchor.box is not None]
        return {
            "step_checks": [{"step_id": step_id, "visible": verdicts[step_id]} for step_id in step_ids],
            "anchor_verdicts": [
                {"anchor_id": anchor.anchor_id, "matches": verdicts[anchor_key(anchor.anchor_id, step_ids)]}
                for anchor in tracked
            ],
            "goal_seen": verdicts[GOAL_KEY],
        }, _usage(body.get("usage"))

    def _verdicts(self, body: dict[str, Any], questions: dict[str, Any]) -> dict[str, str]:
        """One tristate per asked key, or ``ProviderInvalidOutput``: ``envelope`` when there is no answer map,
        ``tool_json`` when the answers are not exactly the asked keys with a well-formed answer each."""
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise ProviderInvalidOutput(InvalidOutputStage.ENVELOPE)
        if set(answers) != set(questions):
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_JSON)
        return {key: self._verdict(answers[key]) for key in questions}

    def _verdict(self, answer: Any) -> str:
        if not isinstance(answer, dict):
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_JSON)
        if self.mode == MODE_CHOICE:
            choice = answer.get("choice")
            if answer.get("type") != "choice" or choice not in CHOICE_VALUES:
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_JSON)
            if self.yes_min is None:
                return choice
            probs = answer.get("probabilities")
            if not isinstance(probs, dict) or not all(
                    isinstance(probs.get(v), (int, float)) and not isinstance(probs.get(v), bool)
                    and math.isfinite(probs[v]) for v in CHOICE_VALUES):
                raise ProviderInvalidOutput(InvalidOutputStage.TOOL_JSON)
            if probs["yes"] >= self.yes_min:
                return "yes"
            return "no" if probs["no"] >= probs["unsure"] else "unsure"
            
        p = answer.get("noul")
        if (answer.get("type") != "noul" or isinstance(p, bool) or not isinstance(p, (int, float))
                or not math.isfinite(p) or not 0 <= p <= 1):
            raise ProviderInvalidOutput(InvalidOutputStage.TOOL_JSON)
        return "yes" if p >= self.threshold else "no"

    async def ready(self) -> bool:
        """Short-cached readiness of the Clef server (``GET /health`` == 200). Never raises."""
        now = time.monotonic()
        seen, ready = self._probe
        if seen and now - seen < HEALTH_TTL_S:
            return ready
        try:
            response = await self.client.get(
                _health_url(self.url), timeout=HEALTH_TIMEOUT_S, follow_redirects=False
            )
            ready = response.status_code == 200
        except httpx.HTTPError:
            ready = False
        self._probe = (now, ready)
        return ready


def _usage(data: Any) -> Usage | None:
    """Clef's ``usage`` (``input_tokens``, ``output_tokens``; the latter is 0: nothing is generated)."""
    if not isinstance(data, dict):
        return None
    counts = {
        name: value for name in ("input_tokens", "output_tokens")
        if isinstance(value := data.get(name), int) and not isinstance(value, bool) and value >= 0
    }
    if not counts:
        return None
    return Usage(**counts, total_tokens=sum(counts.values()))


def build_clef_follower(client: httpx.AsyncClient) -> ClefFollower:
    """Build the Clef follower from its environment, or raise ``StageConfigError`` naming the variable."""
    url = (os.getenv(ENV_URL) or "").strip()
    if not url:
        raise StageConfigError(f"{ENV_URL} is not set")
    model = (os.getenv(ENV_MODEL) or "").strip() or DEFAULT_MODEL
    mode = (os.getenv(ENV_MODE) or "").strip().lower() or DEFAULT_MODE
    raw = (os.getenv(ENV_THRESHOLD) or "").strip()
    try:
        threshold = float(raw) if raw else DEFAULT_THRESHOLD
    except ValueError as exc:
        raise StageConfigError(f"{ENV_THRESHOLD} is not a number") from exc
    raw_yes = (os.getenv(ENV_YES_MIN) or "").strip()
    try:
        yes_min = float(raw_yes) if raw_yes else None
    except ValueError as exc:
        raise StageConfigError(f"{ENV_YES_MIN} is not a number") from exc
    return ClefFollower(
        client,
        url=url,
        model=model,
        mode=mode,
        threshold=threshold,
        timeout_s=_positive_float(ENV_TIMEOUT_S, DEFAULT_TIMEOUT_S),
        yes_min=yes_min,
    )

"""An in-process stand-in for the demo backend's HTTP API (httpx.MockTransport), scriptable per test.

It checks what the real backend checks that matters to the engine (Origin, session cookie, plan revision →
409 stale_plan, frame_seq monotonic, seed only on a run's first frame) and echoes the acceptance envelope.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

STEPS = [
    {"id": "s1", "say": "안경다리를 양손으로 잡으세요", "done_when": "두 손이 안경다리를 잡고 있다",
     "commands": [{"kind": "focus", "anchor": "target", "pad": 0.15},
                  {"kind": "action", "anchor": "target", "action": "grasp", "direction": "none"}]},
    {"id": "s2", "say": "안경을 앞으로 당겨 벗으세요", "done_when": "안경이 얼굴에서 떨어져 있다",
     "commands": [{"kind": "action", "anchor": "target", "action": "pull", "direction": "down"}]},
    {"id": "s3", "say": "안경을 내려놓으세요", "done_when": "안경이 손에서 떨어져 있다",
     "commands": [{"kind": "label", "anchor": "target", "text": "여기"}]},
]
BOX = {"x": 0.3, "y": 0.2, "width": 0.25, "height": 0.2}


@dataclass
class FakeUpstream:
    base_url: str = "http://upstream.test"
    follow_provider: str = "local"
    access_code_required: bool = False
    #: §15: the upstream health's per-profile configured capability (mirrors the real backend's `plan_models`).
    plan_models: dict = field(default_factory=lambda: {"deepseek:high": True, "astra:high": True})
    plan_steps: list = field(default_factory=lambda: [dict(s) for s in STEPS])
    plan_target: str = "eyeglasses"
    #: A question every plan call answers with (the basic, single-turn "stop and say why" path).
    plan_clarification: str | None = None
    #: §15: per-call questions. Each plan call pops one; an empty queue answers with the plan, so
    #: ``deque([q])`` models "ask once, then plan" and ``deque([q1, q2])`` two question turns.
    plan_clarifications: deque = field(default_factory=deque)
    #: §15: public facts the plan answer reports (the real backend merges them per task and caps at six).
    plan_sources: list = field(default_factory=list)
    #: per-frame override queue of (state, box) for the tracker; default: 1st acquiring, then tracking BOX
    track_script: deque = field(default_factory=deque)
    box: dict = field(default_factory=lambda: dict(BOX))
    follow_answers: deque = field(default_factory=deque)
    #: optional ``callable(body) -> dict`` for ``/api/guide/follow``: answered per request instead of popping
    #: ``follow_answers``, so a test can script "yes for the current step" without counting calls.
    follow_fn: Any = None
    #: optional ``async callable(body)`` awaited before a ``/api/guide/follow`` answer is produced: lets a test
    #: hold a follower answer open while the run moves underneath it.
    follow_hold: Any = None
    confirm_answers: deque = field(default_factory=deque)
    #: optional ``async callable(body)`` awaited before a ``/api/guide/confirm`` answer is produced: lets a
    #: test hold the upper's verdict open while the run moves underneath it.
    confirm_hold: Any = None
    #: when true, confirm answers echo an anchor that has drifted (models the anchor changing under a call):
    #: the engine must reject the answer instead of applying it.
    confirm_echo_drift: bool = False
    #: when true, step_done answers carry ``step_check`` but no ``step_id`` echo (a non-conforming upstream).
    confirm_omit_step_echo: bool = False
    talk_answers: deque = field(default_factory=deque)
    requests: list = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token: str = "tok-1"
    plan_id: str | None = None
    plan_revision: int = 0
    #: §15.8: the guide core the plan request named (``classic`` whole-remaining checklist, ``sequential``
    #: current step alone, ``graph`` every plan step including earlier/done ones) — exactly like the real
    #: backend's ``StoredPlan.checklist_ids``.
    plan_core_mode: str = "classic"
    plan_goal_when: str = "안경이 얼굴에서 벗겨져 있다"
    plan_user_goal: str = ""
    plan_context: str | None = None
    plan_task_revision: int = 1
    #: §15.6: replaced plans kept so ``POST /api/guide/plan/revert`` can restore one (bounded, like upstream).
    plan_history: deque = field(default_factory=lambda: deque(maxlen=2))
    record_plan_history: bool = True
    runs: dict = field(default_factory=dict)
    active_run: str | None = None

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # ------------------------------------------------------------ helpers

    def calls(self, path: str) -> list[dict]:
        return [body for p, body in self.requests if p == path]

    def _json(self, status: int, body: Any, headers: dict | None = None) -> httpx.Response:
        return httpx.Response(status, json=body, headers=headers or {})

    def _err(self, status: int, code: str, extra: dict | None = None, retry_after: int | None = None):
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        return self._json(status, {"error": code, **(extra or {})}, headers)

    def _envelope(self, body: dict) -> dict:
        return {
            "intent_id": str(uuid.uuid4()), "frame_id": body["scene"]["frame_id"], "intent_seq": body["intent_seq"],
            "task_revision": 1, "plan_revision": self.plan_revision, "fence_echo": body["fence"],
            "anchors_echo": [{"anchor_id": a["anchor_id"], "track_id": a["track_id"], "generation": a["generation"]}
                             for a in body["anchors"]],
            "provider": "fake", "model": "fake", "usage": None, "latency_ms": 1,
        }

    def _stale(self, body: dict) -> httpx.Response | None:
        if body.get("plan_id") != self.plan_id or body.get("plan_revision") != self.plan_revision:
            return self._err(409, "stale_plan", {"plan_id": self.plan_id, "plan_revision": self.plan_revision})
        return None

    # ------------------------------------------------------------ handler

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.requests.append((path, body))
        if path == "/api/health":
            plan_models = {name: bool(self.plan_models.get(name)) for name in ("deepseek:high", "astra:high")}
            return self._json(200, {"ready": True, "model": "fake", "provider": "deepseek",
                                    "access_code_required": self.access_code_required, "access_mode": "open",
                                    "tracker_ready": True, "guide_ready": True,
                                    "plan_ready": any(plan_models.values()), "plan_models": plan_models,
                                    "follow_provider": self.follow_provider, "follow_ready": True})
        if request.headers.get("origin") != self.base_url:
            return self._err(403, "invalid_origin")
        if path == "/api/session":
            resp = self._json(200, {"session_id": self.session_id})
            resp.headers["set-cookie"] = f"visual_coach_session={self.token}; HttpOnly; Path=/; SameSite=strict; Secure"
            return resp
        if f"visual_coach_session={self.token}" not in request.headers.get("cookie", ""):
            return self._err(401, "session_required")
        if path == "/api/session/end":
            return self._json(200, {"ended": True})
        if path == "/api/track/control":
            if body["action"] == "start":
                self.runs[body["run_id"]] = {"target": body["target"], "last_seq": 0, "frames": 0,
                                             "track": f"k{len(self.runs) + 1}"}
                self.active_run = body["run_id"]
            elif self.active_run == body["run_id"]:
                self.active_run = None
            return self._json(200, {"run_id": body["run_id"], "active": body["action"] == "start",
                                    "target": self.runs.get(body["run_id"], {}).get("target", "x")})
        if path == "/api/track/frame":
            run = self.runs.get(body["run_id"])
            if run is None or self.active_run != body["run_id"]:
                return self._err(409, "stale_run", {"active_run_id": self.active_run})
            if body.get("seed_box") is not None and run["frames"] > 0:
                return self._err(409, "seed_not_first_frame")
            if body["frame_seq"] <= run["last_seq"]:
                return self._err(409, "stale_frame", {"expected_seq_after": run["last_seq"]})
            run["last_seq"] = body["frame_seq"]
            run["frames"] += 1
            if self.track_script:
                state, box = self.track_script.popleft()
            elif body.get("seed_box") is not None:
                state, box = "tracking", body["seed_box"]
            else:
                state, box = ("acquiring", None) if run["frames"] == 1 else ("tracking", dict(self.box))
            return self._json(200, {
                "run_id": body["run_id"], "target": run["target"], "track_id": run["track"], "generation": 1,
                "state": state, "transition": "none", "box": box if state == "tracking" else None,
                "source": "grounder", "frame_id": body["frame_id"], "frame_seq": body["frame_seq"],
                "version": run["frames"], "confidence": 0.9, "ingest_age_ms": 5,
            })
        if path == "/api/guide/plan":
            # §15: the upstream contract is closed — the planning profile is `plan_model` from the two-profile
            # set and the guide core is `core_mode` (``classic``/``sequential``); the retired `plan_mode`
            # selector is rejected as an extra field (as the real backend does).
            if body.get("plan_model") not in ("deepseek:high", "astra:high") or "plan_mode" in body \
                    or body.get("core_mode", "classic") not in ("classic", "sequential", "graph"):
                return self._err(422, "invalid_request")
            # §15: the upstream's closed request — the user's answers (≤12) and reference photos ride the plan
            # call; anything over the bound is refused rather than truncated.
            if len(body.get("answers") or []) > 12 or len(body.get("reference_images") or []) > 12:
                return self._err(422, "invalid_request")
            self._record_history()
            self.plan_id = str(uuid.uuid4())
            self.plan_revision += 1
            self.plan_core_mode = body.get("core_mode", "classic")
            self._steps = None
            self.plan_goal_when = "안경이 얼굴에서 벗겨져 있다"
            self.plan_user_goal = body.get("user_goal") if isinstance(body.get("user_goal"), str) else ""
            self.plan_context = body.get("context")
            selection = {"status": "selected", "target": self.plan_target, "rationale": None}
            final = {"selection": selection, "steps": self.plan_steps, "goal_when": self.plan_goal_when,
                     "evidence_kind": "observed_scene", "needs_clarification": False, "clarification_prompt": None,
                     "research_sources": [dict(s) for s in self.plan_sources],
                     "plan_id": self.plan_id, "plan_revision": self.plan_revision, "frame_id": body["scene"]["frame_id"],
                     "provider": "fake", "model": "fake", "usage": None}
            prompt = self.plan_clarifications.popleft() if self.plan_clarifications else self.plan_clarification
            if prompt:
                final.update(steps=[], needs_clarification=True, clarification_prompt=prompt,
                             evidence_kind="uncertain_view")
            events = [("partial", {"target": self.plan_target}),
                      ("partial", {"first_say": self.plan_steps[0]["say"]} if self.plan_steps else {}),
                      ("final", final)]
            text = ": ping\n\n" + "".join(
                f"event: {k}\ndata: {json.dumps(v, ensure_ascii=False)}\n\n" for k, v in events)
            return httpx.Response(200, content=text.encode(), headers={"content-type": "text/event-stream"})
        if path == "/api/guide/plan/current":
            if self.plan_id is None:
                return self._err(404, "no_plan")
            return self._json(200, self._current_plan())
        if path == "/api/guide/plan/approve":
            stale = self._stale(body)
            if stale:
                return stale
            if not body.get("steps") or not isinstance(body.get("goal_when"), str):
                return self._err(422, "invalid_request")
            self._record_history()
            self._steps = [dict(s) for s in body["steps"]]
            self.plan_goal_when = body["goal_when"]
            if isinstance(body.get("user_goal"), str):
                self.plan_user_goal = body["user_goal"]
            if "context" in body:
                self.plan_context = body["context"]
            self.plan_revision += 1
            return self._json(200, self._current_plan())
        if path == "/api/guide/plan/revert":
            stale = self._stale(body)
            if stale:
                return stale
            if not self.plan_history:
                return self._err(404, "no_plan")
            restored = self.plan_history.pop()
            self._record_history()  # the plan being replaced stays reachable, like the upstream's history
            self.plan_id, self.plan_revision = restored["plan_id"], self.plan_revision + 1
            self._steps = [dict(s) for s in restored["steps"]]
            self.plan_goal_when, self.plan_user_goal = restored["goal_when"], restored["user_goal"]
            return self._json(200, self._current_plan())
        if path == "/api/guide/follow":
            stale = self._stale(body)
            if stale:
                return stale
            answer = self.follow_answers.popleft() if self.follow_answers else {}
            if self.follow_fn is not None:
                answer = self.follow_fn(body)
            if isinstance(answer, httpx.Response):
                return answer
            if self.follow_hold is not None:
                await self.follow_hold(body)
            ids = [s["id"] for s in self.current_steps()]
            start = ids.index(body["current_step"])
            # §15.8: the checklist covers every remaining step under ``classic``, the current step ALONE under
            # ``sequential``, and EVERY plan step (earlier/done ones included) under ``graph`` — the real backend
            # refuses any other id list as a schema violation (an explicit ``checks`` list models a non-conforming
            # upstream that answered about other steps anyway).
            if self.plan_core_mode == "sequential":
                covered = ids[start:start + 1]
            elif self.plan_core_mode == "graph":
                covered = list(ids)
            else:
                covered = ids[start:]
            checks = answer["checks"] if "checks" in answer else \
                [{"step_id": i, "visible": answer.get("visible", {}).get(i, "no")} for i in covered]
            verdicts = [{"anchor_id": a["anchor_id"], "matches": answer.get("matches", "yes")} for a in body["anchors"]]
            return self._json(200, {**self._envelope(body), "step_checks": checks, "anchor_verdicts": verdicts,
                                    "goal_seen": answer.get("goal_seen", "no"), "trigger": body["trigger"],
                                    "needs_reselect": any(v["matches"] == "no" for v in verdicts)})
        if path == "/api/guide/confirm":
            stale = self._stale(body)
            if stale:
                return stale
            if self.confirm_hold is not None:
                await self.confirm_hold(body)
            answer = self.confirm_answers.popleft() if self.confirm_answers else {}
            if isinstance(answer, httpx.Response):
                return answer
            out = {**self._envelope(body), "goal_status": {"status": answer.get("status", "in_progress"),
                                                           "rationale": "fake"},
                   "evidence_kind": "observed_scene", "replan": None, "needs_clarification": False,
                   "clarification_prompt": None, "trigger": body["trigger"], "step_check": None, "step_id": None}
            if self.confirm_echo_drift:
                out["anchors_echo"] = [{**a, "generation": a["generation"] + 1} for a in body["anchors"]]
            if body["trigger"] == "step_done":
                out["step_check"] = answer.get("step_check", "yes")
                if not self.confirm_omit_step_echo:
                    out["step_id"] = body["current_step"]
            if answer.get("replan"):
                self._record_history()
                self.plan_revision += 1
                self._steps = answer["replan"]
                out["replan"] = {"steps": answer["replan"]}
                out["plan_revision"] = self.plan_revision
            return self._json(200, out)
        if path == "/api/guide/talk":
            stale = self._stale(body)
            if stale:
                return stale
            answer = self.talk_answers.popleft() if self.talk_answers else {}
            if isinstance(answer, httpx.Response):
                return answer
            out = {**self._envelope(body), "user_says_done": False, "reply": answer.get("reply", "네, 알겠어요."),
                   "spoken": answer.get("spoken", "네, 알겠어요."), "step_say": None, "target": None,
                   "step_mark": None, "go_to": None, "replan": None}
            for key in ("step_say", "target", "step_mark", "go_to"):
                if key in answer:
                    out[key] = answer[key]
            if answer.get("replan"):
                self._record_history()
                self.plan_revision += 1
                self._steps = answer["replan"]
                out["replan"] = {"steps": answer["replan"]}
            elif answer.get("step_say"):
                self.plan_revision += 1
            out["plan_revision"] = self.plan_revision
            return self._json(200, out)
        return self._err(404, "not_found")

    def current_steps(self) -> list:
        return getattr(self, "_steps", None) or self.plan_steps

    def _current_plan(self) -> dict:
        """The ``/api/guide/plan/current`` shape the engine reads back after ``approve``/``revert``."""
        return {"plan_id": self.plan_id, "plan_revision": self.plan_revision, "core_mode": self.plan_core_mode,
                "task_revision": self.plan_task_revision, "steps": [dict(s) for s in self.current_steps()],
                "goal_when": self.plan_goal_when, "user_goal": self.plan_user_goal}

    def _record_history(self) -> None:
        """§15.6: keep the plan a replacement is about to overwrite, so ``/plan/revert`` can restore it."""
        if self.plan_id is None or not self.record_plan_history:
            return
        self.plan_history.append({"plan_id": self.plan_id, "plan_revision": self.plan_revision,
                                  "steps": [dict(s) for s in self.current_steps()],
                                  "goal_when": self.plan_goal_when, "user_goal": self.plan_user_goal})

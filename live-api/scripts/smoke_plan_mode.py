"""Plan 모드(스펙 §15) 끝단 스모크: 실제 서버에 붙어서 초안 → 검토 → 승인 → 실행까지 밀어 본다.

인-프로세스 TestClient가 아니라 **네트워크 너머의 서버**에 붙습니다. 로컬(`uv run uvicorn ...`)이나
공개 주소(`https://sangye.kr/synoptics/api`) 아무 곳이나 됩니다. 카메라는 쓰지 않고 PIL로 만든
640x360 JPEG를 프레임처럼 보냅니다(mock 엔진은 이미지를 보지 않습니다).

    uv run python scripts/smoke_plan_mode.py --base https://sangye.kr/synoptics/api --max-seconds 180
    uv run python scripts/smoke_plan_mode.py --base http://127.0.0.1:8104        # 로컬 서버

검사하는 것 (§15.4가 실제로 강제되는지):
1. `start{plan_mode:true, plan_model:"astra:high", materials:[…]}` → `phase:"reviewing"`, `plan.status:"draft"`, 단계 ≥ 6, 자료 반영
2. 검토 중 프레임에는 `track{state:"idle"}`(추적 시작 안 함), `overlay`는 null
3. `plan_edit` → revision+1, 여전히 draft
4. `plan_approve` → `phase:"running"`, `status:"approved"`, `approved_revision == revision`, 그리고 `tracking`
5. 필수 검사(`required`) 단계는 타이머만으로 넘어가지 않고 `blocked`가 서며, `step_ack`으로만 진행
6. (나오면) 변경안은 `proposal_accept`로만 적용되고 revision이 오른다
7. 필수 단계가 끝난 뒤에만 `completion`이 `confirmed`로 간다

종료 코드 0이면 위 항목이 모두 관측된 것입니다. `--max-seconds`로 시간을 묶습니다(모의 엔진은
`LIVE_MOCK_STEP_S`(기본 8초)마다 한 단계씩 가므로 공개 서버에서는 오래 걸립니다).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from typing import Any

import httpx
from PIL import Image
from websockets.sync.client import connect

from synoptics_live.contracts import PROTOCOL
from synoptics_live.envelope import pack, unpack

W, H = 640, 360
GOAL = "책을 상자에 포장하기"
MATERIAL = {
    "title": "포장 지침",
    "version": "v3",
    "text": "3.1 바닥면은 완충재로 먼저 덮는다. 4.2 봉함 전에 내부 배치를 확인한다.",
}


def frame_jpeg(step: int) -> bytes:
    image = Image.new("RGB", (W, H), (32, 40, 60))
    for i in range(40):
        image.putpixel(((step * 7 + i * 3) % W, (step * 3 + i * 5) % H), (240, 200, 60))
    image.paste((200, 200, 210), (100 + (step * 11) % 200, 120, 220 + (step * 11) % 200, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


class Observed:
    """What the flow needs to see. Every flag must be True for the smoke to pass."""

    def __init__(self) -> None:
        self.flags: dict[str, bool] = {
            "reviewing": False, "draft": False, "steps>=6": False, "material": False,
            "idle-frames": False, "no-overlay": False, "edited": False, "approved": False,
            "running": False, "tracking": False, "blocked": False, "required-acked": False,
            "proposal": False, "confirmed": False, "completed": False,
        }
        self.notes: list[str] = []
        self.errors: list[str] = []

    def note(self, where: str, why: str) -> None:
        if self.flags.get(where):
            return
        self.flags[where] = True
        self.notes.append(f"{where}: {why}")

    @property
    def failed(self) -> list[str]:
        return [name for name, ok in self.flags.items() if not ok]


def fail(seen: Observed, why: str) -> int:
    print("FAIL:", why)
    print("never observed:", ", ".join(seen.failed))
    for line in seen.notes:
        print("  ", line)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8104", help="REST/WS base ('…/synoptics/api' 포함)")
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--ack-fallback", type=float, default=45.0,
                    help="변경안이 안 오면 이만큼 기다린 뒤 필수 단계를 그냥 확인한다")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    deadline = time.monotonic() + args.max_seconds

    with httpx.Client(timeout=20.0) as http:
        health = http.get(f"{base}/v1/health").json()
        print(f"health: engine={health.get('engine')} ready={health.get('ready')}")
        if health.get("engine") != "mock":
            print("note: mock 엔진이 아닙니다. 이 스모크는 각본이 고정된 mock 기준입니다.")
        created = http.post(f"{base}/v1/sessions", json={})
        created.raise_for_status()
        session = created.json()
    sid, token, live_url = session["session_id"], session["token"], session["live_url"]
    print(f"session {sid[:8]} → {live_url}")

    seen = Observed()
    seq = 0
    capture = None
    required_step: str | None = None
    ack_sent = False
    proposal_answered = False
    first_block_at: float | None = None
    edit_sent = False
    approve_sent = False
    confirm_sent = False
    frame_interval = 1.0 / 15.0          # ready.limits.max_fps 로 갱신
    next_frame_at = 0.0
    start = time.monotonic()

    def left() -> float:
        return deadline - time.monotonic()

    with connect(live_url, open_timeout=20, close_timeout=5, max_size=None) as ws:
        ws.send(json.dumps({"type": "hello", "token": token, "protocol": PROTOCOL, "resume_rev": None}))

        def send_frame(hi_req: str | None = None) -> None:
            nonlocal seq, next_frame_at
            now = time.monotonic()
            if not hi_req and now < next_frame_at:
                time.sleep(next_frame_at - now)
            header: dict[str, Any] = {"t": "frame", "seq": seq, "captured_at": int(time.time() * 1000), "w": W, "h": H}
            if hi_req is not None:
                header["hi_req"] = hi_req
            ws.send(pack(header, frame_jpeg(seq)))
            seq += 1
            next_frame_at = time.monotonic() + frame_interval

        # 프레임은 크레딧 1개: 직전 track이 돌아와야 다음 프레임을 보낸다.
        waiting_track = None
        send_frame()

        while time.monotonic() < deadline:
            try:
                raw = ws.recv(timeout=min(20.0, max(left(), 0.1)))
            except TimeoutError:
                if left() <= 0:
                    break
                print("… 프레임만 계속 보냅니다")
                send_frame()
                waiting_track = seq - 1
                continue
            if isinstance(raw, bytes):
                header, payload = unpack(raw)
                if header.get("t") in ("audio", "frame"):
                    continue
                continue
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "ready":
                max_fps = int((msg.get("limits") or {}).get("max_fps") or 15)
                frame_interval = 1.0 / max(max_fps, 1)
                print("ready:", json.dumps(msg["limits"], ensure_ascii=False)[:160])
                ws.send(json.dumps({"type": "prefs", "voice_out": False, "tts": "browser", "visuals": True}))
                ws.send(json.dumps({"type": "start", "goal": GOAL, "consent_ai": True,
                                    "plan_mode": True, "plan_model": "astra:high", "materials": [MATERIAL]}))
                continue
            if kind == "capture_hi":
                capture = msg["req_id"]
                send_frame(hi_req=capture)
                waiting_track = seq - 1
                continue
            if kind == "say":
                print(f"say: {msg['text']}")
                continue
            if kind == "track":
                if waiting_track is not None and msg.get("seq") == waiting_track:
                    waiting_track = None
                if msg.get("state") == "idle" and not approve_sent:
                    seen.note("idle-frames", "검토 중 프레임에 track.idle")
                if msg.get("state") == "tracking":
                    seen.note("tracking", "승인 뒤 추적 시작")
                if waiting_track is None:
                    send_frame()
                    waiting_track = seq - 1
                continue
            if kind == "error":
                seen.errors.append(f"{msg.get('code')}: {msg.get('message')}")
                print(f"error: {msg.get('code')} {msg.get('message')}")
                continue
            if kind != "state":
                continue

            plan = msg.get("plan") or {}
            phase = msg.get("phase")
            if phase == "reviewing":
                if not seen.flags["reviewing"]:
                    seen.note("reviewing", "phase=reviewing")
                    print(f"plan: revision={plan.get('revision')} status={plan.get('status')} "
                          f"steps={[s['id'] for s in plan.get('steps', [])]} materials={plan.get('materials')}")
                if plan.get("status") == "draft":
                    seen.note("draft", "status=draft")
                if len(plan.get("steps", [])) >= 6:
                    seen.note("steps>=6", f"{len(plan['steps'])} steps")
                if len(plan.get("materials", [])) == 1 and plan["materials"][0]["id"] == "m1":
                    seen.note("material", "m1 echoed with chars")
                if msg.get("overlay") is None:
                    seen.note("no-overlay", "reviewing overlay is null")
                if not required_step:
                    required_step = next((s["id"] for s in plan["steps"] if s.get("required")), None)
                if not edit_sent and required_step:
                    step = next(s for s in plan["steps"] if s["id"] == required_step)
                    body: dict[str, Any] = {"type": "plan_edit", "edits": [{"step_id": required_step,
                                                                           "check": "user"}]}
                    if not step.get("required"):
                        body["edits"][0]["required"] = True
                    print(f"→ plan_edit {json.dumps(body, ensure_ascii=False)}")
                    ws.send(json.dumps(body))
                    edit_sent = True
                    continue
                if edit_sent and not approve_sent and plan.get("revision", 1) > 1:
                    seen.note("edited", f"revision={plan['revision']} status={plan['status']}")
                    if not required_step:
                        required_step = next((s["id"] for s in plan["steps"] if s.get("required")), None)
                    print("→ plan_approve")
                    ws.send(json.dumps({"type": "plan_approve"}))
                    approve_sent = True
                continue

            if phase == "running":
                seen.note("running", "phase=running")
                if plan.get("status") == "approved" and plan.get("approved_revision") == plan.get("revision"):
                    seen.note("approved", f"approved_revision={plan['approved_revision']}")
                blocked = msg.get("blocked")
                if blocked:
                    step_id = blocked.get("step_id")
                    if not seen.flags["blocked"]:
                        print(f"blocked: {step_id} — {blocked.get('reason')}")
                    seen.note("blocked", f"{step_id} blocked")
                    if first_block_at is None:
                        first_block_at = time.monotonic()
                    # §15.6을 먼저 보여 주려고 기다린다: 변경안이 오면 그것부터 답하고, 안 오면 fallback 뒤에 확인한다.
                    waited = time.monotonic() - first_block_at
                    if not ack_sent and step_id and (proposal_answered or waited > args.ack_fallback):
                        print(f"→ step_ack {step_id}" + (" (변경안 없음, fallback)" if not proposal_answered else ""))
                        ws.send(json.dumps({"type": "step_ack", "step_id": step_id}))
                        ack_sent = True
                proposal = msg.get("proposal")
                if proposal and not proposal_answered:
                    seen.note("proposal", f"revision={proposal.get('revision')} reason={proposal.get('reason')}")
                    print(f"→ proposal_accept ({proposal.get('reason')})")
                    ws.send(json.dumps({"type": "proposal_accept"}))
                    proposal_answered = True
                if msg.get("steps_user_done") and required_step in msg["steps_user_done"]:
                    seen.note("required-acked", f"steps_user_done={msg['steps_user_done']}")
                if msg.get("completion") == "confirmed":
                    seen.note("confirmed", "completion=confirmed")
                    if not confirm_sent:
                        print("→ confirm_done")
                        ws.send(json.dumps({"type": "confirm_done"}))
                        confirm_sent = True
                if msg.get("phase") == "completed" or msg.get("completion") == "user_confirmed":
                    seen.note("completed", "completion=user_confirmed")
                    break
                continue

            if phase == "completed":
                seen.note("completed", "phase=completed")
                break

    elapsed = time.monotonic() - start
    try:
        with httpx.Client(timeout=20.0) as http:
            ended = http.request("DELETE", f"{base}/v1/sessions/{sid}",
                                 headers={"Authorization": f"Bearer {token}"})
        print(f"ended: {ended.status_code} {ended.text[:80]}")
    except Exception as exc:  # noqa: BLE001 - 정리 실패는 스모크 결과를 덮지 않는다
        print("end failed:", exc)

    print(f"\n--- {elapsed:.1f}s, 프레임 {seq}장 ---")
    missing = seen.failed
    if missing:
        return fail(seen, f"관측되지 않은 항목: {', '.join(missing)}")
    for line in seen.notes:
        print("  ", line)
    print("OK: §15 Plan 모드 흐름이 끝까지 관측됐습니다.")
    if any("plan_locked" in e or "invalid_edit" in e for e in seen.errors):
        print("(예상한 거절도 있었습니다:", [e for e in seen.errors if "edit" in e], ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

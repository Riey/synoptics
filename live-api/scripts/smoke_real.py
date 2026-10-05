"""LIVE smoke of the real engine against the real upstream (default http://127.0.0.1:8045), in-process.

The app runs inside this process (Starlette TestClient + websocket_connect): no port is bound. TTS is off.
Frames come from a directory of JPEGs (extract them first; the source clip is never modified or copied here):

    ffmpeg -i ~/Projects/aisw/glass.mov -vf fps=10,scale=640:-2 -q:v 5 $DIR/frames/%04d.jpg
    ffmpeg -i ~/Projects/aisw/glass.mov -vf fps=10,scale=1024:-2 -q:v 4 $DIR/frames_hi/%04d.jpg
    uv run python scripts/smoke_real.py --frames $DIR/frames --frames-hi $DIR/frames_hi

The clip plays in real time at --fps and loops (``--hold-end`` freezes on the last frame after the first pass
instead). Frames are credit-paced like a browser: the next frame leaves when the previous one's ``track`` came
back (or after 2 s), never faster than ``limits.max_fps``. Each costs a few DeepSeek calls upstream.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("LIVE_ENGINE", "real")
os.environ["LIVE_TTS_URL"] = ""  # TTS off: the smoke measures the guide, not audio

from starlette.testclient import TestClient  # noqa: E402

from synoptics_live.app import create_app  # noqa: E402
from synoptics_live.contracts import PROTOCOL  # noqa: E402
from synoptics_live.envelope import pack  # noqa: E402
from synoptics_live.settings import Settings  # noqa: E402


def load(directory: str) -> list[tuple[bytes, int, int]]:
    from PIL import Image

    out = []
    for path in sorted(Path(directory).glob("*.jpg")):
        data = path.read_bytes()
        with Image.open(path) as im:
            out.append((data, im.width, im.height))
    if not out:
        sys.exit(f"no frames in {directory}")
    return out


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(p / 100 * (len(values) - 1))))
    return values[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--frames-hi", default=None)
    ap.add_argument("--goal", default="안경을 벗어 주세요")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds after start")
    ap.add_argument("--hold-end", action="store_true", help="freeze on the last frame after one pass")
    ap.add_argument("--talk", default=None, help="an utterance to send once 4 s after running")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    frames = load(args.frames)
    frames_hi = load(args.frames_hi) if args.frames_hi else None
    settings = Settings.from_env()
    app = create_app(settings, tts=None)
    client = TestClient(app)
    client.__enter__()
    events: list[tuple[float, dict]] = []
    inbox: queue.Queue = queue.Queue()
    credit: dict[int, threading.Event] = {}
    sent_at: dict[int, float] = {}
    rtts: list[float] = []
    t0 = time.monotonic()

    health = client.get("/v1/health").json()
    print("health:", json.dumps(health, ensure_ascii=False))
    resp = client.post("/v1/sessions", json={})
    assert resp.status_code == 201, resp.text
    sess = resp.json()
    report: dict = {"health": health, "goal": args.goal, "fps": args.fps, "hold_end": args.hold_end,
                    "frames": len(frames)}
    with client.websocket_connect(f"/v1/sessions/{sess['session_id']}/live") as ws:
        ws.send_text(json.dumps({"type": "hello", "token": sess["token"], "protocol": PROTOCOL, "resume_rev": None}))
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready", ready
        ws.send_text(json.dumps({"type": "prefs", "voice_out": False, "tts": "browser", "visuals": True}))
        stop = threading.Event()
        frame_index = {"i": 0}

        def reader() -> None:
            while not stop.is_set():
                try:
                    msg = ws.receive()
                except Exception:
                    return
                if msg.get("type") == "websocket.close":
                    return
                now = time.monotonic()
                if msg.get("text") is None:
                    continue
                obj = json.loads(msg["text"])
                events.append((now, obj))
                if obj["type"] == "track":
                    if obj["seq"] in sent_at:
                        rtts.append((now - sent_at[obj["seq"]]) * 1000)
                    ev = credit.get(obj["seq"])
                    if ev:
                        ev.set()
                elif obj["type"] == "capture_hi":
                    src = frames_hi or frames
                    data, w, h = src[frame_index["i"] % len(src)]
                    header = {"t": "frame", "seq": 10_000_000 + len(events), "captured_at": int(time.time() * 1000),
                              "w": w, "h": h, "hi_req": obj["req_id"]}
                    ws.send_bytes(pack(header, data))
                inbox.put(obj)

        threading.Thread(target=reader, daemon=True).start()
        seq = 0
        started_at: float | None = None
        running_at: float | None = None
        talk_sent = False
        confirm_sent = False
        last_state: dict = {}
        min_gap = 1 / 15
        last_send = 0.0
        stream_t0 = time.monotonic()
        while True:
            now = time.monotonic()
            # newest state
            for t, obj in events[-50:]:
                if obj["type"] == "state":
                    last_state = obj
            if started_at is None and now - stream_t0 >= 1.0:
                ws.send_text(json.dumps({"type": "start", "goal": args.goal, "consent_ai": True}))
                started_at = time.monotonic()
                print(f"[{started_at - t0:6.2f}] start sent")
            if started_at is not None:
                if running_at is None and last_state.get("phase") == "running":
                    running_at = now
                if args.talk and running_at and not talk_sent and now - running_at >= 4 and \
                        not last_state.get("talk_pending"):
                    ws.send_text(json.dumps({"type": "talk", "utterance": args.talk}))
                    talk_sent = True
                if last_state.get("completion") == "confirmed" and not confirm_sent:
                    ws.send_text(json.dumps({"type": "confirm_done"}))
                    confirm_sent = True
                if last_state.get("phase") in ("completed", "error") or (
                        last_state.get("phase") == "idle" and last_state.get("clarification")):
                    time.sleep(0.5)
                    break
                if now - started_at > args.timeout:
                    print("timeout")
                    break
            # pacing: real-time clip position, credit 1, <= 15 fps
            elapsed = now - stream_t0
            idx = int(elapsed * args.fps)
            if args.hold_end and idx >= len(frames):
                idx = len(frames) - 1
            frame_index["i"] = idx
            data, w, h = frames[idx % len(frames)]
            wait = last_send + min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            ev = threading.Event()
            credit[seq] = ev
            sent_at[seq] = time.monotonic()
            last_send = sent_at[seq]
            ws.send_bytes(pack({"t": "frame", "seq": seq, "captured_at": int(time.time() * 1000), "w": w, "h": h},
                               data))
            ev.wait(2.0)
            seq += 1
            # wait for the next clip frame (real time)
            next_t = stream_t0 + (int((time.monotonic() - stream_t0) * args.fps) + 1) / args.fps
            delay = next_t - time.monotonic()
            if delay > 0 and (time.monotonic() - sent_at[seq - 1]) < 1 / args.fps:
                time.sleep(min(delay, 1 / args.fps))
        stop.set()
        session = app.state.store.sessions.get(sess["session_id"])
        engine = session.engine if session else None
        stats = dict(engine.stats) if engine else {}
        milestones = dict(engine.milestones) if engine else {}
        step_log = list(engine.step_log) if engine else []
        client.delete(f"/v1/sessions/{sess['session_id']}", headers={"Authorization": f"Bearer {sess['token']}"})
    client.__exit__(None, None, None)

    assert started_at is not None

    def first(pred) -> float | None:
        for t, obj in events:
            if t >= started_at and pred(obj):
                return round((t - started_at) * 1000)
        return None

    tracks = [o for t, o in events if o["type"] == "track" and t >= started_at]
    report.update({
        "client_ms": {
            "partial_target": first(lambda o: o["type"] == "state" and (o.get("partial") or {}).get("target")),
            "partial_first_say": first(lambda o: o["type"] == "state" and (o.get("partial") or {}).get("first_say")),
            "running": first(lambda o: o["type"] == "state" and o.get("phase") == "running"),
            "first_step_say": first(lambda o: o["type"] == "say" and "단계" in o["text"]),
            "first_tracking_box": first(lambda o: o["type"] == "track" and o["state"] == "tracking"),
            "first_overlay": first(lambda o: o["type"] == "state" and o.get("overlay")),
            "checking": first(lambda o: o["type"] == "state" and o.get("completion") == "checking"),
            "confirmed": first(lambda o: o["type"] == "state" and o.get("completion") == "confirmed"),
        },
        "frames_sent": seq,
        "track_states": {s: sum(1 for o in tracks if o["state"] == s) for s in
                         ("idle", "acquiring", "tracking", "occluded", "lost", "unavailable")},
        "track_rtt_ms": {"n": len(rtts), "p50": round(pct(rtts, 50), 1), "p90": round(pct(rtts, 90), 1),
                         "max": round(max(rtts), 1) if rtts else None,
                         "mean": round(statistics.mean(rtts), 1) if rtts else None},
        "engine_stats": stats,
        "engine_milestones_ms": {k: round(v) for k, v in milestones.items()},
        "step_log": [(round(t), i, why) for t, i, why in step_log],
        "says": [o["text"] for t, o in events if o["type"] == "say"],
        "errors": [o for t, o in events if o["type"] == "error"],
        "notices": sorted({(o.get("notice") or {}).get("text") for t, o in events
                           if o["type"] == "state" and o.get("notice")}),
        "final_state": {k: last_state.get(k) for k in ("phase", "completion", "step_index", "steps_skipped",
                                                        "steps_user_done", "needs_reselect", "error", "talk")},
        "plan": last_state.get("plan"),
    })
    text = json.dumps(report, ensure_ascii=False, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Follower-only micro-benchmark: the app's real follow request against any llama.cpp model, frame by frame.

Each sampled frame (2 fps) of the three fixture clips is sent through ``backend.app.local_follow.LocalFollower``
exactly as the ``/api/guide/follow`` route sends it: the real ``follow_prompt`` and system prompt, the request-
narrowed strict json_schema, ``enable_thinking: false``, the anchor box drawn by ``intent.prepare_frame_b64``.
The answer is validated like the route does (``GuideFollowToolOutput`` + checklist ids + anchor provenance), so
a "schema reject" here is one the route would have answered with 502.

Inputs (see README "follower_bench"):

* ``--frames-root``: ``<root>/<clip>/f_%05d.jpg`` at 10 fps, extracted with the same crop as ``make_clips.sh``.
* ``--tracks-root``: ``<root>/<clip>/frames.jsonl`` from ``tools/tracker_service/replay_clip.py`` over those
  frames, so every request carries the tracker's own box (or its occluded/lost state) for that frame.

Plans are fixed (``PLANS``): DeepSeek plans recorded by earlier e2e runs (timeline.json of the 2026-10-01
checklist benchmark), so every model judges the same steps. Truth (``TRUTH``) was read by eye from the 2 fps
sheets (``truth/*-tile.png``) and a boundary-frame montage; ``skip`` marks frames whose state is ambiguous.
Every request covers the whole checklist (``current_step = s1``) with trigger ``heartbeat``.

    python3 tools/e2e/follower_bench.py run --url http://127.0.0.1:18082/v1/chat/completions \\
        --model Qwen3.6-35B-A3B-UD-Q4_K_M --frames-root F --tracks-root T --out out/35b.jsonl
    python3 tools/e2e/follower_bench.py score out/*.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError  # noqa: E402

from backend.app.guide import StoredPlan  # noqa: E402
from backend.app.guide_contracts import GuideFollowRequest, GuideFollowToolOutput, GuideStep  # noqa: E402
from backend.app.intent import prepare_frame_b64  # noqa: E402
from backend.app.local_follow import LocalFollower  # noqa: E402
from backend.app.provider import ProviderError  # noqa: E402

FPS_SOURCE = 10
SAMPLE_EVERY = 5  # 10 fps -> 2 fps (``--sample-every 1`` with ``--t-min/--t-max`` for a dense window)

# ------------------------------------------------------------------------------------------------ plans

#: (user_goal, anchor label, steps [(id, say, done_when)], goal_when) — from e2e timelines, 2026-10-01:
#: glass = e2e-final run1 (482d36b), increase_temp = e2e-final run1, put-airpod = e2e-after run2 (84253d2).
PLANS = {
    "glass": ("안경 벗기", "glasses", [
        ("s1", "양손으로 안경 다리를 잡으세요.", "양손이 안경 다리를 잡고 있는 모습이 보임"),
        ("s2", "안경을 얼굴에서 앞으로 들어 올려 빼내세요.", "안경이 얼굴에서 떨어져 손에 들려 있음"),
        ("s3", "안경을 화면 아래쪽으로 내려놓으세요.", "안경이 화면 아래쪽 평평한 곳에 놓여 있음"),
    ], "얼굴에 안경이 없고 안경이 화면 아래쪽에 내려져 있음"),
    "put-airpod": ("에어팟 케이스 열기", "earbud case", [
        ("s1", "케이스 뚜껑을 위로 열어 주세요.", "케이스 뚜껑이 열려 이어버드가 보이는 상태"),
        ("s2", "뚜껑이 완전히 열린 상태로 잡아 주세요.", "뚜껑이 위쪽으로 젖혀져 고정된 상태"),
    ], "케이스 뚜껑이 완전히 열려 안이 보이는 상태"),
    "increase_temp": ("온도를 28도로 올리기", "air conditioner remote control", [
        ("s1", "리모컨 오른쪽 위의 온도 조절 + 버튼을 누르세요.", "액정 온도 숫자가 24.0도에서 25도 이상으로 바뀐다."),
        ("s2", "액정에 28.0도가 표시될 때까지 + 버튼을 계속 누르세요.", "액정에 28.0도가 표시된다."),
    ], "리모컨 액정에 28.0도가 표시된다."),
}

# ------------------------------------------------------------------------------------------------ truth

#: Per clip and key (step id or "goal"): ordered (t_from, t_to, label) inclusive ranges over the 2 fps grid.
#: Labels: yes / no / absent (the thing is out of view: the right answer is unsure, "no" is tolerated, "yes"
#: breaks absence != completion) / skip (ambiguous, not scored).
INF = 99.0
TRUTH = {
    "glass": {
        "s1": [(0, 1.0, "no"), (1.5, 5.0, "yes"), (5.5, 7.0, "skip"), (7.5, INF, "absent")],
        # From ~7.0 s the glasses sit at the bottom edge, mostly cropped out of the 1280x720 frame.
        "s2": [(0, 5.0, "no"), (5.5, 6.5, "yes"), (7.0, 7.5, "skip"), (8.0, INF, "absent")],
        "s3": [(0, 7.0, "no"), (7.5, 7.5, "skip"), (8.0, INF, "absent")],
        "goal": [(0, 5.0, "no"), (5.5, 7.0, "skip"), (7.5, INF, "absent")],
    },
    "put-airpod": {
        key: [(0, 2.5, "no"), (3.0, 3.0, "skip"), (3.5, 8.0, "yes"), (8.5, 8.5, "skip"), (9.0, 10.0, "no"),
              (10.5, 15.5, "yes"), (16.0, 16.5, "no"), (17.0, INF, "absent")]
        for key in ("s1", "s2", "goal")
    },
    "increase_temp": {
        "s1": [(0, 1.5, "no"), (2.0, INF, "yes")],
        "s2": [(0, 5.0, "no"), (5.5, INF, "yes")],
        "goal": [(0, 5.0, "no"), (5.5, INF, "yes")],
    },
}
#: The weak spot the 4B missed in the e2e benchmark: glasses off the face, held in hand.
GLASS_S2_WINDOW = (5.5, 6.5)


def truth(clip: str, key: str, t: float) -> str:
    for lo, hi, label in TRUTH[clip][key]:
        if lo - 1e-6 <= t <= hi + 1e-6:
            return label
    return "skip"


def stored_plan(clip: str) -> StoredPlan:
    user_goal, _, steps, goal_when = PLANS[clip]
    return StoredPlan(
        plan_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"follower-bench/{clip}")),
        plan_revision=1,
        steps=tuple(GuideStep.model_validate({"id": i, "say": say, "commands": [], "done_when": dw})
                    for i, say, dw in steps),
        goal_when=goal_when, user_goal=user_goal, context=None, task_revision=1,
    )


def follow_request(clip: str, plan: StoredPlan, image_b64: str, track: dict, seq: int) -> GuideFollowRequest:
    run_id = f"run-bench-{clip}"
    anchor = {"anchor_id": "a1", "role": "target", "label": PLANS[clip][1], "run_id": run_id,
              "track_id": f"t-bench-{clip}", "generation": 0, "state": track["state"]}
    if track["state"] == "tracking":
        anchor["box"] = track["box"]
    return GuideFollowRequest.model_validate({
        "session_id": str(uuid.uuid4()), "consent_ai": True,
        "scene": {"frame_id": f"bench-{clip}-{seq}", "image_base64": image_b64, "label": "카메라 현재 화면"},
        "plan_id": plan.plan_id, "plan_revision": plan.plan_revision, "current_step": plan.steps[0].id,
        # A track that never acquired has no anchor id yet; the app then sends no anchor at all.
        "anchors": [] if track["state"] == "acquiring" else [anchor],
        "trigger": "heartbeat", "intent_seq": seq,
        "fence": {"task_epoch": f"task-bench-{clip}", "run_id": run_id},
    })


# ------------------------------------------------------------------------------------------------- run

async def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        follower = LocalFollower(client, url=args.url, model=args.model, timeout_s=args.timeout,
                                 max_tokens=args.max_tokens)
        with out.open("a", encoding="utf-8") as sink:
            for clip in args.clips:
                plan = stored_plan(clip)
                tracks = {row["frame_seq"]: row for row in map(json.loads,
                          (Path(args.tracks_root) / clip / "frames.jsonl").read_text().splitlines())}
                frames = sorted((Path(args.frames_root) / clip).glob("f_*.jpg"))
                for repeat in range(args.repeat):
                    for idx in range(0, len(frames), args.sample_every):
                        t = round(idx / FPS_SOURCE, 2)
                        if not args.t_min <= t <= args.t_max:
                            continue
                        row = await one(follower, clip, plan, frames[idx], tracks[idx], idx, t)
                        row.update(model=args.label or args.model, repeat=repeat)
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                        sink.flush()
                        print(f"{row['model']} {clip} t={t:>5} {row['latency_ms']:>6.0f}ms "
                              f"{row.get('checks') or row.get('error')} goal={row.get('goal_seen')}", flush=True)


async def one(follower: LocalFollower, clip: str, plan: StoredPlan, frame: Path, track: dict,
              seq: int, t: float) -> dict:
    image_b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
    request = follow_request(clip, plan, image_b64, track, seq)
    prepared = prepare_frame_b64(image_b64, request.anchors)
    row: dict = {"clip": clip, "t": t, "track_state": track["state"]}
    started = time.perf_counter()
    try:
        body = await follower._post(follower.payload(request, plan, prepared))
        row["latency_ms"] = (time.perf_counter() - started) * 1000
        timings = body.get("timings") or {}
        usage = body.get("usage") or {}
        row.update(prompt_ms=timings.get("prompt_ms"), predicted_ms=timings.get("predicted_ms"),
                   prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
        raw = follower._arguments(body)
        output = GuideFollowToolOutput.model_validate(raw)
        if [c.step_id for c in output.step_checks] != plan.checklist_ids(request.current_step):
            raise ValueError("checklist ids")
        if any(v.anchor_id not in {a.anchor_id for a in request.anchors} for v in output.anchor_verdicts):
            raise ValueError("anchor provenance")
        row["checks"] = {c.step_id: c.visible for c in output.step_checks}
        row["goal_seen"] = output.goal_seen
        row["anchor"] = [v.matches for v in output.anchor_verdicts]
    except (ProviderError, ValidationError, ValueError) as exc:
        row.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
        row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return row


# ----------------------------------------------------------------------------------------------- score

def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))] if ordered else float("nan")


def score(paths: list[str]) -> None:
    rows = [json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line.strip()]
    models = list(dict.fromkeys(row["model"] for row in rows))
    print("| model | n | rejects | follow p50/p95 ms | step acc (yes/no truth) | yes recall | false yes "
          "(no truth / absent) | glass s2 hit | goal false yes | goal recall | unsure used |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    detail = []
    for model in models:
        mine = [r for r in rows if r["model"] == model]
        ok = [r for r in mine if "checks" in r]
        lat = [r["latency_ms"] for r in ok]
        conf: dict[tuple[str, str], int] = {}
        hit = s2n = 0
        g_fy = g_fy_n = g_yes = g_yes_n = 0
        for r in ok:
            for step, verdict in r["checks"].items():
                label = truth(r["clip"], step, r["t"])
                if label != "skip":
                    conf[label, verdict] = conf.get((label, verdict), 0) + 1
                if r["clip"] == "glass" and step == "s2" and GLASS_S2_WINDOW[0] <= r["t"] <= GLASS_S2_WINDOW[1]:
                    s2n += 1
                    hit += verdict == "yes"
            g = truth(r["clip"], "goal", r["t"])
            if g in ("no", "absent"):
                g_fy_n += 1
                g_fy += r["goal_seen"] == "yes"
            elif g == "yes":
                g_yes_n += 1
                g_yes += r["goal_seen"] == "yes"
        def c(label: str, verdict: str) -> int:
            return conf.get((label, verdict), 0)
        yn = sum(c(lb, v) for lb in ("yes", "no") for v in ("yes", "no", "unsure"))
        correct = c("yes", "yes") + c("no", "no")
        yes_n = sum(c("yes", v) for v in ("yes", "no", "unsure"))
        no_n = sum(c("no", v) for v in ("yes", "no", "unsure"))
        ab_n = sum(c("absent", v) for v in ("yes", "no", "unsure"))
        unsure = sum(c(lb, "unsure") for lb in ("yes", "no", "absent"))
        total = sum(conf.values())
        print(f"| {model} | {len(mine)} | {len(mine) - len(ok)} | {pct(lat, .5):.0f} / {pct(lat, .95):.0f} | "
              f"{correct}/{yn} ({correct / max(yn, 1):.0%}) | {c('yes', 'yes')}/{yes_n} | "
              f"{c('no', 'yes')}/{no_n} / {c('absent', 'yes')}/{ab_n} | {hit}/{s2n} | {g_fy}/{g_fy_n} | "
              f"{g_yes}/{g_yes_n} | {unsure}/{total} |")
        detail.append((model, conf, ok))
    print()
    for model, conf, ok in detail:
        print(f"{model}: confusion truth->verdict " + ", ".join(
            f"{lb}->{v}:{conf.get((lb, v), 0)}" for lb in ("yes", "no", "absent") for v in ("yes", "no", "unsure")))
        pm = [r["prompt_ms"] for r in ok if r.get("prompt_ms") is not None]
        dm = [r["predicted_ms"] for r in ok if r.get("predicted_ms") is not None]
        ct = [r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None]
        pt = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None]
        if pm:
            print(f"  server prompt_ms p50 {statistics.median(pm):.0f}, predicted_ms p50 {statistics.median(dm):.0f}, "
                  f"prompt tokens p50 {statistics.median(pt):.0f}, completion tokens p50 {statistics.median(ct):.0f}")
        for clip in PLANS:
            seq = " ".join(
                f"{r['t']:g}:" + "".join((r['checks'][s][0].upper() if truth(clip, s, r['t']) == 'yes' else r['checks'][s][0])
                                       for s in r['checks']) + ("G" if r["goal_seen"] == "yes" else "")
                for r in ok if r["clip"] == clip and r.get("repeat", 0) == 0)
            print(f"  {clip}: {seq}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True, help="loopback chat-completions URL")
    r.add_argument("--model", required=True, help="model alias sent to the server")
    r.add_argument("--label", help="name in the results (default: --model)")
    r.add_argument("--frames-root", required=True)
    r.add_argument("--tracks-root", required=True)
    r.add_argument("--out", required=True, help="JSONL, appended")
    r.add_argument("--clips", nargs="+", default=list(PLANS))
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--sample-every", type=int, default=SAMPLE_EVERY, help="every Nth 10 fps frame")
    r.add_argument("--t-min", type=float, default=0.0)
    r.add_argument("--t-max", type=float, default=INF)
    r.add_argument("--timeout", type=float, default=10.0)
    r.add_argument("--max-tokens", type=int, default=260)
    s = sub.add_parser("score")
    s.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args))
    else:
        score(args.paths)


if __name__ == "__main__":
    main()

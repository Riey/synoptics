#!/usr/bin/env python3
"""Replay a video clip, frame by frame, to a running tracker service and log what it answers.

Measurement harness only: it drives the frozen private ``/runs/*`` wire (``service.py``) exactly as the
application does (``backend/app/tracking.py``) -- ``/runs/start`` ``{run_id, target, fresh}``, then one
``/runs/frame`` ``{run_id, frame_seq, frame_b64, seed_box?}`` per frame, then ``/runs/stop`` -- and changes
nothing in the service. Standard library + the ``ffmpeg`` CLI only.

    python3 tools/tracker_service/replay_clip.py --clip /path/clip.mov --target "glasses" \\
        --base-url http://127.0.0.1:8093 --fps 10 --long-side 1280 --out-dir /tmp/replay/glass-10

Outputs in ``--out-dir``: ``frames.jsonl`` (one line per frame: frame_seq, video_t_s, state, transition,
box, confidence, compute_ms, latency_ms, http_status) and ``summary.json`` (acquisition frame, occluded
episodes, lost events, coverage, compute/latency p50/p95, the service policy read from ``/health``).

Frames are sent closed-loop (the next frame leaves when the previous answer is back) unless ``--realtime``
paces them to the clip clock. The occlusion window counts PROCESSED frames, so the extraction fps -- not
the pacing -- decides how many camera seconds a window covers.

Pre-extracted frames (a host without ffmpeg, e.g. the demo Mac): ``--frames-dir DIR --use-existing-frames``
replays the ``f_*.jpg`` already in DIR as they are -- no ffmpeg, no ``--clip`` needed, and the frames are
never deleted. ``--fps`` must then be the fps they were extracted at (it only sets ``video_t_s`` and the
``--realtime`` clock). Without the flag the same happens automatically when ``--frames-dir`` already holds
``f_*.jpg`` and ffmpeg is NOT on PATH; with ffmpeg present the default is unchanged (re-extract, replacing
the ``f_*.jpg`` in that directory).
"""

from __future__ import annotations

import argparse
import base64
import http.client
import ipaddress
import json
import math
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

STATES = ("acquiring", "tracking", "occluded", "lost")


class ReplayError(RuntimeError):
    pass


# --------------------------------------------------------------------------- frame extraction
def extract_frames(clip: Path, fps: float, long_side: int, out_dir: Path, quality: int) -> list[Path]:
    """Decode ``clip`` at ``fps`` and resize so the LONG side is ``long_side`` (aspect kept, even dims)."""
    if shutil.which("ffmpeg") is None:
        raise ReplayError("ffmpeg not found on PATH")
    if not clip.is_file():
        raise ReplayError(f"clip not found: {clip}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f_*.jpg"):
        old.unlink()
    scale = (f"scale='if(gte(iw,ih),{long_side},-2)':'if(gte(iw,ih),-2,{long_side})'"
             ":flags=bicubic")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(clip),
        "-vf", f"fps={fps},{scale}", "-q:v", str(quality), str(out_dir / "f_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ReplayError(f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}")
    frames = sorted(out_dir.glob("f_*.jpg"))
    if not frames:
        raise ReplayError("ffmpeg produced no frames")
    return frames


def existing_frames(frames_dir: Path) -> list[Path]:
    """The ``f_*.jpg`` already in ``frames_dir``, in extraction order. Read only: nothing is deleted."""
    if not frames_dir.is_dir():
        raise ReplayError(f"frames dir not found: {frames_dir}")
    frames = sorted(frames_dir.glob("f_*.jpg"))
    if not frames:
        raise ReplayError(f"no f_*.jpg frames in {frames_dir}")
    return frames


def jpeg_size(path: Path) -> tuple[int, int]:
    """(width, height) from the first SOF marker; refuses a file that is not a baseline/progressive JPEG."""
    data = path.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ReplayError(f"not a JPEG: {path}")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ReplayError(f"malformed JPEG (marker expected at byte {i}): {path}")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # standalone markers
            i += 2
            continue
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > len(data):
                break
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height
        i += 2 + length
    raise ReplayError(f"no frame header in JPEG: {path}")


def use_existing(args: argparse.Namespace) -> bool:
    """Replay pre-extracted frames instead of extracting: the flag, or frames present and no ffmpeg."""
    if args.use_existing_frames:
        return True
    if not args.frames_dir or shutil.which("ffmpeg") is not None:
        return False
    return any(Path(args.frames_dir).glob("f_*.jpg"))


# --------------------------------------------------------------------------- HTTP
def check_loopback(base_url: str) -> tuple[str, int]:
    """The service is loopback-only (reached through an SSH forward); refuse anything else."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.path not in ("", "/"):
        raise ReplayError(f"base URL must be http://<loopback>:<port>: {base_url!r}")
    host = parsed.hostname
    try:
        loop = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loop = host == "localhost"
    if not loop:
        raise ReplayError(f"base URL host is not loopback: {host!r}")
    return host, parsed.port or 80


class Client:
    """One keep-alive connection with ``TCP_NODELAY`` on the client socket (no per-request reconnect).

    Measured 2026-10-01: through an ``ssh -L`` forward every POST had a ~40 ms round-trip floor whatever the
    work (a tiny ``/runs/start`` 40.8 ms, a 422-rejected frame 42 ms), with this client and with ``urllib``
    alike, while the same frames sent on the GPU node itself carried ~1 ms over ``compute_ms`` (n=39). So
    ``latency_ms`` here includes the forward; ``compute_ms`` is the service's own step time."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.conn: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        conn.connect()
        assert conn.sock is not None
        conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return conn

    def request(self, method: str, path: str, body: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
        data = None if body is None else json.dumps(body).encode()
        headers = {"content-type": "application/json"} if data is not None else {}
        for attempt in (0, 1):
            if self.conn is None:
                self.conn = self._connect()
            try:
                self.conn.request(method, path, body=data, headers=headers)
                resp = self.conn.getresponse()
                raw = resp.read().decode(errors="replace")
                break
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                # A keep-alive connection the server already closed: reconnect once, then fail loudly.
                self.conn.close()
                self.conn = None
                if attempt:
                    raise
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            payload = {"error": raw[:200]}
        return resp.status, payload

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# --------------------------------------------------------------------------- statistics
def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (no interpolation): the reported value is one that was measured."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return round(ordered[rank - 1], 2)


def summarize(rows: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    n = len(rows)
    acq = next((r["frame_seq"] for r in rows if r["state"] == "tracking"), None)
    transitions: dict[str, int] = {}
    states: dict[str, int] = {s: 0 for s in STATES}
    for r in rows:
        transitions[r["transition"]] = transitions.get(r["transition"], 0) + 1
        states[r["state"]] = states.get(r["state"], 0) + 1

    # Occluded episode = a maximal run of consecutive `occluded` frames; it ends in `tracking`
    # (recovered), in `lost`, or with the clip.
    episodes: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for r in rows:
        if r["state"] == "occluded":
            if cur is None:
                cur = {"start": r["frame_seq"], "frames": 0}
            cur["frames"] += 1
        elif cur is not None:
            cur["ended_by"] = r["state"]
            episodes.append(cur)
            cur = None
    if cur is not None:
        cur["ended_by"] = "end_of_clip"
        episodes.append(cur)
    lost_events = [r["frame_seq"] for r in rows if r["transition"] == "lost"]
    with_box = sum(1 for r in rows if r.get("box") is not None)
    after_acq = [r for r in rows if acq is not None and r["frame_seq"] >= acq]
    compute = [r["compute_ms"] for r in rows if r.get("compute_ms") is not None]
    latency = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    # Steady-state SAM steps only: after acquisition and before any loss (lost frames do no model work).
    steady = [r for r in after_acq[1:] if r["state"] in ("tracking", "occluded")]
    return {
        "frames": n,
        "fps": fps,
        "acquisition_frame": acq,
        "acquisition_video_t_s": None if acq is None else round(acq / fps, 3),
        "frames_in_state": states,
        "transitions": transitions,
        "occluded_episodes": len(episodes),
        "occluded_episode_detail": episodes,
        "lost_events": len(lost_events),
        "lost_at_frames": lost_events,
        "lost_at_video_t_s": [round(f / fps, 3) for f in lost_events],
        "frames_with_box": with_box,
        "coverage": round(with_box / n, 4) if n else None,
        "coverage_after_acquisition": (round(sum(1 for r in after_acq if r.get("box") is not None)
                                             / len(after_acq), 4) if after_acq else None),
        "compute_ms": {"p50": percentile(compute, 50), "p95": percentile(compute, 95), "n": len(compute)},
        "latency_ms": {"p50": percentile(latency, 50), "p95": percentile(latency, 95), "n": len(latency)},
        "steady_compute_ms": {"p50": percentile([r["compute_ms"] for r in steady], 50),
                              "p95": percentile([r["compute_ms"] for r in steady], 95), "n": len(steady)},
        "steady_latency_ms": {"p50": percentile([r["latency_ms"] for r in steady], 50),
                              "p95": percentile([r["latency_ms"] for r in steady], 95), "n": len(steady)},
        "first_frame_compute_ms": rows[0].get("compute_ms") if rows else None,
        "acquisition_frame_compute_ms": (next((r["compute_ms"] for r in rows if r["frame_seq"] == acq), None)
                                         if acq is not None else None),
    }


# --------------------------------------------------------------------------- replay
def parse_seed(text: str | None) -> dict[str, float] | None:
    if text is None:
        return None
    parts = text.split(",")
    if len(parts) != 4:
        raise ReplayError("--seed-box must be x,y,width,height (normalised 0..1)")
    x, y, w, h = (float(p) for p in parts)
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ReplayError("--seed-box values must be normalised 0..1 with positive size")
    return {"x": x, "y": y, "width": w, "height": h}


def replay(args: argparse.Namespace) -> dict[str, Any]:
    host, port = check_loopback(args.base_url)
    client = Client(host, port, args.timeout)
    try:
        return _replay(args, client, f"http://{host}:{port}")
    finally:
        client.close()


def _replay(args: argparse.Namespace, client: Client, base: str) -> dict[str, Any]:
    seed = parse_seed(args.seed_box)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir) if args.frames_dir else out_dir / "frames"
    reuse = use_existing(args)
    if reuse:
        frames = existing_frames(frames_dir)
        print(f"replay: using {len(frames)} existing frames in {frames_dir} (no extraction, nothing deleted)",
              file=sys.stderr)
    else:
        if not args.clip:
            raise ReplayError("--clip is required unless existing frames are replayed")
        frames = extract_frames(Path(args.clip), args.fps, args.long_side, frames_dir, args.jpeg_quality)
    frame_size = jpeg_size(frames[0])
    if args.max_frames:
        frames = frames[: args.max_frames]

    status, health_before = client.request("GET", "/health", None)
    if status != 200 or not health_before.get("ready"):
        raise ReplayError(f"tracker not ready ({status}): {str(health_before)[:300]}")
    policy = health_before.get("policy", {})

    run_id = args.run_id or f"replay-{uuid.uuid4().hex[:12]}"
    status, started = client.request("POST", "/runs/start",
                                    {"run_id": run_id, "target": args.target, "fresh": True})
    if status != 200:
        raise ReplayError(f"/runs/start failed ({status}): {started}")

    rows: list[dict[str, Any]] = []
    error: str | None = None
    t0 = time.monotonic()
    try:
        with open(out_dir / "frames.jsonl", "w") as jl:
            for seq, path in enumerate(frames):
                if args.realtime:
                    due = t0 + seq / args.fps
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                body: dict[str, Any] = {
                    "run_id": run_id, "frame_seq": seq,
                    "frame_b64": base64.b64encode(path.read_bytes()).decode(),
                }
                if seed is not None and seq == 0:
                    body["seed_box"] = seed
                sent = time.perf_counter()
                status, resp = client.request("POST", "/runs/frame", body)
                latency = round((time.perf_counter() - sent) * 1e3, 2)
                if status != 200:
                    error = f"/runs/frame seq={seq} failed ({status}): {resp}"
                    break
                row = {
                    "frame_seq": seq,
                    "video_t_s": round(seq / args.fps, 3),
                    "state": resp.get("state"),
                    "transition": resp.get("transition"),
                    "box": resp.get("box"),
                    "confidence": resp.get("confidence"),
                    "source": resp.get("source"),
                    "generation": resp.get("generation"),
                    "compute_ms": resp.get("compute_ms"),
                    "latency_ms": latency,
                    "http_status": status,
                }
                if row["state"] not in STATES:
                    error = f"unexpected state at seq={seq}: {row['state']!r}"
                    break
                rows.append(row)
                jl.write(json.dumps(row) + "\n")
    finally:
        stop_status, _ = client.request("POST", "/runs/stop", {"run_id": run_id})

    _, health_after = client.request("GET", "/health", None)
    summary = {
        "clip": str(args.clip) if args.clip else None,
        "frames_source": "existing" if reuse else "extracted",
        "frames_dir": str(frames_dir),
        "frame_size_first": list(frame_size),
        "target": args.target,
        "seed_box": seed,
        "base_url": base,
        "run_id": run_id,
        # Existing frames were sized by whoever extracted them: frame_size_first is what was measured.
        "long_side": None if reuse else args.long_side,
        "realtime": bool(args.realtime),
        "http_client": "keep-alive, TCP_NODELAY",
        "frames_extracted": len(frames),
        "wall_s": round(time.monotonic() - t0, 2),
        "policy": {"occluded_max_frames": policy.get("occluded_max_frames"),
                   "occluded_max_frames_source": policy.get("occluded_max_frames_source"),
                   "track_keep_score": policy.get("track_keep_score")},
        "inferences_before": health_before.get("inferences"),
        "inferences_after": health_after.get("inferences"),
        "stop_http_status": stop_status,
        "error": error,
        **summarize(rows, args.fps),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", help="video file (read only); required unless existing frames are replayed")
    ap.add_argument("--target", required=True, help="English target noun for GroundingDINO (1..40 chars)")
    ap.add_argument("--base-url", required=True, help="tracker base URL, loopback only (e.g. http://127.0.0.1:8093)")
    ap.add_argument("--fps", type=float, default=10.0, help="extraction fps (default 10)")
    ap.add_argument("--long-side", type=int, default=1280, help="resize so the long side is this many px")
    ap.add_argument("--jpeg-quality", type=int, default=3, help="ffmpeg -q:v (2=best .. 31), default 3")
    ap.add_argument("--seed-box", help="x,y,width,height normalised, sent on frame 0 only (manual selection)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames-dir", help="where extracted JPEGs go (default <out-dir>/frames); with existing "
                    "f_*.jpg and no ffmpeg on PATH they are replayed as they are")
    ap.add_argument("--use-existing-frames", action="store_true",
                    help="replay the f_*.jpg already in --frames-dir: no ffmpeg, never deletes them; "
                    "--fps must be their extraction fps")
    ap.add_argument("--run-id")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--realtime", action="store_true", help="pace frames to the clip clock instead of closed-loop")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)
    if not 1 <= len(args.target) <= 40:
        ap.error("--target must be 1..40 characters (service contract)")
    if args.fps <= 0 or args.long_side < 64:
        ap.error("--fps must be > 0 and --long-side >= 64")
    if args.use_existing_frames and not args.frames_dir:
        ap.error("--use-existing-frames needs --frames-dir")
    if not args.clip and not args.frames_dir:
        ap.error("--clip is required (or --frames-dir with existing frames)")
    try:
        s = replay(args)
    except ReplayError as exc:
        print(f"replay error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: s[k] for k in (
        "target", "fps", "frames", "acquisition_frame", "occluded_episodes", "lost_events", "lost_at_frames",
        "coverage", "compute_ms", "latency_ms", "first_frame_compute_ms", "policy", "error")}, ensure_ascii=False))
    return 1 if s["error"] else 0


if __name__ == "__main__":
    sys.exit(main())

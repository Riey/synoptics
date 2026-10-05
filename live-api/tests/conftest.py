from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

import pytest
from PIL import Image
from starlette.testclient import TestClient

from synoptics_live.app import create_app
from synoptics_live.contracts import PROTOCOL
from synoptics_live.envelope import pack, unpack
from synoptics_live.settings import Settings
from synoptics_live.tts import MIME, AudioClip


def fast_settings(**overrides) -> Settings:
    base = dict(
        mock_speed=50.0,
        mock_step_s=2.0,
        hello_timeout_s=0.3,
        resume_grace_s=0.5,
        tts_url=None,
    )
    base.update(overrides)
    return Settings(**base)


@dataclass
class FakeTts:
    is_ready: bool = True
    calls: list[tuple[str, str]] = field(default_factory=list)

    def ready_now(self) -> bool:
        return self.is_ready

    async def ready(self) -> bool:
        return self.is_ready

    @property
    def voice_label(self) -> str | None:
        return "fake-voice"

    async def synthesize(self, text: str, fmt: str) -> AudioClip | None:
        self.calls.append((text, fmt))
        if not self.is_ready:
            return None
        return AudioClip(data=b"OggS-fake-" + text.encode(), mime=MIME[fmt], duration_ms=1234)


@pytest.fixture
def make_client():
    clients: list[TestClient] = []

    def factory(settings: Settings | None = None, tts: Any = None, **overrides) -> TestClient:
        app = create_app(settings or fast_settings(**overrides), tts=tts)
        client = TestClient(app)
        client.__enter__()  # one event loop (portal) for the whole test: sessions outlive single sockets
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@lru_cache(maxsize=8)
def jpeg(w: int = 640, h: int = 360) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 150)).save(buf, "JPEG", quality=70)
    return buf.getvalue()


def frame(seq: int, w: int = 640, h: int = 360, payload: bytes | None = None, **extra) -> bytes:
    header = {"t": "frame", "seq": seq, "captured_at": 1_759_479_000_000 + seq, "w": w, "h": h, **extra}
    return pack(header, jpeg(w, h) if payload is None else payload)


def create_session(client: TestClient, **body) -> dict:
    resp = client.post("/v1/sessions", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


class Closed(Exception):
    def __init__(self, code: int):
        super().__init__(f"closed {code}")
        self.code = code


class Live:
    """A test-side live socket that records every message it has seen."""

    def __init__(self, ws):
        self.ws = ws
        self.seen: list[dict] = []
        self.audio: list[tuple[dict, bytes]] = []
        self.seq = 0
        self.state: dict | None = None
        self._claimed: set[int] = set()

    def send(self, obj: dict) -> None:
        self.ws.send_text(json.dumps(obj))

    def recv(self) -> dict:
        msg = self.ws.receive()
        if msg["type"] == "websocket.close":
            raise Closed(msg.get("code", 1000))
        if msg.get("bytes") is not None:
            header, payload = unpack(msg["bytes"])
            self.audio.append((header, payload))
            return {"type": "_binary", **header, "_payload": payload}
        obj = json.loads(msg["text"])
        self.seen.append(obj)
        if obj["type"] == "state":
            if self.state is not None:
                assert obj["rev"] >= self.state["rev"], "rev went backwards"
            self.state = obj
        return obj

    def until(self, pred: Callable[[dict], bool], limit: int = 500) -> dict:
        for _ in range(limit):
            m = self.recv()
            if pred(m):
                return m
        raise AssertionError("message never arrived")

    def seen_or_next(self, pred: Callable[[dict], bool], limit: int = 500) -> dict:
        """First match already received (and not returned before by this method), else wait for one."""
        for m in self.seen:
            if id(m) not in self._claimed and pred(m):
                self._claimed.add(id(m))
                return m
        m = self.until(pred, limit)
        self._claimed.add(id(m))
        return m

    def until_type(self, kind: str, limit: int = 500) -> dict:
        return self.until(lambda m: m["type"] == kind, limit)

    def hello(self, token: str, **extra) -> tuple[dict, dict]:
        self.send({"type": "hello", "token": token, "protocol": PROTOCOL, "resume_rev": None, **extra})
        ready = self.recv()
        assert ready["type"] == "ready", ready
        state = self.recv()
        assert state["type"] == "state", state
        return ready, state

    def frame(self, **extra) -> dict:
        """Send one stream frame and return its track (other messages are recorded on the way)."""
        seq = self.seq
        self.seq += 1
        self.ws.send_bytes(frame(seq, **extra))
        return self.until(lambda m: m["type"] == "track" and m["seq"] == seq)

    def drive(self, pred: Callable[[dict], bool], max_frames: int = 400, pause_s: float = 0.004) -> dict:
        """Stream frames (keeping the mock's clock moving) until the latest state satisfies ``pred``."""
        for _ in range(max_frames):
            if self.state is not None and pred(self.state):
                return self.state
            self.frame()
            time.sleep(pause_s)
        raise AssertionError(f"state never satisfied predicate; last: {self.state}")

    def says(self) -> list[dict]:
        return [m for m in self.seen if m["type"] == "say"]

    def errors(self) -> list[dict]:
        return [m for m in self.seen if m["type"] == "error"]


def open_live(client: TestClient, sid: str):
    return client.websocket_connect(f"/v1/sessions/{sid}/live")

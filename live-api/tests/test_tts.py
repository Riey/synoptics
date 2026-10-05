import asyncio
import io
import json
import os
import shutil
import wave

import httpx
import pytest
from conftest import FakeTts, Live, create_session, open_live

from synoptics_live.tts import HttpTtsBridge, pick_format, wav_duration_ms

START = {"type": "start", "goal": "물건을 정리하기", "consent_ai": True}


def make_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


# ------------------------------------------------------------------ through the socket (fake bridge)


def test_ready_tts_sends_pending_say_then_audio(make_client):
    tts = FakeTts(is_ready=True)
    c = make_client(tts=tts)
    s = create_session(c)
    with open_live(c, s["session_id"]) as ws:
        L = Live(ws)
        L.hello(s["token"], audio_accept=["audio/ogg"])
        L.send(START)
        say = L.until_type("say")
        assert say["audio"] == "pending"
        audio = L.until_type("_binary")
        assert audio["t"] == "audio" and audio["line_id"] == say["line_id"]
        assert audio["mime"] == "audio/ogg; codecs=opus" and audio["duration_ms"] == 1234
        assert audio["_payload"] == b"OggS-fake-" + say["text"].encode()
    assert tts.calls[0] == ("계획을 세우는 중입니다.", "ogg")


def test_mp4_when_client_prefers_it(make_client):
    tts = FakeTts(is_ready=True)
    c = make_client(tts=tts)
    s = create_session(c)
    with open_live(c, s["session_id"]) as ws:
        L = Live(ws)
        L.hello(s["token"], audio_accept=["audio/mp4", "audio/ogg"])
        L.send(START)
        L.until_type("say")
        assert L.until_type("_binary")["mime"] == "audio/mp4"
    assert tts.calls[0][1] == "mp4"


def test_not_ready_means_audio_none(make_client):
    tts = FakeTts(is_ready=False)
    c = make_client(tts=tts)
    s = create_session(c)
    with open_live(c, s["session_id"]) as ws:
        L = Live(ws)
        L.hello(s["token"])
        L.send(START)
        assert L.until_type("say")["audio"] == "none"
        L.drive(lambda m: m["phase"] == "running")
    assert all(say["audio"] == "none" for say in L.says())
    assert tts.calls == [] and L.audio == []


@pytest.mark.parametrize("prefs", [{"voice_out": False}, {"tts": "browser"}])
def test_prefs_disable_server_audio(make_client, prefs):
    tts = FakeTts(is_ready=True)
    c = make_client(tts=tts)
    s = create_session(c)
    with open_live(c, s["session_id"]) as ws:
        L = Live(ws)
        L.hello(s["token"])
        L.send({"type": "prefs", **prefs})
        L.send(START)
        assert L.until_type("say")["audio"] == "none"
        L.drive(lambda m: m["phase"] == "running")
        L.seen_or_next(lambda m: m["type"] == "say" and m["text"].startswith("1단계."))
        # turning server TTS back on affects the next line
        L.send({"type": "prefs", "voice_out": True, "tts": "server"})
        L.send({"type": "replan_now"})
        assert L.until_type("say")["audio"] == "pending"
    assert all(say["audio"] == "none" for say in L.says()[:-1])
    assert len(tts.calls) == 1


# ------------------------------------------------------------------ bridge internals (no network)


def test_pick_format():
    assert pick_format(None) == "ogg"
    assert pick_format([]) == "ogg"
    assert pick_format(["audio/mp4"]) == "mp4"
    assert pick_format(["audio/ogg; codecs=opus", "audio/mp4"]) == "ogg"
    assert pick_format(["audio/webm", "audio/aac"]) == "mp4"


def test_wav_duration():
    assert wav_duration_ms(make_wav(0.5)) == 500
    assert wav_duration_ms(make_wav(1.25, 24000)) == 1250
    with pytest.raises(ValueError):
        wav_duration_ms(b"not a wav")


def _bridge(handler, voice: str = "default") -> tuple[HttpTtsBridge, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    bridge = HttpTtsBridge("http://tts.test", voice, client=client)
    return bridge, seen


def test_bridge_readiness_is_the_service_health():
    def handler(request):
        assert request.url.path == "/health"
        return httpx.Response(200, json={"ready": True, "device": "cuda", "weights": "lora=x"})

    async def run():
        bridge, seen = _bridge(handler, voice="op")
        assert bridge.ready_now() is False and bridge.voice_label is None  # never checked yet
        assert await bridge.ready() is True
        assert bridge.voice_label == "op"
        await bridge.ready()  # cached for 10 s
        assert len(seen) <= 2  # ready_now may have refreshed once

    asyncio.run(run())


def test_bridge_not_ready_and_unreachable():
    async def run():
        b1, _ = _bridge(lambda r: httpx.Response(200, json={"ready": False}))
        assert await b1.ready() is False
        assert await b1.synthesize("안녕", "ogg") is None

        def boom(request):
            raise httpx.ConnectError("down")

        b2, _ = _bridge(boom)
        assert await b2.ready() is False

    asyncio.run(run())


@pytest.mark.skipif(not shutil.which(os.environ.get("LIVE_FFMPEG", "/usr/bin/ffmpeg")), reason="ffmpeg missing")
def test_bridge_synthesize_encodes_and_caches():
    wav = make_wav(0.5)

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True})
        assert request.url.path == "/tts"
        assert json.loads(request.content) == {"text": "안녕하세요"}
        return httpx.Response(200, content=wav, headers={"content-type": "audio/wav"})

    async def run():
        bridge, seen = _bridge(handler)
        assert await bridge.ready()
        clip = await bridge.synthesize("안녕하세요", "ogg")
        assert clip is not None and clip.data[:4] == b"OggS" and clip.duration_ms == 500
        assert clip.mime == "audio/ogg; codecs=opus"
        again = await bridge.synthesize("안녕하세요", "ogg")
        assert again is clip
        assert len([r for r in seen if r.url.path == "/tts"]) == 1
        mp4 = await bridge.synthesize("안녕하세요", "mp4")
        assert mp4 is not None and b"ftyp" in mp4.data[:32] and mp4.mime == "audio/mp4"

    asyncio.run(run())

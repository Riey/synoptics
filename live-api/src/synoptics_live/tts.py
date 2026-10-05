"""Optional generic WAV-to-Opus/AAC bridge for protocol compatibility.

No speech model or voice service is included in this submission. The default has no
``LIVE_TTS_URL`` and clients use browser/device speech. If an operator explicitly
configures a separately provided service, this bridge accepts its WAV response.
Otherwise the client uses browser TTS (spec §7).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import struct
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

log = logging.getLogger("synoptics_live.tts")

AudioFormat = Literal["ogg", "mp4"]
MIME = {"ogg": "audio/ogg; codecs=opus", "mp4": "audio/mp4"}
READY_TTL_S = 10.0
HEALTH_TIMEOUT_S = 2.0
SYNTH_TIMEOUT_S = 10.0
CACHE_SIZE = 64


@dataclass(frozen=True)
class AudioClip:
    data: bytes
    mime: str
    duration_ms: int


class TtsBridge(Protocol):
    def ready_now(self) -> bool:
        """Last known readiness, without waiting (a stale value triggers a background refresh)."""

    async def ready(self) -> bool:
        """Readiness, refreshed if older than the cache TTL."""

    @property
    def voice_label(self) -> str | None: ...

    async def synthesize(self, text: str, fmt: AudioFormat) -> AudioClip | None:
        """Encoded audio for ``text``, or None on any failure (never raises)."""


def pick_format(audio_accept: list[str] | None) -> AudioFormat:
    """The first entry of the client's list we can produce; Ogg Opus when absent or nothing matches."""
    for entry in audio_accept or []:
        e = entry.lower().strip()
        if e.startswith("audio/ogg") or e.startswith("audio/opus"):
            return "ogg"
        if e.startswith("audio/mp4") or e.startswith("audio/aac") or e.startswith("audio/m4a"):
            return "mp4"
    return "ogg"


def wav_duration_ms(wav: bytes) -> int:
    """Duration from the RIFF ``fmt `` byte rate and ``data`` size (works for PCM and float WAV)."""
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, byte_rate, data_len = 12, None, None
    while pos + 8 <= len(wav):
        cid, size = wav[pos : pos + 4], struct.unpack_from("<I", wav, pos + 4)[0]
        body = pos + 8
        if cid == b"fmt " and size >= 16:
            byte_rate = struct.unpack_from("<I", wav, body + 8)[0]
        elif cid == b"data":
            # Streaming writers put 0 or 0xFFFFFFFF here; use what is actually there.
            data_len = min(size, len(wav) - body) if size not in (0, 0xFFFFFFFF) else len(wav) - body
            break
        pos = body + size + (size & 1)
    if not byte_rate or data_len is None:
        raise ValueError("WAV without fmt/data chunk")
    return int(round(data_len * 1000 / byte_rate))


class HttpTtsBridge:
    def __init__(
        self,
        base_url: str,
        voice: str = "default",
        ffmpeg: str = "/usr/bin/ffmpeg",
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        #: The label reported as ``tts_voice``; the service itself has exactly one voice.
        self.voice = voice
        self.ffmpeg = ffmpeg
        self._client = client
        self._ready = False
        self._checked_at = -math.inf
        self._refreshing: asyncio.Task | None = None
        self._cache: OrderedDict[tuple, AudioClip] = OrderedDict()

    @property
    def voice_label(self) -> str | None:
        return self.voice if self._ready else None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def refresh(self) -> bool:
        ready = False
        try:
            resp = await self._http().get(f"{self.base_url}/health", timeout=HEALTH_TIMEOUT_S)
            ready = resp.status_code == 200 and resp.json().get("ready") is True
        except Exception as exc:  # unreachable, bad JSON, ...
            log.info("tts health failed: %r", exc)
        self._ready = ready
        self._checked_at = time.monotonic()
        return ready

    def _stale(self) -> bool:
        return time.monotonic() - self._checked_at > READY_TTL_S

    def ready_now(self) -> bool:
        if self._stale() and (self._refreshing is None or self._refreshing.done()):
            try:
                self._refreshing = asyncio.get_running_loop().create_task(self.refresh())
            except RuntimeError:
                pass
        return self._ready

    async def ready(self) -> bool:
        if self._stale():
            return await self.refresh()
        return self._ready

    async def synthesize(self, text: str, fmt: AudioFormat) -> AudioClip | None:
        if not self._ready:
            return None
        key = (text, fmt)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            clip = await asyncio.wait_for(self._synthesize(text, fmt), SYNTH_TIMEOUT_S)
        except Exception as exc:
            log.warning("tts synthesis failed: %r", exc)
            return None
        self._cache[key] = clip
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return clip

    async def _synthesize(self, text: str, fmt: AudioFormat) -> AudioClip:
        resp = await self._http().post(f"{self.base_url}/tts", json={"text": text}, timeout=SYNTH_TIMEOUT_S)
        resp.raise_for_status()
        wav = resp.content
        duration_ms = wav_duration_ms(wav)
        data = await self.encode(wav, fmt)
        return AudioClip(data=data, mime=MIME[fmt], duration_ms=duration_ms)

    async def encode(self, wav: bytes, fmt: AudioFormat) -> bytes:
        base = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-vn"]
        if fmt == "ogg":
            return await _run(base + ["-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "pipe:1"], wav)
        # mp4 needs a seekable output for a normal (non-fragmented) moov atom.
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.m4a")
            await _run(base + ["-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", "-f", "mp4", "-y", out], wav)
            with open(out, "rb") as fh:
                return fh.read()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


async def _run(cmd: list[str], stdin: bytes) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await proc.communicate(stdin)
    except asyncio.CancelledError:
        proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {err.decode(errors='replace')[:300]}")
    return out

"""Runtime configuration, read from the environment once per app instance."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _str(name: str, default: str | None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or None


def _prefixes(raw: str | None) -> dict[str, str]:
    """``host=/prefix,host2=/prefix2`` -> {host: prefix}; hosts lower-cased, prefixes without a trailing slash."""
    out: dict[str, str] = {}
    for item in (raw or "").split(","):
        host, sep, prefix = item.strip().partition("=")
        prefix = prefix.strip().rstrip("/")
        if sep and host.strip() and prefix.startswith("/"):
            out[host.strip().lower()] = prefix
    return out


@dataclass
class Settings:
    access_code: str | None = None
    engine: str = "mock"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    #: host -> mount path, for reverse proxies that strip it without sending X-Forwarded-Prefix (Tailscale Serve).
    path_prefixes: dict[str, str] = field(default_factory=dict)
    # mock engine
    mock_speed: float = 1.0
    mock_step_s: float = 8.0
    # real engine: the demo backend it drives (None disables the real engine)
    upstream_url: str | None = None
    upstream_access_code: str | None = None
    # sessions
    max_sessions: int = 32
    create_per_ip_per_min: int = 5
    idle_ttl_s: float = 15 * 60
    resume_grace_s: float = 30.0
    hello_timeout_s: float = 5.0
    max_message_bytes: int = 512 * 1024
    #: §15: the bound for a control message that may carry reference photos (start / plan_answer). Each photo
    #: is still ≤1.5MB and ≤12 per message; this is the base64-inflated envelope the demo backend also allows.
    max_reference_message_bytes: int = 10 * 1024 * 1024
    credit_violation_limit: int = 20
    credit_stale_s: float = 2.0
    # Submission default: browser/device speech; no server voice endpoint is bundled.
    tts_url: str | None = None
    tts_voice: str = "default"
    ffmpeg: str = "/usr/bin/ffmpeg"
    # build info
    build_commit: str = "dev"
    build_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @classmethod
    def from_env(cls) -> Settings:
        origins = _str("LIVE_CORS_ORIGINS", "*") or "*"
        return cls(
            access_code=_str("LIVE_ACCESS_CODE", None),
            engine=_str("LIVE_ENGINE", "mock") or "mock",
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            path_prefixes=_prefixes(_str("LIVE_PATH_PREFIXES", None)),
            mock_speed=max(_float("LIVE_MOCK_SPEED", 1.0), 0.01),
            mock_step_s=max(_float("LIVE_MOCK_STEP_S", 8.0), 1.5),
            upstream_url=_str("LIVE_UPSTREAM_URL", "http://127.0.0.1:8045"),
            upstream_access_code=_str("LIVE_UPSTREAM_ACCESS_CODE", None),
            hello_timeout_s=_float("LIVE_HELLO_TIMEOUT_S", 5.0),
            resume_grace_s=_float("LIVE_RESUME_GRACE_S", 30.0),
            tts_url=_str("LIVE_TTS_URL", None),
            tts_voice=_str("LIVE_TTS_VOICE", "default") or "default",
            ffmpeg=_str("LIVE_FFMPEG", "/usr/bin/ffmpeg") or "/usr/bin/ffmpeg",
            build_commit=_str("LIVE_BUILD_COMMIT", "dev") or "dev",
            build_time=_str("LIVE_BUILD_TIME", None)
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @property
    def any_origin(self) -> bool:
        return "*" in self.cors_origins

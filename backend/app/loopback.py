"""One rule for every local-only endpoint the app may send a frame to.

A frame leaves the app for these endpoints, so a hostname that merely *looks* local is not enough: the
configured string is parsed and the host is checked, which also rejects userinfo (``user@host``) that
would otherwise smuggle a different host past a prefix test. The same rule therefore covers the local
follower and the standalone tracker service instead of each keeping its own convention.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from backend.app.errors import StageConfigError

LOOPBACK_HELP = "must address a loopback host (127.0.0.0/8, ::1 or localhost)"


def loopback_endpoint(url: str, env_name: str) -> str:
    """Accept only a credential-free http(s) URL whose PARSED host is this machine."""
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise StageConfigError(f"{env_name} is not a parseable URL") from exc
    if parsed.scheme not in {"http", "https"}:
        raise StageConfigError(f"{env_name} must be an http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise StageConfigError(f"{env_name} must not carry credentials")
    host = parsed.hostname
    if not host:
        raise StageConfigError(f"{env_name} has no host")
    if host == "localhost":
        return url
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise StageConfigError(f"{env_name} {LOOPBACK_HELP}") from exc
    if not address.is_loopback:
        raise StageConfigError(f"{env_name} {LOOPBACK_HELP}")
    return url

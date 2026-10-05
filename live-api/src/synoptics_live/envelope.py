"""Binary envelope (§4.2): ``[u32 BE header_len][UTF-8 JSON object header][payload]``."""

from __future__ import annotations

import json
import struct
from typing import Any

#: A header is a few fields; anything larger is a broken or hostile sender.
MAX_HEADER_BYTES = 4096


class EnvelopeError(ValueError):
    """The bytes are not a valid envelope."""


def pack(header: dict[str, Any], payload: bytes = b"") -> bytes:
    if not isinstance(header, dict):
        raise EnvelopeError("header must be a JSON object")
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_HEADER_BYTES:
        raise EnvelopeError(f"header is {len(raw)} bytes (max {MAX_HEADER_BYTES})")
    return struct.pack(">I", len(raw)) + raw + bytes(payload)


def unpack(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < 4:
        raise EnvelopeError("message shorter than the 4-byte header length")
    (length,) = struct.unpack_from(">I", data, 0)
    if length == 0:
        raise EnvelopeError("empty header")
    if length > MAX_HEADER_BYTES:
        raise EnvelopeError(f"header length {length} exceeds {MAX_HEADER_BYTES}")
    if 4 + length > len(data):
        raise EnvelopeError("header length runs past the end of the message")
    try:
        text = data[4 : 4 + length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvelopeError("header is not UTF-8") from exc
    try:
        header = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EnvelopeError("header is not JSON") from exc
    if not isinstance(header, dict):
        raise EnvelopeError("header must be a JSON object")
    return header, bytes(data[4 + length :])

import struct

import pytest

from synoptics_live.envelope import MAX_HEADER_BYTES, EnvelopeError, pack, unpack


def test_roundtrip_with_unicode_and_binary_payload():
    header = {"t": "audio", "line_id": "L1", "text": "안경"}
    payload = bytes(range(256)) * 3
    data = pack(header, payload)
    assert struct.unpack(">I", data[:4])[0] == len(data) - 4 - len(payload)
    assert unpack(data) == (header, payload)


def test_roundtrip_empty_payload():
    assert unpack(pack({"t": "frame"})) == ({"t": "frame"}, b"")


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00\x00\x00",
        struct.pack(">I", 0) + b"{}",
        struct.pack(">I", 50) + b'{"t":1}',
        struct.pack(">I", 2) + b"\xff\xfe",
        struct.pack(">I", 3) + b"{x}",
        struct.pack(">I", 2) + b"[]",
        struct.pack(">I", MAX_HEADER_BYTES + 1) + b"{" + b" " * MAX_HEADER_BYTES,
    ],
    ids=["empty", "short", "zero-len", "overrun", "not-utf8", "not-json", "not-object", "header-too-big"],
)
def test_unpack_rejects(data):
    with pytest.raises(EnvelopeError):
        unpack(data)


def test_pack_rejects_non_object_and_huge_header():
    with pytest.raises(EnvelopeError):
        pack(["x"])  # type: ignore[arg-type]
    with pytest.raises(EnvelopeError):
        pack({"x": "a" * (MAX_HEADER_BYTES + 10)})

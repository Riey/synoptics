"""Debug bug reports: metadata + streamed video segments land in ``AISW_BUGREPORT_DIR``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend.app.bugreport import MAX_SEGMENT_BYTES
from backend.tests.test_guide import ORIGIN, api_client, env  # noqa: F401  (env: fixture)


def test_disabled_without_directory(env: pytest.MonkeyPatch) -> None:  # noqa: F811
    async def exercise() -> None:
        async with api_client() as api:
            env.delenv("AISW_BUGREPORT_DIR", raising=False)
            assert (await api.get("/api/debug/bugreport/config")).json() == {"enabled": False}
            refused = await api.post("/api/debug/bugreport", json={"note": "x"}, headers=ORIGIN)
            assert refused.status_code == 503 and refused.json()["error"] == "bugreport_disabled"

    asyncio.run(exercise())


def test_report_with_segments(env: pytest.MonkeyPatch, tmp_path: Path) -> None:  # noqa: F811
    env.setenv("AISW_BUGREPORT_DIR", str(tmp_path))

    async def exercise() -> None:
        async with api_client() as api:
            assert (await api.get("/api/debug/bugreport/config")).json() == {"enabled": True}
            created = await api.post("/api/debug/bugreport", json={"note": "상자가 늦게 뜸", "goal": "안경 벗기"}, headers=ORIGIN)
            assert created.status_code == 200
            report_id = created.json()["report_id"]
            for index, body in enumerate((b"\x1aE\xdf\xa3first", b"\x1aE\xdf\xa3second")):
                put = await api.put(f"/api/debug/bugreport/{report_id}/video/{index}?ext=webm", content=body, headers=ORIGIN)
                assert put.status_code == 200 and put.json()["bytes"] == len(body)
            updated = await api.put(f"/api/debug/bugreport/{report_id}/meta", json={"note": "갱신", "segments": [1, 2]}, headers=ORIGIN)
            assert updated.status_code == 200
            done = await api.post(f"/api/debug/bugreport/{report_id}/done", headers=ORIGIN)
            assert done.json()["files"] == ["report.json", "video-0000.webm", "video-0001.webm"]

            folder = tmp_path / report_id
            meta = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            assert meta["note"] == "갱신" and meta["segments"] == [1, 2] and meta["received_at"]
            assert (folder / "video-0001.webm").read_bytes().endswith(b"second")
            assert (folder / "DONE").exists()
            assert not list(folder.glob("*.part"))

    asyncio.run(exercise())


def test_rejects_bad_ids_and_segments(env: pytest.MonkeyPatch, tmp_path: Path) -> None:  # noqa: F811
    env.setenv("AISW_BUGREPORT_DIR", str(tmp_path))

    async def exercise() -> None:
        async with api_client() as api:
            missing = await api.put("/api/debug/bugreport/..%2F..%2Fetc/video/0", content=b"x", headers=ORIGIN)
            assert missing.status_code in (404, 405)  # never reaches the handler
            unknown = await api.put("/api/debug/bugreport/20261003-120000-abcdef/video/0", content=b"x", headers=ORIGIN)
            assert unknown.status_code == 404
            newline = await api.put("/api/debug/bugreport/20261003-120000-abcdef%0A/video/0", content=b"x", headers=ORIGIN)
            assert newline.status_code == 404
            report_id = (await api.post("/api/debug/bugreport", json={}, headers=ORIGIN)).json()["report_id"]
            bad_ext = await api.put(f"/api/debug/bugreport/{report_id}/video/0?ext=sh", content=b"x", headers=ORIGIN)
            assert bad_ext.status_code == 400
            bad_index = await api.put(f"/api/debug/bugreport/{report_id}/video/10000", content=b"x", headers=ORIGIN)
            assert bad_index.status_code == 400
            not_object = await api.post("/api/debug/bugreport", content=b"[1]", headers=ORIGIN)
            assert not_object.status_code == 400

    asyncio.run(exercise())


def test_segment_size_cap(env: pytest.MonkeyPatch, tmp_path: Path) -> None:  # noqa: F811
    env.setenv("AISW_BUGREPORT_DIR", str(tmp_path))
    env.setattr("backend.app.bugreport.MAX_SEGMENT_BYTES", 8)

    async def exercise() -> None:
        async with api_client() as api:
            report_id = (await api.post("/api/debug/bugreport", json={}, headers=ORIGIN)).json()["report_id"]
            big = await api.put(f"/api/debug/bugreport/{report_id}/video/0", content=b"x" * 64, headers=ORIGIN)
            assert big.status_code == 413
            assert sorted(p.name for p in (tmp_path / report_id).iterdir()) == ["report.json"]

    assert MAX_SEGMENT_BYTES > 8
    asyncio.run(exercise())


def test_call_frames(env: pytest.MonkeyPatch, tmp_path: Path) -> None:  # noqa: F811
    env.setenv("AISW_BUGREPORT_DIR", str(tmp_path))

    async def exercise() -> None:
        async with api_client() as api:
            report_id = (await api.post("/api/debug/bugreport", json={}, headers=ORIGIN)).json()["report_id"]
            ok = await api.put(f"/api/debug/bugreport/{report_id}/file/call-0007-confirm-before_scene.jpg", content=b"\xff\xd8jpeg", headers=ORIGIN)
            assert ok.status_code == 200 and ok.json()["bytes"] == 6
            assert (tmp_path / report_id / "call-0007-confirm-before_scene.jpg").read_bytes() == b"\xff\xd8jpeg"
            for bad in ("report.json", "call-0001-confirm-x.sh", "call-1-follow-scene.jpg", "..%2Fx.jpg", "call-0001-follow-scene.jpg%0A"):
                refused = await api.put(f"/api/debug/bugreport/{report_id}/file/{bad}", content=b"x", headers=ORIGIN)
                assert refused.status_code in (400, 404, 405), bad
            assert sorted(p.name for p in (tmp_path / report_id).iterdir()) == ["call-0007-confirm-before_scene.jpg", "report.json"]

    asyncio.run(exercise())

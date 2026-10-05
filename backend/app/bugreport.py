"""Debug bug reports: the browser's session recording plus a JSON snapshot, written to a directory.

A report is created with its metadata, then each recorded video segment is streamed into it with its own
``PUT`` (raw body — no multipart dependency, no whole-file buffering) as the browser closes it, with
``PUT meta`` refreshing ``report.json`` alongside, then closed with ``done``. The
directory is ``AISW_BUGREPORT_DIR``; unset means the feature is off. Nothing here reads a report back:
the operator pulls the directory off the host (``deploy/bugreport/``).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.app.errors import ApiFailure

ENV_BUGREPORT_DIR = "AISW_BUGREPORT_DIR"
MAX_META_BYTES = 2 * 1024 * 1024
MAX_SEGMENT_BYTES = 400 * 1024 * 1024
MAX_SEGMENTS = 10000
VIDEO_EXTENSIONS = {"webm", "mp4"}
REPORT_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
FILE_NAME = re.compile(r"^call-\d{4,6}-[a-z]+-[a-z0-9_]{1,40}\.(jpg|png)$")
MAX_FILE_BYTES = 8 * 1024 * 1024


def bugreport_dir() -> Path | None:
    raw = os.getenv(ENV_BUGREPORT_DIR, "").strip()
    return Path(raw) if raw else None


def report_path(report_id: str) -> Path:
    base = bugreport_dir()
    if base is None:
        raise ApiFailure(503, "bugreport_disabled", None, f"{ENV_BUGREPORT_DIR} is not set")
    if not REPORT_ID.fullmatch(report_id):
        raise ApiFailure(404, "bugreport_not_found")
    path = base / report_id
    if not path.is_dir():
        raise ApiFailure(404, "bugreport_not_found")
    return path


async def read_limited(request: Request, max_bytes: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ApiFailure(413, "payload_too_large")
    return bytes(body)


async def read_meta(request: Request) -> dict:
    raw = await read_limited(request, MAX_META_BYTES)
    try:
        meta = json.loads(raw)
    except ValueError:
        raise ApiFailure(400, "invalid_json") from None
    if not isinstance(meta, dict):
        raise ApiFailure(400, "invalid_json")
    return meta


def write_meta(path: Path, meta: dict) -> None:
    partial = path / "report.json.part"
    partial.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path / "report.json")


def build_bugreport_router(admit: Callable[[Request], None]) -> APIRouter:
    """``admit`` runs the app's origin and access gates (no session: a report must survive an ended one)."""
    router = APIRouter()

    @router.get("/api/debug/bugreport/config")
    async def bugreport_config() -> JSONResponse:
        return JSONResponse({"enabled": bugreport_dir() is not None})

    @router.post("/api/debug/bugreport")
    async def create_report(request: Request) -> JSONResponse:
        admit(request)
        base = bugreport_dir()
        if base is None:
            raise ApiFailure(503, "bugreport_disabled", None, f"{ENV_BUGREPORT_DIR} is not set")
        meta = await read_meta(request)
        report_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        path = base / report_id
        path.mkdir(parents=True, exist_ok=False)
        meta["received_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta["client_host"] = request.headers.get("x-forwarded-for") or (request.client.host if request.client else None)
        write_meta(path, meta)
        return JSONResponse({"report_id": report_id})

    @router.put("/api/debug/bugreport/{report_id}/meta")
    async def update_meta(report_id: str, request: Request) -> JSONResponse:
        admit(request)
        path = report_path(report_id)
        meta = await read_meta(request)
        previous = json.loads((path / "report.json").read_text(encoding="utf-8"))
        for key in ("received_at", "client_host"):
            meta[key] = previous.get(key)
        write_meta(path, meta)
        return JSONResponse({"report_id": report_id})

    @router.put("/api/debug/bugreport/{report_id}/video/{index}")
    async def upload_segment(report_id: str, index: int, request: Request, ext: str = "webm") -> JSONResponse:
        admit(request)
        path = report_path(report_id)
        if not 0 <= index < MAX_SEGMENTS or ext not in VIDEO_EXTENSIONS:
            raise ApiFailure(400, "invalid_segment")
        target = path / f"video-{index:04d}.{ext}"
        partial = target.with_suffix(target.suffix + ".part")
        written = 0
        try:
            with partial.open("wb") as handle:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > MAX_SEGMENT_BYTES:
                        raise ApiFailure(413, "payload_too_large")
                    handle.write(chunk)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        return JSONResponse({"report_id": report_id, "index": index, "bytes": written})

    @router.put("/api/debug/bugreport/{report_id}/file/{name}")
    async def upload_file(report_id: str, name: str, request: Request) -> JSONResponse:
        """A frame a guide call sent (``call-0007-confirm-before_scene.jpg``), named by the browser."""
        admit(request)
        path = report_path(report_id)
        if not FILE_NAME.fullmatch(name):
            raise ApiFailure(400, "invalid_file_name")
        body = await read_limited(request, MAX_FILE_BYTES)
        partial = path / f"{name}.part"
        partial.write_bytes(body)
        partial.replace(path / name)
        return JSONResponse({"report_id": report_id, "name": name, "bytes": len(body)})

    @router.post("/api/debug/bugreport/{report_id}/done")
    async def finish_report(report_id: str, request: Request) -> JSONResponse:
        admit(request)
        path = report_path(report_id)
        files = sorted(p.name for p in path.iterdir() if p.is_file())
        (path / "DONE").write_text(json.dumps({"files": files}, ensure_ascii=False) + "\n", encoding="utf-8")
        return JSONResponse({"report_id": report_id, "files": files})

    return router

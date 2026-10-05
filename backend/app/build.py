"""Which build this server is — project version, commit, build time — for ``/api/health`` and the demo page.

Each value has a fallback chain, so every way the app is run reports something true:

* commit: ``AISW_BUILD_COMMIT`` (the tailnet container, set by ``deploy/tailscale/deploy.sh``) -> ``STAGE_INFO``
  ``ref=`` (the Mac stage, written by ``deploy/mac/stage.sh``; it has no ``.git``) -> ``git rev-parse --short HEAD``
  in the checkout -> ``"dev"``.
* built_at: ``AISW_BUILD_TIME`` -> ``STAGE_INFO`` ``built=`` -> when this process started (UTC ISO).
* version: ``[project].version`` of ``pyproject.toml`` (also present in the container image).

Read once per process (``build_info`` is cached).
"""

from __future__ import annotations

import functools
import os
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from backend.app.guide_contracts import BuildInfo

ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stage_info(root: Path) -> dict[str, str]:
    try:
        lines = (root / "STAGE_INFO").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    return dict(line.split("=", 1) for line in lines if "=" in line)


def read_version(root: Path = ROOT) -> str:
    try:
        return str(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def read_commit(root: Path = ROOT) -> str:
    env = os.getenv("AISW_BUILD_COMMIT", "").strip()
    if env:
        return env
    staged = _stage_info(root).get("ref", "").strip()
    if staged:
        return staged
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=2, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = ""
    return head or "dev"


def read_built_at(root: Path = ROOT) -> str:
    env = os.getenv("AISW_BUILD_TIME", "").strip()
    if env:
        return env
    return _stage_info(root).get("built", "").strip() or PROCESS_STARTED


@functools.cache
def build_info() -> BuildInfo:
    return BuildInfo(version=read_version(), commit=read_commit(), built_at=read_built_at())

"""The build identity on ``/api/health`` (shown on the demo page): version, commit, build time."""

from __future__ import annotations

import asyncio
import subprocess
import tomllib
from pathlib import Path

import pytest

from backend.app import build
from backend.tests.test_guide import api_client, env, install  # noqa: F401  (env: fixture)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def fresh_build(monkeypatch: pytest.MonkeyPatch):
    for name in ("AISW_BUILD_COMMIT", "AISW_BUILD_TIME"):
        monkeypatch.delenv(name, raising=False)
    build.build_info.cache_clear()
    yield monkeypatch
    build.build_info.cache_clear()


def test_env_names_the_commit_and_the_build_time(fresh_build: pytest.MonkeyPatch) -> None:
    fresh_build.setenv("AISW_BUILD_COMMIT", "d2376b6")
    fresh_build.setenv("AISW_BUILD_TIME", "2026-10-02T01:30:00Z")
    info = build.build_info()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert (info.version, info.commit, info.built_at) == (version, "d2376b6", "2026-10-02T01:30:00Z")


def test_without_env_the_checkout_and_the_process_start_answer() -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info = build.build_info()
    assert info.commit == (head or "dev")
    assert info.built_at == build.PROCESS_STARTED


def test_a_staged_tree_without_git_reads_stage_info(tmp_path: Path) -> None:
    (tmp_path / "STAGE_INFO").write_text("ref=abc1234\nbuilt=2026-10-02T10:30:00+09:00\n", encoding="utf-8")
    assert build.read_commit(tmp_path) == "abc1234"
    assert build.read_built_at(tmp_path) == "2026-10-02T10:30:00+09:00"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build.read_commit(empty) == "dev"
    assert build.read_built_at(empty) == build.PROCESS_STARTED


def test_health_reports_the_build(env, fresh_build: pytest.MonkeyPatch) -> None:  # noqa: F811
    fresh_build.setenv("AISW_BUILD_COMMIT", "d2376b6")
    fresh_build.setenv("AISW_BUILD_TIME", "2026-10-02T01:30:00Z")

    async def exercise() -> None:
        install()
        async with api_client() as api:
            body = (await api.get("/api/health")).json()
            assert body["build"]["commit"] == "d2376b6"
            assert body["build"]["built_at"] == "2026-10-02T01:30:00Z"
            assert body["build"]["version"]

    asyncio.run(exercise())

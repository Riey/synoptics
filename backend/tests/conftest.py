"""Never let backend tests read the developer's real OpenAI API key."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_astra_auth(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path_factory.mktemp("openai-key") / "absent.key"))

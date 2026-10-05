from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    # Default data paths are relative to the repository root.
    monkeypatch.chdir(ROOT)
    # Never pick up a real license, webhook or data path from the developer's shell.
    for name in ("WLSACCESSID", "WLSSECRET", "LICENSEID", "DISCORD_URL", "EMPLOYEE_PATH", "TASK_PATH"):
        monkeypatch.delenv(name, raising=False)

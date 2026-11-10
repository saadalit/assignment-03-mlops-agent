import subprocess
from pathlib import Path

import pytest

from agent.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(repo=tmp_path, timeout=0.2)


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for args in (["init", "-b", "main"], ["config", "user.name", "Test"],
                 ["config", "user.email", "test@example.invalid"]):
        subprocess.run(["git", *args], check=True, capture_output=True)
    Path("data").mkdir()
    return tmp_path

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from agent.config import Settings
from agent.tools.dvc_tools import DVCTools


def commit_dataset(content):
    Path("data/iris.csv").write_text(content)
    md5 = hashlib.md5(content.encode()).hexdigest()
    Path("data/iris.csv.dvc").write_text(
        yaml.safe_dump({"outs": [{"md5": md5, "path": "iris.csv"}]})
    )
    subprocess.run(["git", "add", "data/iris.csv.dvc"], check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "dataset"], check=True, capture_output=True)
    return md5


def test_provenance_is_read_from_git_not_guessed(git_repo):
    md5 = commit_dataset("a,b\n1,2\n")
    adapter = DVCTools(Settings(repo=git_repo))
    version = adapter.current_version()
    assert version["outputs"][0]["md5"] == md5 and len(version["git_commit"]) == 40
    assert version["matches_recorded_version"]
    assert len(adapter.list_datasets()["datasets"]) == 1


def test_detects_workspace_data_changes(git_repo):
    commit_dataset("a,b\n1,2\n")
    Path("data/iris.csv").write_text("changed")
    assert not DVCTools(Settings(repo=git_repo)).current_version()["matches_recorded_version"]


@pytest.mark.parametrize(
    "ref,path",
    [("--help", "data/iris.csv.dvc"), ("HEAD", "../outside.dvc"), ("HEAD", "/outside.dvc")],
)
def test_rejects_revision_and_path_injection(git_repo, ref, path):
    commit_dataset("hello")
    with pytest.raises(ValueError):
        DVCTools(Settings(repo=git_repo)).current_version(ref, path)


def test_reproduction_instructions_use_immutable_commit(git_repo):
    commit_dataset("hello")
    result = DVCTools(Settings(repo=git_repo)).reproduction()
    assert result["git_commit"] in result["instructions"][0]
    assert "worktree" in result["instructions"][0]
    assert "no checkout" in result["note"]

import subprocess
from pathlib import Path

from scripts.audit_repository import audit


def test_audit_rejects_tracked_runtime_and_secret_without_disclosing_it(git_repo):
    credential = "ghp_" + "a" * 36
    Path(".env").write_text("AGENT_API_TOKEN=" + credential)
    subprocess.run(["git", "add", ".env"], check=True, capture_output=True)
    result = audit()
    assert {v["reason"] for v in result["violations"]} == {
        "runtime or credential path",
        "credential content detected",
    }
    assert credential not in str(result)


def test_history_scan_detects_secret_removed_from_working_tree(git_repo):
    Path("config.txt").write_text("ghp_" + "a" * 36)
    subprocess.run(["git", "add", "config.txt"], check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fixture"], check=True, capture_output=True)
    Path("config.txt").write_text("removed")
    assert not audit()["violations"]
    result = audit(history=True)
    assert result["history_blobs_checked"] == 1
    assert result["violations"][0]["file_or_object"].startswith("git:")

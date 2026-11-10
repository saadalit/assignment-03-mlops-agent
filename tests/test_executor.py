import json
import logging

import httpx
import pytest

from agent.executor import Executor, ToolSpec
from agent.schemas import Empty


def test_execution_records_safe_structured_log(caplog):
    ex = Executor([ToolSpec("dvc.status", "status", Empty, lambda: {"ok": True})])
    with caplog.at_level(logging.INFO, logger="mlops.tools"):
        result = ex.execute("dvc.status", {}, "request-123")
    assert result.status == "ok" and result.data == {"ok": True}
    record = json.loads(caplog.records[-1].message)
    assert record == {
        "tool": "dvc.status",
        "status": "ok",
        "duration_ms": result.duration_ms,
        "request_id": "request-123",
    }


@pytest.mark.parametrize(
    "error,status",
    [(httpx.ReadTimeout("secret"), "timeout"), (ConnectionError("token=abc"), "error")],
)
def test_timeout_and_failure_never_invent_data(error, status):
    def fail():
        raise error

    ex = Executor([ToolSpec("test.status", "status", Empty, fail)])
    result = ex.execute("test.status", {}, "id")
    assert result.status == status and result.data == {}
    assert "secret" not in result.model_dump_json() and "abc" not in result.model_dump_json()


@pytest.mark.parametrize("name,args", [("shell.exec", {}), ("dvc.status", {"command": "rm -rf /"})])
def test_invalid_requests_rejected(name, args):
    ex = Executor([ToolSpec("dvc.status", "status", Empty, lambda: {"ok": True})])
    assert ex.execute(name, args, "id").status == "invalid"


def test_redacts_secrets_in_nested_evidence(monkeypatch, caplog):
    monkeypatch.setenv("AIRFLOW_PASSWORD", "superprivate123")
    ex = Executor(
        [
            ToolSpec(
                "test.logs",
                "logs",
                Empty,
                lambda: {
                    "logs": "password=abc Bearer xyz superprivate123",
                    "nested": {"access_token": "secret"},
                },
            )
        ]
    )
    with caplog.at_level(logging.INFO):
        result = ex.execute("test.logs", {}, "id")
    text = result.model_dump_json() + caplog.text
    assert result.status == "ok"
    assert all(secret not in text for secret in ("abc", "xyz", "superprivate123", '"secret"'))

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


def test_malformed_service_payload_is_an_error_without_claims():
    ex = Executor([ToolSpec("test.status", "status", Empty, lambda: ["unexpected"])])
    result = ex.execute("test.status", {}, "id")
    assert result.status == "error" and result.data == {}


def test_json_credentials_and_jwt_in_log_strings_are_redacted():
    from agent.executor import redact

    jwt = "eyJ" + "a" * 20 + "." + "b" * 25 + "." + "c" * 25
    text = redact('{"password": "private-value", "access_token": "private-token"} ' + jwt)
    assert all(v not in text for v in ("private-value", "private-token", jwt))


def test_kubernetes_sdk_timeout_is_classified(settings):
    from urllib3.exceptions import ReadTimeoutError

    def fail():
        raise ReadTimeoutError(None, "private-url", "private response")

    result = Executor([ToolSpec("kubernetes.events", "events", Empty, fail)]).execute(
        "kubernetes.events",
        {},
        "request-id",
    )
    assert result.status == "timeout" and not result.data
    assert "private" not in result.model_dump_json()


def test_invalid_arguments_return_actionable_schema_feedback_without_echoing_inputs():
    from unittest.mock import Mock

    from agent.schemas import DatasetRef

    handler = Mock()
    result = Executor([ToolSpec("dvc.current_version", "version", DatasetRef, handler)]).execute(
        "dvc.current_version",
        {"ref": "private-secret; shell", "private-field": "private-value"},
        "id",
    )
    assert result.status == "invalid" and not result.data
    assert "ref: string_pattern_mismatch" in result.error
    assert "Allowed fields: path, ref" in result.error
    assert "private" not in result.model_dump_json()
    handler.assert_not_called()

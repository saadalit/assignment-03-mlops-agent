import httpx
import pytest

from agent.executor import Executor
from agent.tools.airflow_tools import AirflowTools


def transport(monkeypatch, handler):
    monkeypatch.setenv("AIRFLOW_USERNAME", "agent")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "private-password")
    monkeypatch.delenv("AIRFLOW_TOKEN", raising=False)

    def wrapped(request):
        if request.url.path == "/auth/token":
            return httpx.Response(200, json={"access_token": "private-jwt"})
        assert request.headers["authorization"] == "Bearer private-jwt"
        assert request.url.path.startswith("/api/v2/")
        return handler(request)

    return httpx.Client(base_url="http://airflow", transport=httpx.MockTransport(wrapped))


def test_airflow_v3_auth_and_task_failure_diagnostics(settings, monkeypatch):
    client = transport(
        monkeypatch,
        lambda r: httpx.Response(
            200,
            json={
                "task_instances": [
                    {"task_id": "validate", "state": "failed"},
                    {"task_id": "train", "state": "upstream_failed"},
                    {"task_id": "evaluate_mlflow", "state": "success"},
                ]
            },
        ),
    )
    result = AirflowTools(settings, client).failed_tasks("training_pipeline", "manual/run")
    assert len(result["failed_tasks"]) == 2


def test_airflow_trigger_requires_confirmation_and_allowlist(settings, monkeypatch):
    client = transport(monkeypatch, lambda r: httpx.Response(200, json={"dag_run_id": "manual"}))
    ex = Executor(AirflowTools(settings, client).specs())
    assert ex.execute("airflow.trigger_training", {}, "id").status == "denied"
    result = ex.execute(
        "airflow.trigger_training", {}, "id", {"airflow.trigger_training:training_pipeline"}
    )
    assert result.status == "ok"
    result = ex.execute(
        "airflow.trigger_training", {"dag_id": "evil"}, "id", {"airflow.trigger_training:evil"}
    )
    assert result.status == "denied"


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_airflow_http_errors_propagate_without_fabrication(settings, monkeypatch, status):
    client = transport(monkeypatch, lambda r: httpx.Response(status, json={"detail": "secret"}))
    result = Executor(AirflowTools(settings, client).specs()).execute("airflow.list_dags", {}, "id")
    assert result.status == "error" and result.data == {} and "secret" not in result.error

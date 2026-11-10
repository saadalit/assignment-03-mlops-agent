import time

import pytest

from agent.executor import Executor, ToolSpec
from agent.orchestrator import Orchestrator
from agent.schemas import AgentRequest, Empty


def mocked_executor(monkeypatch, settings, failing=None, tags=None):
    from agent.orchestrator import build_executor

    ex = build_executor(settings)
    candidate = {
        "run_id": "a" * 32,
        "status": "FINISHED",
        "metrics": {"f1": 0.95},
        "start_time": int(time.time() * 1000),
        "tags": {
            "dataset_git_commit": "b" * 40,
            "dataset_md5": "abc",
            "quality_gate": "passed",
            **(tags or {}),
        },
    }
    values = {
        "mlflow.best_run": {
            "candidate": candidate,
            "metric": "f1",
            "selection_scope": "last 5 runs",
            "caution": "Ranking does not establish production readiness.",
        },
        "dvc.current_version": {
            "pointer": "data/iris.csv.dvc",
            "git_commit": "b" * 40,
            "outputs": [{"md5": "abc"}],
        },
        "kubernetes.deployment_status": {
            "name": "prediction-api",
            "namespace": "mlops",
            "ready": 2,
            "desired": 2,
            "rollout_complete": True,
            "images": ["api:v1"],
            "version": {"MODEL_RUN_ID": "a" * 32},
        },
        "kubernetes.list_workloads": {"pods": []},
        "kubernetes.services": {"endpoints": [{"ready_addresses": ["10.0.0.1"]}]},
        "kubernetes.events": {"events": []},
        "airflow.list_runs": {"dag_runs": [{"dag_run_id": "run-1", "state": "success"}]},
        "airflow.failed_tasks": {"failed_tasks": []},
    }
    for name, spec in ex.tools.items():

        def handler(_name=name, **kwargs):
            if _name == failing:
                raise ConnectionError("service unavailable")
            return values.get(_name, {})

        spec.handler = handler
    return ex


def test_mandatory_scenario_uses_all_four_integrations_without_mutation(settings, monkeypatch):
    ex = mocked_executor(monkeypatch, settings)
    request = AgentRequest(
        message="Find the best recent model, identify its dataset version, "
        "check deployment status, and explain whether retraining is necessary."
    )
    result = Orchestrator(settings, ex).run(request)
    names = {r.tool for r in result.evidence}
    assert {
        "mlflow.best_run",
        "dvc.current_version",
        "kubernetes.deployment_status",
        "airflow.list_runs",
    } <= names
    assert not any("trigger" in n or "restart" in n for n in names)
    assert any("Deployed run matches" in f for f in result.facts)
    assert any("No supplied degradation" in u for u in result.uncertainties)


def test_unavailable_tool_produces_uncertainty_never_operational_claims(settings, monkeypatch):
    ex = mocked_executor(monkeypatch, settings, failing="mlflow.best_run")
    result = Orchestrator(settings, ex).run(AgentRequest(message="Find the best model"))
    assert not result.facts
    assert any("mlflow.best_run" in u for u in result.uncertainties)
    assert not any("Candidate" in f for f in result.facts)


def test_missing_provenance_not_substituted_with_head(settings, monkeypatch):
    ex = mocked_executor(monkeypatch, settings, tags={"dataset_git_commit": ""})
    result = Orchestrator(settings, ex).run(AgentRequest(message="Find the best model"))
    assert "dvc.current_version" not in [r.tool for r in result.evidence]
    assert any("lacks complete" in u for u in result.uncertainties)


def test_unavailable_run_does_not_substitute_current_dataset(settings, monkeypatch):
    ex = mocked_executor(monkeypatch, settings, failing="mlflow.get_run")
    result = Orchestrator(settings, ex).run(
        AgentRequest(message="Which dataset version was used for experiment run " + "a" * 32)
    )
    assert [r.tool for r in result.evidence] == ["mlflow.get_run"]
    assert not result.facts
    assert any("mlflow.get_run" in u for u in result.uncertainties)


def test_freshness_policy_is_explicit(settings, monkeypatch):
    settings.max_model_age_days = -1
    result = Orchestrator(settings, mocked_executor(monkeypatch, settings)).run(
        AgentRequest(message="Find best model")
    )
    assert any("freshness policy" in r for r in result.recommendations)


def test_incomplete_rollout_does_not_claim_serving_model_identity(settings, monkeypatch):
    ex = mocked_executor(monkeypatch, settings)
    handler = ex.tools["kubernetes.deployment_status"].handler
    ex.tools["kubernetes.deployment_status"].handler = lambda **kw: {
        **handler(**kw),
        "rollout_complete": False,
    }
    result = Orchestrator(settings, ex).run(
        AgentRequest(message="Find best model and check deployment")
    )
    assert not any("Deployed run matches" in f for f in result.facts)
    assert any("Serving model identity is unverified" in u for u in result.uncertainties)


def test_dataset_question_selects_dvc_without_mlflow(settings):
    ex = Executor(
        [
            ToolSpec("dvc.list_datasets", "list", Empty, lambda: {"datasets": []}),
            ToolSpec("dvc.status", "status", Empty, lambda: {"changes": {}}),
        ]
    )
    result = Orchestrator(settings, ex).run(
        AgentRequest(message="Which dataset version is current?")
    )
    assert [r.tool for r in result.evidence] == ["dvc.list_datasets", "dvc.status"]


def test_failure_diagnosis_prioritizes_root_failure_over_upstream_failed(settings):
    from agent.schemas import DagQuery, DagRun, TaskLog

    ex = Executor(
        [
            ToolSpec(
                "airflow.list_runs",
                "runs",
                DagQuery,
                lambda **kw: {"dag_runs": [{"dag_run_id": "r1", "state": "failed"}]},
            ),
            ToolSpec(
                "airflow.failed_tasks",
                "tasks",
                DagRun,
                lambda **kw: {
                    "failed_tasks": [
                        {"task_id": "preprocess", "state": "upstream_failed"},
                        {"task_id": "train", "state": "upstream_failed"},
                        {"task_id": "validate", "state": "failed", "try_number": 2},
                    ]
                },
            ),
            ToolSpec(
                "airflow.task_logs",
                "logs",
                TaskLog,
                lambda **kw: {"task_id": kw["task_id"], "log": "DATA_VALIDATION_FAILED"},
            ),
        ]
    )
    result = Orchestrator(settings, ex).run(
        AgentRequest(message="Diagnose failed training pipeline")
    )
    assert result.evidence[-1].tool == "airflow.task_logs"
    assert result.evidence[-1].data["task_id"] == "validate"
    assert any("Validation rejected" in r for r in result.recommendations)


def test_llm_cannot_self_authorize_restart(settings, monkeypatch):
    from agent.orchestrator import build_executor

    settings.planner = "ollama"
    ex = build_executor(settings)
    monkeypatch.setattr(
        Orchestrator,
        "_llm_plan",
        lambda self, msg, call: call(
            "kubernetes.restart_deployment", namespace="mlops", name="prediction-api"
        ),
    )
    result = Orchestrator(settings, ex).run(AgentRequest(message="Inspect the app"))
    assert result.evidence[0].status == "denied"


def test_llm_failure_does_not_invent_results(settings, monkeypatch):
    settings.planner = "ollama"

    def fail(self, message, call):
        raise ConnectionError()

    monkeypatch.setattr(Orchestrator, "_llm_plan", fail)
    result = Orchestrator(settings, Executor([])).run(AgentRequest(message="best model"))
    assert not result.facts
    assert any("planner unavailable" in u for u in result.uncertainties)


@pytest.mark.parametrize("message", ["", "x" * 4001])
def test_invalid_agent_request(message):
    with pytest.raises(ValueError):
        AgentRequest(message=message)

import httpx
import pytest

from agent.executor import Executor, ToolSpec
from agent.orchestrator import Orchestrator
from agent.schemas import AgentRequest, Empty


def mock_llm(monkeypatch, handler):
    client_class = httpx.Client
    monkeypatch.setattr(
        "agent.orchestrator.httpx.Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs),
    )


def reply(*calls, content=""):
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls],
            },
        },
    )


def test_supported_llm_protocol_executes_selected_tool_and_returns_actual_evidence(
    settings, monkeypatch
):
    settings.planner = "ollama"
    requests = []

    def handler(request):
        import json

        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "dvc__status", "arguments": {}}}],
                    }
                },
            )
        assert body["messages"][-1]["role"] == "tool"
        assert body["messages"][-1]["tool_name"] == "dvc__status"
        tool_result = json.loads(body["messages"][-1]["content"])
        assert tool_result["status"] == "ok" and tool_result["data"] == {"changes": {}}
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Fabricated model accuracy is 100%."}
            },
        )

    client_class = httpx.Client
    monkeypatch.setattr(
        "agent.orchestrator.httpx.Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs),
    )
    ex = Executor(
        [ToolSpec("dvc.status", "Read real data changes", Empty, lambda: {"changes": {}})]
    )
    result = Orchestrator(settings, ex).run(AgentRequest(message="Has my tracked dataset changed?"))
    assert [r.tool for r in result.evidence] == ["dvc.status"]
    assert result.evidence[0].data == {"changes": {}}
    assert "100%" not in result.model_dump_json()
    assert len(requests) == 2


def test_llm_multitool_uses_returned_candidate_commit_and_feedback(settings, monkeypatch):
    import json

    from test_orchestrator import mocked_executor

    settings.planner = "ollama"
    executor = mocked_executor(monkeypatch, settings)
    rounds = 0

    def handler(request):
        nonlocal rounds
        rounds += 1
        body = json.loads(request.content)
        if rounds == 1:
            return reply(("mlflow__best_run", {}))
        if rounds == 2:
            result = json.loads(
                next(m["content"] for m in reversed(body["messages"]) if m["role"] == "tool")
            )
            commit = result["data"]["candidate"]["tags"]["dataset_git_commit"]
            assert "mlflow__best_run" not in [d["function"]["name"] for d in body["tools"]]
            return reply(("dvc__current_version", {"ref": commit}))
        if rounds == 3:
            # A premature final answer must prompt the model for missing evidence.
            return reply(content="Everything is healthy. Accuracy is 100%.")
        if rounds >= 4:
            assert any(
                "Operational evidence is incomplete" in m.get("content", "")
                for m in body["messages"]
            )
            assert body["format"]["type"] == "object" and "tools" not in body
            assert r"\\w" not in json.dumps(body["format"])
            assert not any("Accuracy is 100%" in m.get("content", "") for m in body["messages"])
            name = (
                "kubernetes.deployment_status",
                "kubernetes.list_workloads",
                "kubernetes.services",
                "kubernetes.events",
                "airflow.list_runs",
            )[rounds - 4].replace(".", "__")
            assert name in [item["properties"]["name"]["const"] for item in body["format"]["oneOf"]]
            return reply(content=json.dumps({"name": name, "arguments": {}}))
        return reply()

    mock_llm(monkeypatch, handler)
    result = Orchestrator(settings, executor).run(
        AgentRequest(message="Find best recent model, dataset, deployment and whether to retrain")
    )
    assert len(result.evidence) == 7
    assert all(r.status == "ok" for r in result.evidence)
    assert len(result.planner_trace) == 8
    assert all(t["protocol"] == "json_schema" for t in result.planner_trace[3:])
    assert "100%" not in result.model_dump_json()
    assert not any("trigger" in r.tool or "restart" in r.tool for r in result.evidence)
    assert any("Deployed run matches" in f for f in result.facts)


def test_llm_duplicate_mutation_is_not_reexecuted(settings, monkeypatch):
    from unittest.mock import Mock

    from agent.schemas import TriggerDag

    settings.planner = "ollama"
    mutation = Mock(return_value={"dag_run_id": "actual-run", "state": "queued"})
    executor = Executor(
        [
            ToolSpec(
                "airflow.trigger_training",
                "trigger",
                TriggerDag,
                mutation,
                lambda a: "airflow.trigger_training:" + a["dag_id"],
            )
        ]
    )
    rounds = 0

    def handler(request):
        nonlocal rounds
        rounds += 1
        # Explicit defaults and omitted defaults identify the same operation.
        args = {} if rounds == 1 else {"dag_id": "training_pipeline"}
        return reply(("airflow__trigger_training", args))

    mock_llm(monkeypatch, handler)
    result = Orchestrator(settings, executor).run(
        AgentRequest(
            message="Trigger training",
            confirmations=["airflow.trigger_training:training_pipeline"],
        )
    )
    mutation.assert_called_once()
    assert len(result.evidence) == 1
    assert any("budget" in u for u in result.uncertainties)


def test_llm_can_repair_invalid_provenance_call_with_real_candidate_reference(
    settings, monkeypatch
):
    import json

    from test_orchestrator import mocked_executor

    settings.planner = "ollama"
    executor = mocked_executor(monkeypatch, settings)
    rounds = 0

    def handler(request):
        nonlocal rounds
        rounds += 1
        body = json.loads(request.content)
        if rounds == 1:
            return reply(("mlflow__best_run", {}), ("dvc__current_version", {"commit": "invented"}))
        assert "dvc__current_version" in [
            item["properties"]["name"]["const"] for item in body["format"]["oneOf"]
        ]
        pointer_schema = next(
            item["properties"]["arguments"]
            for item in body["format"]["oneOf"]
            if item["properties"]["name"]["const"] == "dvc__current_version"
        )
        assert pointer_schema["required"] == ["ref", "path"]
        invalid = json.loads(
            next(m["content"] for m in reversed(body["messages"]) if m["role"] == "tool")
        )
        assert invalid["status"] == "invalid"
        assert "Allowed fields: path, ref" in invalid["error"]
        candidate = next(
            json.loads(m["content"])["data"]["candidate"]
            for m in body["messages"]
            if m.get("tool_name") == "mlflow__best_run"
        )
        assert (
            pointer_schema["properties"]["ref"]["const"] == candidate["tags"]["dataset_git_commit"]
        )
        return reply(
            ("dvc__current_version", {"ref": candidate["tags"]["dataset_git_commit"]}),
            *[
                (n.replace(".", "__"), {})
                for n in (
                    "kubernetes.deployment_status",
                    "kubernetes.list_workloads",
                    "kubernetes.services",
                    "kubernetes.events",
                    "airflow.list_runs",
                )
            ],
        )

    mock_llm(monkeypatch, handler)
    result = Orchestrator(settings, executor).run(
        AgentRequest(
            message="Find the best recent model, dataset, deployment and whether to retrain"
        )
    )
    assert rounds == 2
    assert [r.status for r in result.evidence[:3]] == ["ok", "invalid", "ok"]
    assert len(result.evidence) == 8
    assert not any("planner unavailable" in u for u in result.uncertainties)


@pytest.mark.parametrize("mode", ["timeout", "http-error", "malformed-json"])
def test_llm_transport_failures_never_fall_back_to_rules(settings, monkeypatch, mode):
    from unittest.mock import Mock

    settings.planner = "ollama"
    tool = Mock(return_value={"changes": {}})

    def handler(request):
        if mode == "timeout":
            raise httpx.ReadTimeout("secret-service-response", request=request)
        if mode == "http-error":
            return httpx.Response(503, text="secret-service-response")
        return httpx.Response(200, text="invalid JSON secret-service-response")

    mock_llm(monkeypatch, handler)
    result = Orchestrator(settings, Executor([ToolSpec("dvc.status", "status", Empty, tool)])).run(
        AgentRequest(message="dataset status")
    )
    tool.assert_not_called()
    assert not result.facts and not result.evidence
    assert "secret-service-response" not in result.model_dump_json()
    assert any("planner unavailable" in u for u in result.uncertainties)


def test_llm_forged_confirmation_and_unknown_tools_rejected(settings, monkeypatch):
    from unittest.mock import Mock

    from agent.schemas import Deployment

    settings.planner = "ollama"
    mutation = Mock(return_value={"accepted": True})
    ex = Executor(
        [
            ToolSpec(
                "kubernetes.restart_deployment",
                "restart",
                Deployment,
                mutation,
                lambda a: f"kubernetes.restart_deployment:{a['namespace']}:{a['name']}",
            )
        ]
    )
    responses = iter(
        [
            reply(("shell__exec", {"command": "echo unsafe"})),
            reply(("kubernetes__restart_deployment", {"confirmations": ["approved"]})),
            reply(("kubernetes__restart_deployment", {})),
            reply(),
        ]
    )
    mock_llm(monkeypatch, lambda r: next(responses))
    result = Orchestrator(settings, ex).run(AgentRequest(message="Inspect deployment"))
    assert [r.status for r in result.evidence] == ["invalid", "invalid", "denied"]
    mutation.assert_not_called()


def test_llm_tool_failure_is_returned_to_model_and_cannot_be_overridden(settings, monkeypatch):
    import json

    settings.planner = "ollama"
    rounds = 0

    def unavailable():
        raise ConnectionError("private service diagnostics")

    def handler(request):
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            return reply(("dvc__status", {}))
        data = json.loads(json.loads(request.content)["messages"][-1]["content"])
        assert data["status"] == "error" and data["data"] == {}
        return reply(content="The dataset is current and all services are healthy.")

    mock_llm(monkeypatch, handler)
    result = Orchestrator(
        settings,
        Executor(
            [
                ToolSpec("dvc.status", "status", Empty, unavailable),
            ]
        ),
    ).run(AgentRequest(message="dataset status"))
    assert not result.facts
    assert any("dvc.status" in u for u in result.uncertainties)
    assert "all services are healthy" not in result.model_dump_json()


def test_large_tool_logs_are_bounded_for_model_but_full_evidence_is_preserved(
    settings, monkeypatch
):
    import json

    settings.planner = "ollama"
    log = "diagnostic line\n" * 1000
    rounds = 0

    def handler(request):
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            return reply(("service__logs", {}))
        evidence = json.loads(json.loads(request.content)["messages"][-1]["content"])
        assert len(evidence["data"]["logs"]) < 2100
        assert "full value in exported evidence" in evidence["data"]["logs"]
        return reply()

    mock_llm(monkeypatch, handler)
    result = Orchestrator(
        settings,
        Executor(
            [
                ToolSpec("service.logs", "logs", Empty, lambda: {"logs": log}),
            ]
        ),
    ).run(AgentRequest(message="Show logs"))
    assert result.evidence[0].data["logs"] == log


def test_compact_llm_schemas_keep_constraints_and_invalid_inputs_cannot_run(settings, monkeypatch):
    import json
    from unittest.mock import Mock

    from agent.schemas import PodLog

    settings.planner = "ollama"
    logs = Mock()
    rounds = 0

    def handler(request):
        nonlocal rounds
        rounds += 1
        body = json.loads(request.content)
        schema = body["tools"][0]["function"]["parameters"]
        assert "title" not in schema and not schema["additionalProperties"]
        assert schema["properties"]["tail_lines"]["maximum"] == 200
        if rounds == 1:
            return reply(("kubernetes__pod_logs", {"pod": "prediction-api-1", "tail_lines": 201}))
        return reply()

    mock_llm(monkeypatch, handler)
    result = Orchestrator(
        settings,
        Executor(
            [
                ToolSpec("kubernetes.pod_logs", "logs", PodLog, logs),
            ]
        ),
    ).run(AgentRequest(message="Get Pod logs"))
    logs.assert_not_called()
    assert result.evidence[0].status == "invalid"

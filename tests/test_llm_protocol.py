import httpx

from agent.executor import Executor, ToolSpec
from agent.orchestrator import Orchestrator
from agent.schemas import AgentRequest, Empty


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

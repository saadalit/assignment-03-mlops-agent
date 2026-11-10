from types import SimpleNamespace as NS
from unittest.mock import Mock

from agent.executor import Executor
from agent.tools.kubernetes_tools import KubernetesTools


def test_unauthorized_cluster_modification_is_rejected(settings):
    core, apps = Mock(), Mock()
    ex = Executor(KubernetesTools(settings, core, apps).specs())
    result = ex.execute("kubernetes.restart_deployment", {}, "id")
    assert result.status == "denied"
    apps.patch_namespaced_deployment.assert_not_called()


def test_confirmed_mutation_cannot_escape_namespace(settings):
    core, apps = Mock(), Mock()
    ex = Executor(KubernetesTools(settings, core, apps).specs())
    result = ex.execute(
        "kubernetes.restart_deployment",
        {"namespace": "kube-system"},
        "id",
        {"kubernetes.restart_deployment:kube-system:prediction-api"},
    )
    assert result.status == "denied"
    apps.patch_namespaced_deployment.assert_not_called()


def test_read_access_cannot_escape_namespace(settings):
    core, apps = Mock(), Mock()
    ex = Executor(KubernetesTools(settings, core, apps).specs())
    assert (
        ex.execute("kubernetes.list_workloads", {"namespace": "default"}, "id").status == "denied"
    )
    core.list_namespaced_pod.assert_not_called()


def test_logs_cannot_read_unrelated_app(settings):
    core, apps = Mock(), Mock()
    core.read_namespaced_pod.return_value = NS(metadata=NS(labels={"app": "private"}))
    ex = Executor(KubernetesTools(settings, core, apps).specs())
    result = ex.execute("kubernetes.pod_logs", {"pod": "private"}, "id")
    assert result.status == "denied"
    core.read_namespaced_pod_log.assert_not_called()


def test_pod_health_includes_restarts_and_waiting(settings):
    waiting = Mock()
    waiting.to_dict.return_value = {"reason": "CrashLoopBackOff"}
    pod = NS(
        metadata=NS(name="prediction-api-1"),
        status=NS(
            phase="Running",
            conditions=[],
            container_statuses=[
                NS(
                    name="api",
                    ready=False,
                    restart_count=5,
                    image="image:v1",
                    image_id="sha256:123",
                    state=NS(waiting=waiting),
                    last_state=NS(terminated=None),
                )
            ],
        ),
    )
    core, apps = Mock(), Mock()
    core.list_namespaced_pod.return_value = NS(items=[pod])
    apps.list_namespaced_deployment.return_value = NS(items=[])
    result = KubernetesTools(settings, core, apps).list_workloads()["pods"][0]
    assert not result["healthy"] and result["containers"][0]["restarts"] == 5
    assert result["containers"][0]["waiting"]["reason"] == "CrashLoopBackOff"

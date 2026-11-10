import os
from datetime import UTC, datetime

from agent.config import Settings
from agent.executor import ToolSpec
from agent.permissions import require_namespace
from agent.schemas import Deployment, Namespace, PodLog
from kubernetes import client, config


class KubernetesTools:
    def __init__(self, settings: Settings, core=None, apps=None):
        self.settings, self.core, self.apps = settings, core, apps

    def _clients(self, namespace):
        require_namespace(namespace, self.settings.namespace)
        if self.core is None:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config(config_file=os.getenv("KUBECONFIG"))
            cfg = client.Configuration.get_default_copy()
            cfg.retries = 0
            api_client = client.ApiClient(cfg)
            self.core, self.apps = client.CoreV1Api(api_client), client.AppsV1Api(api_client)
        return self.core, self.apps

    def _kwargs(self):
        return {"_request_timeout": self.settings.timeout, "label_selector": self.settings.selector}

    def list_workloads(self, namespace="mlops"):
        core, apps = self._clients(namespace)
        pods = core.list_namespaced_pod(namespace, **self._kwargs()).items
        deps = apps.list_namespaced_deployment(namespace, **self._kwargs()).items
        return {
            "namespace": namespace,
            "pods": [self._pod(p) for p in pods],
            "deployments": [
                {
                    "name": d.metadata.name,
                    "desired": d.spec.replicas,
                    "ready": d.status.ready_replicas or 0,
                    "images": [c.image for c in d.spec.template.spec.containers],
                }
                for d in deps
            ],
        }

    def _pod(self, pod):
        containers = pod.status.container_statuses or []
        return {
            "name": pod.metadata.name,
            "phase": pod.status.phase,
            "conditions": [
                {"type": c.type, "status": c.status, "reason": c.reason}
                for c in pod.status.conditions or []
            ],
            "containers": [
                {
                    "name": c.name,
                    "ready": c.ready,
                    "restarts": c.restart_count,
                    "image": c.image,
                    "image_id": c.image_id,
                    "waiting": c.state.waiting.to_dict() if c.state.waiting else None,
                    "last_termination": c.last_state.terminated.to_dict()
                    if c.last_state and c.last_state.terminated
                    else None,
                }
                for c in containers
            ],
            "healthy": bool(containers)
            and all(c.ready for c in containers)
            and pod.status.phase == "Running",
        }

    def services(self, namespace="mlops"):
        core, _ = self._clients(namespace)
        services = core.list_namespaced_service(namespace, **self._kwargs()).items
        endpoints = core.list_namespaced_endpoints(namespace, **self._kwargs()).items
        return {
            "services": [
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "selector": s.spec.selector,
                    "ports": [{"port": p.port, "target_port": p.target_port} for p in s.spec.ports],
                }
                for s in services
            ],
            "endpoints": [
                {
                    "name": e.metadata.name,
                    "ready_addresses": [
                        a.ip for sub in e.subsets or [] for a in sub.addresses or []
                    ],
                }
                for e in endpoints
            ],
        }

    def events(self, namespace="mlops"):
        core, _ = self._clients(namespace)
        # Events usually lack app labels. Filter by involved object after querying namespace.
        events = core.list_namespaced_event(namespace, _request_timeout=self.settings.timeout).items
        return {
            "events": [
                {
                    "reason": e.reason,
                    "message": e.message,
                    "type": e.type,
                    "object": e.involved_object.name,
                    "count": e.count,
                    "last_seen": str(e.last_timestamp or e.event_time),
                }
                for e in events
                if e.involved_object.name.startswith("prediction-api")
            ][-50:]
        }

    def pod_logs(self, namespace, pod, tail_lines=80):
        core, _ = self._clients(namespace)
        obj = core.read_namespaced_pod(pod, namespace, _request_timeout=self.settings.timeout)
        # Object names alone do not grant access to unrelated workloads.
        expected = dict(s.split("=", 1) for s in self.settings.selector.split(","))
        if any((obj.metadata.labels or {}).get(k) != v for k, v in expected.items()):
            from agent.permissions import PermissionDenied

            raise PermissionDenied("Pod is outside the approved application")
        return {
            "pod": pod,
            "logs": core.read_namespaced_pod_log(
                pod, namespace, tail_lines=tail_lines, _request_timeout=self.settings.timeout
            ),
        }

    def deployment_status(self, namespace="mlops", name="prediction-api"):
        core, apps = self._clients(namespace)
        if name != "prediction-api":
            from agent.permissions import PermissionDenied

            raise PermissionDenied("Deployment is outside the application allowlist")
        d = apps.read_namespaced_deployment(name, namespace, _request_timeout=self.settings.timeout)
        env = {
            e.name: e.value
            for c in d.spec.template.spec.containers
            for e in c.env or []
            if e.name in {"MODEL_RUN_ID", "APP_VERSION", "DATASET_MD5"}
        }
        for container in d.spec.template.spec.containers:
            for source in container.env_from or []:
                if source.config_map_ref and source.config_map_ref.name == "prediction-api-config":
                    cfg = core.read_namespaced_config_map(
                        "prediction-api-config", namespace, _request_timeout=self.settings.timeout
                    )
                    env["APP_VERSION"] = (cfg.data or {}).get("APP_VERSION", "unknown")
        return {
            "name": name,
            "namespace": namespace,
            "desired": d.spec.replicas,
            "ready": d.status.ready_replicas or 0,
            "available": d.status.available_replicas or 0,
            "updated": d.status.updated_replicas or 0,
            "rollout_complete": (
                (d.status.observed_generation or 0) >= d.metadata.generation
                and (d.status.updated_replicas or 0) == d.spec.replicas
                and (d.status.ready_replicas or 0) >= d.spec.replicas
                and not d.status.unavailable_replicas
            ),
            "generation": d.metadata.generation,
            "observed_generation": d.status.observed_generation,
            "images": [c.image for c in d.spec.template.spec.containers],
            "version": env,
            "readiness_paths": [
                c.readiness_probe.http_get.path
                for c in d.spec.template.spec.containers
                if c.readiness_probe and c.readiness_probe.http_get
            ],
            "conditions": [
                {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
                for c in d.status.conditions or []
            ],
        }

    def restart_deployment(self, namespace="mlops", name="prediction-api"):
        self.deployment_status(namespace, name)  # scope check before mutation
        _, apps = self._clients(namespace)
        result = apps.patch_namespaced_deployment(
            name,
            namespace,
            {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": datetime.now(UTC).isoformat()
                            }
                        }
                    }
                }
            },
            _request_timeout=self.settings.timeout,
        )
        return {
            "name": result.metadata.name,
            "accepted": True,
            "note": "Restart requested; successful rollout has not been established.",
        }

    def specs(self):
        return [
            ToolSpec(
                "kubernetes.list_workloads",
                "Inspect app Pods, Deployments, restarts and health",
                Namespace,
                self.list_workloads,
            ),
            ToolSpec(
                "kubernetes.services",
                "Inspect Services and ready backend addresses",
                Namespace,
                self.services,
            ),
            ToolSpec(
                "kubernetes.events", "Inspect application Kubernetes events", Namespace, self.events
            ),
            ToolSpec(
                "kubernetes.pod_logs",
                "Retrieve bounded application Pod logs",
                PodLog,
                self.pod_logs,
            ),
            ToolSpec(
                "kubernetes.deployment_status",
                "Read rollout state and deployed model version",
                Deployment,
                self.deployment_status,
            ),
            ToolSpec(
                "kubernetes.restart_deployment",
                "Restart allowlisted app with explicit confirmation",
                Deployment,
                self.restart_deployment,
                lambda a: f"kubernetes.restart_deployment:{a['namespace']}:{a['name']}",
            ),
        ]

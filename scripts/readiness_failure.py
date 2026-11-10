"""Inject a wrong readiness path into the local app and restore it in finally."""

import argparse
import json
import subprocess
import time

from agent.config import Settings
from agent.tools.kubernetes_tools import KubernetesTools
from scripts.demo import converse, save


def kubectl(*args):
    return subprocess.run(
        ["docker", "exec", "assignment3-k3s", "kubectl", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    ).stdout


def patch(path):
    kubectl(
        "patch",
        "deployment",
        "prediction-api",
        "-n",
        "mlops",
        "--type=json",
        "-p",
        json.dumps(
            [
                {
                    "op": "replace",
                    "path": "/spec/template/spec/containers/0/readinessProbe/httpGet/path",
                    "value": path,
                }
            ]
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", required=True, action="store_true")
    parser.parse_args()
    settings = Settings.from_env()
    original = json.loads(
        kubectl("get", "deployment", "prediction-api", "-n", "mlops", "-o", "json")
    )
    original_path = original["spec"]["template"]["spec"]["containers"][0]["readinessProbe"][
        "httpGet"
    ]["path"]
    try:
        patch("/deliberately-missing-readyz")
        adapter = KubernetesTools(settings)
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            events = adapter.events(settings.namespace)["events"]
            if any(
                "Readiness probe failed" in e["message"] and "404" in e["message"] for e in events
            ):
                break
            time.sleep(3)
        else:
            raise TimeoutError("Injected readiness failure did not produce the expected event")
        response = converse(
            settings, "failure-readiness-diagnosis", "Why is my Kubernetes deployment unhealthy?"
        )
        assert any("404" in r for r in response.recommendations)
    finally:
        patch(original_path)
        kubectl("rollout", "status", "deployment/prediction-api", "-n", "mlops", "--timeout=150s")
    recovered = KubernetesTools(settings).deployment_status(settings.namespace)
    assert recovered["rollout_complete"]
    save("failure-readiness-recovery", recovered)
    print("Readiness failure diagnosed; original configuration restored and rollout verified")


if __name__ == "__main__":
    main()

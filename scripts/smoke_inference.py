"""Call the real Kubernetes ClusterIP Service from the local k3s node."""

import json
import subprocess
from pathlib import Path

from scripts.demo import save


def main():
    prefix = ["docker", "exec", "assignment3-k3s"]
    response = subprocess.run(
        [*prefix, "kubectl", "get", "service", "prediction-api", "-n", "mlops", "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    ip = json.loads(response.stdout)["spec"]["clusterIP"]

    def call(path, payload=None):
        command = [*prefix, "wget", "-T", "15", "-qO-"]
        if payload:
            command += [
                "--header",
                "Content-Type: application/json",
                "--post-data",
                json.dumps(payload),
            ]
        result = subprocess.run(
            [*command, f"http://{ip}{path}"], capture_output=True, text=True, check=True, timeout=20
        )
        return json.loads(result.stdout)

    results = {path: call(path) for path in ("/healthz", "/readyz", "/version")}
    results["/predict"] = call("/predict", {"features": [[5.1, 3.5, 1.4, 0.2]]})
    info = json.loads(Path("models/current/metadata.json").read_text())
    assert results["/predict"]["predictions"] == [0]
    assert results["/predict"]["run_id"] == info["run_id"]
    assert results["/readyz"]["dataset_md5"] == info["dataset_md5"]
    save("inference-smoke", {"cluster_ip": ip, "checks": results, "passed": True})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

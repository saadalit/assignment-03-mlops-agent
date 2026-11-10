"""Create ignored admin kubeconfig for operator setup and a restricted agent identity."""

import base64
import subprocess
from pathlib import Path

import yaml


def run(*args):
    return subprocess.run(
        ["docker", "exec", "assignment3-k3s", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout


def main():
    Path(".local").mkdir(exist_ok=True)
    raw = run("cat", "/etc/rancher/k3s/k3s.yaml")
    config = yaml.safe_load(raw)
    config["clusters"][0]["cluster"]["server"] = "https://127.0.0.1:6443"
    admin = Path(".local/admin.kubeconfig")
    admin.write_text(yaml.safe_dump(config))
    admin.chmod(0o600)
    token = run(
        "kubectl", "create", "token", "mlops-agent-reader", "-n", "mlops", "--duration=24h"
    ).strip()
    config["users"] = [{"name": "agent-reader", "user": {"token": token}}]
    config["contexts"][0]["context"]["user"] = "agent-reader"
    config["contexts"][0]["context"]["namespace"] = "mlops"
    target = Path(".local/kubeconfig")
    target.write_text(yaml.safe_dump(config))
    target.chmod(0o600)
    assert base64.b64decode(config["clusters"][0]["cluster"]["certificate-authority-data"])
    print(
        "Created ignored operator and 24-hour read-only agent kubeconfigs; no credentials printed"
    )


if __name__ == "__main__":
    main()

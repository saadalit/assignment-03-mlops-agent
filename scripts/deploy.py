"""Explicit operator deployment: bake exact passed artifact provenance into the manifest."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def kubectl(*args, manifest=None):
    return subprocess.run(
        ["docker", "exec", "-i", "assignment3-k3s", "kubectl", *args],
        input=manifest,
        text=True,
        check=True,
        capture_output=True,
        timeout=180,
    ).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="Confirm deployment changes in the mlops namespace",
    )
    parser.parse_args()
    info = json.loads(Path("models/current/metadata.json").read_text())
    if info.get("quality_gate") != "passed":
        raise ValueError("Refusing to deploy an unapproved model")
    for path in ("namespace", "configmap", "service", "rbac"):
        print(kubectl("apply", "-f", "-", manifest=Path(f"kubernetes/{path}.yaml").read_text()))
    manifest = yaml.safe_load(Path("kubernetes/deployment.yaml").read_text())
    manifest["spec"]["template"]["spec"]["containers"][0]["env"] = [
        {"name": "MODEL_RUN_ID", "value": info["run_id"]},
        {"name": "DATASET_MD5", "value": info["dataset_md5"]},
    ]
    print(kubectl("apply", "-f", "-", manifest=yaml.safe_dump(manifest)))
    print(
        kubectl("rollout", "status", "deployment/prediction-api", "-n", "mlops", "--timeout=150s")
    )


if __name__ == "__main__":
    main()

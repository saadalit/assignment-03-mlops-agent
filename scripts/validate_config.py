"""Static structural checks independent of all external services."""

import ast
from pathlib import Path

import yaml


def main():
    objects = []
    for path in Path("kubernetes").glob("*.yaml"):
        objects.extend(yaml.safe_load_all(path.read_text()))
    deployment = next(d for d in objects if d["kind"] == "Deployment")
    assert deployment["spec"]["replicas"] >= 2
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert {"readinessProbe", "livenessProbe", "resources", "envFrom"} <= container.keys()
    assert {"requests", "limits"} <= container["resources"].keys()
    assert next(d for d in objects if d["kind"] == "Service")["spec"]["type"] == "ClusterIP"
    assert any(d["kind"] == "ConfigMap" for d in objects)
    assert list(yaml.safe_load(Path("dvc.yaml").read_text())["stages"]) == [
        "validate",
        "preprocess",
        "train",
        "evaluate",
        "quality_gate",
    ]
    ast.parse(Path("dags/training_pipeline.py").read_text())
    assert "retries" in Path("dags/training_pipeline.py").read_text()
    print("Kubernetes manifests, DVC stage order, and Airflow DAG syntax validated")


if __name__ == "__main__":
    main()

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
    tree = ast.parse(Path("dags/training_pipeline.py").read_text())
    pipeline = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    decorator = pipeline.decorator_list[0]
    defaults = next(k.value for k in decorator.keywords if k.arg == "default_args")
    retry = next(
        v
        for k, v in zip(defaults.keys, defaults.values, strict=True)
        if isinstance(k, ast.Constant) and k.value == "retries"
    )
    assert isinstance(retry, ast.Constant) and retry.value >= 1
    invocation = pipeline.body[-1].value
    chain = []
    while isinstance(invocation, ast.Call):
        chain.append(invocation.func.id)
        invocation = invocation.args[0] if invocation.args else None
    assert chain[::-1] == [
        "validate",
        "preprocess",
        "train",
        "evaluate_mlflow",
        "quality_gate",
        "register_candidate",
    ], "Airflow must gate registration behind validation, training, evaluation and quality checks"
    for obj in objects:
        if obj["kind"] != "Namespace":
            assert obj["metadata"]["namespace"] == "mlops"
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    service = next(d for d in objects if d["kind"] == "Service")
    assert all(labels.get(k) == v for k, v in service["spec"]["selector"].items())
    print("Kubernetes manifests, DVC stage order, and Airflow DAG syntax validated")


if __name__ == "__main__":
    main()

"""Capture real read-only integration and authorization checks; fails on missing services."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agent.config import Settings
from agent.orchestrator import build_executor
from scripts import demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/review-2026-11-15"))
    args = parser.parse_args()
    demo.EVIDENCE = args.output
    settings = Settings.from_env()
    executor = build_executor(settings)
    results = []

    def call(name, **arguments):
        result = executor.execute(name, arguments, "live-verification")
        results.append(result.model_dump())
        demo.save("integration-checks", {"checks": results})
        if result.status != "ok":
            raise RuntimeError(f"Live {name}: {result.status}; inspect saved evidence")
        return result.data

    for name in ("dvc.list_datasets", "dvc.status", "dvc.reproduction"):
        call(name)
    comparison = call("dvc.compare", ref="v1.0-data", other_ref="v2.0-data")
    assert comparison["changed"]
    for ref in ("v1.0-data", "v2.0-data"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.reproduce_dataset",
                ref,
                "--remote",
                ".local/dvc-remote",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        demo.save("reproduction-" + ref, json.loads(result.stdout))
    experiments = call("mlflow.list_experiments")["experiments"]
    runs = call("mlflow.list_runs")["runs"]
    assert len(runs) >= 2, "Two real runs required for comparison"
    call("mlflow.best_run")
    call("mlflow.get_run", run_id=runs[0]["run_id"])
    comparison_ids = [r["run_id"] for r in runs[:2]]
    baseline = next((e for e in experiments if e["name"] == "iris-classification-baseline"), None)
    if baseline:
        baseline_runs = call("mlflow.list_runs", experiment=baseline["name"])["runs"]
        assert baseline_runs, "Configured baseline experiment must contain a real run"
        comparison_ids = [runs[0]["run_id"], baseline_runs[0]["run_id"]]
    call("mlflow.compare_runs", run_ids=comparison_ids)
    call("mlflow.registry")
    call("airflow.list_dags")
    airflow_runs = call("airflow.list_runs")["dag_runs"]
    assert airflow_runs, "At least one real Airflow execution required"
    call("airflow.failed_tasks", run_id=airflow_runs[0]["dag_run_id"])
    deployment = call("kubernetes.deployment_status")
    assert deployment["rollout_complete"] and deployment["ready"] >= 2
    workloads = call("kubernetes.list_workloads")
    assert len(workloads["pods"]) >= 2 and all(p["healthy"] for p in workloads["pods"])
    services = call("kubernetes.services")
    assert any(e["ready_addresses"] for e in services["endpoints"])
    call("kubernetes.events")
    call("kubernetes.pod_logs", pod=workloads["pods"][0]["name"])
    denied = [
        executor.execute("kubernetes.restart_deployment", {}, "live-safety"),
        executor.execute("kubernetes.list_workloads", {"namespace": "kube-system"}, "live-safety"),
        executor.execute("airflow.trigger_training", {}, "live-safety"),
    ]
    demo.save("safety-controls", {"checks": [r.model_dump() for r in denied]})
    assert all(r.status == "denied" for r in denied)
    print(f"Verified {len(results)} live read operations and 3 authorization denials")


if __name__ == "__main__":
    main()

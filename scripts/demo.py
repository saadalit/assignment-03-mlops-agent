"""Capture real-service conversations and optionally inject recoverable failures.

No mock results are used. Failure injection requires an explicit operator flag.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from agent.config import Settings
from agent.executor import scrub
from agent.orchestrator import Orchestrator
from agent.schemas import AgentRequest
from agent.tools.airflow_tools import AirflowTools
from agent.tools.mlflow_tools import MLflowTools

EVIDENCE = Path("docs/evidence")


def save(name, value):
    value = {
        "collected_at": datetime.now(UTC).isoformat(),
        "source": "live local services",
        "result": scrub(value),
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / f"{name}.json").write_text(json.dumps(value, indent=2, default=str))


def wait_run(airflow, run_id, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = airflow._request("GET", f"{airflow._dag(airflow.settings.airflow_dag)}/{run_id}")
        if run["state"] in {"success", "failed"}:
            tasks = airflow.failed_tasks(airflow.settings.airflow_dag, run_id)
            return {"run": run, "tasks": tasks}
        time.sleep(5)
    raise TimeoutError(f"Airflow run {run_id} has not reached a terminal state")


def converse(settings, name, message, confirmations=None):
    response = Orchestrator(settings).run(
        AgentRequest(message=message, confirmations=confirmations or [])
    )
    save(name, {"user": message, "agent": response.model_dump()})
    print(
        f"{name}: {len(response.evidence)} actual tool calls; "
        f"{sum(r.status != 'ok' for r in response.evidence)} unsuccessful",
        flush=True,
    )
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trigger", action="store_true", help="Explicitly start and await training"
    )
    parser.add_argument(
        "--confirm-failures",
        action="store_true",
        help="Explicitly inject validation and quality-gate failures",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    airflow, mlflow = AirflowTools(settings), MLflowTools(settings)
    if args.trigger:
        response = converse(
            settings,
            "conversation-03-trigger",
            "Trigger the model training pipeline",
            [f"airflow.trigger_training:{settings.airflow_dag}"],
        )
        trigger = next(r for r in response.evidence if r.tool == "airflow.trigger_training")
        if trigger.status != "ok":
            raise RuntimeError(
                "Live training could not be triggered; inspect conversation evidence"
            )
        result = wait_run(airflow, trigger.data["dag_run_id"])
        save("airflow-success-run", result)
        if result["run"]["state"] != "success":
            raise RuntimeError("Training failed; inspect real Airflow task evidence")
        converse(settings, "conversation-03-completion", "Report the training pipeline status")
    best = mlflow.best_run()["candidate"]
    if not best:
        raise RuntimeError("No real MLflow candidate is available")
    converse(
        settings,
        "conversation-01-provenance",
        f"Which dataset version was used for experiment run {best['run_id']}?",
    )
    converse(
        settings,
        "conversation-02-selection",
        "Compare the last five training runs and recommend the best model based on F1 score",
    )
    converse(settings, "conversation-04-deployment", "Check Kubernetes deployment health and logs")
    converse(
        settings,
        "conversation-05-multi-tool",
        "Find the best recent model, identify the dataset version used to train it, "
        "check the deployment status, and explain whether retraining is necessary",
    )
    save("mlflow-experiments", mlflow.list_experiments())
    save("mlflow-runs", mlflow.list_runs())
    save("mlflow-registry", mlflow.registry())
    save("airflow-dags", airflow.list_dags())
    if args.confirm_failures:
        for name, conf in (
            ("validation", {"inject_invalid": True}),
            ("quality-gate", {"min_f1": 1.01}),
        ):
            run = airflow._request(
                "POST",
                airflow._dag(settings.airflow_dag),
                json={"logical_date": datetime.now(UTC).isoformat(), "conf": conf},
            )
            result = wait_run(airflow, run["dag_run_id"])
            save(f"failure-{name}-run", result)
            if result["run"]["state"] != "failed":
                raise RuntimeError(f"Injected {name} failure did not fail the workflow")
            tasks = result["tasks"]["task_instances"]
            promotion = next(t for t in tasks if t["task_id"] == "register_candidate")
            if promotion["state"] == "success":
                raise RuntimeError("Safety violation: failed workflow promoted a candidate")
            converse(settings, f"failure-{name}-diagnosis", "Diagnose the failed training pipeline")
    print("Live evidence exported; no credentials included", flush=True)


if __name__ == "__main__":
    main()

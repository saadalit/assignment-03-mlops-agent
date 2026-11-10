"""Airflow 3 SDK DAG: no infrastructure access while parsing the DAG."""

from datetime import datetime, timedelta

from airflow.sdk import dag, get_current_context, task


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=5)},
    tags=["mlops", "assignment03"],
)
def training_pipeline():
    @task
    def validate():
        import hashlib
        import logging
        import os

        from training import stages

        os.chdir("/workspace")
        ctx = get_current_context()
        conf = ctx["dag_run"].conf or {}
        workspace = ".local/runs/" + hashlib.sha256(ctx["run_id"].encode()).hexdigest()[:16]
        logging.info("Validating DVC provenance and dataset schema")
        return stages.validate(
            workspace=workspace, inject_invalid=conf.get("inject_invalid", False)
        )

    @task
    def preprocess(info):
        import os

        from training import stages

        os.chdir("/workspace")
        return stages.preprocess(info)

    @task
    def train(info):
        import os

        from training import stages

        os.chdir("/workspace")
        return stages.train(
            info, max_depth=(get_current_context()["dag_run"].conf or {}).get("max_depth", 3)
        )

    @task
    def evaluate_mlflow(info):
        import os

        from training import stages

        os.chdir("/workspace")
        return stages.evaluate(
            info, min_f1=(get_current_context()["dag_run"].conf or {}).get("min_f1", 0.85)
        )

    @task
    def quality_gate(info):
        import os

        from training import stages

        os.chdir("/workspace")
        return stages.quality_gate(info)

    @task
    def register_candidate(info):
        import os

        from training import stages

        os.chdir("/workspace")
        # Record a candidate only. Deployment remains a separately authorized operation.
        return stages.register_candidate(info)

    register_candidate(quality_gate(evaluate_mlflow(train(preprocess(validate())))))


training_pipeline()

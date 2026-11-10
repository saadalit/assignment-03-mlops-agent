import math
import os
import time

from mlflow import MlflowClient

from agent.config import Settings
from agent.executor import ToolSpec
from agent.schemas import BestRun, CompareRuns, Empty, RegistryQuery, RunId, RunQuery


def serialize_run(run):
    return {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id,
            "status": run.info.status, "start_time": run.info.start_time,
            "end_time": run.info.end_time, "metrics": dict(run.data.metrics),
            "params": dict(run.data.params), "tags": dict(run.data.tags)}


class MLflowTools:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(int(settings.timeout))
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
        self.client = client or MlflowClient(tracking_uri=settings.mlflow_uri)

    def list_experiments(self):
        return {"experiments": [{"id": e.experiment_id, "name": e.name,
                                 "artifact_location": e.artifact_location}
                                for e in self.client.search_experiments(max_results=100)]}

    def list_runs(self, experiment="iris-classification", limit=5, days=30):
        exp = self.client.get_experiment_by_name(experiment)
        if exp is None:
            return {"runs": [], "experiment": experiment, "note": "Experiment not found"}
        after = int((time.time() - days * 86400) * 1000)
        runs = self.client.search_runs([exp.experiment_id],
                                      filter_string=f"attributes.start_time >= {after}",
                                      order_by=["attributes.start_time DESC"], max_results=limit)
        return {"experiment": experiment, "days": days, "runs": [serialize_run(r) for r in runs]}

    def best_run(self, experiment="iris-classification", limit=5, days=30, metric="f1", mode="max"):
        result = self.list_runs(experiment, limit, days)
        eligible = [r for r in result["runs"] if r["status"] == "FINISHED"
                    and metric in r["metrics"] and math.isfinite(r["metrics"][metric])]
        ordered = sorted(eligible, key=lambda r: (r["metrics"][metric], r["start_time"]),
                         reverse=(mode == "max"))
        return {**result, "metric": metric, "mode": mode,
                "candidate": ordered[0] if ordered else None,
                "selection_scope": f"Most recent {limit} runs in the last {days} days",
                "caution": "Metric ranking alone does not establish production readiness; "
                           "check quality gate, data, deployment, drift and latency."}

    def get_run(self, run_id):
        result = serialize_run(self.client.get_run(run_id))
        result["artifacts"] = [{"path": a.path, "is_dir": a.is_dir, "size": a.file_size}
                               for a in self.client.list_artifacts(run_id)]
        return result

    def compare_runs(self, run_ids):
        from agent.schemas import RunId
        for run_id in run_ids:
            RunId(run_id=run_id)
        return {"runs": [self.get_run(r) for r in run_ids],
                "note": "Compare evaluation splits and dataset provenance before comparing scores."}

    def registry(self, name="iris-classifier"):
        versions = self.client.search_model_versions(f"name='{name}'", max_results=100)
        return {"name": name, "versions": [{"version": v.version, "run_id": v.run_id,
                                            "status": v.status, "source": v.source,
                                            "aliases": list(v.aliases or [])} for v in versions]}

    def specs(self):
        return [
            ToolSpec("mlflow.list_experiments", "List recorded MLflow experiments", Empty,
                     self.list_experiments),
            ToolSpec("mlflow.list_runs", "List recent runs with metrics, parameters and tags", RunQuery,
                     self.list_runs),
            ToolSpec("mlflow.best_run", "Rank recent FINISHED runs by a specified metric", BestRun,
                     self.best_run),
            ToolSpec("mlflow.get_run", "Read a run's metrics, parameters, artifacts and provenance",
                     RunId, self.get_run),
            ToolSpec("mlflow.compare_runs", "Compare runs across one or more experiments", CompareRuns,
                     self.compare_runs),
            ToolSpec("mlflow.registry", "Retrieve registered model versions when available",
                     RegistryQuery, self.registry),
        ]

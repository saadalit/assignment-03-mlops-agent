import os
from datetime import UTC, datetime
from urllib.parse import quote

import httpx

from agent.config import Settings
from agent.executor import ToolSpec
from agent.permissions import PermissionDenied
from agent.schemas import DagQuery, DagRun, Empty, TaskLog, TriggerDag


class AirflowTools:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url=settings.airflow_url, timeout=settings.timeout, follow_redirects=False
        )

    def _headers(self):
        token = os.getenv("AIRFLOW_TOKEN")
        if not token:
            username, password = os.getenv("AIRFLOW_USERNAME"), os.getenv("AIRFLOW_PASSWORD")
            if not username or not password:
                raise PermissionDenied("Airflow authentication is not configured")
            response = self.client.post(
                "/auth/token", json={"username": username, "password": password}
            )
            response.raise_for_status()
            token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def _request(self, method, path, **kwargs):
        response = self.client.request(method, f"/api/v2{path}", headers=self._headers(), **kwargs)
        response.raise_for_status()
        return response.json()

    def _dag(self, dag_id):
        if dag_id != self.settings.airflow_dag:
            raise PermissionDenied("DAG is outside the approved training allowlist")
        return f"/dags/{quote(dag_id, safe='')}/dagRuns"

    def list_dags(self):
        return self._request("GET", "/dags", params={"limit": 100})

    def list_runs(self, dag_id="training_pipeline", limit=5):
        return self._request(
            "GET", self._dag(dag_id), params={"limit": limit, "order_by": "-start_date"}
        )

    def failed_tasks(self, dag_id, run_id, limit=5):
        result = self._request(
            "GET",
            f"{self._dag(dag_id)}/{quote(run_id, safe='')}/taskInstances",
            params={"limit": 100},
        )
        result["failed_tasks"] = [
            t
            for t in result.get("task_instances", [])
            if t.get("state") in {"failed", "upstream_failed", "up_for_retry"}
        ]
        return result

    def task_logs(self, dag_id, run_id, task_id, try_number=1, limit=5):
        path = (
            f"{self._dag(dag_id)}/{quote(run_id, safe='')}/taskInstances/"
            f"{quote(task_id, safe='')}/logs/{try_number}"
        )
        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "task_id": task_id,
            "log": self._request("GET", path),
        }

    def trigger_training(self, dag_id="training_pipeline"):
        return self._request(
            "POST",
            self._dag(dag_id),
            json={
                "logical_date": datetime.now(UTC).isoformat(),
                "conf": {},
            },
        )

    def specs(self):
        return [
            ToolSpec(
                "airflow.list_dags",
                "List DAGs through authenticated Airflow 3 API",
                Empty,
                self.list_dags,
            ),
            ToolSpec(
                "airflow.list_runs",
                "Read recent training workflow states",
                DagQuery,
                self.list_runs,
            ),
            ToolSpec(
                "airflow.failed_tasks",
                "Read task execution states and identify failures",
                DagRun,
                self.failed_tasks,
            ),
            ToolSpec(
                "airflow.task_logs",
                "Retrieve a task attempt's failure diagnostics",
                TaskLog,
                self.task_logs,
            ),
            ToolSpec(
                "airflow.trigger_training",
                "Trigger the approved DAG after explicit confirmation",
                TriggerDag,
                self.trigger_training,
                lambda a: f"airflow.trigger_training:{a['dag_id']}",
            ),
        ]

# Structured tool definitions and permissions

Inputs are generated as JSON Schema from strict Pydantic models in `agent/schemas.py`.
All reject extra fields. Results contain `tool`, `status`, `data`, `error`, `duration_ms`, and
`request_id`. Status is `ok`, `error`, `timeout`, `invalid`, or `denied`. Logs record only tool
name, status, duration, and request ID. Service evidence is redacted before returning it.

| Tool | Inputs (defaults) | Permission / behavior |
| --- | --- | --- |
| `dvc.list_datasets` | none | Read committed DVC pointers |
| `dvc.current_version` | `ref=HEAD`, `path=data/iris.csv.dvc` | Resolve immutable Git commit; read DVC MD5; compare workspace when HEAD |
| `dvc.compare` | `ref`, `other_ref`, `path` | Read two pointers; execute approved `dvc diff --json` |
| `dvc.status` | none | Approved `dvc status --json`; no checkout |
| `dvc.reproduction` | `ref`, `path` | Explain worktree/pull steps; no data mutation |
| `mlflow.list_experiments` | none | MlflowClient; at most 100 experiments |
| `mlflow.list_runs` | `experiment=iris-classification`, `limit=5`, `days=30` | Recent recorded metrics, parameters and tags |
| `mlflow.best_run` | run query + `metric=f1`, `mode=max` | FINISHED runs with finite metrics; comparison included; readiness caveat |
| `mlflow.get_run` | `run_id` | Exact 32-character hex ID; include artifacts |
| `mlflow.compare_runs` | 2–20 `run_ids` | Compare recorded runs, including runs from different experiments |
| `mlflow.registry` | `name=iris-classifier` | Registry versions and run references when available |
| `airflow.list_dags` | none | Authenticated GET `/api/v2/dags` |
| `airflow.list_runs` | `dag_id=training_pipeline`, `limit=5` | Allowlisted DAG only |
| `airflow.failed_tasks` | DAG query + `run_id` | All task states plus failed/retrying/upstream-failed tasks |
| `airflow.task_logs` | DAG run + `task_id`, `try_number=1` | Bounded diagnostic selection by orchestrator |
| `airflow.trigger_training` | `dag_id=training_pipeline` | Mutation; exact trusted confirmation; no arbitrary run configuration |
| `kubernetes.list_workloads` | `namespace=mlops` | App-label selector, Pod readiness, restarts, Deployments |
| `kubernetes.deployment_status` | namespace + `name=prediction-api` | Allowlisted app only; image, run ID, hash, generation and rollout |
| `kubernetes.services` | namespace | App Services and ready backend endpoints |
| `kubernetes.events` | namespace | Application object events; events need not carry app labels |
| `kubernetes.pod_logs` | namespace, `pod`, `tail_lines=80` | App-label membership checked; max 200 lines |
| `kubernetes.restart_deployment` | namespace + name | Mutation; namespace/app allowlist, exact confirmation, Kubernetes RBAC |

Run queries allow at most 100 runs and a 1–3650 day window. Metric/registry names exclude
MLflow filter injection syntax. Dataset paths cannot be absolute or traverse parents.
Subprocesses never invoke a shell; there is no generic exec tool. Tool timeouts come from
bounded subprocess, HTTP and SDK request settings. Kubernetes retries are disabled. Failed
operations never return fabricated service data; mutation timeouts imply an unknown outcome,
not proof of rollback. Inspect state before retrying a mutation.

Trusted confirmations are separate from tool arguments and bind to these exact action IDs:

* `airflow.trigger_training:training_pipeline`
* `kubernetes.restart_deployment:mlops:prediction-api`

Neither an LLM suggestion nor a tool response can supply confirmation. Authenticated API
clients are responsible for collecting human consent before sending action IDs. The local
CLI user's explicit `--confirm` supplies consent. Operator setup/test scripts have separate
explicit `--confirm` / `--confirm-failures` flags and are not agent tools.

The agent Kubernetes ServiceAccount has read-only namespace permissions. The optional
restarter Role grants only `get,patch` on `prediction-api` and has no default RoleBinding.
The generated reader token is short-lived (24h). Credentials remain in ignored local files.
Airflow SimpleAuthManager is for the local demonstration; use DAG-scoped authorization in
a production auth manager. The agent HTTP API rejects access without a configured token.

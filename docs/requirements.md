# Assignment requirements and acceptance evidence

Source: `Assignment_03_MLOps_AI_Agent_Complete.pdf` (7 pages, Fall 2026).

See the [15 November requirement-by-requirement checklist](review-2026-11-15.md) for the expanded
audit, modifications, current evidence, and the distinction between local and hosted CI checks.

| Requirement | Implementation | Verification |
| --- | --- | --- |
| DVC: tracked datasets, Git/hash provenance, comparison, changes, reproduction | `agent/tools/dvc_tools.py`, `dvc.yaml` | Unit tests and actual version checkout in isolated worktree |
| MLflow: experiments/runs, metrics/params/artifacts, best run, comparison, registry | `agent/tools/mlflow_tools.py` | Mocked tests and live tracking server |
| Airflow 3: authenticated v2 API, DAG listing/status/trigger/failure diagnostics | `agent/tools/airflow_tools.py` | API tests and actual DAG executions |
| Training DAG: validation → preprocessing → training → evaluation/MLflow → quality gate | `dags/training_pipeline.py`, `training/` | Dependencies, retries, successful and deliberately failed runs |
| Kubernetes: Pods, Deployments, logs, Services, events, deployed version | `agent/tools/kubernetes_tools.py` | Mocked tests and local Kubernetes cluster |
| Deployment: 2 replicas, ClusterIP, probes, resources, ConfigMap | `kubernetes/` | Live readiness and prediction request |
| Cluster mutations: explicit confirmation and namespace restriction | `agent/permissions.py` | Authorization and scope tests |
| Structured multi-tool AI agent; facts, uncertainty, recommendations | `agent/orchestrator.py`, `agent/schemas.py` | Selection, integration, unavailable-service tests |
| Mandatory four-tool scenario, no automatic retraining | `agent/orchestrator.py` | Live conversation and no-mutation test |
| >=8 pytest tests; timeouts, invalid input, failures; structured safe logs | `tests/`, `agent/executor.py` | pytest and log-redaction assertions |
| Reproducible Python, Docker, GitHub repository, PR lint/test CI | `pyproject.toml`, `uv.lock`, `Dockerfile`, `.github/workflows/ci.yml` | Locked install, lint, tests, container smoke test |
| Five conversations; two injected failures; report and engineering defense | `docs/report.md`, `docs/evidence/`, `architecture.md` | Exported real-service results, diagnosis and resolution |

Evidence must state which services actually ran. Mocks are test fixtures, never operational results.

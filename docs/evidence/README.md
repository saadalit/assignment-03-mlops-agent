# Verification evidence

All `conversation-*.json` exports contain actual tool responses. Conversations 01–05 use
the explicitly identified deterministic planner; conversation 06 uses a real locally hosted
`qwen2.5:1.5b` LLM via Ollama. They are separate from mocked pytest fixtures.

* `verification-summary.json`: test/check summary and tested source revision.
* `test-results.txt`: complete pytest and statement-coverage output.
* `dvc-reproduction-*.json`: actual isolated Git worktrees, DVC pulls and MD5 verification.
* `dvc-version-diff.json`, `dvc-pipeline-run.txt`, `dvc-pipeline-graph.txt`: real DVC results.
* `mlflow-*.json`: experiments, recent training metrics/provenance and registered versions.
* `airflow-*.json`: DAG definitions and actual successful workflow/task states.
* `inference-smoke.json`: real ClusterIP liveness, readiness, version and prediction calls.
* `agent-http-smoke.json`: actual authenticated/unauthenticated API behavior.
* `failure-*.json`: deliberately injected failures, agent diagnosis and recovery evidence.

Collection times are UTC; the report's human verification date uses Asia/Karachi. Exports
are historical snapshots. Run IDs, Pod names, addresses and timestamps will change when the
demonstrations are rerun. Configured passwords/tokens are removed, and local credentials,
database state, model binaries and private logs are excluded from Git.

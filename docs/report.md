# Technical report — Assignment 03

Author: Saad Shahid · Fall 2026 · Verification date: 10 November 2026 (Asia/Karachi)

Repository: [saadalit/assignment-03-mlops-agent](https://github.com/saadalit/assignment-03-mlops-agent)

## Architecture and implementation

[architecture.md](../architecture.md) contains the full component diagram and rationale.
The system separates natural-language planning, typed tool execution, trusted authorization,
and evidence rendering. There are 22 callable tools across four adapters; definitions and
permissions are documented in [tools.md](tools.md). The supported LLM backend is Ollama's
tool-calling API. A transparent deterministic planner supports the required workflows when
no LLM is installed. Responses identify the mode used. Generated model prose cannot override
tool evidence or approve a state change.

DVC uses subprocess argument lists, MLflow uses `MlflowClient`, Airflow uses authenticated
public API **v2**, and Kubernetes uses the official Python SDK. Tools return structured success,
failure, timeout, denial or invalid-input results. Audit logs record names, status, duration and
request IDs. Error bodies and credentials are excluded from logs, and returned diagnostic data
is redacted. Tool failures produce uncertainty rather than a guessed healthy state.

The local training pipeline and Airflow DAG share the same Python stage functions. Stages are
validation → preprocessing → training → evaluation/MLflow → quality gate → registration.
Every DAG task retries once with a five-second delay. Per-run workspaces prevent cross-run
artifact collisions. Validation checks schema, missing values, class coverage, finite values,
workspace MD5, and the committed DVC pointer. A stratified 70/30 split with seed 42 provides
repeatable evaluation; the model is a 60-tree RandomForestClassifier. MLflow records the
dataset hash, immutable Git reference, parameters, metrics, split policy and quality-gate state.

Model registration occurs downstream of a successful quality gate. It does not deploy a model.
Explicit operator setup exports a passed artifact, builds an image tagged with its full run ID,
and deploys it. The FastAPI service validates run/hash identity against baked metadata. The
Kubernetes manifests provide two replicas, a ClusterIP Service, startup/readiness/liveness
probes, CPU/memory requests and limits, a ConfigMap and restricted security contexts.

## Dataset provenance and reproduction

| Dataset reference | Git commit | Actual DVC MD5 |
| --- | --- | --- |
| `v1.0-data` | `12b005559f25d93419ccdc93f3cd470d2ee0c09f` | `4224576f0267bf88902f87f0f6200967` |
| `v2.0-data` | `fca41b0a44a9b26fdad713b17cade6f293d574b4` | `5c2fcf018f5f74ee80f846aabcc97f36` |

Both versions contain 150 Iris rows. Version 2 deterministically shuffles row order; it is a
different content version, not a claim of new field data. Real `dvc pull` operations in separate
temporary Git worktrees recovered both files and reproduced their recorded checksums.
Evidence: [v1 checkout](evidence/dvc-reproduction-v1.json),
[v2 checkout](evidence/dvc-reproduction-v2.json), [diff](evidence/dvc-version-diff.json),
[pipeline execution](evidence/dvc-pipeline-run.txt), and [status](evidence/dvc-status-after-pipeline.json).
The quality-gate stage is intentionally `always_changed` so DVC rechecks it; that status is
distinct from a changed dataset. The shared local remote is populated from deterministic
bootstrap data; configuring a cloud remote is an explicit deployment improvement.

## Recorded model comparison

| Actual MLflow run | Depth | Accuracy | Macro F1 | Gate |
| --- | ---: | ---: | ---: | --- |
| `6138bca51d5f4b0ab91bf5c8d4db3adc` | 1 | 0.933333 | 0.933259 | passed |
| `b70a9a9e59a5423a8264e76d90714a3b` | 3 | 0.955556 | 0.955556 | passed |

Both use DVC hash `5c2fcf018f5f74ee80f846aabcc97f36` and the same split policy. Later DVC
and Airflow executions create additional recorded runs; the exported recent-run comparison
is authoritative for each conversation's collection time. Ranking uses recent FINISHED runs
with finite scores and respects maximization/minimization. The selected run remains a candidate:
F1 alone does not establish readiness, fairness, latency, drift resistance or generalization.
An evaluation can finish even when its downstream quality gate fails, so gate tags must also
be inspected. Registry state and model artifacts are retrieved through supported MLflow APIs.

## Five required conversations and operational evidence

The following exports contain actual service outputs, request IDs, status and elapsed time.
The core demonstration uses `planner=rules`; the LLM backend is separately exercised through
its supported tool-calling protocol. A real local `qwen2.5:1.5b` model selected and executed
`dvc.list_datasets` successfully; see the [LLM conversation](evidence/conversation-06-llm.json).
Mocked unit-test responses are not passed off as live evidence.

| Scenario | Request and evidence |
| --- | --- |
| Dataset provenance | “Which dataset version was used for experiment run …?” resolves the real run's tags, then the exact Git-tracked DVC pointer. [Conversation](evidence/conversation-01-provenance.json) |
| Model selection | “Compare the last five training runs and recommend the best model based on F1 score.” Returns the recorded comparison, candidate, selection scope and production-readiness caveat. [Conversation](evidence/conversation-02-selection.json) |
| Pipeline execution | “Trigger the model training pipeline” includes the explicit action confirmation. The initial response reports acceptance; a later query establishes completion. [Trigger](evidence/conversation-03-trigger.json), [completion](evidence/conversation-03-completion.json), [real successful DAG and task states](evidence/airflow-success-run.json) |
| Deployment diagnosis | “Check Kubernetes deployment health and logs.” Examines desired/updated/ready replicas, generations, Pod states and restarts, image/version, Service endpoints, events and bounded logs. [Conversation](evidence/conversation-04-deployment.json) |
| Multi-tool reasoning | The mandatory best model + dataset + deployment + retraining request queries all four integrations. It correlates run/hash identity and reports remaining uncertainty. It does not trigger training or change the cluster. [Conversation](evidence/conversation-05-multi-tool.json) |

Additional exports: [experiments](evidence/mlflow-experiments.json),
[recent runs](evidence/mlflow-runs.json), [registry](evidence/mlflow-registry.json),
[DAG definitions](evidence/airflow-dags.json), [inference smoke test](evidence/inference-smoke.json),
and [authenticated agent API](evidence/agent-http-smoke.json).
These machine-readable exports satisfy the assignment's “screenshots or exported evidence”
option; they retain more inspectable detail than a UI screenshot.

Retraining is not inferred from an unavailable tool, an unhealthy deployment, or a score ranking.
It needs an explicit freshness policy, a failed quality gate, or supplied degradation evidence.
Without such evidence, the answer states that necessity has not been established. A candidate
different from the deployed run calls for review, not an automatic retraining job.

## Failure investigation 1: deliberately invalid training data

Injection: a real Airflow DAG run with `conf={"inject_invalid": true}` causes validation to
raise `DATA_VALIDATION_FAILED: deliberately injected missing target column`. This is an
explicit fault-injection flag; it does not corrupt the committed dataset. Validation retries
once and fails on its second attempt. All downstream tasks become `upstream_failed`.
The agent reads task states and the failed validation attempt's logs and recommends repairing
schema/provenance before an explicitly requested rerun. No candidate is registered.

Root cause: invalid input at the validation boundary. Resolution: remove the injection and
rerun the approved DAG against the validated DVC content. The successful normal run proves
the underlying dataset/workflow can complete. Evidence: [failed run and tasks](evidence/failure-validation-run.json)
and [agent diagnosis with actual logs](evidence/failure-validation-diagnosis.json).

## Failure investigation 2: deliberately impossible quality threshold

Injection: a real DAG run with `conf={"min_f1": 1.01}` proceeds through training/evaluation,
but a measured macro F1 of approximately 0.9556 cannot pass the deliberately impossible gate.
The gate logs `QUALITY_GATE_FAILED`, retries, and fails. Candidate registration is
`upstream_failed`. The MLflow run records `quality_gate=failed`.

Root cause: an invalid evaluation policy, not demonstrated production degradation. Resolution:
restore the normal threshold of 0.85 and rerun. The agent retrieves the real attempt logs and
recommends checking the threshold and evaluation data before tuning/retraining. It never
promotes the failed candidate. Evidence: [run and tasks](evidence/failure-quality-gate-run.json)
and [agent diagnosis](evidence/failure-quality-gate-diagnosis.json).

## Additional deployment failure and recovery

`scripts/readiness_failure.py --confirm` changes only the demo deployment's readiness path
to a deliberately missing endpoint. Kubernetes emits HTTP 404 readiness warnings; new Pods
cannot become ready. A rolling update can keep old replicas serving, so a stalled rollout
must not be described as a proven complete API outage. The agent uses events, Pod readiness,
rollout generations and logs to identify the readiness problem and recommend verifying
`/readyz`. Its serving-model identity claim remains uncertain during an incomplete rollout.
The script restores the original probe in `finally` and waits for a complete rollout.
Evidence: [diagnosis](evidence/failure-readiness-diagnosis.json) and
[recovery](evidence/failure-readiness-recovery.json).

## Testing and engineering defense

Pytest passes **48 tests** (78% combined statement coverage), exceeding the required eight cases.
Coverage includes successful selection
and execution, four-tool correlation without mutation, timeouts, invalid arguments, service
errors, unavailable evidence, missing provenance, secret redaction, unauthorized/cross-namespace
mutations, unrelated Pod-log access, confirmed scoped restart behavior, LLM protocol, ignoring
fabricated LLM prose, root-task diagnosis ordering, incomplete rollouts, validation, quality-gate
blocking, readiness, prediction shape checks and HTTP authentication.

Unit tests mock external services; real DVC worktrees, MLflow runs/registry, Airflow DAGs,
Kubernetes workloads and inference are demonstrated separately. This keeps PR CI reproducible
while testing integration semantics against real APIs. See [test results](evidence/test-results.txt)
and the repository's [GitHub Actions runs](https://github.com/saadalit/assignment-03-mlops-agent/actions).

An unexpected failure during integration exposed a read-only/ownership problem with Airflow's
password file: its auth manager opens that file for append even when passwords already exist.
Resolution: keep the supplied secret mount read-only, copy it to the writable Airflow home,
and initialize named-volume ownership for UID 50000. This allowed the authenticated API and
all six tasks to run. Actual validation-failure evidence also exposed a diagnosis bug:
`upstream_failed` tasks appeared before the root failure. Filtering root failures before
limiting log retrieval fixed it; a regression test now covers that ordering.

## Known limitations and improvements

* Iris is an educational dataset. There is no live production drift/degradation feed or fairness,
  latency and robustness certification. Absence of those measurements is reported as uncertainty.
* Default deterministic planning covers bounded workflows, not arbitrary conversation. The
  Ollama backend supports structured tool calling; small models can choose poor tools or
  exceed the bounded loop. Use a stronger tool-capable model for broader requests.
  On the 4 GB test machine, an initial full-registry cold request exceeded 90 seconds and
  correctly returned uncertainty. Configurable 180-second LLM timeouts and DVC-only domain
  routing enabled the successful real LLM demonstration. All four tool families remain
  available for multi-tool requests. The model is stopped after verification to free memory.
* Local SQLite, filesystem DVC remote, SimpleAuthManager and single-node k3s are demonstration
  topologies. Production requires durable shared stores, PostgreSQL, TLS, scoped identity,
  high availability, artifact signing and operational monitoring.
* A timed-out mutation has an unknown outcome. Inspect real state before retrying; the agent
  does not claim an automatic rollback. Read-only agent RBAC is intentionally insufficient
  for confirmed restarts until an administrator supplies a separately scoped write identity.
* MLflow model artifacts are trusted local build inputs. A production registry should verify
  artifact signatures before deserialization. No untrusted model-upload endpoint is exposed.
* Recorded evidence is a dated snapshot. Short-lived Kubernetes tokens expire; refresh local
  credentials and rerun demonstrations to establish current health.

The repository includes no API keys, passwords, JWTs, kubeconfigs or service databases.

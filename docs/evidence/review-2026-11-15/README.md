# Live verification evidence, 15 November 2026

These exports came from actual local services. JSON files contain UTC collection timestamps,
source labels, tool status and returned data. Unit tests use mocks and are reported separately.
The original 10 November evidence in the parent directory remains a historical record.

| Check | Result and evidence |
| --- | --- |
| Local CI-equivalent checks | [Command results](local-checks.json), [lint](lint.txt), [format](format.txt), [pytest/coverage](tests.txt), [configuration](configuration.txt), [DVC graph](dvc-graph.txt) |
| Current files and reachable Git history | [CI-equivalent audit](repository-audit.txt) and [final source/history audit](repository-audit-final.json); no violations at collection time |
| Four MLOps integrations | [19 actual read operations](integration-checks.json), all successful, including comparison across two MLflow experiments |
| Dataset reproduction | Two actual isolated worktree checkouts and DVC pulls: [v1](reproduction-v1.0-data.json), [v2](reproduction-v2.0-data.json); recovered MD5 values match pointers |
| Complete DVC pipeline | [Actual pipeline evaluation](dvc-pipeline-evaluation.json); refreshed `dvc.lock` records produced outputs |
| MLflow experiments, metrics and registry | [Experiments](mlflow-experiments.json), [runs](mlflow-runs.json), [registered versions](mlflow-registry.json), run details/artifact listings in integration checks |
| Successful Airflow execution | [Six successful tasks](airflow-success-run.json), including downstream registration |
| Cluster and inference | Two ready replicas and ready Service endpoints in integration checks; [actual ClusterIP prediction](inference-smoke.json) returns class 0 |
| Authenticated agent HTTP | [HTTP smoke](agent-http-smoke.json): missing bearer token rejected; authorized request executes actual DVC reads |
| Safety | [Live denials](safety-controls.json): unconfirmed restart, out-of-scope namespace and unconfirmed training; [reader RBAC](kubernetes-reader-rbac.json) also prohibits patching the Deployment |
| Actual Ollama mandatory scenario | **PASS**: six actual `qwen2.5:1.5b` turns, seven successful required tools, one rejected proposal corrected by LLM recovery, no mutation. [Acceptance checks](llm-acceptance.json), [full LLM conversation and trace](ollama/conversation-05-multi-tool.json). Actual model request time: 482.4 seconds. |

## Five conversations

The first five exports use the explicitly identified deterministic planner and real services.
The separate Ollama export is the required LLM execution; protocol tests are not substituted
for it. The pipeline conversation includes a trusted operator confirmation.

1. [Run-specific dataset provenance](conversation-01-provenance.json): retrieves MLflow tags and
   resolves the selected run's exact Git/DVC pointer.
2. [Recent-run F1 comparison](conversation-02-selection.json): reports measured metrics and the
   limits of ranking a candidate by a single metric.
3. [Training trigger](conversation-03-trigger.json), [completion query](conversation-03-completion.json)
   and [terminal DAG/task states](airflow-success-run.json): distinguishes acceptance from success.
4. [Deployment health and logs](conversation-04-deployment.json): inspects replicas, Pods,
   restarts, Services, events, version metadata and logs.
5. [Four-integration diagnostic](conversation-05-multi-tool.json) and
   [actual Ollama counterpart](ollama/conversation-05-multi-tool.json): correlate the selected
   model, recorded dataset and serving identity; no retraining/restart is authorized.

## Deliberate failures and recovery

| Investigation | Observed root cause | Result and resolution |
| --- | --- | --- |
| Invalid training data | Explicit `inject_invalid=true` makes validation reject a missing target column | [Run/tasks](failure-validation-run.json): validation failed on try 2; downstream registration blocked. [Agent diagnosis](failure-validation-diagnosis.json) reads the actual failed-task logs. Remove the injection and rerun the approved DAG. |
| Impossible quality gate | Explicit `min_f1=1.01` cannot be met by the measured F1, approximately 0.9556 | [Run/tasks](failure-quality-gate-run.json): gate failed on try 2; registration blocked. [Agent diagnosis](failure-quality-gate-diagnosis.json) identifies the threshold/policy problem. Restore the normal 0.85 threshold. |
| Recovery after both training faults | Fault flags removed | [Normal recovery run](failure-recovery-run.json): all six tasks succeeded. |
| Incorrect readiness probe | Explicit operator injection uses a missing endpoint and produces HTTP 404 warnings | [Diagnosis](failure-readiness-diagnosis.json), [restored probe/rollout](failure-readiness-recovery.json): original path restored and two replicas ready. A stalled rollout does not prove a complete API outage. |

## Unexpected LLM runtime failures

[Runtime identity](ollama-runtime.json) records the actual Ollama version and model digest.
The constrained 4 GB WSL environment initially lacked enough memory; [the observed 500 response](ollama-memory-investigation.json)
was retained. Two GB of temporary Linux swap brought total swap to 3 GB. The native installation
was completed with the official image's CPU libraries. Earlier failed conversations under
`ollama/attempt-*.json` retain real failures: memory rejection, request timeouts, poor selection
with the 0.5B model, incomplete 1.5B planning and the schema grammar incompatibility. They are
investigations, not passing evidence. The invalid-grammar attempt was stopped by terminating
its model runner; its partial export is explicitly marked as a failure. Decoding patterns now
expand `\w` to explicit character ranges, with original Pydantic validation still enforced.
The later provenance attempt repeatedly prefixed a dataset pointer with a model-artifact path;
it was safely rejected, then stopped for correction. The mandatory lookup's schema now constrains
the commit/path to values actually returned by MLflow, and recovery drops obsolete model guesses.

The planner bounds requests, reports unavailable evidence, rejects invalid arguments and
discards unsupported model prose. Its schema-output recovery still asks the actual LLM to
choose a tool and arguments; it does not execute rules-planner calls to fill gaps.

Hosted GitHub Actions is separate from these local checks. Inspect the final pushed commit's
run and ensure the evaluator can access the repository. Earlier failed runs may remain in history.

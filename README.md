# Assignment 03 — Evidence-driven MLOps AI Agent

A Python agent that uses **actual DVC, MLflow, Apache Airflow 3, and Kubernetes APIs**.
It reports facts, uncertainties, recommendations, and underlying structured tool evidence.
Diagnostic questions never initiate retraining. The repository includes a reproducible Iris
training workflow, six-task Airflow DAG, containerized FastAPI inference, Kubernetes manifests,
pytest tests, GitHub Actions, and exported demonstrations.

See [architecture](architecture.md), [technical report](docs/report.md),
[tool permissions](docs/tools.md), and [requirement mapping](docs/requirements.md).

The [15 November review](docs/review-2026-11-15.md) records the requirement-by-requirement
audit, changes, current checks, and remaining verification. Historical 10 November exports
are preserved; new results live under [the dated evidence directory](docs/evidence/review-2026-11-15/).

## Install and configure

Use Linux/WSL2, Python **3.12**, Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and Docker Engine with Compose. Allow 10 GB free disk. Six GB or more RAM is recommended
for all services plus a local LLM; core services were also tested on 4 GB WSL.

```bash
git clone git@github.com:saadalit/assignment-03-mlops-agent.git
cd assignment-03-mlops-agent
uv sync --frozen
uv run python scripts/init_local.py
uv run dvc remote add --local -f -d storage "$PWD/.local/dvc-remote"
uv run python scripts/bootstrap_data.py
```

`init_local.py` creates ignored `.env` credentials and the Airflow password file without
printing secrets. Edit `.env` for another environment. Bootstrap regenerates the public Iris
dataset bundled with scikit-learn, checks both committed version hashes, and populates the
local DVC remote. No private dataset download is needed. Replace the local remote with durable
DVC storage for a shared deployment. Dataset v2 deterministically shuffles rows of v1.

`uv.lock` pins the complete environment. Explicit `pathspec==0.12.1` and `sqlalchemy==2.0.44`
constraints fix observed DVC 3.63 and MLflow 3.5 compatibility failures. Inference dependencies
have a separate hashed requirements file. On WSL with an NTFS workspace, keep the virtual
environment on the Linux filesystem for speed (`UV_PROJECT_ENVIRONMENT` plus a `.venv` link).

## Launch services

Run MLflow in a separate terminal and leave it running:

```bash
bash scripts/start_mlflow.sh
```

It uses port **5000**, SQLite, proxied artifacts, and an explicit Host allowlist. It listens
on the host interface so Airflow can reach it. This is a local development server; shared
infrastructure requires TLS and authentication.

In another terminal:

```bash
docker compose build airflow
docker compose up -d airflow kubernetes
bash scripts/kubectl.sh get nodes
```

k3s needs privileged Docker access for its **local** cluster. Airflow uses the official 3.1.5
image, LocalExecutor, a named volume, authenticated API v2, and one retry per task. Wait for
`docker compose ps` to show Airflow healthy. Its UI is `http://localhost:8080`, user `agent`,
with the ignored `AIRFLOW_PASSWORD` from `.env`. MLflow UI is `http://localhost:5000`.
The approved DAG is `training_pipeline`.

## Train and deploy inference

```bash
uv run dvc repro
uv run python -m training.train --workspace .local/runs/deploy --max-depth 3 --register --export
bash scripts/build_model.sh
uv run python scripts/deploy.py --confirm
uv run python scripts/configure_kube.py
uv run python scripts/smoke_inference.py
```

The build script tags the image with its full MLflow run ID and imports it into k3s.
`deploy.py --confirm` authorizes operator setup in `mlops`, configures the exact model/hash,
and waits for **two ready replicas**. A model must pass its quality gate before export.
The smoke script calls the real **ClusterIP Service** from the k3s node and exports
liveness, readiness, version, and prediction results. No public exposure is needed.

Prediction input is `{"features": [[5.1, 3.5, 1.4, 0.2]]}`. Rows require four finite Iris
measurements; batches are limited to 100. `/healthz`, `/readyz`, and `/version` separate
process health, model readiness, and provenance. Missing models fail readiness.

`configure_kube.py` creates ignored operator and **read-only agent** kubeconfigs. The agent
token expires after 24 hours; rerun the command to refresh it. An administrator may bind the
optional restarter Role to a separate identity when cluster writes are needed. Inference Pods
receive no Kubernetes API credentials.

## Agent interaction

The default `rules` planner supports the assignment workflows without an LLM download.
Use `--planner ollama` for supported LLM structured tool calling. Every response identifies
its planner. Neither mode invents results when tools are unavailable.

```bash
uv run mlops-agent "Which dataset version is current?"
uv run mlops-agent "Compare the last five training runs and recommend the best model based on F1"
uv run mlops-agent "Check Kubernetes deployment health and logs"
uv run mlops-agent "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary"
uv run mlops-agent "Trigger the model training pipeline" \
  --confirm airflow.trigger_training:training_pipeline
uv run mlops-agent "Report the training pipeline status"
uv run mlops-agent "Compare dataset v1.0-data and v2.0-data"
uv run mlops-agent "Reproduce dataset v1.0-data"
```

Include the actual 32-character MLflow run ID for run-specific dataset provenance. Trigger
acceptance is not completion: inspect the later DAG/task states. Dataset reproduction advice
uses an isolated checkout and does not change the current workspace.

A cluster restart requires `--confirm kubernetes.restart_deployment:mlops:prediction-api`,
the approved namespace/deployment, and an identity with patch permission. The trusted CLI/API
supplies confirmations; the model cannot add confirmations to tool arguments. The default
reader identity cannot patch the cluster even when an action is confirmed.

Optional local LLM, separate from the core stack:

```bash
docker run -d --name assignment3-ollama -p 127.0.0.1:11434:11434 \
  -v assignment3-ollama:/root/.ollama ollama/ollama:0.13.5
docker exec assignment3-ollama ollama pull qwen2.5:1.5b
uv run mlops-agent "List the DVC tracked datasets" --planner ollama
uv run mlops-agent "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary" --planner ollama
```

For an existing stopped container, use `docker start assignment3-ollama`. Stop it after
verification to free memory. Keep Airflow running for the mandatory four-tool scenario.

The model proposes JSON-schema tool calls via Ollama `/api/chat`. Final operational prose
is rendered from actual tool results. The small model is suitable for demonstration; complex
open-ended requests benefit from a stronger tool-capable model and more memory.
`OLLAMA_TIMEOUT_SECONDS` defaults to 180 to accommodate CPU cold starts. DVC-only questions
receive the five DVC tool definitions. Multi-tool requests initially receive the full registry;
subsequent turns advertise outstanding evidence tools to focus the small model.
The LLM chooses calls and receives their results through a bounded loop (12 turns, 24 proposed
calls, and a configurable `OLLAMA_MAX_SECONDS=600` total deadline). Duplicate calls are not
executed again. An incomplete mandatory scenario prompts the model to collect missing evidence;
it never falls back silently to rules. `planner_trace` records the serving model, token counts,
and proposed tool names/arguments (scrubbed). Full evidence is exported while model context uses bounded
excerpts. Dataset correlation requires the selected run's exact Git reference and pointer.
On a constrained CPU host, use `OLLAMA_TIMEOUT_SECONDS=360` and `OLLAMA_MAX_SECONDS=1200`.
The 4 GB WSL review environment needed 3 GB total swap to load the model alongside all services.
Stopping Airflow is unsuitable for the mandatory four-integration demonstration. Model responses
are capped by `OLLAMA_NUM_PREDICT=256`, and schema display titles are omitted to reduce overhead;
all input constraints remain enforced.

If the small model emits prose before completing the mandatory check or proposes invalid
arguments, planning switches to Ollama's supported JSON-schema response format. The LLM selects
the next function and its arguments from the outstanding tools; execution and authorization use
the same registry. The trace distinguishes `native_tools` and `json_schema` turns. Regex character
classes are expanded for Ollama 0.13's grammar compatibility; Pydantic remains the input authority.
The mandatory dataset lookup's schema uses the commit/pointer actually recorded by MLflow,
preventing the small model from substituting a model-artifact path. Recovery context retains
actual tool results and drops obsolete model prose and argument guesses.

To launch the authenticated agent HTTP interface:

```bash
uv run uvicorn agent.main:app --host 127.0.0.1 --port 8001
```

`POST /chat` takes `{"message": "...", "confirmations": []}` and requires
`Authorization: Bearer <AGENT_API_TOKEN>` from `.env`. The agent API is separate from the
inference API. Local development services should remain private.

## Tests, CI and demonstration

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=agent --cov=api --cov=training --cov-report=term-missing
uv run python scripts/validate_config.py
uv run dvc dag
uv run python scripts/reproduce_dataset.py v1.0-data --remote .local/dvc-remote
uv run python scripts/reproduce_dataset.py v2.0-data --remote .local/dvc-remote
```

Ordinary tests mock network services. Tests cover tool execution/selection, multi-tool requests,
timeouts/failures, unavailable evidence, LLM protocol, missing provenance, namespace and
confirmation enforcement, validation, failed quality gates, safe logs and inference. GitHub
Actions runs lint, formatting, pytest/coverage and configuration checks on PRs and main pushes.
CI needs neither external credentials nor a cluster.

With all real services running and inference deployed:

```bash
uv run python scripts/demo.py --planner rules --trigger --confirm-failures --output docs/evidence/review-2026-11-15
OLLAMA_TIMEOUT_SECONDS=360 OLLAMA_MAX_SECONDS=1200 TOOL_TIMEOUT_SECONDS=45 uv run python scripts/demo.py --planner ollama --only-multi --output docs/evidence/review-2026-11-15/ollama
uv run python -m scripts.verify_live --output docs/evidence/review-2026-11-15
uv run python scripts/smoke_inference.py
uv run python scripts/readiness_failure.py --confirm
```

`--trigger` authorizes a real training run and awaits completion. `--confirm-failures` authorizes
two deliberate failures: invalid data and an impossible quality threshold. Registration must
remain blocked. The readiness script injects an incorrect probe path, captures diagnosis,
then restores the original path and verifies recovery.

To demonstrate comparison across distinct MLflow experiments, create a separately tracked
baseline using the existing training workflow, then rerun the read-only verification:

```bash
MLFLOW_EXPERIMENT_NAME=iris-classification-baseline uv run python -m training.train --workspace .local/runs/review-baseline --max-depth 1
uv run python -m scripts.verify_live --output docs/evidence/review-2026-11-15
```

[docs/evidence](docs/evidence/) contains real service exports, meeting the PDF's screenshots
**or exported evidence** requirement. [The report](docs/report.md) explains the five scenarios,
failure investigations, known limitations and engineering defense. Old run IDs and timestamps
are historical evidence; regenerating demonstrations creates new ones.

## Stop services

```bash
docker compose stop
docker stop assignment3-k3s  # also stops a cluster started manually during setup
docker stop assignment3-ollama  # if started
```

Stop foreground MLflow with Ctrl+C. Volumes and `.local` preserve state. Never commit `.env`,
`config.local`, kubeconfigs, tokens, model binaries, service databases, or private logs.
The assignment PDF is retained locally only.

## Verify the repository submission

```bash
uv run python -m scripts.verify_local
uv run python -m scripts.audit_repository --history
```

`verify_local` saves actual commands, exit codes, timestamps and output for the CI checks.
`audit_repository` checks submission paths, configured local secrets and recognizable credentials,
including reachable Git history. It reports filenames/object IDs without printing credentials.
Submit the GitHub repository link. Source, frozen dependencies, documentation and scrubbed
evidence belong in Git; local credentials, databases, runtime caches and model binaries remain
ignored. No archive is required.

GitHub Actions runs on pull requests and main pushes. Earlier failed runs can remain in history;
the final submitted commit should have a passing run. Local CI-equivalent checks do not establish
a hosted Actions result for changes that have not been pushed.

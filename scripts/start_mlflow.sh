#!/usr/bin/env bash
set -euo pipefail
mkdir -p .local/mlflow
exec .venv/bin/mlflow server \
  --host 0.0.0.0 --port 5000 --workers 1 \
  --backend-store-uri sqlite:///.local/mlflow/tracking.db \
  --artifacts-destination .local/mlflow/artifacts \
  --allowed-hosts 'localhost:*,127.0.0.1:*,host.docker.internal:*'

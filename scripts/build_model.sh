#!/usr/bin/env bash
set -euo pipefail
model_run_id=$(.venv/bin/python -c 'import json; print(json.load(open("models/current/metadata.json"))["run_id"])')
docker build -t "mlops-inference:$model_run_id" .
docker save "mlops-inference:$model_run_id" | docker exec -i assignment3-k3s ctr images import -
echo "Built and imported inference image for run $model_run_id"

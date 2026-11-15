#!/usr/bin/env bash
set -euo pipefail
model_run_id=$(.venv/bin/python -c 'import json; print(json.load(open("models/current/metadata.json"))["run_id"])')
docker build -t "mlops-inference:$model_run_id" .
baked_run_id=$(docker run --rm --entrypoint python "mlops-inference:$model_run_id" \
  -c 'import json; print(json.load(open("/app/models/current/metadata.json"))["run_id"])')
if [[ "$baked_run_id" != "$model_run_id" ]]; then
  echo "Model export changed during build; refusing to import a mislabeled image. Rerun after export finishes." >&2
  exit 1
fi
docker save "mlops-inference:$model_run_id" | docker exec -i assignment3-k3s ctr images import -
echo "Built and imported inference image for run $model_run_id"

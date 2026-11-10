import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

model = None
metadata = {}


@asynccontextmanager
async def lifespan(app):
    global model, metadata
    root = Path(os.getenv("MODEL_DIR", "models/current"))
    if (root / "model.joblib").exists() and (root / "metadata.json").exists():
        candidate_metadata = json.loads((root / "metadata.json").read_text())
        if candidate_metadata.get("quality_gate") != "passed":
            raise RuntimeError("Model has not passed the quality gate")
        for env, key in (("MODEL_RUN_ID", "run_id"), ("DATASET_MD5", "dataset_md5")):
            if os.getenv(env) and os.environ[env] != candidate_metadata.get(key):
                raise RuntimeError("Model artifact provenance disagrees with deployment configuration")
        model, metadata = joblib.load(root / "model.joblib"), candidate_metadata
    yield
    model, metadata = None, {}


app = FastAPI(title="Iris prediction API", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    features: list[list[float]] = Field(min_length=1, max_length=100)


@app.get("/healthz")
def health():
    return {"status": "alive"}


@app.get("/readyz")
def ready():
    if model is None:
        raise HTTPException(503, "Model artifact unavailable")
    return {"status": "ready", "run_id": metadata["run_id"],
            "dataset_md5": metadata["dataset_md5"]}


@app.get("/version")
def version():
    return {"app_version": os.getenv("APP_VERSION", "1.0.0"),
            "model_run_id": metadata.get("run_id"), "dataset_md5": metadata.get("dataset_md5"),
            "git_commit": metadata.get("dataset_git_commit")}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(503, "Model artifact unavailable")
    if any(len(row) != 4 for row in request.features):
        raise HTTPException(422, "Each row must contain exactly four Iris measurements")
    return {"predictions": model.predict(request.features).tolist(), "run_id": metadata["run_id"]}

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def validate(data_path="data/iris.csv", workspace=".local/runs/manual", inject_invalid=False):
    path, work = Path(data_path), Path(workspace)
    if inject_invalid:
        raise ValueError("DATA_VALIDATION_FAILED: deliberately injected missing target column")
    df = pd.read_csv(path)
    if list(df.columns) != FEATURES + ["target"]:
        raise ValueError("DATA_VALIDATION_FAILED: unexpected columns")
    if len(df) < 30 or df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("DATA_VALIDATION_FAILED: insufficient rows, missing or non-finite values")
    if not set(df["target"].unique()) == {0, 1, 2}:
        raise ValueError("DATA_VALIDATION_FAILED: expected all three Iris classes")
    pointer = Path(str(path) + ".dvc")
    recorded = yaml.safe_load(pointer.read_text())["outs"][0]["md5"]
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != recorded:
        raise ValueError("DATA_VALIDATION_FAILED: data differs from recorded DVC content address")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    committed = yaml.safe_load(
        subprocess.run(
            ["git", "show", f"{commit}:{pointer.as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    )["outs"][0]["md5"]
    if committed != actual:
        raise ValueError("DATA_VALIDATION_FAILED: DVC pointer is not committed to Git")
    work.mkdir(parents=True, exist_ok=True)
    info = {
        "workspace": str(work),
        "data_path": str(path),
        "dataset_md5": actual,
        "dataset_git_commit": commit,
        "dataset_pointer": str(pointer),
        "rows": len(df),
        "dataset_version": f"md5:{actual}",
    }
    (work / "validation.json").write_text(json.dumps(info, indent=2))
    return info


def preprocess(info):
    df = pd.read_csv(info["data_path"])
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES].to_numpy(),
        df["target"].to_numpy(),
        test_size=0.3,
        random_state=42,
        stratify=df["target"],
    )
    np.savez(
        Path(info["workspace"]) / "split.npz",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
    return info


def train(info, max_depth=3):
    split = np.load(Path(info["workspace"]) / "split.npz")
    model = RandomForestClassifier(n_estimators=60, max_depth=max_depth, random_state=42, n_jobs=1)
    model.fit(split["x_train"], split["y_train"])
    joblib.dump(model, Path(info["workspace"]) / "model.joblib")
    return {**info, "max_depth": max_depth}


def evaluate(info, min_f1=0.85):
    work = Path(info["workspace"])
    split = np.load(work / "split.npz")
    model = joblib.load(work / "model.joblib")
    predictions = model.predict(split["x_test"])
    metrics = {
        "accuracy": float(accuracy_score(split["y_test"], predictions)),
        "f1": float(f1_score(split["y_test"], predictions, average="macro")),
    }
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2))
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification"))
    with mlflow.start_run(run_name=work.name) as run:
        mlflow.log_params(
            {
                "max_depth": info["max_depth"],
                "n_estimators": 60,
                "split_seed": 42,
                "test_size": 0.3,
                "min_f1": min_f1,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                k: info[k]
                for k in ("dataset_md5", "dataset_git_commit", "dataset_pointer", "dataset_version")
            }
        )
        mlflow.set_tags(
            {"quality_gate": "pending", "evaluation_split": "stratified-seed42-test0.3"}
        )
        mlflow.log_artifact(str(work / "validation.json"))
        mlflow.log_artifact(str(work / "model.joblib"))
        mlflow.sklearn.log_model(model, name="model", input_example=split["x_test"][:2])
        result = {
            **info,
            "metrics": metrics,
            "min_f1": min_f1,
            "run_id": run.info.run_id,
            "app_version": "1.0.0",
        }
        (work / "evaluation.json").write_text(json.dumps(result, indent=2))
        mlflow.log_dict(result, "evaluation.json")
    return result


def quality_gate(info):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    passed = info["metrics"]["f1"] >= info["min_f1"]
    MlflowClient().set_tag(info["run_id"], "quality_gate", "passed" if passed else "failed")
    if not passed:
        raise ValueError(
            f"QUALITY_GATE_FAILED: f1={info['metrics']['f1']:.4f} < "
            f"required={info['min_f1']}; promotion blocked"
        )
    return {**info, "quality_gate": "passed"}


def register_candidate(info):
    # Downstream of quality_gate only; failure must block registration/export.
    if info.get("quality_gate") != "passed":
        raise ValueError("Promotion requires a passed quality gate")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    version = mlflow.register_model(f"runs:/{info['run_id']}/model", "iris-classifier")
    return {**info, "registry_version": version.version}


def export_model(info, destination="models/current"):
    if info.get("quality_gate") != "passed":
        raise ValueError("Export requires a passed quality gate")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(info["workspace"]) / "model.joblib", target / "model.joblib")
    (target / "metadata.json").write_text(json.dumps(info, indent=2))
    return info

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import joblib
import pytest
import yaml
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from api import main as inference
from training import stages


def test_training_validation_and_preprocessing_enforce_provenance(git_repo):
    df = load_iris(as_frame=True).frame
    df.columns = stages.FEATURES + ["target"]
    df.to_csv("data/iris.csv", index=False)
    md5 = hashlib.md5(Path("data/iris.csv").read_bytes()).hexdigest()
    Path("data/iris.csv.dvc").write_text(
        yaml.safe_dump({"outs": [{"md5": md5, "path": "iris.csv"}]})
    )
    subprocess.run(["git", "add", "data/iris.csv.dvc"], check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "data"], check=True, capture_output=True)
    info = stages.preprocess(stages.validate(workspace="work"))
    assert Path(info["workspace"], "split.npz").exists()
    trained = stages.train(info)
    assert Path(trained["workspace"], "model.joblib").exists()
    Path("data/iris.csv").write_text(Path("data/iris.csv").read_text() + "\n")
    with pytest.raises(ValueError, match="DVC content address"):
        stages.validate()


def test_failed_quality_gate_blocks_export_and_registration(monkeypatch):
    client = Mock()
    monkeypatch.setattr(stages, "MlflowClient", lambda: client)
    monkeypatch.setattr(stages.mlflow, "set_tracking_uri", lambda uri: None)
    info = {"run_id": "a" * 32, "metrics": {"f1": 0.7}, "min_f1": 0.85}
    with pytest.raises(ValueError, match="promotion blocked"):
        stages.quality_gate(info)
    client.set_tag.assert_called_once_with("a" * 32, "quality_gate", "failed")
    for fn in (stages.export_model, stages.register_candidate):
        with pytest.raises(ValueError, match="passed quality gate"):
            fn(info)


def test_inference_readiness_fails_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(inference.app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.post("/predict", json={"features": [[1, 2, 3, 4]]}).status_code == 503


def test_inference_predicts_and_rejects_bad_shape(tmp_path, monkeypatch):
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=2, random_state=42).fit(iris.data, iris.target)
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"run_id": "a" * 32, "quality_gate": "passed", "dataset_md5": "abc"})
    )
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(inference.app) as client:
        assert client.get("/readyz").json()["dataset_md5"] == "abc"
        response = client.post("/predict", json={"features": [[5.1, 3.5, 1.4, 0.2]]})
        assert response.status_code == 200 and response.json()["predictions"] == [0]
        assert client.post("/predict", json={"features": [[1, 2]]}).status_code == 422
        assert client.post("/predict", json={"features": []}).status_code == 422


def test_inference_rejects_mislabeled_artifact(tmp_path, monkeypatch):
    (tmp_path / "model.joblib").touch()
    (tmp_path / "metadata.json").write_text(
        json.dumps({"quality_gate": "passed", "run_id": "actual"})
    )
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_RUN_ID", "different")
    with pytest.raises(RuntimeError, match="provenance disagrees"):
        with TestClient(inference.app):
            pass


def test_agent_http_api_requires_credentials(monkeypatch, tmp_path):
    from agent.main import app

    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        assert client.post("/chat", json={"message": "best model"}).status_code == 503
        monkeypatch.setenv("AGENT_API_TOKEN", "private-test-token")
        assert client.post("/chat", json={"message": "best model"}).status_code == 401


def test_agent_http_accepts_correct_token_and_rejects_wrong_token(monkeypatch, tmp_path):
    from agent import main
    from agent.executor import Executor, ToolSpec
    from agent.orchestrator import Orchestrator
    from agent.schemas import Empty

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_API_TOKEN", "private-test-token")
    monkeypatch.setenv("AGENT_PLANNER", "rules")
    ex = Executor(
        [
            ToolSpec("dvc.list_datasets", "datasets", Empty, lambda: {"datasets": []}),
            ToolSpec("dvc.status", "status", Empty, lambda: {"changes": {}}),
        ]
    )
    monkeypatch.setattr(main, "Orchestrator", lambda settings: Orchestrator(settings, ex))
    with TestClient(main.app) as client:
        payload = {"message": "Which dataset version is current?"}
        assert (
            client.post(
                "/chat",
                json=payload,
                headers={
                    "Authorization": "Bearer incorrect-token",
                },
            ).status_code
            == 401
        )
        response = client.post(
            "/chat",
            json=payload,
            headers={
                "Authorization": "Bearer private-test-token",
            },
        )
        assert response.status_code == 200
        assert [r["tool"] for r in response.json()["evidence"]] == [
            "dvc.list_datasets",
            "dvc.status",
        ]

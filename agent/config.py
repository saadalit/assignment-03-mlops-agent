import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    repo: Path = field(default_factory=lambda: Path(os.getenv("MLOPS_REPO", ".")).resolve())
    mlflow_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    experiment: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification")
    )
    airflow_url: str = field(
        default_factory=lambda: os.getenv("AIRFLOW_URL", "http://127.0.0.1:8080")
    )
    airflow_dag: str = field(
        default_factory=lambda: os.getenv("AIRFLOW_DAG_ID", "training_pipeline")
    )
    namespace: str = field(default_factory=lambda: os.getenv("KUBE_NAMESPACE", "mlops"))
    selector: str = field(
        default_factory=lambda: os.getenv("KUBE_APP_SELECTOR", "app=prediction-api")
    )
    timeout: float = field(default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_SECONDS", "15")))
    planner: str = field(default_factory=lambda: os.getenv("AGENT_PLANNER", "rules"))
    max_model_age_days: float | None = field(
        default_factory=lambda: float(os.environ["MAX_MODEL_AGE_DAYS"])
        if os.getenv("MAX_MODEL_AGE_DAYS")
        else None
    )

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls()

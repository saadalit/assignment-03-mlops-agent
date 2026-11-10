from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Empty(StrictModel):
    pass


class DatasetRef(StrictModel):
    ref: str = Field(default="HEAD", min_length=1, max_length=100, pattern=r"^[\w./-]+$")
    path: str = Field(default="data/iris.csv.dvc", pattern=r"^[\w./-]+\.dvc$")


class DatasetCompare(DatasetRef):
    other_ref: str = Field(min_length=1, max_length=100, pattern=r"^[\w./-]+$")


class RunQuery(StrictModel):
    experiment: str = Field(default="iris-classification", min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=100)
    days: int = Field(default=30, ge=1, le=3650)


class BestRun(RunQuery):
    metric: str = Field(default="f1", pattern=r"^[a-zA-Z][\w.-]{0,63}$")
    mode: Literal["max", "min"] = "max"


class RunId(StrictModel):
    run_id: str = Field(pattern=r"^[a-fA-F0-9]{32}$")


class CompareRuns(StrictModel):
    run_ids: list[str] = Field(min_length=2, max_length=20)


class RegistryQuery(StrictModel):
    name: str = Field(default="iris-classifier", pattern=r"^[\w.-]{1,100}$")


class DagQuery(StrictModel):
    dag_id: str = Field(default="training_pipeline", pattern=r"^[\w.-]{1,100}$")
    limit: int = Field(default=5, ge=1, le=100)


class DagRun(DagQuery):
    run_id: str = Field(min_length=1, max_length=250)


class TaskLog(DagRun):
    task_id: str = Field(pattern=r"^[\w.-]{1,100}$")
    try_number: int = Field(default=1, ge=1, le=100)


class TriggerDag(StrictModel):
    dag_id: str = Field(default="training_pipeline", pattern=r"^[\w.-]{1,100}$")


class Namespace(StrictModel):
    namespace: str = Field(default="mlops", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")


class Deployment(Namespace):
    name: str = Field(default="prediction-api", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")


class PodLog(Namespace):
    pod: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]{0,252}$")
    tail_lines: int = Field(default=80, ge=1, le=200)


class ToolResult(BaseModel):
    tool: str
    status: Literal["ok", "error", "timeout", "denied", "invalid"]
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float
    request_id: str


class AgentRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    confirmations: list[str] = Field(default_factory=list, max_length=10)


class AgentResponse(BaseModel):
    request_id: str
    planner: str
    facts: list[str]
    uncertainties: list[str]
    recommendations: list[str]
    evidence: list[ToolResult]

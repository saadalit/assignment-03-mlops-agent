import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agent.tools.mlflow_tools import MLflowTools


def run(id, f1, status="FINISHED", extra=None):
    return NS(
        info=NS(
            run_id=id * 32,
            experiment_id="1",
            status=status,
            start_time=int(time.time() * 1000),
            end_time=1,
        ),
        data=NS(
            metrics={"f1": f1, **(extra or {})},
            params={"max_depth": "3"},
            tags={"dataset_md5": "abcd"},
        ),
    )


def test_best_run_ignores_failed_and_nan(settings):
    client = Mock()
    client.get_experiment_by_name.return_value = NS(experiment_id="1")
    client.search_runs.return_value = [
        run("a", 1, "FAILED"),
        run("b", float("nan")),
        run("c", 0.91),
        run("d", 0.82),
    ]
    result = MLflowTools(settings, client).best_run()
    assert result["candidate"]["run_id"] == "c" * 32
    assert "production readiness" in result["caution"]
    assert "start_time >=" in client.search_runs.call_args.kwargs["filter_string"]


def test_minimization_metric(settings):
    client = Mock()
    client.get_experiment_by_name.return_value = NS(experiment_id="1")
    client.search_runs.return_value = [
        run("a", 0.8, extra={"rmse": 0.3}),
        run("b", 0.7, extra={"rmse": 0.2}),
    ]
    result = MLflowTools(settings, client).best_run(metric="rmse", mode="min")
    assert result["candidate"]["run_id"] == "b" * 32


def test_missing_experiment_does_not_fabricate(settings):
    client = Mock()
    client.get_experiment_by_name.return_value = None
    assert MLflowTools(settings, client).best_run()["candidate"] is None


def test_compare_reads_metrics_params_artifacts(settings):
    client = Mock()
    a, b = run("a", 0.8), run("b", 0.9)
    b.info.experiment_id = "2"
    client.get_run.side_effect = [a, b]
    client.list_artifacts.return_value = [NS(path="model.joblib", is_dir=False, file_size=100)]
    results = MLflowTools(settings, client).compare_runs(["a" * 32, "b" * 32])["runs"]
    assert results[0]["artifacts"][0]["path"] == "model.joblib"
    assert results[1]["metrics"]["f1"] == 0.9
    assert {r["experiment_id"] for r in results} == {"1", "2"}
    with pytest.raises(ValueError):
        MLflowTools(settings, client).compare_runs(["bad/id", "b" * 32])

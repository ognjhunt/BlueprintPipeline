from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _find_step(steps: list[dict[str, object]], name: str) -> dict[str, object]:
    for step in steps:
        if name in step:
            return step[name]
    raise AssertionError(f"Step '{name}' not found")


@pytest.fixture()
def training_pipeline_workflow_path() -> Path:
    return Path(__file__).resolve().parents[1] / "workflows" / "training-pipeline.yaml"


@pytest.fixture()
def training_pipeline_workflow(training_pipeline_workflow_path: Path) -> dict[str, object]:
    return yaml.safe_load(training_pipeline_workflow_path.read_text())


def test_training_pipeline_event_parsing_requires_job_id_and_output_path(
    training_pipeline_workflow: dict[str, object],
) -> None:
    steps = training_pipeline_workflow["main"]["steps"]
    extract_event = _find_step(steps, "extract_event")
    fallback_steps = extract_event["switch"][-1]["steps"]
    return_invalid_event = _find_step(fallback_steps, "return_invalid_event")
    error_payload = return_invalid_event["return"]["error"]

    assert error_payload["code"] == "invalid_event_payload"
    assert error_payload["required_fields"] == ["job_id", "output_path"]
    assert "job_id" in error_payload["message"] and "output_path" in error_payload["message"]


def test_training_pipeline_missing_job_id_payload(
    training_pipeline_workflow: dict[str, object],
) -> None:
    steps = training_pipeline_workflow["main"]["steps"]
    validate_inputs = _find_step(steps, "validate_inputs")
    missing_job_id_steps = validate_inputs["switch"][0]["steps"]
    return_missing_job_id = _find_step(missing_job_id_steps, "return_missing_job_id")
    error_payload = return_missing_job_id["return"]["error"]

    assert error_payload["code"] == "missing_job_id"
    assert error_payload["required_fields"] == ["job_id"]
    assert "job_id" in error_payload["message"]


def test_training_pipeline_missing_output_path_payload(
    training_pipeline_workflow: dict[str, object],
) -> None:
    steps = training_pipeline_workflow["main"]["steps"]
    validate_inputs = _find_step(steps, "validate_inputs")
    missing_output_steps = validate_inputs["switch"][1]["steps"]
    return_missing_output_path = _find_step(missing_output_steps, "return_missing_output_path")
    error_payload = return_missing_output_path["return"]["error"]

    assert error_payload["code"] == "missing_output_path"
    assert error_payload["required_fields"] == ["output_path"]
    assert "output_path" in error_payload["message"]


def test_training_pipeline_launch_and_monitoring_steps(
    training_pipeline_workflow: dict[str, object],
) -> None:
    steps = training_pipeline_workflow["main"]["steps"]
    init_step = _find_step(steps, "init")
    init_assignment_keys = {next(iter(item.keys())) for item in init_step["assign"]}
    assert "trainingJobName" in init_assignment_keys
    assert "trainingTimeoutSeconds" in init_assignment_keys

    set_training_markers = _find_step(steps, "set_training_markers")
    marker_assignment = set_training_markers["assign"][0]["trainingFailedMarker"]
    assert ".failed" in marker_assignment

    run_training_job = _find_step(steps, "run_training_job")
    run_args = run_training_job["try"]["args"]
    assert "trainingJobName" in run_args["name"]

    emit_training_metrics_start = _find_step(steps, "emit_training_metrics_start")
    metrics_timeout = emit_training_metrics_start["args"]["data"]["timeout_seconds"]
    assert metrics_timeout == "${trainingTimeoutSeconds}"

    wait_for_training = _find_step(steps, "wait_for_training")
    assert wait_for_training["call"] == "googleapis.run.v2.projects.locations.jobs.executions.get"

    training_status_switch = _find_step(steps, "training_status_switch")
    switch_conditions = [case["condition"] for case in training_status_switch["switch"]]
    assert any("FAILED" in condition for condition in switch_conditions)
    assert any("SUCCEEDED" in condition for condition in switch_conditions)

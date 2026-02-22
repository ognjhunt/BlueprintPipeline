from pathlib import Path

import pytest

from tools.cost_tracking import estimate, tracker

pytestmark = pytest.mark.usefixtures("add_repo_to_path")


@pytest.mark.unit
def test_estimate_gpu_costs_handles_missing_steps_and_rates() -> None:
    config = estimate.EstimateConfig(
        instance_rates={"custom": {"hourly_rate": 2.0, "gpu_count": 2}},
        step_config={
            "step-a": {"duration_minutes": 30, "instance_type": "custom"},
            "step-b": {"duration_hours": 2, "instance_type": "missing-rate"},
        },
        default_instance_type="custom",
    )

    summary = estimate.estimate_gpu_costs(["step-a", "step-b", "step-c"], config)

    assert summary.total_gpu_hours == pytest.approx(1.0)
    assert summary.total_cost == pytest.approx(1.0)
    assert summary.missing_steps == ["step-c"]
    assert summary.missing_rates == ["missing-rate"]
    assert summary.steps[0].step == "step-a"


@pytest.mark.unit
def test_format_estimate_summary_includes_missing_sections() -> None:
    summary = estimate.EstimateSummary(
        total_gpu_hours=1.0,
        total_cost=2.0,
        steps=[],
        missing_steps=["step-x"],
        missing_rates=["rate-y"],
    )

    output = estimate.format_estimate_summary(summary)

    assert "Missing step configs: step-x" in output
    assert "Missing rate configs: rate-y" in output


@pytest.mark.unit
def test_cost_tracker_aggregates_scene_costs(tmp_path: Path) -> None:
    cost_tracker = tracker.CostTracker(data_dir=tmp_path, enable_logging=False)

    gemini_cost = cost_tracker.track_gemini_call("scene-1", tokens_in=1000, tokens_out=500)
    run_cost = cost_tracker.track_compute(
        "scene-1",
        "stage1-job",
        duration_seconds=120,
        vcpu_count=2,
        memory_gb=4.0,
    )
    build_cost = cost_tracker.track_cloud_build("scene-1", "build-1", duration_minutes=3)
    storage_cost = cost_tracker.track_storage("scene-1", bytes_written=1024 ** 3, operation_count=2)
    geniesim_cost = cost_tracker.track_geniesim_job("scene-1", "job-1", episode_count=5)

    breakdown = cost_tracker.get_scene_cost("scene-1")

    expected_total = gemini_cost + run_cost + build_cost + storage_cost + geniesim_cost
    assert breakdown.total == pytest.approx(expected_total)
    assert breakdown.gemini == pytest.approx(gemini_cost)
    assert breakdown.cloud_run == pytest.approx(run_cost)
    assert breakdown.cloud_build == pytest.approx(build_cost)
    assert breakdown.gcs_storage == pytest.approx(storage_cost)
    assert breakdown.geniesim == pytest.approx(geniesim_cost)
    assert breakdown.by_job["stage1-job"] == pytest.approx(run_cost)


@pytest.mark.unit
def test_get_period_cost_includes_recent_entries(tmp_path: Path) -> None:
    cost_tracker = tracker.CostTracker(data_dir=tmp_path, enable_logging=False)
    cost_tracker.track_gemini_call("scene-1", tokens_in=1000, tokens_out=0)
    cost_tracker.track_cloud_build("scene-2", "build-1", duration_minutes=1)

    report = cost_tracker.get_period_cost(days=1)

    assert report["scene_count"] == 2
    assert report["total"] > 0
    assert report["top_scenes"][0]["scene_id"] in {"scene-1", "scene-2"}


@pytest.mark.unit
def test_cost_tracker_rejects_missing_pricing_in_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.delenv("GENIESIM_JOB_COST", raising=False)
    monkeypatch.delenv("GENIESIM_EPISODE_COST", raising=False)

    with pytest.raises(RuntimeError, match="Genie Sim pricing environment variables"):
        tracker.CostTracker(data_dir=tmp_path, enable_logging=False)


@pytest.mark.unit
def test_cost_tracker_rejects_missing_pricing_config_in_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("GENIESIM_JOB_COST", "1.25")
    monkeypatch.setenv("GENIESIM_EPISODE_COST", "0.03")
    monkeypatch.delenv("COST_TRACKING_PRICING_JSON", raising=False)
    monkeypatch.delenv("COST_TRACKING_PRICING_PATH", raising=False)

    with pytest.raises(RuntimeError, match="Cost tracking defaults are dev-only"):
        tracker.CostTracker(data_dir=tmp_path, enable_logging=False)


@pytest.mark.unit
def test_cost_tracker_uses_defaults_in_non_prod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PIPELINE_ENV", raising=False)
    monkeypatch.delenv("GENIESIM_JOB_COST", raising=False)
    monkeypatch.delenv("GENIESIM_EPISODE_COST", raising=False)

    cost_tracker = tracker.CostTracker(data_dir=tmp_path, enable_logging=False)

    assert cost_tracker.pricing["geniesim_job"] == pytest.approx(
        tracker.DEFAULT_PRICING["geniesim_job"]
    )
    assert cost_tracker.pricing["geniesim_episode"] == pytest.approx(
        tracker.DEFAULT_PRICING["geniesim_episode"]
    )


@pytest.mark.unit
def test_cost_tracker_raises_on_scene_quota_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COST_HARD_QUOTA_PER_SCENE_USD", "0.001")

    cost_tracker = tracker.CostTracker(data_dir=tmp_path, enable_logging=False)

    with pytest.raises(tracker.CostQuotaExceeded, match="Scene cost hard quota exceeded"):
        cost_tracker.track_gemini_call("scene-1", tokens_in=1000, tokens_out=0)


@pytest.mark.unit
def test_cost_tracker_raises_on_total_quota_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COST_HARD_QUOTA_TOTAL_USD", "0.001")

    cost_tracker = tracker.CostTracker(data_dir=tmp_path, enable_logging=False)

    with pytest.raises(tracker.CostQuotaExceeded, match="Total cost hard quota exceeded"):
        cost_tracker.track_gemini_call("scene-1", tokens_in=1000, tokens_out=0)


@pytest.mark.unit
def test_cost_tracker_rejects_invalid_quota_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COST_HARD_QUOTA_TOTAL_USD", "0")

    with pytest.raises(ValueError, match="COST_HARD_QUOTA_TOTAL_USD must be greater than zero"):
        tracker.CostTracker(data_dir=tmp_path, enable_logging=False)

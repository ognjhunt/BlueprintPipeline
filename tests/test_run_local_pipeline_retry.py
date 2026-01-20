import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest

from tools.error_handling.retry import RetryConfig, RetryableError
from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep, StepResult


def _make_runner(tmp_path, *, verbose=True):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=verbose,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )
    runner.retry_config = RetryConfig(
        max_retries=2,
        base_delay=0.0,
        max_delay=0.0,
        backoff_factor=1.0,
        jitter=False,
    )
    return runner


def test_run_step_retries_transient_failure(tmp_path, capsys, monkeypatch):
    runner = _make_runner(tmp_path, verbose=True)
    attempts = {"count": 0}

    def flaky_scale():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RetryableError("transient")
        return StepResult(
            step=PipelineStep.SCALE,
            success=True,
            duration_seconds=0,
            message="ok",
        )

    monkeypatch.setattr(runner, "_run_scale", flaky_scale)

    result = runner._run_step(PipelineStep.SCALE)
    output = capsys.readouterr().out

    assert attempts["count"] == 2
    assert result.success
    assert "scale retry 1/2" in output


def test_run_step_retryable_failure_returns_step_result(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, verbose=False)
    runner.retry_config = RetryConfig(
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        backoff_factor=1.0,
        jitter=False,
    )

    def failing_scale():
        raise RetryableError("still failing")

    monkeypatch.setattr(runner, "_run_scale", failing_scale)

    result = runner._run_step(PipelineStep.SCALE)

    assert not result.success
    assert result.message.startswith("Error:")

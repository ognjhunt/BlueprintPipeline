import json
from pathlib import Path

import pytest

from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep, StepResult


@pytest.fixture
def local_pipeline_runner(mock_scene_dir, mock_regen3d_output):
    (mock_scene_dir / "seg").mkdir(exist_ok=True)
    return LocalPipelineRunner(
        scene_dir=mock_scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )


def _write_placeholder(path: Path, payload=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        path.write_text(json.dumps(payload))
    else:
        path.write_text("ok")


def test_resume_skips_completed_steps(local_pipeline_runner, monkeypatch):
    steps = [PipelineStep.REGEN3D, PipelineStep.SIMREADY, PipelineStep.USD]
    run_calls = []

    def fake_run_step(step):
        run_calls.append(step)
        if step == PipelineStep.USD:
            return StepResult(
                step=step,
                success=False,
                duration_seconds=0,
                message="boom",
            )

        expected_paths = local_pipeline_runner._expected_output_paths(step)
        for path in expected_paths:
            if path.name == "scene_manifest.json":
                _write_placeholder(path, {"objects": [], "scene": {}})
            elif path.name.endswith(".json"):
                _write_placeholder(path, {"ok": True})
            else:
                _write_placeholder(path)

        return StepResult(
            step=step,
            success=True,
            duration_seconds=0,
            message="ok",
            outputs={"expected_outputs": [str(path) for path in expected_paths]},
        )

    monkeypatch.setattr(local_pipeline_runner, "_run_step", fake_run_step)

    success = local_pipeline_runner.run(steps=steps)

    assert not success
    assert run_calls == steps

    resume_runner = LocalPipelineRunner(
        scene_dir=local_pipeline_runner.scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    run_calls = []

    def fake_run_step_resume(step):
        run_calls.append(step)
        expected_paths = resume_runner._expected_output_paths(step)
        for path in expected_paths:
            if not path.exists():
                if path.name == "scene_manifest.json":
                    _write_placeholder(path, {"objects": [], "scene": {}})
                elif path.name.endswith(".json"):
                    _write_placeholder(path, {"ok": True})
                else:
                    _write_placeholder(path)
        return StepResult(
            step=step,
            success=True,
            duration_seconds=0,
            message="ok",
            outputs={"expected_outputs": [str(path) for path in expected_paths]},
        )

    monkeypatch.setattr(resume_runner, "_run_step", fake_run_step_resume)

    success = resume_runner.run(steps=steps, resume_from=PipelineStep.REGEN3D)

    assert success
    assert run_calls == [PipelineStep.USD]

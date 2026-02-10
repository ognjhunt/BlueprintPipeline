import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fixtures.generate_mock_regen3d import generate_mock_regen3d
from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep, StepResult


@pytest.fixture
def local_pipeline_runner(temp_test_dir):
    scene_id = "test_scene"
    generate_mock_regen3d(
        output_dir=temp_test_dir,
        scene_id=scene_id,
        environment_type="kitchen",
    )
    scene_dir = temp_test_dir / "scenes" / scene_id
    (scene_dir / "seg").mkdir(exist_ok=True)
    return LocalPipelineRunner(
        scene_dir=scene_dir,
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
    marker_path = local_pipeline_runner.assets_dir / ".regen3d_complete"

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
    monkeypatch.setattr(local_pipeline_runner, "_apply_quality_gates", lambda step, result, **kwargs: result)

    success = local_pipeline_runner.run(steps=steps)

    assert not success
    assert run_calls == steps
    assert marker_path.exists()

    marker_path.unlink()

    resume_runner_missing_marker = LocalPipelineRunner(
        scene_dir=local_pipeline_runner.scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    run_calls_missing_marker = []

    def fake_run_step_resume_missing_marker(step):
        run_calls_missing_marker.append(step)
        expected_paths = resume_runner_missing_marker._expected_output_paths(step)
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
            success=step != PipelineStep.USD,
            duration_seconds=0,
            message="boom" if step == PipelineStep.USD else "ok",
            outputs={"expected_outputs": [str(path) for path in expected_paths]},
        )

    monkeypatch.setattr(
        resume_runner_missing_marker,
        "_run_step",
        fake_run_step_resume_missing_marker,
    )
    monkeypatch.setattr(resume_runner_missing_marker, "_apply_quality_gates", lambda step, result, **kwargs: result)

    success = resume_runner_missing_marker.run(
        steps=steps,
        resume_from=PipelineStep.REGEN3D,
    )

    assert not success
    assert run_calls_missing_marker == [PipelineStep.REGEN3D, PipelineStep.SIMREADY, PipelineStep.USD]
    assert marker_path.exists()

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
    monkeypatch.setattr(resume_runner, "_apply_quality_gates", lambda step, result, **kwargs: result)

    success = resume_runner.run(steps=steps, resume_from=PipelineStep.REGEN3D)

    assert success
    assert run_calls == [PipelineStep.REGEN3D, PipelineStep.SIMREADY, PipelineStep.USD]


def test_resume_checkpointed_step_reapplies_quality_gates(local_pipeline_runner, monkeypatch):
    step = PipelineStep.REGEN3D

    resume_runner = LocalPipelineRunner(
        scene_dir=local_pipeline_runner.scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )
    monkeypatch.setattr(
        resume_runner,
        "_run_step",
        lambda _step: pytest.fail("Checkpointed step should not execute _run_step"),
    )
    monkeypatch.setattr(
        resume_runner,
        "_steps_require_geniesim_preflight",
        lambda _steps: False,
    )
    monkeypatch.setattr(
        resume_runner,
        "_check_step_prerequisites",
        lambda *_args, **_kwargs: None,
    )

    class FakeCheckpointStore:
        def should_skip_step(self, *_args, **_kwargs):
            return True

        def load_checkpoint(self, *_args, **_kwargs):
            return SimpleNamespace(outputs={"checkpointed": True})

        def write_checkpoint(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        "tools.run_local_pipeline.get_checkpoint_store",
        lambda *_args, **_kwargs: FakeCheckpointStore(),
    )

    applied = []

    def record_quality_gate(step_to_run: PipelineStep, result: StepResult, **kwargs) -> StepResult:
        applied.append((step_to_run, kwargs.get("checkpointed", False)))
        return result

    monkeypatch.setattr(resume_runner, "_apply_quality_gates", record_quality_gate)

    assert resume_runner.run(steps=[step], resume_from=step) is True
    assert applied == [(step, True)]


def test_resume_geniesim_import_ignores_stale_completion_marker(local_pipeline_runner, monkeypatch):
    step = PipelineStep.GENIESIM_IMPORT
    local_pipeline_runner.geniesim_dir.mkdir(parents=True, exist_ok=True)
    stale_job_payload = {"job_id": "job-old", "run_id": "run-old", "status": "completed"}
    _write_placeholder(local_pipeline_runner.geniesim_dir / "job.json", stale_job_payload)

    def seed_import_checkpoint(step_to_run: PipelineStep) -> StepResult:
        local_pipeline_runner._write_marker(
            local_pipeline_runner.geniesim_dir / ".geniesim_import_complete",
            status="completed",
            payload={"job_id": "job-old", "run_id": "run-old"},
        )
        return StepResult(
            step=step_to_run,
            success=True,
            duration_seconds=0,
            message="ok",
            outputs={"seeded": True},
        )

    monkeypatch.setattr(local_pipeline_runner, "_run_step", seed_import_checkpoint)
    monkeypatch.setattr(local_pipeline_runner, "_apply_quality_gates", lambda step, result, **kwargs: result)
    monkeypatch.setattr(local_pipeline_runner, "_steps_require_geniesim_preflight", lambda _steps: False)
    monkeypatch.setattr(local_pipeline_runner, "_check_step_prerequisites", lambda *_args, **_kwargs: None)
    assert local_pipeline_runner.run(steps=[step]) is True

    resume_runner = LocalPipelineRunner(
        scene_dir=local_pipeline_runner.scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )
    monkeypatch.setattr(resume_runner, "_steps_require_geniesim_preflight", lambda _steps: False)
    monkeypatch.setattr(resume_runner, "_check_step_prerequisites", lambda *_args, **_kwargs: None)
    _write_placeholder(
        resume_runner.geniesim_dir / "job.json",
        {"job_id": "job-new", "run_id": "run-new", "status": "completed"},
    )
    resume_runner._write_marker(
        resume_runner.geniesim_dir / ".geniesim_import_complete",
        status="completed",
        payload={"job_id": "job-old", "run_id": "run-old"},
    )

    run_calls = []

    def rerun_import(step_to_run: PipelineStep) -> StepResult:
        run_calls.append(step_to_run)
        resume_runner._write_marker(
            resume_runner.geniesim_dir / ".geniesim_import_complete",
            status="completed",
            payload={"job_id": "job-new", "run_id": "run-new"},
        )
        return StepResult(
            step=step_to_run,
            success=True,
            duration_seconds=0,
            message="ok",
            outputs={"rerun": True},
        )

    monkeypatch.setattr(resume_runner, "_run_step", rerun_import)
    monkeypatch.setattr(resume_runner, "_apply_quality_gates", lambda step, result, **kwargs: result)

    assert resume_runner.run(steps=[step], resume_from=step) is True
    assert run_calls == [step]
    marker_payload = json.loads((resume_runner.geniesim_dir / ".geniesim_import_complete").read_text())
    assert marker_payload["job_id"] == "job-new"
    assert marker_payload["run_id"] == "run-new"

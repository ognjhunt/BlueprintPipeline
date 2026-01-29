import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest

from tools.error_handling.retry import NonRetryableError
from fixtures.generate_mock_regen3d import generate_mock_regen3d
from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep, StepResult


def test_geniesim_submit_requires_robot_types(tmp_path, monkeypatch):
    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
    )

    geniesim_dir = tmp_path / "geniesim"
    geniesim_dir.mkdir(parents=True, exist_ok=True)
    (geniesim_dir / "scene_graph.json").write_text("{}")
    (geniesim_dir / "asset_index.json").write_text("{}")
    (geniesim_dir / "task_config.json").write_text("{}")

    monkeypatch.setattr(runner, "_resolve_geniesim_robot_types", lambda: [])

    with pytest.raises(NonRetryableError, match="No robot types configured"):
        runner._run_geniesim_submit()


def test_geniesim_mock_mode_rejected_in_production(tmp_path, monkeypatch):
    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
    )

    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("GENIESIM_MOCK_MODE", "true")

    with pytest.raises(
        NonRetryableError,
        match="Disable GENIESIM_MOCK_MODE in production",
    ):
        runner._geniesim_requires_server([PipelineStep.GENIESIM_SUBMIT])


def test_auto_trigger_import_after_submit(temp_test_dir, monkeypatch):
    scene_id = "test_scene"
    generate_mock_regen3d(
        output_dir=temp_test_dir,
        scene_id=scene_id,
        environment_type="kitchen",
    )
    scene_dir = temp_test_dir / "scenes" / scene_id
    (scene_dir / "seg").mkdir(exist_ok=True)
    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
    )

    run_calls = []

    def fake_run_step(step):
        run_calls.append(step)
        expected_paths = runner._expected_output_paths(step)
        for path in expected_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text("{}")
            else:
                path.write_text("ok")
        outputs = {"job_status": "completed"} if step == PipelineStep.GENIESIM_SUBMIT else {}
        return StepResult(
            step=step,
            success=True,
            duration_seconds=0,
            message="ok",
            outputs=outputs,
        )

    monkeypatch.setattr(runner, "_run_step", fake_run_step)
    monkeypatch.setattr(runner, "_run_geniesim_preflight", lambda require_server: True)
    monkeypatch.setattr(runner, "_apply_quality_gates", lambda step, result: result)
    monkeypatch.setattr(runner, "_check_step_prerequisites", lambda step, deps, req: None)

    success = runner.run(
        steps=[PipelineStep.GENIESIM_EXPORT, PipelineStep.GENIESIM_SUBMIT],
        auto_trigger_import=True,
    )

    assert success
    assert run_calls == [
        PipelineStep.GENIESIM_EXPORT,
        PipelineStep.GENIESIM_SUBMIT,
        PipelineStep.GENIESIM_IMPORT,
    ]

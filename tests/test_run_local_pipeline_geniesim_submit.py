import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest

from tools.error_handling.retry import NonRetryableError
from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep


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

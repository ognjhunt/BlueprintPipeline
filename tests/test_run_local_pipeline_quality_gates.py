from dataclasses import dataclass
import sys
import types

import pytest

stub_gcs_upload = types.ModuleType("tools.gcs_upload")
stub_gcs_upload.calculate_md5_base64 = lambda *args, **kwargs: ""
stub_gcs_upload.verify_blob_upload = lambda *args, **kwargs: True
sys.modules.setdefault("tools.gcs_upload", stub_gcs_upload)

from tools.error_handling.retry import NonRetryableError
from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep, StepResult


@dataclass
class DummyCollectionResult:
    success: bool
    task_name: str
    episodes_collected: int = 0
    episodes_passed: int = 0
    average_quality_score: float = 0.0


def test_geniesim_submit_blocks_when_episode_metadata_missing(tmp_path):
    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    output_dir = tmp_path / "episodes" / "geniesim_test"
    (output_dir / "lerobot").mkdir(parents=True)

    result = DummyCollectionResult(
        success=True,
        task_name="demo",
        episodes_collected=3,
        episodes_passed=3,
        average_quality_score=0.95,
    )

    runner._geniesim_local_run_results = {"franka": result}
    runner._geniesim_output_dirs = {"franka": output_dir}

    step_result = StepResult(
        step=PipelineStep.GENIESIM_SUBMIT,
        success=True,
        duration_seconds=0.01,
    )

    updated = runner._apply_quality_gates(PipelineStep.GENIESIM_SUBMIT, step_result)

    assert not updated.success
    assert updated.outputs.get("quality_gate_blocked") is True
    assert "Quality gates blocked" in updated.message


def test_skip_quality_gates_rejected_in_production(tmp_path, monkeypatch):
    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )
    monkeypatch.setenv("SKIP_QUALITY_GATES", "true")
    monkeypatch.setattr(runner, "_is_production_mode", lambda: True)

    with pytest.raises(NonRetryableError, match="not allowed in production"):
        runner._should_skip_quality_gates()

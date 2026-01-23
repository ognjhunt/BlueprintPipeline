from __future__ import annotations

from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Type google\._upb\..*",
    category=DeprecationWarning,
)

import pytest

from fixtures.generate_mock_regen3d import generate_mock_regen3d
from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep


@pytest.mark.e2e
def test_geniesim_mock_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_id = "mock_geniesim_scene"
    generate_mock_regen3d(
        output_dir=tmp_path,
        scene_id=scene_id,
        environment_type="kitchen",
    )
    scene_dir = tmp_path / "scenes" / scene_id
    (scene_dir / "seg").mkdir(exist_ok=True)

    isaac_sim_path = tmp_path / "isaac-sim"
    isaac_sim_path.mkdir(parents=True, exist_ok=True)
    (isaac_sim_path / "python.sh").write_text("#!/usr/bin/env bash\n")

    local_upload_dir = tmp_path / "firebase_uploads"

    monkeypatch.setenv("PIPELINE_ENV", "test")
    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "1")
    monkeypatch.setenv("GENIESIM_MOCK_MODE", "true")
    monkeypatch.setenv("USE_GENIESIM", "true")
    monkeypatch.setenv("SCENE_ID", scene_id)
    monkeypatch.setenv("LEROBOT_EXPORT_FORMAT", "lerobot_v3")
    monkeypatch.setenv("REQUIRE_LEROBOT", "true")
    monkeypatch.setenv("ISAAC_SIM_PATH", str(isaac_sim_path))
    monkeypatch.setenv("EPISODES_PER_TASK", "1")
    monkeypatch.setenv("NUM_VARIATIONS", "1")
    monkeypatch.setenv("FIREBASE_UPLOAD_MODE", "local")
    monkeypatch.setenv("FIREBASE_UPLOAD_LOCAL_DIR", str(local_upload_dir))
    monkeypatch.setenv("FIREBASE_UPLOAD_PREFIX", "local-datasets")

    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    steps = [
        PipelineStep.REGEN3D,
        PipelineStep.SIMREADY,
        PipelineStep.USD,
        PipelineStep.GENIESIM_EXPORT,
        PipelineStep.GENIESIM_SUBMIT,
        PipelineStep.GENIESIM_IMPORT,
    ]

    success = runner.run(steps=steps)
    assert success

    steps_seen = [result.step for result in runner.results]
    assert steps_seen == steps
    results_by_step = {result.step: result for result in runner.results}
    for step in steps:
        assert results_by_step[step].success

    import_result = results_by_step[PipelineStep.GENIESIM_IMPORT]
    output_dir = Path(import_result.outputs["output_dir"])
    lerobot_dir = Path(import_result.outputs["lerobot_path"])

    expected_paths = [
        lerobot_dir / "meta" / "info.json",
        lerobot_dir / "meta" / "stats.json",
        lerobot_dir / "meta" / "episodes" / "chunk-000" / "file-0000.parquet",
        lerobot_dir / "data" / "chunk-000" / "file-0000.parquet",
    ]
    for path in expected_paths:
        assert path.exists()

    local_prefix = Path("local-datasets") / runner.scene_id
    uploaded_info = local_upload_dir / local_prefix / (expected_paths[0].relative_to(output_dir))
    uploaded_data = local_upload_dir / local_prefix / (expected_paths[-1].relative_to(output_dir))
    assert uploaded_info.exists()
    assert uploaded_data.exists()

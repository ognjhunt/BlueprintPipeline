from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
    monkeypatch.setenv("DISABLE_ARTICULATED_ASSETS", "true")
    monkeypatch.setenv("SKIP_QUALITY_GATES", "true")
    monkeypatch.setenv("REGEN3D_ALLOW_MATERIALLESS", "true")
    monkeypatch.setenv("REGEN3D_ALLOW_TEXTURELESS", "true")

    # Mock the geniesim export job
    def fake_run_geniesim_export_job(
        root, scene_id, assets_prefix, geniesim_prefix,
        robot_type, variation_assets_prefix, replicator_prefix, copy_usd,
    ):
        geniesim_dir = scene_dir / "geniesim"
        geniesim_dir.mkdir(parents=True, exist_ok=True)
        (geniesim_dir / "scene_graph.json").write_text(
            json.dumps({"scene_id": scene_id, "nodes": []})
        )
        (geniesim_dir / "asset_index.json").write_text(json.dumps({"assets": []}))
        (geniesim_dir / "task_config.json").write_text(
            json.dumps({"tasks": [{"task_id": "task-1"}]})
        )
        (geniesim_dir / "merged_scene_manifest.json").write_text(
            json.dumps({"scene_id": scene_id, "objects": []})
        )
        return 0

    export_module = types.ModuleType("export_to_geniesim")
    export_module.run_geniesim_export_job = fake_run_geniesim_export_job
    monkeypatch.setitem(sys.modules, "export_to_geniesim", export_module)

    # Mock the geniesim preflight and data collection
    from tools.geniesim_adapter import local_framework

    def fake_run_geniesim_preflight(name, require_server=False):
        return {"ok": True, "name": name, "require_server": require_server}

    def fake_run_local_data_collection(
        scene_manifest_path, task_config_path, output_dir,
        robot_type, episodes_per_task,
        max_duration_seconds=None, verbose=False,
    ):
        recordings_dir = Path(output_dir) / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        (recordings_dir / "episode_000.json").write_text(
            json.dumps({"episode": 0, "robot": robot_type})
        )
        lerobot_dir = Path(output_dir) / "lerobot"
        lerobot_dir.mkdir(parents=True, exist_ok=True)
        meta_dir = lerobot_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir = lerobot_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "info.json").write_text(
            json.dumps({"format": "lerobot", "export_format": "lerobot_v2", "version": "2.0"})
        )
        (meta_dir / "stats.json").write_text(json.dumps({"num_episodes": 1}))
        episodes_chunk_dir = meta_dir / "episodes" / "chunk-000"
        episodes_chunk_dir.mkdir(parents=True, exist_ok=True)
        (episodes_chunk_dir / "file-0000.parquet").write_bytes(b"\x00")
        parquet_path = data_dir / "episode_000000.parquet"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pydict({
                "episode_id": ["episode_000000"],
                "frame_index": [0],
                "timestamp": [0.0],
                "observation": ["state"],
                "action": [0.0],
                "reward": [0.0],
                "done": [False],
                "task_name": ["task"],
                "task_id": ["task_0"],
            })
            pq.write_table(table, parquet_path)
        except ImportError:
            parquet_path.write_bytes(b"PAR1\x00\x00\x00\x00PAR1")
        (lerobot_dir / "dataset_info.json").write_text(
            json.dumps({"format": "lerobot", "version": "2.0", "episodes": 1, "robot": robot_type})
        )
        (lerobot_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_id": "episode_000000", "task": "task-1"}) + "\n"
        )
        return SimpleNamespace(success=True, episodes_collected=1, episodes_passed=1)

    monkeypatch.setattr(local_framework, "run_geniesim_preflight", fake_run_geniesim_preflight)
    monkeypatch.setattr(local_framework, "run_local_data_collection", fake_run_local_data_collection)

    # Mock Firebase upload
    import tools.firebase_upload.uploader
    monkeypatch.setattr(tools.firebase_upload.uploader, "init_firebase", lambda: None)

    # Mock the import job
    @dataclass
    class ImportConfig:
        job_id: str
        output_dir: Path
        min_quality_score: float
        enable_validation: bool
        filter_low_quality: bool
        require_lerobot: bool
        wait_for_completion: bool
        poll_interval: int
        job_metadata_path: str
        local_episodes_prefix: str

    def fake_run_local_import_job(config, job_metadata):
        import_manifest_path = Path(config.output_dir) / "import_manifest.json"
        import_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.3",
                    "scene_id": job_metadata.get("scene_id"),
                    "run_id": job_metadata.get("run_id", config.job_id),
                    "status": "completed",
                    "recordings_format": "json",
                    "quality": {"average_score": 0.95, "threshold": config.min_quality_score},
                    "validation": {"episodes": {"enabled": True, "episode_results": []}},
                    "job_id": config.job_id,
                    "episodes": {"passed_validation": 1},
                }
            )
        )
        return SimpleNamespace(success=True, import_manifest_path=import_manifest_path)

    import_module = types.ModuleType("import_from_geniesim")
    import_module.ImportConfig = ImportConfig
    import_module.run_local_import_job = fake_run_local_import_job
    monkeypatch.setitem(sys.modules, "import_from_geniesim", import_module)

    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
        disable_articulated_assets=True,
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
    ]
    for path in expected_paths:
        assert path.exists(), f"Expected path missing: {path}"

    recordings_path = output_dir / "recordings" / "episode_000.json"
    assert recordings_path.is_file()

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep


def test_geniesim_full_flow_e2e(tmp_path, monkeypatch):
    scene_id = "mini_scene"
    scene_dir = tmp_path / "scenes" / scene_id
    assets_dir = scene_dir / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "scene_manifest.json").write_text(
        json.dumps({"scene": {"id": scene_id}, "objects": []})
    )

    def fake_run_geniesim_export_job(
        root,
        scene_id,
        assets_prefix,
        geniesim_prefix,
        robot_type,
        variation_assets_prefix,
        replicator_prefix,
        copy_usd,
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
        return 0

    export_module = types.ModuleType("export_to_geniesim")
    export_module.run_geniesim_export_job = fake_run_geniesim_export_job
    monkeypatch.setitem(sys.modules, "export_to_geniesim", export_module)

    from tools.geniesim_adapter import local_framework

    def fake_run_geniesim_preflight(name, require_server=False):
        return {"ok": True, "name": name, "require_server": require_server}

    def fake_run_local_data_collection(
        scene_manifest_path,
        task_config_path,
        output_dir,
        robot_type,
        episodes_per_task,
        max_duration_seconds=None,
        verbose,
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
            json.dumps(
                {
                    "format": "lerobot",
                    "export_format": "lerobot_v2",
                    "version": "2.0",
                }
            )
        )
        parquet_path = data_dir / "episode_000000.parquet"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            parquet_path.write_text("")
        else:
            table = pa.Table.from_pydict(
                {
                    "episode_id": ["episode_000000"],
                    "frame_index": [0],
                    "timestamp": [0.0],
                    "observation": ["state"],
                    "action": [0.0],
                    "reward": [0.0],
                    "done": [False],
                    "task_name": ["task"],
                    "task_id": ["task_0"],
                }
            )
            pq.write_table(table, parquet_path)
        (lerobot_dir / "dataset_info.json").write_text(
            json.dumps({"format": "lerobot", "version": "2.0", "episodes": 1, "robot": robot_type})
        )
        return SimpleNamespace(success=True, episodes_collected=1, episodes_passed=1)

    monkeypatch.setattr(local_framework, "run_geniesim_preflight", fake_run_geniesim_preflight)
    monkeypatch.setattr(local_framework, "run_local_data_collection", fake_run_local_data_collection)

    upload_calls = []

    def fake_upload_episodes_to_firebase(episodes_dir, scene_id, prefix):
        upload_calls.append(
            {"episodes_dir": Path(episodes_dir), "scene_id": scene_id, "prefix": prefix}
        )
        return {"uploaded": True, "prefix": prefix}

    import tools.firebase_upload
    import tools.firebase_upload.uploader

    monkeypatch.setattr(
        tools.firebase_upload, "upload_episodes_to_firebase", fake_upload_episodes_to_firebase
    )
    monkeypatch.setattr(tools.firebase_upload.uploader, "init_firebase", lambda: None)

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
            json.dumps({"job_id": config.job_id, "scene_id": job_metadata.get("scene_id")})
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
    )

    success = runner.run(
        steps=[
            PipelineStep.GENIESIM_EXPORT,
            PipelineStep.GENIESIM_SUBMIT,
            PipelineStep.GENIESIM_IMPORT,
        ]
    )

    assert success

    job_path = scene_dir / "geniesim" / "job.json"
    assert job_path.is_file()
    job_payload = json.loads(job_path.read_text())
    job_id = job_payload["job_id"]
    expected_output_dir = scene_dir / "episodes" / f"geniesim_{job_id}"

    assert upload_calls
    assert upload_calls[0]["prefix"] == "datasets"
    assert upload_calls[0]["episodes_dir"] == expected_output_dir

    recordings_path = expected_output_dir / "recordings" / "episode_000.json"
    lerobot_path = expected_output_dir / "lerobot" / "dataset_info.json"
    assert recordings_path.is_file()
    assert lerobot_path.is_file()

import json
import socket
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage

from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return
            except OSError:
                time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {host}:{port}")


@pytest.fixture(scope="module")
def _firebase_storage_emulator():
    pytest.importorskip("testcontainers")
    from testcontainers.core.container import DockerContainer

    project_id = "local-emulator"
    container = (
        DockerContainer("firebase/emulators")
        .with_exposed_ports(9199)
        .with_command(
            "firebase emulators:start --only storage "
            f"--project {project_id} --host 0.0.0.0 --port 9199"
        )
    )
    container.start()
    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(9199))
    _wait_for_port(host, port, timeout_s=30.0)
    try:
        yield {
            "host": host,
            "port": port,
            "project": project_id,
        }
    finally:
        container.stop()


@pytest.mark.integration
def test_geniesim_full_flow_integration(tmp_path, monkeypatch, _firebase_storage_emulator):
    pytest.importorskip("grpc")
    pytest.importorskip("pyarrow")

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
            json.dumps(
                {
                    "suggested_tasks": [
                        {
                            "task_id": "task-1",
                            "task_name": "integration-task",
                            "target_position": [0.5, 0.0, 0.8],
                            "place_position": [0.3, 0.2, 0.8],
                        }
                    ]
                }
            )
        )
        return 0

    export_module = types.ModuleType("export_to_geniesim")
    export_module.run_geniesim_export_job = fake_run_geniesim_export_job
    monkeypatch.setitem(sys.modules, "export_to_geniesim", export_module)

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
        return types.SimpleNamespace(success=True, import_manifest_path=import_manifest_path)

    import_module = types.ModuleType("import_from_geniesim")
    import_module.ImportConfig = ImportConfig
    import_module.run_local_import_job = fake_run_local_import_job
    monkeypatch.setitem(sys.modules, "import_from_geniesim", import_module)

    geniesim_port = _reserve_free_port()
    geniesim_host = "127.0.0.1"
    server_log = tmp_path / "geniesim_server.log"
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.geniesim_adapter.geniesim_server",
            "--host",
            geniesim_host,
            "--port",
            str(geniesim_port),
        ],
        stdout=server_log.open("w"),
        stderr=subprocess.STDOUT,
    )
    _wait_for_port(geniesim_host, geniesim_port, timeout_s=30.0)

    emulator_host = _firebase_storage_emulator["host"]
    emulator_port = _firebase_storage_emulator["port"]
    emulator_project = _firebase_storage_emulator["project"]
    bucket_name = "test-bucket"

    emulator_endpoint = f"http://{emulator_host}:{emulator_port}"
    emulator_client = storage.Client(
        project=emulator_project,
        credentials=AnonymousCredentials(),
        client_options={"api_endpoint": emulator_endpoint},
    )
    bucket = emulator_client.bucket(bucket_name)
    try:
        emulator_client.create_bucket(bucket)
    except Exception:
        bucket = emulator_client.bucket(bucket_name)

    isaac_sim_path = tmp_path / "isaac-sim"
    isaac_sim_path.mkdir()
    (isaac_sim_path / "python.sh").write_text("#!/bin/sh\nexit 0\n")

    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "1")
    monkeypatch.setenv("ISAAC_SIM_PATH", str(isaac_sim_path))
    monkeypatch.setenv("GENIESIM_HOST", geniesim_host)
    monkeypatch.setenv("GENIESIM_PORT", str(geniesim_port))
    monkeypatch.setenv("GENIESIM_CLEANUP_TMP", "0")
    monkeypatch.setenv("GENIESIM_ALLOW_IK_FAILURE_FALLBACK", "1")
    monkeypatch.setenv("EPISODES_PER_TASK", "1")
    monkeypatch.setenv("NUM_VARIATIONS", "1")

    monkeypatch.setenv("FIREBASE_STORAGE_BUCKET", bucket_name)
    monkeypatch.setenv("FIREBASE_STORAGE_EMULATOR_HOST", f"{emulator_host}:{emulator_port}")
    monkeypatch.setenv("FIREBASE_PROJECT", emulator_project)
    monkeypatch.setenv("FIREBASE_UPLOAD_REQUIRED", "1")
    monkeypatch.setenv("FIREBASE_UPLOAD_CONCURRENCY", "1")

    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    try:
        success = runner.run(
            steps=[
                PipelineStep.GENIESIM_EXPORT,
                PipelineStep.GENIESIM_SUBMIT,
            ]
        )
    finally:
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait(timeout=10)

    assert success

    job_path = scene_dir / "geniesim" / "job.json"
    assert job_path.is_file()
    job_payload = json.loads(job_path.read_text())
    job_id = job_payload["job_id"]
    expected_output_dir = scene_dir / "episodes" / f"geniesim_{job_id}"

    recording_files = list((expected_output_dir / "recordings").rglob("*.json"))
    assert recording_files
    recording_payload = json.loads(recording_files[0].read_text())
    assert recording_payload["episode_id"]

    lerobot_info_path = expected_output_dir / "lerobot" / "dataset_info.json"
    assert lerobot_info_path.is_file()
    lerobot_info = json.loads(lerobot_info_path.read_text())
    assert lerobot_info["episodes"] >= 1

    firebase_payload = job_payload["firebase_upload"]
    summary = firebase_payload["summary"]
    assert summary["uploaded"] > 0

    expected_file_count = len([path for path in expected_output_dir.rglob("*") if path.is_file()])
    assert summary["total_files"] == expected_file_count

    remote_prefix = f"datasets/{scene_id}/"
    blob_names = [blob.name for blob in bucket.list_blobs(prefix=remote_prefix)]
    assert blob_names
    assert any("recordings/" in name for name in blob_names)
    assert any("lerobot/" in name for name in blob_names)

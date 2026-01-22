"""
Staging E2E test for Genie Sim using a real gRPC server.

Usage (inside Isaac Sim on a GPU host):
    RUN_GENIESIM_STAGING_E2E=1 \
    STAGING_DATA_ROOT=/mnt/gcs \
    STAGING_SCENE_ID=<scene_id> \
    STAGING_ENVIRONMENT_TYPE=kitchen \
    GENIESIM_HOST=localhost \
    GENIESIM_PORT=50051 \
    /isaac-sim/python.sh -m pytest tests/test_geniesim_staging_e2e.py -v
"""

import json
import os
from pathlib import Path

import pytest


def _staging_enabled() -> bool:
    return os.environ.get("RUN_GENIESIM_STAGING_E2E") == "1"


def _resolve_scene_dir() -> Path:
    data_root = os.environ.get("STAGING_DATA_ROOT")
    scene_id = os.environ.get("STAGING_SCENE_ID")
    if data_root and scene_id:
        return Path(data_root).expanduser().resolve() / "scenes" / scene_id
    raise RuntimeError(
        "Set STAGING_DATA_ROOT and STAGING_SCENE_ID to point at a staging scene."
    )


def _validate_scene_inputs(scene_dir: Path) -> None:
    required_paths = [
        scene_dir / "assets" / "scene_manifest.json",
        scene_dir / "usd" / "scene.usda",
        scene_dir / "variation_assets" / "variation_assets.json",
        scene_dir / ".usd_assembly_complete",
        scene_dir / ".replicator_complete",
    ]
    for path in required_paths:
        if not path.exists():
            raise AssertionError(f"Missing required staging artifact: {path}")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"Missing expected JSON artifact: {path}")
    return json.loads(path.read_text())


@pytest.mark.staging
@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.requires_secrets
@pytest.mark.requires_gcs
def test_geniesim_staging_e2e():
    if not _staging_enabled():
        pytest.skip("Set RUN_GENIESIM_STAGING_E2E=1 to run staging Genie Sim tests.")

    scene_dir = _resolve_scene_dir()
    _validate_scene_inputs(scene_dir)

    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=True,
        skip_interactive=True,
        environment_type=os.environ.get("STAGING_ENVIRONMENT_TYPE", "kitchen"),
    )

    success = runner.run(
        steps=[
            PipelineStep.GENIESIM_EXPORT,
            PipelineStep.GENIESIM_SUBMIT,
            PipelineStep.GENIESIM_IMPORT,
        ]
    )
    assert success, "Genie Sim staging E2E failed"

    job_path = scene_dir / "geniesim" / "job.json"
    job_payload = _load_json(job_path)
    job_id = job_payload.get("job_id")
    assert job_id, "Genie Sim job payload missing job_id"

    artifacts = job_payload.get("artifacts", {})
    episodes_root = (
        artifacts.get("episodes_path")
        or artifacts.get("episodes_prefix")
        or str(scene_dir / "episodes" / f"geniesim_{job_id}")
    )
    episodes_dir = Path(episodes_root)
    recordings_dir = episodes_dir / "recordings"
    assert recordings_dir.is_dir(), f"Missing recordings directory: {recordings_dir}"

    episode_files = list(recordings_dir.rglob("episode_*.json"))
    assert episode_files, "Expected at least one recorded episode in staging run"

    import_manifest_path = episodes_dir / "import_manifest.json"
    import_manifest = _load_json(import_manifest_path)
    assert import_manifest.get("job_id") == job_id

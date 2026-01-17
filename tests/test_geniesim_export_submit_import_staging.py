#!/usr/bin/env python3
"""
Staging test for Genie Sim export → submit → import using real services.

Prerequisites:
- Isaac Sim installed (run via /isaac-sim/python.sh).
- Genie Sim gRPC server running and reachable (GENIESIM_HOST/GENIESIM_PORT).
- Scene reconstruction outputs available (STAGING_SCENE_DIR or STAGING_DATA_ROOT +
  STAGING_SCENE_ID).

Usage:
    RUN_GENIESIM_STAGING_EXPORT_SUBMIT_IMPORT=1 \
    STAGING_SCENE_DIR=/mnt/gcs/scenes/<scene_id> \
    /isaac-sim/python.sh -m pytest tests/test_geniesim_export_submit_import_staging.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GENIESIM_IMPORT_DIR = REPO_ROOT / "genie-sim-import-job"
if str(GENIESIM_IMPORT_DIR) not in sys.path:
    sys.path.insert(0, str(GENIESIM_IMPORT_DIR))

from import_manifest_utils import compute_manifest_checksum


def _staging_enabled() -> bool:
    return os.environ.get("RUN_GENIESIM_STAGING_EXPORT_SUBMIT_IMPORT") == "1"


def _resolve_scene_dir() -> Path:
    scene_dir_env = os.environ.get("STAGING_SCENE_DIR")
    if scene_dir_env:
        return Path(scene_dir_env).expanduser().resolve()

    data_root = os.environ.get("STAGING_DATA_ROOT")
    scene_id = os.environ.get("STAGING_SCENE_ID")
    if data_root and scene_id:
        return Path(data_root).expanduser().resolve() / "scenes" / scene_id

    raise RuntimeError(
        "Set STAGING_SCENE_DIR or STAGING_DATA_ROOT + STAGING_SCENE_ID to "
        "point at real reconstruction outputs."
    )


def _require_isaac_sim() -> None:
    try:
        import omni  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Sim modules not available. Run with /isaac-sim/python.sh."
        ) from exc

    from tools.run_full_isaacsim_pipeline import initialize_isaac_sim

    if not initialize_isaac_sim(headless=True):
        raise RuntimeError("Failed to initialize Isaac Sim in staging test.")


def _assert_no_mock_flags() -> None:
    mock_mode = os.environ.get("GENIESIM_MOCK_MODE", "false").lower() == "true"
    allow_mock = os.environ.get("ALLOW_GENIESIM_MOCK", "0") == "1"
    if mock_mode or allow_mock:
        raise AssertionError(
            "Genie Sim staging test requires real gRPC + Isaac Sim. "
            "Disable GENIESIM_MOCK_MODE and ALLOW_GENIESIM_MOCK."
        )


def _validate_required_artifacts(scene_dir: Path) -> None:
    assets_dir = scene_dir / "assets"
    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise AssertionError(f"Missing scene manifest: {manifest_path}")

    usd_marker = assets_dir / ".usd_assembly_complete"
    usd_marker_alt = scene_dir / ".usd_assembly_complete"
    if not usd_marker.exists() and not usd_marker_alt.exists():
        raise AssertionError(
            "Missing USD assembly marker: "
            f"expected {usd_marker} or {usd_marker_alt}"
        )

    replicator_marker = scene_dir / "replicator" / ".replicator_complete"
    replicator_marker_alt = assets_dir / ".replicator_complete"
    if not replicator_marker.exists() and not replicator_marker_alt.exists():
        raise AssertionError(
            "Missing replicator marker: "
            f"expected {replicator_marker} or {replicator_marker_alt}"
        )

    usd_scene = scene_dir / "usd" / "scene.usda"
    if not usd_scene.is_file():
        raise AssertionError(f"Missing USD scene: {usd_scene}")

    variation_assets = scene_dir / "variation_assets" / "variation_assets.json"
    if not variation_assets.is_file():
        raise AssertionError(
            "Missing variation assets JSON required for commercial-safe export: "
            f"{variation_assets}"
        )


@pytest.mark.staging
def test_geniesim_staging_export_submit_import(monkeypatch):
    if not _staging_enabled():
        pytest.skip(
            "Set RUN_GENIESIM_STAGING_EXPORT_SUBMIT_IMPORT=1 to run staging test."
        )

    scene_dir = _resolve_scene_dir()
    _validate_required_artifacts(scene_dir)
    _assert_no_mock_flags()

    monkeypatch.setenv("GENIESIM_FORCE_LOCAL", "true")
    monkeypatch.setenv("GENIESIM_MOCK_MODE", "false")
    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "0")

    _require_isaac_sim()

    from tools.geniesim_adapter.local_framework import (
        GenieSimGRPCClient,
        check_geniesim_availability,
    )

    status = check_geniesim_availability()
    assert status.get("isaac_sim_available"), f"Isaac Sim not available: {status}"
    assert status.get("geniesim_installed"), f"Genie Sim not installed: {status}"
    assert status.get("grpc_available"), f"gRPC not available: {status}"
    assert status.get("grpc_stubs_available"), f"gRPC stubs not available: {status}"
    assert status.get("server_running"), f"Genie Sim gRPC server not running: {status}"
    assert status.get("available"), f"Genie Sim preflight failed: {status}"

    client = GenieSimGRPCClient(
        host=os.environ.get("GENIESIM_HOST", "localhost"),
        port=int(os.environ.get("GENIESIM_PORT", "50051")),
    )
    assert client.connect(), "Failed to connect to Genie Sim gRPC server"
    assert client.ping(timeout=10.0), "Genie Sim gRPC ping failed"
    client.disconnect()

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
        ],
        run_validation=False,
    )
    assert success, "Genie Sim export/submit/import pipeline failed"

    geniesim_dir = scene_dir / "geniesim"
    job_path = geniesim_dir / "job.json"
    assert job_path.is_file(), "Genie Sim job.json missing after submit"

    job_payload = json.loads(job_path.read_text())
    assert job_payload.get("status") == "completed", f"Job not completed: {job_payload}"

    local_execution = job_payload.get("local_execution", {})
    assert local_execution.get("success"), f"Local execution failed: {job_payload}"
    preflight = local_execution.get("preflight", {})
    assert preflight.get("available"), f"Preflight not available: {job_payload}"

    job_id = job_payload.get("job_id")
    assert job_id, f"Missing job_id in Genie Sim job payload: {job_payload}"

    episodes_dir = scene_dir / "episodes" / f"geniesim_{job_id}"
    import_manifest = episodes_dir / "import_manifest.json"
    lerobot_dir = episodes_dir / "lerobot"

    assert episodes_dir.is_dir(), f"Episodes directory missing: {episodes_dir}"
    assert import_manifest.is_file(), f"Missing import manifest: {import_manifest}"
    assert lerobot_dir.is_dir(), f"Missing LeRobot output: {lerobot_dir}"

    episode_files = sorted(lerobot_dir.glob("episode_*.parquet"))
    assert episode_files, "No episodes generated in LeRobot output"

    manifest_payload = json.loads(import_manifest.read_text())
    assert manifest_payload.get("schema_version") == "1.2", (
        "Unexpected import manifest schema_version"
    )

    manifest_checksum = (
        manifest_payload.get("checksums", {})
        .get("metadata", {})
        .get("import_manifest.json", {})
        .get("sha256")
    )
    assert manifest_checksum, "Import manifest checksum missing"
    assert (
        manifest_checksum == compute_manifest_checksum(manifest_payload)
    ), "Import manifest checksum mismatch"

    episodes_summary = manifest_payload.get("episodes", {})
    assert episodes_summary.get("passed_validation", 0) > 0, (
        "No episodes passed validation"
    )

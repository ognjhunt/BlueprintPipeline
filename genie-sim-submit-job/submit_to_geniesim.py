#!/usr/bin/env python3
"""
Genie Sim Submission Job.

Submits a Genie Sim generation job using the export bundle produced by
`genie-sim-export-job/` and persists the resulting job ID to GCS.

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    GENIESIM_PREFIX: Prefix where export bundle is stored (default: scenes/<scene>/geniesim)
    JOB_OUTPUT_PATH: GCS path to write job metadata (default: scenes/<scene>/geniesim/job.json)
    ROBOT_TYPE: Robot type (default: franka)
    EPISODES_PER_TASK: Episodes per task (default: 10)
    NUM_VARIATIONS: Scene variations (default: 5)
    MIN_QUALITY_SCORE: Minimum quality score (default: 0.85)
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))
from geniesim_client import GenieSimClient, GenerationParams

sys.path.insert(0, str(REPO_ROOT / "tools"))
from geniesim_adapter.local_framework import (
    check_geniesim_availability,
    run_local_data_collection,
)
from tools.metrics.pipeline_metrics import get_metrics

EXPECTED_EXPORT_SCHEMA_VERSION = "1.0.0"
EXPECTED_GENIESIM_API_VERSION = "3.0.0"


def _write_local_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _read_json_blob(client: storage.Client, bucket: str, blob_name: str) -> Dict[str, Any]:
    blob = client.bucket(bucket).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"Missing required export bundle file: gs://{bucket}/{blob_name}")
    return json.loads(blob.download_as_text())


def _write_json_blob(client: storage.Client, bucket: str, blob_name: str, payload: Dict[str, Any]) -> None:
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _parse_version(version: str) -> tuple:
    parts = version.split(".")
    padded = (parts + ["0", "0", "0"])[:3]
    parsed = []
    for part in padded:
        if part.lower() in {"x", "*"}:
            parsed.append(None)
        else:
            parsed.append(int(part))
    return tuple(parsed)


def _version_tuple(version: str) -> tuple:
    major, minor, patch = _parse_version(version)
    return (major or 0, minor or 0, patch or 0)


def _is_version_compatible(expected_version: str, min_version: str, max_version: str) -> bool:
    expected = _version_tuple(expected_version)
    min_required = _version_tuple(min_version)
    max_parsed = _parse_version(max_version)
    if expected < min_required:
        return False
    if max_parsed[1] is None or max_parsed[2] is None:
        return expected[0] == (max_parsed[0] or expected[0])
    max_required = _version_tuple(max_version)
    return expected <= max_required


def _validate_export_marker(marker: Dict[str, Any]) -> None:
    export_schema = marker.get("export_schema_version")
    geniesim_schema = marker.get("geniesim_schema_version")
    compatibility = marker.get("schema_compatibility", {})

    if not export_schema:
        raise RuntimeError("Missing export_schema_version in _GENIESIM_EXPORT_COMPLETE.")
    if export_schema != EXPECTED_EXPORT_SCHEMA_VERSION:
        raise RuntimeError(
            "Export schema mismatch: expected "
            f"{EXPECTED_EXPORT_SCHEMA_VERSION}, found {export_schema}."
        )

    if not geniesim_schema:
        raise RuntimeError("Missing geniesim_schema_version in _GENIESIM_EXPORT_COMPLETE.")
    if _parse_version(geniesim_schema)[0] != _parse_version(EXPECTED_GENIESIM_API_VERSION)[0]:
        raise RuntimeError(
            "Genie Sim API version incompatibility: "
            f"expected major {EXPECTED_GENIESIM_API_VERSION}, found {geniesim_schema}."
        )

    min_version = compatibility.get("min_geniesim_version")
    max_version = compatibility.get("max_geniesim_version")
    if not min_version or not max_version:
        raise RuntimeError(
            "Missing schema_compatibility ranges in _GENIESIM_EXPORT_COMPLETE."
        )
    if not _is_version_compatible(EXPECTED_GENIESIM_API_VERSION, min_version, max_version):
        raise RuntimeError(
            "Genie Sim API version incompatibility: expected "
            f"{EXPECTED_GENIESIM_API_VERSION} to be within [{min_version}, {max_version}]."
        )


def main() -> int:
    bucket = os.getenv("BUCKET")
    scene_id = os.getenv("SCENE_ID")
    if not bucket or not scene_id:
        raise RuntimeError("BUCKET and SCENE_ID must be set")

    geniesim_prefix = os.getenv("GENIESIM_PREFIX", f"scenes/{scene_id}/geniesim")
    job_output_path = os.getenv("JOB_OUTPUT_PATH", f"{geniesim_prefix}/job.json")

    robot_type = os.getenv("ROBOT_TYPE", "franka")
    episodes_per_task = int(os.getenv("EPISODES_PER_TASK", "10"))
    num_variations = int(os.getenv("NUM_VARIATIONS", "5"))
    min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.85"))

    storage_client = storage.Client()

    scene_graph = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/scene_graph.json")
    asset_index = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/asset_index.json")
    task_config = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/task_config.json")
    export_marker = _read_json_blob(
        storage_client,
        bucket,
        f"{geniesim_prefix}/_GENIESIM_EXPORT_COMPLETE",
    )
    _validate_export_marker(export_marker)

    generation_params = GenerationParams(
        episodes_per_task=episodes_per_task,
        num_variations=num_variations,
        robot_type=robot_type,
        min_quality_score=min_quality_score,
    )

    api_key = os.getenv("GENIE_SIM_API_KEY")
    force_local = _env_flag("GENIESIM_FORCE_LOCAL")
    enable_api_submission = _env_flag("GENIESIM_SUBMIT_API")
    submission_mode = "api" if api_key and enable_api_submission and not force_local else "local"
    job_id = None
    submission_message = None
    local_run_result = None
    preflight_status = None
    missing_components = []
    remediation_guidance = None
    episodes_output_prefix = os.getenv("OUTPUT_PREFIX", f"scenes/{scene_id}/episodes")

    if submission_mode == "api":
        client = GenieSimClient()
        try:
            metrics = get_metrics()
            with metrics.track_api_call("genie-sim", "submit_generation_job", scene_id):
                result = client.submit_generation_job(
                    scene_graph=scene_graph,
                    asset_index=asset_index,
                    task_config=task_config,
                    generation_params=generation_params,
                    job_name=f"{scene_id}-geniesim",
                )
            if not result.success or not result.job_id:
                raise RuntimeError(result.message or "Genie Sim submission failed")
            job_id = result.job_id
            submission_message = result.message
        finally:
            client.close()
    else:
        job_id = f"local-{uuid.uuid4()}"
        submission_message = "Local Genie Sim execution started (no API key provided)."
        preflight_status = check_geniesim_availability()
        missing_components = []
        if not preflight_status.get("isaac_sim_available", False):
            missing_components.append("Isaac Sim path")
        if not preflight_status.get("grpc_available", False):
            missing_components.append("gRPC stubs")
        if not preflight_status.get("server_running", False):
            missing_components.append("server running")
        remediation_guidance = (
            "Set ISAAC_SIM_PATH to your Isaac Sim install, ensure gRPC stubs are installed, "
            "and start or expose the Genie Sim server at the configured host/port."
        )

        if not preflight_status.get("available", False):
            submission_message = (
                "Local Genie Sim preflight failed; missing: "
                f"{', '.join(missing_components) or 'unknown components'}. "
                "See local_execution.remediation for next steps."
            )
        else:
            try:
                scene_manifest = _read_json_blob(
                    storage_client,
                    bucket,
                    f"{geniesim_prefix}/merged_scene_manifest.json",
                )
            except FileNotFoundError:
                scene_manifest = {"scene_graph": scene_graph}
            task_config_local = task_config

            gcs_root = Path("/mnt/gcs") / bucket
            use_gcs_fuse = gcs_root.exists()
            local_root = gcs_root if use_gcs_fuse else Path("/tmp") / "geniesim-local"
            output_dir = local_root / episodes_output_prefix / f"geniesim_{job_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            config_dir = output_dir / "config"
            scene_manifest_path = config_dir / "scene_manifest.json"
            task_config_path = config_dir / "task_config.json"
            _write_local_json(scene_manifest_path, scene_manifest)
            _write_local_json(task_config_path, task_config_local)

            local_run_result = run_local_data_collection(
                scene_manifest_path=scene_manifest_path,
                task_config_path=task_config_path,
                output_dir=output_dir,
                robot_type=robot_type,
                episodes_per_task=episodes_per_task,
                verbose=True,
            )
            if local_run_result and local_run_result.success:
                submission_message = "Local Genie Sim execution completed."
            else:
                submission_message = "Local Genie Sim execution failed."

            if not use_gcs_fuse:
                for file_path in output_dir.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_root)
                        blob = storage_client.bucket(bucket).blob(str(relative_path))
                        blob.upload_from_filename(str(file_path))

    metrics = get_metrics()
    metrics_summary = {
        "backend": metrics.backend.value,
        "stats": metrics.get_stats(),
    }
    job_status = (
        "completed"
        if local_run_result and local_run_result.success
        else ("failed" if local_run_result else "submitted")
    )
    if submission_mode == "local" and preflight_status and not preflight_status.get("available", False):
        job_status = "failed"

    job_payload = {
        "job_id": job_id,
        "scene_id": scene_id,
        "status": job_status,
        "submission_mode": submission_mode,
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "message": submission_message,
        "bundle": {
            "scene_graph": f"gs://{bucket}/{geniesim_prefix}/scene_graph.json",
            "asset_index": f"gs://{bucket}/{geniesim_prefix}/asset_index.json",
            "task_config": f"gs://{bucket}/{geniesim_prefix}/task_config.json",
        },
        "generation_params": {
            "robot_type": robot_type,
            "episodes_per_task": episodes_per_task,
            "num_variations": num_variations,
            "min_quality_score": min_quality_score,
        },
        "export_schema": {
            "export_schema_version": export_marker.get("export_schema_version"),
            "geniesim_schema_version": export_marker.get("geniesim_schema_version"),
            "blueprintpipeline_version": export_marker.get("blueprintpipeline_version"),
            "export_timestamp": export_marker.get("export_timestamp"),
            "schema_compatibility": export_marker.get("schema_compatibility", {}),
        },
        "metrics_summary": metrics_summary,
    }
    if submission_mode == "local":
        job_payload["artifacts"] = {
            "episodes_prefix": f"gs://{bucket}/{episodes_output_prefix}/geniesim_{job_id}",
            "lerobot_prefix": (
                f"gs://{bucket}/{episodes_output_prefix}/geniesim_{job_id}/lerobot"
            ),
        }
        job_payload["local_execution"] = {
            "success": bool(local_run_result and local_run_result.success),
            "episodes_collected": getattr(local_run_result, "episodes_collected", 0) if local_run_result else 0,
            "episodes_passed": getattr(local_run_result, "episodes_passed", 0) if local_run_result else 0,
            "preflight": {
                "available": preflight_status.get("available", False) if preflight_status else False,
                "isaac_sim_available": (
                    preflight_status.get("isaac_sim_available", False) if preflight_status else False
                ),
                "grpc_available": (
                    preflight_status.get("grpc_available", False) if preflight_status else False
                ),
                "server_running": (
                    preflight_status.get("server_running", False) if preflight_status else False
                ),
                "missing_components": missing_components if preflight_status else [],
                "details": preflight_status.get("details", {}) if preflight_status else {},
            },
            "remediation": (
                remediation_guidance
                if preflight_status and not preflight_status.get("available", False)
                else None
            ),
        }

    _write_json_blob(storage_client, bucket, job_output_path, job_payload)
    print(f"[GENIESIM-SUBMIT] Stored job metadata at gs://{bucket}/{job_output_path}")
    return 0


if __name__ == "__main__":
    metrics = get_metrics()
    scene_id = os.getenv("SCENE_ID", "")
    with metrics.track_job("genie-sim-submit-job", scene_id):
        raise SystemExit(main())

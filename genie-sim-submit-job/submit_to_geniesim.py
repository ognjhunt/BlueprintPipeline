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
from geniesim_adapter.local_framework import run_local_data_collection


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

    generation_params = GenerationParams(
        episodes_per_task=episodes_per_task,
        num_variations=num_variations,
        robot_type=robot_type,
        min_quality_score=min_quality_score,
    )

    api_key = os.getenv("GENIE_SIM_API_KEY")
    force_local = os.getenv("GENIESIM_FORCE_LOCAL", "false").lower() == "true"
    submission_mode = "api" if api_key and not force_local else "local"
    job_id = None
    submission_message = None
    local_run_result = None
    episodes_output_prefix = os.getenv("OUTPUT_PREFIX", f"scenes/{scene_id}/episodes")

    if submission_mode == "api":
        client = GenieSimClient()
        try:
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

    job_payload = {
        "job_id": job_id,
        "scene_id": scene_id,
        "status": (
            "completed"
            if local_run_result and local_run_result.success
            else ("failed" if local_run_result else "submitted")
        ),
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
        }

    _write_json_blob(storage_client, bucket, job_output_path, job_payload)
    print(f"[GENIESIM-SUBMIT] Stored job metadata at gs://{bucket}/{job_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

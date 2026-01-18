#!/usr/bin/env python3
"""
Cloud Run entrypoint for DWM Preparation Job.

Reads configuration from environment variables and runs the DWM preparation pipeline.
Downloads scene data from GCS, processes it, and uploads results back to GCS.

Environment Variables:
    Required:
        BUCKET: GCS bucket name
        SCENE_ID: Scene identifier

    Optional:
        ASSETS_PREFIX: GCS prefix for scene assets (default: scenes/{SCENE_ID}/assets)
        USD_PREFIX: GCS prefix for USD files (default: scenes/{SCENE_ID}/usd)
        DWM_PREFIX: GCS prefix for DWM output (default: scenes/{SCENE_ID}/dwm)
        NUM_TRAJECTORIES: Number of trajectories to generate (default: 5)
        RESOLUTION_WIDTH: Video width (default: 720)
        RESOLUTION_HEIGHT: Video height (default: 480)
        NUM_FRAMES: Frames per video (default: 49)
        FPS: Frames per second (default: 24)
        MANO_MODEL_PATH: Path to MANO model files (optional)
        REQUIRE_MANO: Require MANO assets (true/false, optional)
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
import logging
from pathlib import Path
from typing import Optional

from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring.alerting import send_alert
from prepare_dwm_bundle import DWMJobConfig, DWMPreparationJob
from models import TrajectoryType, HandActionType
from tools.gcs_upload import upload_blob_from_filename
from tools.validation.entrypoint_checks import (
    validate_required_env_vars,
    validate_scene_manifest,
)


def download_scene_data(
    client: storage.Client,
    bucket_name: str,
    scene_id: str,
    assets_prefix: str,
    usd_prefix: str,
    local_dir: Path,
) -> tuple[Path, Path]:
    """
    Download scene data from GCS to local directory.

    Returns:
        Tuple of (manifest_path, scene_usd_path)
    """
    bucket = client.bucket(bucket_name)

    # Create local directories
    assets_dir = local_dir / "assets"
    usd_dir = local_dir / "usd"
    assets_dir.mkdir(parents=True, exist_ok=True)
    usd_dir.mkdir(parents=True, exist_ok=True)

    # Download scene manifest
    manifest_blob = bucket.blob(f"{assets_prefix}/scene_manifest.json")
    manifest_path = assets_dir / "scene_manifest.json"
    if manifest_blob.exists():
        manifest_blob.download_to_filename(str(manifest_path))
        print(f"[DWM-ENTRYPOINT] Downloaded manifest: {manifest_path}")
    else:
        raise FileNotFoundError(f"Scene manifest not found: gs://{bucket_name}/{assets_prefix}/scene_manifest.json")

    # Download scene USD (optional - may not exist yet)
    scene_usd_path = usd_dir / "scene.usda"
    usd_blob = bucket.blob(f"{usd_prefix}/scene.usda")
    if usd_blob.exists():
        usd_blob.download_to_filename(str(scene_usd_path))
        print(f"[DWM-ENTRYPOINT] Downloaded USD: {scene_usd_path}")
    else:
        # Try .usd extension
        usd_blob = bucket.blob(f"{usd_prefix}/scene.usd")
        if usd_blob.exists():
            scene_usd_path = usd_dir / "scene.usd"
            usd_blob.download_to_filename(str(scene_usd_path))
            print(f"[DWM-ENTRYPOINT] Downloaded USD: {scene_usd_path}")
        else:
            print(f"[DWM-ENTRYPOINT] WARNING: Scene USD not found - will use mock renderer")
            scene_usd_path = None

    return manifest_path, scene_usd_path


def upload_dwm_bundles(
    client: storage.Client,
    bucket_name: str,
    dwm_prefix: str,
    local_output_dir: Path,
) -> tuple[int, list[dict[str, str]]]:
    """
    Upload DWM bundles to GCS.

    Returns:
        Tuple of (number of files uploaded, list of upload failures)
    """
    bucket = client.bucket(bucket_name)
    upload_count = 0
    attempted_count = 0
    failures: list[dict[str, str]] = []
    logger = logging.getLogger("dwm-preparation-job")

    for local_path in local_output_dir.rglob("*"):
        if local_path.is_file():
            relative_path = local_path.relative_to(local_output_dir)
            blob_path = f"{dwm_prefix}/{relative_path}"

            blob = bucket.blob(blob_path)
            gcs_uri = f"gs://{bucket_name}/{blob_path}"
            result = upload_blob_from_filename(
                blob,
                local_path,
                gcs_uri,
                logger=logger,
                verify_upload=True,
            )
            attempted_count += 1
            if result.success:
                upload_count += 1
            else:
                failures.append(
                    {
                        "path": blob_path,
                        "error": result.error or "unknown error",
                    }
                )

            if attempted_count % 50 == 0:
                print(
                    "[DWM-ENTRYPOINT] Uploaded "
                    f"{upload_count}/{attempted_count} files..."
                )

    print(f"[DWM-ENTRYPOINT] Uploaded {upload_count} files to gs://{bucket_name}/{dwm_prefix}/")
    if failures:
        print(f"[DWM-ENTRYPOINT] WARNING: {len(failures)} uploads failed.")
        for failure in failures[:5]:
            print(
                "[DWM-ENTRYPOINT]   - "
                f"{failure['path']}: {failure['error']}"
            )
    return upload_count, failures


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_bool(value: str | None) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _is_production_level(level: str | None) -> bool:
    if level is None:
        return False
    normalized = level.strip().lower()
    return normalized in {"production", "prod", "high", "strict"}


def main():
    """Main entrypoint."""
    print("[DWM-ENTRYPOINT] Starting DWM Preparation Job")

    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[DWM-ENTRYPOINT]",
    )

    # Get required environment variables
    bucket_name = os.environ.get("BUCKET")
    scene_id = os.environ.get("SCENE_ID")

    # Get optional environment variables
    assets_prefix = os.environ.get("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    usd_prefix = os.environ.get("USD_PREFIX", f"scenes/{scene_id}/usd")
    dwm_prefix = os.environ.get("DWM_PREFIX", f"scenes/{scene_id}/dwm")
    num_trajectories = int(os.environ.get("NUM_TRAJECTORIES", "5"))
    resolution_width = int(os.environ.get("RESOLUTION_WIDTH", "720"))
    resolution_height = int(os.environ.get("RESOLUTION_HEIGHT", "480"))
    num_frames = int(os.environ.get("NUM_FRAMES", "49"))
    fps = float(os.environ.get("FPS", "24"))
    mano_model_path = os.environ.get("MANO_MODEL_PATH")
    skip_dwm = os.environ.get("SKIP_DWM", "").strip().lower() in {"1", "true", "yes"}
    data_quality_level = os.environ.get("DATA_QUALITY_LEVEL")
    allow_mock_rendering = _is_truthy(os.environ.get("ALLOW_MOCK_RENDERING"))
    require_mano = _parse_optional_bool(os.environ.get("REQUIRE_MANO"))

    print(f"[DWM-ENTRYPOINT] Configuration:")
    print(f"  BUCKET: {bucket_name}")
    print(f"  SCENE_ID: {scene_id}")
    print(f"  ASSETS_PREFIX: {assets_prefix}")
    print(f"  USD_PREFIX: {usd_prefix}")
    print(f"  DWM_PREFIX: {dwm_prefix}")
    print(f"  NUM_TRAJECTORIES: {num_trajectories}")
    print(f"  RESOLUTION: {resolution_width}x{resolution_height}")
    print(f"  FRAMES: {num_frames} @ {fps}fps")
    print(f"  SKIP_DWM: {skip_dwm}")
    if data_quality_level:
        print(f"  DATA_QUALITY_LEVEL: {data_quality_level}")
    print(f"  ALLOW_MOCK_RENDERING: {allow_mock_rendering}")
    if require_mano is not None:
        print(f"  REQUIRE_MANO: {require_mano}")
    if mano_model_path:
        print(f"  MANO_MODEL_PATH: {mano_model_path}")

    if skip_dwm:
        print("[DWM-ENTRYPOINT] SKIP_DWM enabled - exiting without running DWM preparation.")
        sys.exit(0)

    # Create GCS client
    client = storage.Client()

    # Create temporary working directory
    with tempfile.TemporaryDirectory(prefix="dwm_") as temp_dir:
        temp_path = Path(temp_dir)
        local_scene_dir = temp_path / "scene"
        local_output_dir = temp_path / "output"
        local_scene_dir.mkdir(parents=True)
        local_output_dir.mkdir(parents=True)

        try:
            # Step 1: Download scene data
            print("[DWM-ENTRYPOINT] Downloading scene data from GCS...")
            manifest_path, scene_usd_path = download_scene_data(
                client=client,
                bucket_name=bucket_name,
                scene_id=scene_id,
                assets_prefix=assets_prefix,
                usd_prefix=usd_prefix,
                local_dir=local_scene_dir,
            )
            validate_scene_manifest(manifest_path, label="[DWM-ENTRYPOINT]")
            if scene_usd_path is None:
                if _is_production_level(data_quality_level):
                    raise FileNotFoundError(
                        "Scene USD is required for production DATA_QUALITY_LEVEL but was not found."
                    )
                if not allow_mock_rendering:
                    raise FileNotFoundError(
                        "Scene USD missing and mock rendering is not allowed. "
                        "Set ALLOW_MOCK_RENDERING=true for CI smoke tests."
                    )

            # Step 2: Configure and run DWM preparation
            print("[DWM-ENTRYPOINT] Running DWM preparation...")
            config = DWMJobConfig(
                manifest_path=manifest_path,
                scene_usd_path=scene_usd_path,
                output_dir=local_output_dir,
                num_trajectories=num_trajectories,
                resolution=(resolution_width, resolution_height),
                num_frames=num_frames,
                fps=fps,
                trajectory_types=[
                    TrajectoryType.APPROACH,
                    TrajectoryType.REACH_MANIPULATE,
                    TrajectoryType.ORBIT,
                ],
                action_types=[
                    HandActionType.GRASP,
                    HandActionType.PULL,
                    HandActionType.PUSH,
                ],
                data_quality_level=data_quality_level,
                allow_mock_rendering=allow_mock_rendering,
                require_mano=require_mano,
                verbose=True,
            )

            job = DWMPreparationJob(config)
            output = job.run()

            # Step 3: Upload results to GCS
            print("[DWM-ENTRYPOINT] Uploading DWM bundles to GCS...")
            upload_count, upload_failures = upload_dwm_bundles(
                client=client,
                bucket_name=bucket_name,
                dwm_prefix=dwm_prefix,
                local_output_dir=local_output_dir,
            )

            # Summary
            print("[DWM-ENTRYPOINT] " + "=" * 50)
            print("[DWM-ENTRYPOINT] DWM PREPARATION COMPLETE")
            print("[DWM-ENTRYPOINT] " + "=" * 50)
            print(f"[DWM-ENTRYPOINT] Scene: {scene_id}")
            print(f"[DWM-ENTRYPOINT] Bundles: {len(output.bundles)}")
            print(f"[DWM-ENTRYPOINT] Files uploaded: {upload_count}")
            print(f"[DWM-ENTRYPOINT] Time: {output.generation_time_seconds:.2f}s")
            print(f"[DWM-ENTRYPOINT] Output: gs://{bucket_name}/{dwm_prefix}/")
            if output.errors:
                print(f"[DWM-ENTRYPOINT] Warnings: {len(output.errors)}")
                for err in output.errors[:5]:
                    print(f"[DWM-ENTRYPOINT]   - {err}")
            if upload_failures:
                print(f"[DWM-ENTRYPOINT] Upload failures: {len(upload_failures)}")

            # Exit successfully
            if output.success and not upload_failures:
                sys.exit(0)
            else:
                print("[DWM-ENTRYPOINT] Job completed with errors")
                sys.exit(1)

        except Exception as e:
            print(f"[DWM-ENTRYPOINT] ERROR: {e}")
            traceback.print_exc()
            send_alert(
                event_type="dwm_job_fatal_exception",
                summary="DWM preparation job failed with an unhandled exception",
                details={
                    "job": "dwm-preparation-job",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                severity=os.getenv("ALERT_JOB_EXCEPTION_SEVERITY", "critical"),
            )
            sys.exit(1)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="DWM-PREPARATION", validate_gcs=True)
    main()

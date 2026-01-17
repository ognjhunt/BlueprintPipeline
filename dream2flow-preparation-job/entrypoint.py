#!/usr/bin/env python3
"""
Cloud Run entrypoint for Dream2Flow Preparation Job.

Reads configuration from environment variables and runs the Dream2Flow preparation pipeline.
Downloads scene data from GCS, processes it, and uploads results back to GCS.

Environment Variables:
    Required:
        BUCKET: GCS bucket name
        SCENE_ID: Scene identifier

    Optional:
        ASSETS_PREFIX: GCS prefix for scene assets (default: scenes/{SCENE_ID}/assets)
        USD_PREFIX: GCS prefix for USD files (default: scenes/{SCENE_ID}/usd)
        DREAM2FLOW_PREFIX: GCS prefix for output (default: scenes/{SCENE_ID}/dream2flow)
        NUM_TASKS: Number of tasks to generate (default: 5)
        RESOLUTION_WIDTH: Video width (default: 720)
        RESOLUTION_HEIGHT: Video height (default: 480)
        NUM_FRAMES: Frames per video (default: 49)
        FPS: Frames per second (default: 24)
        ROBOT: Robot embodiment (default: franka_panda)
        VIDEO_API_ENDPOINT: API endpoint for video generation (optional)
        DREAM2FLOW_VIDEO_CHECKPOINT: Local checkpoint path for video generation (optional)
        SEGMENTATION_API: API endpoint for segmentation (optional)
        DEPTH_API: API endpoint for depth estimation (optional)
        TRACKING_API: API endpoint for point tracking (optional)
        ROBOT_TRACKING_API: API endpoint for robot tracking (optional)
        DREAM2FLOW_ENABLE_VIDEO: Enable video generation stage (default: true)
        DREAM2FLOW_ENABLE_FLOW: Enable flow extraction stage (default: true)
        DREAM2FLOW_ENABLE_ROBOT: Enable robot tracking stage (default: true)
        DREAM2FLOW_ALLOW_PLACEHOLDER: Allow placeholder fallbacks (default: true)
"""

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring.alerting import send_alert
from prepare_dream2flow_bundle import Dream2FlowJobConfig, Dream2FlowPreparationJob
from models import TaskType, RobotEmbodiment
from tools.config.env_flags import env_flag
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
        print(f"[D2F-ENTRYPOINT] Downloaded manifest: {manifest_path}")
    else:
        raise FileNotFoundError(f"Scene manifest not found: gs://{bucket_name}/{assets_prefix}/scene_manifest.json")

    # Download scene USD (optional)
    scene_usd_path = usd_dir / "scene.usda"
    usd_blob = bucket.blob(f"{usd_prefix}/scene.usda")
    if usd_blob.exists():
        usd_blob.download_to_filename(str(scene_usd_path))
        print(f"[D2F-ENTRYPOINT] Downloaded USD: {scene_usd_path}")
    else:
        usd_blob = bucket.blob(f"{usd_prefix}/scene.usd")
        if usd_blob.exists():
            scene_usd_path = usd_dir / "scene.usd"
            usd_blob.download_to_filename(str(scene_usd_path))
            print(f"[D2F-ENTRYPOINT] Downloaded USD: {scene_usd_path}")
        else:
            print("[D2F-ENTRYPOINT] WARNING: Scene USD not found - using placeholder rendering")
            scene_usd_path = None

    return manifest_path, scene_usd_path


def upload_dream2flow_bundles(
    client: storage.Client,
    bucket_name: str,
    dream2flow_prefix: str,
    local_output_dir: Path,
) -> int:
    """
    Upload Dream2Flow bundles to GCS.

    Returns:
        Number of files uploaded
    """
    bucket = client.bucket(bucket_name)
    upload_count = 0

    for local_path in local_output_dir.rglob("*"):
        if local_path.is_file():
            relative_path = local_path.relative_to(local_output_dir)
            blob_path = f"{dream2flow_prefix}/{relative_path}"

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))
            upload_count += 1

            if upload_count % 50 == 0:
                print(f"[D2F-ENTRYPOINT] Uploaded {upload_count} files...")

    print(f"[D2F-ENTRYPOINT] Uploaded {upload_count} files to gs://{bucket_name}/{dream2flow_prefix}/")
    return upload_count


def main():
    """Main entrypoint."""
    print("[D2F-ENTRYPOINT] Starting Dream2Flow Preparation Job")

    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[D2F-ENTRYPOINT]",
    )

    # Get required environment variables
    bucket_name = os.environ.get("BUCKET")
    scene_id = os.environ.get("SCENE_ID")

    # Get optional environment variables
    assets_prefix = os.environ.get("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    usd_prefix = os.environ.get("USD_PREFIX", f"scenes/{scene_id}/usd")
    dream2flow_prefix = os.environ.get("DREAM2FLOW_PREFIX", f"scenes/{scene_id}/dream2flow")
    num_tasks = int(os.environ.get("NUM_TASKS", "5"))
    resolution_width = int(os.environ.get("RESOLUTION_WIDTH", "720"))
    resolution_height = int(os.environ.get("RESOLUTION_HEIGHT", "480"))
    num_frames = int(os.environ.get("NUM_FRAMES", "49"))
    fps = float(os.environ.get("FPS", "24"))
    robot_name = os.environ.get("ROBOT", "franka_panda")
    video_api_endpoint = os.environ.get("VIDEO_API_ENDPOINT")
    video_checkpoint_path = os.environ.get("DREAM2FLOW_VIDEO_CHECKPOINT")
    segmentation_api = os.environ.get("SEGMENTATION_API")
    depth_api = os.environ.get("DEPTH_API")
    tracking_api = os.environ.get("TRACKING_API")
    robot_tracking_api = os.environ.get("ROBOT_TRACKING_API")
    enable_video = env_flag(os.environ.get("DREAM2FLOW_ENABLE_VIDEO"), default=True)
    enable_flow = env_flag(os.environ.get("DREAM2FLOW_ENABLE_FLOW"), default=True)
    enable_robot = env_flag(os.environ.get("DREAM2FLOW_ENABLE_ROBOT"), default=True)
    allow_placeholder = env_flag(os.environ.get("DREAM2FLOW_ALLOW_PLACEHOLDER"), default=True)
    require_real_backends = env_flag(os.environ.get("DREAM2FLOW_REQUIRE_REAL_BACKENDS"), default=False)

    # Map robot name to enum
    robot_map = {
        "franka_panda": RobotEmbodiment.FRANKA_PANDA,
        "ur5e": RobotEmbodiment.UR5E,
        "spot": RobotEmbodiment.BOSTON_DYNAMICS_SPOT,
        "gr1": RobotEmbodiment.FOURIER_GR1,
    }
    robot = robot_map.get(robot_name, RobotEmbodiment.FRANKA_PANDA)

    print("[D2F-ENTRYPOINT] Configuration:")
    print(f"  BUCKET: {bucket_name}")
    print(f"  SCENE_ID: {scene_id}")
    print(f"  ASSETS_PREFIX: {assets_prefix}")
    print(f"  USD_PREFIX: {usd_prefix}")
    print(f"  DREAM2FLOW_PREFIX: {dream2flow_prefix}")
    print(f"  NUM_TASKS: {num_tasks}")
    print(f"  RESOLUTION: {resolution_width}x{resolution_height}")
    print(f"  FRAMES: {num_frames} @ {fps}fps")
    print(f"  ROBOT: {robot_name}")
    if video_api_endpoint:
        print(f"  VIDEO_API_ENDPOINT: {video_api_endpoint}")
    if video_checkpoint_path:
        print(f"  VIDEO_CHECKPOINT: {video_checkpoint_path}")
    if segmentation_api or depth_api or tracking_api:
        print("  Flow APIs:")
        if segmentation_api:
            print(f"    SEGMENTATION_API: {segmentation_api}")
        if depth_api:
            print(f"    DEPTH_API: {depth_api}")
        if tracking_api:
            print(f"    TRACKING_API: {tracking_api}")
    if robot_tracking_api:
        print(f"  ROBOT_TRACKING_API: {robot_tracking_api}")
    print(f"  ENABLE_VIDEO: {enable_video}")
    print(f"  ENABLE_FLOW: {enable_flow}")
    print(f"  ENABLE_ROBOT: {enable_robot}")
    print(f"  ALLOW_PLACEHOLDER: {allow_placeholder}")
    print(f"  REQUIRE_REAL_BACKENDS: {require_real_backends}")

    # Create GCS client
    client = storage.Client()

    # Create temporary working directory
    with tempfile.TemporaryDirectory(prefix="dream2flow_") as temp_dir:
        temp_path = Path(temp_dir)
        local_scene_dir = temp_path / "scene"
        local_output_dir = temp_path / "output"
        local_scene_dir.mkdir(parents=True)
        local_output_dir.mkdir(parents=True)

        try:
            # Step 1: Download scene data
            print("[D2F-ENTRYPOINT] Downloading scene data from GCS...")
            manifest_path, scene_usd_path = download_scene_data(
                client=client,
                bucket_name=bucket_name,
                scene_id=scene_id,
                assets_prefix=assets_prefix,
                usd_prefix=usd_prefix,
                local_dir=local_scene_dir,
            )
            validate_scene_manifest(manifest_path, label="[D2F-ENTRYPOINT]")

            # Step 2: Configure and run Dream2Flow preparation
            print("[D2F-ENTRYPOINT] Running Dream2Flow preparation...")
            config = Dream2FlowJobConfig(
                manifest_path=manifest_path,
                scene_usd_path=scene_usd_path,
                output_dir=local_output_dir,
                num_tasks=num_tasks,
                resolution=(resolution_width, resolution_height),
                num_frames=num_frames,
                fps=fps,
                robot_embodiment=robot,
                video_api_endpoint=video_api_endpoint,
                video_checkpoint_path=Path(video_checkpoint_path) if video_checkpoint_path else None,
                segmentation_api=segmentation_api,
                depth_api=depth_api,
                tracking_api=tracking_api,
                robot_tracking_api=robot_tracking_api,
                enable_video_generation=enable_video,
                enable_flow_extraction=enable_flow,
                enable_robot_tracking=enable_robot,
                allow_placeholder=allow_placeholder,
                require_real_backends=require_real_backends,
                verbose=True,
            )

            job = Dream2FlowPreparationJob(config)
            output = job.run()

            # Step 3: Upload results to GCS
            print("[D2F-ENTRYPOINT] Uploading Dream2Flow bundles to GCS...")
            upload_count = upload_dream2flow_bundles(
                client=client,
                bucket_name=bucket_name,
                dream2flow_prefix=dream2flow_prefix,
                local_output_dir=local_output_dir,
            )

            # Summary
            print("[D2F-ENTRYPOINT] " + "=" * 50)
            print("[D2F-ENTRYPOINT] DREAM2FLOW PREPARATION COMPLETE")
            print("[D2F-ENTRYPOINT] " + "=" * 50)
            print(f"[D2F-ENTRYPOINT] Scene: {scene_id}")
            print(f"[D2F-ENTRYPOINT] Bundles: {len(output.bundles)}")
            print(f"[D2F-ENTRYPOINT] Video success: {output.num_successful_videos}/{len(output.bundles)}")
            print(f"[D2F-ENTRYPOINT] Flow success: {output.num_successful_flows}/{len(output.bundles)}")
            print(f"[D2F-ENTRYPOINT] Files uploaded: {upload_count}")
            print(f"[D2F-ENTRYPOINT] Time: {output.generation_time_seconds:.2f}s")
            print(f"[D2F-ENTRYPOINT] Output: gs://{bucket_name}/{dream2flow_prefix}/")
            if output.errors:
                print(f"[D2F-ENTRYPOINT] Warnings: {len(output.errors)}")
                for err in output.errors[:5]:
                    print(f"[D2F-ENTRYPOINT]   - {err}")

            # Exit successfully
            if output.success:
                sys.exit(0)
            else:
                print("[D2F-ENTRYPOINT] Job completed with errors")
                sys.exit(1)

        except Exception as e:
            print(f"[D2F-ENTRYPOINT] ERROR: {e}")
            traceback.print_exc()
            send_alert(
                event_type="dream2flow_job_fatal_exception",
                summary="Dream2Flow preparation job failed with an unhandled exception",
                details={
                    "job": "dream2flow-preparation-job",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                severity=os.getenv("ALERT_JOB_EXCEPTION_SEVERITY", "critical"),
            )
            sys.exit(1)


if __name__ == "__main__":
    main()

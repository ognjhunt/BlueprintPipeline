#!/usr/bin/env python3
"""
Cloud Run entrypoint for the DWM inference job.

Downloads prepared DWM bundles from GCS, runs the DWM inference pipeline,
and uploads interaction videos + frames back to the same prefix.
"""

import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

from google.cloud import storage

from dwm_inference_job import DWMInferenceConfig, DWMInferenceJob

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.gcs_upload import upload_blob_from_filename


def download_dwm_bundles(
    client: storage.Client,
    bucket_name: str,
    dwm_prefix: str,
    local_dir: Path,
) -> int:
    """Download all DWM bundle assets under the given prefix."""
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket_name, prefix=f"{dwm_prefix}/")

    count = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_path = Path(blob.name).relative_to(dwm_prefix)
        target_path = local_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target_path))
        count += 1

    return count


def upload_dwm_bundles(
    client: storage.Client,
    bucket_name: str,
    dwm_prefix: str,
    local_dir: Path,
) -> int:
    """Upload updated DWM bundles back to GCS."""
    bucket = client.bucket(bucket_name)
    upload_count = 0
    logger = logging.getLogger("dwm-inference-job")

    for local_path in local_dir.rglob("*"):
        if local_path.is_file():
            relative_path = local_path.relative_to(local_dir)
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
            if not result.success:
                print(
                    "[DWM-INFERENCE] WARNING: Upload verification failed for "
                    f"{gcs_uri}: {result.error}"
                )
            upload_count += 1

            if upload_count % 50 == 0:
                print(f"[DWM-INFERENCE] Uploaded {upload_count} files...")

    print(f"[DWM-INFERENCE] Uploaded {upload_count} files to gs://{bucket_name}/{dwm_prefix}/")
    return upload_count


def main() -> None:
    print("[DWM-INFERENCE] Starting inference job")

    bucket_name = os.environ.get("BUCKET")
    scene_id = os.environ.get("SCENE_ID")
    dwm_prefix = os.environ.get("DWM_PREFIX", f"scenes/{scene_id}/dwm" if scene_id else None)
    api_endpoint = os.environ.get("DWM_API_ENDPOINT")
    checkpoint_path = os.environ.get("DWM_CHECKPOINT_PATH")
    overwrite = os.environ.get("OVERWRITE_INTERACTION", "false").lower() == "true"
    save_frames = os.environ.get("SAVE_INTERACTION_FRAMES", "true").lower() == "true"

    if not bucket_name or not dwm_prefix:
        print("[DWM-INFERENCE] ERROR: BUCKET and DWM_PREFIX environment variables are required")
        sys.exit(1)

    client = storage.Client()

    with tempfile.TemporaryDirectory(prefix="dwm_inference_") as temp_dir:
        local_dir = Path(temp_dir) / "dwm"
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[DWM-INFERENCE] Downloading bundles from gs://{bucket_name}/{dwm_prefix}/")
            count = download_dwm_bundles(
                client=client,
                bucket_name=bucket_name,
                dwm_prefix=dwm_prefix,
                local_dir=local_dir,
            )
            print(f"[DWM-INFERENCE] Downloaded {count} objects")

            config = DWMInferenceConfig(
                bundles_dir=local_dir,
                api_endpoint=api_endpoint,
                checkpoint_path=checkpoint_path,
                save_frames=save_frames,
                overwrite=overwrite,
                verbose=True,
            )
            job = DWMInferenceJob(config)
            output = job.run()

            print("[DWM-INFERENCE] Uploading interaction outputs...")
            upload_count = upload_dwm_bundles(
                client=client,
                bucket_name=bucket_name,
                dwm_prefix=dwm_prefix,
                local_dir=local_dir,
            )

            print("[DWM-INFERENCE] Inference summary")
            print(f"[DWM-INFERENCE]   Bundles: {len(output.bundles_processed)}")
            print(f"[DWM-INFERENCE]   Uploads: {upload_count}")
            if output.errors:
                print("[DWM-INFERENCE] Errors encountered:")
                for err in output.errors:
                    print(f"  - {err}")

            sys.exit(0 if output.success else 1)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[DWM-INFERENCE] ERROR: {exc}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

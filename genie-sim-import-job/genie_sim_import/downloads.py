import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from tools.utils.atomic_write import write_json_atomic


def _parse_gcs_uri(uri: str) -> Optional[Dict[str, str]]:
    if not uri.startswith("gs://"):
        return None
    remainder = uri[len("gs://"):]
    if "/" not in remainder:
        return {"bucket": remainder, "object": ""}
    bucket, obj = remainder.split("/", 1)
    return {"bucket": bucket, "object": obj}


def _resolve_local_path(bucket: str, uri_or_path: str) -> Path:
    if uri_or_path.startswith("/mnt/gcs/"):
        return Path(uri_or_path)
    parsed = _parse_gcs_uri(uri_or_path)
    if parsed:
        return Path("/mnt/gcs") / parsed["bucket"] / parsed["object"]
    return Path("/mnt/gcs") / bucket / uri_or_path


def _resolve_local_output_dir(
    bucket: str,
    output_prefix: str,
    job_id: str,
    local_episodes_prefix: Optional[str],
) -> Path:
    if local_episodes_prefix:
        return _resolve_local_path(bucket, local_episodes_prefix)
    return Path("/mnt/gcs") / bucket / output_prefix / f"geniesim_{job_id}"


def _resolve_gcs_output_path(
    output_dir: Path,
    *,
    bucket: Optional[str],
    output_prefix: Optional[str],
    job_id: str,
    explicit_gcs_output_path: Optional[str],
) -> Optional[str]:
    if explicit_gcs_output_path:
        return explicit_gcs_output_path
    output_dir_str = str(output_dir)
    if output_dir_str.startswith("/mnt/gcs/"):
        return "gs://" + output_dir_str[len("/mnt/gcs/"):]
    if bucket and output_prefix:
        return f"gs://{bucket}/{output_prefix}/geniesim_{job_id}"
    return None


def _resolve_gcs_recordings_path(
    *,
    bucket: str,
    output_prefix: str,
    job_id: str,
    local_episodes_prefix: Optional[str],
) -> str:
    if local_episodes_prefix:
        if local_episodes_prefix.startswith("gs://"):
            base = local_episodes_prefix
        elif local_episodes_prefix.startswith("/mnt/gcs/"):
            base = "gs://" + local_episodes_prefix[len("/mnt/gcs/"):]
        else:
            base = f"gs://{bucket}/{local_episodes_prefix.lstrip('/')}"
    else:
        base = f"gs://{bucket}/{output_prefix}/geniesim_{job_id}"
    return f"{base.rstrip('/')}/recordings"


def _download_recordings_from_gcs(recordings_uri: str, destination: Path) -> Path:
    try:
        from google.cloud import storage  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"google-cloud-storage unavailable for recordings download: {exc}"
        ) from exc

    parsed = _parse_gcs_uri(recordings_uri)
    if not parsed:
        raise ValueError(f"Invalid GCS recordings path: {recordings_uri}")

    bucket_name = parsed["bucket"]
    prefix = parsed["object"].rstrip("/")
    if prefix:
        prefix += "/"
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No recordings found at {recordings_uri}")

    for blob in blobs:
        blob_name = getattr(blob, "name", "")
        if not blob_name or blob_name.endswith("/"):
            continue
        rel_path = blob_name[len(prefix):] if prefix and blob_name.startswith(prefix) else blob_name
        local_path = destination / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
    return destination


def _relative_recordings_path(
    recordings_dir: Path,
    output_dir: Path,
    path: Path,
) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        try:
            rel_path = path.relative_to(recordings_dir)
        except ValueError:
            return path.as_posix()
        return (Path("recordings") / rel_path).as_posix()


def _resolve_recordings_dir(
    config: "ImportConfig",
    *,
    bucket: Optional[str],
    output_prefix: str,
    log: logging.LoggerAdapter,
) -> Path:
    recordings_dir = config.output_dir / "recordings"
    output_dir_str = str(config.output_dir)
    on_shared_volume = output_dir_str.startswith("/mnt/gcs/")

    if on_shared_volume and recordings_dir.exists():
        log.info("Using recordings from shared volume: %s", recordings_dir)
        return recordings_dir

    if recordings_dir.exists():
        log.info("Using local recordings directory: %s", recordings_dir)
        return recordings_dir

    if not bucket:
        raise ValueError("BUCKET is required to resolve recordings from GCS.")

    recordings_uri = _resolve_gcs_recordings_path(
        bucket=bucket,
        output_prefix=output_prefix,
        job_id=config.job_id,
        local_episodes_prefix=config.local_episodes_prefix,
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="geniesim_recordings_"))
    log.info(
        "Recordings not available on shared volume; downloading from %s to %s",
        recordings_uri,
        temp_dir,
    )
    downloaded_dir = _download_recordings_from_gcs(recordings_uri, temp_dir)
    log.info("Using downloaded recordings directory: %s", downloaded_dir)
    return downloaded_dir


def _load_local_job_metadata(
    bucket: str,
    job_metadata_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not job_metadata_path:
        return None
    metadata_path = _resolve_local_path(bucket, job_metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Job metadata not found at {metadata_path}")
    with open(metadata_path, "r") as handle:
        return json.load(handle)


def _write_local_job_metadata(
    bucket: str,
    job_metadata_path: Optional[str],
    job_metadata: Dict[str, Any],
) -> None:
    if not job_metadata_path:
        return
    metadata_path = _resolve_local_path(bucket, job_metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(metadata_path, job_metadata, indent=2)

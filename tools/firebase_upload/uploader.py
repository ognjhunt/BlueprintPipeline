"""Firebase upload helpers for dataset episodes."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials, storage

from tools.error_handling.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_FIREBASE_APP: Optional[firebase_admin.App] = None
_SHA256_METADATA_KEY = "sha256"


class FirebaseUploadError(RuntimeError):
    """Raised when Firebase uploads complete with failures."""

    def __init__(self, summary: dict, message: str) -> None:
        super().__init__(message)
        self.summary = summary


def init_firebase() -> firebase_admin.App:
    """Initialize the Firebase app singleton."""
    global _FIREBASE_APP
    if _FIREBASE_APP is not None:
        return _FIREBASE_APP

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")

    if not bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET is required to initialize Firebase storage")

    if service_account_json:
        try:
            service_account_payload = json.loads(service_account_json)
        except json.JSONDecodeError as exc:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
        cred = credentials.Certificate(service_account_payload)
    elif service_account_path:
        path = Path(service_account_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"FIREBASE_SERVICE_ACCOUNT_PATH not found: {path}")
        cred = credentials.Certificate(str(path))
    else:
        raise ValueError(
            "Set FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_SERVICE_ACCOUNT_PATH to "
            "initialize Firebase storage"
        )

    _FIREBASE_APP = firebase_admin.initialize_app(
        cred,
        {
            "storageBucket": bucket_name,
        },
    )
    logger.info("Initialized Firebase app with storage bucket %s", bucket_name)
    return _FIREBASE_APP


@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=10.0)
def _upload_file(blob, file_path: Path, content_type: Optional[str]) -> None:
    if content_type:
        blob.upload_from_filename(str(file_path), content_type=content_type)
    else:
        blob.upload_from_filename(str(file_path))


def _calculate_file_hashes(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> dict:
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    return {
        "md5_base64": base64.b64encode(md5_hash.digest()).decode("utf-8"),
        "sha256_hex": sha256_hash.hexdigest(),
    }


def _set_blob_sha256_metadata(blob, sha256_hex: str) -> None:
    metadata = dict(blob.metadata or {})
    metadata[_SHA256_METADATA_KEY] = sha256_hex
    blob.metadata = metadata


def _verify_blob_checksum(
    blob,
    file_path: Path,
    *,
    local_hashes: Optional[dict] = None,
) -> tuple[bool, dict]:
    if local_hashes is None:
        local_hashes = _calculate_file_hashes(file_path)
    local_md5 = local_hashes["md5_base64"]
    local_sha256 = local_hashes["sha256_hex"]

    blob.reload()
    remote_md5 = blob.md5_hash
    metadata = blob.metadata or {}
    remote_sha256 = metadata.get(_SHA256_METADATA_KEY)

    errors = []
    if remote_sha256:
        if remote_sha256 != local_sha256:
            errors.append("sha256_mismatch")
    if remote_md5:
        if remote_md5 != local_md5:
            errors.append("md5_mismatch")
    if not remote_sha256 and not remote_md5:
        errors.append("missing_remote_hashes")

    if remote_sha256 and remote_md5:
        strategy = "sha256_metadata+md5_base64"
    elif remote_sha256:
        strategy = "sha256_metadata"
    elif remote_md5:
        strategy = "md5_base64"
    else:
        strategy = "missing_remote_hashes"

    detail = {
        "expected_md5": local_md5,
        "actual_md5": remote_md5,
        "expected_sha256": local_sha256,
        "actual_sha256": remote_sha256,
        "hash_strategy": strategy,
    }
    if errors:
        detail["error"] = ", ".join(errors)
    return not errors, detail


def upload_episodes_to_firebase(
    episodes_dir: Path,
    scene_id: str,
    prefix: str = "datasets",
) -> dict:
    """Upload episode artifacts to Firebase Storage."""
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")

    init_firebase()
    bucket = storage.bucket()

    concurrency = int(os.getenv("FIREBASE_UPLOAD_CONCURRENCY", "8"))
    if concurrency < 1:
        raise ValueError("FIREBASE_UPLOAD_CONCURRENCY must be >= 1")

    total_files = 0
    uploaded_files = 0
    failures = []
    verification_failed = []

    file_paths = [
        file_path
        for file_path in sorted(episodes_dir.rglob("*"))
        if file_path.is_file()
    ]
    total_files = len(file_paths)

    def _upload_single(
        path: Path,
        remote_path: str,
        content_type: Optional[str],
    ) -> Optional[dict]:
        blob = bucket.blob(remote_path)
        local_hashes = _calculate_file_hashes(path)
        _set_blob_sha256_metadata(blob, local_hashes["sha256_hex"])
        _upload_file(blob, path, content_type)
        verified, verification_detail = _verify_blob_checksum(
            blob,
            path,
            local_hashes=local_hashes,
        )
        if not verified:
            verification_detail = {
                "local_path": str(path),
                "remote_path": remote_path,
                **verification_detail,
            }
            return {
                "local_path": str(path),
                "remote_path": remote_path,
                "error": "upload verification failed",
                "verification": verification_detail,
            }
        return None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for file_path in file_paths:
            relative_path = file_path.relative_to(episodes_dir).as_posix()
            remote_path = f"{prefix}/{scene_id}/{relative_path}"
            content_type, _ = mimetypes.guess_type(file_path.name)
            future = executor.submit(_upload_single, file_path, remote_path, content_type)
            futures[future] = {
                "local_path": str(file_path),
                "remote_path": remote_path,
            }

        for future in as_completed(futures):
            info = futures[future]
            try:
                verification_failure = future.result()
                if verification_failure:
                    failures.append(verification_failure)
                    verification_failed.append(verification_failure["verification"])
                    continue
                uploaded_files += 1
            except Exception as exc:
                logger.error(
                    "Failed to upload %s to %s: %s",
                    info["local_path"],
                    info["remote_path"],
                    exc,
                )
                failures.append({
                    "local_path": info["local_path"],
                    "remote_path": info["remote_path"],
                    "error": str(exc),
                })

    failures.sort(key=lambda failure: failure["local_path"])
    verification_failed.sort(key=lambda failure: failure["local_path"])

    summary = {
        "total_files": total_files,
        "uploaded": uploaded_files,
        "failed": len(failures),
        "failures": failures,
        "verification_failed": verification_failed,
        "verification_strategy": "sha256_metadata+md5_base64",
    }

    logger.info(
        "Firebase upload summary: %s/%s files uploaded",
        uploaded_files,
        total_files,
    )

    if failures:
        raise FirebaseUploadError(
            summary,
            f"Firebase upload failed for {len(failures)} of {total_files} files"
        )

    return summary

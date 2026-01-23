"""Firebase upload helpers for dataset episodes."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import base64
import hashlib
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

import firebase_admin
from firebase_admin import credentials, storage
from google.auth.credentials import AnonymousCredentials

from tools.error_handling.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_FIREBASE_APP: Optional[firebase_admin.App] = None
_FIREBASE_STORAGE_CLIENT: Optional[storage.Client] = None
_FIREBASE_STORAGE_BUCKET = None
_SHA256_METADATA_KEY = "sha256"
_LOCAL_UPLOAD_ROOT: Optional[Path] = None


class FirebaseUploadError(RuntimeError):
    """Raised when Firebase uploads complete with failures."""

    def __init__(self, summary: dict, message: str) -> None:
        super().__init__(message)
        self.summary = summary


def _get_emulator_endpoint() -> Optional[str]:
    emulator_host = os.getenv("FIREBASE_STORAGE_EMULATOR_HOST")
    if not emulator_host:
        return None
    if emulator_host.startswith(("http://", "https://")):
        return emulator_host
    return f"http://{emulator_host}"


def _resolve_firebase_upload_mode() -> str:
    return (os.getenv("FIREBASE_UPLOAD_MODE", "firebase") or "firebase").strip().lower()


def get_firebase_upload_mode() -> str:
    """Public wrapper for resolving Firebase upload mode."""
    return _resolve_firebase_upload_mode()


def _resolve_local_upload_root() -> Path:
    global _LOCAL_UPLOAD_ROOT
    if _LOCAL_UPLOAD_ROOT is not None:
        return _LOCAL_UPLOAD_ROOT
    local_root = os.getenv("FIREBASE_UPLOAD_LOCAL_DIR")
    if local_root:
        _LOCAL_UPLOAD_ROOT = Path(local_root).expanduser()
        return _LOCAL_UPLOAD_ROOT
    _LOCAL_UPLOAD_ROOT = Path(tempfile.mkdtemp(prefix="firebase-upload-local-"))
    return _LOCAL_UPLOAD_ROOT


def resolve_firebase_local_upload_root() -> Path:
    """Public wrapper for resolving local Firebase upload root."""
    return _resolve_local_upload_root()


def _preflight_firebase_credentials() -> None:
    if _resolve_firebase_upload_mode() == "local":
        return
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    if not bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET is required to initialize Firebase storage")

    if _get_emulator_endpoint():
        return

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if service_account_json:
        try:
            json.loads(service_account_json)
        except json.JSONDecodeError as exc:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
    elif service_account_path:
        path = Path(service_account_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"FIREBASE_SERVICE_ACCOUNT_PATH not found: {path}")
    else:
        raise ValueError(
            "Set FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_SERVICE_ACCOUNT_PATH to "
            "initialize Firebase storage"
        )


def init_firebase() -> firebase_admin.App:
    """Initialize the Firebase app singleton."""
    global _FIREBASE_APP
    if _FIREBASE_APP is not None:
        return _FIREBASE_APP

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    emulator_endpoint = _get_emulator_endpoint()

    if not bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET is required to initialize Firebase storage")

    if emulator_endpoint:
        _FIREBASE_APP = firebase_admin.initialize_app(
            options={"storageBucket": bucket_name},
        )
        logger.info("Initialized Firebase app for emulator at %s", emulator_endpoint)
        return _FIREBASE_APP

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


def _get_storage_bucket():
    emulator_endpoint = _get_emulator_endpoint()
    if emulator_endpoint:
        bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
        if not bucket_name:
            raise ValueError("FIREBASE_STORAGE_BUCKET is required to initialize Firebase storage")

        global _FIREBASE_STORAGE_CLIENT, _FIREBASE_STORAGE_BUCKET
        if _FIREBASE_STORAGE_CLIENT is None:
            project = os.getenv("FIREBASE_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "demo-firebase"))
            _FIREBASE_STORAGE_CLIENT = storage.Client(
                project=project,
                credentials=AnonymousCredentials(),
                client_options={"api_endpoint": emulator_endpoint},
            )
        if (
            _FIREBASE_STORAGE_BUCKET is None
            or _FIREBASE_STORAGE_BUCKET.name != bucket_name
        ):
            _FIREBASE_STORAGE_BUCKET = _FIREBASE_STORAGE_CLIENT.bucket(bucket_name)
        return _FIREBASE_STORAGE_BUCKET

    init_firebase()
    return storage.bucket()


def get_firebase_storage_bucket():
    """Public wrapper for resolving Firebase storage bucket."""
    _preflight_firebase_credentials()
    return _get_storage_bucket()


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


def _upload_firebase_files(
    file_paths: Sequence[Path],
    *,
    base_dir: Path,
    scene_id: str,
    prefix: str,
) -> dict:
    if _resolve_firebase_upload_mode() == "local":
        return _upload_local_files(
            file_paths,
            base_dir=base_dir,
            scene_id=scene_id,
            prefix=prefix,
        )

    bucket = _get_storage_bucket()

    concurrency = int(os.getenv("FIREBASE_UPLOAD_CONCURRENCY", "8"))
    if concurrency < 1:
        raise ValueError("FIREBASE_UPLOAD_CONCURRENCY must be >= 1")

    total_files = len(file_paths)
    uploaded_files = 0
    skipped_files = 0
    reuploaded_files = 0
    failures = []
    verification_failed = []
    file_statuses = []

    def _upload_single(
        path: Path,
        remote_path: str,
        content_type: Optional[str],
    ) -> dict:
        blob = bucket.blob(remote_path)
        local_hashes = _calculate_file_hashes(path)
        status_payload = {
            "local_path": str(path),
            "remote_path": remote_path,
        }

        if blob.exists():
            verified, verification_detail = _verify_blob_checksum(
                blob,
                path,
                local_hashes=local_hashes,
            )
            if verified:
                return {
                    **status_payload,
                    "status": "skipped",
                }

            blob.delete()
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
                    **status_payload,
                    "status": "failed",
                    "error": "reupload verification failed",
                    "verification": verification_detail,
                }
            return {
                **status_payload,
                "status": "reuploaded",
            }

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
                **status_payload,
                "status": "failed",
                "error": "upload verification failed",
                "verification": verification_detail,
            }
        return {
            **status_payload,
            "status": "uploaded",
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for file_path in file_paths:
            relative_path = file_path.relative_to(base_dir).as_posix()
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
                status_result = future.result()
                status = status_result.get("status")
                file_statuses.append(status_result)
                if status == "failed":
                    failures.append(status_result)
                    verification = status_result.get("verification")
                    if verification:
                        verification_failed.append(verification)
                    continue
                if status == "uploaded":
                    uploaded_files += 1
                elif status == "skipped":
                    skipped_files += 1
                elif status == "reuploaded":
                    reuploaded_files += 1
            except Exception as exc:
                logger.error(
                    "Failed to upload %s to %s: %s",
                    info["local_path"],
                    info["remote_path"],
                    exc,
                )
                failure = {
                    "local_path": info["local_path"],
                    "remote_path": info["remote_path"],
                    "status": "failed",
                    "error": str(exc),
                }
                failures.append(failure)
                file_statuses.append(failure)

    failures.sort(key=lambda failure: failure["local_path"])
    verification_failed.sort(key=lambda failure: failure["local_path"])
    file_statuses.sort(key=lambda status: status["local_path"])

    summary = {
        "total_files": total_files,
        "uploaded": uploaded_files,
        "skipped": skipped_files,
        "reuploaded": reuploaded_files,
        "failed": len(failures),
        "file_statuses": file_statuses,
        "failures": failures,
        "verification_failed": verification_failed,
        "verification_strategy": "sha256_metadata+md5_base64",
    }

    logger.info(
        "Firebase upload summary: %s uploaded, %s skipped, %s reuploaded, %s failed (%s total)",
        uploaded_files,
        skipped_files,
        reuploaded_files,
        len(failures),
        total_files,
    )

    if failures:
        raise FirebaseUploadError(
            summary,
            f"Firebase upload failed for {len(failures)} of {total_files} files"
        )

    return summary


def _upload_local_files(
    file_paths: Sequence[Path],
    *,
    base_dir: Path,
    scene_id: str,
    prefix: str,
) -> dict:
    local_root = _resolve_local_upload_root()
    local_root.mkdir(parents=True, exist_ok=True)

    concurrency = int(os.getenv("FIREBASE_UPLOAD_CONCURRENCY", "8"))
    if concurrency < 1:
        raise ValueError("FIREBASE_UPLOAD_CONCURRENCY must be >= 1")

    total_files = len(file_paths)
    uploaded_files = 0
    skipped_files = 0
    reuploaded_files = 0
    failures = []
    verification_failed = []
    file_statuses = []

    def _upload_single(path: Path) -> dict:
        relative_path = path.relative_to(base_dir).as_posix()
        remote_path = f"{prefix}/{scene_id}/{relative_path}"
        dest_path = local_root / remote_path
        status_payload = {
            "local_path": str(path),
            "remote_path": remote_path,
            "local_destination": str(dest_path),
        }
        local_hashes = _calculate_file_hashes(path)

        if dest_path.exists():
            existing_hashes = _calculate_file_hashes(dest_path)
            if existing_hashes == local_hashes:
                return {
                    **status_payload,
                    "status": "skipped",
                }
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_path)
            verified_hashes = _calculate_file_hashes(dest_path)
            if verified_hashes != local_hashes:
                return {
                    **status_payload,
                    "status": "failed",
                    "error": "local reupload verification failed",
                    "verification": {
                        "expected_sha256": local_hashes["sha256_hex"],
                        "actual_sha256": verified_hashes["sha256_hex"],
                    },
                }
            return {
                **status_payload,
                "status": "reuploaded",
            }

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_path)
        verified_hashes = _calculate_file_hashes(dest_path)
        if verified_hashes != local_hashes:
            return {
                **status_payload,
                "status": "failed",
                "error": "local upload verification failed",
                "verification": {
                    "expected_sha256": local_hashes["sha256_hex"],
                    "actual_sha256": verified_hashes["sha256_hex"],
                },
            }
        return {
            **status_payload,
            "status": "uploaded",
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_upload_single, file_path): file_path for file_path in file_paths}
        for future in as_completed(futures):
            try:
                status_result = future.result()
                status = status_result.get("status")
                file_statuses.append(status_result)
                if status == "failed":
                    failures.append(status_result)
                    verification = status_result.get("verification")
                    if verification:
                        verification_failed.append(verification)
                    continue
                if status == "uploaded":
                    uploaded_files += 1
                elif status == "skipped":
                    skipped_files += 1
                elif status == "reuploaded":
                    reuploaded_files += 1
            except Exception as exc:
                failure = {
                    "local_path": str(futures[future]),
                    "remote_path": "unknown",
                    "status": "failed",
                    "error": str(exc),
                }
                failures.append(failure)
                file_statuses.append(failure)

    failures.sort(key=lambda failure: failure["local_path"])
    verification_failed.sort(key=lambda failure: failure["local_path"])
    file_statuses.sort(key=lambda status: status["local_path"])

    summary = {
        "total_files": total_files,
        "uploaded": uploaded_files,
        "skipped": skipped_files,
        "reuploaded": reuploaded_files,
        "failed": len(failures),
        "file_statuses": file_statuses,
        "failures": failures,
        "verification_failed": verification_failed,
        "verification_strategy": "sha256_local+md5_local",
        "upload_mode": "local",
        "local_root": str(local_root),
    }

    if failures:
        raise FirebaseUploadError(
            summary,
            f"Firebase upload failed for {len(failures)} of {total_files} files"
        )

    return summary


def upload_episodes_to_firebase(
    episodes_dir: Path,
    scene_id: str,
    prefix: str = "datasets",
) -> dict:
    """Upload episode artifacts to Firebase Storage."""
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")

    _preflight_firebase_credentials()

    file_paths = [
        file_path
        for file_path in sorted(episodes_dir.rglob("*"))
        if file_path.is_file()
    ]
    return _upload_firebase_files(
        file_paths,
        base_dir=episodes_dir,
        scene_id=scene_id,
        prefix=prefix,
    )


def upload_firebase_files(
    paths: Iterable[str | Path],
    prefix: str,
    scene_id: str,
) -> dict:
    """Upload explicit file paths to Firebase Storage."""
    file_paths = [Path(path) for path in paths if path]
    if not file_paths:
        return {
            "total_files": 0,
            "uploaded": 0,
            "skipped": 0,
            "reuploaded": 0,
            "failed": 0,
            "file_statuses": [],
            "failures": [],
            "verification_failed": [],
            "verification_strategy": "sha256_metadata+md5_base64",
        }

    _preflight_firebase_credentials()

    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"Retry file path not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Retry path is not a file: {file_path}")

    common_root = Path(os.path.commonpath([str(path) for path in file_paths]))
    base_dir = common_root.parent if common_root.is_file() else common_root
    return _upload_firebase_files(
        file_paths,
        base_dir=base_dir,
        scene_id=scene_id,
        prefix=prefix,
    )


def cleanup_firebase_paths(
    *,
    prefix: Optional[str] = None,
    paths: Optional[Iterable[str]] = None,
) -> dict:
    """Delete Firebase blobs by prefix or by explicit paths."""
    if not prefix and not paths:
        raise ValueError("cleanup_firebase_paths requires a prefix or paths to delete")

    if _resolve_firebase_upload_mode() == "local":
        local_root = _resolve_local_upload_root()
        deleted = []
        failed = []
        requested = []
        mode = "paths" if paths else "prefix"

        if paths:
            for blob_path in paths:
                if not blob_path:
                    continue
                requested.append(blob_path)
                local_path = local_root / blob_path
                try:
                    if local_path.exists():
                        local_path.unlink()
                        deleted.append(blob_path)
                except Exception as exc:
                    failed.append({"path": blob_path, "error": str(exc)})
        elif prefix:
            prefix_path = local_root / prefix
            requested.append(prefix)
            try:
                if prefix_path.exists():
                    shutil.rmtree(prefix_path)
                    deleted.append(prefix)
            except Exception as exc:
                failed.append({"path": prefix, "error": str(exc)})

        return {
            "mode": mode,
            "prefix": prefix,
            "requested": requested,
            "deleted": deleted,
            "failed": failed,
        }

    bucket = _get_storage_bucket()

    deleted = []
    failed = []
    requested = []
    mode = "paths" if paths else "prefix"

    if paths:
        for blob_path in paths:
            if not blob_path:
                continue
            requested.append(blob_path)
            blob = bucket.blob(blob_path)
            try:
                blob.delete()
                deleted.append(blob_path)
            except Exception as exc:
                failed.append({"path": blob_path, "error": str(exc)})
    elif prefix:
        for blob in bucket.list_blobs(prefix=prefix):
            requested.append(blob.name)
            try:
                blob.delete()
                deleted.append(blob.name)
            except Exception as exc:
                failed.append({"path": blob.name, "error": str(exc)})

    return {
        "mode": mode,
        "prefix": prefix,
        "requested": requested,
        "deleted": deleted,
        "failed": failed,
    }

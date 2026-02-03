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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from math import ceil
from pathlib import Path
from typing import Iterable, Optional, Sequence

from types import SimpleNamespace

try:
    import firebase_admin
    from firebase_admin import credentials, storage
except ImportError:  # Optional dependency for local/test environments.
    firebase_admin = SimpleNamespace(initialize_app=None, App=object)
    credentials = SimpleNamespace(Certificate=None, ApplicationDefault=None)
    storage = SimpleNamespace(bucket=None, Client=None)
try:
    import google.auth
    from google.auth.credentials import AnonymousCredentials
except Exception:  # Optional dependency when google is stubbed in tests.
    class _StubGoogleAuth:
        def default(self):
            raise ImportError("google.auth is required for ADC credentials")

    google = SimpleNamespace(auth=_StubGoogleAuth())

    class AnonymousCredentials:  # type: ignore[no-redef]
        pass

from tools.error_handling.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_FIREBASE_APP: Optional[firebase_admin.App] = None
_FIREBASE_STORAGE_CLIENT: Optional[storage.Client] = None
_FIREBASE_STORAGE_BUCKET = None
_SHA256_METADATA_KEY = "sha256"
_LOCAL_UPLOAD_ROOT: Optional[Path] = None
_DEFAULT_FIREBASE_UPLOAD_FILE_TIMEOUT_SECONDS = 300.0
_DEFAULT_FIREBASE_UPLOAD_TOTAL_TIMEOUT_SECONDS = 3600.0
_DEFAULT_FIREBASE_UPLOAD_RATE_LIMIT_PER_SEC = 0.0


def _require_firebase_admin_init() -> None:
    if hasattr(firebase_admin, "initialize_app"):
        return
    raise ImportError(
        "firebase_admin is required for Firebase uploads. "
        "Install firebase-admin or use FIREBASE_UPLOAD_MODE=local."
    )


def _require_firebase_certificate() -> None:
    if hasattr(credentials, "Certificate"):
        return
    raise ImportError(
        "firebase_admin credentials.Certificate is required for Firebase uploads."
    )


def _require_firebase_adc() -> None:
    if hasattr(credentials, "ApplicationDefault"):
        return
    raise ImportError(
        "firebase_admin credentials.ApplicationDefault is required for Firebase uploads."
    )


def _require_firebase_storage() -> None:
    if hasattr(storage, "bucket") or hasattr(storage, "Client"):
        return
    raise ImportError(
        "firebase_admin storage client is required for Firebase uploads."
    )


class _RateLimiter:
    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self._rate_per_sec = rate_per_sec
        self._capacity = burst
        self._tokens = float(burst)
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, *, remote_path: str) -> None:
        if self._rate_per_sec <= 0:
            return

        while True:
            wait_seconds = 0.0
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated_at
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity,
                        self._tokens + elapsed * self._rate_per_sec,
                    )
                    self._updated_at = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait_seconds = (1 - self._tokens) / self._rate_per_sec
            if wait_seconds > 0:
                logger.info(
                    "Rate limiting Firebase upload for %s; sleeping %.2f seconds",
                    remote_path,
                    wait_seconds,
                )
                time.sleep(wait_seconds)


def _resolve_upload_rate_limit_per_sec() -> float:
    raw_rate = os.getenv(
        "FIREBASE_UPLOAD_RATE_LIMIT_PER_SEC",
        str(_DEFAULT_FIREBASE_UPLOAD_RATE_LIMIT_PER_SEC),
    )
    try:
        rate_limit = float(raw_rate)
    except ValueError as exc:
        raise ValueError("FIREBASE_UPLOAD_RATE_LIMIT_PER_SEC must be a number") from exc
    if rate_limit < 0:
        raise ValueError("FIREBASE_UPLOAD_RATE_LIMIT_PER_SEC must be >= 0")
    return rate_limit


def _resolve_upload_burst(rate_limit_per_sec: float) -> int:
    raw_burst = os.getenv("FIREBASE_UPLOAD_BURST")
    if raw_burst is None or raw_burst == "":
        if rate_limit_per_sec <= 0:
            return 0
        return max(1, int(ceil(rate_limit_per_sec)))
    try:
        burst = int(raw_burst)
    except ValueError as exc:
        raise ValueError("FIREBASE_UPLOAD_BURST must be an integer") from exc
    if burst < 1:
        raise ValueError("FIREBASE_UPLOAD_BURST must be >= 1 when rate limiting is enabled")
    return burst


def _build_upload_rate_limiter() -> Optional[_RateLimiter]:
    rate_limit = _resolve_upload_rate_limit_per_sec()
    if rate_limit <= 0:
        return None
    burst = _resolve_upload_burst(rate_limit)
    return _RateLimiter(rate_limit, burst)


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


def _resolve_upload_file_timeout_seconds() -> float:
    raw_timeout = os.getenv(
        "FIREBASE_UPLOAD_FILE_TIMEOUT_SECONDS",
        str(_DEFAULT_FIREBASE_UPLOAD_FILE_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            "FIREBASE_UPLOAD_FILE_TIMEOUT_SECONDS must be a number of seconds"
        ) from exc
    if timeout <= 0:
        raise ValueError("FIREBASE_UPLOAD_FILE_TIMEOUT_SECONDS must be > 0")
    return timeout


def _resolve_upload_total_timeout_seconds() -> float:
    raw_timeout = os.getenv(
        "FIREBASE_UPLOAD_TIMEOUT_TOTAL_SECONDS",
        str(_DEFAULT_FIREBASE_UPLOAD_TOTAL_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            "FIREBASE_UPLOAD_TIMEOUT_TOTAL_SECONDS must be a number of seconds"
        ) from exc
    if timeout <= 0:
        raise ValueError("FIREBASE_UPLOAD_TIMEOUT_TOTAL_SECONDS must be > 0")
    return timeout


def resolve_firebase_local_upload_root() -> Path:
    """Public wrapper for resolving local Firebase upload root."""
    return _resolve_local_upload_root()


def _preflight_firebase_credentials() -> str:
    if _resolve_firebase_upload_mode() == "local":
        return "local"
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    if not bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET is required to initialize Firebase storage")

    if _get_emulator_endpoint():
        return "emulator"

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if service_account_json:
        try:
            json.loads(service_account_json)
        except json.JSONDecodeError as exc:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
        return "service_account_json"
    elif service_account_path:
        path = Path(service_account_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"FIREBASE_SERVICE_ACCOUNT_PATH not found: {path}")
        return "service_account_path"

    try:
        adc_credentials, _ = google.auth.default()
    except Exception as exc:
        raise ValueError(
            "Set FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_SERVICE_ACCOUNT_PATH, or configure "
            "Application Default Credentials to initialize Firebase storage"
        ) from exc
    if isinstance(adc_credentials, AnonymousCredentials):
        raise ValueError("Application Default Credentials are anonymous")
    return "adc"


def preflight_firebase_connectivity(*, timeout_seconds: float = 10.0) -> dict:
    """
    Test actual connectivity to Firebase Storage bucket.

    This performs a lightweight check to verify:
    1. Credentials are valid and can authenticate
    2. The bucket exists and is accessible
    3. We have permission to list/write to the bucket

    Returns a dict with connectivity status and any errors.
    Raises ValueError or FirebaseUploadError on failure.

    Args:
        timeout_seconds: Maximum time to wait for connectivity check

    Returns:
        dict with keys: success, bucket_name, mode, auth_mode, latency_ms, error (if any)
    """
    import time

    start_time = time.time()
    mode = _resolve_firebase_upload_mode()

    result = {
        "success": False,
        "bucket_name": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "mode": mode,
        "auth_mode": None,
        "latency_ms": None,
        "error": None,
    }

    if mode == "local":
        # For local mode, just verify the local root is writable
        try:
            local_root = _resolve_local_upload_root()
            local_root.mkdir(parents=True, exist_ok=True)
            test_file = local_root / ".connectivity_check"
            test_file.write_text("connectivity_check")
            test_file.unlink()
            result["success"] = True
            result["latency_ms"] = int((time.time() - start_time) * 1000)
            result["local_root"] = str(local_root)
            result["auth_mode"] = "local"
            logger.info("Firebase local mode connectivity check passed: %s", local_root)
            return result
        except Exception as exc:
            result["error"] = f"Local mode connectivity check failed: {exc}"
            logger.error(result["error"])
            raise ValueError(result["error"]) from exc

    # Validate credentials first
    try:
        result["auth_mode"] = _preflight_firebase_credentials()
    except (ValueError, FileNotFoundError) as exc:
        result["error"] = f"Firebase credentials invalid: {exc}"
        logger.error(result["error"])
        raise

    # Test actual bucket connectivity
    try:
        bucket = _get_storage_bucket()
        bucket_name = bucket.name

        # Try to list blobs with a limit of 1 to verify read access
        # This is a lightweight operation that confirms bucket exists and is accessible
        blobs = list(bucket.list_blobs(max_results=1, timeout=timeout_seconds))

        # Connectivity verified
        result["success"] = True
        result["bucket_name"] = bucket_name
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        result["blobs_found"] = len(blobs)
        logger.info(
            "Firebase connectivity check passed: bucket=%s, latency=%dms",
            bucket_name,
            result["latency_ms"],
        )
        return result

    except Exception as exc:
        result["error"] = f"Firebase bucket connectivity check failed: {exc}"
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        logger.error(result["error"])
        raise ValueError(result["error"]) from exc


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
        _require_firebase_admin_init()
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
        _require_firebase_certificate()
        cred = credentials.Certificate(service_account_payload)
    elif service_account_path:
        path = Path(service_account_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"FIREBASE_SERVICE_ACCOUNT_PATH not found: {path}")
        _require_firebase_certificate()
        cred = credentials.Certificate(str(path))
    else:
        _require_firebase_adc()
        cred = credentials.ApplicationDefault()

    _require_firebase_admin_init()
    _FIREBASE_APP = firebase_admin.initialize_app(
        cred,
        {
            "storageBucket": bucket_name,
        },
    )
    logger.info("Initialized Firebase app with storage bucket %s", bucket_name)
    return _FIREBASE_APP


def _get_storage_bucket():
    _require_firebase_storage()
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
    upload_timeout_seconds = _resolve_upload_file_timeout_seconds()
    total_timeout_seconds = _resolve_upload_total_timeout_seconds()
    rate_limiter = _build_upload_rate_limiter()

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
        if rate_limiter:
            rate_limiter.acquire(remote_path=remote_path)
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

    def _record_global_timeout(pending_futures: Iterable) -> None:
        for pending_future in pending_futures:
            info = futures[pending_future]
            failure = {
                "local_path": info["local_path"],
                "remote_path": info["remote_path"],
                "status": "failed",
                "error": f"global timeout after {total_timeout_seconds} seconds",
            }
            failures.append(failure)
            file_statuses.append(failure)

    start_time = time.monotonic()
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

        pending_futures = set(futures)
        while pending_futures:
            remaining = total_timeout_seconds - (time.monotonic() - start_time)
            if remaining <= 0:
                _record_global_timeout(pending_futures)
                break
            try:
                for future in as_completed(pending_futures, timeout=remaining):
                    info = futures[future]
                    remaining = total_timeout_seconds - (time.monotonic() - start_time)
                    if remaining <= 0:
                        _record_global_timeout(pending_futures)
                        pending_futures.clear()
                        break
                    try:
                        status_result = future.result(
                            timeout=min(upload_timeout_seconds, remaining)
                        )
                        status = status_result.get("status")
                        file_statuses.append(status_result)
                        if status == "failed":
                            failures.append(status_result)
                            verification = status_result.get("verification")
                            if verification:
                                verification_failed.append(verification)
                            pending_futures.remove(future)
                            continue
                        if status == "uploaded":
                            uploaded_files += 1
                        elif status == "skipped":
                            skipped_files += 1
                        elif status == "reuploaded":
                            reuploaded_files += 1
                    except TimeoutError:
                        logger.error(
                            "Timed out uploading %s to %s after %s seconds",
                            info["local_path"],
                            info["remote_path"],
                            upload_timeout_seconds,
                        )
                        failure = {
                            "local_path": info["local_path"],
                            "remote_path": info["remote_path"],
                            "status": "failed",
                            "error": f"timeout after {upload_timeout_seconds} seconds",
                        }
                        failures.append(failure)
                        file_statuses.append(failure)
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
                    pending_futures.remove(future)
            except TimeoutError:
                _record_global_timeout(pending_futures)
                break

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
    upload_timeout_seconds = _resolve_upload_file_timeout_seconds()
    total_timeout_seconds = _resolve_upload_total_timeout_seconds()
    rate_limiter = _build_upload_rate_limiter()

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
        if rate_limiter:
            rate_limiter.acquire(remote_path=remote_path)
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

    def _record_global_timeout(pending_futures: Iterable) -> None:
        for pending_future in pending_futures:
            info = futures[pending_future]
            failure = {
                "local_path": info["local_path"],
                "remote_path": info["remote_path"],
                "status": "failed",
                "error": f"global timeout after {total_timeout_seconds} seconds",
            }
            failures.append(failure)
            file_statuses.append(failure)

    start_time = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for file_path in file_paths:
            relative_path = file_path.relative_to(base_dir).as_posix()
            remote_path = f"{prefix}/{scene_id}/{relative_path}"
            futures[executor.submit(_upload_single, file_path)] = {
                "local_path": str(file_path),
                "remote_path": remote_path,
            }

        pending_futures = set(futures)
        while pending_futures:
            remaining = total_timeout_seconds - (time.monotonic() - start_time)
            if remaining <= 0:
                _record_global_timeout(pending_futures)
                break
            try:
                for future in as_completed(pending_futures, timeout=remaining):
                    info = futures[future]
                    remaining = total_timeout_seconds - (time.monotonic() - start_time)
                    if remaining <= 0:
                        _record_global_timeout(pending_futures)
                        pending_futures.clear()
                        break
                    try:
                        status_result = future.result(
                            timeout=min(upload_timeout_seconds, remaining)
                        )
                        status = status_result.get("status")
                        file_statuses.append(status_result)
                        if status == "failed":
                            failures.append(status_result)
                            verification = status_result.get("verification")
                            if verification:
                                verification_failed.append(verification)
                            pending_futures.remove(future)
                            continue
                        if status == "uploaded":
                            uploaded_files += 1
                        elif status == "skipped":
                            skipped_files += 1
                        elif status == "reuploaded":
                            reuploaded_files += 1
                    except TimeoutError:
                        failure = {
                            "local_path": info["local_path"],
                            "remote_path": info["remote_path"],
                            "status": "failed",
                            "error": f"timeout after {upload_timeout_seconds} seconds",
                        }
                        failures.append(failure)
                        file_statuses.append(failure)
                    except Exception as exc:
                        failure = {
                            "local_path": info["local_path"],
                            "remote_path": info["remote_path"],
                            "status": "failed",
                            "error": str(exc),
                        }
                        failures.append(failure)
                        file_statuses.append(failure)
                    pending_futures.remove(future)
            except TimeoutError:
                _record_global_timeout(pending_futures)
                break

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


def verify_firebase_upload_manifest(
    *,
    remote_prefix: str,
    expected_paths: Sequence[str],
    verify_checksums: bool = False,
    local_path_map: Optional[dict[str, Path]] = None,
) -> dict:
    """Verify that all expected paths exist in Firebase Storage.

    This is a post-upload verification step to ensure data integrity.
    After uploading files, call this function to confirm all files
    actually exist in Firebase Storage.

    Args:
        remote_prefix: The Firebase Storage prefix to check (e.g., datasets/scene_id)
        expected_paths: List of relative paths expected to exist under the prefix
        verify_checksums: If True, also verify SHA256 checksums (requires local_path_map)
        local_path_map: Map of relative path -> local Path for checksum verification

    Returns:
        dict with keys: success, verified, missing, extra, checksum_mismatches, errors
    """
    mode = _resolve_firebase_upload_mode()

    # For local mode, verify files exist on disk
    if mode == "local":
        local_root = _resolve_local_upload_root()
        return _verify_local_upload_manifest(
            local_root=local_root,
            remote_prefix=remote_prefix,
            expected_paths=expected_paths,
            verify_checksums=verify_checksums,
            local_path_map=local_path_map,
        )

    # For Firebase mode, list blobs and compare
    _preflight_firebase_credentials()
    bucket = _get_storage_bucket()

    result = {
        "success": False,
        "remote_prefix": remote_prefix,
        "verified": [],
        "missing": [],
        "extra": [],
        "checksum_mismatches": [],
        "errors": [],
    }

    # List all blobs under the prefix
    try:
        remote_blobs = {}
        for blob in bucket.list_blobs(prefix=remote_prefix):
            # Strip prefix to get relative path
            relative_path = blob.name
            if relative_path.startswith(remote_prefix):
                relative_path = relative_path[len(remote_prefix):].lstrip("/")
            remote_blobs[relative_path] = blob
    except Exception as exc:
        result["errors"].append(f"Failed to list blobs under {remote_prefix}: {exc}")
        return result

    # Check for expected paths
    expected_set = set(expected_paths)
    remote_set = set(remote_blobs.keys())

    result["missing"] = list(expected_set - remote_set)
    result["extra"] = list(remote_set - expected_set)

    # Verify files that exist
    for path in expected_set & remote_set:
        blob = remote_blobs[path]
        try:
            if verify_checksums and local_path_map and path in local_path_map:
                local_path = local_path_map[path]
                verified, detail = _verify_blob_checksum(blob, local_path)
                if not verified:
                    result["checksum_mismatches"].append({
                        "path": path,
                        "detail": detail,
                    })
                else:
                    result["verified"].append(path)
            else:
                # Just verify existence
                result["verified"].append(path)
        except Exception as exc:
            result["errors"].append(f"Error verifying {path}: {exc}")

    result["success"] = (
        len(result["missing"]) == 0
        and len(result["checksum_mismatches"]) == 0
        and len(result["errors"]) == 0
    )

    logger.info(
        "Firebase manifest verification: prefix=%s, verified=%d, missing=%d, extra=%d, checksum_mismatches=%d",
        remote_prefix,
        len(result["verified"]),
        len(result["missing"]),
        len(result["extra"]),
        len(result["checksum_mismatches"]),
    )

    return result


def _verify_local_upload_manifest(
    *,
    local_root: Path,
    remote_prefix: str,
    expected_paths: Sequence[str],
    verify_checksums: bool = False,
    local_path_map: Optional[dict[str, Path]] = None,
) -> dict:
    """Verify manifest for local upload mode."""
    result = {
        "success": False,
        "remote_prefix": remote_prefix,
        "verified": [],
        "missing": [],
        "extra": [],
        "checksum_mismatches": [],
        "errors": [],
    }

    target_dir = local_root / remote_prefix

    if not target_dir.exists():
        result["missing"] = list(expected_paths)
        return result

    # List files that exist
    existing_files = set()
    for file_path in target_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(target_dir).as_posix()
            existing_files.add(relative)

    expected_set = set(expected_paths)
    result["missing"] = list(expected_set - existing_files)
    result["extra"] = list(existing_files - expected_set)

    for path in expected_set & existing_files:
        if verify_checksums and local_path_map and path in local_path_map:
            local_path = local_path_map[path]
            target_path = target_dir / path
            local_hashes = _calculate_file_hashes(local_path)
            target_hashes = _calculate_file_hashes(target_path)
            if local_hashes["sha256_hex"] != target_hashes["sha256_hex"]:
                result["checksum_mismatches"].append({
                    "path": path,
                    "expected_sha256": local_hashes["sha256_hex"],
                    "actual_sha256": target_hashes["sha256_hex"],
                })
            else:
                result["verified"].append(path)
        else:
            result["verified"].append(path)

    result["success"] = (
        len(result["missing"]) == 0
        and len(result["checksum_mismatches"]) == 0
        and len(result["errors"]) == 0
    )

    return result

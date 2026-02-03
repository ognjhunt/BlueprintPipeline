from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from google.api_core import exceptions as gcs_exceptions
    from google.api_core.retry import Retry
except Exception:  # Optional dependency for local/test stubs.
    class _StubRetry:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __call__(self, func, *args, **kwargs):
            return func(*args, **kwargs)

    class _StubExceptions:
        class ServiceUnavailable(Exception):
            pass

        class TooManyRequests(Exception):
            pass

        class DeadlineExceeded(Exception):
            pass

    gcs_exceptions = _StubExceptions()
    Retry = _StubRetry

TRANSIENT_GCS_EXCEPTIONS = (
    gcs_exceptions.ServiceUnavailable,
    gcs_exceptions.TooManyRequests,
    gcs_exceptions.DeadlineExceeded,
)


@dataclass(frozen=True)
class UploadResult:
    success: bool
    gcs_uri: str
    error: Optional[str]
    attempts: int


def _is_retryable(exception: Exception) -> bool:
    return isinstance(exception, TRANSIENT_GCS_EXCEPTIONS)


def _deadline_for_attempts(
    max_attempts: int,
    initial: float,
    multiplier: float,
    maximum: float,
    buffer_seconds: float = 10.0,
) -> float:
    delay = initial
    total_sleep = 0.0
    for _ in range(max_attempts - 1):
        total_sleep += min(delay, maximum)
        delay *= multiplier
    return total_sleep + buffer_seconds


def upload_blob_from_filename(
    blob,
    filename: Path | str,
    gcs_uri: str,
    *,
    logger: Optional[logging.Logger] = None,
    content_type: Optional[str] = None,
    verify_upload: bool = False,
    max_attempts: int = 6,
    initial_backoff: float = 0.5,
    max_backoff: float = 8.0,
    multiplier: float = 2.0,
) -> UploadResult:
    log = logger or logging.getLogger(__name__)
    attempts = 0

    def _upload() -> None:
        nonlocal attempts
        attempts += 1
        log.info(
            "gcs_upload_attempt",
            extra={
                "gcs_uri": gcs_uri,
                "local_path": str(filename),
                "attempt": attempts,
                "max_attempts": max_attempts,
            },
        )
        blob.upload_from_filename(str(filename), content_type=content_type)

    retry = Retry(
        predicate=_is_retryable,
        initial=initial_backoff,
        maximum=max_backoff,
        multiplier=multiplier,
        deadline=_deadline_for_attempts(
            max_attempts=max_attempts,
            initial=initial_backoff,
            multiplier=multiplier,
            maximum=max_backoff,
        ),
    )

    try:
        retry(_upload)
        if verify_upload:
            try:
                file_path = Path(filename)
                expected_size = file_path.stat().st_size
                expected_md5 = calculate_file_md5_base64(file_path)
                verified, failure_reason = verify_blob_upload(
                    blob,
                    gcs_uri=gcs_uri,
                    expected_size=expected_size,
                    expected_md5=expected_md5,
                    logger=log,
                )
                if not verified:
                    return UploadResult(
                        success=False,
                        gcs_uri=gcs_uri,
                        error=failure_reason or "upload verification failed",
                        attempts=attempts,
                    )
            except Exception as exc:
                log.error(
                    "gcs_upload_verification_error",
                    extra={
                        "gcs_uri": gcs_uri,
                        "local_path": str(filename),
                        "attempts": attempts,
                    },
                    exc_info=exc,
                )
                return UploadResult(
                    success=False,
                    gcs_uri=gcs_uri,
                    error=str(exc),
                    attempts=attempts,
                )
        log.info(
            "gcs_upload_success",
            extra={
                "gcs_uri": gcs_uri,
                "local_path": str(filename),
                "attempts": attempts,
            },
        )
        return UploadResult(
            success=True,
            gcs_uri=gcs_uri,
            error=None,
            attempts=attempts,
        )
    except Exception as exc:
        log.error(
            "gcs_upload_failed",
            extra={
                "gcs_uri": gcs_uri,
                "local_path": str(filename),
                "attempts": attempts,
            },
            exc_info=exc,
        )
        return UploadResult(
            success=False,
            gcs_uri=gcs_uri,
            error=str(exc),
            attempts=attempts,
        )


def calculate_md5_base64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode("utf-8")


def calculate_file_md5_base64(filename: Path | str, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(filename).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("utf-8")


def verify_blob_upload(
    blob,
    *,
    gcs_uri: str,
    expected_size: int,
    expected_md5: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[bool, Optional[str]]:
    log = logger or logging.getLogger(__name__)
    if not blob.exists():
        log.error(
            "gcs_upload_verification_failed",
            extra={
                "gcs_uri": gcs_uri,
                "reason": "missing_blob",
            },
        )
        return False, "blob missing after upload"

    blob.reload()
    actual_size = blob.size
    if actual_size != expected_size:
        log.error(
            "gcs_upload_verification_failed",
            extra={
                "gcs_uri": gcs_uri,
                "reason": "size_mismatch",
                "expected_size": expected_size,
                "actual_size": actual_size,
            },
        )
        return False, f"size mismatch: expected {expected_size}, got {actual_size}"

    if expected_md5:
        if not blob.md5_hash:
            log.error(
                "gcs_upload_verification_failed",
                extra={
                    "gcs_uri": gcs_uri,
                    "reason": "missing_md5",
                    "expected_md5": expected_md5,
                },
            )
            return False, "blob md5 missing after upload"
        if blob.md5_hash != expected_md5:
            log.error(
                "gcs_upload_verification_failed",
                extra={
                    "gcs_uri": gcs_uri,
                    "reason": "md5_mismatch",
                    "expected_md5": expected_md5,
                    "actual_md5": blob.md5_hash,
                },
            )
            return False, "md5 mismatch after upload"

    log.info(
        "gcs_upload_verified",
        extra={
            "gcs_uri": gcs_uri,
            "expected_size": expected_size,
        },
    )
    return True, None

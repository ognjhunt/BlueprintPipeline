from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google.api_core import exceptions as gcs_exceptions
from google.api_core.retry import Retry

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

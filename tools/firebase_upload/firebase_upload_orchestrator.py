"""Shared orchestration helpers for Firebase uploads."""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.firebase_upload.uploader import (
    FirebaseUploadError,
    cleanup_firebase_paths,
    upload_firebase_files,
    upload_episodes_to_firebase,
)

logger = logging.getLogger(__name__)

FIREBASE_UPLOAD_PREFIX_ENV = "FIREBASE_UPLOAD_PREFIX"
DEFAULT_FIREBASE_UPLOAD_PREFIX = "datasets"


@dataclass
class FirebaseUploadResult:
    """Summary of an orchestrated Firebase upload."""

    summary: dict
    remote_prefix: str
    initial_summary: Optional[dict] = None
    retry_summary: Optional[dict] = None
    retry_attempted: bool = False
    retry_failed_count: int = 0


class FirebaseUploadOrchestratorError(RuntimeError):
    """Raised when an orchestrated Firebase upload fails."""

    def __init__(
        self,
        message: str,
        *,
        summary: Optional[dict] = None,
        retry_summary: Optional[dict] = None,
        retry_attempted: bool = False,
        retry_failed_count: int = 0,
        remote_prefix: Optional[str] = None,
        cleanup_result: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.summary = summary
        self.retry_summary = retry_summary
        self.retry_attempted = retry_attempted
        self.retry_failed_count = retry_failed_count
        self.remote_prefix = remote_prefix
        self.cleanup_result = cleanup_result


def resolve_firebase_upload_prefix(
    prefix: Optional[str] = None,
    *,
    default: str = DEFAULT_FIREBASE_UPLOAD_PREFIX,
) -> str:
    """Resolve the Firebase upload prefix from env or override."""
    if prefix:
        return prefix
    return os.getenv(FIREBASE_UPLOAD_PREFIX_ENV, default)


def build_firebase_upload_scene_id(scene_id: str, robot_type: Optional[str]) -> str:
    """Build the scene_id value used by the uploader."""
    if robot_type:
        return f"{scene_id}/{robot_type}"
    return scene_id


def build_firebase_upload_prefix(
    scene_id: str,
    *,
    robot_type: Optional[str] = None,
    prefix: Optional[str] = None,
) -> str:
    """Build the remote prefix used for uploads and cleanup."""
    resolved_prefix = resolve_firebase_upload_prefix(prefix)
    if robot_type:
        return f"{resolved_prefix}/{scene_id}/{robot_type}"
    return f"{resolved_prefix}/{scene_id}"


def _sleep_backoff(
    attempt: int,
    *,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: float = 0.2,
) -> None:
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    if jitter:
        delay *= 1 + random.uniform(-jitter, jitter)
    time.sleep(delay)


def upload_episodes_with_retry(
    *,
    episodes_dir: Path,
    scene_id: str,
    robot_type: Optional[str] = None,
    prefix: Optional[str] = None,
    cleanup_on_failure: bool = True,
    second_pass_max: Optional[int] = None,
) -> FirebaseUploadResult:
    """Upload episodes with a consistent path layout and retry/backoff."""
    resolved_prefix = resolve_firebase_upload_prefix(prefix)
    upload_scene_id = build_firebase_upload_scene_id(scene_id, robot_type)
    remote_prefix = build_firebase_upload_prefix(
        scene_id,
        robot_type=robot_type,
        prefix=resolved_prefix,
    )
    if second_pass_max is None:
        second_pass_max = int(os.getenv("FIREBASE_UPLOAD_SECOND_PASS_MAX", "1"))
    if second_pass_max < 0:
        raise ValueError("FIREBASE_UPLOAD_SECOND_PASS_MAX must be >= 0")

    try:
        summary = upload_episodes_to_firebase(
            episodes_dir=episodes_dir,
            scene_id=upload_scene_id,
            prefix=resolved_prefix,
        )
        return FirebaseUploadResult(
            summary=summary,
            initial_summary=summary,
            remote_prefix=remote_prefix,
        )
    except FirebaseUploadError as exc:
        initial_summary = exc.summary
        retry_summary = None
        retry_attempted = False
        retry_failed_count = 0
        failures = initial_summary.get("failures", []) if initial_summary else []

        if failures and second_pass_max > 0:
            retry_paths = [
                Path(failure["local_path"])
                for failure in failures
                if failure.get("local_path")
            ]
            if retry_paths:
                retry_attempted = True
                for attempt in range(1, second_pass_max + 1):
                    try:
                        retry_summary = upload_firebase_files(
                            retry_paths,
                            prefix=resolved_prefix,
                            scene_id=upload_scene_id,
                        )
                        retry_failed_count = len(retry_summary.get("failures", []))
                        if retry_failed_count == 0:
                            return FirebaseUploadResult(
                                summary=retry_summary,
                                initial_summary=initial_summary,
                                retry_summary=retry_summary,
                                retry_attempted=True,
                                retry_failed_count=0,
                                remote_prefix=remote_prefix,
                            )
                    except FirebaseUploadError as retry_exc:
                        retry_summary = retry_exc.summary
                        retry_failed_count = len(retry_summary.get("failures", []))
                        if attempt < second_pass_max:
                            _sleep_backoff(attempt)
                            continue
                    except Exception as retry_exc:
                        logger.warning(
                            "Firebase retry attempt %s failed: %s",
                            attempt,
                            retry_exc,
                        )
                        if attempt < second_pass_max:
                            _sleep_backoff(attempt)
                            continue
                    break

        cleanup_result = None
        if cleanup_on_failure:
            try:
                cleanup_result = cleanup_firebase_paths(prefix=remote_prefix)
            except Exception as cleanup_exc:
                logger.warning(
                    "Failed to cleanup Firebase prefix %s: %s",
                    remote_prefix,
                    cleanup_exc,
                )

        raise FirebaseUploadOrchestratorError(
            "Firebase upload failed after retries",
            summary=initial_summary,
            retry_summary=retry_summary,
            retry_attempted=retry_attempted,
            retry_failed_count=retry_failed_count,
            remote_prefix=remote_prefix,
            cleanup_result=cleanup_result,
        ) from exc
    except Exception as exc:
        cleanup_result = None
        if cleanup_on_failure:
            try:
                cleanup_result = cleanup_firebase_paths(prefix=remote_prefix)
            except Exception as cleanup_exc:
                logger.warning(
                    "Failed to cleanup Firebase prefix %s after exception: %s",
                    remote_prefix,
                    cleanup_exc,
                )
        raise FirebaseUploadOrchestratorError(
            "Firebase upload failed",
            remote_prefix=remote_prefix,
            cleanup_result=cleanup_result,
        ) from exc


def cleanup_firebase_upload_prefix(
    *,
    scene_id: str,
    robot_type: Optional[str] = None,
    prefix: Optional[str] = None,
) -> dict:
    """Cleanup Firebase uploads for a scene/robot prefix."""
    remote_prefix = build_firebase_upload_prefix(
        scene_id,
        robot_type=robot_type,
        prefix=prefix,
    )
    return cleanup_firebase_paths(prefix=remote_prefix)

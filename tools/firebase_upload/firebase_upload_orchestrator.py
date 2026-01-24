"""Shared orchestration helpers for Firebase uploads."""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tools.firebase_upload.uploader import (
    FirebaseUploadError,
    cleanup_firebase_paths,
    upload_firebase_files,
    upload_episodes_to_firebase,
    verify_firebase_upload_manifest,
)
from tools.config.production_mode import resolve_production_mode

logger = logging.getLogger(__name__)

FIREBASE_UPLOAD_PREFIX_ENV = "FIREBASE_UPLOAD_PREFIX"
DEFAULT_FIREBASE_UPLOAD_PREFIX = "datasets"
FIREBASE_REQUIRE_ATOMIC_UPLOAD_ENV = "FIREBASE_REQUIRE_ATOMIC_UPLOAD"


def _resolve_require_atomic_upload() -> bool:
    """Resolve whether atomic uploads are required.

    In production mode, atomic uploads are REQUIRED by default to prevent
    partial dataset delivery. This ensures all robots either succeed together
    or all uploads are rolled back.

    Set FIREBASE_REQUIRE_ATOMIC_UPLOAD=false to disable (not recommended).
    """
    raw_value = os.getenv(FIREBASE_REQUIRE_ATOMIC_UPLOAD_ENV)
    if raw_value is not None:
        return raw_value.lower() in ("true", "1", "yes")
    # Default to True in production mode
    return resolve_production_mode()


@dataclass
class FirebaseUploadResult:
    """Summary of an orchestrated Firebase upload."""

    summary: dict
    remote_prefix: str
    initial_summary: Optional[dict] = None
    retry_summary: Optional[dict] = None
    retry_attempted: bool = False
    retry_failed_count: int = 0


@dataclass
class AtomicUploadTransaction:
    """Tracks an atomic multi-robot upload transaction.

    This ensures that if any robot upload fails, all successful uploads
    are rolled back to prevent partial dataset delivery.
    """

    scene_id: str
    successful_prefixes: List[str] = field(default_factory=list)
    successful_results: Dict[str, FirebaseUploadResult] = field(default_factory=dict)
    failed_robots: List[Tuple[str, BaseException]] = field(default_factory=list)
    committed: bool = False
    rolled_back: bool = False

    def record_success(self, robot_type: str, result: FirebaseUploadResult) -> None:
        """Record a successful robot upload."""
        self.successful_prefixes.append(result.remote_prefix)
        self.successful_results[robot_type] = result

    def record_failure(self, robot_type: str, error: BaseException) -> None:
        """Record a failed robot upload."""
        self.failed_robots.append((robot_type, error))

    def should_rollback(self) -> bool:
        """Check if transaction should be rolled back."""
        return len(self.failed_robots) > 0 and not self.committed

    def rollback(self) -> Dict[str, any]:
        """Roll back all successful uploads.

        Returns:
            Summary of rollback operations with cleanup results per prefix.
        """
        if self.rolled_back:
            return {"status": "already_rolled_back", "prefixes": []}

        rollback_results = {
            "status": "rollback_complete",
            "prefixes": [],
            "errors": [],
        }

        for prefix in self.successful_prefixes:
            try:
                cleanup_result = cleanup_firebase_paths(prefix=prefix)
                rollback_results["prefixes"].append({
                    "prefix": prefix,
                    "status": "cleaned",
                    "result": cleanup_result,
                })
                logger.info(
                    "Atomic rollback: cleaned Firebase prefix %s (deleted=%d)",
                    prefix,
                    cleanup_result.get("deleted", 0) if cleanup_result else 0,
                )
            except Exception as exc:
                rollback_results["errors"].append({
                    "prefix": prefix,
                    "error": str(exc),
                })
                logger.error(
                    "Atomic rollback: failed to clean prefix %s: %s",
                    prefix,
                    exc,
                )

        self.rolled_back = True
        rollback_results["status"] = (
            "rollback_complete" if not rollback_results["errors"]
            else "rollback_partial"
        )
        return rollback_results

    def commit(self) -> None:
        """Mark transaction as committed (no rollback possible)."""
        self.committed = True


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
        cleanup_error: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.summary = summary
        self.retry_summary = retry_summary
        self.retry_attempted = retry_attempted
        self.retry_failed_count = retry_failed_count
        self.remote_prefix = remote_prefix
        self.cleanup_result = cleanup_result
        self.cleanup_error = cleanup_error


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
    file_paths: Optional[list[Path]] = None,
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
        if file_paths is None:
            summary = upload_episodes_to_firebase(
                episodes_dir=episodes_dir,
                scene_id=upload_scene_id,
                prefix=resolved_prefix,
            )
        else:
            summary = upload_firebase_files(
                file_paths,
                prefix=resolved_prefix,
                scene_id=upload_scene_id,
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
        cleanup_error = None
        if cleanup_on_failure:
            try:
                cleanup_result = cleanup_firebase_paths(prefix=remote_prefix)
            except Exception as cleanup_exc:
                cleanup_error = cleanup_exc
                logger.error(
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
            cleanup_error=cleanup_error,
        ) from exc
    except Exception as exc:
        cleanup_result = None
        cleanup_error = None
        if cleanup_on_failure:
            try:
                cleanup_result = cleanup_firebase_paths(prefix=remote_prefix)
            except Exception as cleanup_exc:
                cleanup_error = cleanup_exc
                logger.error(
                    "Failed to cleanup Firebase prefix %s after exception: %s",
                    remote_prefix,
                    cleanup_exc,
                )
        raise FirebaseUploadOrchestratorError(
            "Firebase upload failed",
            remote_prefix=remote_prefix,
            cleanup_result=cleanup_result,
            cleanup_error=cleanup_error,
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


def verify_firebase_upload(
    *,
    scene_id: str,
    expected_paths: list[str],
    robot_type: Optional[str] = None,
    prefix: Optional[str] = None,
    verify_checksums: bool = False,
    local_path_map: Optional[dict[str, Path]] = None,
) -> dict:
    """Verify that all expected files exist in Firebase Storage after upload.

    This is a post-upload verification step to ensure data integrity.
    Call this after upload_episodes_with_retry to confirm all files
    actually exist in Firebase Storage.

    Args:
        scene_id: Scene identifier
        expected_paths: List of relative paths expected to exist
        robot_type: Optional robot type for prefix construction
        prefix: Optional Firebase prefix override
        verify_checksums: If True, also verify SHA256 checksums
        local_path_map: Map of relative path -> local Path for checksum verification

    Returns:
        dict with keys: success, verified, missing, extra, checksum_mismatches, errors

    Raises:
        FirebaseUploadOrchestratorError: If verification fails and strict mode enabled
    """
    remote_prefix = build_firebase_upload_prefix(
        scene_id,
        robot_type=robot_type,
        prefix=prefix,
    )

    result = verify_firebase_upload_manifest(
        remote_prefix=remote_prefix,
        expected_paths=expected_paths,
        verify_checksums=verify_checksums,
        local_path_map=local_path_map,
    )

    if not result["success"]:
        logger.warning(
            "Firebase upload verification failed for %s: missing=%d, checksum_mismatches=%d",
            remote_prefix,
            len(result.get("missing", [])),
            len(result.get("checksum_mismatches", [])),
        )

    return result


def create_atomic_upload_transaction(scene_id: str) -> AtomicUploadTransaction:
    """Create a new atomic upload transaction for multi-robot uploads.

    Use this when uploading multiple robots to ensure all-or-nothing semantics.
    If any robot upload fails, call transaction.rollback() to clean up all
    successful uploads.

    Example:
        transaction = create_atomic_upload_transaction(scene_id)
        for robot_type in robot_types:
            try:
                result = upload_episodes_with_retry(...)
                transaction.record_success(robot_type, result)
            except Exception as exc:
                transaction.record_failure(robot_type, exc)
                if require_atomic:
                    transaction.rollback()
                    raise
        transaction.commit()

    Args:
        scene_id: Scene identifier for the transaction

    Returns:
        AtomicUploadTransaction for tracking the multi-robot upload
    """
    return AtomicUploadTransaction(scene_id=scene_id)


def require_atomic_upload() -> bool:
    """Check if atomic uploads are required for this environment.

    Returns True in production mode by default. Can be overridden with
    FIREBASE_REQUIRE_ATOMIC_UPLOAD environment variable.
    """
    return _resolve_require_atomic_upload()

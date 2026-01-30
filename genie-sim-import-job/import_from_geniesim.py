#!/usr/bin/env python3
"""
Genie Sim Episode Import Job.

This job completes the bidirectional Genie Sim integration by:
1. Polling for completed generation jobs
2. Downloading generated episodes
3. Validating episode quality
4. Converting to LeRobot format
5. Integrating with existing pipeline

This is the missing "import" side of the Genie Sim integration, which previously
only had export capabilities.

Environment Variables:
    BUCKET: GCS bucket name
    GENIE_SIM_JOB_ID: Job ID to import (if monitoring specific job)
    GENIE_SIM_POLL_INTERVAL: Polling interval in seconds (default: 30)
    OUTPUT_PREFIX: Output path for imported episodes (default: scenes/{scene_id}/episodes)
    MIN_EPISODES_REQUIRED: Minimum number of episodes required for import (default: 1)
    MIN_QUALITY_SCORE: Minimum quality score for import (default: from quality_config.json)
    ENABLE_VALIDATION: Enable quality validation (default: true)
    FILTER_LOW_QUALITY: Filter low-quality episodes during import (default: true)
    REQUIRE_LEROBOT: Treat LeRobot conversion failure as job failure (default: production/service mode)
    LEROBOT_SKIP_RATE_MAX: Max allowed LeRobot skip rate percentage (default: 0.0 in production)
    ENABLE_FIREBASE_UPLOAD: Enable Firebase Storage upload of local episodes (default: true)
    DISABLE_FIREBASE_UPLOAD: Explicitly disable Firebase uploads (overrides ENABLE_FIREBASE_UPLOAD)
    FIREBASE_STORAGE_BUCKET: Firebase Storage bucket name for uploads
    FIREBASE_SERVICE_ACCOUNT_JSON: Service account JSON payload for Firebase
    FIREBASE_SERVICE_ACCOUNT_PATH: Path to service account JSON for Firebase
    FIREBASE_UPLOAD_PREFIX: Remote prefix for Firebase uploads (default: datasets)
    FIREBASE_UPLOAD_MAX_WORKERS: Max workers for parallel Firebase uploads
    FIREBASE_VERIFY_CHECKSUMS: Verify Firebase upload checksums (default: true)
    ALLOW_PARTIAL_FIREBASE_UPLOADS: Allow Firebase uploads to proceed for
        successful robots even if others fail (default: true)
    ARTIFACTS_BY_ROBOT: JSON map of robot type to artifacts payload for multi-robot imports
    ALLOW_IDEMPOTENT_RETRY: Allow retrying a local import when a prior manifest
        indicates a failed or partial run (default: false)
    ENABLE_REALTIME_STREAMING: Enable real-time feedback streaming (default: false in production)
    REALTIME_STREAM_PROTOCOL: Streaming protocol (http_post, grpc, websocket, message_queue, file_watch)
    REALTIME_STREAM_ENDPOINT: Endpoint URL or path for streaming
    REALTIME_STREAM_API_KEY: API key for streaming authentication
    REALTIME_STREAM_BATCH_SIZE: Batch size for streaming episodes (default: 10)
    ENABLE_COSMOS_POLICY_EXPORT: Enable Cosmos Policy format export (default: true)
    COSMOS_POLICY_ACTION_CHUNK_SIZE: Action chunk size for Cosmos Policy (default: 16)
    COSMOS_POLICY_IMAGE_SIZE: Image resize target for Cosmos Policy (default: 256)
    COSMOS_POLICY_CAMERAS: Comma-separated camera list for Cosmos Policy (default: wrist,overhead)
    COSMOS_POLICY_FIREBASE_PREFIX: Firebase upload prefix for Cosmos Policy (default: datasets/cosmos_policy)
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import json
import os
import logging
import shutil
import sys
import tarfile
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tools.metrics.job_metrics_exporter import export_job_metrics

from import_manifest_utils import (
    ENV_SNAPSHOT_KEYS,
    MANIFEST_SCHEMA_DEFINITION,
    MANIFEST_SCHEMA_VERSION,
    build_directory_checksums,
    build_file_inventory,
    collect_provenance,
    compute_manifest_checksum,
    get_git_sha,
    get_episode_file_paths,
    get_lerobot_metadata_paths,
    snapshot_env,
    verify_checksums_manifest,
    verify_import_manifest_checksum,
)
from verify_import_manifest import verify_manifest
from genie_sim_import.constants import MIN_EPISODES_REQUIRED
from genie_sim_import.downloads import (
    _download_recordings_from_gcs,
    _load_local_job_metadata,
    _parse_gcs_uri,
    _relative_recordings_path,
    _resolve_gcs_output_path,
    _resolve_gcs_recordings_path,
    _resolve_local_output_dir,
    _resolve_local_path,
    _resolve_recordings_dir,
    _write_local_job_metadata,
)
from genie_sim_import.integrity import (
    _format_checksums_verification_errors,
    _resolve_firebase_verify_checksums,
    _sha256_file,
    _write_checksums_file,
)
from genie_sim_import.manifest import (
    _build_episode_hash_index_path,
    _compute_episode_content_hash,
    _compute_episode_content_manifest,
    _load_json_file,
    _load_existing_import_manifest,
    _load_import_manifest_with_migration,
    _lookup_episode_hash_index,
    _persist_episode_hash_index,
    _resolve_manifest_import_status,
    _update_import_manifest_dedup_summary,
    _update_import_manifest_firebase_summary,
    _update_import_manifest_status,
)
from genie_sim_import.reporting import (
    _aggregate_metrics_summary,
    _normalize_gcs_output_path,
    _relative_to_bundle,
    _resolve_asset_provenance_reference,
    _resolve_job_idempotency,
    _write_combined_import_manifest,
)

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.alerting import send_alert

from tools.dataset_regression.metrics import compute_regression_metrics
from tools.error_handling.retry import NonRetryableError, RetryConfig, RetryContext
from tools.error_handling.errors import classify_exception
from tools.error_handling.logging import log_pipeline_error
from tools.geniesim_adapter.local_framework import GeneratedEpisodeMetadata
from tools.metrics.pipeline_metrics import get_metrics
from tools.quality_gates.quality_gate import (
    QualityGate,
    QualityGateCheckpoint,
    QualityGateRegistry,
    QualityGateResult,
    QualityGateSeverity,
)
from tools.geniesim_adapter.multi_robot_config import validate_geniesim_robot_allowlist
from tools.geniesim_adapter.mock_mode import resolve_geniesim_mock_mode
from tools.config import load_pipeline_config
from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode
from tools.lerobot_format import LeRobotExportFormat, parse_lerobot_export_format
from tools.logging_config import init_logging
from tools.secret_store.secret_manager import get_secret_or_env
from tools.schema_migrations import (
    DATASET_INFO_SCHEMA_VERSION,
    SchemaMigrationError,
    migrate_dataset_info_payload,
    migrate_import_manifest_payload,
)
from tools.tracing.correlation import ensure_request_id
from tools.firebase_upload.firebase_upload_orchestrator import (
    AtomicUploadTransaction,
    FirebaseUploadOrchestratorError,
    build_firebase_upload_prefix,
    create_atomic_upload_transaction,
    require_atomic_upload,
    resolve_firebase_upload_prefix,
    upload_episodes_with_retry,
    verify_firebase_upload,
)
from tools.firebase_upload.uploader import (
    get_firebase_storage_bucket,
    get_firebase_upload_mode,
    preflight_firebase_connectivity,
    resolve_firebase_local_upload_root,
)
from tools.error_handling.job_wrapper import run_job_with_dead_letter_queue
from tools.tracing import init_tracing
from tools.gcs_upload import (
    calculate_file_md5_base64,
    upload_blob_from_filename,
    verify_blob_upload,
)
from tools.dataset_catalog import DatasetCatalogClient, build_dataset_document
from tools.cost_tracking import get_cost_tracker
from tools.validation.entrypoint_checks import validate_required_env_vars
from tools.utils.atomic_write import write_json_atomic, write_text_atomic
from tools.quality.quality_config import (
    DEFAULT_FILTER_LOW_QUALITY,
    DEFAULT_MIN_QUALITY_SCORE,
    QUALITY_CONFIG,
    ResolvedQualitySettings,
    resolve_quality_settings,
)
from tools.training import DataStreamConfig, DataStreamProtocol, RealtimeFeedbackLoop

# Import quality validation
try:
    sys.path.insert(0, str(REPO_ROOT / "episode-generation-job"))
    from quality_certificate import (
        QualityCertificate,
        TrajectoryQualityMetrics,
        VisualQualityMetrics,
        TaskQualityMetrics,
    )
    HAVE_QUALITY_VALIDATION = True
except ImportError:
    HAVE_QUALITY_VALIDATION = False


# =============================================================================
# Data Models
# =============================================================================

JOB_NAME = "genie-sim-import-job"
logger = logging.getLogger(__name__)
DEFAULT_VIDEO_CAMERA_ID = "camera"
VIDEO_CHUNK_SIZE = 1000


class DeliveryMarkerExistsError(RuntimeError):
    """Raised when a delivery marker already exists and retries are disallowed."""


def _resolve_debug_mode() -> bool:
    debug_flag = parse_bool_env(os.getenv("BLUEPRINT_DEBUG"))
    if debug_flag is None:
        debug_flag = parse_bool_env(os.getenv("DEBUG"), default=False)
    return bool(debug_flag)


def _resolve_realtime_stream_config(
    min_quality_score: float,
    production_mode: bool,
    log: logging.LoggerAdapter,
) -> Optional[DataStreamConfig]:
    raw_enabled = os.getenv("ENABLE_REALTIME_STREAMING")
    realtime_enabled = parse_bool_env(
        raw_enabled,
        default=not production_mode,
    )
    if not realtime_enabled:
        log.info(
            "Realtime streaming disabled (ENABLE_REALTIME_STREAMING=%s).",
            raw_enabled,
        )
        return None

    endpoint = os.getenv("REALTIME_STREAM_ENDPOINT")
    if not endpoint:
        log.warning(
            "Realtime streaming enabled but REALTIME_STREAM_ENDPOINT is not set; skipping."
        )
        return None

    protocol_raw = os.getenv(
        "REALTIME_STREAM_PROTOCOL",
        DataStreamProtocol.HTTP_POST.value,
    ).lower()
    try:
        protocol = DataStreamProtocol(protocol_raw)
    except ValueError:
        log.warning(
            "Unsupported REALTIME_STREAM_PROTOCOL=%s; defaulting to %s.",
            protocol_raw,
            DataStreamProtocol.HTTP_POST.value,
        )
        protocol = DataStreamProtocol.HTTP_POST

    batch_size = 10
    batch_size_raw = os.getenv("REALTIME_STREAM_BATCH_SIZE")
    if batch_size_raw:
        try:
            batch_size = int(batch_size_raw)
            if batch_size <= 0:
                raise ValueError("batch size must be positive")
        except ValueError as exc:
            log.warning(
                "Invalid REALTIME_STREAM_BATCH_SIZE=%s (%s); using default=%s.",
                batch_size_raw,
                exc,
                batch_size,
            )

    config = DataStreamConfig(
        protocol=protocol,
        endpoint_url=endpoint,
        api_key=os.getenv("REALTIME_STREAM_API_KEY") or None,
        batch_size=batch_size,
        min_quality_score=min_quality_score,
    )
    log.info(
        "Realtime streaming enabled (protocol=%s, endpoint=%s, batch_size=%s).",
        protocol.value,
        endpoint,
        batch_size,
    )
    return config


def _resolve_firebase_upload_suppression(
    *,
    partial_failure: bool,
    allow_partial_firebase_uploads: bool,
    fail_on_partial_error: bool,
    quality_gate_failures: List[str],
) -> tuple[bool, Optional[str], bool]:
    if quality_gate_failures:
        return True, "quality_gate_failure", False
    if partial_failure:
        if not allow_partial_firebase_uploads:
            reason = (
                "partial_failure_fail_on_partial_error"
                if fail_on_partial_error
                else "partial_failure_partial_uploads_disabled"
            )
            return True, reason, False
        return False, None, True
    return False, None, False


def _apply_production_filter_override(
    quality_settings: ResolvedQualitySettings,
    production_mode: bool,
    log: logging.Logger,
    env: Mapping[str, str],
) -> ResolvedQualitySettings:
    if not production_mode:
        return quality_settings
    if quality_settings.filter_low_quality:
        return quality_settings
    raw_value = env.get("FILTER_LOW_QUALITY") if env else None
    parsed = parse_bool_env(raw_value, default=None)
    if parsed is False:
        log.warning(
            "Production mode forces FILTER_LOW_QUALITY=true; overriding "
            "FILTER_LOW_QUALITY=%s.",
            raw_value,
        )
    return ResolvedQualitySettings(
        min_quality_score=quality_settings.min_quality_score,
        filter_low_quality=True,
        dimension_thresholds=quality_settings.dimension_thresholds,
        config=quality_settings.config,
    )


def _stream_realtime_episodes(
    loop: RealtimeFeedbackLoop,
    episodes: List[GeneratedEpisodeMetadata],
    job_id: str,
    scene_id: Optional[str],
    robot_type: Optional[str],
    log: logging.LoggerAdapter,
) -> None:
    if not episodes:
        log.info("Realtime streaming enabled but no validated episodes to stream.")
        return

    async def _run_stream() -> None:
        await loop.start()
        try:
            max_queue_size = max(loop.config.batch_size * 10, 100)
            for episode in episodes:
                payload = {
                    "episode_id": episode.episode_id,
                    "task_name": episode.task_name,
                    "frame_count": episode.frame_count,
                    "duration_seconds": episode.duration_seconds,
                    "validation_passed": episode.validation_passed,
                    "file_size_bytes": episode.file_size_bytes,
                    "quality_score": episode.quality_score,
                    "quality_components": episode.quality_components,
                    "job_id": job_id,
                    "scene_id": scene_id,
                    "robot_type": robot_type,
                }
                loop.queue_episode(payload, quality_score=episode.quality_score)
                while True:
                    async with loop.queue_lock:
                        queue_size = len(loop.queue)
                    if queue_size < max_queue_size:
                        break
                    log.warning(
                        "Realtime streaming queue at %s episodes (max=%s); pausing enqueue.",
                        queue_size,
                        max_queue_size,
                    )
                    await asyncio.sleep(0.5)
        finally:
            await loop.stop()

    try:
        asyncio.run(_run_stream())
    except RuntimeError as exc:
        log.warning("Realtime streaming skipped: %s", exc)


def _is_service_mode() -> bool:
    return (
        os.getenv("SERVICE_MODE", "").lower() in {"1", "true", "yes", "y", "on"}
        or os.getenv("K_SERVICE") is not None
        or os.getenv("KUBERNETES_SERVICE_HOST") is not None
    )


def _resolve_upload_file_list(
    output_dir: Path,
    deduplicated_episode_ids: List[str],
) -> List[Path]:
    if not deduplicated_episode_ids:
        return [path for path in sorted(output_dir.rglob("*")) if path.is_file()]

    deduped_set = set(deduplicated_episode_ids)
    excluded_paths: set[Path] = set()
    recordings_dir = output_dir / "recordings"
    if recordings_dir.exists():
        for path in recordings_dir.rglob("*"):
            if path.is_file() and path.stem in deduped_set:
                excluded_paths.add(path)

    lerobot_dir = output_dir / "lerobot"
    dataset_info_path = lerobot_dir / "dataset_info.json"
    if dataset_info_path.exists():
        dataset_info = _load_dataset_info_with_migration(
            dataset_info_path,
            logger,
            required=True,
        ) or {}
        for entry in dataset_info.get("episodes", []):
            episode_id = entry.get("episode_id")
            episode_file = entry.get("file")
            if episode_id in deduped_set and episode_file:
                excluded_paths.add(lerobot_dir / episode_file)

    return [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path not in excluded_paths
    ]


def _prepare_deduplication_summary(
    *,
    result: ImportResult,
    scene_id: str,
    robot_type: Optional[str],
    prefix: str,
    log: logging.LoggerAdapter,
) -> Dict[str, Any]:
    episode_hashes = result.episode_content_hashes
    duplicates = _lookup_episode_hash_index(
        episode_hashes=episode_hashes,
        scene_id=scene_id,
        robot_type=robot_type,
        prefix=prefix,
        log=log,
    )
    deduplicated_episode_ids = sorted(duplicates.keys())
    result.deduplicated_episode_ids = deduplicated_episode_ids
    base_prefix = build_firebase_upload_prefix(
        scene_id,
        robot_type=robot_type,
        prefix=prefix,
    )
    return {
        "strategy": "firebase_storage_index_v1",
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "scene_id": scene_id,
        "robot_type": robot_type,
        "index_prefix": f"{base_prefix}/episode_hash_index",
        "deduplicated_count": len(deduplicated_episode_ids),
        "deduplicated_episode_ids": deduplicated_episode_ids,
        "deduplicated_hashes": duplicates,
        "content_hash_count": len(episode_hashes),
    }


def _filter_lerobot_episode_metadata(
    lerobot_dir: Path,
    removed_episode_indices: List[int],
    log: logging.LoggerAdapter,
) -> int:
    if not removed_episode_indices:
        return 0

    meta_root = lerobot_dir / "meta" / "episodes"
    if not meta_root.exists():
        return 0

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError:
        log.warning(
            "Skipping LeRobot meta episode filtering; pyarrow is not available.",
        )
        return 0

    filtered_count = 0
    remove_set = set(removed_episode_indices)
    value_set = pa.array(sorted(remove_set))
    for meta_file in sorted(meta_root.rglob("*.parquet")):
        try:
            table = pq.read_table(meta_file)
        except Exception as exc:
            log.warning("Failed to read LeRobot metadata %s: %s", meta_file, exc)
            continue
        if "episode_index" not in table.schema.names:
            continue
        mask = ~pc.is_in(table["episode_index"], value_set=value_set)
        filtered_table = table.filter(mask)
        removed_rows = table.num_rows - filtered_table.num_rows
        if removed_rows <= 0:
            continue
        filtered_count += removed_rows
        if filtered_table.num_rows == 0:
            meta_file.unlink(missing_ok=True)
        else:
            pq.write_table(filtered_table, meta_file)
    return filtered_count


def _filter_deduplicated_outputs(
    output_dir: Path,
    deduplicated_episode_ids: List[str],
    log: logging.LoggerAdapter,
) -> Dict[str, Any]:
    if not deduplicated_episode_ids:
        return {
            "filtered_count": 0,
            "filtered_episode_ids": [],
            "filtered_from_outputs": False,
        }

    lerobot_dir = output_dir / "lerobot"
    dataset_info_path = lerobot_dir / "dataset_info.json"
    if not dataset_info_path.exists():
        log.warning(
            "LeRobot dataset_info.json missing; cannot filter deduplicated episodes."
        )
        return {
            "filtered_count": 0,
            "filtered_episode_ids": [],
            "filtered_from_outputs": False,
        }

    dataset_info = _load_dataset_info_with_migration(
        dataset_info_path,
        log,
        required=True,
    ) or {}
    episodes = dataset_info.get("episodes", [])
    if not isinstance(episodes, list):
        log.warning(
            "LeRobot dataset_info episodes payload is invalid; skipping dedup filtering."
        )
        return {
            "filtered_count": 0,
            "filtered_episode_ids": [],
            "filtered_from_outputs": False,
        }

    deduped_set = set(deduplicated_episode_ids)
    removed_entries = [
        entry for entry in episodes if entry.get("episode_id") in deduped_set
    ]
    if not removed_entries:
        return {
            "filtered_count": 0,
            "filtered_episode_ids": [],
            "filtered_from_outputs": False,
        }

    kept_entries = [
        entry for entry in episodes if entry.get("episode_id") not in deduped_set
    ]
    removed_indices: List[int] = []
    removed_files: List[str] = []
    for entry in removed_entries:
        file_name = entry.get("file")
        if file_name:
            removed_files.append(str(file_name))
            (lerobot_dir / file_name).unlink(missing_ok=True)
        video_paths = entry.get("video_paths")
        if isinstance(video_paths, dict):
            for video_path in video_paths.values():
                if video_path:
                    (lerobot_dir / video_path).unlink(missing_ok=True)
        episode_index = entry.get("episode_index")
        if isinstance(episode_index, int):
            removed_indices.append(episode_index)

    dataset_info["episodes"] = kept_entries
    dataset_info["total_episodes"] = len(kept_entries)
    frame_counts = [
        entry.get("num_frames")
        for entry in kept_entries
        if isinstance(entry.get("num_frames"), int)
    ]
    if frame_counts and len(frame_counts) == len(kept_entries):
        dataset_info["total_frames"] = int(sum(frame_counts))
    quality_scores = [
        entry.get("quality_score")
        for entry in kept_entries
        if isinstance(entry.get("quality_score"), (int, float))
    ]
    if quality_scores:
        dataset_info["average_quality_score"] = float(np.mean(quality_scores))
        dataset_info["min_quality_score"] = float(np.min(quality_scores))
        dataset_info["max_quality_score"] = float(np.max(quality_scores))

    write_json_atomic(dataset_info_path, dataset_info, indent=2)

    episodes_file = lerobot_dir / "episodes.jsonl"
    episode_lines = [json.dumps(ep_info) for ep_info in kept_entries]
    episodes_payload = "\n".join(episode_lines)
    if episodes_payload:
        episodes_payload += "\n"
    write_text_atomic(episodes_file, episodes_payload)

    filtered_meta_count = _filter_lerobot_episode_metadata(
        lerobot_dir,
        removed_indices,
        log,
    )

    log.info(
        "Filtered %s deduplicated episode(s) from LeRobot outputs (files=%s, meta=%s).",
        len(removed_entries),
        len(removed_files),
        filtered_meta_count,
    )
    return {
        "filtered_count": len(removed_entries),
        "filtered_episode_ids": [
            entry.get("episode_id")
            for entry in removed_entries
            if entry.get("episode_id")
        ],
        "filtered_from_outputs": True,
        "lerobot_total_episodes": dataset_info.get("total_episodes"),
        "lerobot_total_frames": dataset_info.get("total_frames"),
        "quality_average_score": dataset_info.get("average_quality_score"),
        "quality_min_score": dataset_info.get("min_quality_score"),
        "quality_max_score": dataset_info.get("max_quality_score"),
    }


def _apply_deduplication_filters(
    result: ImportResult,
    dedup_summary: Dict[str, Any],
    log: logging.LoggerAdapter,
) -> Dict[str, Any]:
    deduplicated_ids = dedup_summary.get("deduplicated_episode_ids") or []
    if not deduplicated_ids or result.output_dir is None:
        return dedup_summary

    filter_result = _filter_deduplicated_outputs(
        result.output_dir,
        deduplicated_ids,
        log,
    )
    filtered_count = int(filter_result.get("filtered_count", 0))
    if filtered_count:
        result.episodes_passed_validation = max(
            0,
            result.episodes_passed_validation - filtered_count,
        )
        result.episodes_filtered += filtered_count
        avg_score = filter_result.get("quality_average_score")
        min_score = filter_result.get("quality_min_score")
        max_score = filter_result.get("quality_max_score")
        if isinstance(avg_score, (int, float)):
            result.average_quality_score = float(avg_score)
        if isinstance(min_score, (int, float)):
            result.quality_min_score = float(min_score)
        if isinstance(max_score, (int, float)):
            result.quality_max_score = float(max_score)

    dedup_summary = dict(dedup_summary)
    dedup_summary["filtered_from_outputs"] = filter_result.get(
        "filtered_from_outputs",
        False,
    )
    dedup_summary["filtered_count"] = filtered_count
    dedup_summary["filtered_episode_ids"] = filter_result.get(
        "filtered_episode_ids",
        [],
    )
    if filter_result.get("lerobot_total_episodes") is not None:
        dedup_summary["post_dedup_lerobot_converted_count"] = filter_result.get(
            "lerobot_total_episodes"
        )
    if filter_result.get("lerobot_total_frames") is not None:
        dedup_summary["post_dedup_lerobot_total_frames"] = filter_result.get(
            "lerobot_total_frames"
        )
    dedup_summary["post_dedup_passed_validation"] = result.episodes_passed_validation
    dedup_summary["post_dedup_filtered"] = result.episodes_filtered
    return dedup_summary


def _upload_robot_payload_to_firebase(
    *,
    payload: Dict[str, Any],
    scene_id: str,
    job_id: str,
    firebase_upload_prefix: str,
    allow_partial_firebase_uploads: bool,
    fail_on_partial_error: bool,
    log: logging.LoggerAdapter,
) -> tuple[str, Optional[Dict[str, Any]], Optional[BaseException]]:
    robot_type = payload["robot_type"]
    result = payload["result"]
    entry = payload["entry"]
    if not result.success:
        return robot_type, None, None

    dedup_summary = _prepare_deduplication_summary(
        result=result,
        scene_id=scene_id,
        robot_type=robot_type,
        prefix=firebase_upload_prefix,
        log=log,
    )
    dedup_summary = _apply_deduplication_filters(
        result,
        dedup_summary,
        log,
    )
    deduplicated_ids = dedup_summary.get("deduplicated_episode_ids", [])
    entry["episodes"]["passed_validation"] = result.episodes_passed_validation
    entry["episodes"]["filtered"] = result.episodes_filtered
    entry["episodes"]["deduplicated"] = len(deduplicated_ids)
    entry["episodes"]["deduplicated_ids"] = deduplicated_ids
    entry["episodes"]["deduplicated_filtered_from_outputs"] = dedup_summary.get(
        "filtered_from_outputs",
        False,
    )
    entry["episodes"]["deduplicated_filtered"] = dedup_summary.get(
        "filtered_count",
        0,
    )
    entry["episodes"]["deduplicated_filtered_ids"] = dedup_summary.get(
        "filtered_episode_ids",
        [],
    )
    _update_import_manifest_dedup_summary(
        result.import_manifest_path,
        dedup_summary,
    )
    if deduplicated_ids:
        log.info(
            "Deduplicating %s episode(s) for robot %s via content hash index.",
            len(deduplicated_ids),
            robot_type,
        )
    upload_file_paths = _resolve_upload_file_list(
        result.output_dir,
        deduplicated_ids,
    )
    expected_paths = [
        path.relative_to(result.output_dir).as_posix() for path in upload_file_paths
    ]
    local_path_map = {
        rel_path: path.resolve()
        for rel_path, path in zip(expected_paths, upload_file_paths)
    }
    verify_checksums = _resolve_firebase_verify_checksums(
        os.getenv("FIREBASE_VERIFY_CHECKSUMS"),
    )
    log.info(
        "Uploading episodes to Firebase Storage for robot %s...",
        robot_type,
    )
    try:
        firebase_result = upload_episodes_with_retry(
            episodes_dir=result.output_dir,
            scene_id=scene_id,
            robot_type=robot_type,
            prefix=firebase_upload_prefix,
            file_paths=upload_file_paths,
        )
        firebase_summary = firebase_result.summary
        try:
            firebase_verification = verify_firebase_upload(
                scene_id=scene_id,
                robot_type=robot_type,
                prefix=firebase_upload_prefix,
                expected_paths=expected_paths,
                verify_checksums=verify_checksums,
                local_path_map=local_path_map,
            )
        except Exception as exc:
            firebase_verification = {
                "success": False,
                "verified": [],
                "missing": [],
                "extra": [],
                "checksum_mismatches": [],
                "errors": [str(exc)],
            }
        firebase_summary["firebase_verification"] = firebase_verification
    except Exception as exc:
        _alert_firebase_upload_failure(
            scene_id=scene_id,
            job_id=job_id,
            robot_type=robot_type,
            error=str(exc),
        )
        log.error("Firebase upload failed: %s", exc)
        return robot_type, None, exc

    firebase_summary["deduplicated_episodes"] = len(deduplicated_ids)
    firebase_summary["deduplicated_episode_ids"] = deduplicated_ids
    firebase_summary["remote_prefix"] = firebase_result.remote_prefix
    _update_import_manifest_firebase_summary(
        result.import_manifest_path,
        firebase_summary,
    )
    firebase_verification = firebase_summary.get("firebase_verification") or {}
    if firebase_verification and not firebase_verification.get("success", True):
        verification_errors = _format_firebase_verification_errors(firebase_verification)
        error_message = "Firebase upload verification failed"
        if verification_errors:
            error_message += f": {'; '.join(verification_errors)}"
        log.warning("%s for robot %s.", error_message, robot_type)
        entry.setdefault("errors", []).append(error_message)
        if _should_fail_firebase_verification(
            allow_partial_firebase_uploads,
            fail_on_partial_error,
        ):
            return (
                robot_type,
                firebase_summary,
                FirebaseUploadOrchestratorError(error_message),
            )
    return robot_type, firebase_summary, None


def _run_firebase_uploads_for_robot_payloads(
    *,
    robot_results: List[Dict[str, Any]],
    scene_id: str,
    job_id: str,
    firebase_upload_prefix: str,
    firebase_upload_max_workers: int,
    allow_partial_firebase_uploads: bool,
    fail_on_partial_error: bool,
    log: logging.LoggerAdapter,
    overall_success: bool,
) -> bool:
    """Upload robot payloads to Firebase with atomic transaction support.

    P0 FIX: Implements atomic uploads - if any robot fails and atomic uploads
    are required, ALL successful uploads are rolled back to prevent partial
    dataset delivery to labs.

    Atomic upload behavior:
    - In production mode (default): If ANY robot upload fails, ALL successful
      uploads are rolled back automatically.
    - Set FIREBASE_REQUIRE_ATOMIC_UPLOAD=false to disable (not recommended).
    - ALLOW_PARTIAL_FIREBASE_UPLOADS only affects error handling AFTER atomic
      rollback decision - it cannot prevent rollback in production.
    """
    successful_payloads = [
        payload for payload in robot_results if payload["result"].success
    ]
    if not successful_payloads:
        return overall_success

    # P0 FIX: Create atomic transaction for multi-robot uploads
    atomic_required = require_atomic_upload()
    transaction = create_atomic_upload_transaction(scene_id) if atomic_required else None

    if atomic_required:
        log.info(
            "Atomic Firebase upload enabled for scene %s with %d robot(s). "
            "All uploads will be rolled back if any robot fails.",
            scene_id,
            len(successful_payloads),
        )

    firebase_upload_failures: List[tuple[str, BaseException]] = []
    successful_uploads: List[Dict[str, Any]] = []  # Track for potential rollback

    with ThreadPoolExecutor(max_workers=firebase_upload_max_workers) as executor:
        futures = {
            executor.submit(
                _upload_robot_payload_to_firebase,
                payload=payload,
                scene_id=scene_id,
                job_id=job_id,
                firebase_upload_prefix=firebase_upload_prefix,
                allow_partial_firebase_uploads=allow_partial_firebase_uploads,
                fail_on_partial_error=fail_on_partial_error,
                log=log,
            ): payload
            for payload in successful_payloads
        }
        for future in as_completed(futures):
            payload = futures[future]
            entry = payload["entry"]
            result = payload["result"]
            robot_type = payload.get("robot_type", "unknown")
            try:
                resolved_robot_type, firebase_summary, error = future.result()
            except Exception as exc:
                _alert_firebase_upload_failure(
                    scene_id=scene_id,
                    job_id=job_id,
                    robot_type=robot_type,
                    error=str(exc),
                )
                log.error("Firebase upload failed for robot %s: %s", robot_type, exc)
                entry.setdefault("errors", []).append(f"Firebase upload failed: {exc}")
                firebase_upload_failures.append((robot_type, exc))
                if transaction:
                    transaction.record_failure(robot_type, exc)
                overall_success = False
                continue
            if resolved_robot_type:
                robot_type = resolved_robot_type
            if error is not None:
                log.error("Firebase upload failed for robot %s: %s", robot_type, error)
                entry.setdefault("errors", []).append(
                    f"Firebase upload failed: {error}"
                )
                firebase_upload_failures.append((robot_type, error))
                if transaction:
                    transaction.record_failure(robot_type, error)
                overall_success = False
                continue
            if not firebase_summary:
                continue

            # P0 FIX: Track successful upload for atomic transaction
            remote_prefix = firebase_summary.get("remote_prefix") or build_firebase_upload_prefix(
                scene_id, robot_type=robot_type, prefix=firebase_upload_prefix
            )
            if transaction:
                from tools.firebase_upload.firebase_upload_orchestrator import FirebaseUploadResult
                upload_result = FirebaseUploadResult(
                    summary=firebase_summary,
                    remote_prefix=remote_prefix,
                )
                transaction.record_success(robot_type, upload_result)

            successful_uploads.append({
                "robot_type": robot_type,
                "remote_prefix": remote_prefix,
                "entry": entry,
                "result": result,
                "firebase_summary": firebase_summary,
            })

            entry["firebase_upload"] = firebase_summary
            log.info(
                "Firebase upload complete: uploaded=%s skipped=%s reuploaded=%s failed=%s total=%s",
                firebase_summary.get("uploaded", 0),
                firebase_summary.get("skipped", 0),
                firebase_summary.get("reuploaded", 0),
                firebase_summary.get("failed", 0),
                firebase_summary.get("total_files", 0),
            )
            failed_count = firebase_summary.get("failed", 0)
            verification_failed = False
            verification_should_fail = False
            if failed_count > 0:
                _alert_firebase_upload_failure(
                    scene_id=scene_id,
                    job_id=job_id,
                    robot_type=robot_type,
                    error=f"{failed_count} file(s) failed to upload to Firebase",
                )
                log.error(
                    "Firebase upload incomplete for robot %s: %d file(s) failed.",
                    robot_type,
                    failed_count,
                )
                entry.setdefault("errors", []).append(
                    f"Firebase upload incomplete: {failed_count} file(s) failed"
                )
                firebase_upload_failures.append(
                    (robot_type, FirebaseUploadOrchestratorError(f"{failed_count} file(s) failed"))
                )
                if transaction:
                    transaction.record_failure(
                        robot_type,
                        FirebaseUploadOrchestratorError(f"{failed_count} file(s) failed to upload"),
                    )
                overall_success = False
            firebase_verification = firebase_summary.get("firebase_verification") or {}
            if firebase_verification and not firebase_verification.get("success", True):
                verification_failed = True
                verification_should_fail = _should_fail_firebase_verification(
                    allow_partial_firebase_uploads,
                    fail_on_partial_error,
                )
                verification_errors = _format_firebase_verification_errors(
                    firebase_verification
                )
                error_message = "Firebase upload verification failed"
                if verification_errors:
                    error_message += f": {'; '.join(verification_errors)}"
                entry.setdefault("errors", []).append(error_message)
                firebase_upload_failures.append(
                    (robot_type, FirebaseUploadOrchestratorError(error_message))
                )
                if transaction:
                    transaction.record_failure(
                        robot_type,
                        FirebaseUploadOrchestratorError(error_message),
                    )
                if verification_should_fail:
                    overall_success = False

    # P0 FIX: Atomic rollback if any failures occurred
    if transaction and transaction.should_rollback():
        failed_robot_types = sorted({robot for robot, _ in firebase_upload_failures})
        log.error(
            "Atomic Firebase upload: Rolling back %d successful upload(s) because "
            "robot(s) %s failed. This prevents partial dataset delivery.",
            len(transaction.successful_prefixes),
            failed_robot_types,
        )
        rollback_result = transaction.rollback()
        log.info(
            "Atomic rollback complete: status=%s, cleaned_prefixes=%d, errors=%d",
            rollback_result.get("status"),
            len(rollback_result.get("prefixes", [])),
            len(rollback_result.get("errors", [])),
        )

        # Clear successful uploads since they've been rolled back
        for upload_info in successful_uploads:
            entry = upload_info["entry"]
            entry["firebase_upload"] = {
                "status": "rolled_back",
                "reason": f"Atomic rollback due to failure of robot(s): {failed_robot_types}",
                "rollback_result": rollback_result,
            }
            entry.setdefault("errors", []).append(
                f"Firebase upload rolled back due to atomic transaction failure"
            )

        raise FirebaseUploadOrchestratorError(
            f"Atomic Firebase upload failed: {len(firebase_upload_failures)} robot(s) failed "
            f"({', '.join(failed_robot_types)}). All {len(transaction.successful_prefixes)} "
            f"successful upload(s) have been rolled back to prevent partial dataset delivery."
        )

    # Commit transaction if no failures
    if transaction:
        transaction.commit()
        log.info(
            "Atomic Firebase upload committed: %d robot(s) uploaded successfully.",
            len(transaction.successful_results),
        )

    # Process successful uploads (persist hash index, publish catalog)
    for upload_info in successful_uploads:
        robot_type = upload_info["robot_type"]
        entry = upload_info["entry"]
        result = upload_info["result"]
        firebase_summary = upload_info["firebase_summary"]

        # Check if this robot had failures (already tracked above)
        robot_failed = any(r == robot_type for r, _ in firebase_upload_failures)
        if robot_failed:
            continue

        if firebase_summary.get("deduplicated_episode_ids"):
            deduplicated_ids = firebase_summary.get(
                "deduplicated_episode_ids",
                [],
            )
            new_hashes = {
                episode_id: content_hash
                for episode_id, content_hash in result.episode_content_hashes.items()
                if episode_id not in deduplicated_ids
            }
            _persist_episode_hash_index(
                episode_hashes=new_hashes,
                scene_id=scene_id,
                robot_type=robot_type,
                prefix=firebase_upload_prefix,
                job_id=job_id,
                log=log,
            )
        else:
            _persist_episode_hash_index(
                episode_hashes=result.episode_content_hashes,
                scene_id=scene_id,
                robot_type=robot_type,
                prefix=firebase_upload_prefix,
                job_id=job_id,
                log=log,
            )
        _publish_dataset_catalog_document(
            scene_id=scene_id,
            job_id=job_id,
            robot_type=robot_type,
            result=result,
            firebase_summary=firebase_summary,
            gcs_output_path=entry.get("gcs_output_path"),
            log=log,
        )

    if firebase_upload_failures and not atomic_required:
        failed_robot_types = sorted({robot for robot, _ in firebase_upload_failures})
        if allow_partial_firebase_uploads:
            log.warning(
                "Firebase uploads failed for robot(s) %s; continuing because "
                "ALLOW_PARTIAL_FIREBASE_UPLOADS=true and atomic uploads not required.",
                failed_robot_types,
            )
        else:
            raise FirebaseUploadOrchestratorError(
                "Firebase uploads failed for robot types: "
                + ", ".join(failed_robot_types)
            )

    return overall_success


def _resolve_run_id(job_id: str) -> str:
    return (
        os.getenv("RUN_ID")
        or os.getenv("GENIE_SIM_RUN_ID")
        or os.getenv("GENIESIM_RUN_ID")
        or job_id
    )


@dataclass(frozen=True)
class DeliveryMarkerResult:
    payload: Dict[str, Any]
    gcs_uri: Optional[str]
    local_path: Optional[Path]
    marker_exists: bool


def _delivery_marker_object_path(scene_id: str, run_id: str) -> str:
    return f"scenes/{scene_id}/geniesim/delivery/{run_id}.json"


def _resolve_delivery_marker_gcs_uri(
    bucket: Optional[str],
    scene_id: str,
    run_id: str,
) -> Optional[str]:
    if not bucket:
        return None
    return f"gs://{bucket}/{_delivery_marker_object_path(scene_id, run_id)}"


def _resolve_delivery_marker_local_path(
    bucket: Optional[str],
    scene_id: str,
    run_id: str,
    *,
    local_root: Optional[Path] = None,
) -> Optional[Path]:
    if not bucket:
        return None
    object_path = _delivery_marker_object_path(scene_id, run_id)
    if local_root is not None:
        return local_root / bucket / object_path
    return _resolve_local_path(bucket, object_path)


def _build_delivery_marker_payload(
    *,
    scene_id: str,
    job_id: str,
    run_id: str,
    idempotency_key: Optional[str],
) -> Dict[str, Any]:
    return {
        "scene_id": scene_id,
        "job_id": job_id,
        "run_id": run_id,
        "idempotency_key": idempotency_key,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _is_gcs_precondition_failure(exc: BaseException) -> bool:
    if hasattr(exc, "code") and getattr(exc, "code") == 412:
        return True
    if hasattr(exc, "status_code") and getattr(exc, "status_code") == 412:
        return True
    return False


def _load_delivery_marker_payload(
    *,
    bucket: Optional[str],
    scene_id: str,
    run_id: str,
    storage_client: Optional[Any] = None,
    local_path: Optional[Path] = None,
    log: Optional[logging.LoggerAdapter] = None,
) -> Optional[Dict[str, Any]]:
    if bucket and storage_client is not None:
        try:
            blob = storage_client.bucket(bucket).blob(
                _delivery_marker_object_path(scene_id, run_id)
            )
            if not blob.exists():
                return None
            payload_text = blob.download_as_text()
            return json.loads(payload_text)
        except Exception as exc:
            if log:
                log.warning("Failed to read delivery marker from GCS: %s", exc)
            return None
    if local_path and local_path.exists():
        try:
            with open(local_path, "r") as handle:
                return json.load(handle)
        except Exception as exc:
            if log:
                log.warning("Failed to read local delivery marker: %s", exc)
            return None
    return None


def _update_import_manifest_delivery(
    manifest_path: Optional[Path],
    delivery_payload: Dict[str, Any],
    *,
    gcs_uri: Optional[str],
    local_path: Optional[Path],
    marker_exists: bool,
) -> None:
    if manifest_path is None or not manifest_path.exists():
        return
    with open(manifest_path, "r") as handle:
        import_manifest = json.load(handle)
    import_manifest["delivery"] = {
        "marker": delivery_payload,
        "gcs_uri": gcs_uri,
        "local_path": str(local_path) if local_path else None,
        "marker_exists": marker_exists,
        "recorded_at": datetime.utcnow().isoformat() + "Z",
    }
    checksums = import_manifest.setdefault("checksums", {})
    metadata_checksums = checksums.setdefault("metadata", {})
    metadata_checksums.setdefault("import_manifest.json", {})
    metadata_checksums["import_manifest.json"]["sha256"] = compute_manifest_checksum(
        import_manifest
    )
    write_json_atomic(manifest_path, import_manifest, indent=2)


def _write_delivery_marker(
    *,
    bucket: Optional[str],
    scene_id: str,
    job_id: str,
    run_id: str,
    idempotency_key: Optional[str],
    allow_idempotent_retry: bool,
    log: logging.LoggerAdapter,
    import_manifest_paths: Optional[List[Path]] = None,
    storage_client: Optional[Any] = None,
    local_root: Optional[Path] = None,
) -> DeliveryMarkerResult:
    marker_payload = _build_delivery_marker_payload(
        scene_id=scene_id,
        job_id=job_id,
        run_id=run_id,
        idempotency_key=idempotency_key,
    )
    gcs_uri = _resolve_delivery_marker_gcs_uri(bucket, scene_id, run_id)
    local_path = _resolve_delivery_marker_local_path(
        bucket,
        scene_id,
        run_id,
        local_root=local_root,
    )

    gcs_blob = None
    if bucket:
        if storage_client is None:
            try:
                from google.cloud import storage  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"google-cloud-storage unavailable for delivery marker: {exc}"
                ) from exc
            storage_client = storage.Client()
        gcs_blob = storage_client.bucket(bucket).blob(
            _delivery_marker_object_path(scene_id, run_id)
        )

    marker_exists = False
    if gcs_blob is not None and gcs_blob.exists():
        marker_exists = True
    elif local_path is not None and local_path.exists():
        marker_exists = True

    if marker_exists:
        existing_payload = _load_delivery_marker_payload(
            bucket=bucket,
            scene_id=scene_id,
            run_id=run_id,
            storage_client=storage_client,
            local_path=local_path,
            log=log,
        )
        if not allow_idempotent_retry:
            raise DeliveryMarkerExistsError(
                f"Delivery marker already exists for scene {scene_id} "
                f"run {run_id}. Set ALLOW_IDEMPOTENT_RETRY=true to proceed."
            )
        if existing_payload:
            marker_payload = existing_payload
        log.warning(
            "Delivery marker already exists for scene %s (run_id=%s); "
            "continuing due to ALLOW_IDEMPOTENT_RETRY=true.",
            scene_id,
            run_id,
        )
    else:
        if gcs_blob is not None:
            try:
                gcs_blob.upload_from_string(
                    json.dumps(marker_payload, indent=2),
                    content_type="application/json",
                    if_generation_match=0,
                )
            except Exception as exc:
                if _is_gcs_precondition_failure(exc):
                    raise DeliveryMarkerExistsError(
                        f"Delivery marker already exists for scene {scene_id} "
                        f"run {run_id}."
                    ) from exc
                raise
        if local_path is not None:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(local_path, marker_payload, indent=2)
            except Exception as exc:
                log.warning("Failed to write local delivery marker: %s", exc)

    for manifest_path in import_manifest_paths or []:
        _update_import_manifest_delivery(
            manifest_path,
            marker_payload,
            gcs_uri=gcs_uri,
            local_path=local_path,
            marker_exists=marker_exists,
        )

    return DeliveryMarkerResult(
        payload=marker_payload,
        gcs_uri=gcs_uri,
        local_path=local_path,
        marker_exists=marker_exists,
    )


def _resolve_export_schema_version() -> str:
    return (
        os.getenv("GENIESIM_EXPORT_SCHEMA_VERSION")
        or os.getenv("EXPORT_SCHEMA_VERSION")
        or "1.0.0"
    )


def _resolve_skip_rate_max(raw_value: Optional[str]) -> float:
    if raw_value is None:
        production_mode = resolve_production_mode()
        return 0.0 if production_mode else 100.0
    try:
        skip_rate = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid LEROBOT_SKIP_RATE_MAX value: {raw_value}") from exc
    if skip_rate < 0.0:
        raise ValueError("LEROBOT_SKIP_RATE_MAX must be >= 0")
    return skip_rate


def _resolve_min_episodes_required(raw_value: Optional[str]) -> int:
    if raw_value is None:
        return MIN_EPISODES_REQUIRED
    try:
        min_episodes = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid MIN_EPISODES_REQUIRED value: {raw_value}") from exc
    if min_episodes < 1:
        raise ValueError("MIN_EPISODES_REQUIRED must be >= 1")
    return min_episodes


def _resolve_firebase_upload_max_workers(raw_value: Optional[str]) -> Optional[int]:
    if raw_value is None or raw_value == "":
        return None
    try:
        max_workers = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid FIREBASE_UPLOAD_MAX_WORKERS value: {raw_value}"
        ) from exc
    if max_workers < 1:
        raise ValueError("FIREBASE_UPLOAD_MAX_WORKERS must be >= 1")
    return max_workers


def _should_fail_firebase_verification(
    allow_partial_firebase_uploads: bool,
    fail_on_partial_error: bool,
) -> bool:
    if not allow_partial_firebase_uploads:
        return True
    return fail_on_partial_error


def _format_firebase_verification_errors(result: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for error in result.get("errors") or []:
        errors.append(str(error))
    if result.get("missing"):
        errors.append("Missing files: " + ", ".join(result["missing"]))
    if result.get("checksum_mismatches"):
        mismatch_paths = [
            mismatch.get("path")
            for mismatch in result["checksum_mismatches"]
            if isinstance(mismatch, dict) and mismatch.get("path")
        ]
        if mismatch_paths:
            errors.append("Checksum mismatches: " + ", ".join(mismatch_paths))
    if result.get("extra"):
        errors.append("Extra files: " + ", ".join(result["extra"]))
    return errors


@dataclass(frozen=True)
class RequireLerobotResolution:
    value: bool
    default: bool
    source: str
    raw_value: Optional[str]


def _resolve_require_lerobot(
    raw_value: Optional[str],
    *,
    production_mode: bool,
    service_mode: bool,
) -> RequireLerobotResolution:
    default_value = production_mode or service_mode
    if raw_value is None:
        return RequireLerobotResolution(
            value=default_value,
            default=default_value,
            source="default",
            raw_value=None,
        )
    resolved = parse_bool_env(raw_value, default=default_value)
    return RequireLerobotResolution(
        value=resolved,
        default=default_value,
        source="env",
        raw_value=raw_value,
    )


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return parse_bool_env(value, default=None)
    return None


def _coerce_threshold_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    thresholds: Dict[str, float] = {}
    for key, raw_value in value.items():
        coerced = _coerce_float(raw_value)
        if coerced is None:
            continue
        thresholds[str(key)] = coerced
    return thresholds


def _coerce_quality_component_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_collision_component(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int) and not isinstance(value, bool):
        return 1.0 if value <= 0 else 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0.0:
        return None
    if numeric > 1.0:
        return 0.0
    return numeric


def _extract_quality_components(payload: Dict[str, Any]) -> Dict[str, float]:
    components: Dict[str, float] = {}
    for source_key in ("quality_components", "quality_metrics", "quality_scores"):
        raw = payload.get(source_key)
        if not isinstance(raw, dict):
            continue
        for key, value in raw.items():
            if key == "collision_count":
                normalized = _normalize_collision_component(value)
                if normalized is not None:
                    components[key] = normalized
                continue
            coerced = _coerce_quality_component_value(value)
            if coerced is not None:
                components[key] = coerced

    cert = payload.get("quality_certificate")
    if isinstance(cert, dict):
        trajectory = cert.get("trajectory_metrics")
        task = cert.get("task_metrics")
        sim2real = cert.get("sim2real_metrics")
        if isinstance(trajectory, dict):
            smoothness = _coerce_quality_component_value(
                trajectory.get("smoothness_score")
            )
            if smoothness is not None:
                components.setdefault("trajectory_smoothness", smoothness)
            collision_count = trajectory.get("collision_count")
            if collision_count is not None and "collision_count" not in components:
                normalized = _normalize_collision_component(collision_count)
                if normalized is not None:
                    components["collision_count"] = normalized
        if isinstance(task, dict):
            completion = _coerce_quality_component_value(
                task.get("goal_achievement_score")
            )
            if completion is not None:
                components.setdefault("task_completion", completion)
            grasp_success = _coerce_quality_component_value(
                task.get("skill_correctness_ratio")
            )
            if grasp_success is not None:
                components.setdefault("grasp_success", grasp_success)
        if isinstance(sim2real, dict):
            plausibility = _coerce_quality_component_value(
                sim2real.get("physics_plausibility_score")
            )
            if plausibility is not None:
                components.setdefault("physics_plausibility", plausibility)

    if "task_success" in payload and "task_completion" not in components:
        completion = _coerce_quality_component_value(payload.get("task_success"))
        if completion is not None:
            components["task_completion"] = completion
    if "grasp_success" in payload and "grasp_success" not in components:
        grasp_success = _coerce_quality_component_value(payload.get("grasp_success"))
        if grasp_success is not None:
            components["grasp_success"] = grasp_success
    if "collision_free" in payload and "collision_count" not in components:
        collision_free = _coerce_quality_component_value(payload.get("collision_free"))
        if collision_free is not None:
            components["collision_count"] = collision_free

    return components


def _evaluate_quality_component_thresholds(
    components: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> List[str]:
    if not thresholds:
        return []
    failures = []
    for key, min_value in thresholds.items():
        if key not in components:
            continue
        if components[key] < min_value:
            failures.append(key)
    return failures


def _guard_quality_thresholds(
    job_metadata: Optional[Dict[str, Any]],
    quality_settings: ResolvedQualitySettings,
    production_mode: bool,
) -> None:
    if not job_metadata:
        return
    quality_metadata = job_metadata.get("quality_config", {})
    generation_params = job_metadata.get("generation_params", {})
    submitted_min_quality = _coerce_float(
        quality_metadata.get("min_quality_score")
    ) or _coerce_float(generation_params.get("min_quality_score"))
    submitted_filter_low_quality = _coerce_bool(
        quality_metadata.get("filter_low_quality")
    )
    submitted_dimension_thresholds = _coerce_threshold_map(
        quality_metadata.get("dimension_thresholds")
    )
    if not submitted_dimension_thresholds:
        submitted_dimension_thresholds = _coerce_threshold_map(
            generation_params.get("dimension_thresholds")
        )

    mismatches = []
    if submitted_min_quality is not None and abs(
        submitted_min_quality - quality_settings.min_quality_score
    ) > 1e-6:
        mismatches.append(
            "min_quality_score "
            f"(submit={submitted_min_quality}, import={quality_settings.min_quality_score})"
        )
    if (
        submitted_filter_low_quality is not None
        and submitted_filter_low_quality != quality_settings.filter_low_quality
    ):
        mismatches.append(
            "filter_low_quality "
            f"(submit={submitted_filter_low_quality}, import={quality_settings.filter_low_quality})"
        )
    if submitted_dimension_thresholds:
        config_thresholds = dict(quality_settings.dimension_thresholds)
        threshold_keys = set(submitted_dimension_thresholds) | set(config_thresholds)
        for key in sorted(threshold_keys):
            submitted_value = submitted_dimension_thresholds.get(key)
            config_value = config_thresholds.get(key)
            if submitted_value is None or config_value is None:
                mismatches.append(
                    "dimension_thresholds mismatch "
                    f"(dimension={key}, submit={submitted_value}, import={config_value})"
                )
            elif abs(submitted_value - config_value) > 1e-6:
                mismatches.append(
                    "dimension_thresholds mismatch "
                    f"(dimension={key}, submit={submitted_value}, import={config_value})"
                )

    if not mismatches:
        return

    message = (
        "[GENIE-SIM-IMPORT] Quality config mismatch detected: "
        + "; ".join(mismatches)
    )
    if production_mode:
        print(f"[GENIE-SIM-IMPORT] ERROR: {message}")
        sys.exit(1)
    print(f"[GENIE-SIM-IMPORT] WARNING: {message}")


def _alert_low_quality(
    *,
    scene_id: str,
    job_id: str,
    robot_type: str,
    average_quality_score: float,
    min_quality_score: float,
    episodes_passed_validation: int,
    episodes_filtered: int,
) -> None:
    if average_quality_score >= min_quality_score:
        return
    send_alert(
        event_type="geniesim_import_low_quality",
        summary="Genie Sim import quality score below threshold",
        details={
            "scene_id": scene_id,
            "job_id": job_id,
            "robot_type": robot_type,
            "average_quality_score": average_quality_score,
            "min_quality_score": min_quality_score,
            "episodes_passed_validation": episodes_passed_validation,
            "episodes_filtered": episodes_filtered,
        },
        severity=os.getenv("ALERT_LOW_QUALITY_SEVERITY", "warning"),
    )


def _alert_firebase_upload_failure(
    *,
    scene_id: str,
    job_id: str,
    robot_type: str,
    error: str,
) -> None:
    send_alert(
        event_type="geniesim_import_firebase_upload_failed",
        summary="Genie Sim import Firebase upload failed",
        details={
            "scene_id": scene_id,
            "job_id": job_id,
            "robot_type": robot_type,
            "error": error,
        },
        severity=os.getenv("ALERT_FIREBASE_UPLOAD_SEVERITY", "error"),
    )


def _read_parquet_dataframe(
    episode_file: Path,
    allow_fallback: bool,
) -> "pd.DataFrame":
    if importlib.util.find_spec("pyarrow.parquet") is not None:
        pq = importlib.import_module("pyarrow.parquet")
        table = pq.read_table(episode_file)
        return table.to_pandas()

    if not allow_fallback:
        raise RuntimeError(
            "Parquet validation requires pyarrow; install pyarrow or allow a fallback reader."
        )

    if parse_bool_env("PARQUET_STREAM_VALIDATE_ONLY", False):
        raise RuntimeError(
            "Parquet validation fallback disabled by PARQUET_STREAM_VALIDATE_ONLY=1."
        )

    max_bytes_raw = os.getenv("PARQUET_PANDAS_FALLBACK_MAX_BYTES", "268435456")
    try:
        max_bytes = int(max_bytes_raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid PARQUET_PANDAS_FALLBACK_MAX_BYTES value: {max_bytes_raw}"
        ) from exc
    file_size = episode_file.stat().st_size
    if file_size > max_bytes:
        raise RuntimeError(
            "Parquet validation fallback refused: "
            f"{file_size} bytes exceeds PARQUET_PANDAS_FALLBACK_MAX_BYTES={max_bytes}."
        )

    if importlib.util.find_spec("pandas") is None:
        raise RuntimeError(
            "Parquet validation fallback requires pandas and fastparquet."
        )
    if importlib.util.find_spec("fastparquet") is None:
        raise RuntimeError(
            "Parquet validation fallback requires fastparquet (pip install fastparquet)."
        )
    pd = importlib.import_module("pandas")
    return pd.read_parquet(episode_file, engine="fastparquet")


def _collect_parquet_column_names(schema: Any) -> List[str]:
    if schema is None:
        return []
    try:
        pa = importlib.import_module("pyarrow")
    except ImportError:
        return list(getattr(schema, "names", []))
    column_names = set(getattr(schema, "names", []))
    for field in schema:
        if pa.types.is_struct(field.type):
            for child in field.type:
                column_names.add(f"{field.name}.{child.name}")
    return sorted(column_names)


def _stream_parquet_validation(
    episode_file: Path,
    require_parquet_validation: bool,
    episode_index: Optional[int] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    batch_count = 0
    row_count = 0
    mode = "streaming"

    allow_pandas_fallback = parse_bool_env(
        os.getenv("ALLOW_PANDAS_PARQUET_FALLBACK"),
        default=False,
    )
    if (
        importlib.util.find_spec("pyarrow.parquet") is None
        and require_parquet_validation
        and not allow_pandas_fallback
    ):
        raise RuntimeError(
            "Parquet validation requires pyarrow; install pyarrow or set "
            "ALLOW_PANDAS_PARQUET_FALLBACK=1 to use the pandas fallback."
        )

    if importlib.util.find_spec("pyarrow.parquet") is not None:
        pq = importlib.import_module("pyarrow.parquet")
        pf = pq.ParquetFile(episode_file)
        schema = pf.schema_arrow
        column_names = _collect_parquet_column_names(schema)
        batch_size_raw = os.getenv("PARQUET_VALIDATE_BATCH_SIZE", "65536")
        try:
            batch_size = int(batch_size_raw)
        except ValueError:
            batch_size = 65536

        required_fields = ["observation", "action"]
        for field in required_fields:
            field_variations = [field, f"observation.{field}", f"{field}s"]
            if not any(
                variation in column_names
                or any(variation in col for col in column_names)
                for variation in field_variations
            ):
                errors.append(f"Missing required field: {field}")

        timestamp_cols = [
            col
            for col in column_names
            if "timestamp" in col.lower() or "time" in col.lower()
        ]
        obs_cols = [col for col in column_names if col.startswith("observation")]
        action_cols = [col for col in column_names if "action" in col.lower()]

        nan_columns: set[str] = set()
        inf_columns: set[str] = set()
        timestamp_errors: set[str] = set()
        last_timestamps: Dict[str, Optional[float]] = {col: None for col in timestamp_cols}
        obs_shapes: Dict[str, set] = {col: set() for col in obs_cols}
        action_max_abs: Dict[str, Optional[float]] = {col: None for col in action_cols}

        obs_shape_error_cols: set[str] = set()
        action_value_error_cols: set[str] = set()

        for batch in pf.iter_batches(batch_size=batch_size):
            batch_count += 1
            row_count += batch.num_rows
            df = batch.to_pandas()

            for col in df.columns:
                series = df[col]
                if np.issubdtype(series.dtype, np.floating):
                    if col not in nan_columns and series.isna().any():
                        nan_columns.add(col)
                        errors.append(f"Column '{col}' contains NaN values")
                    if col not in inf_columns and np.isinf(series.to_numpy()).any():
                        inf_columns.add(col)
                        errors.append(f"Column '{col}' contains Inf values")

            for ts_col in timestamp_cols:
                if ts_col not in df.columns or ts_col in timestamp_errors:
                    continue
                series = df[ts_col]
                if not np.issubdtype(series.dtype, np.number):
                    continue
                numeric_series = series.dropna()
                if numeric_series.empty:
                    continue
                last_value = last_timestamps.get(ts_col)
                if last_value is not None and numeric_series.iloc[0] < last_value:
                    timestamp_errors.add(ts_col)
                    errors.append(f"Timestamps in '{ts_col}' are not monotonic")
                    continue
                if not numeric_series.is_monotonic_increasing:
                    timestamp_errors.add(ts_col)
                    errors.append(f"Timestamps in '{ts_col}' are not monotonic")
                    continue
                last_timestamps[ts_col] = float(numeric_series.iloc[-1])

            for obs_col in obs_cols:
                if obs_col not in df.columns:
                    continue
                series = df[obs_col]
                for value in series:
                    if hasattr(value, "__len__"):
                        try:
                            obs_shapes[obs_col].add(np.array(value).shape)
                        except (TypeError, ValueError, np.AxisError) as exc:
                            if obs_col not in obs_shape_error_cols:
                                obs_shape_error_cols.add(obs_col)
                                logger.warning(
                                    "Failed to parse observation shape "
                                    "(episode_index=%s, column=%s, record_count=%s): %s",
                                    episode_index,
                                    obs_col,
                                    row_count,
                                    exc,
                                )
                            continue

            for action_col in action_cols:
                if action_col not in df.columns:
                    continue
                series = df[action_col]
                current_max = action_max_abs.get(action_col)
                if np.issubdtype(series.dtype, np.number):
                    values = series.to_numpy()
                    if values.size:
                        batch_max = float(np.nanmax(np.abs(values)))
                        if current_max is None or batch_max > current_max:
                            action_max_abs[action_col] = batch_max
                    continue
                for value in series:
                    if value is None:
                        continue
                    try:
                        arr = np.array(value)
                        if arr.size == 0:
                            continue
                        batch_max = float(np.nanmax(np.abs(arr)))
                    except (TypeError, ValueError, np.AxisError) as exc:
                        try:
                            batch_max = abs(float(value))
                        except (TypeError, ValueError) as inner_exc:
                            if action_col not in action_value_error_cols:
                                action_value_error_cols.add(action_col)
                                logger.warning(
                                    "Failed to parse action values "
                                    "(episode_index=%s, column=%s, record_count=%s): %s",
                                    episode_index,
                                    action_col,
                                    row_count,
                                    inner_exc,
                                )
                            continue
                        if action_col not in action_value_error_cols:
                            action_value_error_cols.add(action_col)
                            logger.warning(
                                "Failed to parse action array values "
                                "(episode_index=%s, column=%s, record_count=%s): %s",
                                episode_index,
                                action_col,
                                row_count,
                                exc,
                            )
                    if current_max is None or batch_max > current_max:
                        current_max = batch_max
                        action_max_abs[action_col] = current_max

        for obs_col, shapes in obs_shapes.items():
            if len(shapes) > 1:
                warnings.append(f"Inconsistent shapes in '{obs_col}': {sorted(shapes)}")

        for action_col, max_abs in action_max_abs.items():
            if max_abs is not None and max_abs > 10.0:
                warnings.append(
                    "Action values in "
                    f"'{action_col}' exceed reasonable bounds (max: {max_abs:.2f})"
                )

        warnings.append(
            "Parquet validation used streaming mode; "
            f"inspected {row_count} rows across {batch_count} batch(es)."
        )
        return {
            "errors": errors,
            "warnings": warnings,
            "batch_count": batch_count,
            "row_count": row_count,
            "mode": mode,
        }

    mode = "pandas-fallback"
    df = _read_parquet_dataframe(
        episode_file,
        allow_fallback=allow_pandas_fallback or not require_parquet_validation,
    )

    required_fields = ["observation", "action"]
    for field in required_fields:
        field_variations = [field, f"observation.{field}", f"{field}s"]
        if not any(
            variation in df.columns or any(variation in col for col in df.columns)
            for variation in field_variations
        ):
            errors.append(f"Missing required field: {field}")

    for col in df.columns:
        if df[col].dtype in [np.float32, np.float64]:
            if df[col].isna().any():
                errors.append(f"Column '{col}' contains NaN values")
            if np.isinf(df[col]).any():
                errors.append(f"Column '{col}' contains Inf values")

    timestamp_cols = [
        col
        for col in df.columns
        if "timestamp" in col.lower() or "time" in col.lower()
    ]
    for ts_col in timestamp_cols:
        if df[ts_col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            if not df[ts_col].is_monotonic_increasing:
                errors.append(f"Timestamps in '{ts_col}' are not monotonic")

    obs_cols = [col for col in df.columns if col.startswith("observation")]
    for obs_col in obs_cols:
        if df[obs_col].apply(lambda x: hasattr(x, "__len__")).any():
            shapes = df[obs_col].apply(
                lambda x: np.array(x).shape if hasattr(x, "__len__") else None
            )
            unique_shapes = shapes.dropna().unique()
            if len(unique_shapes) > 1:
                warnings.append(f"Inconsistent shapes in '{obs_col}': {unique_shapes}")

    action_cols = [col for col in df.columns if "action" in col.lower()]
    for action_col in action_cols:
        if df[action_col].dtype in [np.float32, np.float64]:
            action_values = df[action_col].values
            if hasattr(action_values[0], "__len__"):
                action_arr = np.array([np.array(a) for a in action_values])
                max_abs = np.abs(action_arr).max()
                if max_abs > 10.0:
                    warnings.append(
                        "Action values in "
                        f"'{action_col}' exceed reasonable bounds (max: {max_abs:.2f})"
                    )
            else:
                max_abs = np.abs(action_values).max()
                if max_abs > 10.0:
                    warnings.append(
                        "Action values in "
                        f"'{action_col}' exceed reasonable bounds (max: {max_abs:.2f})"
                    )

    warnings.append(
        "Parquet validation used pandas fallback mode; "
        f"inspected {len(df)} rows in 1 batch."
    )

    return {
        "errors": errors,
        "warnings": warnings,
        "batch_count": 1,
        "row_count": len(df),
        "mode": mode,
    }


def _load_episode_frames_for_cosmos(episode_file: Path) -> List[Dict[str, Any]]:
    """Load episode frames from a Parquet file for Cosmos Policy export.

    Reads the Parquet file and converts rows to the frame dict format
    expected by CosmosPolicyExporter.

    Returns:
        List of frame dicts, or empty list if loading fails.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(episode_file)
        df_dict = table.to_pydict()
    except Exception:
        try:
            with open(episode_file) as f:
                df_dict = json.load(f)
        except Exception:
            return []

    num_rows = len(next(iter(df_dict.values()))) if df_dict else 0
    if num_rows == 0:
        return []

    frames: List[Dict[str, Any]] = []
    for i in range(num_rows):
        frame: Dict[str, Any] = {}

        # Timestamp
        if "timestamp" in df_dict:
            frame["timestamp"] = float(df_dict["timestamp"][i])
        else:
            frame["timestamp"] = i / 30.0

        # Joint positions
        jp_keys = sorted(k for k in df_dict if k.startswith("observation.state.") or k.startswith("joint_position"))
        if jp_keys:
            frame["joint_positions"] = [float(df_dict[k][i]) for k in jp_keys[:7]]
        elif "joint_positions" in df_dict:
            val = df_dict["joint_positions"][i]
            frame["joint_positions"] = list(val) if hasattr(val, "__iter__") else [float(val)]

        # Gripper
        if "gripper_position" in df_dict:
            frame["gripper_position"] = float(df_dict["gripper_position"][i])
        elif "observation.gripper_position" in df_dict:
            frame["gripper_position"] = float(df_dict["observation.gripper_position"][i])
        else:
            frame["gripper_position"] = 0.0

        # Actions
        action_keys = sorted(k for k in df_dict if k.startswith("action.") or k == "action")
        if action_keys and action_keys[0] == "action":
            val = df_dict["action"][i]
            frame["action"] = list(val) if hasattr(val, "__iter__") else [float(val)]
        elif action_keys:
            frame["action"] = [float(df_dict[k][i]) for k in action_keys[:8]]

        # End-effector
        ee_pos_keys = sorted(k for k in df_dict if k.startswith("ee_position") or k.startswith("observation.ee_position"))
        if ee_pos_keys:
            frame["ee_position"] = [float(df_dict[k][i]) for k in ee_pos_keys[:3]]

        ee_orient_keys = sorted(k for k in df_dict if k.startswith("ee_orientation") or k.startswith("observation.ee_orientation"))
        if ee_orient_keys:
            frame["ee_orientation"] = [float(df_dict[k][i]) for k in ee_orient_keys[:4]]

        # Joint velocities (optional)
        jv_keys = sorted(k for k in df_dict if k.startswith("joint_velocit"))
        if jv_keys:
            frame["joint_velocities"] = [float(df_dict[k][i]) for k in jv_keys[:7]]

        frames.append(frame)

    return frames


def _build_dataset_info(
    job_id: str,
    scene_id: str,
    source: str,
    converted_at: str,
) -> Dict[str, Any]:
    pipeline_commit = get_git_sha(REPO_ROOT) or "unknown"
    return {
        "dataset_type": "lerobot",
        "format_version": "1.0",
        "schema_version": DATASET_INFO_SCHEMA_VERSION,
        "scene_id": scene_id,
        "job_id": job_id,
        "run_id": _resolve_run_id(job_id),
        "pipeline_commit": pipeline_commit,
        "export_schema_version": _resolve_export_schema_version(),
        "episodes": [],
        "total_frames": 0,
        "average_quality_score": 0.0,
        "source": source,
        "converted_at": converted_at,
    }


def _build_cost_summary(
    scene_id: str,
    log: logging.LoggerAdapter,
) -> Optional[Dict[str, Any]]:
    if not scene_id:
        return None
    try:
        tracker = get_cost_tracker()
        breakdown = tracker.get_scene_cost(scene_id)
    except Exception as exc:
        log.warning("Cost tracking unavailable for scene %s: %s", scene_id, exc)
        return None

    gcs_total = breakdown.gcs_storage + breakdown.gcs_operations
    return {
        "total_usd": breakdown.total,
        "categories": {
            "geniesim": breakdown.geniesim,
            "cloud_run": breakdown.cloud_run,
            "cloud_build": breakdown.cloud_build,
            "gcs": gcs_total,
            "gcs_storage": breakdown.gcs_storage,
            "gcs_operations": breakdown.gcs_operations,
            "gemini": breakdown.gemini,
            "other_apis": breakdown.other_apis,
        },
        "by_job": dict(breakdown.by_job),
        "pricing_source": tracker.pricing_source,
    }


def _update_dataset_info_cost_summary(
    cost_summary: Optional[Dict[str, Any]],
    dataset_info_payload: Optional[Dict[str, Any]],
    dataset_info_path: Optional[Path],
) -> None:
    if not cost_summary or not isinstance(dataset_info_payload, dict):
        return
    dataset_info_payload["cost_summary"] = cost_summary
    if dataset_info_path and dataset_info_path.exists():
        write_json_atomic(dataset_info_path, dataset_info_payload, indent=2)


def _attach_cost_summary(
    import_manifest: Dict[str, Any],
    cost_summary: Optional[Dict[str, Any]],
) -> None:
    if cost_summary is None:
        return
    import_manifest["cost_summary"] = cost_summary


def _write_lerobot_readme(output_dir: Path, lerobot_dir: Path) -> Path:
    readme_path = output_dir / "README.md"
    lerobot_rel = _relative_to_bundle(output_dir, lerobot_dir)
    content = f"""# Genie Sim LeRobot Bundle

This bundle includes a LeRobot-compatible dataset generated from Genie Sim episodes.

## Dataset layout
- `{lerobot_rel}/`: LeRobot dataset directory
- `{lerobot_rel}/dataset_info.json`: dataset summary and episode metadata
- `{lerobot_rel}/episodes.jsonl`: episode index
- `{lerobot_rel}/episode_*.parquet`: per-episode data

## Checksums
To verify bundle integrity, use the checksum manifest at `checksums.json`.
It contains SHA-256 checksums keyed by relative file path within the bundle root.

## Load with LeRobot
Install LeRobot and load the dataset directory:

```bash
pip install lerobot
```

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("{lerobot_rel}")
sample = dataset[0]
print(sample.keys())
```

## Train with LeRobot
Use your preferred LeRobot training entrypoint and point it at `{lerobot_rel}`.
For example, supply the dataset path in your training config or CLI to start training.
"""
    readme_path.write_text(content)
    return readme_path


def _load_contract_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = REPO_ROOT / "fixtures" / "contracts" / schema_name
    return json.loads(schema_path.read_text())


def _validate_minimal_schema(payload: Any, schema: Dict[str, Any], path: str) -> None:
    any_of = schema.get("anyOf") or schema.get("oneOf")
    if any_of:
        errors = []
        for idx, option in enumerate(any_of):
            try:
                _validate_minimal_schema(payload, option, path)
            except ValueError as exc:
                errors.append(f"Option {idx}: {exc}")
            else:
                return
        raise ValueError(f"{path}: payload did not match anyOf/oneOf: {errors}")
    for option in schema.get("allOf", []):
        _validate_minimal_schema(payload, option, path)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        last_error: Optional[Exception] = None
        for candidate in schema_type:
            candidate_schema = dict(schema)
            candidate_schema["type"] = candidate
            try:
                _validate_minimal_schema(payload, candidate_schema, path)
            except ValueError as exc:
                last_error = exc
                continue
            else:
                return
        if last_error is not None:
            raise last_error
        return
    if schema_type == "object":
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected object")
        for key in schema.get("required", []):
            if key not in payload:
                raise ValueError(f"{path}: missing required field '{key}'")
        for key, prop_schema in schema.get("properties", {}).items():
            if key in payload:
                _validate_minimal_schema(payload[key], prop_schema, f"{path}.{key}")
    elif schema_type == "array":
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(payload) < min_items:
            raise ValueError(f"{path}: expected at least {min_items} items")
        if max_items is not None and len(payload) > max_items:
            raise ValueError(f"{path}: expected at most {max_items} items")
        items_schema = schema.get("items")
        if items_schema:
            for idx, item in enumerate(payload):
                _validate_minimal_schema(item, items_schema, f"{path}[{idx}]")
    elif schema_type == "string":
        if not isinstance(payload, str):
            raise ValueError(f"{path}: expected string")
        enum = schema.get("enum")
        if enum and payload not in enum:
            raise ValueError(f"{path}: value '{payload}' not in enum {enum}")
    elif schema_type == "integer":
        if not isinstance(payload, int):
            raise ValueError(f"{path}: expected integer")
    elif schema_type == "number":
        if not isinstance(payload, (int, float)):
            raise ValueError(f"{path}: expected number")
    elif schema_type == "boolean":
        if not isinstance(payload, bool):
            raise ValueError(f"{path}: expected boolean")
    elif schema_type == "null":
        if payload is not None:
            raise ValueError(f"{path}: expected null")


def _resolve_lerobot_info_path(output_dir: Path, lerobot_dir: Path) -> Optional[Path]:
    candidates = [
        lerobot_dir / "meta" / "info.json",
        output_dir / "episodes" / "lerobot" / "meta" / "info.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_info_path(output_dir: Path, path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return output_dir / candidate


def _detect_lerobot_export_format(
    info_payload: Optional[Dict[str, Any]],
) -> Optional[LeRobotExportFormat]:
    if not isinstance(info_payload, dict):
        return None
    raw_value = info_payload.get("export_format") or info_payload.get("format")
    if raw_value:
        try:
            return parse_lerobot_export_format(str(raw_value))
        except ValueError:
            return None
    version = info_payload.get("version")
    if version:
        normalized = str(version).strip()
        if normalized == "3.0":
            return LeRobotExportFormat.LEROBOT_V3
        if normalized.startswith("0.4"):
            return LeRobotExportFormat.LEROBOT_V0_4
        if normalized == "0.3.3":
            return LeRobotExportFormat.LEROBOT_V0_3_3
    return LeRobotExportFormat.LEROBOT_V2


def _resolve_lerobot_episode_index_path(
    lerobot_root: Path,
    export_format: Optional[LeRobotExportFormat],
) -> Path:
    primary = lerobot_root / "episodes.jsonl"
    meta_candidate = lerobot_root / "meta" / "episodes.jsonl"
    if primary.exists():
        return primary
    if meta_candidate.exists():
        return meta_candidate
    if export_format == LeRobotExportFormat.LEROBOT_V0_4:
        return meta_candidate
    return primary


def _resolve_lerobot_v3_paths(
    output_dir: Path,
    lerobot_root: Path,
    info_payload: Optional[Dict[str, Any]],
) -> Dict[str, Path]:
    data_path = None
    if isinstance(info_payload, dict):
        data_path = info_payload.get("data_path")
    data_dir = _resolve_info_path(output_dir, data_path) if data_path else None
    if data_dir is None:
        data_dir = lerobot_root / "data"

    # LeRobot v3.0 official spec: data/chunk-{idx}/file-{idx}.parquet
    # Also support legacy episodes.parquet for backward compatibility
    data_parquet_path = data_dir / "chunk-000" / "file-0000.parquet"
    if not data_parquet_path.exists():
        legacy_path = data_dir / "chunk-000" / "episodes.parquet"
        if legacy_path.exists():
            data_parquet_path = legacy_path

    # Episode metadata: meta/episodes/chunk-000/file-0000.parquet (v3 official spec)
    # Fallback to legacy meta/episode_index.json for backward compatibility
    episodes_meta_path = lerobot_root / "meta" / "episodes" / "chunk-000" / "file-0000.parquet"
    if not episodes_meta_path.exists():
        legacy_index = lerobot_root / "meta" / "episode_index.json"
        if legacy_index.exists():
            episodes_meta_path = legacy_index
        else:
            episode_index_value = None
            if isinstance(info_payload, dict):
                episode_index_value = info_payload.get("episode_index")
            resolved = _resolve_info_path(output_dir, episode_index_value)
            if resolved is not None:
                episodes_meta_path = resolved

    return {
        "episodes_parquet": data_parquet_path,
        "episode_index": episodes_meta_path,
    }


def _validate_lerobot_metadata_files(
    output_dir: Path,
    lerobot_dir: Path,
    require_parquet_validation: bool,
) -> Dict[str, Any]:
    schema_errors: List[str] = []
    info_path = _resolve_lerobot_info_path(output_dir, lerobot_dir)
    info_payload = None
    if info_path is not None:
        try:
            info_payload = _load_json_file(info_path)
        except Exception as exc:
            schema_errors.append(
                f"metadata {_relative_to_bundle(output_dir, info_path)}: {exc}"
            )
    export_format = _detect_lerobot_export_format(info_payload)
    lerobot_root = info_path.parent.parent if info_path is not None else lerobot_dir

    episodes_index_path = _resolve_lerobot_episode_index_path(lerobot_root, export_format)
    episodes_parquet_path = None
    episode_index_path = None
    if export_format == LeRobotExportFormat.LEROBOT_V3:
        v3_paths = _resolve_lerobot_v3_paths(output_dir, lerobot_root, info_payload)
        episodes_parquet_path = v3_paths["episodes_parquet"]
        episode_index_path = v3_paths["episode_index"]
        if not episodes_parquet_path.exists():
            schema_errors.append(
                f"metadata {_relative_to_bundle(output_dir, episodes_parquet_path)}: missing data parquet file"
            )
        if episode_index_path.exists():
            # Handle both Parquet (v3 official) and JSON (legacy) episode metadata
            if episode_index_path.suffix == ".parquet":
                try:
                    import pyarrow.parquet as _pq
                    _meta_table = _pq.read_table(episode_index_path)
                    required_cols = {"episode_index", "num_frames"}
                    missing_cols = required_cols - set(_meta_table.column_names)
                    if missing_cols:
                        schema_errors.append(
                            f"metadata {_relative_to_bundle(output_dir, episode_index_path)}: "
                            f"missing columns {missing_cols}"
                        )
                except ImportError:
                    if require_parquet_validation:
                        schema_errors.append(
                            "metadata "
                            f"{_relative_to_bundle(output_dir, episode_index_path)}: "
                            "pyarrow is required to validate parquet metadata"
                        )
                except Exception as exc:
                    schema_errors.append(
                        f"metadata {_relative_to_bundle(output_dir, episode_index_path)}: {exc}"
                    )
            else:
                try:
                    episode_index_payload = _load_json_file(episode_index_path)
                    schema_errors.extend(
                        _validate_schema_payload(
                            episode_index_payload,
                            "geniesim_local_episode_index_v3.schema.json",
                            f"metadata {_relative_to_bundle(output_dir, episode_index_path)}",
                        )
                    )
                except Exception as exc:
                    schema_errors.append(
                        f"metadata {_relative_to_bundle(output_dir, episode_index_path)}: {exc}"
                    )
        else:
            schema_errors.append(
                f"metadata {_relative_to_bundle(output_dir, episode_index_path)}: missing episode metadata"
            )
    else:
        if episodes_index_path.exists():
            try:
                with open(episodes_index_path, "r") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        entry = json.loads(stripped)
                        schema_errors.extend(
                            _validate_schema_payload(
                                entry,
                                "geniesim_local_episodes_index.schema.json",
                                f"metadata {_relative_to_bundle(output_dir, episodes_index_path)}:{line_number}",
                            )
                        )
            except Exception as exc:
                schema_errors.append(
                    f"metadata {_relative_to_bundle(output_dir, episodes_index_path)}: {exc}"
                )
        else:
            schema_errors.append(
                f"metadata {_relative_to_bundle(output_dir, episodes_index_path)}: missing episodes.jsonl"
            )

    return {
        "export_format": export_format.value if export_format else None,
        "schema_errors": schema_errors,
        "info_path": info_path,
        "episodes_index_path": episodes_index_path,
        "episodes_parquet_path": episodes_parquet_path,
        "episode_index_path": episode_index_path,
    }


def _validate_json_schema(payload: Any, schema: Dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        _validate_minimal_schema(payload, schema, path="$")
    else:
        jsonschema.validate(instance=payload, schema=schema)


def _validate_schema_payload(
    payload: Any,
    schema_name: str,
    payload_label: str,
) -> List[str]:
    schema = _load_contract_schema(schema_name)
    try:
        _validate_json_schema(payload, schema)
    except Exception as exc:
        return [f"{payload_label}: {exc}"]
    return []


def _load_dataset_info_with_migration(
    path: Path,
    log: logging.Logger,
    *,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    payload = _load_json_file(path)
    try:
        migration = migrate_dataset_info_payload(payload)
    except SchemaMigrationError as exc:
        message = f"Unsupported dataset_info schema version in {path}: {exc}"
        if required:
            raise ValueError(message) from exc
        log.warning(message)
        return None
    if migration.applied_steps:
        log.info(
            "Applied dataset_info migrations for %s: %s",
            path,
            ", ".join(migration.applied_steps),
        )
    return migration.payload


def _resolve_dataset_document_id(job_id: str, robot_type: Optional[str]) -> str:
    if robot_type:
        return f"{job_id}-{robot_type}"
    return job_id


def _publish_dataset_catalog_document(
    *,
    scene_id: str,
    job_id: str,
    robot_type: Optional[str],
    result: "ImportResult",
    firebase_summary: Optional[Dict[str, Any]],
    gcs_output_path: Optional[str],
    log: logging.LoggerAdapter,
) -> None:
    if not result.import_manifest_path:
        log.warning("Skipping dataset catalog publish: missing import manifest path.")
        return

    try:
        import_manifest = _load_import_manifest_with_migration(
            result.import_manifest_path,
            log,
            required=False,
        )
    except ValueError as exc:
        log.warning("Skipping dataset catalog publish: %s", exc)
        return
    if import_manifest is None:
        log.warning(
            "Skipping dataset catalog publish: import_manifest schema unsupported."
        )
        return

    if result.import_manifest_path:
        import_manifest["import_manifest_path"] = _resolve_gcs_path(result.import_manifest_path)

    dataset_info_payload: Optional[Dict[str, Any]] = None
    if result.output_dir:
        dataset_info_path = result.output_dir / "lerobot" / "dataset_info.json"
        if dataset_info_path.exists():
            try:
                dataset_info_payload = _load_dataset_info_with_migration(
                    dataset_info_path,
                    log,
                    required=False,
                )
            except ValueError as exc:
                log.warning("Failed to load dataset_info.json for catalog: %s", exc)

    document_id = _resolve_dataset_document_id(job_id, robot_type)
    dataset_document = build_dataset_document(
        scene_id=scene_id,
        job_id=job_id,
        import_manifest=import_manifest,
        dataset_info=dataset_info_payload,
        firebase_summary=firebase_summary,
        gcs_output_path=gcs_output_path,
        robot_types=[robot_type] if robot_type else [],
        document_id=document_id,
    )
    try:
        DatasetCatalogClient().upsert_dataset_document(dataset_document)
        log.info("Published dataset catalog entry %s.", document_id)
    except Exception as exc:  # pragma: no cover - network/firestore errors
        log.warning("Failed to publish dataset catalog entry %s: %s", document_id, exc)


def _create_bundle_package(
    output_dir: Path,
    package_name: str,
    files: List[Path],
    directories: List[Path],
) -> Path:
    package_path = output_dir / package_name
    with tarfile.open(package_path, "w:gz") as archive:
        for path in files:
            if path.exists():
                archive.add(path, arcname=_relative_to_bundle(output_dir, path))
        for directory in directories:
            if directory.exists():
                archive.add(directory, arcname=_relative_to_bundle(output_dir, directory))
    return package_path


def _resolve_import_status(result: "ImportResult") -> str:
    if result.success:
        return "success"
    if result.episodes_passed_validation or result.episodes_filtered:
        return "partial"
    return "failed"


def _resolve_artifacts_by_robot(
    job_metadata: Optional[Dict[str, Any]],
    artifacts_by_robot_env: Optional[str],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if job_metadata:
        artifacts_by_robot = job_metadata.get("artifacts_by_robot")
        if isinstance(artifacts_by_robot, dict):
            return artifacts_by_robot
    if artifacts_by_robot_env:
        try:
            payload = json.loads(artifacts_by_robot_env)
        except json.JSONDecodeError as exc:
            raise ValueError("ARTIFACTS_BY_ROBOT is not valid JSON") from exc
        if isinstance(payload, dict):
            return payload
    return None


def _build_robot_job_metadata(
    job_metadata: Optional[Dict[str, Any]],
    robot_type: str,
    artifacts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if job_metadata is None:
        return None
    payload = copy.deepcopy(job_metadata)
    payload["robot_type"] = robot_type
    payload["artifacts"] = artifacts
    return payload


def _resolve_gcs_path(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    path_str = str(path)
    if path_str.startswith("/mnt/gcs/"):
        return "gs://" + path_str[len("/mnt/gcs/"):]
    return path_str


def _resolve_gcs_upload_target(gcs_output_path: str) -> Dict[str, str]:
    parsed = _parse_gcs_uri(gcs_output_path)
    if not parsed or not parsed.get("bucket"):
        raise ValueError(f"Invalid GCS output path: {gcs_output_path}")
    return parsed


def _build_gcs_object_path(prefix: str, rel_path: str) -> str:
    normalized_prefix = prefix.strip("/")
    normalized_rel = rel_path.lstrip("/")
    if not normalized_prefix:
        return normalized_rel
    return f"{normalized_prefix}/{normalized_rel}"


def _upload_output_dir(
    output_dir: Path,
    gcs_output_path: str,
) -> Dict[str, Any]:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        return {
            "status": "failed",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "total_files": 0,
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
            "failures": [
                {
                    "path": None,
                    "error": f"google-cloud-storage unavailable: {exc}",
                }
            ],
        }

    parsed = _resolve_gcs_upload_target(gcs_output_path)
    bucket_name = parsed["bucket"]
    prefix = parsed.get("object", "")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    log = logging.getLogger(__name__)

    started_at = datetime.utcnow().isoformat() + "Z"
    failures: List[Dict[str, Any]] = []
    uploaded = 0
    skipped = 0
    failed = 0
    files = [path for path in output_dir.rglob("*") if path.is_file()]
    total_files = len(files)

    for path in sorted(files):
        rel_path = path.relative_to(output_dir).as_posix()
        object_name = _build_gcs_object_path(prefix, rel_path)
        gcs_uri = f"gs://{bucket_name}/{object_name}"
        blob = bucket.blob(object_name)
        expected_size = path.stat().st_size
        expected_md5 = calculate_file_md5_base64(path)

        if blob.exists():
            verified, reason = verify_blob_upload(
                blob,
                gcs_uri=gcs_uri,
                expected_size=expected_size,
                expected_md5=expected_md5,
                logger=log,
            )
            if verified:
                skipped += 1
                continue

        result = upload_blob_from_filename(
            blob,
            path,
            gcs_uri,
            logger=log,
            verify_upload=True,
        )
        if result.success:
            uploaded += 1
        else:
            failed += 1
            failures.append(
                {
                    "path": rel_path,
                    "gcs_uri": gcs_uri,
                    "error": result.error,
                }
            )

    completed_at = datetime.utcnow().isoformat() + "Z"
    status = "completed" if failed == 0 else "failed"
    return {
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "total_files": total_files,
        "uploaded": uploaded,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
    }


def _collect_local_episode_metadata(
    recordings_dir: Path,
) -> Dict[str, Any]:
    episode_metadata_list: List[GeneratedEpisodeMetadata] = []
    parse_failures: List[Dict[str, str]] = []
    total_files = 0
    for episode_file in sorted(recordings_dir.rglob("*.json")):
        total_files += 1
        try:
            with open(episode_file, "r") as handle:
                payload = json.load(handle)
            frames = payload.get("frames", [])
            frame_count = payload.get("frame_count", len(frames))
            duration_seconds = 0.0
            if frames:
                last_timestamp = frames[-1].get("timestamp")
                if isinstance(last_timestamp, (int, float)):
                    duration_seconds = float(last_timestamp)
                else:
                    duration_seconds = max(0.0, frame_count / 30.0)
            episode_metadata_list.append(
                GeneratedEpisodeMetadata(
                    episode_id=payload.get("episode_id", episode_file.stem),
                    task_name=payload.get("task_name", "unknown"),
                    quality_score=float(payload.get("quality_score", 0.0)),
                    quality_components=_extract_quality_components(payload),
                    frame_count=int(frame_count),
                    duration_seconds=duration_seconds,
                    validation_passed=bool(payload.get("validation_passed", True)),
                    file_size_bytes=episode_file.stat().st_size,
                )
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            parse_failures.append(
                {
                    "file": episode_file.as_posix(),
                    "error": error_message,
                }
            )
            print(f"[IMPORT] ⚠️  Failed to parse local episode {episode_file}: {exc}")
    return {
        "episodes": episode_metadata_list,
        "parse_failures": parse_failures,
        "parse_failure_count": len(parse_failures),
        "total_files": total_files,
    }


class ImportConfig(BaseModel):
    """Configuration for episode import."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Genie Sim job
    job_id: str

    # Output
    output_dir: Path
    gcs_output_path: Optional[str] = None
    enable_gcs_uploads: bool = True

    # Quality filtering
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE
    quality_component_thresholds: Dict[str, float] = Field(default_factory=dict)
    min_episodes_required: int = MIN_EPISODES_REQUIRED
    enable_validation: bool = True
    filter_low_quality: bool = DEFAULT_FILTER_LOW_QUALITY
    require_lerobot: bool = False
    require_lerobot_default: bool = False
    require_lerobot_source: str = "default"
    require_lerobot_raw_value: Optional[str] = None
    lerobot_skip_rate_max: float = 0.0

    # Polling (if waiting for completion)
    poll_interval: int = 30
    wait_for_completion: bool = True

    # Error handling for partial failures
    fail_on_partial_error: bool = False  # If True, fail the job if any episodes failed

    job_metadata_path: Optional[str] = None
    local_episodes_prefix: Optional[str] = None

    @field_validator("output_dir", mode="before")
    @classmethod
    def _validate_output_dir(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                raise ValueError("output_dir must be a non-empty path.")
            return Path(candidate)
        raise ValueError(f"output_dir must be a path-like value (got {value!r}).")

    @field_validator("min_quality_score", mode="before")
    @classmethod
    def _validate_min_quality_score(cls, value: Any) -> float:
        if value is None:
            raise ValueError("min_quality_score is required.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"min_quality_score must be a float (got {value!r})."
            ) from exc
        if parsed < QUALITY_CONFIG.min_allowed or parsed > QUALITY_CONFIG.max_allowed:
            raise ValueError(
                f"min_quality_score must be between {QUALITY_CONFIG.min_allowed} "
                f"and {QUALITY_CONFIG.max_allowed} (got {parsed})."
            )
        return parsed

    @field_validator("lerobot_skip_rate_max", mode="before")
    @classmethod
    def _validate_lerobot_skip_rate_max(cls, value: Any) -> float:
        if value is None:
            raise ValueError("lerobot_skip_rate_max is required.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"lerobot_skip_rate_max must be a float (got {value!r})."
            ) from exc
        if parsed < 0.0:
            raise ValueError("lerobot_skip_rate_max must be >= 0.")
        return parsed

    @field_validator("min_episodes_required", mode="before")
    @classmethod
    def _validate_min_episodes_required(cls, value: Any) -> int:
        if value is None:
            raise ValueError("min_episodes_required is required.")
        if isinstance(value, bool):
            raise ValueError("min_episodes_required must be an integer, not a boolean.")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"min_episodes_required must be an integer (got {value!r})."
            ) from exc
        if parsed < 1:
            raise ValueError("min_episodes_required must be >= 1.")
        return parsed


def _format_validation_error(error: ValidationError) -> str:
    details = []
    for entry in error.errors():
        location = ".".join(str(part) for part in entry.get("loc", []))
        message = entry.get("msg", "Invalid value")
        details.append(f"{location}: {message}" if location else message)
    return "; ".join(details) if details else str(error)


def _create_import_config(payload: Dict[str, Any]) -> ImportConfig:
    try:
        return ImportConfig.model_validate(payload)
    except ValidationError as exc:
        details = _format_validation_error(exc)
        print(
            "[GENIE-SIM-IMPORT] ERROR: Invalid import configuration. "
            f"Fix the environment inputs and retry. Details: {details}"
        )
        sys.exit(1)


@dataclass
class ImportResult:
    """Result of episode import."""

    success: bool
    job_id: str

    # Statistics
    total_episodes_downloaded: int = 0
    episodes_passed_validation: int = 0
    episodes_filtered: int = 0
    episodes_parse_failed: int = 0
    episode_parse_failures: List[Dict[str, str]] = field(default_factory=list)
    episode_content_hashes: Dict[str, str] = field(default_factory=dict)
    episode_content_manifests: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    deduplicated_episode_ids: List[str] = field(default_factory=list)

    # Quality metrics
    average_quality_score: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    quality_min_score: float = 0.0
    quality_max_score: float = 0.0
    quality_component_failed_count: int = 0
    quality_component_failure_counts: Dict[str, int] = field(default_factory=dict)
    quality_component_thresholds: Dict[str, float] = field(default_factory=dict)

    # Output
    output_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None
    import_manifest_path: Optional[Path] = None
    lerobot_conversion_success: bool = True
    checksum_verification_passed: bool = True
    checksum_verification_errors: List[str] = field(default_factory=list)
    upload_status: Optional[str] = None
    upload_failures: List[Dict[str, Any]] = field(default_factory=list)
    upload_started_at: Optional[str] = None
    upload_completed_at: Optional[str] = None

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Episode Validator
# =============================================================================


class ImportedEpisodeValidator:
    """Validates imported episodes from Genie Sim."""

    def __init__(
        self,
        min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
        require_parquet_validation: bool = True,
        dimension_thresholds: Optional[Mapping[str, float]] = None,
    ):
        """
        Initialize validator.

        Args:
            min_quality_score: Minimum quality score to pass
        """
        self.min_quality_score = min_quality_score
        self.require_parquet_validation = require_parquet_validation
        self.dimension_thresholds = dict(dimension_thresholds or {})

    def validate_episode(
        self,
        episode_metadata: GeneratedEpisodeMetadata,
        episode_file: Path,
        episode_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate a single episode.

        Enhanced validation with comprehensive checks for:
        - Required fields
        - Observation/action shapes
        - NaN/Inf values
        - Timestamp monotonicity
        - Physics plausibility

        Args:
            episode_metadata: Episode metadata from Genie Sim
            episode_file: Path to episode file

        Returns:
            Validation result dict with:
                - passed: bool
                - quality_score: float
                - errors: List[str]
                - warnings: List[str]
        """
        errors = []
        warnings = []
        parquet_validation_error: Optional[str] = None
        parquet_errors: List[str] = []

        # Check file exists
        if not episode_file.exists():
            errors.append(f"Episode file not found: {episode_file}")
            return {
                "passed": False,
                "quality_score": 0.0,
                "errors": errors,
                "warnings": warnings,
            }

        # Check file size
        actual_size = episode_file.stat().st_size
        if actual_size == 0:
            errors.append("Episode file is empty")
        elif actual_size < 1024:  # Less than 1KB
            warnings.append(f"Episode file is suspiciously small: {actual_size} bytes")

        # Load and validate episode data structure
        try:
            parquet_results = _stream_parquet_validation(
                episode_file,
                require_parquet_validation=self.require_parquet_validation,
                episode_index=episode_index,
            )
            parquet_errors = parquet_results["errors"]
            errors.extend(parquet_errors)
            warnings.extend(parquet_results["warnings"])
        except RuntimeError as exc:
            error_message = str(exc)
            errors.append(error_message)
            if (
                self.require_parquet_validation
                and "pyarrow" in error_message.lower()
            ):
                parquet_validation_error = error_message
        except Exception as e:
            warnings.append(f"Failed to load episode data for validation: {e}")

        # Check quality score
        quality_score = episode_metadata.quality_score
        if quality_score < self.min_quality_score:
            warnings.append(
                f"Quality score {quality_score:.2f} below threshold {self.min_quality_score:.2f}"
            )

        component_failures = _evaluate_quality_component_thresholds(
            episode_metadata.quality_components,
            self.dimension_thresholds,
        )
        if component_failures:
            warnings.append(
                "Quality component thresholds failed for: "
                + ", ".join(component_failures)
            )

        # Check validation status
        if not episode_metadata.validation_passed:
            warnings.append("Episode failed Genie Sim validation")

        # Check frame count
        if episode_metadata.frame_count < 10:
            warnings.append(f"Episode has very few frames: {episode_metadata.frame_count}")

        # Check duration
        if episode_metadata.duration_seconds < 0.1:
            warnings.append(f"Episode duration suspiciously short: {episode_metadata.duration_seconds}s")

        # Determine if passed
        passed = (
            len(errors) == 0
            and quality_score >= self.min_quality_score
            and not component_failures
            and episode_metadata.validation_passed
        )

        return {
            "passed": passed,
            "quality_score": quality_score,
            "quality_components": episode_metadata.quality_components,
            "quality_component_failures": component_failures,
            "errors": errors,
            "warnings": warnings,
            "parquet_validation_error": parquet_validation_error,
            "parquet_errors": parquet_errors,
        }

    def validate_batch(
        self,
        episodes: List[GeneratedEpisodeMetadata],
        episode_dir: Path,
    ) -> Dict[str, Any]:
        """
        Validate a batch of episodes.

        Args:
            episodes: List of episode metadata
            episode_dir: Directory containing episode files

        Returns:
            Batch validation result with statistics
        """
        results = []
        passed_count = 0
        quality_scores = []
        component_failure_counts = {
            key: 0 for key in (self.dimension_thresholds or {}).keys()
        }
        component_failed_episodes = []
        parquet_validation_errors: set[str] = set()
        component_failed_count = 0

        for episode_index, episode in enumerate(episodes):
            episode_file = episode_dir / f"{episode.episode_id}.parquet"
            result = self.validate_episode(
                episode,
                episode_file,
                episode_index=episode_index,
            )
            results.append({
                "episode_id": episode.episode_id,
                **result,
            })
            parquet_validation_error = result.get("parquet_validation_error")
            if parquet_validation_error:
                parquet_validation_errors.add(str(parquet_validation_error))
            parquet_errors = result.get("parquet_errors") or []
            for error in parquet_errors:
                parquet_validation_errors.add(str(error))

            if result["passed"]:
                passed_count += 1

            quality_scores.append(result["quality_score"])
            component_failures = result.get("quality_component_failures") or []
            if component_failures:
                component_failed_count += 1
                component_failed_episodes.append(
                    {
                        "episode_id": episode.episode_id,
                        "failed_dimensions": component_failures,
                    }
                )
                for dimension in component_failures:
                    component_failure_counts[dimension] = (
                        component_failure_counts.get(dimension, 0) + 1
                    )

        return {
            "total_episodes": len(episodes),
            "passed_count": passed_count,
            "failed_count": len(episodes) - passed_count,
            "average_quality_score": np.mean(quality_scores) if quality_scores else 0.0,
            "min_quality_score": np.min(quality_scores) if quality_scores else 0.0,
            "max_quality_score": np.max(quality_scores) if quality_scores else 0.0,
            "quality_component_thresholds": dict(self.dimension_thresholds),
            "quality_component_failed_count": component_failed_count,
            "quality_component_failure_counts": component_failure_counts,
            "quality_component_failed_episodes": component_failed_episodes,
            "episode_results": results,
            "parquet_validation_errors": sorted(parquet_validation_errors),
        }


# =============================================================================
# LeRobot Conversion
# =============================================================================


def _compute_lerobot_stats(
    output_dir: Path,
    num_episodes: int,
) -> Optional[Dict[str, Any]]:
    """Compute per-feature normalization statistics (min/max/mean/std).

    Reads all converted episode Parquet files and computes running statistics
    for observation.state and action features. Required by LeRobot v3 for
    dataset.meta.stats normalization in training scripts.

    Returns:
        Dict with per-feature stats, or None if computation fails.
    """
    if num_episodes == 0:
        return None

    try:
        import pyarrow.parquet as _pq
    except ImportError:
        logger.warning("pyarrow not available; skipping stats.json generation")
        return None

    all_states = []
    all_actions = []

    for ep_idx in range(num_episodes):
        ep_path = output_dir / f"episode_{ep_idx:06d}.parquet"
        if not ep_path.exists():
            continue
        try:
            table = _pq.read_table(ep_path)
            if "observation.state" in table.column_names:
                col = table["observation.state"]
                for row in col.to_pylist():
                    if isinstance(row, (list, np.ndarray)):
                        all_states.append(np.array(row, dtype=np.float64))
            if "action" in table.column_names:
                col = table["action"]
                for row in col.to_pylist():
                    if isinstance(row, (list, np.ndarray)):
                        all_actions.append(np.array(row, dtype=np.float64))
        except Exception as exc:
            logger.warning("Failed to read episode %d for stats: %s", ep_idx, exc)
            continue

    stats: Dict[str, Any] = {}

    if all_states:
        states_arr = np.array(all_states)
        stats["observation.state"] = {
            "min": states_arr.min(axis=0).tolist(),
            "max": states_arr.max(axis=0).tolist(),
            "mean": states_arr.mean(axis=0).tolist(),
            "std": states_arr.std(axis=0).tolist(),
        }

    if all_actions:
        actions_arr = np.array(all_actions)
        stats["action"] = {
            "min": actions_arr.min(axis=0).tolist(),
            "max": actions_arr.max(axis=0).tolist(),
            "mean": actions_arr.mean(axis=0).tolist(),
            "std": actions_arr.std(axis=0).tolist(),
        }

    return stats if stats else None


def _resolve_video_config() -> Any:
    pipeline_config = load_pipeline_config(use_cache=False)
    return pipeline_config.video


def _resolve_video_codec(codec: str) -> str:
    codec_map = {
        "h264": "libx264",
        "h265": "libx265",
    }
    normalized = codec.strip().lower()
    return codec_map.get(normalized, codec)


def _normalize_rgb_frame(
    frame: Any,
    expected_height: int,
    expected_width: int,
    *,
    episode_id: str,
    frame_index: int,
    camera_id: str,
) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            "RGB frame has invalid shape for "
            f"{episode_id} (camera={camera_id}, frame={frame_index}): "
            f"expected HxWx3 but got {array.shape}"
        )
    height, width = array.shape[:2]
    if height != expected_height or width != expected_width:
        raise ValueError(
            "RGB frame resolution mismatch for "
            f"{episode_id} (camera={camera_id}, frame={frame_index}): "
            f"expected {expected_height}x{expected_width} but got {height}x{width}"
        )
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _collect_video_frames(
    rgb_frames: List[Any],
    *,
    episode_id: str,
    expected_height: int,
    expected_width: int,
) -> Dict[str, List[np.ndarray]]:
    camera_frames: Dict[str, List[np.ndarray]] = {}
    camera_ids: Optional[set[str]] = None
    for frame_index, frame_value in enumerate(rgb_frames):
        if frame_value is None:
            raise ValueError(
                f"Missing RGB frame for {episode_id} at index {frame_index}."
            )
        if isinstance(frame_value, dict):
            frame_camera_ids = set(frame_value.keys())
            if camera_ids is None:
                camera_ids = frame_camera_ids
                for camera_id in camera_ids:
                    camera_frames[camera_id] = []
            elif frame_camera_ids != camera_ids:
                raise ValueError(
                    "Inconsistent camera IDs for "
                    f"{episode_id} at frame {frame_index}: "
                    f"expected {sorted(camera_ids)} but got {sorted(frame_camera_ids)}"
                )
            for camera_id, camera_frame in frame_value.items():
                camera_frames[camera_id].append(
                    _normalize_rgb_frame(
                        camera_frame,
                        expected_height,
                        expected_width,
                        episode_id=episode_id,
                        frame_index=frame_index,
                        camera_id=camera_id,
                    )
                )
        else:
            if camera_ids is None:
                camera_ids = {DEFAULT_VIDEO_CAMERA_ID}
                camera_frames[DEFAULT_VIDEO_CAMERA_ID] = []
            elif camera_ids != {DEFAULT_VIDEO_CAMERA_ID}:
                raise ValueError(
                    "Mixed RGB frame formats for "
                    f"{episode_id}: expected camera map but found single frame."
                )
            camera_frames[DEFAULT_VIDEO_CAMERA_ID].append(
                _normalize_rgb_frame(
                    frame_value,
                    expected_height,
                    expected_width,
                    episode_id=episode_id,
                    frame_index=frame_index,
                    camera_id=DEFAULT_VIDEO_CAMERA_ID,
                )
            )
    return camera_frames


def _get_video_rel_path(episode_index: int, camera_id: str) -> str:
    chunk_idx = episode_index // VIDEO_CHUNK_SIZE
    file_idx = episode_index % VIDEO_CHUNK_SIZE
    return f"videos/{camera_id}/chunk-{chunk_idx:03d}/file-{file_idx:04d}.mp4"


def _write_video_frames(
    frames: List[np.ndarray],
    output_path: Path,
    *,
    fps: int,
    codec: str,
) -> None:
    if not frames:
        return
    if importlib.util.find_spec("imageio") is None:
        raise ImportError("imageio is required for video encoding.")
    import imageio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_handle = tempfile.NamedTemporaryFile(
        delete=False,
        dir=output_path.parent,
        suffix=output_path.suffix,
    )
    tmp_path = Path(tmp_handle.name)
    tmp_handle.close()
    writer = imageio.get_writer(
        str(tmp_path),
        fps=fps,
        codec=codec,
        quality=8,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    tmp_path.replace(output_path)


class VideoValidationError(ValueError):
    def __init__(self, message: str, details: Dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def _validate_video_output(
    frames: List[np.ndarray],
    output_path: Path,
    *,
    expected_height: int,
    expected_width: int,
) -> Dict[str, Any]:
    if not frames:
        return {
            "passed": True,
            "expected_frames": 0,
            "actual_frames": 0,
            "expected_resolution": {
                "height": expected_height,
                "width": expected_width,
            },
            "actual_resolution": None,
        }
    if importlib.util.find_spec("imageio") is None:
        raise ImportError("imageio is required for video validation.")
    import imageio

    reader = imageio.get_reader(str(output_path))
    try:
        metadata = reader.get_meta_data() or {}
        size = metadata.get("size") or metadata.get("source_size")
        actual_width = None
        actual_height = None
        if size and len(size) == 2:
            actual_width, actual_height = size
        else:
            first_frame = reader.get_data(0)
            actual_height, actual_width = first_frame.shape[:2]

        try:
            actual_frames = reader.count_frames()
        except Exception:
            actual_frames = None
        if actual_frames in (None, float("inf")):
            actual_frames = sum(1 for _ in reader)

        expected_frames = len(frames)
        passed = True
        errors: Dict[str, str] = {}

        if actual_frames != expected_frames:
            passed = False
            errors["frame_count"] = (
                f"Expected {expected_frames} frame(s), got {actual_frames}."
            )
        if (actual_height, actual_width) != (expected_height, expected_width):
            passed = False
            errors["resolution"] = (
                "Expected "
                f"{expected_width}x{expected_height}, got {actual_width}x{actual_height}."
            )

        return {
            "passed": passed,
            "expected_frames": expected_frames,
            "actual_frames": actual_frames,
            "expected_resolution": {
                "height": expected_height,
                "width": expected_width,
            },
            "actual_resolution": {
                "height": actual_height,
                "width": actual_width,
            },
            "errors": errors,
        }
    finally:
        reader.close()


def _append_conversion_failure(
    conversion_error: Exception,
    episode_id: str,
    retry_attempts: int,
) -> Dict[str, Any]:
    failure_entry: Dict[str, Any] = {
        "episode_id": episode_id,
        "error": str(conversion_error),
        "retry_attempts": retry_attempts,
        "final_exception": repr(conversion_error),
    }
    if isinstance(conversion_error, VideoValidationError):
        failure_entry["video_validation"] = conversion_error.details
    return failure_entry


def convert_to_lerobot(
    episodes_dir: Path,
    output_dir: Path,
    episode_metadata_list: List[GeneratedEpisodeMetadata],
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
    quality_component_thresholds: Optional[Mapping[str, float]] = None,
    job_id: str = "unknown",
    scene_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Convert Genie Sim episodes to LeRobot format.

    This implements the conversion from Genie Sim's native format to
    LeRobot's Parquet-based format with proper metadata standardization
    and quality metrics calculation.

    Args:
        episodes_dir: Directory containing episode .parquet files
        output_dir: Output directory for LeRobot dataset
        episode_metadata_list: List of episode metadata from Genie Sim
        min_quality_score: Minimum quality score for inclusion

    Returns:
        Dict with conversion statistics
    """
    component_thresholds = dict(quality_component_thresholds or {})
    mock_decision = resolve_geniesim_mock_mode()
    if mock_decision.requested and mock_decision.production_mode:
        print(
            "[GENIE-SIM-IMPORT] Mock mode requested but production mode detected; "
            "mock conversion is ignored."
        )
    if mock_decision.enabled:
        output_dir.mkdir(parents=True, exist_ok=True)
        converted_count = 0
        skipped_count = 0
        total_frames = 0
        conversion_failures: List[Dict[str, str]] = []
        dataset_info = _build_dataset_info(
            job_id=job_id,
            scene_id=scene_id,
            source="genie_sim_mock",
            converted_at="2025-01-01T00:00:00Z",
        )
        quality_scores = []

        for ep_metadata in episode_metadata_list:
            component_failures = _evaluate_quality_component_thresholds(
                ep_metadata.quality_components,
                component_thresholds,
            )
            if ep_metadata.quality_score < min_quality_score or component_failures:
                skipped_count += 1
                continue

            episode_file = episodes_dir / f"{ep_metadata.episode_id}.parquet"
            if not episode_file.exists():
                skipped_count += 1
                conversion_failures.append(
                    {
                        "episode_id": ep_metadata.episode_id,
                        "error": "Episode file missing",
                    }
                )
                continue

            episode_output = output_dir / f"episode_{converted_count:06d}.parquet"
            shutil.copyfile(episode_file, episode_output)

            dataset_info["episodes"].append({
                "episode_id": ep_metadata.episode_id,
                "episode_index": converted_count,
                "num_frames": ep_metadata.frame_count,
                "duration_seconds": ep_metadata.duration_seconds,
                "quality_score": ep_metadata.quality_score,
                "quality_components": ep_metadata.quality_components,
                "validation_passed": ep_metadata.validation_passed,
                "file": str(episode_output.name),
            })
            if ep_metadata.episode_content_hash:
                dataset_info["episodes"][-1]["content_hash"] = ep_metadata.episode_content_hash

            converted_count += 1
            total_frames += ep_metadata.frame_count
            quality_scores.append(ep_metadata.quality_score)

        dataset_info["total_episodes"] = converted_count
        dataset_info["total_frames"] = total_frames
        dataset_info["skipped_episodes"] = skipped_count
        total_source_episodes = len(episode_metadata_list)
        skip_rate_percent = (
            (skipped_count / total_source_episodes) * 100.0
            if total_source_episodes
            else 0.0
        )
        dataset_info["skip_rate_percent"] = skip_rate_percent
        dataset_info["conversion_failures"] = conversion_failures
        if quality_scores:
            dataset_info["average_quality_score"] = float(np.mean(quality_scores))
            dataset_info["min_quality_score"] = float(np.min(quality_scores))
            dataset_info["max_quality_score"] = float(np.max(quality_scores))

        metadata_file = output_dir / "dataset_info.json"
        write_json_atomic(metadata_file, dataset_info, indent=2)

        episodes_file = output_dir / "episodes.jsonl"
        episode_lines = [json.dumps(ep_info) for ep_info in dataset_info["episodes"]]
        episodes_payload = "\n".join(episode_lines)
        if episodes_payload:
            episodes_payload += "\n"
        write_text_atomic(episodes_file, episodes_payload)

        return {
            "success": True,
            "converted_count": converted_count,
            "skipped_count": skipped_count,
            "skip_rate_percent": skip_rate_percent,
            "total_frames": total_frames,
            "conversion_failures": conversion_failures,
            "output_dir": output_dir,
            "metadata_file": metadata_file,
        }

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("PyArrow is required for LeRobot conversion: pip install pyarrow")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track conversion statistics
    converted_count = 0
    skipped_count = 0
    total_frames = 0
    conversion_failures: List[Dict[str, Any]] = []

    # LeRobot dataset metadata
    dataset_info = _build_dataset_info(
        job_id=job_id,
        scene_id=scene_id,
        source="genie_sim",
        converted_at=datetime.utcnow().isoformat() + "Z",
    )

    quality_scores = []
    v3_episode_records: List[Dict[str, Any]] = []
    video_config = _resolve_video_config()
    video_codec = _resolve_video_codec(video_config.codec)

    retryable_exceptions = {
        OSError,
        IOError,
        pa.ArrowIOError,
    }
    if hasattr(pa, "lib") and hasattr(pa.lib, "ArrowIOError"):
        retryable_exceptions.add(pa.lib.ArrowIOError)

    retry_config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0,
        backoff_factor=2.0,
    )
    retry_config.retryable_exceptions.update(retryable_exceptions)

    for ep_metadata in episode_metadata_list:
        # Skip low-quality episodes
        component_failures = _evaluate_quality_component_thresholds(
            ep_metadata.quality_components,
            component_thresholds,
        )
        if ep_metadata.quality_score < min_quality_score or component_failures:
            skipped_count += 1
            continue

        episode_file = episodes_dir / f"{ep_metadata.episode_id}.parquet"
        if not episode_file.exists():
            skipped_count += 1
            conversion_failures.append(
                {
                    "episode_id": ep_metadata.episode_id,
                    "error": "Episode file missing",
                }
            )
            continue

        retry_ctx = RetryContext(config=retry_config)
        conversion_error: Optional[Exception] = None
        df = None

        while retry_ctx.should_continue():
            try:
                # Read Genie Sim episode
                table = pq.read_table(episode_file)
                df = table.to_pandas()
                rgb_column = None
                if "rgb_image" in df.columns:
                    rgb_column = "rgb_image"
                elif "image" in df.columns:
                    rgb_column = "image"

                video_paths: Dict[str, str] = {}
                if rgb_column is not None:
                    rgb_frames = df[rgb_column].tolist()
                    camera_frames = _collect_video_frames(
                        rgb_frames,
                        episode_id=ep_metadata.episode_id,
                        expected_height=video_config.height,
                        expected_width=video_config.width,
                    )
                    for camera_id, frames in camera_frames.items():
                        rel_path = _get_video_rel_path(converted_count, camera_id)
                        video_path = output_dir / rel_path
                        _write_video_frames(
                            frames,
                            video_path,
                            fps=video_config.fps,
                            codec=video_codec,
                        )
                        video_validation = _validate_video_output(
                            frames,
                            video_path,
                            expected_height=video_config.height,
                            expected_width=video_config.width,
                        )
                        if not video_validation["passed"]:
                            raise VideoValidationError(
                                "Video validation failed.",
                                video_validation,
                            )
                        video_paths[camera_id] = rel_path

                # Convert to LeRobot schema
                # LeRobot expects: observations, actions, rewards, episode metadata
                lerobot_data = {
                    # Observations (assume RGB images + robot state)
                    "observation.image": df.get("rgb_image", df.get("image", [])),
                    "observation.state": df.get("robot_state", []),

                    # Actions (joint positions/velocities)
                    "action": df.get("action", df.get("joint_positions", [])),

                    # Episode metadata
                    "episode_index": [converted_count] * len(df),
                    "frame_index": list(range(len(df))),
                    "timestamp": df.get("timestamp", list(range(len(df)))),

                    # Quality metrics
                    "quality_score": [ep_metadata.quality_score] * len(df),
                }
                for component_name, component_value in ep_metadata.quality_components.items():
                    lerobot_data[f"quality_components.{component_name}"] = [
                        component_value
                    ] * len(df)

                # Add optional fields if available
                if "depth_image" in df.columns:
                    lerobot_data["observation.depth"] = df["depth_image"]
                if "reward" in df.columns:
                    lerobot_data["reward"] = df["reward"]
                else:
                    # Default reward: 0 for all frames except last (1 if successful)
                    rewards = [0.0] * len(df)
                    if ep_metadata.validation_passed:
                        rewards[-1] = 1.0
                    lerobot_data["reward"] = rewards

                # Convert to PyArrow table
                lerobot_table = pa.Table.from_pydict(lerobot_data)

                # Write episode to LeRobot format
                episode_output = output_dir / f"episode_{converted_count:06d}.parquet"
                pq.write_table(lerobot_table, episode_output)

                # Validate written Parquet file integrity
                readback = pq.ParquetFile(episode_output)
                readback_rows = readback.metadata.num_rows
                expected_rows = len(df)
                if readback_rows != expected_rows:
                    raise IOError(
                        f"Parquet integrity check failed for episode "
                        f"{ep_metadata.episode_id}: expected {expected_rows} rows "
                        f"but file contains {readback_rows}"
                    )

                # Update dataset info
                dataset_info["episodes"].append({
                    "episode_id": ep_metadata.episode_id,
                    "episode_index": converted_count,
                    "num_frames": len(df),
                    "duration_seconds": ep_metadata.duration_seconds,
                    "quality_score": ep_metadata.quality_score,
                    "quality_components": ep_metadata.quality_components,
                    "validation_passed": ep_metadata.validation_passed,
                    "file": str(episode_output.name),
                })
                if video_paths:
                    dataset_info["episodes"][-1]["video_paths"] = dict(video_paths)
                if ep_metadata.episode_content_hash:
                    dataset_info["episodes"][-1]["content_hash"] = ep_metadata.episode_content_hash
                v3_episode_records.append(
                    {
                        "episode_index": converted_count,
                        "num_frames": len(df),
                        "video_paths": json.dumps(video_paths) if video_paths else "",
                    }
                )

                converted_count += 1
                total_frames += len(df)
                quality_scores.append(ep_metadata.quality_score)
                conversion_error = None
                break
            except Exception as exc:
                retry_exception = exc
                if not isinstance(exc, tuple(retryable_exceptions)):
                    retry_exception = NonRetryableError(str(exc))
                conversion_error = exc
                if not retry_ctx.record_failure(retry_exception):
                    break

        if conversion_error is not None:
            print(
                "[LEROBOT] Warning: Failed to convert episode "
                f"{ep_metadata.episode_id} after {retry_ctx.attempt} attempt(s): {conversion_error}"
            )
            conversion_failures.append(
                _append_conversion_failure(
                    conversion_error,
                    ep_metadata.episode_id,
                    retry_ctx.attempt,
                )
            )
            skipped_count += 1
            continue

    # Update dataset statistics
    dataset_info["total_episodes"] = converted_count
    dataset_info["total_frames"] = total_frames
    dataset_info["skipped_episodes"] = skipped_count
    total_source_episodes = len(episode_metadata_list)
    skip_rate_percent = (
        (skipped_count / total_source_episodes) * 100.0
        if total_source_episodes
        else 0.0
    )
    dataset_info["skip_rate_percent"] = skip_rate_percent
    dataset_info["conversion_failures"] = conversion_failures
    if quality_scores:
        dataset_info["average_quality_score"] = float(np.mean(quality_scores))
        dataset_info["min_quality_score"] = float(np.min(quality_scores))
        dataset_info["max_quality_score"] = float(np.max(quality_scores))

    # Compute and write per-feature normalization stats (required by LeRobot v3)
    stats = _compute_lerobot_stats(output_dir, converted_count)
    if stats:
        stats_file = output_dir / "stats.json"
        write_json_atomic(stats_file, stats, indent=2)

    if v3_episode_records:
        episodes_meta_root = output_dir / "meta" / "episodes"
        episodes_meta_root.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([
            ("episode_index", pa.int64()),
            ("num_frames", pa.int64()),
            ("video_paths", pa.string()),
        ])
        for chunk_start in range(0, len(v3_episode_records), VIDEO_CHUNK_SIZE):
            chunk_idx = chunk_start // VIDEO_CHUNK_SIZE
            chunk_dir = episodes_meta_root / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            episodes_meta_path = chunk_dir / f"file-{chunk_idx:04d}.parquet"
            chunk_records = v3_episode_records[
                chunk_start:chunk_start + VIDEO_CHUNK_SIZE
            ]
            table = pa.table(
                {col: [rec[col] for rec in chunk_records] for col in schema.names},
                schema=schema,
            )
            pq.write_table(table, episodes_meta_path)

    # Write dataset metadata
    metadata_file = output_dir / "dataset_info.json"
    write_json_atomic(metadata_file, dataset_info, indent=2)

    # Write episode index
    episodes_file = output_dir / "episodes.jsonl"
    episode_lines = [json.dumps(ep_info) for ep_info in dataset_info["episodes"]]
    episodes_payload = "\n".join(episode_lines)
    if episodes_payload:
        episodes_payload += "\n"
    write_text_atomic(episodes_file, episodes_payload)

    return {
        "success": True,
        "converted_count": converted_count,
        "skipped_count": skipped_count,
        "skip_rate_percent": skip_rate_percent,
        "total_frames": total_frames,
        "conversion_failures": conversion_failures,
        "output_dir": output_dir,
        "metadata_file": metadata_file,
    }


# =============================================================================
# Import Job
# =============================================================================


def run_local_import_job(
    config: ImportConfig,
    job_metadata: Optional[Dict[str, Any]] = None,
) -> ImportResult:
    """
    Run episode import using already-generated local artifacts.

    Args:
        config: Import configuration
        job_metadata: Optional job.json payload for additional context

    Returns:
        ImportResult with statistics and output paths
    """
    print("\n" + "=" * 80)
    print("GENIE SIM LOCAL EPISODE IMPORT JOB")
    print("=" * 80)
    print(f"Job ID: {config.job_id}")
    print(f"Output: {config.output_dir}")
    print("=" * 80 + "\n")

    result = ImportResult(
        success=False,
        job_id=config.job_id,
    )
    result.output_dir = config.output_dir
    scene_id = os.environ.get("SCENE_ID", "unknown")
    log = logging.LoggerAdapter(logger, {"job_id": JOB_NAME, "scene_id": scene_id})

    idempotency = _resolve_job_idempotency(job_metadata)
    if idempotency:
        existing_manifest = _load_existing_import_manifest(config.output_dir, log)
        existing_idempotency = (
            existing_manifest.get("job_idempotency", {}) if existing_manifest else {}
        )
        if existing_idempotency.get("key") == idempotency.get("key"):
            allow_retry = parse_bool_env(
                os.getenv("ALLOW_IDEMPOTENT_RETRY"), default=False
            )
            import_status = (
                _resolve_manifest_import_status(existing_manifest)
                if existing_manifest
                else "unknown"
            )
            if import_status in {"success", "succeeded", "complete", "completed"}:
                result.success = True
                result.warnings.append(
                    "Duplicate import detected; matching import_manifest.json "
                    "already exists for this idempotency key."
                )
                return result
            if import_status in {"failed", "partial"} and not allow_retry:
                result.errors.append(
                    "Previous import for this idempotency key did not complete "
                    f"successfully (status: {import_status}). Set "
                    "ALLOW_IDEMPOTENT_RETRY=true to rerun."
                )
                return result
            if import_status in {"failed", "partial"} and allow_retry:
                warning_message = (
                    "Retrying local import after previous "
                    f"{import_status} run for this idempotency key."
                )
                print(f"[IMPORT] ⚠️  {warning_message}")
                result.warnings.append(warning_message)
            elif import_status == "unknown" and not allow_retry:
                result.errors.append(
                    "Existing import_manifest.json found for this idempotency key "
                    "with unknown status. Set ALLOW_IDEMPOTENT_RETRY=true to rerun "
                    "or delete the manifest to proceed."
                )
                return result
            elif import_status == "unknown" and allow_retry:
                warning_message = (
                    "Retrying local import with unknown prior status for this "
                    "idempotency key."
                )
                print(f"[IMPORT] ⚠️  {warning_message}")
                result.warnings.append(warning_message)

    bucket = os.getenv("BUCKET")
    output_prefix = os.getenv("OUTPUT_PREFIX", f"scenes/{scene_id}/episodes")

    try:
        recordings_dir = _resolve_recordings_dir(
            config,
            bucket=bucket,
            output_prefix=output_prefix,
            log=log,
        )
    except Exception as exc:
        result.errors.append(f"Failed to resolve recordings directory: {exc}")
        return result

    bundle_root = config.output_dir.resolve()
    input_checksums_verification = None
    input_checksums_path = None
    input_bundle_root = None
    for candidate_root in (bundle_root, recordings_dir):
        candidate_path = candidate_root / "checksums.json"
        if candidate_path.exists():
            input_checksums_path = candidate_path
            input_bundle_root = candidate_root
            input_checksums_verification = verify_checksums_manifest(
                candidate_root,
                candidate_path,
            )
            if not input_checksums_verification["success"]:
                verification_errors = _format_checksums_verification_errors(
                    input_checksums_verification,
                )
                if verification_errors:
                    result.errors.extend(verification_errors)
                else:
                    result.errors.append(
                        "Checksum verification failed for the input bundle."
                    )
                result.success = False
                return result
            break

    schema_errors: List[str] = []
    for episode_file in sorted(recordings_dir.rglob("*.json")):
        try:
            payload = _load_json_file(episode_file)
        except Exception as exc:
            schema_errors.append(
                f"recording {_relative_recordings_path(recordings_dir, config.output_dir, episode_file)}: {exc}"
            )
            continue
        schema_errors.extend(
            _validate_schema_payload(
                payload,
                "geniesim_local_episode.schema.json",
                f"recording {_relative_recordings_path(recordings_dir, config.output_dir, episode_file)}",
            )
        )

    episode_metadata_payload = _collect_local_episode_metadata(recordings_dir)
    episode_metadata_list = episode_metadata_payload["episodes"]
    parse_failure_count = episode_metadata_payload["parse_failure_count"]
    parse_failures = episode_metadata_payload["parse_failures"]
    if not episode_metadata_list:
        result.errors.append(f"No local episode files found under {recordings_dir}")
        return result
    result.total_episodes_downloaded = len(episode_metadata_list)
    if result.total_episodes_downloaded < config.min_episodes_required:
        result.errors.append(
            "Insufficient episodes discovered for import: "
            f"{result.total_episodes_downloaded} < {config.min_episodes_required}"
        )
        return result
    if parse_failure_count > 0:
        parse_failure_message = (
            f"{parse_failure_count} local episode files failed to parse"
        )
        if config.fail_on_partial_error:
            result.errors.append(parse_failure_message)
            result.success = False
            return result
        result.warnings.append(parse_failure_message)
    result.episodes_parse_failed = parse_failure_count
    result.episode_parse_failures = parse_failures

    validator = ImportedEpisodeValidator(
        min_quality_score=config.min_quality_score,
        require_parquet_validation=config.enable_validation,
        dimension_thresholds=config.quality_component_thresholds,
    )
    validation_summary = validator.validate_batch(episode_metadata_list, recordings_dir)
    result.quality_component_failed_count = int(
        validation_summary.get("quality_component_failed_count", 0)
    )
    result.quality_component_failure_counts = dict(
        validation_summary.get("quality_component_failure_counts", {}) or {}
    )
    result.quality_component_thresholds = dict(config.quality_component_thresholds)
    parquet_validation_errors = validation_summary.get("parquet_validation_errors") or []
    production_mode = resolve_production_mode()
    if parquet_validation_errors:
        result.errors.extend(parquet_validation_errors)
        if config.enable_validation or production_mode:
            result.success = False
            return result
    failed_episode_ids = [
        entry["episode_id"]
        for entry in validation_summary["episode_results"]
        if not entry["passed"]
    ]
    failed_episode_id_set = set(failed_episode_ids)
    filtered_episode_ids: List[str] = []
    filtered_episode_metadata_list = episode_metadata_list
    if not config.fail_on_partial_error and failed_episode_ids:
        filtered_episode_ids = failed_episode_ids
        filtered_episode_metadata_list = [
            ep for ep in episode_metadata_list if ep.episode_id not in failed_episode_id_set
        ]
    if validation_summary["failed_count"] > 0:
        failure_message = (
            f"{validation_summary['failed_count']} local episodes failed validation"
        )
        if config.fail_on_partial_error:
            result.errors.append(failure_message)
            result.success = False
        else:
            result.warnings.append(failure_message)
            if filtered_episode_ids:
                result.warnings.append(
                    "Excluding failed episodes from downstream processing and manifest: "
                    + ", ".join(filtered_episode_ids)
                )

    validated_episode_metadata_list = [
        ep for ep in episode_metadata_list if ep.episode_id not in failed_episode_id_set
    ]
    realtime_config = _resolve_realtime_stream_config(
        min_quality_score=config.min_quality_score,
        production_mode=resolve_production_mode(),
        log=log,
    )
    if realtime_config:
        feedback_loop = RealtimeFeedbackLoop(realtime_config, enable_logging=True)
        log.info(
            "Streaming %s validated episodes to realtime feedback loop.",
            len(validated_episode_metadata_list),
        )
        _stream_realtime_episodes(
            feedback_loop,
            validated_episode_metadata_list,
            job_id=config.job_id,
            scene_id=scene_id,
            robot_type=(job_metadata or {}).get("robot_type"),
            log=log,
        )
        stats = feedback_loop.get_statistics()
        log.info(
            "Realtime streaming summary: queued=%s filtered=%s sent=%s rejected=%s",
            stats.get("episodes_queued"),
            stats.get("episodes_filtered"),
            stats.get("episodes_sent"),
            stats.get("episodes_rejected"),
        )

    lerobot_dir = config.output_dir / "lerobot"
    dataset_info_path = lerobot_dir / "dataset_info.json"
    dataset_info_payload = None
    conversion_result: Optional[Dict[str, Any]] = None
    lerobot_error = None
    force_conversion_raw = os.getenv("FORCE_LEROBOT_CONVERSION")
    force_conversion = parse_bool_env(force_conversion_raw, default=False)
    require_lerobot_explicit = (
        config.require_lerobot_raw_value is not None and config.require_lerobot
    )
    should_convert = force_conversion or require_lerobot_explicit or not lerobot_dir.exists()
    if should_convert:
        try:
            conversion_result = convert_to_lerobot(
                episodes_dir=recordings_dir,
                output_dir=lerobot_dir,
                episode_metadata_list=validated_episode_metadata_list,
                min_quality_score=config.min_quality_score,
                quality_component_thresholds=config.quality_component_thresholds,
                job_id=config.job_id,
                scene_id=scene_id,
            )
            result.lerobot_conversion_success = bool(
                conversion_result.get("success", False)
                if conversion_result is not None
                else False
            )
        except Exception as exc:
            result.lerobot_conversion_success = False
            lerobot_error = f"LeRobot conversion failed: {exc}"
            if config.require_lerobot:
                result.errors.append(lerobot_error)
            else:
                result.warnings.append(lerobot_error)

    # =========================================================================
    # Cosmos Policy Export (runs alongside LeRobot conversion)
    # =========================================================================
    cosmos_policy_dir = config.output_dir / "cosmos_policy"
    cosmos_policy_error = None
    cosmos_policy_enabled = parse_bool_env(
        os.getenv("ENABLE_COSMOS_POLICY_EXPORT"), default=True
    )
    if cosmos_policy_enabled and result.lerobot_conversion_success:
        try:
            from tools.cosmos_policy_adapter import CosmosPolicyExporter, CosmosPolicyConfig

            cosmos_config = CosmosPolicyConfig()
            cosmos_exporter = CosmosPolicyExporter(
                output_dir=cosmos_policy_dir,
                config=cosmos_config,
                verbose=True,
            )

            # Build episode dicts from validated metadata + parquet files
            cosmos_episodes = []
            for ep_metadata in validated_episode_metadata_list:
                episode_file = recordings_dir / f"{ep_metadata.episode_id}.parquet"
                if not episode_file.exists():
                    continue

                # Load frame data from parquet
                frames = _load_episode_frames_for_cosmos(episode_file)
                if not frames:
                    continue

                cosmos_episodes.append({
                    "episode_id": ep_metadata.episode_id,
                    "task": ep_metadata.task_description if hasattr(ep_metadata, "task_description") else "manipulation task",
                    "frames": frames,
                    "success": ep_metadata.validation_passed,
                    "quality_score": ep_metadata.quality_score,
                })

            if cosmos_episodes:
                robot_type = os.getenv("ROBOT_TYPE", "franka")
                cosmos_exporter.export_episodes(
                    episodes=cosmos_episodes,
                    robot_type=robot_type,
                    scene_id=scene_id or "unknown",
                    source_videos_dir=config.output_dir / "lerobot" / "videos",
                )
                log.info(
                    "Cosmos Policy export complete: %d episodes -> %s",
                    len(cosmos_episodes),
                    cosmos_policy_dir,
                )
            else:
                log.warning("No valid episodes for Cosmos Policy export")
        except Exception as exc:
            cosmos_policy_error = f"Cosmos Policy export failed: {exc}"
            result.warnings.append(cosmos_policy_error)
            log.warning(cosmos_policy_error)

    if dataset_info_path.exists():
        try:
            dataset_info_payload = _load_dataset_info_with_migration(
                dataset_info_path,
                log,
                required=True,
            )
            schema_errors.extend(
                _validate_schema_payload(
                    dataset_info_payload,
                    "geniesim_local_dataset_info.schema.json",
                    f"metadata {dataset_info_path.relative_to(config.output_dir)}",
                )
            )
        except Exception as exc:
            schema_errors.append(
                f"metadata {dataset_info_path.relative_to(config.output_dir)}: {exc}"
            )
    else:
        schema_errors.append(
            f"metadata {dataset_info_path.relative_to(config.output_dir)}: missing dataset_info.json"
        )

    lerobot_metadata_validation = _validate_lerobot_metadata_files(
        config.output_dir,
        lerobot_dir,
        config.enable_validation,
    )
    schema_errors.extend(lerobot_metadata_validation["schema_errors"])

    if schema_errors:
        if resolve_production_mode():
            result.errors.extend(schema_errors)
            result.success = False
            import_manifest_path = config.output_dir / "import_manifest.json"
            if import_manifest_path.exists():
                result.import_manifest_path = import_manifest_path
                _update_import_manifest_status(
                    import_manifest_path,
                    "failed",
                    success=False,
                )
            return result
        else:
            result.warnings.extend(schema_errors)

    episode_content_hashes: Dict[str, str] = {}
    episode_content_manifests: Dict[str, List[Dict[str, Any]]] = {}
    missing_content_hashes: List[str] = []
    for episode in episode_metadata_list:
        manifest_entries = _compute_episode_content_manifest(
            recordings_dir,
            episode.episode_id,
        )
        if not manifest_entries:
            missing_content_hashes.append(episode.episode_id)
            continue
        content_hash = _compute_episode_content_hash(manifest_entries)
        episode.episode_content_hash = content_hash
        episode_content_hashes[episode.episode_id] = content_hash
        episode_content_manifests[episode.episode_id] = manifest_entries
    result.episode_content_hashes = episode_content_hashes
    result.episode_content_manifests = episode_content_manifests
    if missing_content_hashes:
        result.warnings.append(
            "Missing content hash sources for episodes: "
            + ", ".join(sorted(missing_content_hashes))
        )
    if isinstance(dataset_info_payload, dict):
        updated = False
        for entry in dataset_info_payload.get("episodes", []):
            episode_id = entry.get("episode_id")
            if not episode_id:
                continue
            content_hash = episode_content_hashes.get(episode_id)
            if content_hash and entry.get("content_hash") != content_hash:
                entry["content_hash"] = content_hash
                updated = True
        if updated and dataset_info_path.exists():
            write_json_atomic(dataset_info_path, dataset_info_payload, indent=2)

    total_size_bytes = 0
    for episode_file in recordings_dir.rglob("*.json"):
        total_size_bytes += episode_file.stat().st_size

    low_quality_episodes = [
        ep
        for ep in filtered_episode_metadata_list
        if ep.quality_score < config.min_quality_score
    ]
    component_failed_episodes = []
    if config.quality_component_thresholds:
        for ep in filtered_episode_metadata_list:
            failures = _evaluate_quality_component_thresholds(
                ep.quality_components,
                config.quality_component_thresholds,
            )
            if failures:
                component_failed_episodes.append(
                    {"episode_id": ep.episode_id, "failed_dimensions": failures}
                )
    result.episodes_passed_validation = validation_summary["passed_count"]
    result.episodes_filtered = len(filtered_episode_ids)
    quality_scores = [ep.quality_score for ep in filtered_episode_metadata_list]
    result.average_quality_score = float(np.mean(quality_scores)) if quality_scores else 0.0
    quality_min_score = float(np.min(quality_scores)) if quality_scores else 0.0
    quality_max_score = float(np.max(quality_scores)) if quality_scores else 0.0
    result.quality_min_score = quality_min_score
    result.quality_max_score = quality_max_score

    if low_quality_episodes:
        result.errors.append(
            f"{len(low_quality_episodes)} local episodes below min_quality_score={config.min_quality_score}"
        )
    if component_failed_episodes:
        failed_dimensions = sorted(
            {dim for entry in component_failed_episodes for dim in entry["failed_dimensions"]}
        )
        result.errors.append(
            "Local episodes below quality component thresholds: "
            f"{len(component_failed_episodes)} episode(s) failed "
            f"({', '.join(failed_dimensions)})."
        )

    if config.enable_validation:
        result.warnings.append("Local import skipped API validation; local episodes are assumed valid.")

    lerobot_episode_files = []
    lerobot_skipped_count = 0
    lerobot_skip_rate_percent = 0.0
    conversion_failures: List[Dict[str, Any]] = []
    converted_count: Optional[int] = None
    if conversion_result is not None:
        converted_count = int(conversion_result.get("converted_count", 0))
        lerobot_skipped_count = int(conversion_result.get("skipped_count", 0))
        lerobot_skip_rate_percent = float(conversion_result.get("skip_rate_percent", 0.0))
        conversion_failures = conversion_result.get("conversion_failures", []) or []
    if lerobot_dir.exists():
        lerobot_episode_files = [
            path for path in lerobot_dir.glob("*.json") if path.name != "dataset_info.json"
        ]
        if conversion_result is None:
            result.lerobot_conversion_success = True
        if isinstance(dataset_info_payload, dict):
            if converted_count is None:
                converted_count = int(
                    dataset_info_payload.get("total_episodes", len(lerobot_episode_files))
                )
            lerobot_skipped_count = int(dataset_info_payload.get("skipped_episodes", 0))
            lerobot_skip_rate_percent = float(
                dataset_info_payload.get("skip_rate_percent", 0.0)
            )
            conversion_failures = dataset_info_payload.get("conversion_failures", []) or []
            if not dataset_info_payload.get("skip_rate_percent"):
                total_source_episodes = (
                    int(dataset_info_payload.get("total_episodes", 0)) + lerobot_skipped_count
                )
                lerobot_skip_rate_percent = (
                    (lerobot_skipped_count / total_source_episodes) * 100.0
                    if total_source_episodes
                    else 0.0
                )
    else:
        if conversion_result is None:
            result.lerobot_conversion_success = False
            if lerobot_error is None:
                lerobot_error = "LeRobot output directory not found for local import."
            if config.require_lerobot:
                result.errors.append(lerobot_error)
            else:
                result.warnings.append(lerobot_error)

    if conversion_failures:
        failure_id_set = {entry.get("episode_id", "unknown") for entry in conversion_failures}
        failure_ids = ", ".join(sorted(failure_id_set))
        log.warning(
            "LeRobot conversion failures treated as skipped episodes: %s",
            failure_ids,
        )
        if lerobot_skipped_count < len(failure_id_set):
            lerobot_skipped_count = len(failure_id_set)
            if not (
                isinstance(dataset_info_payload, dict)
                and dataset_info_payload.get("skip_rate_percent")
            ):
                total_source_episodes = (
                    int(dataset_info_payload.get("total_episodes", 0))
                    if isinstance(dataset_info_payload, dict)
                    else max(converted_count or 0, 0)
                ) + lerobot_skipped_count
                lerobot_skip_rate_percent = (
                    (lerobot_skipped_count / total_source_episodes) * 100.0
                    if total_source_episodes
                    else 0.0
                )
        if config.require_lerobot:
            result.errors.append(
                "LeRobot conversion failures: "
                + ", ".join(sorted(failure_id_set))
            )

    if converted_count is None:
        converted_count = len(lerobot_episode_files)

    if lerobot_skip_rate_percent > config.lerobot_skip_rate_max:
        result.errors.append(
            "LeRobot skip rate "
            f"{lerobot_skip_rate_percent:.2f}% exceeded max "
            f"{config.lerobot_skip_rate_max:.2f}%"
        )

    if lerobot_skipped_count > 0:
        print(
            "[IMPORT] ⚠️  LeRobot conversion skipped episodes: "
            f"{lerobot_skipped_count} ({lerobot_skip_rate_percent:.2f}%)"
        )

    cost_summary = _build_cost_summary(scene_id, log)
    _update_dataset_info_cost_summary(
        cost_summary=cost_summary,
        dataset_info_payload=dataset_info_payload,
        dataset_info_path=dataset_info_path if dataset_info_path.exists() else None,
    )

    # Write machine-readable import manifest for workflows
    import_manifest_path = config.output_dir / "import_manifest.json"
    gcs_output_path = config.gcs_output_path
    output_dir_str = str(config.output_dir)
    if not gcs_output_path and output_dir_str.startswith("/mnt/gcs/"):
        gcs_output_path = "gs://" + output_dir_str[len("/mnt/gcs/"):]

    metrics = get_metrics()
    if lerobot_skipped_count > 0:
        metrics.geniesim_import_episodes_skipped_total.inc(
            lerobot_skipped_count,
            labels={"scene_id": scene_id, "job_id": config.job_id},
        )
        if metrics.enable_logging:
            print(
                "[METRICS] Genie Sim import skipped episodes: "
                f"{lerobot_skipped_count} (scene: {scene_id}, job: {config.job_id})"
            )
    metrics_summary = {
        "backend": metrics.backend.value,
        "stats": metrics.get_stats(),
    }
    try:
        if job_metadata:
            export_job_metrics(job_payload=job_metadata, scene_id=scene_id)
        elif config.job_metadata_path and bucket:
            job_metadata_path = _resolve_local_path(bucket, config.job_metadata_path)
            export_job_metrics(job_json_path=job_metadata_path, scene_id=scene_id)
    except Exception as exc:
        log.warning("Failed to export Genie Sim job metrics summary: %s", exc)

    episode_checksums = []
    for episode_file in sorted(recordings_dir.rglob("*.json")):
        episode_id = episode_file.stem
        episode_checksums.append({
            "episode_id": episode_id,
            "file_name": _relative_recordings_path(
                recordings_dir,
                config.output_dir,
                episode_file,
            ),
            "sha256": _sha256_file(episode_file),
            "content_hash": episode_content_hashes.get(episode_id),
        })

    lerobot_checksums = {
        "dataset_info": None,
        "episodes_index": None,
        "episodes_parquet": None,
        "episode_index": None,
        "episodes": [],
        "videos": [],
    }
    dataset_info_path = lerobot_dir / "dataset_info.json"
    if dataset_info_path.exists():
        lerobot_checksums["dataset_info"] = _sha256_file(dataset_info_path)
    episodes_index_path = lerobot_metadata_validation.get("episodes_index_path")
    if isinstance(episodes_index_path, Path) and episodes_index_path.exists():
        lerobot_checksums["episodes_index"] = _sha256_file(episodes_index_path)
    episodes_parquet_path = lerobot_metadata_validation.get("episodes_parquet_path")
    if isinstance(episodes_parquet_path, Path) and episodes_parquet_path.exists():
        lerobot_checksums["episodes_parquet"] = _sha256_file(episodes_parquet_path)
    episode_index_path = lerobot_metadata_validation.get("episode_index_path")
    if isinstance(episode_index_path, Path) and episode_index_path.exists():
        lerobot_checksums["episode_index"] = _sha256_file(episode_index_path)
    for lerobot_file in sorted(lerobot_episode_files):
        lerobot_checksums["episodes"].append({
            "file_name": lerobot_file.name,
            "sha256": _sha256_file(lerobot_file),
        })
    videos_dir = lerobot_dir / "videos"
    if videos_dir.exists():
        for video_file in sorted(videos_dir.rglob("*.mp4")):
            lerobot_checksums["videos"].append({
                "file_name": video_file.relative_to(lerobot_dir).as_posix(),
                "sha256": _sha256_file(video_file),
            })

    episode_paths = sorted(recordings_dir.rglob("*.json"))
    metadata_paths = get_lerobot_metadata_paths(config.output_dir)
    missing_metadata_files = []
    lerobot_info_path = lerobot_metadata_validation.get("info_path")
    if isinstance(lerobot_info_path, Path):
        info_rel_path = lerobot_info_path.relative_to(config.output_dir).as_posix()
    else:
        info_rel_path = (config.output_dir / "lerobot" / "meta" / "info.json").relative_to(
            config.output_dir
        ).as_posix()
    if not lerobot_info_path or not Path(lerobot_info_path).exists():
        missing_metadata_files.append(info_rel_path)
    readme_path = _write_lerobot_readme(config.output_dir, lerobot_dir)
    directory_checksums = build_directory_checksums(
        config.output_dir,
        exclude_paths=[import_manifest_path],
    )
    if not _relative_to_bundle(config.output_dir, recordings_dir).startswith("recordings"):
        recordings_checksums = build_directory_checksums(recordings_dir)
        for rel_path, checksum in recordings_checksums.items():
            recordings_key = (Path("recordings") / rel_path).as_posix()
            directory_checksums[recordings_key] = checksum
    episode_rel_paths = {
        _relative_recordings_path(recordings_dir, config.output_dir, path)
        for path in episode_paths
    }
    metadata_rel_paths = {path.relative_to(config.output_dir).as_posix() for path in metadata_paths}
    file_checksums = {
        "episodes": {
            rel_path: checksum
            for rel_path, checksum in directory_checksums.items()
            if rel_path in episode_rel_paths
        },
        "metadata": {
            rel_path: checksum
            for rel_path, checksum in directory_checksums.items()
            if rel_path in metadata_rel_paths
        },
        "missing_episode_ids": [],
        "missing_metadata_files": missing_metadata_files,
    }
    checksums_payload = {
        "download_manifest": None,
        "episodes": episode_checksums,
        "episode_content_hashes": episode_content_hashes,
        "episode_content_manifest": episode_content_manifests,
        "filtered_episodes": filtered_episode_ids,
        "lerobot": lerobot_checksums,
        "metadata": file_checksums["metadata"],
        "missing_episode_ids": file_checksums["missing_episode_ids"],
        "missing_metadata_files": file_checksums["missing_metadata_files"],
        "episode_files": file_checksums["episodes"],
        "bundle_files": dict(directory_checksums),
    }
    checksums_path, checksums_signature = _write_checksums_file(
        config.output_dir,
        directory_checksums,
    )
    if checksums_signature:
        checksums_payload["signature"] = checksums_signature
    checksums_rel_path = checksums_path.relative_to(config.output_dir).as_posix()
    checksums_entry = {
        "sha256": _sha256_file(checksums_path),
        "size_bytes": checksums_path.stat().st_size,
    }
    checksums_payload["metadata"][checksums_rel_path] = checksums_entry
    checksums_payload["bundle_files"][checksums_rel_path] = checksums_entry
    config_snapshot = {
        "env": snapshot_env(ENV_SNAPSHOT_KEYS),
        "config": {
            "job_id": config.job_id,
            "output_dir": output_dir_str,
            "gcs_output_path": gcs_output_path,
            "enable_gcs_uploads": config.enable_gcs_uploads,
            "min_quality_score": config.min_quality_score,
            "quality_component_thresholds": config.quality_component_thresholds,
            "enable_validation": config.enable_validation,
            "filter_low_quality": config.filter_low_quality,
            "require_lerobot": config.require_lerobot,
            "require_lerobot_default": config.require_lerobot_default,
            "require_lerobot_source": config.require_lerobot_source,
            "require_lerobot_raw_value": config.require_lerobot_raw_value,
            "lerobot_skip_rate_max": config.lerobot_skip_rate_max,
            "min_episodes_required": config.min_episodes_required,
            "poll_interval": config.poll_interval,
            "wait_for_completion": config.wait_for_completion,
            "fail_on_partial_error": config.fail_on_partial_error,
            "job_metadata_path": config.job_metadata_path,
            "local_episodes_prefix": config.local_episodes_prefix,
        },
        "job_metadata": job_metadata or {},
    }
    base_provenance = {
        "source": "genie_sim",
        "job_id": config.job_id,
        "scene_id": scene_id or None,
        "imported_by": "BlueprintPipeline",
        "importer": "genie-sim-import-job",
        "client_mode": "local",
    }
    provenance = collect_provenance(REPO_ROOT, config_snapshot)
    provenance.update(base_provenance)

    regression_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": {},
        "error": None,
    }
    try:
        regression_payload["metrics"] = compute_regression_metrics(config.output_dir)
    except Exception as exc:
        regression_payload["error"] = f"{type(exc).__name__}: {exc}"

    input_checksums_payload = None
    if input_checksums_verification is not None:
        input_checksums_payload = dict(input_checksums_verification)
        if input_checksums_path is not None:
            input_checksums_payload["path"] = _relative_to_bundle(
                bundle_root,
                input_checksums_path,
            )
        if input_bundle_root is not None:
            input_checksums_payload["bundle_root"] = _relative_to_bundle(
                bundle_root,
                input_bundle_root,
            )

    import_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "schema_definition": MANIFEST_SCHEMA_DEFINITION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "job_id": config.job_id,
        "output_dir": output_dir_str,
        "gcs_output_path": gcs_output_path,
        "import_status": "in_progress",
        "success": False,
        "job_idempotency": {
            "key": idempotency.get("key") if idempotency else None,
            "first_submitted_at": idempotency.get("first_submitted_at") if idempotency else None,
            "job_metadata_path": config.job_metadata_path,
        },
        "readme_path": _relative_to_bundle(bundle_root, readme_path),
        "checksums_path": _relative_to_bundle(bundle_root, checksums_path),
        "episodes": {
            "downloaded": result.total_episodes_downloaded,
            "passed_validation": result.episodes_passed_validation,
            "filtered": result.episodes_filtered,
            "excluded_failed_count": len(filtered_episode_ids),
            "excluded_failed_ids": filtered_episode_ids,
            "download_errors": 0,
            "parse_failed": result.episodes_parse_failed,
            "parse_failures": result.episode_parse_failures,
            "min_required": config.min_episodes_required,
        },
        "quality": {
            "average_score": result.average_quality_score,
            "min_score": quality_min_score,
            "max_score": quality_max_score,
            "threshold": config.min_quality_score,
            "validation_enabled": config.enable_validation,
            "component_thresholds": config.quality_component_thresholds,
            "component_failed_episodes": result.quality_component_failed_count,
            "component_failure_counts": result.quality_component_failure_counts,
        },
        "quality_config": {
            "min_quality_score": config.min_quality_score,
            "filter_low_quality": config.filter_low_quality,
            "dimension_thresholds": config.quality_component_thresholds,
            "range": {
                "min_allowed": QUALITY_CONFIG.min_allowed,
                "max_allowed": QUALITY_CONFIG.max_allowed,
            },
            "defaults": {
                "min_quality_score": QUALITY_CONFIG.default_min_quality_score,
                "filter_low_quality": QUALITY_CONFIG.default_filter_low_quality,
            },
            "source_path": QUALITY_CONFIG.source_path,
        },
        "lerobot": {
            "conversion_success": result.lerobot_conversion_success,
            "converted_count": converted_count,
            "skipped_episodes": lerobot_skipped_count,
            "skip_rate_percent": lerobot_skip_rate_percent,
            "conversion_failures": conversion_failures,
            "output_dir": _relative_to_bundle(bundle_root, lerobot_dir),
            "error": lerobot_error,
            "required": config.require_lerobot,
            "required_default": config.require_lerobot_default,
            "required_source": config.require_lerobot_source,
            "required_raw_value": config.require_lerobot_raw_value,
            "skip_rate_max": config.lerobot_skip_rate_max,
        },
        "cosmos_policy": {
            "enabled": cosmos_policy_enabled,
            "export_success": cosmos_policy_dir.exists() and cosmos_policy_error is None,
            "output_dir": _relative_to_bundle(bundle_root, cosmos_policy_dir) if cosmos_policy_dir.exists() else None,
            "error": cosmos_policy_error,
        },
        "validation": {
            "episodes": {
                **validation_summary,
                "failed_episode_ids": failed_episode_ids,
                "filtered_episode_ids": filtered_episode_ids,
                "filtered_episode_count": len(filtered_episode_ids),
                "fail_on_partial_error": config.fail_on_partial_error,
            },
        },
        "verification": {
            "checksums": {
                "input_bundle": input_checksums_payload,
            }
            if input_checksums_payload is not None
            else {},
        },
        "metrics_summary": metrics_summary,
        "regression_metrics": regression_payload,
        "checksums": checksums_payload,
        "provenance": provenance,
    }

    _attach_cost_summary(import_manifest, cost_summary)

    with open(import_manifest_path, "w") as f:
        json.dump(import_manifest, f, indent=2)

    bundle_directories = [lerobot_dir]
    if cosmos_policy_dir.exists():
        bundle_directories.append(cosmos_policy_dir)
    package_path = _create_bundle_package(
        config.output_dir,
        f"lerobot_bundle_{config.job_id}.tar.gz",
        files=[import_manifest_path, readme_path, checksums_path],
        directories=bundle_directories,
    )
    package_checksum = _sha256_file(package_path)
    checksums_payload["bundle_files"][_relative_to_bundle(bundle_root, package_path)] = {
        "sha256": package_checksum,
        "size_bytes": package_path.stat().st_size,
    }
    file_inventory = build_file_inventory(config.output_dir, exclude_paths=[import_manifest_path])
    upload_summary: Dict[str, Any] = {
        "status": "not_configured",
        "started_at": None,
        "completed_at": None,
        "total_files": 0,
        "uploaded": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
    }
    if not config.enable_gcs_uploads:
        upload_summary["status"] = "disabled"
    elif output_dir_str.startswith("/mnt/gcs/"):
        upload_summary["status"] = "not_required"
    elif gcs_output_path:
        upload_summary = _upload_output_dir(config.output_dir, gcs_output_path)
    result.upload_status = upload_summary["status"]
    result.upload_failures = upload_summary["failures"]
    result.upload_started_at = upload_summary["started_at"]
    result.upload_completed_at = upload_summary["completed_at"]
    if upload_summary["status"] == "failed":
        failed_count = upload_summary.get("failed", 0)
        result.errors.append(
            f"GCS upload failed: {failed_count} file(s) could not be uploaded"
        )

    asset_provenance_path = _resolve_asset_provenance_reference(
        bundle_root=bundle_root,
        output_dir=config.output_dir,
        job_metadata=job_metadata,
    )
    import_manifest["package"] = {
        "path": _relative_to_bundle(bundle_root, package_path),
        "sha256": package_checksum,
        "size_bytes": package_path.stat().st_size,
        "format": "tar.gz",
        "includes": [
            _relative_to_bundle(bundle_root, import_manifest_path),
            _relative_to_bundle(bundle_root, readme_path),
            _relative_to_bundle(bundle_root, checksums_path),
            _relative_to_bundle(bundle_root, lerobot_dir),
        ],
    }
    import_manifest["file_inventory"] = file_inventory
    import_manifest["asset_provenance_path"] = asset_provenance_path
    import_manifest["upload_status"] = upload_summary["status"]
    import_manifest["upload_failures"] = upload_summary["failures"]
    import_manifest["upload_started_at"] = upload_summary["started_at"]
    import_manifest["upload_completed_at"] = upload_summary["completed_at"]
    import_manifest["upload_summary"] = {
        "total_files": upload_summary["total_files"],
        "uploaded": upload_summary["uploaded"],
        "skipped": upload_summary["skipped"],
        "failed": upload_summary["failed"],
    }
    checksums_verification = verify_checksums_manifest(bundle_root, checksums_path)
    import_manifest.setdefault("verification", {}).setdefault("checksums", {})[
        "output_bundle"
    ] = checksums_verification
    import_manifest["checksums"]["metadata"]["import_manifest.json"] = {
        "sha256": compute_manifest_checksum(import_manifest),
    }

    with open(import_manifest_path, "w") as f:
        json.dump(import_manifest, f, indent=2)

    result.import_manifest_path = import_manifest_path

    print("=" * 80)
    print("IMPORT MANIFEST CHECKSUM VERIFICATION")
    print("=" * 80)
    verify_exit_code = verify_manifest(import_manifest_path)
    if verify_exit_code != 0:
        result.errors.append("Import manifest verification failed.")
        result.success = False
        print("[IMPORT] ❌ Import manifest verification failed; aborting job.")
        _update_import_manifest_status(import_manifest_path, "failed", success=False)
        print("=" * 80 + "\n")
        return result
    print("[IMPORT] ✅ Import manifest verification succeeded")
    print("=" * 80 + "\n")

    print("=" * 80)
    print("CHECKSUM VERIFICATION")
    print("=" * 80)
    if not checksums_verification["success"]:
        verification_errors = _format_checksums_verification_errors(checksums_verification)
        result.checksum_verification_errors.extend(verification_errors)
        remediation = (
            "Re-download the bundle to ensure artifacts are intact, or rerun the import "
            "to regenerate checksums.json from the source files."
        )
        result.errors.append(
            "Checksum verification failed. " + remediation
        )
        print("[IMPORT] ❌ " + result.errors[-1])
        print("[IMPORT] ❌ Checksum verification details:")
        for error in verification_errors:
            print(f"[IMPORT]   - {error}")
        result.success = False
        _update_import_manifest_status(import_manifest_path, "failed", success=False)
        print("=" * 80 + "\n")
        return result
    else:
        print("[IMPORT] ✅ Checksums.json verification succeeded")
    print("=" * 80 + "\n")

    manifest_checksum_result = verify_import_manifest_checksum(import_manifest_path)
    if not manifest_checksum_result["success"]:
        result.checksum_verification_errors.extend(manifest_checksum_result["errors"])
        result.errors.append(
            "Import manifest checksum validation failed. "
            "Re-run the import to regenerate a consistent manifest."
        )
        print("[IMPORT] ❌ " + result.errors[-1])
        print("[IMPORT] ❌ Import manifest checksum verification details:")
        for error in manifest_checksum_result["errors"]:
            print(f"[IMPORT]   - {error}")
        result.success = False
        _update_import_manifest_status(import_manifest_path, "failed", success=False)
        print("=" * 80 + "\n")
        return result

    result.checksum_verification_passed = (
        checksums_verification["success"] and manifest_checksum_result["success"]
    )
    if not result.checksum_verification_passed:
        result.success = False
        _update_import_manifest_status(import_manifest_path, "failed", success=False)
        return result

    result.success = len(result.errors) == 0
    import_status = _resolve_import_status(result)
    _update_import_manifest_status(
        import_manifest_path, import_status, success=result.success
    )

    print("=" * 80)
    print("LOCAL IMPORT COMPLETE")
    print("=" * 80)
    print(f"{'✅' if result.success else '❌'} Imported {result.episodes_passed_validation} local episodes")
    if result.episodes_parse_failed:
        print(f"[IMPORT] ⚠️  Episode parse failures: {result.episodes_parse_failed}")
    print(f"Output directory: {result.output_dir}")
    print(f"Manifest: {result.import_manifest_path}")
    print(
        "Checksum verification: "
        f"{'✅' if result.checksum_verification_passed else '❌'}"
    )
    print("=" * 80 + "\n")

    return result


# =============================================================================
# Main Entry Point
# =============================================================================


def _emit_import_quality_gate(
    result: ImportResult,
    scene_id: str,
    job_metadata: Optional[Dict[str, Any]] = None,
) -> List[QualityGateResult]:
    checkpoint = QualityGateCheckpoint.GENIESIM_IMPORT_COMPLETE
    registry = QualityGateRegistry(verbose=_resolve_debug_mode())

    local_execution = (job_metadata or {}).get("local_execution", {})
    robot_type = (job_metadata or {}).get("robot_type")
    if robot_type and isinstance(local_execution, dict) and "by_robot" in local_execution:
        per_robot = local_execution.get("by_robot", {}).get(robot_type, {})
        collision_free_rate = per_robot.get("collision_free_rate")
        task_success_rate = per_robot.get("task_success_rate")
        episodes_collected = per_robot.get("episodes_collected")
        by_robot = {robot_type: per_robot}
    else:
        collision_free_rate = local_execution.get("collision_free_rate")
        task_success_rate = local_execution.get("task_success_rate")
        episodes_collected = local_execution.get("episodes_collected")
        by_robot = local_execution.get("by_robot")

    def _maybe_emit_rate(metric_name: str, metric_value: Any) -> None:
        if metric_value is None:
            return
        try:
            rate_value = float(metric_value)
        except (TypeError, ValueError):
            return
        metrics = get_metrics()
        labels = {"scene_id": scene_id, "job": JOB_NAME}
        if robot_type:
            labels["robot_type"] = robot_type
        metric = getattr(metrics, metric_name, None)
        if metric is not None:
            metric.set(rate_value, labels=labels)

    _maybe_emit_rate("collision_free_rate", collision_free_rate)
    _maybe_emit_rate("task_success_rate", task_success_rate)

    def _check_import(ctx: Dict[str, Any]) -> QualityGateResult:
        passed = ctx["success"]
        # If GCS upload was configured and failed, the import is not complete
        upload_status = ctx.get("upload_status")
        if upload_status == "failed":
            passed = False
        severity = QualityGateSeverity.INFO if passed else QualityGateSeverity.ERROR
        message = (
            "Genie Sim import completed successfully"
            if passed
            else "Genie Sim import completed with errors"
        )
        details = {
            "episodes_passed_validation": ctx["episodes_passed_validation"],
            "episodes_filtered": ctx["episodes_filtered"],
            "episodes_parse_failed": ctx["episodes_parse_failed"],
            "average_quality_score": ctx["average_quality_score"],
            "import_manifest_path": ctx["import_manifest_path"],
            "errors": ctx["errors"],
            "warnings": ctx["warnings"],
            "checksum_verification_passed": ctx["checksum_verification_passed"],
            "upload_status": upload_status,
            "firebase_upload_required": ctx.get("firebase_upload_required", False),
        }
        return QualityGateResult(
            gate_id="import_complete",
            checkpoint=checkpoint,
            passed=passed,
            severity=severity,
            message=message,
            details=details,
        )

    registry.register(QualityGate(
        id="import_complete",
        name="Genie Sim Import Complete",
        checkpoint=checkpoint,
        severity=QualityGateSeverity.INFO,
        description="Emit a completion gate for Genie Sim import validation.",
        check_fn=_check_import,
    ))

    context = {
        "scene_id": scene_id,
        "success": result.success,
        "episodes_passed_validation": result.episodes_passed_validation,
        "episodes_filtered": result.episodes_filtered,
        "episodes_parse_failed": result.episodes_parse_failed,
        "average_quality_score": result.average_quality_score,
        "collision_free_rate": collision_free_rate,
        "task_success_rate": task_success_rate,
        "episodes_collected": episodes_collected,
        "by_robot": by_robot,
        "import_manifest_path": str(result.import_manifest_path)
        if result.import_manifest_path
        else None,
        "errors": result.errors,
        "warnings": result.warnings,
        "checksum_verification_passed": result.checksum_verification_passed,
        "upload_status": result.upload_status,
    }
    return registry.run_checkpoint(checkpoint, context)


def _quality_gate_failure_detected(
    results: Optional[List[QualityGateResult]],
) -> bool:
    if not results:
        return False
    for result in results:
        severity = (
            result.severity.value
            if isinstance(result.severity, QualityGateSeverity)
            else str(result.severity)
        )
        if result.passed is False and severity.lower() in {"error", "critical"}:
            return True
    return False


def main(input_params: Optional[Dict[str, Any]] = None):
    """Main entry point for import job."""
    if input_params is None:
        input_params = {}
    debug_mode = _resolve_debug_mode()
    if debug_mode:
        os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["REQUEST_ID"] = ensure_request_id()
    init_logging(level=logging.DEBUG if debug_mode else None)
    log = logging.LoggerAdapter(logger, {"job_id": JOB_NAME, "scene_id": os.getenv("SCENE_ID")})

    log.info("Starting import job...")

    # Get configuration from environment
    job_id = os.getenv("GENIE_SIM_JOB_ID")
    if not job_id:
        log.error("GENIE_SIM_JOB_ID is required")
        sys.exit(1)

    # Output configuration
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[GENIE-SIM-IMPORT]",
    )
    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]
    log = logging.LoggerAdapter(logger, {"job_id": JOB_NAME, "scene_id": scene_id})
    output_prefix = os.getenv("OUTPUT_PREFIX", f"scenes/{scene_id}/episodes")
    explicit_gcs_output_path = os.getenv("GCS_OUTPUT_PATH") or None
    job_metadata_path = os.getenv("JOB_METADATA_PATH") or None
    local_episodes_prefix = os.getenv("LOCAL_EPISODES_PREFIX") or None
    artifacts_by_robot_env = os.getenv("ARTIFACTS_BY_ROBOT") or None
    input_params.update(
        {
            "bucket": bucket,
            "scene_id": scene_id,
            "job_id": job_id,
            "output_prefix": output_prefix,
            "job_metadata_path": job_metadata_path,
            "local_episodes_prefix": local_episodes_prefix,
        }
    )
    log.debug("Input params: %s", input_params)

    job_metadata = None
    if job_metadata_path:
        try:
            job_metadata = _load_local_job_metadata(bucket, job_metadata_path)
            artifacts = job_metadata.get("artifacts", {})
            artifacts_by_robot = job_metadata.get("artifacts_by_robot")
            if not local_episodes_prefix and not isinstance(artifacts_by_robot, dict):
                local_episodes_prefix = artifacts.get("episodes_prefix")
        except FileNotFoundError as e:
            log.warning("%s", e)
            if not local_episodes_prefix:
                sys.exit(1)
    idempotency = _resolve_job_idempotency(job_metadata)

    # Quality configuration
    production_mode = resolve_production_mode()
    try:
        quality_settings = resolve_quality_settings()
    except ValueError as exc:
        log_pipeline_error(
            classify_exception(exc),
            "Failed to resolve quality settings",
            logger=log,
        )
        sys.exit(1)
    quality_settings = _apply_production_filter_override(
        quality_settings,
        production_mode=production_mode,
        log=log,
        env=os.environ,
    )
    min_quality_score = quality_settings.min_quality_score
    enable_validation = parse_bool_env(os.getenv("ENABLE_VALIDATION"), default=True)
    filter_low_quality = quality_settings.filter_low_quality
    service_mode = _is_service_mode()
    require_lerobot_resolution = _resolve_require_lerobot(
        os.getenv("REQUIRE_LEROBOT"),
        production_mode=production_mode,
        service_mode=service_mode,
    )
    require_lerobot = require_lerobot_resolution.value
    disable_gcs_upload = parse_bool_env(os.getenv("DISABLE_GCS_UPLOAD"), default=False)
    _guard_quality_thresholds(job_metadata, quality_settings, production_mode)
    disable_firebase_upload = parse_bool_env(
        os.getenv("DISABLE_FIREBASE_UPLOAD"),
        default=False,
    )
    enable_firebase_upload = parse_bool_env(
        os.getenv("ENABLE_FIREBASE_UPLOAD"),
        default=True,
    )
    if disable_firebase_upload:
        enable_firebase_upload = False
    firebase_upload_prefix = resolve_firebase_upload_prefix()
    try:
        lerobot_skip_rate_max = _resolve_skip_rate_max(
            os.getenv("LEROBOT_SKIP_RATE_MAX")
        )
    except ValueError as exc:
        log_pipeline_error(
            classify_exception(exc),
            "Failed to resolve LeRobot skip rate max",
            logger=log,
        )
        sys.exit(1)
    try:
        min_episodes_required = _resolve_min_episodes_required(
            os.getenv("MIN_EPISODES_REQUIRED")
        )
    except ValueError as exc:
        log_pipeline_error(
            classify_exception(exc),
            "Failed to resolve minimum episodes required",
            logger=log,
        )
        sys.exit(1)

    # Polling configuration
    poll_interval = int(os.getenv("GENIE_SIM_POLL_INTERVAL", "30"))
    wait_for_completion = parse_bool_env(os.getenv("WAIT_FOR_COMPLETION"), default=True)

    # Error handling configuration
    # P0 FIX: Default to True in production to fail fast on partial errors
    fail_on_partial_error = parse_bool_env(
        os.getenv("FAIL_ON_PARTIAL_ERROR"),
        default=production_mode,  # Default True in production, False otherwise
    )
    # P0 FIX: Partial uploads DISABLED by default in production mode
    # In production, atomic uploads are required - all robots succeed or all fail
    # This prevents delivering incomplete datasets to labs
    # Set ALLOW_PARTIAL_FIREBASE_UPLOADS=true to override (not recommended in production)
    allow_partial_firebase_uploads = parse_bool_env(
        os.getenv("ALLOW_PARTIAL_FIREBASE_UPLOADS"),
        default=not production_mode,  # Default False in production, True otherwise
    )
    allow_idempotent_retry = parse_bool_env(
        os.getenv("ALLOW_IDEMPOTENT_RETRY"),
        default=False,
    )
    if production_mode and allow_partial_firebase_uploads:
        log.warning(
            "ALLOW_PARTIAL_FIREBASE_UPLOADS=true in production mode. "
            "This may result in incomplete datasets being delivered. "
            "Consider setting ALLOW_PARTIAL_FIREBASE_UPLOADS=false for production safety."
        )
    try:
        firebase_upload_max_workers = _resolve_firebase_upload_max_workers(
            os.getenv("FIREBASE_UPLOAD_MAX_WORKERS")
        )
    except ValueError as exc:
        log_pipeline_error(
            classify_exception(exc),
            "Failed to resolve Firebase upload max workers",
            logger=log,
        )
        sys.exit(1)

    # Validate credentials at startup
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        from startup_validation import validate_and_fail_fast
        validate_and_fail_fast(
            job_name="GENIE-SIM-IMPORT",
            require_geniesim=False,
            require_gemini=False,
            validate_gcs=True,
            validate_firebase=enable_firebase_upload,
            require_firebase=enable_firebase_upload,
        )
    except ImportError as e:
        log.warning("Startup validation unavailable: %s", e)
    except SystemExit:
        # Re-raise to exit immediately
        raise

    firebase_upload_mode = get_firebase_upload_mode()
    firebase_preflight = None
    if enable_firebase_upload and firebase_upload_mode != "local":
        try:
            firebase_preflight = preflight_firebase_connectivity(timeout_seconds=30)
        except Exception as exc:
            log_pipeline_error(
                classify_exception(exc),
                "Firebase preflight failed",
                logger=log,
            )
            sys.exit(1)
    elif enable_firebase_upload:
        firebase_preflight = {
            "success": True,
            "mode": firebase_upload_mode,
            "note": "local mode preflight skipped",
        }
    else:
        firebase_preflight = {
            "success": False,
            "mode": firebase_upload_mode,
            "note": "firebase uploads disabled",
        }

    log.info("Configuration:")
    log.info("  Job ID: %s", job_id)
    log.info("  Output Prefix: %s", output_prefix)
    log.info("  Min Quality: %s", min_quality_score)
    if quality_settings.dimension_thresholds:
        log.info(
            "  Quality Component Thresholds: %s",
            quality_settings.dimension_thresholds,
        )
    log.info(
        "  Quality Range: %s - %s",
        QUALITY_CONFIG.min_allowed,
        QUALITY_CONFIG.max_allowed,
    )
    log.info("  Min Episodes Required: %s", min_episodes_required)
    log.info("  Enable Validation: %s", enable_validation)
    log.info(
        "  Require LeRobot: %s (default=%s, source=%s)",
        require_lerobot,
        require_lerobot_resolution.default,
        require_lerobot_resolution.source,
    )
    log.info("  LeRobot Skip Rate Max: %.2f%%", lerobot_skip_rate_max)
    log.info("  GCS Uploads Enabled: %s", not disable_gcs_upload)
    log.info(
        "  Firebase Uploads Enabled: %s (disable_env=%s)",
        enable_firebase_upload,
        disable_firebase_upload,
    )
    log.info("  Firebase Preflight: %s", firebase_preflight)
    log.info(
        "  Allow Partial Firebase Uploads: %s",
        allow_partial_firebase_uploads,
    )
    log.info("  Wait for Completion: %s", wait_for_completion)
    log.info("  Fail on Partial Error: %s", fail_on_partial_error)

    # Setup paths
    output_dir = _resolve_local_output_dir(
        bucket=bucket,
        output_prefix=output_prefix,
        job_id=job_id,
        local_episodes_prefix=local_episodes_prefix,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gcs_output_path = _resolve_gcs_output_path(
        output_dir,
        bucket=bucket,
        output_prefix=output_prefix,
        job_id=job_id,
        explicit_gcs_output_path=explicit_gcs_output_path,
    )

    # Create configuration
    config = _create_import_config(
        {
            "job_id": job_id,
            "output_dir": output_dir,
            "gcs_output_path": gcs_output_path,
            "enable_gcs_uploads": not disable_gcs_upload,
            "min_quality_score": min_quality_score,
            "quality_component_thresholds": quality_settings.dimension_thresholds,
            "enable_validation": enable_validation,
            "filter_low_quality": filter_low_quality,
            "require_lerobot": require_lerobot,
            "require_lerobot_default": require_lerobot_resolution.default,
            "require_lerobot_source": require_lerobot_resolution.source,
            "require_lerobot_raw_value": require_lerobot_resolution.raw_value,
            "lerobot_skip_rate_max": lerobot_skip_rate_max,
            "min_episodes_required": min_episodes_required,
            "poll_interval": poll_interval,
            "wait_for_completion": wait_for_completion,
            "fail_on_partial_error": fail_on_partial_error,
            "job_metadata_path": job_metadata_path,
            "local_episodes_prefix": local_episodes_prefix,
        }
    )

    artifacts_by_robot = _resolve_artifacts_by_robot(job_metadata, artifacts_by_robot_env)
    if artifacts_by_robot:
        validate_geniesim_robot_allowlist(
            artifacts_by_robot.keys(),
            strict=production_mode,
            logger=log.logger,
            context="GENIESIM import artifacts_by_robot",
        )
    else:
        job_robot_type = (job_metadata or {}).get("robot_type")
        if isinstance(job_robot_type, str) and job_robot_type.strip():
            validate_geniesim_robot_allowlist(
                [job_robot_type],
                strict=production_mode,
                logger=log.logger,
                context="GENIESIM import robot_type",
            )

    # Run import
    try:
        metrics = get_metrics()
        if artifacts_by_robot:
            robot_entries = []
            overall_success = True
            robot_results: List[Dict[str, Any]] = []
            quality_gate_failures: List[str] = []
            for robot_type, artifacts in sorted(artifacts_by_robot.items()):
                episodes_prefix = artifacts.get("episodes_prefix") or artifacts.get("episodes_path")
                if not episodes_prefix:
                    error_result = ImportResult(success=False, job_id=job_id)
                    error_result.errors.append(
                        f"Missing episodes_prefix for robot {robot_type}"
                    )
                    robot_entry = {
                        "robot_type": robot_type,
                        "success": False,
                        "output_dir": None,
                        "gcs_output_path": None,
                        "import_manifest_path": None,
                        "episodes": {
                            "downloaded": 0,
                            "passed_validation": 0,
                            "filtered": 0,
                            "download_errors": 0,
                            "parse_failed": 0,
                            "parse_failures": [],
                            "min_required": min_episodes_required,
                        },
                        "quality": {
                            "average_score": 0.0,
                            "min_score": 0.0,
                            "max_score": 0.0,
                            "threshold": min_quality_score,
                            "validation_enabled": enable_validation,
                            "component_thresholds": quality_settings.dimension_thresholds,
                            "component_failed_episodes": 0,
                            "component_failure_counts": {},
                        },
                        "errors": error_result.errors,
                        "warnings": [],
                        "upload_status": None,
                        "upload_failures": [],
                        "firebase_upload": None,
                    }
                    robot_entries.append(robot_entry)
                    robot_results.append(
                        {
                            "robot_type": robot_type,
                            "result": error_result,
                            "entry": robot_entry,
                        }
                    )
                    overall_success = False
                    continue

                robot_output_dir = _resolve_local_output_dir(
                    bucket=bucket,
                    output_prefix=output_prefix,
                    job_id=job_id,
                    local_episodes_prefix=episodes_prefix,
                )
                robot_output_dir.mkdir(parents=True, exist_ok=True)
                robot_gcs_output_path = _resolve_gcs_output_path(
                    robot_output_dir,
                    bucket=bucket,
                    output_prefix=output_prefix,
                    job_id=job_id,
                    explicit_gcs_output_path=explicit_gcs_output_path,
                )
                robot_config = _create_import_config(
                    {
                        "job_id": job_id,
                        "output_dir": robot_output_dir,
                        "gcs_output_path": robot_gcs_output_path,
                        "enable_gcs_uploads": not disable_gcs_upload,
                        "min_quality_score": min_quality_score,
                        "quality_component_thresholds": quality_settings.dimension_thresholds,
                        "enable_validation": enable_validation,
                        "filter_low_quality": filter_low_quality,
                        "require_lerobot": require_lerobot,
                        "require_lerobot_default": require_lerobot_resolution.default,
                        "require_lerobot_source": require_lerobot_resolution.source,
                        "require_lerobot_raw_value": require_lerobot_resolution.raw_value,
                        "lerobot_skip_rate_max": lerobot_skip_rate_max,
                        "min_episodes_required": min_episodes_required,
                        "poll_interval": poll_interval,
                        "wait_for_completion": wait_for_completion,
                        "fail_on_partial_error": fail_on_partial_error,
                        "job_metadata_path": job_metadata_path,
                        "local_episodes_prefix": episodes_prefix,
                    }
                )
                robot_job_metadata = _build_robot_job_metadata(
                    job_metadata,
                    robot_type,
                    artifacts,
                )
                with metrics.track_job("genie-sim-import-job", scene_id):
                    result = run_local_import_job(
                        robot_config,
                        job_metadata=robot_job_metadata,
                    )
                _alert_low_quality(
                    scene_id=scene_id,
                    job_id=job_id,
                    robot_type=robot_type,
                    average_quality_score=result.average_quality_score,
                    min_quality_score=min_quality_score,
                    episodes_passed_validation=result.episodes_passed_validation,
                    episodes_filtered=result.episodes_filtered,
                )
                try:
                    gate_results = _emit_import_quality_gate(
                        result,
                        scene_id,
                        job_metadata=robot_job_metadata,
                    )
                    gate_failed = _quality_gate_failure_detected(gate_results)
                except Exception as exc:
                    log.warning("Quality gate emission failed: %s", exc)
                    gate_failed = False

                if gate_failed:
                    quality_gate_failures.append(robot_type)
                    result.success = False
                    result.errors.append("Quality gate failure")
                    log.error(
                        "Quality gate failure detected for robot %s. "
                        "Production mode enforces a non-zero exit.",
                        robot_type,
                    )

                overall_success = overall_success and result.success
                robot_entry = {
                    "robot_type": robot_type,
                    "success": result.success,
                    "output_dir": str(result.output_dir) if result.output_dir else None,
                    "gcs_output_path": robot_gcs_output_path,
                    "import_manifest_path": _resolve_gcs_path(result.import_manifest_path),
                    "import_manifest_local_path": (
                        str(result.import_manifest_path) if result.import_manifest_path else None
                    ),
                    "episodes": {
                        "downloaded": result.total_episodes_downloaded,
                        "passed_validation": result.episodes_passed_validation,
                        "filtered": result.episodes_filtered,
                        "download_errors": 0,
                        "parse_failed": result.episodes_parse_failed,
                        "parse_failures": result.episode_parse_failures,
                        "min_required": min_episodes_required,
                    },
                    "quality": {
                        "average_score": result.average_quality_score,
                        "min_score": result.quality_min_score,
                        "max_score": result.quality_max_score,
                        "threshold": min_quality_score,
                        "validation_enabled": enable_validation,
                        "component_thresholds": result.quality_component_thresholds,
                        "component_failed_episodes": result.quality_component_failed_count,
                        "component_failure_counts": result.quality_component_failure_counts,
                    },
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "upload_status": result.upload_status,
                    "upload_failures": result.upload_failures,
                    "firebase_upload": None,
                }
                robot_entries.append(robot_entry)
                robot_results.append(
                    {
                        "robot_type": robot_type,
                        "result": result,
                        "entry": robot_entry,
                    }
                )

            failed_robots = [
                payload["robot_type"]
                for payload in robot_results
                if not payload["result"].success
            ]
            if job_metadata is not None:
                component_failure_counts: Dict[str, int] = {}
                component_failed_total = 0
                per_robot_components: Dict[str, Any] = {}
                for entry in robot_entries:
                    quality = entry.get("quality", {})
                    robot_type = entry.get("robot_type", "unknown")
                    failed_count = int(quality.get("component_failed_episodes", 0))
                    component_failed_total += failed_count
                    counts = quality.get("component_failure_counts", {})
                    if isinstance(counts, dict):
                        per_robot_components[robot_type] = {
                            "failed_episodes": failed_count,
                            "failure_counts": counts,
                        }
                        for key, value in counts.items():
                            if isinstance(value, (int, float)):
                                component_failure_counts[key] = (
                                    component_failure_counts.get(key, 0) + int(value)
                                )
                job_summary = job_metadata.setdefault("job_summary", {})
                job_summary["quality_components"] = {
                    "thresholds": quality_settings.config.dimension_thresholds,
                    "failed_episodes": component_failed_total,
                    "failure_counts": component_failure_counts,
                    "robots": per_robot_components,
                }
                _write_local_job_metadata(
                    bucket=bucket,
                    job_metadata_path=job_metadata_path,
                    job_metadata=job_metadata,
                )
            partial_failure = bool(failed_robots)
            delivery_marker_result: Optional[DeliveryMarkerResult] = None
            (
                firebase_upload_suppressed,
                suppression_reason,
                log_partial_uploads,
            ) = _resolve_firebase_upload_suppression(
                partial_failure=partial_failure,
                allow_partial_firebase_uploads=allow_partial_firebase_uploads,
                fail_on_partial_error=fail_on_partial_error,
                quality_gate_failures=quality_gate_failures,
            )
            if log_partial_uploads:
                # Respect allow_partial_firebase_uploads even in production mode:
                # - If allow_partial_firebase_uploads=True (new default), upload successful robots
                # - If allow_partial_firebase_uploads=False, suppress all uploads on partial failure
                # - fail_on_partial_error controls job exit code, not upload behavior
                log.warning(
                    "Partial failure detected (%d robot(s) failed), but uploading "
                    "successful robots since ALLOW_PARTIAL_FIREBASE_UPLOADS=true. "
                    "Failed robots: %s",
                    len(failed_robots),
                    list(failed_robots),
                )

            if enable_firebase_upload and not firebase_upload_suppressed:
                manifest_paths = [
                    payload["result"].import_manifest_path
                    for payload in robot_results
                    if payload.get("result") and payload["result"].import_manifest_path
                ]
                delivery_marker_result = _write_delivery_marker(
                    bucket=bucket,
                    scene_id=scene_id,
                    job_id=job_id,
                    run_id=_resolve_run_id(job_id),
                    idempotency_key=idempotency.get("key") if idempotency else None,
                    allow_idempotent_retry=allow_idempotent_retry,
                    log=log,
                    import_manifest_paths=manifest_paths,
                )
                overall_success = _run_firebase_uploads_for_robot_payloads(
                    robot_results=robot_results,
                    scene_id=scene_id,
                    job_id=job_id,
                    firebase_upload_prefix=firebase_upload_prefix,
                    firebase_upload_max_workers=firebase_upload_max_workers,
                    allow_partial_firebase_uploads=allow_partial_firebase_uploads,
                    fail_on_partial_error=fail_on_partial_error,
                    log=log,
                    overall_success=overall_success,
                )
            elif enable_firebase_upload and firebase_upload_suppressed:
                log.warning(
                    "Firebase uploads suppressed (reason=%s).",
                    suppression_reason,
                )

            if partial_failure:
                robots_not_uploaded = []
                if firebase_upload_suppressed:
                    robots_not_uploaded = [payload["robot_type"] for payload in robot_results]
                else:
                    robots_not_uploaded = [
                        payload["robot_type"]
                        for payload in robot_results
                        if not payload["result"].success
                    ]
                if job_metadata is not None:
                    job_summary = job_metadata.setdefault("job_summary", {})
                    job_summary["partial_failures"] = {
                        "failed_robots": failed_robots,
                        "robots_not_uploaded": robots_not_uploaded,
                        "firebase_upload_suppressed": firebase_upload_suppressed,
                        "suppression_reason": suppression_reason,
                        "allow_partial_firebase_uploads": allow_partial_firebase_uploads,
                        "production_mode": production_mode,
                        "fail_on_partial_error": fail_on_partial_error,
                    }
                    _write_local_job_metadata(
                        bucket=bucket,
                        job_metadata_path=job_metadata_path,
                        job_metadata=job_metadata,
                    )
            if job_metadata is not None and quality_gate_failures:
                job_summary = job_metadata.setdefault("job_summary", {})
                job_summary["quality_gate"] = {
                    "failed_robots": quality_gate_failures,
                    "firebase_upload_suppressed": firebase_upload_suppressed,
                    "suppression_reason": suppression_reason,
                    "exit_nonzero_in_production": production_mode,
                }
                _write_local_job_metadata(
                    bucket=bucket,
                    job_metadata_path=job_metadata_path,
                    job_metadata=job_metadata,
                )

            combined_output_dir = _resolve_local_output_dir(
                bucket=bucket,
                output_prefix=output_prefix,
                job_id=job_id,
                local_episodes_prefix=None,
            )
            combined_gcs_output_path = _resolve_gcs_output_path(
                combined_output_dir,
                bucket=bucket,
                output_prefix=output_prefix,
                job_id=job_id,
                explicit_gcs_output_path=explicit_gcs_output_path,
            )
            combined_manifest_path = _write_combined_import_manifest(
                combined_output_dir,
                job_id,
                combined_gcs_output_path,
                job_metadata,
                robot_entries,
                quality_settings,
                logger=logger,
            )
            if delivery_marker_result is not None:
                _update_import_manifest_delivery(
                    combined_manifest_path,
                    delivery_marker_result.payload,
                    gcs_uri=delivery_marker_result.gcs_uri,
                    local_path=delivery_marker_result.local_path,
                    marker_exists=delivery_marker_result.marker_exists,
                )
            log.info("Combined import manifest: %s", combined_manifest_path)

            if overall_success:
                log.info("Multi-robot import succeeded")
                sys.exit(0)
            log.error("Multi-robot import failed")
            for entry in robot_entries:
                if entry["errors"]:
                    log.error(
                        "  - %s: %s",
                        entry["robot_type"],
                        ", ".join(entry["errors"]),
                    )
            sys.exit(1)

        result = None
        with metrics.track_job("genie-sim-import-job", scene_id):
            result = run_local_import_job(config, job_metadata=job_metadata)
        if job_metadata is not None:
            job_summary = job_metadata.setdefault("job_summary", {})
            job_summary["quality_components"] = {
                "thresholds": result.quality_component_thresholds,
                "failed_episodes": result.quality_component_failed_count,
                "failure_counts": result.quality_component_failure_counts,
            }
            _write_local_job_metadata(
                bucket=bucket,
                job_metadata_path=job_metadata_path,
                job_metadata=job_metadata,
            )
        _alert_low_quality(
            scene_id=scene_id,
            job_id=job_id,
            robot_type="default",
            average_quality_score=result.average_quality_score,
            min_quality_score=min_quality_score,
            episodes_passed_validation=result.episodes_passed_validation,
            episodes_filtered=result.episodes_filtered,
        )
        try:
            gate_results = _emit_import_quality_gate(
                result,
                scene_id,
                job_metadata=job_metadata,
            )
            gate_failed = _quality_gate_failure_detected(gate_results)
        except Exception as exc:
            log.warning("Quality gate emission failed: %s", exc)
            gate_failed = False

        if gate_failed:
            result.success = False
            result.errors.append("Quality gate failure")
            log.error(
                "Quality gate failure detected. Production mode enforces a non-zero exit."
            )
            if enable_firebase_upload:
                log.warning(
                    "Firebase uploads suppressed due to quality gate failure "
                    "(reason=quality_gate_failure)."
                )
            if job_metadata is not None:
                job_summary = job_metadata.setdefault("job_summary", {})
                job_summary["quality_gate"] = {
                    "failed_robots": ["default"],
                    "firebase_upload_suppressed": True,
                    "suppression_reason": "quality_gate_failure",
                    "exit_nonzero_in_production": production_mode,
                }
                _write_local_job_metadata(
                    bucket=bucket,
                    job_metadata_path=job_metadata_path,
                    job_metadata=job_metadata,
                )

        if result.success:
            log.info("Import succeeded")
            log.info("Episodes imported: %s", result.episodes_passed_validation)
            log.info("Average quality: %.2f", result.average_quality_score)
            if enable_firebase_upload:
                log.info("Uploading episodes to Firebase Storage...")
                _write_delivery_marker(
                    bucket=bucket,
                    scene_id=scene_id,
                    job_id=job_id,
                    run_id=_resolve_run_id(job_id),
                    idempotency_key=idempotency.get("key") if idempotency else None,
                    allow_idempotent_retry=allow_idempotent_retry,
                    log=log,
                    import_manifest_paths=[
                        result.import_manifest_path
                    ]
                    if result.import_manifest_path
                    else [],
                )
                dedup_summary = _prepare_deduplication_summary(
                    result=result,
                    scene_id=scene_id,
                    robot_type="default",
                    prefix=firebase_upload_prefix,
                    log=log,
                )
                dedup_summary = _apply_deduplication_filters(
                    result,
                    dedup_summary,
                    log,
                )
                deduplicated_ids = dedup_summary.get("deduplicated_episode_ids", [])
                _update_import_manifest_dedup_summary(
                    result.import_manifest_path,
                    dedup_summary,
                )
                if deduplicated_ids:
                    log.info(
                        "Deduplicating %s episode(s) via content hash index.",
                        len(deduplicated_ids),
                    )
                upload_file_paths = _resolve_upload_file_list(
                    result.output_dir,
                    deduplicated_ids,
                )
                expected_paths = [
                    path.relative_to(result.output_dir).as_posix()
                    for path in upload_file_paths
                ]
                local_path_map = {
                    rel_path: path.resolve()
                    for rel_path, path in zip(expected_paths, upload_file_paths)
                }
                verify_checksums = _resolve_firebase_verify_checksums(
                    os.getenv("FIREBASE_VERIFY_CHECKSUMS"),
                )
                robot_type = None
                try:
                    firebase_result = upload_episodes_with_retry(
                        episodes_dir=result.output_dir,
                        scene_id=scene_id,
                        prefix=firebase_upload_prefix,
                        file_paths=upload_file_paths,
                    )
                except FirebaseUploadOrchestratorError as exc:
                    _alert_firebase_upload_failure(
                        scene_id=scene_id,
                        job_id=job_id,
                        robot_type="default",
                        error=str(exc),
                    )
                    log.error("Firebase upload failed: %s", exc)
                    raise
                upload_summary = firebase_result.summary
                try:
                    firebase_verification = verify_firebase_upload(
                        scene_id=scene_id,
                        robot_type=robot_type,
                        prefix=firebase_upload_prefix,
                        expected_paths=expected_paths,
                        verify_checksums=verify_checksums,
                        local_path_map=local_path_map,
                    )
                except Exception as exc:
                    firebase_verification = {
                        "success": False,
                        "verified": [],
                        "missing": [],
                        "extra": [],
                        "checksum_mismatches": [],
                        "errors": [str(exc)],
                    }
                upload_summary["firebase_verification"] = firebase_verification
                upload_summary["deduplicated_episodes"] = len(deduplicated_ids)
                upload_summary["deduplicated_episode_ids"] = deduplicated_ids
                upload_summary["remote_prefix"] = firebase_result.remote_prefix
                _update_import_manifest_firebase_summary(
                    result.import_manifest_path,
                    upload_summary,
                )
                log.info(
                    "Firebase upload complete: uploaded=%s skipped=%s reuploaded=%s failed=%s total=%s",
                    upload_summary.get("uploaded", 0),
                    upload_summary.get("skipped", 0),
                    upload_summary.get("reuploaded", 0),
                    upload_summary.get("failed", 0),
                    upload_summary.get("total_files", 0),
                )
                failed_count = upload_summary.get("failed", 0)
                if failed_count > 0:
                    _alert_firebase_upload_failure(
                        scene_id=scene_id,
                        job_id=job_id,
                        robot_type="default",
                        error=f"{failed_count} file(s) failed to upload to Firebase",
                    )
                    log.error(
                        "Firebase upload incomplete: %d file(s) failed. "
                        "Completion marker will NOT be written.",
                        failed_count,
                    )
                    sys.exit(1)
                if firebase_verification and not firebase_verification.get("success", True):
                    verification_errors = _format_firebase_verification_errors(
                        firebase_verification
                    )
                    error_message = "Firebase upload verification failed"
                    if verification_errors:
                        error_message += f": {'; '.join(verification_errors)}"
                    _alert_firebase_upload_failure(
                        scene_id=scene_id,
                        job_id=job_id,
                        robot_type="default",
                        error=error_message,
                    )
                    if _should_fail_firebase_verification(
                        allow_partial_firebase_uploads,
                        fail_on_partial_error,
                    ):
                        log.error(
                            "%s Completion marker will NOT be written.",
                            error_message,
                        )
                        sys.exit(1)
                    log.warning(
                        "%s Continuing because ALLOW_PARTIAL_FIREBASE_UPLOADS=true "
                        "and FAIL_ON_PARTIAL_ERROR=false.",
                        error_message,
                    )
                elif deduplicated_ids:
                    new_hashes = {
                        episode_id: content_hash
                        for episode_id, content_hash in result.episode_content_hashes.items()
                        if episode_id not in deduplicated_ids
                    }
                    _persist_episode_hash_index(
                        episode_hashes=new_hashes,
                        scene_id=scene_id,
                        robot_type="default",
                        prefix=firebase_upload_prefix,
                        job_id=job_id,
                        log=log,
                    )
                else:
                    _persist_episode_hash_index(
                        episode_hashes=result.episode_content_hashes,
                        scene_id=scene_id,
                        robot_type="default",
                        prefix=firebase_upload_prefix,
                        job_id=job_id,
                        log=log,
                    )
                _publish_dataset_catalog_document(
                    scene_id=scene_id,
                    job_id=job_id,
                    robot_type="default",
                    result=result,
                    firebase_summary=upload_summary,
                    gcs_output_path=gcs_output_path,
                    log=log,
                )

                # Upload Cosmos Policy dataset to Firebase (separate prefix)
                cosmos_policy_output = result.output_dir / "cosmos_policy"
                if cosmos_policy_output.exists():
                    try:
                        cosmos_upload_prefix = os.getenv(
                            "COSMOS_POLICY_FIREBASE_PREFIX",
                            "datasets/cosmos_policy",
                        )
                        cosmos_files = list(cosmos_policy_output.rglob("*"))
                        cosmos_files = [f for f in cosmos_files if f.is_file()]
                        if cosmos_files:
                            cosmos_firebase_result = upload_episodes_with_retry(
                                episodes_dir=cosmos_policy_output,
                                scene_id=scene_id,
                                prefix=cosmos_upload_prefix,
                                file_paths=cosmos_files,
                            )
                            log.info(
                                "Cosmos Policy Firebase upload complete: uploaded=%s failed=%s",
                                cosmos_firebase_result.summary.get("uploaded", 0),
                                cosmos_firebase_result.summary.get("failed", 0),
                            )
                    except Exception as cosmos_upload_exc:
                        log.warning(
                            "Cosmos Policy Firebase upload failed (non-fatal): %s",
                            cosmos_upload_exc,
                        )

            sys.exit(0)
        else:
            log.error("Import failed")
            for error in result.errors:
                log.error("  - %s", error)
            sys.exit(1)
    except Exception as exc:
        log_pipeline_error(
            classify_exception(exc),
            "Import failed with exception",
            logger=log,
        )
        log.exception("Import failed with exception: %s", exc)
        send_alert(
            event_type="geniesim_import_job_fatal_exception",
            summary="Genie Sim import job failed with an unhandled exception",
            details={
                "job": "genie-sim-import-job",
                "job_id": job_id,
                "scene_id": scene_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            severity=os.getenv("ALERT_JOB_EXCEPTION_SEVERITY", "critical"),
        )
        sys.exit(1)



if __name__ == "__main__":
    try:
        init_tracing(service_name=os.getenv("OTEL_SERVICE_NAME", JOB_NAME))
        input_params: Dict[str, Any] = {}

        def _run_import_job() -> Optional[int]:
            return main(input_params)

        run_job_with_dead_letter_queue(
            _run_import_job,
            scene_id=os.getenv("SCENE_ID"),
            job_type=JOB_NAME,
            step="import",
            input_params=input_params,
        )
    except Exception as exc:
        send_alert(
            event_type="geniesim_import_job_fatal_exception",
            summary="Genie Sim import job failed with an unhandled exception",
            details={
                "job": "genie-sim-import-job",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            severity=os.getenv("ALERT_JOB_EXCEPTION_SEVERITY", "critical"),
        )
        raise

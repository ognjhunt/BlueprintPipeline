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
    ALLOW_PARTIAL_FIREBASE_UPLOADS: Allow Firebase uploads to proceed for
        successful robots even if others fail (default: false)
    ARTIFACTS_BY_ROBOT: JSON map of robot type to artifacts payload for multi-robot imports
    ALLOW_IDEMPOTENT_RETRY: Allow retrying a local import when a prior manifest
        indicates a failed or partial run (default: false)
    ENABLE_REALTIME_STREAMING: Enable real-time feedback streaming (default: false in production)
    REALTIME_STREAM_PROTOCOL: Streaming protocol (http_post, grpc, websocket, message_queue, file_watch)
    REALTIME_STREAM_ENDPOINT: Endpoint URL or path for streaming
    REALTIME_STREAM_API_KEY: API key for streaming authentication
    REALTIME_STREAM_BATCH_SIZE: Batch size for streaming episodes (default: 10)
"""

import asyncio
import copy
import hashlib
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.alerting import send_alert

from tools.dataset_regression.metrics import compute_regression_metrics
from tools.error_handling.retry import NonRetryableError, RetryConfig, RetryContext
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
from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode
from tools.lerobot_format import LeRobotExportFormat, parse_lerobot_export_format
from tools.logging_config import init_logging
from tools.firebase_upload.firebase_upload_orchestrator import (
    FirebaseUploadOrchestratorError,
    build_firebase_upload_prefix,
    resolve_firebase_upload_prefix,
    upload_episodes_with_retry,
)
from tools.firebase_upload.uploader import (
    get_firebase_storage_bucket,
    get_firebase_upload_mode,
    resolve_firebase_local_upload_root,
)
from tools.error_handling.job_wrapper import run_job_with_dead_letter_queue
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

DATASET_INFO_SCHEMA_VERSION = "1.0.0"
MIN_EPISODES_REQUIRED = 1
JOB_NAME = "genie-sim-import-job"
logger = logging.getLogger(__name__)


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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_episode_content_manifest(
    recordings_dir: Path,
    episode_id: str,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(recordings_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.stem != episode_id:
            continue
        try:
            rel_path = path.relative_to(recordings_dir).as_posix()
        except ValueError:
            rel_path = path.as_posix()
        entries.append(
            {
                "path": rel_path,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    entries.sort(key=lambda entry: entry["path"])
    return entries


def _compute_episode_content_hash(entries: List[Dict[str, Any]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_episode_hash_index_path(
    *,
    scene_id: str,
    robot_type: Optional[str],
    prefix: str,
    content_hash: str,
) -> str:
    base_prefix = build_firebase_upload_prefix(
        scene_id,
        robot_type=robot_type,
        prefix=prefix,
    )
    return f"{base_prefix}/episode_hash_index/{content_hash}.json"


def _lookup_episode_hash_index(
    *,
    episode_hashes: Dict[str, str],
    scene_id: str,
    robot_type: Optional[str],
    prefix: str,
    log: logging.LoggerAdapter,
) -> Dict[str, str]:
    if not episode_hashes:
        return {}
    upload_mode = get_firebase_upload_mode()
    existing: Dict[str, str] = {}
    try:
        if upload_mode == "local":
            local_root = resolve_firebase_local_upload_root()
            for episode_id, content_hash in episode_hashes.items():
                index_path = _build_episode_hash_index_path(
                    scene_id=scene_id,
                    robot_type=robot_type,
                    prefix=prefix,
                    content_hash=content_hash,
                )
                if (local_root / index_path).exists():
                    existing[episode_id] = content_hash
            return existing

        bucket = get_firebase_storage_bucket()
        for episode_id, content_hash in episode_hashes.items():
            blob_path = _build_episode_hash_index_path(
                scene_id=scene_id,
                robot_type=robot_type,
                prefix=prefix,
                content_hash=content_hash,
            )
            blob = bucket.blob(blob_path)
            if blob.exists():
                existing[episode_id] = content_hash
    except Exception as exc:
        log.warning("Episode hash index lookup failed: %s", exc)
        return {}
    return existing


def _persist_episode_hash_index(
    *,
    episode_hashes: Dict[str, str],
    scene_id: str,
    robot_type: Optional[str],
    prefix: str,
    job_id: str,
    log: logging.LoggerAdapter,
) -> None:
    if not episode_hashes:
        return
    upload_mode = get_firebase_upload_mode()
    payload_template = {
        "scene_id": scene_id,
        "robot_type": robot_type,
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    if upload_mode == "local":
        local_root = resolve_firebase_local_upload_root()
        for episode_id, content_hash in episode_hashes.items():
            index_path = _build_episode_hash_index_path(
                scene_id=scene_id,
                robot_type=robot_type,
                prefix=prefix,
                content_hash=content_hash,
            )
            full_path = local_root / index_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                **payload_template,
                "episode_id": episode_id,
                "content_hash": content_hash,
            }
            write_json_atomic(full_path, payload, indent=2)
        return

    try:
        bucket = get_firebase_storage_bucket()
    except Exception as exc:
        log.warning("Episode hash index write skipped: %s", exc)
        return
    for episode_id, content_hash in episode_hashes.items():
        blob_path = _build_episode_hash_index_path(
            scene_id=scene_id,
            robot_type=robot_type,
            prefix=prefix,
            content_hash=content_hash,
        )
        blob = bucket.blob(blob_path)
        if blob.exists():
            continue
        payload = {
            **payload_template,
            "episode_id": episode_id,
            "content_hash": content_hash,
        }
        blob.upload_from_string(
            json.dumps(payload, sort_keys=True),
            content_type="application/json",
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
        try:
            dataset_info = _load_json_file(dataset_info_path)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(
                "Failed to load dataset info from %s: %s",
                dataset_info_path,
                exc,
            )
            dataset_info = {}
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


def _relative_to_bundle(bundle_root: Path, path: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(bundle_root.resolve())
    except ValueError:
        return path.as_posix()
    rel_str = rel_path.as_posix()
    return rel_str if rel_str else "."


def _resolve_asset_provenance_reference(
    bundle_root: Path,
    output_dir: Path,
    job_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    local_path = output_dir / "legal" / "asset_provenance.json"
    if local_path.exists():
        return _relative_to_bundle(bundle_root, local_path)
    if job_metadata:
        artifacts = job_metadata.get("artifacts", {})
        bundle = job_metadata.get("bundle", {})
        reference = (
            artifacts.get("asset_provenance")
            or artifacts.get("asset_provenance_path")
            or bundle.get("asset_provenance")
            or bundle.get("asset_provenance_path")
        )
        if reference:
            return reference
    return None


def _resolve_run_id(job_id: str) -> str:
    return (
        os.getenv("RUN_ID")
        or os.getenv("GENIE_SIM_RUN_ID")
        or os.getenv("GENIESIM_RUN_ID")
        or job_id
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
        allow_fallback=require_parquet_validation,
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


def _write_checksums_file(output_dir: Path, checksums: Dict[str, Any]) -> Path:
    checksums_path = output_dir / "checksums.json"
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "root": ".",
        "algorithm": "sha256",
        "files": checksums,
    }
    with open(checksums_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return checksums_path


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
    if version and str(version).strip() == "3.0":
        return LeRobotExportFormat.LEROBOT_V3
    return LeRobotExportFormat.LEROBOT_V2


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

    episodes_index_path = lerobot_root / "episodes.jsonl"
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
                    pass  # pyarrow not available, skip parquet validation
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


def _load_json_file(path: Path) -> Any:
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to load JSON file {path}: {exc}") from exc


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
        import_manifest = _load_json_file(result.import_manifest_path)
    except ValueError as exc:
        log.warning("Skipping dataset catalog publish: %s", exc)
        return

    if result.import_manifest_path:
        import_manifest["import_manifest_path"] = _resolve_gcs_path(result.import_manifest_path)

    dataset_info_payload: Optional[Dict[str, Any]] = None
    if result.output_dir:
        dataset_info_path = result.output_dir / "lerobot" / "dataset_info.json"
        if dataset_info_path.exists():
            try:
                dataset_info_payload = _load_json_file(dataset_info_path)
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


def _load_existing_import_manifest(output_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = output_dir / "import_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r") as handle:
        return json.load(handle)


def _resolve_manifest_import_status(import_manifest: Dict[str, Any]) -> str:
    status = import_manifest.get("import_status")
    if isinstance(status, str):
        return status.strip().lower()
    success = import_manifest.get("success")
    if isinstance(success, bool):
        return "success" if success else "failed"
    checksums_success = (
        import_manifest.get("verification", {}).get("checksums", {}).get("success")
    )
    if checksums_success is True:
        return "success"
    if checksums_success is False:
        return "failed"
    return "unknown"


def _resolve_import_status(result: "ImportResult") -> str:
    if result.success:
        return "success"
    if result.episodes_passed_validation or result.episodes_filtered:
        return "partial"
    return "failed"


def _update_import_manifest_status(
    manifest_path: Optional[Path],
    status: str,
    success: Optional[bool] = None,
) -> None:
    if manifest_path is None or not manifest_path.exists():
        return
    with open(manifest_path, "r") as handle:
        import_manifest = json.load(handle)
    import_manifest["import_status"] = status
    if success is not None:
        import_manifest["success"] = success
    import_manifest["status_updated_at"] = datetime.utcnow().isoformat() + "Z"
    checksums = import_manifest.setdefault("checksums", {})
    metadata_checksums = checksums.setdefault("metadata", {})
    metadata_checksums.setdefault("import_manifest.json", {})
    metadata_checksums["import_manifest.json"]["sha256"] = compute_manifest_checksum(
        import_manifest
    )
    write_json_atomic(manifest_path, import_manifest, indent=2)


def _update_import_manifest_firebase_summary(
    manifest_path: Optional[Path],
    firebase_summary: Dict[str, Any],
) -> None:
    if manifest_path is None:
        return
    if not manifest_path.exists():
        print(
            "[GENIE-SIM-IMPORT] ⚠️  Import manifest not found; "
            "skipping Firebase summary update."
        )
        return
    with open(manifest_path, "r") as handle:
        import_manifest = json.load(handle)
    import_manifest["firebase_upload"] = firebase_summary
    checksums = import_manifest.setdefault("checksums", {})
    metadata_checksums = checksums.setdefault("metadata", {})
    metadata_checksums.setdefault("import_manifest.json", {})
    metadata_checksums["import_manifest.json"]["sha256"] = compute_manifest_checksum(
        import_manifest
    )
    write_json_atomic(manifest_path, import_manifest, indent=2)


def _update_import_manifest_dedup_summary(
    manifest_path: Optional[Path],
    dedup_summary: Dict[str, Any],
) -> None:
    if manifest_path is None:
        return
    if not manifest_path.exists():
        print(
            "[GENIE-SIM-IMPORT] ⚠️  Import manifest not found; "
            "skipping dedup summary update."
        )
        return
    with open(manifest_path, "r") as handle:
        import_manifest = json.load(handle)
    import_manifest["deduplication"] = dedup_summary
    episodes_summary = import_manifest.setdefault("episodes", {})
    episodes_summary["deduplicated"] = dedup_summary.get("deduplicated_count", 0)
    episodes_summary["deduplicated_ids"] = dedup_summary.get(
        "deduplicated_episode_ids", []
    )
    checksums = import_manifest.setdefault("checksums", {})
    metadata_checksums = checksums.setdefault("metadata", {})
    metadata_checksums.setdefault("import_manifest.json", {})
    metadata_checksums["import_manifest.json"]["sha256"] = compute_manifest_checksum(
        import_manifest
    )
    write_json_atomic(manifest_path, import_manifest, indent=2)


def _resolve_job_idempotency(job_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job_metadata:
        return None
    idempotency = job_metadata.get("idempotency")
    return idempotency if isinstance(idempotency, dict) else None


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


def _normalize_gcs_output_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    if path_value.startswith("gs://"):
        return path_value
    if path_value.startswith("/mnt/gcs/"):
        return "gs://" + path_value[len("/mnt/gcs/"):]
    return path_value


def _aggregate_metrics_summary(robot_metrics: Dict[str, Any]) -> Dict[str, Any]:
    aggregated_stats: Dict[str, Any] = {}
    backends = set()
    for metrics in robot_metrics.values():
        if not isinstance(metrics, dict):
            continue
        backend = metrics.get("backend")
        if backend:
            backends.add(backend)
        stats = metrics.get("stats")
        if isinstance(stats, dict):
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    aggregated_stats[key] = aggregated_stats.get(key, 0) + value
    if not backends:
        backend = "unknown"
    elif len(backends) == 1:
        backend = next(iter(backends))
    else:
        backend = "mixed"
    return {
        "backend": backend,
        "stats": aggregated_stats,
        "robots": robot_metrics,
    }


def _write_combined_import_manifest(
    output_dir: Path,
    job_id: str,
    gcs_output_path: Optional[str],
    job_metadata: Optional[Dict[str, Any]],
    robot_entries: List[Dict[str, Any]],
    quality_settings: ResolvedQualitySettings,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_gcs_output_path = _normalize_gcs_output_path(gcs_output_path)
    total_downloaded = sum(entry["episodes"]["downloaded"] for entry in robot_entries)
    total_passed = sum(entry["episodes"]["passed_validation"] for entry in robot_entries)
    total_filtered = sum(entry["episodes"]["filtered"] for entry in robot_entries)
    total_parse_failed = sum(entry["episodes"]["parse_failed"] for entry in robot_entries)
    idempotency = _resolve_job_idempotency(job_metadata)

    weighted_quality_sum = 0.0
    quality_weight_total = 0
    min_quality_scores = []
    max_quality_scores = []
    for entry in robot_entries:
        quality = entry["quality"]
        episodes_downloaded = entry["episodes"]["downloaded"]
        weighted_quality_sum += quality["average_score"] * episodes_downloaded
        quality_weight_total += episodes_downloaded
        if episodes_downloaded > 0:
            min_quality_scores.append(quality["min_score"])
            max_quality_scores.append(quality["max_score"])

    average_quality = (
        weighted_quality_sum / quality_weight_total if quality_weight_total else 0.0
    )
    min_quality = min(min_quality_scores) if min_quality_scores else 0.0
    max_quality = max(max_quality_scores) if max_quality_scores else 0.0
    component_failure_counts: Dict[str, int] = {}
    component_failed_total = 0
    for entry in robot_entries:
        quality = entry.get("quality", {})
        component_failed_total += int(quality.get("component_failed_episodes", 0))
        counts = quality.get("component_failure_counts")
        if isinstance(counts, dict):
            for key, value in counts.items():
                if isinstance(value, (int, float)):
                    component_failure_counts[key] = component_failure_counts.get(key, 0) + int(
                        value
                    )

    normalized_robot_entries = []
    robot_metrics: Dict[str, Any] = {}
    robot_provenance: Dict[str, Any] = {}
    lerobot_metadata_paths: Dict[str, List[str]] = {}
    for entry in robot_entries:
        entry_copy = copy.deepcopy(entry)
        entry_copy["gcs_output_path"] = _normalize_gcs_output_path(
            entry_copy.get("gcs_output_path")
        )
        entry_copy["import_manifest_path"] = _normalize_gcs_output_path(
            entry_copy.get("import_manifest_path")
        )
        normalized_robot_entries.append(entry_copy)
        robot_type = entry_copy.get("robot_type", "unknown")
        output_dir_str = entry_copy.get("output_dir")
        if output_dir_str:
            robot_output_dir = Path(output_dir_str)
            metadata_paths = get_lerobot_metadata_paths(robot_output_dir)
            lerobot_metadata_paths[robot_type] = [
                path.relative_to(robot_output_dir).as_posix()
                for path in metadata_paths
            ]
        manifest_path_str = entry.get("import_manifest_local_path")
        if manifest_path_str:
            manifest_path = Path(manifest_path_str)
            if manifest_path.exists():
                try:
                    manifest_payload = json.loads(manifest_path.read_text())
                except json.JSONDecodeError as exc:
                    print(
                        "[GENIE-SIM-IMPORT] ⚠️  Failed to load robot manifest "
                        f"{manifest_path}: {exc}"
                    )
                else:
                    robot_metrics_payload = manifest_payload.get("metrics_summary")
                    if robot_metrics_payload:
                        robot_metrics[robot_type] = robot_metrics_payload
                    robot_provenance_payload = manifest_payload.get("provenance")
                    if robot_provenance_payload:
                        robot_provenance[robot_type] = robot_provenance_payload

    metrics_summary = _aggregate_metrics_summary(robot_metrics)
    bundle_root = output_dir.resolve()
    manifest_path = output_dir / "import_manifest.json"
    directory_checksums = build_directory_checksums(
        output_dir, exclude_paths=[manifest_path]
    )
    checksums_path = _write_checksums_file(output_dir, directory_checksums)
    checksums_rel_path = checksums_path.relative_to(output_dir).as_posix()
    checksums_entry = {
        "sha256": _sha256_file(checksums_path),
        "size_bytes": checksums_path.stat().st_size,
    }
    checksums_payload = {
        "download_manifest": None,
        "episodes": {},
        "filtered_episodes": [],
        "lerobot": {
            "dataset_info": None,
            "episodes_index": None,
            "episodes": [],
        },
        "metadata": dict(directory_checksums),
        "missing_episode_ids": [],
        "missing_metadata_files": [],
        "episode_files": {},
        "bundle_files": dict(directory_checksums),
    }
    checksums_payload["metadata"][checksums_rel_path] = checksums_entry
    checksums_payload["bundle_files"][checksums_rel_path] = checksums_entry
    file_inventory = build_file_inventory(output_dir, exclude_paths=[manifest_path])
    asset_provenance_path = _resolve_asset_provenance_reference(
        bundle_root=bundle_root,
        output_dir=output_dir,
        job_metadata=job_metadata,
    )
    config_snapshot = {
        "env": snapshot_env(ENV_SNAPSHOT_KEYS),
        "config": {
            "job_id": job_id,
            "output_dir": str(output_dir),
            "gcs_output_path": normalized_gcs_output_path,
            "robots": [entry["robot_type"] for entry in normalized_robot_entries],
        },
        "job_metadata": job_metadata or {},
    }
    base_provenance = {
        "source": "genie_sim",
        "job_id": job_id,
        "imported_by": "BlueprintPipeline",
        "importer": "genie-sim-import-job",
        "client_mode": "local",
    }
    provenance = collect_provenance(REPO_ROOT, config_snapshot)
    provenance.update(base_provenance)
    provenance["robots"] = robot_provenance

    import_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "schema_definition": MANIFEST_SCHEMA_DEFINITION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "job_id": job_id,
        "output_dir": str(output_dir),
        "gcs_output_path": normalized_gcs_output_path,
        "job_idempotency": {
            "key": idempotency.get("key") if idempotency else None,
            "first_submitted_at": idempotency.get("first_submitted_at") if idempotency else None,
        },
        "success": all(entry["success"] for entry in normalized_robot_entries),
        "episodes": {
            "downloaded": total_downloaded,
            "passed_validation": total_passed,
            "filtered": total_filtered,
            "download_errors": 0,
            "parse_failed": total_parse_failed,
            "parse_failures": [],
            "min_required": min(
                (entry["episodes"].get("min_required", MIN_EPISODES_REQUIRED) for entry in robot_entries),
                default=MIN_EPISODES_REQUIRED,
            ),
        },
        "quality": {
            "average_score": average_quality,
            "min_score": min_quality,
            "max_score": max_quality,
            "threshold": min(
                (entry["quality"]["threshold"] for entry in robot_entries),
                default=0.0,
            ),
            "validation_enabled": any(
                entry["quality"]["validation_enabled"] for entry in normalized_robot_entries
            ),
            "component_thresholds": quality_settings.config.dimension_thresholds,
            "component_failed_episodes": component_failed_total,
            "component_failure_counts": component_failure_counts,
        },
        "quality_config": {
            "min_quality_score": quality_settings.min_quality_score,
            "filter_low_quality": quality_settings.filter_low_quality,
            "dimension_thresholds": quality_settings.config.dimension_thresholds,
            "range": {
                "min_allowed": quality_settings.config.min_allowed,
                "max_allowed": quality_settings.config.max_allowed,
            },
            "defaults": {
                "min_quality_score": quality_settings.config.default_min_quality_score,
                "filter_low_quality": quality_settings.config.default_filter_low_quality,
            },
            "source_path": quality_settings.config.source_path,
        },
        "robots": normalized_robot_entries,
        "job_metadata": job_metadata or {},
        "readme_path": None,
        "checksums_path": _relative_to_bundle(bundle_root, checksums_path),
        "asset_provenance_path": asset_provenance_path,
        "lerobot_metadata_paths": lerobot_metadata_paths,
        "metrics_summary": metrics_summary,
        "provenance": provenance,
        "checksums": checksums_payload,
        "file_inventory": file_inventory,
        "verification": {
            "checksums": {},
        },
        "upload_status": "not_configured",
        "upload_failures": [],
        "upload_started_at": None,
        "upload_completed_at": None,
        "upload_summary": {
            "total_files": 0,
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
        },
    }

    import_manifest["checksums"]["metadata"]["import_manifest.json"] = {
        "sha256": compute_manifest_checksum(import_manifest),
    }

    write_json_atomic(manifest_path, import_manifest, indent=2)

    print("=" * 80)
    print("COMBINED IMPORT MANIFEST CHECKSUM VERIFICATION")
    print("=" * 80)
    verify_exit_code = verify_manifest(manifest_path)
    if verify_exit_code != 0:
        print("[GENIE-SIM-IMPORT] ❌ Combined import manifest verification failed.")
        raise RuntimeError("Combined import manifest verification failed.")
    print("[GENIE-SIM-IMPORT] ✅ Combined import manifest verification succeeded")
    print("=" * 80 + "\n")

    manifest_checksum_result = verify_import_manifest_checksum(manifest_path)
    if not manifest_checksum_result["success"]:
        print(
            "[GENIE-SIM-IMPORT] ❌ Combined import manifest checksum validation failed."
        )
        print("[GENIE-SIM-IMPORT] ❌ Import manifest checksum verification details:")
        for error in manifest_checksum_result["errors"]:
            print(f"[GENIE-SIM-IMPORT]   - {error}")
        raise RuntimeError("Combined import manifest checksum verification failed.")
    return manifest_path


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
            errors.extend(parquet_results["errors"])
            warnings.extend(parquet_results["warnings"])
        except RuntimeError as exc:
            errors.append(str(exc))
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
                if ep_metadata.episode_content_hash:
                    dataset_info["episodes"][-1]["content_hash"] = ep_metadata.episode_content_hash

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
                {
                    "episode_id": ep_metadata.episode_id,
                    "error": str(conversion_error),
                    "retry_attempts": retry_ctx.attempt,
                    "final_exception": repr(conversion_error),
                }
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
        existing_manifest = _load_existing_import_manifest(config.output_dir)
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
    if dataset_info_path.exists():
        try:
            dataset_info_payload = _load_json_file(dataset_info_path)
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
    )
    schema_errors.extend(lerobot_metadata_validation["schema_errors"])

    if schema_errors:
        result.errors.extend(schema_errors)

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

    lerobot_error = None
    lerobot_episode_files = []
    lerobot_skipped_count = 0
    lerobot_skip_rate_percent = 0.0
    conversion_failures: List[Dict[str, str]] = []
    if lerobot_dir.exists():
        lerobot_episode_files = [
            path for path in lerobot_dir.glob("*.json") if path.name != "dataset_info.json"
        ]
        result.lerobot_conversion_success = True
        if isinstance(dataset_info_payload, dict):
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
        result.lerobot_conversion_success = False
        lerobot_error = "LeRobot output directory not found for local import."
        if config.require_lerobot:
            result.errors.append(lerobot_error)
        else:
            result.warnings.append(lerobot_error)

    if conversion_failures and config.require_lerobot:
        failure_ids = ", ".join(
            sorted({entry.get("episode_id", "unknown") for entry in conversion_failures})
        )
        result.errors.append(
            "LeRobot conversion failures detected with REQUIRE_LEROBOT=true: "
            + failure_ids
        )

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
    bundle_root = config.output_dir.resolve()
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
    checksums_path = _write_checksums_file(config.output_dir, directory_checksums)
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
            "converted_count": len(lerobot_episode_files),
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
            "checksums": {},
        },
        "metrics_summary": metrics_summary,
        "regression_metrics": regression_payload,
        "checksums": checksums_payload,
        "provenance": provenance,
    }

    _attach_cost_summary(import_manifest, cost_summary)

    with open(import_manifest_path, "w") as f:
        json.dump(import_manifest, f, indent=2)

    package_path = _create_bundle_package(
        config.output_dir,
        f"lerobot_bundle_{config.job_id}.tar.gz",
        files=[import_manifest_path, readme_path, checksums_path],
        directories=[lerobot_dir],
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
    import_manifest["verification"]["checksums"] = checksums_verification
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
        verification_errors = []
        if checksums_verification["missing_files"]:
            verification_errors.append(
                "Missing files: " + ", ".join(checksums_verification["missing_files"])
            )
        if checksums_verification["checksum_mismatches"]:
            mismatch_paths = [
                mismatch["path"] for mismatch in checksums_verification["checksum_mismatches"]
            ]
            verification_errors.append("Checksum mismatches: " + ", ".join(mismatch_paths))
        if checksums_verification["size_mismatches"]:
            mismatch_paths = [
                mismatch["path"] for mismatch in checksums_verification["size_mismatches"]
            ]
            verification_errors.append("Size mismatches: " + ", ".join(mismatch_paths))
        if checksums_verification["errors"]:
            verification_errors.extend(checksums_verification["errors"])
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

    # Quality configuration
    try:
        quality_settings = resolve_quality_settings()
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)
    min_quality_score = quality_settings.min_quality_score
    enable_validation = parse_bool_env(os.getenv("ENABLE_VALIDATION"), default=True)
    filter_low_quality = quality_settings.filter_low_quality
    production_mode = resolve_production_mode()
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
        log.error("%s", exc)
        sys.exit(1)
    try:
        min_episodes_required = _resolve_min_episodes_required(
            os.getenv("MIN_EPISODES_REQUIRED")
        )
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Polling configuration
    poll_interval = int(os.getenv("GENIE_SIM_POLL_INTERVAL", "30"))
    wait_for_completion = parse_bool_env(os.getenv("WAIT_FOR_COMPLETION"), default=True)

    # Error handling configuration
    fail_on_partial_error = parse_bool_env(os.getenv("FAIL_ON_PARTIAL_ERROR"), default=False)
    allow_partial_firebase_uploads = parse_bool_env(
        os.getenv("ALLOW_PARTIAL_FIREBASE_UPLOADS"),
        default=False,
    )

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
            firebase_upload_suppressed = False
            suppression_reason = None
            if quality_gate_failures:
                firebase_upload_suppressed = True
                suppression_reason = "quality_gate_failure"
            if partial_failure:
                if production_mode or fail_on_partial_error:
                    firebase_upload_suppressed = True
                    if suppression_reason is None:
                        suppression_reason = (
                            "partial_failure_production_or_fail_on_partial_error"
                        )
                elif not allow_partial_firebase_uploads:
                    firebase_upload_suppressed = True
                    if suppression_reason is None:
                        suppression_reason = "partial_failure_partial_uploads_disabled"

            if enable_firebase_upload and not firebase_upload_suppressed:
                for payload in robot_results:
                    robot_type = payload["robot_type"]
                    result = payload["result"]
                    entry = payload["entry"]
                    if not result.success:
                        continue
                    dedup_summary = _prepare_deduplication_summary(
                        result=result,
                        scene_id=scene_id,
                        robot_type=robot_type,
                        prefix=firebase_upload_prefix,
                        log=log,
                    )
                    deduplicated_ids = dedup_summary.get("deduplicated_episode_ids", [])
                    entry["episodes"]["deduplicated"] = len(deduplicated_ids)
                    entry["episodes"]["deduplicated_ids"] = deduplicated_ids
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
                    except Exception as exc:
                        _alert_firebase_upload_failure(
                            scene_id=scene_id,
                            job_id=job_id,
                            robot_type=robot_type,
                            error=str(exc),
                        )
                        log.error("Firebase upload failed: %s", exc)
                        raise
                    firebase_summary["deduplicated_episodes"] = len(deduplicated_ids)
                    firebase_summary["deduplicated_episode_ids"] = deduplicated_ids
                    firebase_summary["remote_prefix"] = firebase_result.remote_prefix
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
                        entry["errors"].append(
                            f"Firebase upload incomplete: {failed_count} file(s) failed"
                        )
                        overall_success = False
                    elif deduplicated_ids:
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
                        gcs_output_path=robot_gcs_output_path,
                        log=log,
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
                dedup_summary = _prepare_deduplication_summary(
                    result=result,
                    scene_id=scene_id,
                    robot_type="default",
                    prefix=firebase_upload_prefix,
                    log=log,
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
            sys.exit(0)
        else:
            log.error("Import failed")
            for error in result.errors:
                log.error("  - %s", error)
            sys.exit(1)
    except Exception as exc:
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

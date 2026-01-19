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
    MIN_QUALITY_SCORE: Minimum quality score for import (default: from quality_config.json)
    ENABLE_VALIDATION: Enable quality validation (default: true)
    REQUIRE_LEROBOT: Treat LeRobot conversion failure as job failure (default: false)
    LEROBOT_SKIP_RATE_MAX: Max allowed LeRobot skip rate percentage (default: 0.0 in production)
    ENABLE_FIREBASE_UPLOAD: Enable Firebase Storage upload of local episodes (default: false)
    FIREBASE_STORAGE_BUCKET: Firebase Storage bucket name for uploads
    FIREBASE_SERVICE_ACCOUNT_JSON: Service account JSON payload for Firebase
    FIREBASE_SERVICE_ACCOUNT_PATH: Path to service account JSON for Firebase
    FIREBASE_UPLOAD_PREFIX: Remote prefix for Firebase uploads (default: datasets)
"""

import hashlib
import json
import os
import logging
import shutil
import sys
import tarfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

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
from tools.geniesim_adapter.mock_mode import resolve_geniesim_mock_mode
from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode
from tools.gcs_upload import (
    calculate_file_md5_base64,
    upload_blob_from_filename,
    verify_blob_upload,
)
from tools.validation.entrypoint_checks import validate_required_env_vars
from tools.utils.atomic_write import write_json_atomic, write_text_atomic
from quality_config import (
    DEFAULT_MIN_QUALITY_SCORE,
    QUALITY_CONFIG,
    resolve_min_quality_score,
)

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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def _classify_validation_error_message(message: str) -> Optional[str]:
    normalized = message.lower()
    if "json" in normalized and "parse" in normalized:
        return "json_parse_failed"
    if "episode file not found" in normalized or "missing" in normalized and "episode file" in normalized:
        return "parquet_missing_file"
    if "missing required field" in normalized or "missing columns" in normalized:
        return "parquet_missing_column"
    if "nan" in normalized:
        return "parquet_nan"
    if "inf" in normalized:
        return "parquet_inf"
    if "timestamp" in normalized and "monotonic" in normalized:
        return "timestamp_not_monotonic"
    if "failed to load episode data" in normalized or "failed to read parquet" in normalized:
        return "parquet_read_failed"
    if "episode failed genie sim validation" in normalized:
        return "geniesim_validation_failed"
    if "missing camera" in normalized:
        return "camera_missing"
    return None


def _collect_validation_error_types(
    result: Dict[str, Any],
    *,
    min_quality_score: float,
) -> List[str]:
    error_types: Set[str] = set()
    for message in result.get("errors", []):
        error_type = _classify_validation_error_message(message)
        if error_type:
            error_types.add(error_type)
    for message in result.get("warnings", []):
        error_type = _classify_validation_error_message(message)
        if error_type:
            error_types.add(error_type)
    if result.get("quality_score", 0.0) < min_quality_score:
        error_types.add("quality_below_threshold")
    if not error_types and not result.get("passed", True):
        error_types.add("validation_failed")
    return sorted(error_types)


def _classify_conversion_error_message(message: str) -> str:
    normalized = message.lower()
    if "missing" in normalized and "episode file" in normalized:
        return "parquet_missing_file"
    if "pyarrow" in normalized:
        return "parquet_dependency_missing"
    if "schema" in normalized or "column" in normalized:
        return "parquet_missing_column"
    return "lerobot_conversion_error"


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
    schema_type = schema.get("type")
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


def _load_existing_import_manifest(output_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = output_dir / "import_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r") as handle:
        return json.load(handle)


def _resolve_job_idempotency(job_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job_metadata:
        return None
    idempotency = job_metadata.get("idempotency")
    return idempotency if isinstance(idempotency, dict) else None


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
) -> List[GeneratedEpisodeMetadata]:
    episode_metadata_list: List[GeneratedEpisodeMetadata] = []
    for episode_file in sorted(recordings_dir.rglob("*.json")):
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
                    frame_count=int(frame_count),
                    duration_seconds=duration_seconds,
                    validation_passed=bool(payload.get("validation_passed", True)),
                    file_size_bytes=episode_file.stat().st_size,
                )
            )
        except Exception as exc:
            print(f"[IMPORT] ⚠️  Failed to parse local episode {episode_file}: {exc}")
    return episode_metadata_list


@dataclass
class ImportConfig:
    """Configuration for episode import."""

    # Genie Sim job
    job_id: str

    # Output
    output_dir: Path
    gcs_output_path: Optional[str] = None
    enable_gcs_uploads: bool = True

    # Quality filtering
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE
    enable_validation: bool = True
    filter_low_quality: bool = True
    require_lerobot: bool = False
    lerobot_skip_rate_max: float = 0.0

    # Polling (if waiting for completion)
    poll_interval: int = 30
    wait_for_completion: bool = True

    # Error handling for partial failures
    fail_on_partial_error: bool = False  # If True, fail the job if any episodes failed

    job_metadata_path: Optional[str] = None
    local_episodes_prefix: Optional[str] = None


@dataclass
class ImportResult:
    """Result of episode import."""

    success: bool
    job_id: str

    # Statistics
    total_episodes_downloaded: int = 0
    episodes_passed_validation: int = 0
    episodes_filtered: int = 0

    # Quality metrics
    average_quality_score: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)

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

    def __init__(self, min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE):
        """
        Initialize validator.

        Args:
            min_quality_score: Minimum quality score to pass
        """
        self.min_quality_score = min_quality_score

    def validate_episode(
        self,
        episode_metadata: GeneratedEpisodeMetadata,
        episode_file: Path,
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
            import pyarrow.parquet as pq
            table = pq.read_table(episode_file)
            df = table.to_pandas()

            # Check required fields
            required_fields = ["observation", "action"]
            optional_fields = ["reward", "done", "timestamp"]

            for field in required_fields:
                # Check variations of field names
                field_variations = [field, f"observation.{field}", f"{field}s"]
                if not any(variation in df.columns or any(variation in col for col in df.columns) for variation in field_variations):
                    errors.append(f"Missing required field: {field}")

            # Check for NaN/Inf values
            for col in df.columns:
                if df[col].dtype in [np.float32, np.float64]:
                    if df[col].isna().any():
                        errors.append(f"Column '{col}' contains NaN values")
                    if np.isinf(df[col]).any():
                        errors.append(f"Column '{col}' contains Inf values")

            # Check timestamp monotonicity
            timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
            for ts_col in timestamp_cols:
                if df[ts_col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                    if not df[ts_col].is_monotonic_increasing:
                        errors.append(f"Timestamps in '{ts_col}' are not monotonic")

            # Check observation shapes are consistent
            obs_cols = [col for col in df.columns if col.startswith('observation')]
            for obs_col in obs_cols:
                if df[obs_col].apply(lambda x: hasattr(x, '__len__')).any():
                    shapes = df[obs_col].apply(lambda x: np.array(x).shape if hasattr(x, '__len__') else None)
                    unique_shapes = shapes.dropna().unique()
                    if len(unique_shapes) > 1:
                        warnings.append(f"Inconsistent shapes in '{obs_col}': {unique_shapes}")

            # Check action bounds are reasonable
            action_cols = [col for col in df.columns if 'action' in col.lower()]
            for action_col in action_cols:
                if df[action_col].dtype in [np.float32, np.float64]:
                    action_values = df[action_col].values
                    if hasattr(action_values[0], '__len__'):
                        # Multi-dimensional action
                        action_arr = np.array([np.array(a) for a in action_values])
                        if np.abs(action_arr).max() > 10.0:  # Reasonable joint limits
                            warnings.append(f"Action values in '{action_col}' exceed reasonable bounds (max: {np.abs(action_arr).max():.2f})")
                    else:
                        # Scalar action
                        if np.abs(action_values).max() > 10.0:
                            warnings.append(f"Action values in '{action_col}' exceed reasonable bounds (max: {np.abs(action_values).max():.2f})")

        except ImportError:
            warnings.append("PyArrow not available - skipping detailed validation")
        except Exception as e:
            warnings.append(f"Failed to load episode data for validation: {e}")

        # Check quality score
        quality_score = episode_metadata.quality_score
        if quality_score < self.min_quality_score:
            warnings.append(
                f"Quality score {quality_score:.2f} below threshold {self.min_quality_score:.2f}"
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
            and episode_metadata.validation_passed
        )

        return {
            "passed": passed,
            "quality_score": quality_score,
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

        for episode in episodes:
            episode_file = episode_dir / f"{episode.episode_id}.parquet"
            result = self.validate_episode(episode, episode_file)
            results.append({
                "episode_id": episode.episode_id,
                **result,
            })

            if result["passed"]:
                passed_count += 1

            quality_scores.append(result["quality_score"])

        return {
            "total_episodes": len(episodes),
            "passed_count": passed_count,
            "failed_count": len(episodes) - passed_count,
            "average_quality_score": np.mean(quality_scores) if quality_scores else 0.0,
            "min_quality_score": np.min(quality_scores) if quality_scores else 0.0,
            "max_quality_score": np.max(quality_scores) if quality_scores else 0.0,
            "episode_results": results,
        }


# =============================================================================
# LeRobot Conversion
# =============================================================================


def convert_to_lerobot(
    episodes_dir: Path,
    output_dir: Path,
    episode_metadata_list: List[GeneratedEpisodeMetadata],
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
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
            if ep_metadata.quality_score < min_quality_score:
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
                "validation_passed": ep_metadata.validation_passed,
                "file": str(episode_output.name),
            })

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
        if ep_metadata.quality_score < min_quality_score:
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

                # Update dataset info
                dataset_info["episodes"].append({
                    "episode_id": ep_metadata.episode_id,
                    "episode_index": converted_count,
                    "num_frames": len(df),
                    "duration_seconds": ep_metadata.duration_seconds,
                    "quality_score": ep_metadata.quality_score,
                    "validation_passed": ep_metadata.validation_passed,
                    "file": str(episode_output.name),
                })

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
    metrics = get_metrics()
    metrics_labels = {
        "job": "genie-sim-import-job",
        "scene_id": scene_id,
        "job_id": config.job_id,
    }

    idempotency = _resolve_job_idempotency(job_metadata)
    if idempotency:
        existing_manifest = _load_existing_import_manifest(config.output_dir)
        existing_idempotency = (
            existing_manifest.get("job_idempotency", {}) if existing_manifest else {}
        )
        if existing_idempotency.get("key") == idempotency.get("key"):
            result.success = True
            result.warnings.append(
                "Duplicate import detected; matching import_manifest.json "
                "already exists for this idempotency key."
            )
            return result

    recordings_dir = config.output_dir / "recordings"
    if not recordings_dir.exists():
        result.errors.append(f"Local recordings directory missing: {recordings_dir}")
        return result

    schema_errors: List[str] = []
    for episode_file in sorted(recordings_dir.rglob("*.json")):
        try:
            payload = _load_json_file(episode_file)
        except Exception as exc:
            schema_errors.append(f"recording {episode_file.relative_to(config.output_dir)}: {exc}")
            metrics.episode_validation_error_total.inc(labels={
                **metrics_labels,
                "status": "failure",
                "error_type": "json_parse_failed",
            })
            continue
        schema_errors.extend(
            _validate_schema_payload(
                payload,
                "geniesim_local_episode.schema.json",
                f"recording {episode_file.relative_to(config.output_dir)}",
            )
        )

    episode_metadata_list = _collect_local_episode_metadata(recordings_dir)
    if not episode_metadata_list:
        result.errors.append(f"No local episode files found under {recordings_dir}")
        return result

    validator = ImportedEpisodeValidator(min_quality_score=config.min_quality_score)
    validation_summary = validator.validate_batch(episode_metadata_list, recordings_dir)
    for entry in validation_summary["episode_results"]:
        metrics.episode_quality_score.observe(
            entry["quality_score"],
            labels=metrics_labels,
        )
        if entry["passed"]:
            metrics.episode_validation_pass_total.inc(labels={
                **metrics_labels,
                "status": "success",
            })
            continue
        metrics.episode_validation_fail_total.inc(labels={
            **metrics_labels,
            "status": "failure",
        })
        for error_type in _collect_validation_error_types(
            entry,
            min_quality_score=config.min_quality_score,
        ):
            metrics.episode_validation_error_total.inc(labels={
                **metrics_labels,
                "status": "failure",
                "error_type": error_type,
            })
    failed_episode_ids = [
        entry["episode_id"]
        for entry in validation_summary["episode_results"]
        if not entry["passed"]
    ]
    filtered_episode_ids = (
        failed_episode_ids if not config.fail_on_partial_error else []
    )
    if validation_summary["failed_count"] > 0:
        failure_message = (
            f"{validation_summary['failed_count']} local episodes failed validation"
        )
        if config.fail_on_partial_error:
            result.errors.append(failure_message)
            result.success = False
        else:
            result.warnings.append(failure_message)
            result.warnings.append(
                "Excluding failed episodes from manifest: "
                + ", ".join(failed_episode_ids)
            )

    lerobot_dir = config.output_dir / "lerobot"
    dataset_info_path = lerobot_dir / "dataset_info.json"
    episodes_index_path = lerobot_dir / "episodes.jsonl"
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
                            f"metadata {episodes_index_path.relative_to(config.output_dir)}:{line_number}",
                        )
                    )
        except Exception as exc:
            schema_errors.append(
                f"metadata {episodes_index_path.relative_to(config.output_dir)}: {exc}"
            )
    else:
        schema_errors.append(
            f"metadata {episodes_index_path.relative_to(config.output_dir)}: missing episodes.jsonl"
        )

    if schema_errors:
        result.errors.extend(schema_errors)

    total_size_bytes = 0
    for episode_file in recordings_dir.rglob("*.json"):
        total_size_bytes += episode_file.stat().st_size

    result.total_episodes_downloaded = len(episode_metadata_list)
    low_quality_episodes = [
        ep for ep in episode_metadata_list if ep.quality_score < config.min_quality_score
    ]
    result.episodes_passed_validation = validation_summary["passed_count"]
    result.episodes_filtered = validation_summary["failed_count"]
    quality_scores = [ep.quality_score for ep in episode_metadata_list]
    result.average_quality_score = float(np.mean(quality_scores)) if quality_scores else 0.0
    quality_min_score = float(np.min(quality_scores)) if quality_scores else 0.0
    quality_max_score = float(np.max(quality_scores)) if quality_scores else 0.0

    if low_quality_episodes:
        result.errors.append(
            f"{len(low_quality_episodes)} local episodes below min_quality_score={config.min_quality_score}"
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

    # Write machine-readable import manifest for workflows
    import_manifest_path = config.output_dir / "import_manifest.json"
    gcs_output_path = config.gcs_output_path
    output_dir_str = str(config.output_dir)
    if not gcs_output_path and output_dir_str.startswith("/mnt/gcs/"):
        gcs_output_path = "gs://" + output_dir_str[len("/mnt/gcs/"):]

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
    if conversion_failures:
        for failure in conversion_failures:
            failure_error = failure.get("error", "unknown")
            metrics.lerobot_conversion_fail_total.inc(labels={
                **metrics_labels,
                "status": "failure",
                "error_type": _classify_conversion_error_message(str(failure_error)),
            })
    metrics_summary = {
        "backend": metrics.backend.value,
        "stats": metrics.get_stats(),
    }

    episode_checksums = []
    for episode_file in sorted(recordings_dir.rglob("*.json")):
        episode_checksums.append({
            "episode_id": episode_file.stem,
            "file_name": episode_file.relative_to(config.output_dir).as_posix(),
            "sha256": _sha256_file(episode_file),
        })

    lerobot_checksums = {
        "dataset_info": None,
        "episodes_index": None,
        "episodes": [],
    }
    dataset_info_path = lerobot_dir / "dataset_info.json"
    if dataset_info_path.exists():
        lerobot_checksums["dataset_info"] = _sha256_file(dataset_info_path)
    for lerobot_file in sorted(lerobot_episode_files):
        lerobot_checksums["episodes"].append({
            "file_name": lerobot_file.name,
            "sha256": _sha256_file(lerobot_file),
        })

    episode_paths = sorted(recordings_dir.rglob("*.json"))
    metadata_paths = get_lerobot_metadata_paths(config.output_dir)
    missing_metadata_files = []
    lerobot_info_path = config.output_dir / "lerobot" / "meta" / "info.json"
    if not lerobot_info_path.exists():
        missing_metadata_files.append(lerobot_info_path.relative_to(config.output_dir).as_posix())
    bundle_root = config.output_dir.resolve()
    readme_path = _write_lerobot_readme(config.output_dir, lerobot_dir)
    directory_checksums = build_directory_checksums(config.output_dir, exclude_paths=[import_manifest_path])
    episode_rel_paths = {path.relative_to(config.output_dir).as_posix() for path in episode_paths}
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
            "enable_validation": config.enable_validation,
            "filter_low_quality": config.filter_low_quality,
            "require_lerobot": config.require_lerobot,
            "lerobot_skip_rate_max": config.lerobot_skip_rate_max,
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
            "download_errors": 0,
        },
        "quality": {
            "average_score": result.average_quality_score,
            "min_score": quality_min_score,
            "max_score": quality_max_score,
            "threshold": config.min_quality_score,
            "validation_enabled": config.enable_validation,
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
            "skip_rate_max": config.lerobot_skip_rate_max,
        },
        "validation": {
            "episodes": {
                **validation_summary,
                "failed_episode_ids": failed_episode_ids,
                "filtered_episode_ids": filtered_episode_ids,
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
        print("=" * 80 + "\n")
        return result

    result.checksum_verification_passed = (
        checksums_verification["success"] and manifest_checksum_result["success"]
    )
    if not result.checksum_verification_passed:
        result.success = False
        return result

    result.success = len(result.errors) == 0

    print("=" * 80)
    print("LOCAL IMPORT COMPLETE")
    print("=" * 80)
    print(f"{'✅' if result.success else '❌'} Imported {result.episodes_passed_validation} local episodes")
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


def main():
    """Main entry point for import job."""
    print("\n[GENIE-SIM-IMPORT] Starting import job...")

    # Get configuration from environment
    job_id = os.getenv("GENIE_SIM_JOB_ID")
    if not job_id:
        print("[GENIE-SIM-IMPORT] ERROR: GENIE_SIM_JOB_ID is required")
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
    output_prefix = os.getenv("OUTPUT_PREFIX", f"scenes/{scene_id}/episodes")
    explicit_gcs_output_path = os.getenv("GCS_OUTPUT_PATH") or None
    job_metadata_path = os.getenv("JOB_METADATA_PATH") or None
    local_episodes_prefix = os.getenv("LOCAL_EPISODES_PREFIX") or None

    job_metadata = None
    if job_metadata_path:
        try:
            job_metadata = _load_local_job_metadata(bucket, job_metadata_path)
            artifacts = job_metadata.get("artifacts", {})
            if not local_episodes_prefix:
                local_episodes_prefix = artifacts.get("episodes_prefix")
        except FileNotFoundError as e:
            print(f"[GENIE-SIM-IMPORT] WARNING: {e}")
            if not local_episodes_prefix:
                sys.exit(1)

    # Quality configuration
    try:
        min_quality_score = resolve_min_quality_score(
            os.getenv("MIN_QUALITY_SCORE"),
            QUALITY_CONFIG,
        )
    except ValueError as exc:
        print(f"[GENIE-SIM-IMPORT] ERROR: {exc}")
        sys.exit(1)
    enable_validation = parse_bool_env(os.getenv("ENABLE_VALIDATION"), default=True)
    filter_low_quality = parse_bool_env(os.getenv("FILTER_LOW_QUALITY"), default=True)
    require_lerobot = parse_bool_env(os.getenv("REQUIRE_LEROBOT"), default=False)
    disable_gcs_upload = parse_bool_env(os.getenv("DISABLE_GCS_UPLOAD"), default=False)
    enable_firebase_upload = parse_bool_env(os.getenv("ENABLE_FIREBASE_UPLOAD"), default=False)
    firebase_upload_prefix = os.getenv("FIREBASE_UPLOAD_PREFIX", "datasets")
    try:
        lerobot_skip_rate_max = _resolve_skip_rate_max(
            os.getenv("LEROBOT_SKIP_RATE_MAX")
        )
    except ValueError as exc:
        print(f"[GENIE-SIM-IMPORT] ERROR: {exc}")
        sys.exit(1)

    # Polling configuration
    poll_interval = int(os.getenv("GENIE_SIM_POLL_INTERVAL", "30"))
    wait_for_completion = parse_bool_env(os.getenv("WAIT_FOR_COMPLETION"), default=True)

    # Error handling configuration
    fail_on_partial_error = parse_bool_env(os.getenv("FAIL_ON_PARTIAL_ERROR"), default=False)

    # Validate credentials at startup
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        from startup_validation import validate_and_fail_fast
        validate_and_fail_fast(
            job_name="GENIE-SIM-IMPORT",
            require_geniesim=False,
            require_gemini=False,
            validate_gcs=True,
        )
    except ImportError as e:
        print(f"[GENIE-SIM-IMPORT] WARNING: Startup validation unavailable: {e}")
    except SystemExit:
        # Re-raise to exit immediately
        raise

    print(f"[GENIE-SIM-IMPORT] Configuration:")
    print(f"[GENIE-SIM-IMPORT]   Job ID: {job_id}")
    print(f"[GENIE-SIM-IMPORT]   Output Prefix: {output_prefix}")
    print(f"[GENIE-SIM-IMPORT]   Min Quality: {min_quality_score}")
    print(
        "[GENIE-SIM-IMPORT]   Quality Range: "
        f"{QUALITY_CONFIG.min_allowed} - {QUALITY_CONFIG.max_allowed}"
    )
    print(f"[GENIE-SIM-IMPORT]   Enable Validation: {enable_validation}")
    print(f"[GENIE-SIM-IMPORT]   Require LeRobot: {require_lerobot}")
    print(f"[GENIE-SIM-IMPORT]   LeRobot Skip Rate Max: {lerobot_skip_rate_max:.2f}%")
    print(f"[GENIE-SIM-IMPORT]   GCS Uploads Enabled: {not disable_gcs_upload}")
    print(f"[GENIE-SIM-IMPORT]   Firebase Uploads Enabled: {enable_firebase_upload}")
    print(f"[GENIE-SIM-IMPORT]   Wait for Completion: {wait_for_completion}")
    print(f"[GENIE-SIM-IMPORT]   Fail on Partial Error: {fail_on_partial_error}\n")

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
    config = ImportConfig(
        job_id=job_id,
        output_dir=output_dir,
        gcs_output_path=gcs_output_path,
        enable_gcs_uploads=not disable_gcs_upload,
        min_quality_score=min_quality_score,
        enable_validation=enable_validation,
        filter_low_quality=filter_low_quality,
        require_lerobot=require_lerobot,
        lerobot_skip_rate_max=lerobot_skip_rate_max,
        poll_interval=poll_interval,
        wait_for_completion=wait_for_completion,
        fail_on_partial_error=fail_on_partial_error,
        job_metadata_path=job_metadata_path,
        local_episodes_prefix=local_episodes_prefix,
    )

    # Run import
    try:
        metrics = get_metrics()
        with metrics.track_job("genie-sim-import-job", scene_id):
            result = run_local_import_job(config, job_metadata=job_metadata)

        if result.success:
            print(f"[GENIE-SIM-IMPORT] ✅ Import succeeded")
            print(f"[GENIE-SIM-IMPORT] Episodes imported: {result.episodes_passed_validation}")
            print(f"[GENIE-SIM-IMPORT] Average quality: {result.average_quality_score:.2f}")
            if enable_firebase_upload:
                from tools.firebase_upload.uploader import upload_episodes_to_firebase

                print("[GENIE-SIM-IMPORT] Uploading episodes to Firebase Storage...")
                try:
                    upload_summary = upload_episodes_to_firebase(
                        result.output_dir,
                        scene_id,
                        prefix=firebase_upload_prefix,
                    )
                except Exception as exc:
                    print(f"[GENIE-SIM-IMPORT] ❌ Firebase upload failed: {exc}")
                    raise
                print(
                    "[GENIE-SIM-IMPORT] Firebase upload complete: "
                    f"{upload_summary['uploaded']}/{upload_summary['total_files']} files"
                )
            sys.exit(0)
        else:
            print(f"[GENIE-SIM-IMPORT] ❌ Import failed")
            for error in result.errors:
                print(f"[GENIE-SIM-IMPORT]   - {error}")
            sys.exit(1)



if __name__ == "__main__":
    try:
        main()
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

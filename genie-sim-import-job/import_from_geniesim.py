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
"""

import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
)
from verify_import_manifest import verify_manifest

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.alerting import send_alert

# Import Genie Sim client
sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))
from geniesim_client import (
    GenieSimClient,
    GenieSimAPIError,
    JobStatus,
    JobProgress,
    DownloadResult,
    GeneratedEpisodeMetadata,
)
from tools.metrics.pipeline_metrics import get_metrics
from tools.geniesim_adapter.mock_mode import resolve_geniesim_mock_mode
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

    # Quality filtering
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE
    enable_validation: bool = True
    filter_low_quality: bool = True
    require_lerobot: bool = False

    # Polling (if waiting for completion)
    poll_interval: int = 30
    wait_for_completion: bool = True

    # Error handling for partial failures
    fail_on_partial_error: bool = False  # If True, fail the job if any episodes failed

    # Submission mode
    submission_mode: str = "api"
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
            "total_frames": total_frames,
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

    # LeRobot dataset metadata
    dataset_info = _build_dataset_info(
        job_id=job_id,
        scene_id=scene_id,
        source="genie_sim",
        converted_at=datetime.utcnow().isoformat() + "Z",
    )

    quality_scores = []

    for ep_metadata in episode_metadata_list:
        # Skip low-quality episodes
        if ep_metadata.quality_score < min_quality_score:
            skipped_count += 1
            continue

        episode_file = episodes_dir / f"{ep_metadata.episode_id}.parquet"
        if not episode_file.exists():
            skipped_count += 1
            continue

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

        except Exception as e:
            print(f"[LEROBOT] Warning: Failed to convert episode {ep_metadata.episode_id}: {e}")
            skipped_count += 1
            continue

    # Update dataset statistics
    dataset_info["total_episodes"] = converted_count
    dataset_info["total_frames"] = total_frames
    dataset_info["skipped_episodes"] = skipped_count
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
        "total_frames": total_frames,
        "output_dir": output_dir,
        "metadata_file": metadata_file,
    }


# =============================================================================
# Import Job
# =============================================================================


def run_import_job(
    config: ImportConfig,
    client: GenieSimClient,
    job_metadata: Optional[Dict[str, Any]] = None,
) -> ImportResult:
    """
    Run episode import job.

    Args:
        config: Import configuration
        client: Genie Sim client

    Returns:
        ImportResult with statistics and output paths
    """
    print("\n" + "=" * 80)
    print("GENIE SIM EPISODE IMPORT JOB")
    print("=" * 80)
    print(f"Job ID: {config.job_id}")
    print(f"Output: {config.output_dir}")
    print(f"Min Quality: {config.min_quality_score}")
    print("=" * 80 + "\n")

    result = ImportResult(
        success=False,
        job_id=config.job_id,
    )
    quality_min_score = 0.0
    quality_max_score = 0.0
    scene_id = os.environ.get("SCENE_ID", "unknown")

    try:
        # Step 1: Check/wait for job completion
        if config.wait_for_completion:
            print(f"[IMPORT] Waiting for job {config.job_id} to complete...")

            def progress_callback(progress: JobProgress):
                print(
                    f"[IMPORT] Progress: {progress.progress_percent:.1f}% - "
                    f"{progress.episodes_generated}/{progress.total_episodes_target} episodes - "
                    f"{progress.current_task}"
                )

            try:
                metrics = get_metrics()
                with metrics.track_api_call("genie-sim", "wait_for_completion", scene_id):
                    final_progress = client.wait_for_completion(
                        config.job_id,
                        poll_interval=config.poll_interval,
                        callback=progress_callback,
                    )
                print(f"[IMPORT] ✅ Job completed: {final_progress.episodes_generated} episodes generated\n")

            except GenieSimAPIError as e:
                result.errors.append(f"Job failed or was cancelled: {e}")
                return result

        else:
            # Just check status
            metrics = get_metrics()
            with metrics.track_api_call("genie-sim", "get_job_status", scene_id):
                progress = client.get_job_status(config.job_id)
            if progress.status != JobStatus.COMPLETED:
                result.errors.append(f"Job not completed (status: {progress.status.value})")
                return result

        # Step 2: Download episodes
        print(f"[IMPORT] Downloading episodes...")
        metrics = get_metrics()
        with metrics.track_api_call("genie-sim", "download_episodes", scene_id):
            download_result = client.download_episodes(
                config.job_id,
                config.output_dir,
                validate=True,
            )

        if not download_result.success:
            result.errors.extend(download_result.errors)

            # Write failed episodes details for debugging
            if download_result.errors:
                failed_episodes_path = config.output_dir / "failed_episodes.json"
                failed_episodes_data = {
                    "job_id": config.job_id,
                    "total_episodes_attempted": download_result.episode_count,
                    "failed_count": len(download_result.errors),
                    "errors": [
                        {
                            "error": str(err),
                            "suggested_remediation": "Check Genie Sim job logs for details",
                        }
                        for err in download_result.errors
                    ],
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }

                try:
                    config.output_dir.mkdir(parents=True, exist_ok=True)
                    with open(failed_episodes_path, "w") as f:
                        json.dump(failed_episodes_data, f, indent=2)
                    print(f"[IMPORT] ⚠️  Failed episodes details written to: {failed_episodes_path}")
                except Exception as e:
                    print(f"[IMPORT] WARNING: Could not write failed episodes file: {e}")

            return result

        print(f"[IMPORT] ✅ Downloaded {download_result.episode_count} episodes")
        print(f"[IMPORT]    Total size: {download_result.total_size_bytes / 1024 / 1024:.1f} MB\n")

        # Check for partial failures (some episodes downloaded, others failed)
        if download_result.errors:
            print(f"[IMPORT] ⚠️  WARNING: {len(download_result.errors)} episodes had errors during download")
            result.warnings.extend(download_result.errors)

            # Write failed episodes details
            failed_episodes_path = config.output_dir / "failed_episodes.json"
            failed_episodes_data = {
                "job_id": config.job_id,
                "total_episodes_attempted": download_result.episode_count + len(download_result.errors),
                "successful_downloads": download_result.episode_count,
                "failed_count": len(download_result.errors),
                "errors": [
                    {
                        "error": str(err),
                        "suggested_remediation": "Check Genie Sim job logs; episodes may have been filtered by quality",
                    }
                    for err in download_result.errors
                ],
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            try:
                with open(failed_episodes_path, "w") as f:
                    json.dump(failed_episodes_data, f, indent=2)
                print(f"[IMPORT]    Failed episodes details: {failed_episodes_path}")
            except Exception as e:
                print(f"[IMPORT]    WARNING: Could not write failed episodes file: {e}")

            # Optionally fail on partial errors
            if config.fail_on_partial_error:
                result.errors.append(
                    f"{len(download_result.errors)} episodes failed and fail_on_partial_error=True"
                )
                print(f"[IMPORT] ❌ Failing due to partial errors (fail_on_partial_error=True)")
                return result

        result.total_episodes_downloaded = download_result.episode_count
        result.output_dir = download_result.output_dir
        result.manifest_path = download_result.manifest_path

        # Step 3: Validate episodes
        if config.enable_validation:
            print(f"[IMPORT] Validating episodes...")
            validator = ImportedEpisodeValidator(config.min_quality_score)

            validation_result = validator.validate_batch(
                download_result.episodes,
                config.output_dir,
            )

            print(f"[IMPORT] Validation results:")
            print(f"[IMPORT]   Total: {validation_result['total_episodes']}")
            print(f"[IMPORT]   Passed: {validation_result['passed_count']}")
            print(f"[IMPORT]   Failed: {validation_result['failed_count']}")
            print(f"[IMPORT]   Avg Quality: {validation_result['average_quality_score']:.2f}")
            print(
                f"[IMPORT]   Quality Range: "
                f"[{validation_result['min_quality_score']:.2f}, {validation_result['max_quality_score']:.2f}]\n"
            )

            result.episodes_passed_validation = validation_result['passed_count']
            result.episodes_filtered = validation_result['failed_count']
            result.average_quality_score = validation_result['average_quality_score']
            quality_min_score = validation_result["min_quality_score"]
            quality_max_score = validation_result["max_quality_score"]

            if validation_result["failed_count"] > 0:
                result.errors.append(
                    f"{validation_result['failed_count']} episodes failed validation (min_quality_score={config.min_quality_score})"
                )

            low_quality_episodes = [
                ep.episode_id
                for ep in download_result.episodes
                if ep.quality_score < config.min_quality_score
            ]
            if low_quality_episodes:
                result.errors.append(
                    f"{len(low_quality_episodes)} episodes below min_quality_score={config.min_quality_score}"
                )

            # Step 4: Filter low-quality episodes
            if config.filter_low_quality:
                print(f"[IMPORT] Filtering low-quality episodes...")
                filtered_count = 0

                for ep_result in validation_result['episode_results']:
                    if not ep_result['passed']:
                        episode_id = ep_result['episode_id']
                        episode_file = config.output_dir / f"{episode_id}.parquet"

                        # Move to filtered directory
                        filtered_dir = config.output_dir / "filtered"
                        filtered_dir.mkdir(exist_ok=True)

                        if episode_file.exists():
                            shutil.move(str(episode_file), str(filtered_dir / episode_file.name))
                            filtered_count += 1

                            # Log reason
                            reason_file = filtered_dir / f"{episode_id}.reason.txt"
                            with open(reason_file, "w") as f:
                                f.write(f"Quality Score: {ep_result['quality_score']:.2f}\n")
                                f.write(f"Threshold: {config.min_quality_score:.2f}\n")
                                f.write("\nErrors:\n")
                                for error in ep_result['errors']:
                                    f.write(f"  - {error}\n")
                                f.write("\nWarnings:\n")
                                for warning in ep_result['warnings']:
                                    f.write(f"  - {warning}\n")

                print(f"[IMPORT]   Filtered {filtered_count} low-quality episodes\n")

        else:
            result.episodes_passed_validation = result.total_episodes_downloaded
            quality_scores = [ep.quality_score for ep in download_result.episodes]
            result.average_quality_score = np.mean(quality_scores) if quality_scores else 0.0
            quality_min_score = float(np.min(quality_scores)) if quality_scores else 0.0
            quality_max_score = float(np.max(quality_scores)) if quality_scores else 0.0
            low_quality_episodes = [
                ep.episode_id
                for ep in download_result.episodes
                if ep.quality_score < config.min_quality_score
            ]
            if low_quality_episodes:
                result.errors.append(
                    f"{len(low_quality_episodes)} episodes below min_quality_score={config.min_quality_score}"
                )

        # Step 5: Convert episodes to LeRobot format
        print(f"[IMPORT] Converting episodes to LeRobot format...")
        lerobot_dir = config.output_dir / "lerobot"
        lerobot_error = None
        convert_result: Dict[str, Any] = {}
        try:
            convert_result = convert_to_lerobot(
                episodes_dir=config.output_dir,
                output_dir=lerobot_dir,
                episode_metadata_list=download_result.episodes,
                min_quality_score=config.min_quality_score,
                job_id=config.job_id,
                scene_id=scene_id,
            )
            result.lerobot_conversion_success = True
            print(f"[IMPORT] ✅ Converted {convert_result['converted_count']} episodes to LeRobot format")
            print(f"[IMPORT]    Output: {convert_result['output_dir']}\n")
        except Exception as e:
            lerobot_error = str(e)
            result.lerobot_conversion_success = False
            result.warnings.append(f"LeRobot conversion failed: {lerobot_error}")
            print(f"[IMPORT] ⚠️  LeRobot conversion failed: {lerobot_error}\n")

            if config.enable_validation or config.require_lerobot:
                result.errors.append(
                    "LeRobot conversion failed and is required "
                    f"(enable_validation={config.enable_validation}, require_lerobot={config.require_lerobot}): "
                    f"{lerobot_error}"
                )

        # Step 6: Update manifest with import metadata
        if result.manifest_path and result.manifest_path.exists():
            with open(result.manifest_path, "r") as f:
                manifest = json.load(f)

            manifest["import_metadata"] = {
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "imported_by": "BlueprintPipeline",
                "total_downloaded": result.total_episodes_downloaded,
                "passed_validation": result.episodes_passed_validation,
                "filtered": result.episodes_filtered,
                "average_quality_score": result.average_quality_score,
                "lerobot_converted": convert_result.get('converted_count', 0),
                "lerobot_conversion_success": result.lerobot_conversion_success,
                "lerobot_conversion_error": lerobot_error,
            }

            with open(result.manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        # Step 7: Write machine-readable import manifest for workflows
        import_manifest_path = config.output_dir / "import_manifest.json"
        gcs_output_path = None
        output_dir_str = str(config.output_dir)
        if output_dir_str.startswith("/mnt/gcs/"):
            gcs_output_path = "gs://" + output_dir_str[len("/mnt/gcs/"):]
        bundle_root = config.output_dir.resolve()
        output_dir_str = _relative_to_bundle(bundle_root, config.output_dir)

        metrics = get_metrics()
        metrics_summary = {
            "backend": metrics.backend.value,
            "stats": metrics.get_stats(),
        }
        episode_checksums = []
        filtered_checksums = []
        for episode in download_result.episodes:
            episode_file = config.output_dir / f"{episode.episode_id}.parquet"
            if episode_file.exists():
                episode_checksums.append({
                    "episode_id": episode.episode_id,
                    "file_name": episode_file.name,
                    "sha256": _sha256_file(episode_file),
                })
        filtered_dir = config.output_dir / "filtered"
        if filtered_dir.exists():
            for filtered_file in sorted(filtered_dir.glob("*.parquet")):
                filtered_checksums.append({
                    "episode_id": filtered_file.stem,
                    "file_name": filtered_file.name,
                    "sha256": _sha256_file(filtered_file),
                })

        lerobot_checksums = {
            "dataset_info": None,
            "episodes_index": None,
            "episodes": [],
        }
        dataset_info_path = lerobot_dir / "dataset_info.json"
        episodes_index_path = lerobot_dir / "episodes.jsonl"
        if dataset_info_path.exists():
            lerobot_checksums["dataset_info"] = _sha256_file(dataset_info_path)
        if episodes_index_path.exists():
            lerobot_checksums["episodes_index"] = _sha256_file(episodes_index_path)
        if lerobot_dir.exists():
            for lerobot_file in sorted(lerobot_dir.glob("episode_*.parquet")):
                lerobot_checksums["episodes"].append({
                    "file_name": lerobot_file.name,
                    "sha256": _sha256_file(lerobot_file),
                })

        download_manifest_checksum = (
            _sha256_file(result.manifest_path)
            if result.manifest_path and result.manifest_path.exists()
            else None
        )
        base_provenance = {
            "source": "genie_sim",
            "job_id": config.job_id,
            "scene_id": scene_id or None,
            "imported_by": "BlueprintPipeline",
            "importer": "genie-sim-import-job",
            "client_mode": "mock" if getattr(client, "mock_mode", False) else "api",
        }
        episode_ids = [episode.episode_id for episode in download_result.episodes]
        episode_paths, missing_episode_ids = get_episode_file_paths(config.output_dir, episode_ids)
        metadata_paths = get_lerobot_metadata_paths(config.output_dir)
        lerobot_info_path = config.output_dir / "lerobot" / "meta" / "info.json"
        missing_metadata_files = []
        if not lerobot_info_path.exists():
            missing_metadata_files.append(lerobot_info_path.relative_to(config.output_dir).as_posix())
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
            "missing_episode_ids": missing_episode_ids,
            "missing_metadata_files": missing_metadata_files,
        }
        checksums_payload = {
            "download_manifest": download_manifest_checksum,
            "episodes": episode_checksums,
            "filtered_episodes": filtered_checksums,
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
                "min_quality_score": config.min_quality_score,
                "enable_validation": config.enable_validation,
                "filter_low_quality": config.filter_low_quality,
                "require_lerobot": config.require_lerobot,
                "poll_interval": config.poll_interval,
                "wait_for_completion": config.wait_for_completion,
                "fail_on_partial_error": config.fail_on_partial_error,
                "submission_mode": config.submission_mode,
                "job_metadata_path": config.job_metadata_path,
                "local_episodes_prefix": config.local_episodes_prefix,
            },
        }
        provenance = collect_provenance(REPO_ROOT, config_snapshot)
        provenance.update(base_provenance)
        import_manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "schema_definition": MANIFEST_SCHEMA_DEFINITION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "job_id": config.job_id,
            "output_dir": output_dir_str,
            "gcs_output_path": gcs_output_path,
            "readme_path": _relative_to_bundle(bundle_root, readme_path),
            "checksums_path": _relative_to_bundle(bundle_root, checksums_path),
            "episodes": {
                "downloaded": result.total_episodes_downloaded,
                "passed_validation": result.episodes_passed_validation,
                "filtered": result.episodes_filtered,
                "download_errors": len(download_result.errors),
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
                "converted_count": convert_result.get("converted_count", 0),
                "output_dir": _relative_to_bundle(bundle_root, lerobot_dir),
                "error": lerobot_error,
                "required": config.require_lerobot or config.enable_validation,
            },
            "metrics_summary": metrics_summary,
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

        # Success
        result.success = len(result.errors) == 0

        print("=" * 80)
        print("IMPORT COMPLETE")
        print("=" * 80)
        print(f"{'✅' if result.success else '❌'} Successfully imported {result.episodes_passed_validation} episodes")
        print(f"Output directory: {result.output_dir}")
        print(f"Manifest: {result.manifest_path}")
        print("=" * 80 + "\n")

        return result

    except Exception as e:
        print(f"\n❌ ERROR during import: {e}")
        traceback.print_exc()
        result.errors.append(str(e))
        return result


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

    lerobot_dir = config.output_dir / "lerobot"
    dataset_info_path = lerobot_dir / "dataset_info.json"
    episodes_index_path = lerobot_dir / "episodes.jsonl"
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
    result.episodes_passed_validation = len(episode_metadata_list) - len(low_quality_episodes)
    result.episodes_filtered = len(low_quality_episodes)
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
    if lerobot_dir.exists():
        lerobot_episode_files = [
            path for path in lerobot_dir.glob("*.json") if path.name != "dataset_info.json"
        ]
        result.lerobot_conversion_success = True
    else:
        result.lerobot_conversion_success = False
        lerobot_error = "LeRobot output directory not found for local import."
        result.warnings.append(lerobot_error)

    # Write machine-readable import manifest for workflows
    import_manifest_path = config.output_dir / "import_manifest.json"
    gcs_output_path = None
    output_dir_str = str(config.output_dir)
    if output_dir_str.startswith("/mnt/gcs/"):
        gcs_output_path = "gs://" + output_dir_str[len("/mnt/gcs/"):]

    metrics = get_metrics()
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
        "filtered_episodes": [],
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
            "min_quality_score": config.min_quality_score,
            "enable_validation": config.enable_validation,
            "filter_low_quality": config.filter_low_quality,
            "require_lerobot": config.require_lerobot,
            "poll_interval": config.poll_interval,
            "wait_for_completion": config.wait_for_completion,
            "fail_on_partial_error": config.fail_on_partial_error,
            "submission_mode": config.submission_mode,
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

    import_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "schema_definition": MANIFEST_SCHEMA_DEFINITION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "job_id": config.job_id,
        "output_dir": output_dir_str,
        "gcs_output_path": gcs_output_path,
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
            "output_dir": _relative_to_bundle(bundle_root, lerobot_dir),
            "error": lerobot_error,
            "required": False,
        },
        "metrics_summary": metrics_summary,
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

    result.success = len(result.errors) == 0

    print("=" * 80)
    print("LOCAL IMPORT COMPLETE")
    print("=" * 80)
    print(f"{'✅' if result.success else '❌'} Imported {result.episodes_passed_validation} local episodes")
    print(f"Output directory: {result.output_dir}")
    print(f"Manifest: {result.import_manifest_path}")
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
    submission_mode = os.getenv("GENIESIM_SUBMISSION_MODE", "api").lower()
    job_metadata_path = os.getenv("JOB_METADATA_PATH") or None
    local_episodes_prefix = os.getenv("LOCAL_EPISODES_PREFIX") or None

    job_metadata = None
    if job_metadata_path:
        try:
            job_metadata = _load_local_job_metadata(bucket, job_metadata_path)
            submission_mode = job_metadata.get("submission_mode", submission_mode)
            artifacts = job_metadata.get("artifacts", {})
            if not local_episodes_prefix:
                local_episodes_prefix = artifacts.get("episodes_prefix")
        except FileNotFoundError as e:
            print(f"[GENIE-SIM-IMPORT] WARNING: {e}")
            if submission_mode == "local" and not local_episodes_prefix:
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
    enable_validation = os.getenv("ENABLE_VALIDATION", "true").lower() == "true"
    filter_low_quality = os.getenv("FILTER_LOW_QUALITY", "true").lower() == "true"
    require_lerobot = os.getenv("REQUIRE_LEROBOT", "false").lower() == "true"

    # Polling configuration
    poll_interval = int(os.getenv("GENIE_SIM_POLL_INTERVAL", "30"))
    wait_for_completion = os.getenv("WAIT_FOR_COMPLETION", "true").lower() == "true"

    # Error handling configuration
    fail_on_partial_error = os.getenv("FAIL_ON_PARTIAL_ERROR", "false").lower() == "true"

    # Validate credentials at startup
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        from startup_validation import validate_and_fail_fast
        validate_and_fail_fast(
            job_name="GENIE-SIM-IMPORT",
            require_geniesim=submission_mode != "local",
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
    print(f"[GENIE-SIM-IMPORT]   Submission Mode: {submission_mode}")
    print(f"[GENIE-SIM-IMPORT]   Min Quality: {min_quality_score}")
    print(
        "[GENIE-SIM-IMPORT]   Quality Range: "
        f"{QUALITY_CONFIG.min_allowed} - {QUALITY_CONFIG.max_allowed}"
    )
    print(f"[GENIE-SIM-IMPORT]   Enable Validation: {enable_validation}")
    print(f"[GENIE-SIM-IMPORT]   Require LeRobot: {require_lerobot}")
    print(f"[GENIE-SIM-IMPORT]   Wait for Completion: {wait_for_completion}")
    print(f"[GENIE-SIM-IMPORT]   Fail on Partial Error: {fail_on_partial_error}\n")

    # Setup paths
    GCS_ROOT = Path("/mnt/gcs")
    if submission_mode == "local":
        output_dir = _resolve_local_output_dir(
            bucket=bucket,
            output_prefix=output_prefix,
            job_id=job_id,
            local_episodes_prefix=local_episodes_prefix,
        )
    else:
        output_dir = GCS_ROOT / bucket / output_prefix / f"geniesim_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create configuration
    config = ImportConfig(
        job_id=job_id,
        output_dir=output_dir,
        min_quality_score=min_quality_score,
        enable_validation=enable_validation,
        filter_low_quality=filter_low_quality,
        require_lerobot=require_lerobot,
        poll_interval=poll_interval,
        wait_for_completion=wait_for_completion,
        fail_on_partial_error=fail_on_partial_error,
        submission_mode=submission_mode,
        job_metadata_path=job_metadata_path,
        local_episodes_prefix=local_episodes_prefix,
    )

    # Run import
    client = None
    try:
        metrics = get_metrics()
        with metrics.track_job("genie-sim-import-job", scene_id):
            if submission_mode == "local":
                result = run_local_import_job(config, job_metadata=job_metadata)
            else:
                try:
                    client = GenieSimClient(
                        mock_mode=resolve_geniesim_mock_mode().enabled,
                        validate_on_init=False,
                    )
                except Exception as e:
                    print(f"[GENIE-SIM-IMPORT] ERROR: Failed to create Genie Sim local client: {e}")
                    print("[GENIE-SIM-IMPORT] Make sure local Genie Sim gRPC server is running (GENIESIM_HOST:GENIESIM_PORT)")
                    sys.exit(1)
                result = run_import_job(config, client, job_metadata=job_metadata)

        if result.success:
            print(f"[GENIE-SIM-IMPORT] ✅ Import succeeded")
            print(f"[GENIE-SIM-IMPORT] Episodes imported: {result.episodes_passed_validation}")
            print(f"[GENIE-SIM-IMPORT] Average quality: {result.average_quality_score:.2f}")
            sys.exit(0)
        else:
            print(f"[GENIE-SIM-IMPORT] ❌ Import failed")
            for error in result.errors:
                print(f"[GENIE-SIM-IMPORT]   - {error}")
            sys.exit(1)

    finally:
        if client is not None:
            client.close()


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

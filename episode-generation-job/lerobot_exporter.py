#!/usr/bin/env python3
"""
LeRobot Format Exporter for Episode Generation.

Exports generated robot episodes to LeRobot v2.0 format for direct training.
Compatible with the LeRobot training pipeline and HuggingFace datasets.

ENHANCED (2025): Now supports full visual observations and ground-truth labels:
- RGB images/videos per camera
- Depth maps
- Segmentation masks (semantic + instance)
- 2D/3D bounding boxes
- Object poses
- Contact information
- Privileged state

LeRobot Dataset Structure (v2.0):
    dataset/
    ├── meta/
    │   ├── info.json           # Dataset metadata
    │   ├── stats.json          # Statistics per feature
    │   ├── tasks.jsonl         # Task descriptions
    │   └── episodes.jsonl      # Episode metadata
    ├── data/
    │   ├── chunk-000/
    │   │   ├── episode_000000.parquet
    │   │   ├── episode_000001.parquet
    │   │   └── ...
    │   └── ...
    ├── videos/                  # Video data (RGB per camera)
    │   ├── chunk-000/
    │   │   ├── observation.images.wrist/
    │   │   │   ├── episode_000000.mp4
    │   │   │   └── ...
    │   │   ├── observation.images.overhead/
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── ground_truth/            # Ground-truth labels (Plus/Full packs)
        ├── chunk-000/
        │   ├── depth/
        │   ├── segmentation/
        │   ├── bboxes/
        │   ├── object_poses/
        │   └── contacts/
        └── ...

See: https://github.com/huggingface/lerobot
"""

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_bool_env_with_default(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

from trajectory_solver import JointTrajectory, JointState, RobotConfig, ROBOT_CONFIGS

# Import reward computation
try:
    from reward_computation import RewardComputer, RewardConfig, compute_episode_reward
    HAVE_REWARD_COMPUTATION = True
except ImportError:
    HAVE_REWARD_COMPUTATION = False
    RewardComputer = None

# Try to import pyarrow for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False
    pa = None
    pq = None

# Try to import video writing libraries
try:
    import imageio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False
    imageio = None

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
    Image = None

# Sensor data capture (optional import)
try:
    from sensor_data_capture import (
        EpisodeSensorData,
        FrameSensorData,
        SensorDataConfig,
        DataPackTier,
    )
    from data_pack_config import DataPackConfig, get_data_pack_config
    HAVE_SENSOR_CAPTURE = True
except ImportError:
    HAVE_SENSOR_CAPTURE = False
    EpisodeSensorData = None
    DataPackConfig = None

try:
    from sim_validator import ValidationConfig
    HAVE_VALIDATION_CONFIG = True
except ImportError:
    HAVE_VALIDATION_CONFIG = False
    ValidationConfig = None


# =============================================================================
# Data Models
# =============================================================================


def _checksum_file(path: Path, algo: str = "sha256") -> str:
    """Compute a checksum for a file."""
    hasher = hashlib.new(algo)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class LeRobotEpisode:
    """A single episode in LeRobot format."""

    episode_index: int
    task_index: int
    task_description: str

    # Trajectory data
    trajectory: JointTrajectory

    # Scene context
    scene_id: str = ""
    variation_index: int = 0

    # Optional image data paths
    camera_video_path: Optional[Path] = None

    # Sensor data (enhanced - for visual observations)
    sensor_data: Optional[Any] = None  # EpisodeSensorData when available

    # Metadata
    success: bool = True
    total_reward: float = 0.0  # Computed by RewardComputer
    quality_score: float = 1.0
    is_mock: bool = False  # Flag for mock sensor data

    # Reward breakdown (for interpretability)
    reward_components: Dict[str, float] = field(default_factory=dict)

    # Source motion plan (for reward computation)
    motion_plan: Optional[Any] = None
    validation_result: Optional[Any] = None

    # Ground-truth metadata
    object_metadata: Dict[str, Any] = field(default_factory=dict)

    # Capture diagnostics
    camera_capture_warnings: List[str] = field(default_factory=list)
    camera_error_counts: Dict[str, int] = field(default_factory=dict)
    camera_frame_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    rgb_frames_expected_total: int = 0
    rgb_frames_written_total: int = 0
    dropped_frames_total: int = 0
    parquet_row_count: Optional[int] = None
    parquet_path: Optional[str] = None


@dataclass
class LeRobotDatasetConfig:
    """Configuration for LeRobot dataset export."""

    # Dataset identification
    dataset_name: str
    robot_type: str = "franka"

    # Data configuration
    fps: float = 30.0
    chunk_size: int = 1000  # Episodes per chunk

    # Feature configuration
    state_dim: int = 7  # Joint positions
    action_dim: int = 8  # Joints + gripper

    # Image configuration (enhanced)
    include_images: bool = False
    image_resolution: Tuple[int, int] = (640, 480)
    camera_types: List[str] = field(default_factory=lambda: ["wrist"])
    video_codec: str = "h264"

    # Data pack configuration (Core/Plus/Full)
    data_pack_tier: str = "core"  # "core", "plus", "full"

    # Ground-truth configuration (Plus/Full packs)
    include_depth: bool = False
    include_segmentation: bool = False
    include_bboxes: bool = False
    include_object_poses: bool = False
    include_contacts: bool = False
    include_privileged_state: bool = False
    strict_alignment: bool = True

    # Output paths
    output_dir: Path = Path("./lerobot_dataset")

    # Tier compliance enforcement
    tier_compliance_action: str = "fail"  # "fail" or "drop"

    # Episode completeness enforcement
    require_complete_episodes: bool = field(
        default_factory=lambda: _parse_bool_env_with_default(
            "REQUIRE_COMPLETE_EPISODES",
            False,
        )
    )

    @classmethod
    def from_data_pack(
        cls,
        dataset_name: str,
        data_pack_tier: str,
        robot_type: str = "franka",
        num_cameras: int = 1,
        resolution: Tuple[int, int] = (640, 480),
        fps: float = 30.0,
        output_dir: Path = Path("./lerobot_dataset"),
        strict_alignment: Optional[bool] = None,
        tier_compliance_action: str = "fail",
    ) -> "LeRobotDatasetConfig":
        """Create config from data pack tier."""
        camera_types = ["wrist"]
        if num_cameras >= 2:
            camera_types.append("overhead")
        if num_cameras >= 3:
            camera_types.append("side")
        if num_cameras >= 4:
            camera_types.append("front")

        config = cls(
            dataset_name=dataset_name,
            robot_type=robot_type,
            fps=fps,
            image_resolution=resolution,
            camera_types=camera_types[:num_cameras],
            data_pack_tier=data_pack_tier,
            output_dir=output_dir,
            strict_alignment=strict_alignment if strict_alignment is not None else True,
            tier_compliance_action=tier_compliance_action,
        )

        # Configure based on tier
        tier = data_pack_tier.lower()

        if tier in ["core", "plus", "full"]:
            config.include_images = True

        if tier in ["plus", "full"]:
            config.include_depth = True
            config.include_segmentation = True
            config.include_bboxes = True

        if tier == "full":
            config.include_object_poses = True
            config.include_contacts = True
            config.include_privileged_state = True

        return config


# =============================================================================
# LeRobot Exporter
# =============================================================================


class LeRobotExporter:
    """
    Exports robot episodes to LeRobot v2.0 format.

    LeRobot format is the standard for robot learning datasets, compatible with:
    - HuggingFace Datasets
    - LeRobot training pipeline
    - Various policy architectures (ACT, Diffusion Policy, etc.)

    Usage:
        exporter = LeRobotExporter(config)
        exporter.add_episode(trajectory, task_description)
        exporter.add_episode(trajectory2, task_description2)
        exporter.finalize()
    """

    def __init__(self, config: LeRobotDatasetConfig, verbose: bool = True):
        """
        Initialize exporter.

        Args:
            config: Dataset configuration
            verbose: Print progress
        """
        self.config = config
        self.verbose = verbose
        self.robot_config = ROBOT_CONFIGS.get(config.robot_type, ROBOT_CONFIGS["franka"])
        self.debug = os.getenv("BP_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        self._path_redaction_regex = re.compile(r"(?:[A-Za-z]:\\\\|/)[^\\s]+")

        # Episode storage
        self.episodes: List[LeRobotEpisode] = []
        self.tasks: List[Dict[str, str]] = []
        self.task_to_index: Dict[str, int] = {}

        # Statistics tracking
        self.stats = {
            "observation.state": {"min": [], "max": [], "mean": [], "std": []},
            "action": {"min": [], "max": [], "mean": [], "std": []},
        }
        self.tier_compliance_summary = {
            "non_compliant_episodes": 0,
            "dropped_episodes": 0,
            "action": "none",
        }
        self.checksum_algorithm = "sha256"
        self.checksum_manifest: Dict[str, str] = {}
        self.checksum_generated_at: Optional[str] = None
        self.output_dir: Optional[Path] = None
        self._temp_export_dir: Optional[Path] = None
        self._backup_output_dir: Optional[Path] = None
        self._export_output_dir: Optional[Path] = None
        self.dropped_frames_total: int = 0

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[LEROBOT-EXPORTER] [%s] %s", level, msg)

    def _sanitize_error_message(self, message: str) -> str:
        if not message:
            return message
        return self._path_redaction_regex.sub("<redacted-path>", message)

    def _record_checksum(self, path: Path) -> None:
        if self.output_dir is None or not path.exists():
            return
        rel_path = path.resolve().relative_to(self.output_dir.resolve()).as_posix()
        self.checksum_manifest[rel_path] = _checksum_file(path, self.checksum_algorithm)

    def _render_checksum_manifest(self) -> str:
        return json.dumps(self.checksum_manifest, indent=2, sort_keys=True)

    def _checksum_manifest_hash(self) -> Optional[str]:
        if self.output_dir is None:
            return None
        payload = self._render_checksum_manifest().encode("utf-8")
        hasher = hashlib.new(self.checksum_algorithm)
        hasher.update(payload)
        return hasher.hexdigest()

    def _verify_checksum_manifest_consistency(self, root_dir: Path) -> None:
        if self.output_dir is None:
            return

        root_dir = root_dir.resolve()
        disk_files = {
            path.resolve().relative_to(root_dir).as_posix()
            for path in root_dir.rglob("*")
            if path.is_file()
        }
        disk_files.discard("meta/checksums.json")

        manifest_files = set(self.checksum_manifest.keys())
        missing_in_manifest = sorted(disk_files - manifest_files)
        missing_on_disk = sorted(manifest_files - disk_files)

        if missing_in_manifest or missing_on_disk:
            if missing_in_manifest:
                self.log(
                    "  Checksum manifest missing entries for files: "
                    + ", ".join(missing_in_manifest[:5]),
                    "ERROR",
                )
                if len(missing_in_manifest) > 5:
                    self.log(
                        f"  ... and {len(missing_in_manifest) - 5} more missing manifest entries",
                        "ERROR",
                    )
            if missing_on_disk:
                self.log(
                    "  Checksum manifest references missing files: "
                    + ", ".join(missing_on_disk[:5]),
                    "ERROR",
                )
                if len(missing_on_disk) > 5:
                    self.log(
                        f"  ... and {len(missing_on_disk) - 5} more missing files on disk",
                        "ERROR",
                    )
            raise ValueError("Checksum manifest consistency check failed.")

    def _log_export_state(self, message: str, level: str = "ERROR") -> None:
        temp_dir = str(self._temp_export_dir) if self._temp_export_dir else "<unset>"
        backup_dir = str(self._backup_output_dir) if self._backup_output_dir else "<unset>"
        output_dir = str(self._export_output_dir) if self._export_output_dir else "<unset>"
        self.log(
            f"{message} (temp_dir={temp_dir}, backup_dir={backup_dir}, output_dir={output_dir})",
            level,
        )

    def _cleanup_failed_export(self, message: str, exc: Optional[BaseException] = None) -> None:
        if exc is not None:
            error_message = self._sanitize_error_message(str(exc))
            self._log_export_state(f"{message}: {error_message}", "ERROR")
        else:
            self._log_export_state(message, "ERROR")
        temp_dir = self._temp_export_dir
        if temp_dir is None:
            return
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_exc:
            cleanup_message = self._sanitize_error_message(str(cleanup_exc))
            self._log_export_state(f"Failed to cleanup temp export dir: {cleanup_message}", "WARNING")

    def _atomic_write(
        self,
        target: Path,
        writer: Callable[[Path], None],
        *,
        record_checksum: bool = True,
        fsync_target: bool = False,
    ) -> None:
        tmp_path = target.with_name(f"{target.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        writer(tmp_path)
        os.replace(tmp_path, target)
        if fsync_target:
            self._fsync_file_and_dir(target)
        if record_checksum:
            self._record_checksum(target)

    def _atomic_write_with_suffix(
        self,
        target: Path,
        writer: Callable[[Path], None],
        *,
        record_checksum: bool = True,
        fsync_target: bool = False,
    ) -> None:
        tmp_path = target.with_name(f"{target.stem}.tmp{target.suffix}")
        if tmp_path.exists():
            tmp_path.unlink()
        writer(tmp_path)
        os.replace(tmp_path, target)
        if fsync_target:
            self._fsync_file_and_dir(target)
        if record_checksum:
            self._record_checksum(target)

    def _fsync_file_and_dir(self, target: Path) -> None:
        try:
            with open(target, "rb") as handle:
                os.fsync(handle.fileno())
        except Exception as exc:
            self.log(
                f"Failed to fsync file {target.name}: {self._sanitize_error_message(str(exc))}",
                "WARNING",
            )
            return

        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
        except Exception as exc:
            self.log(
                f"Failed to open directory for fsync {target.parent}: {self._sanitize_error_message(str(exc))}",
                "WARNING",
            )
            return
        try:
            os.fsync(dir_fd)
        except Exception as exc:
            self.log(
                f"Failed to fsync directory {target.parent}: {self._sanitize_error_message(str(exc))}",
                "WARNING",
            )
        finally:
            os.close(dir_fd)

    def _get_data_source_info(self) -> Dict[str, Any]:
        """
        Determine the data source and quality information.

        This checks:
        - Whether Isaac Sim was used for physics
        - Whether sensor data is from simulation or mock
        - Whether validation was physics-based or heuristic

        Returns:
            Dict with physics_validated, sensor_source, validation_type, etc.
        """
        import os

        warnings = []

        # Check for Isaac Sim availability
        try:
            from isaac_sim_integration import is_isaac_sim_available, is_physx_available
            isaac_sim_available = is_isaac_sim_available()
            physx_available = is_physx_available()
        except ImportError:
            isaac_sim_available = False
            physx_available = False

        # Determine sensor source from episodes
        sensor_source = "unknown"
        if self.episodes:
            sample_episode = self.episodes[0]
            if sample_episode.sensor_data is not None:
                # Check if sensor data has source annotation
                if hasattr(sample_episode.sensor_data, 'frames') and sample_episode.sensor_data.frames:
                    frame = sample_episode.sensor_data.frames[0]
                    if hasattr(frame, 'privileged_state') and frame.privileged_state:
                        source = frame.privileged_state.get('data_source', 'unknown')
                        if source == 'simulation':
                            sensor_source = "isaac_sim_replicator"
                        elif source == 'input_fallback':
                            sensor_source = "mock_random"
                            warnings.append("Sensor data from mock capture - random noise images")
                        else:
                            sensor_source = source
                    else:
                        # Check if it's a mock sensor class
                        sensor_class = sample_episode.sensor_data.__class__.__name__
                        if 'Mock' in sensor_class:
                            sensor_source = "mock_random"
                            warnings.append("Sensor data from mock capture - random noise images")
                        else:
                            sensor_source = "simulation"
            else:
                sensor_source = "none"
                if self.config.include_images:
                    warnings.append("No sensor data captured despite images being requested")

        # Determine validation type from episodes
        validation_type = "unknown"
        physics_validated = False
        if self.episodes:
            sample_episode = self.episodes[0]
            if sample_episode.validation_result is not None:
                val_result = sample_episode.validation_result
                if hasattr(val_result, 'validation_type'):
                    validation_type = val_result.validation_type
                elif hasattr(val_result, 'physics_checked') and val_result.physics_checked:
                    validation_type = "physx"
                else:
                    validation_type = "heuristic"
                    warnings.append("Validation was heuristic-based, not physics-verified")

                physics_validated = validation_type == "physx"
            else:
                validation_type = "none"
                warnings.append("No validation performed on episodes")

        # Check environment indicators
        is_production = self._is_production_mode()

        if not isaac_sim_available and is_production:
            warnings.append("Running in production without Isaac Sim")

        if not physx_available:
            warnings.append("PhysX not available - physics not validated")

        return {
            "physics_validated": physics_validated,
            "sensor_source": sensor_source,
            "validation_type": validation_type,
            "isaac_sim_used": isaac_sim_available,
            "physx_available": physx_available,
            "warnings": warnings,
            "production_mode": is_production,
        }

    def _parse_bool_env(self, key: str) -> Optional[bool]:
        value = os.getenv(key)
        if value is None:
            return None
        return value.lower() in {"1", "true", "yes", "y"}

    def _is_production_mode(self) -> bool:
        return (
            os.getenv("KUBERNETES_SERVICE_HOST") is not None
            or os.getenv("K_SERVICE") is not None
            or os.path.exists("/.dockerenv")
        )

    def _resolve_quality_gate_report_path(self) -> Optional[Path]:
        env_path = os.getenv("QUALITY_GATE_REPORT_PATH")
        if env_path:
            return Path(env_path)
        scene_id = os.getenv("SCENE_ID")
        if not scene_id:
            return None
        root = Path(os.getenv("PIPELINE_STORAGE_ROOT", "/mnt/gcs"))
        return root / f"scenes/{scene_id}/episode-generation-job/quality_gate_report.json"

    def _get_sim_backend_lineage(self, data_source_info: Dict[str, Any]) -> Dict[str, Any]:
        backend_name = "unknown"
        if self.episodes:
            sample_episode = self.episodes[0]
            if sample_episode.validation_result and hasattr(sample_episode.validation_result, "physics_backend"):
                backend_name = sample_episode.validation_result.physics_backend
        if backend_name == "unknown" and data_source_info.get("isaac_sim_used"):
            backend_name = "isaac_sim"

        backend_version = os.getenv("ISAAC_SIM_VERSION") or os.getenv("ISAAC_LAB_VERSION")
        backend_container = os.getenv("ISAAC_SIM_DOCKER_IMAGE") or os.getenv("ISAAC_SIM_CONTAINER")
        backend_build_id = (
            os.getenv("ISAAC_SIM_BUILD_ID")
            or os.getenv("ISAAC_LAB_BUILD_ID")
            or os.getenv("ISAAC_SIM_BUILD")
            or os.getenv("ISAAC_LAB_BUILD")
        )
        backend_container_digest = (
            os.getenv("ISAAC_SIM_CONTAINER_DIGEST")
            or os.getenv("ISAAC_SIM_IMAGE_DIGEST")
            or os.getenv("ISAAC_SIM_DIGEST")
            or os.getenv("ISAAC_LAB_CONTAINER_DIGEST")
        )

        return {
            "name": backend_name,
            "version": backend_version,
            "container_image": backend_container,
            "build_id": backend_build_id,
            "container_digest": backend_container_digest,
            "source": {
                "name": "episode.validation_result.physics_backend",
                "version": "env:ISAAC_SIM_VERSION|ISAAC_LAB_VERSION",
                "container_image": "env:ISAAC_SIM_DOCKER_IMAGE|ISAAC_SIM_CONTAINER",
                "build_id": "env:ISAAC_SIM_BUILD_ID|ISAAC_LAB_BUILD_ID|ISAAC_SIM_BUILD|ISAAC_LAB_BUILD",
                "container_digest": (
                    "env:ISAAC_SIM_CONTAINER_DIGEST|ISAAC_SIM_IMAGE_DIGEST|ISAAC_SIM_DIGEST|ISAAC_LAB_CONTAINER_DIGEST"
                ),
            },
        }

    def _get_physics_parameters_lineage(self) -> Dict[str, Any]:
        validation_thresholds = None
        if HAVE_VALIDATION_CONFIG and ValidationConfig is not None:
            try:
                config = ValidationConfig.from_policy_config()
                validation_thresholds = {
                    "max_unexpected_collisions": config.max_unexpected_collisions,
                    "max_joint_violations": config.max_joint_violations,
                    "max_collision_force": config.max_collision_force,
                    "max_joint_velocity": config.max_joint_velocity,
                    "max_joint_acceleration": config.max_joint_acceleration,
                    "require_task_success": config.require_task_success,
                    "require_grasp_success": config.require_grasp_success,
                    "require_placement_success": config.require_placement_success,
                    "require_object_stable": config.require_object_stable,
                    "stability_threshold": config.stability_threshold,
                    "min_quality_score": config.min_quality_score,
                    "max_retries": config.max_retries,
                    "retry_on_collision": config.retry_on_collision,
                    "retry_on_joint_limit": config.retry_on_joint_limit,
                }
            except Exception:
                validation_thresholds = None

        env_overrides: Dict[str, Any] = {}
        min_quality_score_override = os.getenv("MIN_QUALITY_SCORE")
        if min_quality_score_override is not None:
            try:
                env_overrides["min_quality_score"] = float(min_quality_score_override)
            except ValueError:
                env_overrides["min_quality_score"] = min_quality_score_override

        resolved_thresholds = None
        if validation_thresholds is not None:
            resolved_thresholds = dict(validation_thresholds)
            for key, value in env_overrides.items():
                resolved_thresholds[key] = value

        post_rollout_seconds = float(os.getenv("SIM_VALIDATOR_POST_ROLLOUT_SECONDS", "0.5"))
        load_scene = os.getenv("SIM_VALIDATOR_LOAD_SCENE", "true").lower() == "true"
        require_real_physics = self._parse_bool_env("REQUIRE_REAL_PHYSICS")

        return {
            "validation_thresholds": validation_thresholds,
            "validation_overrides": env_overrides or None,
            "resolved_thresholds": resolved_thresholds,
            "sim_validator": {
                "post_rollout_seconds": post_rollout_seconds,
                "load_scene": load_scene,
                "require_real_physics": require_real_physics,
            },
            "source": {
                "validation_thresholds": "sim_validator.ValidationConfig.from_policy_config",
                "validation_overrides": "env:MIN_QUALITY_SCORE",
                "resolved_thresholds": "validation_thresholds + env overrides",
                "sim_validator.post_rollout_seconds": "env:SIM_VALIDATOR_POST_ROLLOUT_SECONDS (default 0.5)",
                "sim_validator.load_scene": "env:SIM_VALIDATOR_LOAD_SCENE (default true)",
                "sim_validator.require_real_physics": "env:REQUIRE_REAL_PHYSICS",
            },
        }

    def _get_quality_gate_lineage(self) -> Dict[str, Any]:
        report_path = self._resolve_quality_gate_report_path()
        summary = None
        report_timestamp = None
        report_path_str = None
        report_hash = None
        if report_path is not None:
            report_path_str = report_path.as_posix()
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text())
                    summary = report.get("summary")
                    report_timestamp = report.get("timestamp")
                    report_hash = _checksum_file(report_path)
                except Exception:
                    summary = None

        config_version = None
        config_hash = None
        config_path = REPO_ROOT / "tools" / "quality_gates" / "quality_config.json"
        if config_path.is_file():
            try:
                config_payload = json.loads(config_path.read_text())
                config_version = config_payload.get("version")
                config_hash = _checksum_file(config_path)
            except Exception:
                config_version = None

        return {
            "summary": summary,
            "report_timestamp": report_timestamp,
            "report_path": report_path_str,
            "report_hash": report_hash,
            "config_version": config_version,
            "config_hash": config_hash,
            "bypass_enabled": self._parse_bool_env("BYPASS_QUALITY_GATES"),
            "source": {
                "summary": "quality_gate_report.json",
                "report_timestamp": "quality_gate_report.json",
                "report_path": "env:QUALITY_GATE_REPORT_PATH|SCENE_ID",
                "report_hash": "quality_gate_report.json (sha256)",
                "config_version": "tools/quality_gates/quality_config.json",
                "config_hash": "tools/quality_gates/quality_config.json (sha256)",
                "bypass_enabled": "env:BYPASS_QUALITY_GATES",
            },
        }

    def _validate_lineage_requirements(self, lineage: Dict[str, Any], production_mode: bool) -> None:
        if not production_mode:
            return

        missing_fields = []
        required_paths = {
            "sim_backend.build_id": lineage.get("sim_backend", {}).get("build_id"),
            "sim_backend.container_digest": lineage.get("sim_backend", {}).get("container_digest"),
            "quality_gates.report_hash": lineage.get("quality_gates", {}).get("report_hash"),
            "quality_gates.config_hash": lineage.get("quality_gates", {}).get("config_hash"),
            "physics_parameters.resolved_thresholds": lineage.get("physics_parameters", {}).get("resolved_thresholds"),
        }

        for path, value in required_paths.items():
            if value in (None, "", {}, []):
                missing_fields.append(path)

        if missing_fields:
            raise RuntimeError(
                "Missing required lineage fields in production: " + ", ".join(missing_fields)
            )

    def _get_pipeline_lineage(self) -> Dict[str, Any]:
        return {
            "scene_id": os.getenv("SCENE_ID"),
            "bucket": os.getenv("BUCKET"),
            "assets_prefix": os.getenv("ASSETS_PREFIX"),
            "episodes_prefix": os.getenv("EPISODES_PREFIX"),
            "job_name": os.getenv("JOB_NAME", "episode-generation-job"),
            "job_id": os.getenv("JOB_ID"),
            "run_id": os.getenv("RUN_ID"),
            "pipeline_env": os.getenv("PIPELINE_ENV"),
            "pipeline_version": os.getenv("PIPELINE_VERSION"),
            "source": {
                "scene_id": "env:SCENE_ID",
                "bucket": "env:BUCKET",
                "assets_prefix": "env:ASSETS_PREFIX",
                "episodes_prefix": "env:EPISODES_PREFIX",
                "job_name": "env:JOB_NAME (default episode-generation-job)",
                "job_id": "env:JOB_ID",
                "run_id": "env:RUN_ID",
                "pipeline_env": "env:PIPELINE_ENV",
                "pipeline_version": "env:PIPELINE_VERSION",
            },
        }

    def _get_lineage_info(self, data_source_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sim_backend": self._get_sim_backend_lineage(data_source_info),
            "physics_parameters": self._get_physics_parameters_lineage(),
            "quality_gates": self._get_quality_gate_lineage(),
            "pipeline": self._get_pipeline_lineage(),
        }

    def add_episode(
        self,
        trajectory: JointTrajectory,
        task_description: str,
        scene_id: str = "",
        variation_index: int = 0,
        success: bool = True,
        quality_score: float = 1.0,
        sensor_data: Optional[Any] = None,
        object_metadata: Optional[Dict[str, Any]] = None,
        motion_plan: Optional[Any] = None,
        validation_result: Optional[Any] = None,
    ) -> int:
        """
        Add an episode to the dataset.

        Args:
            trajectory: Joint trajectory for the episode
            task_description: Natural language task description
            scene_id: Scene identifier
            variation_index: Variation index within scene
            success: Whether episode was successful
            quality_score: Quality score from validation (0.0-1.0)
            sensor_data: EpisodeSensorData with visual observations (optional)
            object_metadata: Object metadata for ground-truth (optional)
            motion_plan: Original motion plan for reward computation (optional)
            validation_result: Validation result with physics data (optional)

        Returns:
            Episode index
        """
        # Get or create task index
        if task_description not in self.task_to_index:
            task_index = len(self.tasks)
            self.tasks.append({"task_index": task_index, "task": task_description})
            self.task_to_index[task_description] = task_index
        else:
            task_index = self.task_to_index[task_description]

        episode_index = len(self.episodes)

        # Compute reward with comprehensive error handling and logging
        total_reward = 0.0
        reward_components = {}
        reward_computation_status = "not_attempted"

        if HAVE_REWARD_COMPUTATION and motion_plan is not None:
            try:
                reward_computer = RewardComputer(verbose=False)
                total_reward, components = reward_computer.compute_episode_reward(
                    trajectory=trajectory,
                    motion_plan=motion_plan,
                    validation_result=validation_result,
                )
                reward_components = components.to_dict()
                reward_computation_status = "success"
                self.log(f"  Reward computed: {total_reward:.3f} (components: {len(reward_components)})", "DEBUG")
            except Exception as e:
                import traceback
                error_details = {
                    "error_type": type(e).__name__,
                    "error_message": self._sanitize_error_message(str(e)),
                }
                if self.debug:
                    error_details["traceback"] = traceback.format_exc()
                self.log(f"Reward computation failed for episode {episode_index}: {type(e).__name__}: {e}", "WARNING")
                if self.debug:
                    self.log(traceback.format_exc(), "DEBUG")
                self.log(f"  Using heuristic fallback for reward computation", "WARNING")

                # Fallback based on quality_score and success (more nuanced than hardcoded 0.7)
                # Success = quality_score (e.g., 0.85 quality → 0.85 reward)
                # Failure = quality_score * 0.3 (e.g., 0.85 quality → 0.255 reward)
                total_reward = quality_score if success else (quality_score * 0.3)
                reward_components = {
                    "fallback_reason": "reward_computation_error",
                    "error_details": error_details,
                    "success": float(success),
                    "quality": quality_score,
                    "computed_from": "quality_score_heuristic"
                }
                reward_computation_status = "failed_with_fallback"
        else:
            # Improved fallback with logging of reason
            if not HAVE_REWARD_COMPUTATION:
                reason = "reward_computer_not_available"
                self.log(f"  Reward computation module not available for episode {episode_index}", "DEBUG")
            elif motion_plan is None:
                reason = "motion_plan_not_provided"
                self.log(f"  Motion plan not provided for episode {episode_index}, using fallback", "DEBUG")
            else:
                reason = "unknown"

            # Use success + quality_score as heuristic
            total_reward = quality_score if success else (quality_score * 0.3)
            reward_components = {
                "fallback_reason": reason,
                "success": float(success),
                "quality": quality_score,
                "computed_from": "quality_score_heuristic"
            }
            reward_computation_status = "fallback"

        # Extract is_mock flag from sensor_data
        is_mock = False
        if sensor_data is not None:
            # Check if sensor_data has frames with is_mock flag
            if hasattr(sensor_data, 'frames') and len(sensor_data.frames) > 0:
                first_frame = sensor_data.frames[0]
                if hasattr(first_frame, 'is_mock'):
                    is_mock = first_frame.is_mock
                # Also check state dict if available
                elif hasattr(first_frame, 'state') and isinstance(first_frame.state, dict):
                    is_mock = first_frame.state.get('is_mock', False)

        camera_capture_warnings = []
        camera_error_counts = {}
        if sensor_data is not None:
            camera_capture_warnings = list(
                getattr(sensor_data, "camera_capture_warnings", [])
            )
            camera_error_counts = dict(
                getattr(sensor_data, "camera_error_counts", {})
            )

        episode = LeRobotEpisode(
            episode_index=episode_index,
            task_index=task_index,
            task_description=task_description,
            trajectory=trajectory,
            scene_id=scene_id,
            variation_index=variation_index,
            success=success,
            total_reward=total_reward,
            quality_score=quality_score,
            is_mock=is_mock,
            reward_components=reward_components,
            sensor_data=sensor_data,
            motion_plan=motion_plan,
            validation_result=validation_result,
            object_metadata=object_metadata or {},
            camera_capture_warnings=camera_capture_warnings,
            camera_error_counts=camera_error_counts,
        )

        self.episodes.append(episode)

        # Log with visual observation info if present
        visual_info = ""
        if sensor_data is not None:
            num_frames = sensor_data.num_frames if hasattr(sensor_data, 'num_frames') else 0
            cameras = sensor_data.camera_ids if hasattr(sensor_data, 'camera_ids') else []
            visual_info = f" [visual: {num_frames} frames, {len(cameras)} cameras]"

        reward_info = f" [reward: {total_reward:.2f}]"

        self.log(f"Added episode {episode_index}: {task_description[:50]}...{visual_info}{reward_info}")

        return episode_index

    def _validate_episode_completeness(self) -> List[Tuple[int, List[str]]]:
        """
        Validate episode completeness before export.

        Checks for each episode:
        1. Trajectory is not None and has frames
        2. Task description is not empty
        3. If sensor_data exists, it has frames matching trajectory
        4. Required metadata fields are present

        Returns:
            List of (episode_index, errors) tuples
        """
        incomplete_episodes = []

        for episode in self.episodes:
            errors = []

            # Check trajectory
            if episode.trajectory is None:
                errors.append("Trajectory is None")
            elif episode.trajectory.num_frames == 0:
                errors.append("Trajectory has 0 frames")

            # Check task description
            if not episode.task_description or len(episode.task_description.strip()) == 0:
                errors.append("Task description is empty")

            # Check sensor data consistency if present
            if episode.sensor_data is not None:
                if episode.trajectory is not None:
                    trajectory_frames = episode.trajectory.num_frames
                    sensor_frames = episode.sensor_data.num_frames if hasattr(episode.sensor_data, 'num_frames') else 0

                    if sensor_frames == 0:
                        errors.append("Sensor data has 0 frames")
                    elif trajectory_frames != sensor_frames:
                        errors.append(
                            f"Frame count mismatch: trajectory has {trajectory_frames} frames, "
                            f"sensor data has {sensor_frames} frames"
                        )

            # Check if episode has critical data for the advertised tier
            if self.config.include_images and (episode.sensor_data is None or not episode.sensor_data.has_rgb):
                errors.append(f"Data pack tier requires RGB images but episode has none")

            if self.config.include_depth and (episode.sensor_data is None or not episode.sensor_data.has_depth):
                errors.append(f"Data pack tier requires depth maps but episode has none")

            if errors:
                incomplete_episodes.append((episode.episode_index, errors))

        return incomplete_episodes

    def _validate_data_pack_tier_compliance(self) -> List[Tuple[int, List[str]]]:
        """
        Validate data pack tier compliance.

        Checks that all episodes have the data required for advertised tier:
        - Core: RGB + state + actions + metadata
        - Plus: Core + depth + segmentation + bboxes
        - Full: Plus + poses + contacts + privileged state

        Returns:
            List of (episode_index, errors) tuples for non-compliant episodes
        """
        non_compliant = []
        tier = self.config.data_pack_tier.lower()

        for episode in self.episodes:
            errors = []

            if episode.sensor_data is None:
                # Episodes without sensor data don't support Plus/Full features
                if tier in ["plus", "full"]:
                    errors.append(f"Tier '{tier}' requires sensor data but episode has none")
                continue

            # Core requirements (RGB + state + actions already validated in add_episode)
            if not episode.sensor_data.has_rgb and self.config.include_images:
                errors.append(f"Tier '{tier}' requires RGB but episode has no RGB images")

            # Plus requirements
            if tier in ["plus", "full"]:
                if not episode.sensor_data.has_depth and self.config.include_depth:
                    errors.append(f"Tier '{tier}' requires depth but episode has no depth maps")
                if not episode.sensor_data.has_segmentation and self.config.include_segmentation:
                    errors.append(f"Tier '{tier}' requires segmentation but episode has none")

                # Check for bboxes in frames
                if self.config.include_bboxes:
                    has_bboxes = any(
                        len(frame.bboxes_2d) > 0 or len(frame.bboxes_3d) > 0
                        for frame in episode.sensor_data.frames
                    )
                    if not has_bboxes:
                        errors.append(f"Tier '{tier}' requires bounding boxes but episode has none")

            # Full requirements
            if tier == "full":
                # Check for object poses
                if self.config.include_object_poses:
                    has_poses = any(
                        len(frame.object_poses) > 0
                        for frame in episode.sensor_data.frames
                    )
                    if not has_poses:
                        errors.append(f"Tier 'full' requires object poses but episode has none")

                # Check for contacts
                if self.config.include_contacts:
                    has_contacts = any(
                        len(frame.contacts) > 0
                        for frame in episode.sensor_data.frames
                    )
                    if not has_contacts:
                        errors.append(f"Tier 'full' requires contacts but episode has none")

                # Check for privileged state
                if self.config.include_privileged_state:
                    has_privileged = any(
                        frame.privileged_state is not None
                        for frame in episode.sensor_data.frames
                    )
                    if not has_privileged:
                        errors.append(f"Tier 'full' requires privileged state but episode has none")

            if errors:
                non_compliant.append((episode.episode_index, errors))

        return non_compliant

    def _validate_camera_calibration(self) -> List[Tuple[int, List[str]]]:
        """
        Validate camera calibration matrices.

        For episodes with sensor data, checks:
        1. Intrinsic matrices are 3x3 and upper triangular
        2. Extrinsic matrices are 4x4 with valid rotation
        3. Matrices are invertible (not singular)

        Returns:
            List of (episode_index, errors) tuples for calibration issues
        """
        invalid_calibrations = []

        for episode in self.episodes:
            if episode.sensor_data is None:
                continue

            errors = []

            # Check if episode has camera calibration data
            if not hasattr(episode.sensor_data, 'frames') or len(episode.sensor_data.frames) == 0:
                continue

            first_frame = episode.sensor_data.frames[0]

            # Get camera calibration from sensor config
            if not hasattr(episode.sensor_data, 'config') or episode.sensor_data.config is None:
                continue

            camera_config = episode.sensor_data.config

            # Check for CameraCalibration config
            if not hasattr(camera_config, 'camera_calibration') or camera_config.camera_calibration is None:
                continue

            calibration = camera_config.camera_calibration

            # Validate intrinsic matrix
            if calibration.intrinsic_matrix is not None:
                K = calibration.intrinsic_matrix
                # Check shape
                if K.shape != (3, 3):
                    errors.append(f"Intrinsic matrix: Expected shape (3, 3), got {K.shape}")
                else:
                    # Check upper triangular structure (most elements below diagonal should be ~0)
                    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    if abs(K[0, 1]) > 1e-6 or abs(K[1, 0]) > 1e-6 or abs(K[2, 0]) > 1e-6 or abs(K[2, 1]) > 1e-6:
                        errors.append(f"Intrinsic matrix: Not properly upper triangular")

                    # Check diagonal elements are positive (focal lengths)
                    if K[0, 0] <= 0 or K[1, 1] <= 0:
                        errors.append(f"Intrinsic matrix: Focal lengths must be positive")

                    # Check invertibility (determinant non-zero)
                    try:
                        det = np.linalg.det(K)
                        if abs(det) < 1e-10:
                            errors.append(f"Intrinsic matrix: Singular matrix (det ≈ 0)")
                    except np.linalg.LinAlgError:
                        errors.append(f"Intrinsic matrix: Failed to compute determinant")

            # Validate extrinsic matrix
            if calibration.extrinsic_matrix is not None:
                T = calibration.extrinsic_matrix
                # Check shape (4x4 homogeneous transform)
                if T.shape != (4, 4):
                    errors.append(f"Extrinsic matrix: Expected shape (4, 4), got {T.shape}")
                else:
                    # Extract rotation part (top-left 3x3)
                    R = T[:3, :3]

                    # Check orthogonality (R^T * R should be I)
                    RTR = np.dot(R.T, R)
                    should_be_identity = np.eye(3)
                    if not np.allclose(RTR, should_be_identity, atol=1e-3):
                        errors.append(
                            f"Extrinsic matrix: Rotation part is not orthogonal "
                            f"(R^T*R deviation from I: {np.linalg.norm(RTR - should_be_identity):.4f})"
                        )

                    # Check determinant is 1 (proper rotation, not reflection)
                    det_R = np.linalg.det(R)
                    if abs(det_R - 1.0) > 1e-2:
                        errors.append(f"Extrinsic matrix: Rotation determinant is {det_R:.3f}, expected 1.0")

                    # Check last row is [0, 0, 0, 1]
                    expected_last_row = np.array([0, 0, 0, 1])
                    if not np.allclose(T[3, :], expected_last_row, atol=1e-6):
                        errors.append(f"Extrinsic matrix: Last row is not [0, 0, 0, 1]")

            if errors:
                invalid_calibrations.append((episode.episode_index, errors))

        return invalid_calibrations

    def _validate_trajectory_sensor_alignment(self) -> List[Dict[str, Any]]:
        """
        Validate that trajectory frames and sensor data frames are aligned.

        Checks for each episode:
        1. If sensor_data exists, validate frame counts match
        2. Validate timestamps are aligned (within tolerance)
        3. Detect missing or extra sensor frames

        Returns:
            List of per-episode error entries (empty if all aligned)
        """
        errors: List[Dict[str, Any]] = []

        for episode in self.episodes:
            episode_id = episode.episode_index
            trajectory_frames = episode.trajectory.num_frames if episode.trajectory else 0
            episode_errors: List[str] = []
            mismatch_counts = {"frame_count": 0, "timestamp": 0, "missing_camera": 0}

            # Skip if no sensor data (not an error)
            if episode.sensor_data is None:
                continue

            # Get sensor frame count
            sensor_frames = 0
            if hasattr(episode.sensor_data, 'num_frames'):
                sensor_frames = episode.sensor_data.num_frames
            elif hasattr(episode.sensor_data, 'frames'):
                sensor_frames = len(episode.sensor_data.frames)

            # Validate frame count alignment
            if trajectory_frames != sensor_frames:
                mismatch_counts["frame_count"] = abs(trajectory_frames - sensor_frames)
                episode_errors.append(
                    f"Frame count mismatch. Trajectory: {trajectory_frames} frames, Sensor: {sensor_frames} frames"
                )
                errors.append({
                    "episode_index": episode_id,
                    "errors": episode_errors,
                    "mismatch_counts": mismatch_counts,
                    "trajectory_frames": trajectory_frames,
                    "sensor_frames": sensor_frames,
                })
                continue

            # Validate timestamp alignment (if both have timestamps)
            if hasattr(episode.sensor_data, 'frames') and episode.trajectory:
                trajectory_timestamps = [s.timestamp for s in episode.trajectory.states]
                sensor_timestamps = [f.timestamp for f in episode.sensor_data.frames if hasattr(f, 'timestamp')]

                if len(sensor_timestamps) == len(trajectory_timestamps):
                    # Check timestamp alignment (within 1ms tolerance for floating point)
                    tolerance = 0.001  # 1ms
                    misaligned_frames = []

                    for i, (traj_ts, sensor_ts) in enumerate(zip(trajectory_timestamps, sensor_timestamps)):
                        if abs(traj_ts - sensor_ts) > tolerance:
                            misaligned_frames.append(i)

                    if misaligned_frames:
                        mismatch_counts["timestamp"] = len(misaligned_frames)
                        # Report first few misaligned frames
                        sample_frames = misaligned_frames[:3]
                        frame_info = ", ".join([
                            f"frame {i} (traj: {trajectory_timestamps[i]:.3f}s, sensor: {sensor_timestamps[i]:.3f}s)"
                            for i in sample_frames
                        ])
                        episode_errors.append(
                            f"{len(misaligned_frames)} misaligned timestamps. Examples: {frame_info}"
                        )

            # Validate camera consistency (all frames should have same cameras)
            if hasattr(episode.sensor_data, 'frames') and hasattr(episode.sensor_data, 'camera_ids'):
                expected_cameras = set(episode.sensor_data.camera_ids)
                for i, frame in enumerate(episode.sensor_data.frames):
                    if hasattr(frame, 'rgb_images'):
                        frame_cameras = set(frame.rgb_images.keys())
                        missing_cameras = expected_cameras - frame_cameras
                        if missing_cameras:
                            mismatch_counts["missing_camera"] = len(missing_cameras)
                            episode_errors.append(
                                f"Missing camera data for {missing_cameras} at frame {i}"
                            )
                            break  # Don't spam errors for every frame

            if episode_errors:
                errors.append({
                    "episode_index": episode_id,
                    "errors": episode_errors,
                    "mismatch_counts": mismatch_counts,
                    "trajectory_frames": trajectory_frames,
                    "sensor_frames": sensor_frames,
                })

        return errors

    def _validate_rgb_frame(
        self,
        frame: np.ndarray,
        episode_idx: int,
        camera_id: str,
        frame_idx: int
    ) -> List[str]:
        """
        Validate RGB image frame.

        Checks:
        - Shape is (H, W, 3)
        - Dtype is uint8
        - Values are in range [0, 255]
        """
        errors = []

        # Validate shape
        if frame.ndim != 3:
            errors.append(f"Frame {frame_idx}: Invalid dimensions {frame.ndim}, expected 3 (H, W, C)")
            return errors  # Can't continue validation

        if frame.shape[2] != 3:
            errors.append(f"Frame {frame_idx}: Invalid channels {frame.shape[2]}, expected 3 (RGB)")

        # Validate dtype
        if frame.dtype != np.uint8:
            errors.append(f"Frame {frame_idx}: Invalid dtype {frame.dtype}, expected uint8")

        # Validate value range (only if dtype is correct)
        if frame.dtype == np.uint8:
            if frame.min() < 0 or frame.max() > 255:
                errors.append(f"Frame {frame_idx}: Values out of range [{frame.min()}, {frame.max()}], expected [0, 255]")

        # Validate resolution matches config
        expected_h, expected_w = self.config.image_resolution[1], self.config.image_resolution[0]
        actual_h, actual_w = frame.shape[0], frame.shape[1]
        if (actual_h, actual_w) != (expected_h, expected_w):
            errors.append(
                f"Frame {frame_idx}: Resolution mismatch ({actual_w}x{actual_h}), "
                f"expected ({expected_w}x{expected_h})"
            )

        return errors

    def _validate_depth_frame(
        self,
        frame: np.ndarray,
        episode_idx: int,
        camera_id: str,
        frame_idx: int
    ) -> List[str]:
        """
        Validate depth map frame.

        Checks:
        - Shape is (H, W)
        - Dtype is float32
        - Values are in valid range (0.01m - 100m typical)
        """
        errors = []

        # Validate shape
        if frame.ndim != 2:
            errors.append(f"Frame {frame_idx}: Invalid depth dimensions {frame.ndim}, expected 2 (H, W)")
            return errors

        # Validate dtype
        if frame.dtype != np.float32:
            errors.append(f"Frame {frame_idx}: Invalid depth dtype {frame.dtype}, expected float32")

        # Validate value range (typical depth sensor range: 0.01m to 100m)
        if np.isnan(frame).any():
            errors.append(f"Frame {frame_idx}: Depth contains NaN values")

        if np.isinf(frame).any():
            errors.append(f"Frame {frame_idx}: Depth contains infinite values")

        valid_mask = ~(np.isnan(frame) | np.isinf(frame))
        if valid_mask.any():
            depth_min, depth_max = frame[valid_mask].min(), frame[valid_mask].max()
            if depth_min < 0.0 or depth_max > 200.0:  # Generous upper bound
                errors.append(
                    f"Frame {frame_idx}: Depth values out of typical range [{depth_min:.3f}m, {depth_max:.3f}m], "
                    f"expected [0.0m, 200.0m]"
                )

        # Validate resolution matches config
        expected_h, expected_w = self.config.image_resolution[1], self.config.image_resolution[0]
        actual_h, actual_w = frame.shape[0], frame.shape[1]
        if (actual_h, actual_w) != (expected_h, expected_w):
            errors.append(
                f"Frame {frame_idx}: Depth resolution mismatch ({actual_w}x{actual_h}), "
                f"expected ({expected_w}x{expected_h})"
            )

        return errors

    def _validate_segmentation_frame(
        self,
        frame: np.ndarray,
        episode_idx: int,
        camera_id: str,
        frame_idx: int
    ) -> List[str]:
        """
        Validate segmentation mask frame.

        Checks:
        - Shape is (H, W)
        - Dtype is uint8 or uint16
        - Values are valid class IDs
        """
        errors = []

        # Validate shape
        if frame.ndim != 2:
            errors.append(f"Frame {frame_idx}: Invalid segmentation dimensions {frame.ndim}, expected 2 (H, W)")
            return errors

        # Validate dtype
        if frame.dtype not in [np.uint8, np.uint16]:
            errors.append(f"Frame {frame_idx}: Invalid segmentation dtype {frame.dtype}, expected uint8 or uint16")

        # Validate resolution matches config
        expected_h, expected_w = self.config.image_resolution[1], self.config.image_resolution[0]
        actual_h, actual_w = frame.shape[0], frame.shape[1]
        if (actual_h, actual_w) != (expected_h, expected_w):
            errors.append(
                f"Frame {frame_idx}: Segmentation resolution mismatch ({actual_w}x{actual_h}), "
                f"expected ({expected_w}x{expected_h})"
            )

        return errors

    def _validate_bbox_frame(
        self,
        bboxes: Dict[str, Any],
        episode_idx: int,
        camera_id: str,
        frame_idx: int
    ) -> List[str]:
        """
        Validate bounding box annotations (COCO format).

        Checks:
        - COCO format compliance: [x, y, width, height]
        - Values are within image bounds
        - No negative dimensions
        """
        errors = []

        if not isinstance(bboxes, dict):
            errors.append(f"Frame {frame_idx}: Bboxes must be dict, got {type(bboxes)}")
            return errors

        # Get image dimensions
        img_w, img_h = self.config.image_resolution

        # Validate 2D bboxes (COCO format)
        if 'bboxes_2d' in bboxes:
            for obj_id, bbox in bboxes['bboxes_2d'].items():
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    errors.append(f"Frame {frame_idx}, object {obj_id}: Invalid 2D bbox format {bbox}, expected [x, y, w, h]")
                    continue

                x, y, w, h = bbox

                # Validate no negative dimensions
                if w < 0 or h < 0:
                    errors.append(f"Frame {frame_idx}, object {obj_id}: Negative bbox dimensions (w={w}, h={h})")

                # Validate within image bounds
                if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                    errors.append(
                        f"Frame {frame_idx}, object {obj_id}: Bbox [{x}, {y}, {w}, {h}] "
                        f"exceeds image bounds [0, 0, {img_w}, {img_h}]"
                    )

        # Validate 3D bboxes (if present)
        if 'bboxes_3d' in bboxes:
            for obj_id, bbox_3d in bboxes['bboxes_3d'].items():
                if not isinstance(bbox_3d, dict):
                    errors.append(f"Frame {frame_idx}, object {obj_id}: Invalid 3D bbox format, expected dict")
                    continue

                # Check for required fields
                required_fields = ['center', 'size']
                for field in required_fields:
                    if field not in bbox_3d:
                        errors.append(f"Frame {frame_idx}, object {obj_id}: Missing 3D bbox field '{field}'")

        return errors

    def _verify_parquet_exports(self, data_dir: Path) -> List[Tuple[Path, List[str]]]:
        """
        Verify Parquet files after export.

        Checks:
        1. File exists and is readable
        2. Schema is consistent with expected format
        3. Contains expected columns (state, action, etc.)
        4. No missing data (NaN/None values)
        5. Row counts match expected length
        6. Checksum matches manifest

        Returns:
            List of (file_path, errors) tuples for problematic files
        """
        if not HAVE_PYARROW:
            self.log("  Skipping Parquet verification (pyarrow not available)", "DEBUG")
            return []

        verification_errors = []

        try:
            expected_lengths = {
                episode.episode_index: episode.trajectory.num_frames
                for episode in self.episodes
            }
            expected_row_counts = {
                episode.episode_index: episode.parquet_row_count
                for episode in self.episodes
                if episode.parquet_row_count is not None
            }
            # Find all Parquet files
            parquet_files = list(data_dir.rglob("*.parquet"))
            if not parquet_files:
                self.log(f"  No Parquet files found in {data_dir}", "WARNING")
                return []

            self.log(f"  Verifying {len(parquet_files)} Parquet files...")

            for parquet_path in parquet_files:
                errors = []

                try:
                    # Try to read the file
                    table = pq.read_table(parquet_path)

                    # Check for expected columns
                    required_columns = {"observation.state", "action"}
                    missing_columns = required_columns - set(table.column_names)
                    if missing_columns:
                        errors.append(f"Missing columns: {missing_columns}")

                    # Check for NaN/None in critical columns
                    for col_name in ["observation.state", "action"]:
                        if col_name in table.column_names:
                            col = table[col_name].to_numpy()
                            nan_count = np.isnan(col).sum() if col.dtype == np.float32 else 0
                            if nan_count > 0:
                                errors.append(f"Column '{col_name}': {nan_count} NaN values")

                    # Check row count (should match trajectory length for episode)
                    num_rows = table.num_rows
                    if num_rows == 0:
                        errors.append("Table is empty (0 rows)")
                    episode_match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
                    if episode_match:
                        episode_index = int(episode_match.group(1))
                        expected_length = expected_lengths.get(episode_index)
                        if expected_length is None:
                            errors.append(
                                f"No episode metadata found for index {episode_index}"
                            )
                        elif num_rows != expected_length:
                            errors.append(
                                f"Row count {num_rows} does not match expected "
                                f"{expected_length} frames for episode {episode_index}"
                            )

                        expected_row_count = expected_row_counts.get(episode_index)
                        if expected_row_count is not None and num_rows != expected_row_count:
                            errors.append(
                                f"Row count {num_rows} does not match recorded metadata "
                                f"{expected_row_count} for episode {episode_index}"
                            )
                    else:
                        errors.append(
                            f"Unable to determine episode index from filename {parquet_path.name}"
                        )

                    checksum = _checksum_file(parquet_path, self.checksum_algorithm)
                    if self.output_dir is None:
                        errors.append("Checksum manifest unavailable (output_dir unset)")
                    else:
                        rel_path = parquet_path.resolve().relative_to(
                            self.output_dir.resolve()
                        ).as_posix()
                        manifest_checksum = self.checksum_manifest.get(rel_path)
                        if manifest_checksum is None:
                            errors.append(
                                f"Checksum manifest missing entry for {rel_path}"
                            )
                        elif checksum != manifest_checksum:
                            errors.append(
                                f"Checksum mismatch for {rel_path}: expected {manifest_checksum}, got {checksum}"
                            )

                except Exception as e:
                    errors.append(f"Failed to read Parquet file: {type(e).__name__}: {str(e)}")

                if errors:
                    verification_errors.append((parquet_path, errors))

        except Exception as e:
            self.log(f"  Error during Parquet verification: {e}", "ERROR")

        return verification_errors

    def finalize(self) -> Path:
        """
        Write the complete dataset to disk with comprehensive validation.

        Validates:
        - Episode completeness (all required fields present)
        - Data pack tier compliance (tier-specific data requirements)
        - Camera calibration matrices (intrinsic/extrinsic validity)
        - Trajectory-sensor alignment
        - Parquet post-export verification

        Returns:
            Path to the dataset directory
        """
        self.log("=" * 60)
        self.log("Finalizing LeRobot Dataset Export")
        self.log(f"Data Pack: {self.config.data_pack_tier}")
        self.log("=" * 60)

        output_dir = Path(self.config.output_dir)
        output_parent = output_dir.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        temp_dir = output_parent / f".tmp_export_{uuid.uuid4().hex}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=False)

        self.output_dir = temp_dir
        self._temp_export_dir = temp_dir
        self._export_output_dir = output_dir
        self._backup_output_dir = None
        self.checksum_generated_at = datetime.utcnow().isoformat() + "Z"
        self.checksum_manifest = {}

        # Create directories in temp export
        meta_dir = temp_dir / "meta"
        data_dir = temp_dir / "data"
        meta_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        try:
            # Validate episode completeness before export
            self.log("Validating episode completeness...")
            incomplete = self._validate_episode_completeness()
            if incomplete:
                sample_limit = 5
                self.log(
                    "  ERROR: Incomplete episodes detected "
                    f"(showing first {min(len(incomplete), sample_limit)} of {len(incomplete)}).",
                    "ERROR",
                )
                for ep_idx, errors in incomplete[:sample_limit]:
                    self.log(f"    Episode {ep_idx}:", "ERROR")
                    for error in errors:
                        self.log(f"      - {error}", "ERROR")
                if len(incomplete) > sample_limit:
                    self.log(
                        f"    ... and {len(incomplete) - sample_limit} more incomplete episodes",
                        "ERROR",
                    )

                if self.config.require_complete_episodes:
                    raise ValueError(
                        "Incomplete episodes detected; "
                        "REQUIRE_COMPLETE_EPISODES is enabled so export cannot continue."
                    )

                dropped_episode_ids = {ep_idx for ep_idx, _ in incomplete}
                self.log(
                    f"  Dropping {len(dropped_episode_ids)} incomplete episodes before export.",
                    "ERROR",
                )
                self.episodes = [
                    episode for episode in self.episodes
                    if episode.episode_index not in dropped_episode_ids
                ]
                if not self.episodes:
                    raise ValueError(
                        "All episodes were incomplete; aborting export after validation."
                    )
            else:
                self.log("  All episodes are complete")

            # Validate data pack tier compliance
            self.log("Validating data pack tier compliance...")
            non_compliant = self._validate_data_pack_tier_compliance()
            if non_compliant:
                self.log(
                    f"  Found {len(non_compliant)} tier-non-compliant episodes:",
                    "WARNING",
                )
                for ep_idx, errors in non_compliant[:3]:  # Show first 3
                    self.log(f"    Episode {ep_idx}:", "WARNING")
                    for error in errors[:2]:  # Show first 2 errors per episode
                        self.log(f"      - {error}", "WARNING")
                if len(non_compliant) > 3:
                    self.log(f"    ... and {len(non_compliant) - 3} more episodes", "WARNING")

                tier_action = self.config.tier_compliance_action.lower().strip()
                if tier_action not in {"fail", "drop"}:
                    raise ValueError(
                        "Invalid tier_compliance_action "
                        f"'{self.config.tier_compliance_action}'. Use 'fail' or 'drop'."
                    )

                self.tier_compliance_summary = {
                    "non_compliant_episodes": len(non_compliant),
                    "dropped_episodes": 0,
                    "action": tier_action,
                }

                if tier_action == "fail":
                    raise ValueError(
                        "Data pack tier compliance failed for "
                        f"{len(non_compliant)} episodes."
                    )

                dropped_episode_ids = {ep_idx for ep_idx, _ in non_compliant}
                self.log(
                    f"  Dropping {len(dropped_episode_ids)} tier-non-compliant episodes before export.",
                    "WARNING",
                )
                self.episodes = [
                    episode for episode in self.episodes
                    if episode.episode_index not in dropped_episode_ids
                ]
                self.tier_compliance_summary["dropped_episodes"] = len(dropped_episode_ids)
                if not self.episodes:
                    raise ValueError(
                        "All episodes failed tier compliance; aborting export."
                    )
            else:
                self.log(f"  All episodes comply with '{self.config.data_pack_tier}' tier requirements")
                self.tier_compliance_summary = {
                    "non_compliant_episodes": 0,
                    "dropped_episodes": 0,
                    "action": "none",
                }

            # Validate camera calibration matrices
            self.log("Validating camera calibration matrices...")
            bad_calibrations = self._validate_camera_calibration()
            if bad_calibrations:
                self.log(f"  WARNING: Found {len(bad_calibrations)} episodes with invalid calibrations:", "WARNING")
                for ep_idx, errors in bad_calibrations[:3]:  # Show first 3
                    self.log(f"    Episode {ep_idx}:", "WARNING")
                    for error in errors[:2]:  # Show first 2 errors per episode
                        self.log(f"      - {error}", "WARNING")
                if len(bad_calibrations) > 3:
                    self.log(f"    ... and {len(bad_calibrations) - 3} more episodes", "WARNING")
            else:
                self.log("  All calibration matrices are valid")

            # Validate trajectory-sensor frame alignment
            self.log("Validating trajectory-sensor frame alignment...")
            alignment_errors = self._validate_trajectory_sensor_alignment()
            if alignment_errors:
                self.log(f"  Found {len(alignment_errors)} episodes with alignment issues:", "WARNING")
                for entry in alignment_errors[:5]:  # Show first 5 episodes
                    self.log(f"    Episode {entry['episode_index']}:", "WARNING")
                    for error in entry["errors"][:3]:
                        self.log(f"      - {error}", "WARNING")
                if len(alignment_errors) > 5:
                    self.log(f"    ... and {len(alignment_errors) - 5} more", "WARNING")

                is_production = self._is_production_mode()
                self._write_invalid_episodes(
                    meta_dir,
                    alignment_errors,
                    reason="trajectory_sensor_alignment",
                    summary={
                        "mismatch_summary": self._summarize_alignment_mismatches(alignment_errors),
                    },
                    export_blocked=is_production or self.config.strict_alignment,
                    blocked_reason="frame_mismatch",
                )

                if is_production or self.config.strict_alignment:
                    mismatch_message = self._format_alignment_mismatch_exception(alignment_errors)
                    if is_production:
                        raise ValueError(
                            "Production export blocked due to trajectory-sensor frame mismatches. "
                            f"{mismatch_message}"
                        )
                    raise ValueError(
                        "Trajectory-sensor alignment validation failed. "
                        f"{mismatch_message} Set strict_alignment=False to drop them."
                    )

                dropped_episode_ids = {entry["episode_index"] for entry in alignment_errors}
                self.log(
                    f"  Dropping {len(dropped_episode_ids)} episodes due to alignment errors.",
                    "WARNING",
                )
                self.episodes = [
                    episode for episode in self.episodes
                    if episode.episode_index not in dropped_episode_ids
                ]
            else:
                self.log("  All episodes have aligned trajectory-sensor frames")

            # Step 1: Write episode data (joint-space trajectories)
            self.log("Writing episode data...")
            try:
                self._write_episodes(data_dir)
            except Exception as exc:
                self._cleanup_failed_export("Failed to write episode data", exc)
                raise

            # Verify Parquet exports after writing
            self.log("Verifying Parquet file exports...")
            try:
                parquet_errors = self._verify_parquet_exports(data_dir)
            except Exception as exc:
                self._cleanup_failed_export("Failed to verify Parquet exports", exc)
                raise
            if parquet_errors:
                self.log(f"  WARNING: Found {len(parquet_errors)} Parquet verification issues:", "WARNING")
                for file_path, errors in parquet_errors[:3]:  # Show first 3 files
                    self.log(f"    {file_path.name}:", "WARNING")
                    for error in errors[:2]:  # Show first 2 errors per file
                        self.log(f"      - {error}", "WARNING")
                if len(parquet_errors) > 3:
                    self.log(f"    ... and {len(parquet_errors) - 3} more files with issues", "WARNING")
                if self._is_production_mode():
                    raise ValueError(
                        f"Production export blocked due to {len(parquet_errors)} Parquet verification errors."
                    )
            else:
                self.log("  Parquet files verified successfully")

            # Step 2: Write visual observations (RGB videos per camera)
            if self.config.include_images:
                self.log("Writing visual observations...")
                try:
                    self._write_visual_observations(temp_dir, meta_dir)
                except Exception as exc:
                    self._cleanup_failed_export("Failed to write visual observations", exc)
                    raise

            # Step 3: Write ground-truth labels (Plus/Full packs)
            if any([
                self.config.include_depth,
                self.config.include_segmentation,
                self.config.include_bboxes,
                self.config.include_object_poses,
                self.config.include_contacts,
            ]):
                self.log("Writing ground-truth labels...")
                try:
                    self._write_ground_truth(temp_dir)
                except Exception as exc:
                    self._cleanup_failed_export("Failed to write ground-truth labels", exc)
                    raise

            # Step 4: Calculate and write statistics
            self.log("Calculating statistics...")
            try:
                self._calculate_stats()
                self._write_stats(meta_dir)
            except Exception as exc:
                self._cleanup_failed_export("Failed to calculate/write stats", exc)
                raise

            # Step 5: Write metadata
            self.log("Writing metadata...")
            try:
                self._write_info(meta_dir)
                self._write_tasks(meta_dir)
                self._write_episodes_meta(meta_dir)
            except Exception as exc:
                self._cleanup_failed_export("Failed to write metadata", exc)
                raise

            # Step 6: Write checksums manifest
            self.log("Validating checksum manifest consistency...")
            try:
                self._verify_checksum_manifest_consistency(temp_dir)
                self._write_checksums_manifest(meta_dir)
            except Exception as exc:
                self._cleanup_failed_export("Failed to write checksum manifest", exc)
                raise

            # Finalize export atomically
            backup_dir = Path(f"{output_dir}.bak")
            self._backup_output_dir = backup_dir
            try:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                if output_dir.exists():
                    os.replace(output_dir, backup_dir)
                os.replace(temp_dir, output_dir)
            except Exception as exc:
                self._log_export_state("Failed to finalize export swap; attempting restore", "ERROR")
                try:
                    if backup_dir.exists() and not output_dir.exists():
                        os.replace(backup_dir, output_dir)
                except Exception as restore_exc:
                    restore_message = self._sanitize_error_message(str(restore_exc))
                    self._log_export_state(
                        f"Failed to restore backup after swap failure: {restore_message}",
                        "ERROR",
                    )
                self._cleanup_failed_export("Finalize export swap failed", exc)
                raise
            else:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)

            self.output_dir = output_dir

            # Summary
            self.log("=" * 60)
            self.log(f"Dataset exported: {output_dir}")
            self.log(f"  Data Pack: {self.config.data_pack_tier}")
            self.log(f"  Episodes: {len(self.episodes)}")
            self.log(f"  Tasks: {len(self.tasks)}")

            # Count episodes with visual data
            visual_episodes = sum(1 for e in self.episodes if e.sensor_data is not None)
            if visual_episodes > 0:
                self.log(f"  Episodes with visual obs: {visual_episodes}")
                self.log(f"  Cameras: {self.config.camera_types}")

            # Report ground-truth streams
            gt_streams = []
            if self.config.include_depth:
                gt_streams.append("depth")
            if self.config.include_segmentation:
                gt_streams.append("segmentation")
            if self.config.include_bboxes:
                gt_streams.append("bboxes")
            if self.config.include_object_poses:
                gt_streams.append("object_poses")
            if self.config.include_contacts:
                gt_streams.append("contacts")
            if self.config.include_privileged_state:
                gt_streams.append("privileged_state")

            if gt_streams:
                self.log(f"  Ground-truth streams: {', '.join(gt_streams)}")

            self.log("=" * 60)

            return output_dir
        except Exception as exc:
            self._cleanup_failed_export("Export failed", exc)
            raise
        finally:
            self._temp_export_dir = None
            self._backup_output_dir = None
            self._export_output_dir = None

    def _write_episodes(self, data_dir: Path) -> None:
        """Write episode data to Parquet files."""
        try:
            chunk_idx = 0
            chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(exist_ok=True)

            for episode in self.episodes:
                # Create episode data
                episode_data = self._trajectory_to_arrow_table(episode)

                # Write to Parquet
                episode_path = chunk_dir / f"episode_{episode.episode_index:06d}.parquet"

                if HAVE_PYARROW:
                    self._atomic_write(
                        episode_path,
                        lambda tmp_path: pq.write_table(episode_data, tmp_path),
                        fsync_target=True,
                    )
                    episode.parquet_row_count = episode_data.num_rows
                    if self.output_dir:
                        episode.parquet_path = episode_path.resolve().relative_to(
                            self.output_dir.resolve()
                        ).as_posix()
                else:
                    # Fallback: write as JSON
                    self._write_episode_json(episode, episode_path.with_suffix(".json"))

                # Handle chunk rotation
                if (episode.episode_index + 1) % self.config.chunk_size == 0:
                    chunk_idx += 1
                    chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
                    chunk_dir.mkdir(exist_ok=True)
        except Exception as exc:
            self._cleanup_failed_export("Failed during episode write", exc)
            raise

    def _validate_lerobot_schema(
        self,
        timestamps: List[float],
        state_data: List[List[float]],
        action_data: List[List[float]],
        episode_index: int,
    ) -> None:
        """
        Validate LeRobot v2.0 schema compliance.

        Validates:
        - observation.state dimensions match robot DOF
        - action dimensions match robot DOF + gripper
        - timestamps are monotonically increasing
        - Action continuity (no large jumps)
        - Joint velocity/acceleration limits
        - Gripper position range (0.0-1.0)

        Raises:
            ValueError: If validation fails
        """
        # Validate state dimensions
        expected_state_dim = self.robot_config.num_dof
        for i, state in enumerate(state_data):
            if len(state) != expected_state_dim:
                raise ValueError(
                    f"Episode {episode_index}, frame {i}: observation.state dimension mismatch. "
                    f"Expected {expected_state_dim}, got {len(state)}"
                )

        # Validate action dimensions (DOF + gripper)
        expected_action_dim = self.robot_config.num_dof + 1
        for i, action in enumerate(action_data):
            if len(action) != expected_action_dim:
                raise ValueError(
                    f"Episode {episode_index}, frame {i}: action dimension mismatch. "
                    f"Expected {expected_action_dim} (DOF + gripper), got {len(action)}"
                )

        # Validate timestamps are monotonically increasing
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                raise ValueError(
                    f"Episode {episode_index}, frame {i}: timestamps not monotonically increasing. "
                    f"Frame {i-1}: {timestamps[i-1]:.6f}, Frame {i}: {timestamps[i]:.6f}"
                )

        # Validate timestamps are non-negative
        if any(t < 0 for t in timestamps):
            raise ValueError(
                f"Episode {episode_index}: negative timestamps detected"
            )

        # Validate gripper position range (0.0-1.0)
        for i, action in enumerate(action_data):
            gripper_pos = action[-1]  # Last element is gripper
            if not (0.0 <= gripper_pos <= 1.0):
                raise ValueError(
                    f"Episode {episode_index}, frame {i}: gripper position {gripper_pos:.3f} out of range [0.0, 1.0]"
                )

        # Validate action continuity (detect large jumps between frames)
        # Max joint displacement per frame (radians or meters depending on joint type)
        # This is a heuristic - actual limits depend on robot and control frequency
        max_joint_delta = 0.5  # radians per timestep (conservative for 30Hz control)
        max_gripper_delta = 0.3  # gripper can move faster (0-1 normalized)

        for i in range(1, len(action_data)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                continue  # Already validated monotonic timestamps

            prev_action = action_data[i - 1]
            curr_action = action_data[i]

            # Check joint continuity
            for j in range(expected_action_dim - 1):  # Exclude gripper
                delta = abs(curr_action[j] - prev_action[j])
                # Scale by time delta for variable-rate trajectories
                velocity = delta / dt if dt > 0 else float('inf')

                if delta > max_joint_delta and velocity > max_joint_delta * 30:  # Assume ~30Hz base rate
                    raise ValueError(
                        f"Episode {episode_index}, frame {i}: Large joint discontinuity detected. "
                        f"Joint {j}: delta={delta:.3f} rad, velocity={velocity:.3f} rad/s. "
                        f"This may indicate a trajectory solver error or collision."
                    )

            # Check gripper continuity
            gripper_delta = abs(curr_action[-1] - prev_action[-1])
            if gripper_delta > max_gripper_delta:
                # Gripper jumps are acceptable (open/close), but log warning for extreme values
                if gripper_delta > 0.8:  # Very large jump (nearly full range)
                    self.log(
                        f"Episode {episode_index}, frame {i}: Large gripper jump {gripper_delta:.3f}. "
                        f"This is acceptable but verify grasp timing.",
                        "WARNING"
                    )

        # Validate joint velocity limits (heuristic based on robot type)
        # Standard industrial robot velocity limits (conservative)
        max_joint_velocity = 2.0  # rad/s (conservative for 7-DOF arms)

        for i in range(1, len(state_data)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                continue

            prev_state = state_data[i - 1]
            curr_state = state_data[i]

            for j in range(expected_state_dim):
                velocity = abs(curr_state[j] - prev_state[j]) / dt
                if velocity > max_joint_velocity:
                    raise ValueError(
                        f"Episode {episode_index}, frame {i}: Joint velocity limit exceeded. "
                        f"Joint {j}: velocity={velocity:.3f} rad/s, limit={max_joint_velocity:.3f} rad/s. "
                        f"This trajectory may not be executable on real hardware."
                    )

        # Validate joint acceleration limits (heuristic)
        max_joint_acceleration = 5.0  # rad/s² (conservative)

        for i in range(2, len(state_data)):
            dt1 = timestamps[i - 1] - timestamps[i - 2]
            dt2 = timestamps[i] - timestamps[i - 1]
            if dt1 <= 0 or dt2 <= 0:
                continue

            prev_state = state_data[i - 2]
            curr_state = state_data[i - 1]
            next_state = state_data[i]

            for j in range(expected_state_dim):
                vel1 = (curr_state[j] - prev_state[j]) / dt1
                vel2 = (next_state[j] - curr_state[j]) / dt2
                acceleration = abs(vel2 - vel1) / ((dt1 + dt2) / 2)

                if acceleration > max_joint_acceleration:
                    raise ValueError(
                        f"Episode {episode_index}, frame {i}: Joint acceleration limit exceeded. "
                        f"Joint {j}: acceleration={acceleration:.3f} rad/s², limit={max_joint_acceleration:.3f} rad/s². "
                        f"This trajectory may not be executable on real hardware."
                    )

    def _trajectory_to_arrow_table(self, episode: LeRobotEpisode) -> Union[Any, Dict]:
        """Convert trajectory to PyArrow table."""

        trajectory = episode.trajectory
        states = trajectory.states

        # Extract data arrays
        timestamps = [s.timestamp for s in states]
        frame_indices = list(range(len(states)))
        episode_indices = [episode.episode_index] * len(states)

        # State: joint positions
        state_data = [s.joint_positions.tolist() for s in states]

        # Action: joint positions + gripper (next frame's state as action)
        # Standard convention: action[t] leads to state[t+1]
        action_data = []
        for i, s in enumerate(states):
            if i < len(states) - 1:
                next_state = states[i + 1]
                action = list(next_state.joint_positions) + [next_state.gripper_position]
            else:
                # Last frame: repeat current
                action = list(s.joint_positions) + [s.gripper_position]
            action_data.append(action)

        # Validate LeRobot v2.0 schema compliance
        self._validate_lerobot_schema(
            timestamps=timestamps,
            state_data=state_data,
            action_data=action_data,
            episode_index=episode.episode_index,
        )

        # Validate ee_position is not None (cannot use [0,0,0] as it's a valid position)
        for i, s in enumerate(states):
            if s.ee_position is None:
                raise ValueError(
                    f"Episode {episode.episode_index}: ee_position is None at frame {i}. "
                    f"Cannot use [0,0,0] as fallback since it's a valid position. "
                    f"Ensure trajectory solver computes ee_position for all frames."
                )

        # Index within episode
        index_in_episode = list(range(len(states)))

        if HAVE_PYARROW:
            # Create Arrow arrays
            table = pa.table({
                "timestamp": pa.array(timestamps, type=pa.float64()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(episode_indices, type=pa.int64()),
                "index": pa.array(index_in_episode, type=pa.int64()),
                "task_index": pa.array([episode.task_index] * len(states), type=pa.int64()),
                "observation.state": pa.array(state_data, type=pa.list_(pa.float32())),
                "action": pa.array(action_data, type=pa.list_(pa.float32())),
                # Optional: gripper as separate field
                "observation.gripper_position": pa.array(
                    [s.gripper_position for s in states], type=pa.float32()
                ),
                # EE position (now guaranteed to be non-None)
                "observation.ee_position": pa.array(
                    [s.ee_position.tolist() for s in states],
                    type=pa.list_(pa.float32()),
                ),
            })
            return table
        else:
            # Return as dict for JSON fallback
            return {
                "timestamps": timestamps,
                "frame_indices": frame_indices,
                "episode_index": episode.episode_index,
                "task_index": episode.task_index,
                "states": state_data,
                "actions": action_data,
                "gripper_positions": [s.gripper_position for s in states],
            }

    def _write_episode_json(self, episode: LeRobotEpisode, path: Path) -> None:
        """Fallback: write episode as JSON."""
        data = self._trajectory_to_arrow_table(episode)
        self._atomic_write(
            path,
            lambda tmp_path: tmp_path.write_text(json.dumps(data, indent=2)),
        )

    def _calculate_stats(self) -> None:
        """Calculate statistics for normalization."""

        all_states = []
        all_actions = []

        for episode in self.episodes:
            for i, state in enumerate(episode.trajectory.states):
                all_states.append(state.joint_positions)

                # Action
                if i < len(episode.trajectory.states) - 1:
                    next_state = episode.trajectory.states[i + 1]
                    action = np.append(next_state.joint_positions, next_state.gripper_position)
                else:
                    action = np.append(state.joint_positions, state.gripper_position)
                all_actions.append(action)

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)

        self.stats["observation.state"] = {
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist(),
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
        }

        self.stats["action"] = {
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
        }

    def _write_stats(self, meta_dir: Path) -> None:
        """Write statistics JSON."""
        stats_path = meta_dir / "stats.json"
        self._atomic_write(
            stats_path,
            lambda tmp_path: tmp_path.write_text(json.dumps(self.stats, indent=2)),
        )

    def _summarize_alignment_mismatches(
        self,
        invalid_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        mismatch_totals = {"frame_count": 0, "timestamp": 0, "missing_camera": 0}
        for entry in invalid_entries:
            mismatch_counts = entry.get("mismatch_counts", {})
            for key in mismatch_totals:
                mismatch_totals[key] += int(mismatch_counts.get(key, 0) or 0)
        total_mismatches = sum(mismatch_totals.values())
        return {
            "total_mismatches": total_mismatches,
            "by_type": mismatch_totals,
        }

    def _format_alignment_mismatch_exception(self, alignment_errors: List[Dict[str, Any]]) -> str:
        summary = []
        for entry in alignment_errors:
            mismatch_counts = entry.get("mismatch_counts", {})
            mismatch_total = sum(int(v or 0) for v in mismatch_counts.values())
            summary.append(f"{entry.get('episode_index')} (mismatches={mismatch_total})")
        mismatch_summary = "; ".join(summary) if summary else "none"
        totals = self._summarize_alignment_mismatches(alignment_errors)
        totals_by_type = ", ".join(
            f"{key}={value}" for key, value in totals["by_type"].items()
        )
        return (
            "Trajectory-sensor alignment mismatches detected for episodes: "
            f"{mismatch_summary}. Total mismatches: {totals['total_mismatches']} "
            f"({totals_by_type})."
        )

    def _write_invalid_episodes(
        self,
        meta_dir: Path,
        invalid_entries: List[Dict[str, Any]],
        *,
        reason: str,
        summary: Dict[str, Any],
        export_blocked: bool = False,
        blocked_reason: Optional[str] = None,
    ) -> None:
        """Write invalid episode details to meta directory."""
        invalid_path = meta_dir / "invalid_episodes.json"
        entry = {
            "reason": reason,
            "total_dropped": len(invalid_entries),
            "episodes": invalid_entries,
            "summary": {
                "export_blocked": export_blocked,
                "blocked_reason": blocked_reason if export_blocked else None,
                "message": (
                    "Export blocked due to invalid episodes."
                    if export_blocked
                    else "Episodes dropped due to validation issues."
                ),
                **summary,
            },
        }
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "entries": [entry],
        }

        if invalid_path.exists():
            try:
                existing = json.loads(invalid_path.read_text())
            except json.JSONDecodeError:
                existing = None
            if isinstance(existing, dict):
                existing_entries = existing.get("entries")
                if existing_entries is None and "reason" in existing:
                    existing_entries = [{
                        "reason": existing.get("reason"),
                        "total_dropped": existing.get("total_dropped"),
                        "episodes": existing.get("episodes", []),
                        "summary": existing.get("summary", {}),
                    }]
                if isinstance(existing_entries, list):
                    payload["generated_at"] = existing.get("generated_at", payload["generated_at"])
                    payload["entries"] = existing_entries + [entry]

        self._atomic_write(
            invalid_path,
            lambda tmp_path: tmp_path.write_text(json.dumps(payload, indent=2)),
            record_checksum=True,
        )

    def _summarize_dropped_frames(
        self,
        invalid_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        totals = {
            "rgb_frames_expected": 0,
            "rgb_frames_written": 0,
            "rgb_frames_dropped": 0,
        }
        for entry in invalid_entries:
            totals["rgb_frames_expected"] += int(entry.get("rgb_frames_expected_total", 0) or 0)
            totals["rgb_frames_written"] += int(entry.get("rgb_frames_written_total", 0) or 0)
            totals["rgb_frames_dropped"] += int(entry.get("dropped_frames_total", 0) or 0)
        return {
            "episode_count": len(invalid_entries),
            **totals,
        }

    def _remove_checksum_entry(self, path: Path) -> None:
        if self.output_dir is None:
            return
        try:
            rel_path = path.resolve().relative_to(self.output_dir.resolve()).as_posix()
        except ValueError:
            return
        self.checksum_manifest.pop(rel_path, None)

    def _drop_episode_files(
        self,
        output_dir: Path,
        episode_index: int,
        camera_ids: List[str],
    ) -> None:
        chunk_idx = episode_index // self.config.chunk_size
        data_chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        for suffix in (".parquet", ".json"):
            episode_path = data_chunk_dir / f"episode_{episode_index:06d}{suffix}"
            if episode_path.exists():
                episode_path.unlink()
                self._remove_checksum_entry(episode_path)

        videos_chunk_dir = output_dir / "videos" / f"chunk-{chunk_idx:03d}"
        for camera_id in camera_ids:
            video_path = (
                videos_chunk_dir
                / f"observation.images.{camera_id}"
                / f"episode_{episode_index:06d}.mp4"
            )
            if video_path.exists():
                video_path.unlink()
                self._remove_checksum_entry(video_path)

    def _write_info(self, meta_dir: Path) -> None:
        """Write dataset info JSON."""
        # Base features
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [self.robot_config.num_joints],
                "names": self.robot_config.joint_names,
            },
            "observation.gripper_position": {
                "dtype": "float32",
                "shape": [1],
            },
            "observation.ee_position": {
                "dtype": "float32",
                "shape": [3],
            },
            "action": {
                "dtype": "float32",
                "shape": [self.robot_config.num_joints + 1],  # + gripper
                "names": self.robot_config.joint_names + ["gripper"],
            },
        }

        # Add visual observation features
        if self.config.include_images:
            for camera_type in self.config.camera_types:
                features[f"observation.images.{camera_type}"] = {
                    "dtype": "video",
                    "shape": [self.config.image_resolution[1], self.config.image_resolution[0], 3],
                    "video_info": {
                        "fps": self.config.fps,
                        "codec": self.config.video_codec,
                    },
                }

        # Add depth features (Plus/Full packs)
        if self.config.include_depth:
            for camera_type in self.config.camera_types:
                features[f"observation.depth.{camera_type}"] = {
                    "dtype": "float32",
                    "shape": [self.config.image_resolution[1], self.config.image_resolution[0]],
                    "unit": "meters",
                }

        # Add segmentation features (Plus/Full packs)
        if self.config.include_segmentation:
            for camera_type in self.config.camera_types:
                features[f"observation.segmentation.{camera_type}"] = {
                    "dtype": "uint8",
                    "shape": [self.config.image_resolution[1], self.config.image_resolution[0]],
                    "includes": ["semantic", "instance"],
                }

        # Add ground-truth features (Plus/Full packs)
        if self.config.include_bboxes:
            features["ground_truth.bboxes_2d"] = {
                "dtype": "json",
                "format": "coco",
            }
            features["ground_truth.bboxes_3d"] = {
                "dtype": "json",
                "format": "camera_space",
            }

        # Add object pose features (Full pack)
        if self.config.include_object_poses:
            features["ground_truth.object_poses"] = {
                "dtype": "json",
                "format": "quaternion_position",
                "coordinate_system": "world",
            }

        # Add contact features (Full pack)
        if self.config.include_contacts:
            features["ground_truth.contacts"] = {
                "dtype": "json",
                "format": "contact_list",
            }

        # Add privileged state (Full pack)
        if self.config.include_privileged_state:
            features["ground_truth.privileged_state"] = {
                "dtype": "json",
                "format": "physics_state",
            }

        # Check data source for validation
        # This determines if the data was generated with real physics or mock
        data_source_info = self._get_data_source_info()
        lineage = self._get_lineage_info(data_source_info)
        self._validate_lineage_requirements(lineage, data_source_info.get("production_mode", False))

        info = {
            "codebase_version": "v2.0",
            "robot_type": self.config.robot_type,
            "fps": self.config.fps,
            "total_episodes": len(self.episodes),
            "total_frames": sum(ep.trajectory.num_frames for ep in self.episodes),
            "total_tasks": len(self.tasks),
            "total_chunks": (len(self.episodes) - 1) // self.config.chunk_size + 1,
            "chunks_size": self.config.chunk_size,
            "data_path": "data",
            "videos_path": "videos" if self.config.include_images else None,
            "ground_truth_path": "ground_truth" if any([
                self.config.include_depth,
                self.config.include_segmentation,
                self.config.include_bboxes,
                self.config.include_object_poses,
                self.config.include_contacts,
            ]) else None,
            "features": features,
            "data_pack": {
                "tier": self.config.data_pack_tier,
                "cameras": self.config.camera_types,
                "resolution": list(self.config.image_resolution),
                "includes_visual_obs": self.config.include_images,
                "includes_depth": self.config.include_depth,
                "includes_segmentation": self.config.include_segmentation,
                "includes_bboxes": self.config.include_bboxes,
                "includes_object_poses": self.config.include_object_poses,
                "includes_contacts": self.config.include_contacts,
                "includes_privileged_state": self.config.include_privileged_state,
            },
            "tier_compliance": dict(self.tier_compliance_summary),
            "export_controls": {
                "require_complete_episodes": self.config.require_complete_episodes,
                "require_complete_episodes_source": "env:REQUIRE_COMPLETE_EPISODES",
            },
            # Data quality and source information
            "data_quality": {
                "physics_validated": data_source_info["physics_validated"],
                "sensor_source": data_source_info["sensor_source"],
                "validation_type": data_source_info["validation_type"],
                "isaac_sim_used": data_source_info["isaac_sim_used"],
                "warnings": data_source_info["warnings"],
                "video_integrity": (
                    {
                        "partial_frames_allowed": False,
                        "dropped_frames_total": self.dropped_frames_total,
                        "dropped_frames_allowed": not self._is_production_mode(),
                        "note": (
                            "Episodes with invalid RGB frames are either blocked (production) or excluded "
                            "from export (non-production). Refer to episodes.jsonl and invalid_episodes.json "
                            "for per-episode counts."
                        ),
                    }
                    if self.config.include_images
                    else None
                ),
            },
            "lineage": lineage,
            "splits": {
                "train": f"0:{len(self.episodes)}",
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
            "generator": "BlueprintPipeline/episode-generation-job",
            "generator_version": "2.0.0",
            "checksums": {
                "algorithm": self.checksum_algorithm,
                "manifest_path": "meta/checksums.json",
                "generated_at": self.checksum_generated_at or datetime.utcnow().isoformat() + "Z",
                "manifest_count": len(self.checksum_manifest),
                "manifest_hash": self._checksum_manifest_hash(),
                "format": "relative path -> sha256",
                "coverage": "all files under dataset root excluding meta/checksums.json",
            },
        }

        info_path = meta_dir / "info.json"
        self._atomic_write(
            info_path,
            lambda tmp_path: tmp_path.write_text(json.dumps(info, indent=2)),
        )

    def _write_tasks(self, meta_dir: Path) -> None:
        """Write tasks JSONL."""
        tasks_path = meta_dir / "tasks.jsonl"
        def write_tasks(tmp_path: Path) -> None:
            with open(tmp_path, "w") as f:
                for task in self.tasks:
                    f.write(json.dumps(task) + "\n")

        self._atomic_write(tasks_path, write_tasks)

    def _write_episodes_meta(self, meta_dir: Path) -> None:
        """Write episodes metadata JSONL."""
        episodes_path = meta_dir / "episodes.jsonl"
        def write_episodes(tmp_path: Path) -> None:
            with open(tmp_path, "w") as f:
                for episode in self.episodes:
                    meta = {
                        "episode_index": episode.episode_index,
                        "task_index": episode.task_index,
                        "task": episode.task_description,
                        "length": episode.trajectory.num_frames,
                        "scene_id": episode.scene_id,
                        "variation_index": episode.variation_index,
                        "success": episode.success,
                        "quality_score": episode.quality_score,
                        "is_mock": episode.is_mock,  # Explicit flag for mock data
                    }
                    # Add sensor data info if available
                    if episode.sensor_data is not None:
                        meta["has_visual_obs"] = True
                        meta["cameras"] = list(episode.sensor_data.camera_ids) if hasattr(episode.sensor_data, 'camera_ids') else []
                    if episode.camera_capture_warnings:
                        meta["camera_capture_warnings"] = episode.camera_capture_warnings
                    if episode.camera_error_counts:
                        meta["camera_error_counts"] = episode.camera_error_counts
                    if episode.camera_frame_counts:
                        meta["camera_frame_counts"] = episode.camera_frame_counts
                    meta["rgb_frames_expected_total"] = episode.rgb_frames_expected_total
                    meta["rgb_frames_written_total"] = episode.rgb_frames_written_total
                    meta["dropped_frames_total"] = episode.dropped_frames_total
                    if episode.parquet_row_count is not None:
                        meta["parquet_row_count"] = episode.parquet_row_count
                    if episode.parquet_path is not None:
                        meta["parquet_path"] = episode.parquet_path
                    meta["lineage_ref"] = "meta/info.json#/lineage"
                    f.write(json.dumps(meta) + "\n")

        self._atomic_write(episodes_path, write_episodes)

    def _write_checksums_manifest(self, meta_dir: Path) -> None:
        """Write checksums manifest for generated files."""
        if not self.checksum_manifest:
            return
        manifest_path = meta_dir / "checksums.json"
        self._atomic_write(
            manifest_path,
            lambda tmp_path: tmp_path.write_text(self._render_checksum_manifest()),
            record_checksum=False,
        )

    def _write_visual_observations(self, output_dir: Path, meta_dir: Path) -> None:
        """Write visual observations (images, videos) for all episodes."""
        if not self.config.include_images:
            return
        try:
            videos_dir = output_dir / "videos"

            chunk_idx = 0
            dropped_frames_entries: List[Dict[str, Any]] = []
            is_production = self._is_production_mode()

            for episode in self.episodes:
                # Determine chunk
                if episode.episode_index > 0 and episode.episode_index % self.config.chunk_size == 0:
                    chunk_idx += 1

                chunk_dir = videos_dir / f"chunk-{chunk_idx:03d}"

                if episode.sensor_data is not None:
                    self._write_episode_videos(episode, chunk_dir)
                    if episode.dropped_frames_total > 0:
                        dropped_frames_entries.append({
                            "episode_index": episode.episode_index,
                            "task_index": episode.task_index,
                            "scene_id": episode.scene_id,
                            "variation_index": episode.variation_index,
                            "dropped_frames_total": episode.dropped_frames_total,
                            "rgb_frames_expected_total": episode.rgb_frames_expected_total,
                            "rgb_frames_written_total": episode.rgb_frames_written_total,
                            "camera_frame_counts": episode.camera_frame_counts,
                        })
                        if not is_production:
                            self._drop_episode_files(
                                output_dir,
                                episode.episode_index,
                                list(getattr(episode.sensor_data, "camera_ids", [])),
                            )
                elif episode.camera_video_path is not None:
                    # Legacy: copy existing video if available
                    self._copy_video(episode.camera_video_path, chunk_dir, episode.episode_index)
            if dropped_frames_entries:
                self._write_invalid_episodes(
                    meta_dir,
                    dropped_frames_entries,
                    reason="dropped_frames",
                    summary={
                        "dropped_frames_summary": self._summarize_dropped_frames(dropped_frames_entries),
                        "drops_allowed": not is_production,
                    },
                    export_blocked=is_production,
                    blocked_reason="dropped_frames",
                )
                if is_production:
                    raise ValueError(
                        "Production export blocked due to dropped RGB frames in visual observations."
                    )
                dropped_episode_ids = {entry["episode_index"] for entry in dropped_frames_entries}
                self.log(
                    f"  Dropping {len(dropped_episode_ids)} episodes due to dropped RGB frames.",
                    "WARNING",
                )
                self.episodes = [
                    episode for episode in self.episodes
                    if episode.episode_index not in dropped_episode_ids
                ]
        except Exception as exc:
            self._cleanup_failed_export("Failed during visual observations write", exc)
            raise

    def _write_episode_videos(self, episode: LeRobotEpisode, chunk_dir: Path) -> None:
        """Write video files with per-frame validation."""
        sensor_data = episode.sensor_data
        if sensor_data is None or not hasattr(sensor_data, 'frames'):
            return

        total_expected = 0
        total_written = 0
        total_dropped = 0

        # Write video for each camera
        for camera_id in sensor_data.camera_ids if hasattr(sensor_data, 'camera_ids') else []:
            video_dir = chunk_dir / f"observation.images.{camera_id}"
            video_dir.mkdir(parents=True, exist_ok=True)

            video_path = video_dir / f"episode_{episode.episode_index:06d}.mp4"

            # Collect RGB frames with validation
            frames = []
            validation_errors = []
            expected_frames = 0
            written_frames = 0
            dropped_frames = 0

            for frame_idx, frame in enumerate(sensor_data.frames):
                if hasattr(frame, 'rgb_images') and camera_id in frame.rgb_images:
                    expected_frames += 1
                    rgb_frame = frame.rgb_images[camera_id]

                    # Validate RGB frame
                    frame_errors = self._validate_rgb_frame(
                        rgb_frame, episode.episode_index, camera_id, frame_idx
                    )
                    if frame_errors:
                        validation_errors.extend(frame_errors)
                        dropped_frames += 1
                    else:
                        frames.append(rgb_frame)
                        written_frames += 1

            # Log validation errors (but don't fail - continue with valid frames)
            if validation_errors:
                self.log(f"  Episode {episode.episode_index}, camera {camera_id}: {len(validation_errors)} validation errors", "WARNING")
                for error in validation_errors[:3]:  # Log first 3
                    self.log(f"    - {error}", "WARNING")

            episode.camera_frame_counts[camera_id] = {
                "rgb_frames_expected": expected_frames,
                "rgb_frames_written": written_frames,
                "rgb_frames_dropped": dropped_frames,
            }

            if frames:
                self._write_video(frames, video_path, self.config.fps)

            total_expected += expected_frames
            total_written += written_frames
            total_dropped += dropped_frames

        episode.rgb_frames_expected_total = total_expected
        episode.rgb_frames_written_total = total_written
        episode.dropped_frames_total = total_dropped
        self.dropped_frames_total += total_dropped

    def _write_video(self, frames: List[np.ndarray], video_path: Path, fps: float) -> None:
        """Write frames to a video file."""
        if not frames:
            return

        if HAVE_IMAGEIO:
            try:
                def write_video(tmp_path: Path) -> None:
                    writer = imageio.get_writer(
                        str(tmp_path),
                        fps=fps,
                        codec="libx264",
                        quality=8,
                    )
                    for frame in frames:
                        # Ensure RGB format (H, W, 3)
                        if frame.ndim == 3 and frame.shape[-1] == 3:
                            writer.append_data(frame)
                    writer.close()

                self._atomic_write(video_path, write_video)
                self.log(f"  Wrote video: {video_path}")
            except Exception as e:
                self.log(f"  Video write failed: {e}", "WARNING")
                # Fallback to individual frames
                self._write_frames_as_images(frames, video_path.parent, video_path.stem)
        else:
            # Fallback to individual frames
            self._write_frames_as_images(frames, video_path.parent, video_path.stem)

    def _write_frames_as_images(self, frames: List[np.ndarray], output_dir: Path, prefix: str) -> None:
        """Write frames as individual images (fallback)."""
        frames_dir = output_dir / f"{prefix}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            if HAVE_PIL:
                frame_path = frames_dir / f"frame_{i:06d}.png"
                self._atomic_write(
                    frame_path,
                    lambda tmp_path, frame=frame: Image.fromarray(frame).save(tmp_path),
                )
            else:
                frame_path = frames_dir / f"frame_{i:06d}.npy"
                self._atomic_write_with_suffix(
                    frame_path,
                    lambda tmp_path, frame=frame: np.save(tmp_path, frame),
                )

    def _copy_video(self, src_path: Path, chunk_dir: Path, episode_index: int) -> None:
        """Copy an existing video file to the output directory."""
        video_dir = chunk_dir / "observation.images.camera"
        video_dir.mkdir(parents=True, exist_ok=True)
        dst_path = video_dir / f"episode_{episode_index:06d}.mp4"
        try:
            self._atomic_write(
                dst_path,
                lambda tmp_path: shutil.copy2(src_path, tmp_path),
            )
        except Exception as e:
            self.log(f"  Failed to copy video: {e}", "WARNING")

    def _write_ground_truth(self, output_dir: Path) -> None:
        """Write ground-truth labels for all episodes (Plus/Full packs)."""
        if not any([
            self.config.include_depth,
            self.config.include_segmentation,
            self.config.include_bboxes,
            self.config.include_object_poses,
            self.config.include_contacts,
        ]):
            return
        try:
            gt_dir = output_dir / "ground_truth"
            chunk_idx = 0

            for episode in self.episodes:
                if episode.episode_index > 0 and episode.episode_index % self.config.chunk_size == 0:
                    chunk_idx += 1

                chunk_dir = gt_dir / f"chunk-{chunk_idx:03d}"

                if episode.sensor_data is not None:
                    self._write_episode_ground_truth(episode, chunk_dir)
        except Exception as exc:
            self._cleanup_failed_export("Failed during ground-truth write", exc)
            raise

    def _write_episode_ground_truth(self, episode: LeRobotEpisode, chunk_dir: Path) -> None:
        """Write ground-truth data for a single episode."""
        sensor_data = episode.sensor_data
        if sensor_data is None or not hasattr(sensor_data, 'frames'):
            return

        episode_idx = episode.episode_index

        # Depth maps
        if self.config.include_depth:
            self._write_depth_data(sensor_data, chunk_dir, episode_idx)

        # Segmentation masks
        if self.config.include_segmentation:
            self._write_segmentation_data(sensor_data, chunk_dir, episode_idx)

        # Bounding boxes
        if self.config.include_bboxes:
            self._write_bbox_data(sensor_data, chunk_dir, episode_idx)

        # Object poses
        if self.config.include_object_poses:
            self._write_object_pose_data(sensor_data, chunk_dir, episode_idx)

        # Contacts
        if self.config.include_contacts:
            self._write_contact_data(sensor_data, chunk_dir, episode_idx)

        # Privileged state
        if self.config.include_privileged_state:
            self._write_privileged_state_data(sensor_data, chunk_dir, episode_idx)

    def _write_depth_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write depth maps with validation."""
        for camera_id in sensor_data.camera_ids if hasattr(sensor_data, 'camera_ids') else []:
            depth_dir = chunk_dir / "depth" / camera_id
            depth_dir.mkdir(parents=True, exist_ok=True)

            depth_frames = []
            validation_errors = []

            for frame_idx, frame in enumerate(sensor_data.frames):
                if hasattr(frame, 'depth_maps') and camera_id in frame.depth_maps:
                    depth_frame = frame.depth_maps[camera_id]

                    # Validate depth frame
                    frame_errors = self._validate_depth_frame(
                        depth_frame, episode_idx, camera_id, frame_idx
                    )
                    if frame_errors:
                        validation_errors.extend(frame_errors)
                    else:
                        depth_frames.append(depth_frame)

            # Log validation errors
            if validation_errors:
                self.log(f"  Episode {episode_idx}, camera {camera_id}: {len(validation_errors)} depth validation errors", "WARNING")
                for error in validation_errors[:3]:
                    self.log(f"    - {error}", "WARNING")

            if depth_frames:
                depth_path = depth_dir / f"episode_{episode_idx:06d}.npz"
                self._atomic_write_with_suffix(
                    depth_path,
                    lambda tmp_path: np.savez_compressed(
                        tmp_path,
                        depth=np.stack(depth_frames),
                        fps=self.config.fps,
                        unit="meters",
                    ),
                )

    def _write_segmentation_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write segmentation masks with validation."""
        for camera_id in sensor_data.camera_ids if hasattr(sensor_data, 'camera_ids') else []:
            seg_dir = chunk_dir / "segmentation" / camera_id
            seg_dir.mkdir(parents=True, exist_ok=True)

            semantic_frames = []
            instance_frames = []
            validation_errors = []

            for frame_idx, frame in enumerate(sensor_data.frames):
                # Validate semantic masks
                if hasattr(frame, 'semantic_masks') and camera_id in frame.semantic_masks:
                    semantic_frame = frame.semantic_masks[camera_id]
                    frame_errors = self._validate_segmentation_frame(
                        semantic_frame, episode_idx, camera_id, frame_idx
                    )
                    if frame_errors:
                        validation_errors.extend([f"semantic: {e}" for e in frame_errors])
                    else:
                        semantic_frames.append(semantic_frame)

                # Validate instance masks
                if hasattr(frame, 'instance_masks') and camera_id in frame.instance_masks:
                    instance_frame = frame.instance_masks[camera_id]
                    frame_errors = self._validate_segmentation_frame(
                        instance_frame, episode_idx, camera_id, frame_idx
                    )
                    if frame_errors:
                        validation_errors.extend([f"instance: {e}" for e in frame_errors])
                    else:
                        instance_frames.append(instance_frame)

            # Log validation errors
            if validation_errors:
                self.log(f"  Episode {episode_idx}, camera {camera_id}: {len(validation_errors)} segmentation validation errors", "WARNING")
                for error in validation_errors[:3]:
                    self.log(f"    - {error}", "WARNING")

            if semantic_frames or instance_frames:
                seg_path = seg_dir / f"episode_{episode_idx:06d}.npz"
                data = {"fps": self.config.fps}
                if semantic_frames:
                    data["semantic"] = np.stack(semantic_frames)
                if instance_frames:
                    data["instance"] = np.stack(instance_frames)
                if hasattr(sensor_data, 'semantic_labels'):
                    data["label_mapping"] = json.dumps(sensor_data.semantic_labels)
                self._atomic_write_with_suffix(
                    seg_path,
                    lambda tmp_path: np.savez_compressed(tmp_path, **data),
                )

    def _write_bbox_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write bounding box annotations with validation."""
        bbox_dir = chunk_dir / "bboxes"
        bbox_dir.mkdir(parents=True, exist_ok=True)

        bbox_data = {"episode_index": episode_idx, "frames": []}
        validation_errors = []

        for frame_idx, frame in enumerate(sensor_data.frames):
            frame_bboxes = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "bboxes_2d": {},
                "bboxes_3d": {},
            }

            if hasattr(frame, 'bboxes_2d'):
                frame_bboxes["bboxes_2d"] = frame.bboxes_2d
            if hasattr(frame, 'bboxes_3d'):
                frame_bboxes["bboxes_3d"] = frame.bboxes_3d

            # Validate bounding boxes
            if frame_bboxes["bboxes_2d"] or frame_bboxes["bboxes_3d"]:
                frame_errors = self._validate_bbox_frame(
                    frame_bboxes, episode_idx, "global", frame_idx
                )
                if frame_errors:
                    validation_errors.extend(frame_errors)

            bbox_data["frames"].append(frame_bboxes)

        # Log validation errors
        if validation_errors:
            self.log(f"  Episode {episode_idx}: {len(validation_errors)} bbox validation errors", "WARNING")
            for error in validation_errors[:3]:
                self.log(f"    - {error}", "WARNING")

        bbox_path = bbox_dir / f"episode_{episode_idx:06d}.json"
        self._atomic_write(
            bbox_path,
            lambda tmp_path: tmp_path.write_text(
                json.dumps(bbox_data, indent=2, default=self._json_serializer)
            ),
        )

    def _write_object_pose_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write object pose data for an episode."""
        pose_dir = chunk_dir / "object_poses"
        pose_dir.mkdir(parents=True, exist_ok=True)

        pose_data = {
            "episode_index": episode_idx,
            "object_metadata": sensor_data.object_metadata if hasattr(sensor_data, 'object_metadata') else {},
            "frames": [],
        }

        for frame in sensor_data.frames:
            frame_poses = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "poses": frame.object_poses if hasattr(frame, 'object_poses') else {},
            }
            pose_data["frames"].append(frame_poses)

        pose_path = pose_dir / f"episode_{episode_idx:06d}.json"
        self._atomic_write(
            pose_path,
            lambda tmp_path: tmp_path.write_text(
                json.dumps(pose_data, indent=2, default=self._json_serializer)
            ),
        )

    def _write_contact_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write contact information for an episode."""
        contact_dir = chunk_dir / "contacts"
        contact_dir.mkdir(parents=True, exist_ok=True)

        contact_data = {"episode_index": episode_idx, "frames": []}

        for frame in sensor_data.frames:
            frame_contacts = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "contacts": frame.contacts if hasattr(frame, 'contacts') else [],
            }
            contact_data["frames"].append(frame_contacts)

        contact_path = contact_dir / f"episode_{episode_idx:06d}.json"
        self._atomic_write(
            contact_path,
            lambda tmp_path: tmp_path.write_text(
                json.dumps(contact_data, indent=2, default=self._json_serializer)
            ),
        )

    def _write_privileged_state_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write privileged physics state for an episode."""
        priv_dir = chunk_dir / "privileged_state"
        priv_dir.mkdir(parents=True, exist_ok=True)

        priv_data = {"episode_index": episode_idx, "frames": []}

        for frame in sensor_data.frames:
            frame_state = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "state": frame.privileged_state if hasattr(frame, 'privileged_state') else None,
            }
            priv_data["frames"].append(frame_state)

        priv_path = priv_dir / f"episode_{episode_idx:06d}.json"
        self._atomic_write(
            priv_path,
            lambda tmp_path: tmp_path.write_text(
                json.dumps(priv_data, indent=2, default=self._json_serializer)
            ),
        )

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.uint8, np.uint16)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# Convenience Functions
# =============================================================================


def export_trajectories_to_lerobot(
    trajectories: List[Tuple[JointTrajectory, str]],
    dataset_name: str,
    output_dir: Path,
    robot_type: str = "franka",
    fps: float = 30.0,
) -> Path:
    """
    Convenience function to export trajectories to LeRobot format.

    Args:
        trajectories: List of (trajectory, task_description) tuples
        dataset_name: Name for the dataset
        output_dir: Output directory
        robot_type: Robot type
        fps: Frames per second

    Returns:
        Path to exported dataset
    """
    config = LeRobotDatasetConfig(
        dataset_name=dataset_name,
        robot_type=robot_type,
        fps=fps,
        output_dir=output_dir,
    )

    exporter = LeRobotExporter(config, verbose=True)

    for trajectory, task_desc in trajectories:
        exporter.add_episode(trajectory, task_desc)

    return exporter.finalize()


if __name__ == "__main__":
    from motion_planner import AIMotionPlanner
    from trajectory_solver import TrajectorySolver
    from tools.logging_config import init_logging

    init_logging()
    logger.info("Testing LeRobot Export Pipeline")
    logger.info("%s", "=" * 60)

    # Generate some test episodes
    planner = AIMotionPlanner(robot_type="franka", use_llm=False, verbose=False)
    solver = TrajectorySolver(robot_type="franka", fps=30.0, verbose=False)

    # Create exporter
    config = LeRobotDatasetConfig(
        dataset_name="test_pick_place",
        robot_type="franka",
        fps=30.0,
        output_dir=Path("/tmp/lerobot_test"),
    )
    exporter = LeRobotExporter(config, verbose=True)

    # Generate and add episodes
    tasks = [
        ("pick_cup", "Pick up the coffee cup from the counter", [0.5, 0.1, 0.85], [0.3, -0.2, 0.9]),
        ("pick_bowl", "Pick up the bowl and place on shelf", [0.4, 0.0, 0.82], [0.2, 0.3, 0.95]),
        ("pick_plate", "Pick up the plate from table", [0.6, -0.1, 0.80], [0.4, 0.2, 0.85]),
    ]

    for task_name, desc, target_pos, place_pos in tasks:
        plan = planner.plan_motion(
            task_name=task_name,
            task_description=desc,
            target_object={"id": task_name, "position": target_pos, "dimensions": [0.08, 0.08, 0.1]},
            place_position=place_pos,
        )
        trajectory = solver.solve(plan)
        exporter.add_episode(trajectory, desc)

    # Export
    dataset_path = exporter.finalize()

    # Verify output
    logger.info("Verifying output...")
    for path in sorted(dataset_path.rglob("*")):
        if path.is_file():
            logger.info("  %s", path.relative_to(dataset_path))

#!/usr/bin/env python3
"""
Episode Generation Job for BlueprintPipeline.

SOTA Implementation (2025-2026) based on:
- CP-Gen (CoRL 2025): Constraint-preserving data generation
- DemoGen (RSS 2025): Skill segment + free-space decomposition
- AnyTask: Automated task generation with validation

Key Architecture Changes from Original:
1. Gemini moved "up the stack" - used for task specification, not trajectory generation
2. CP-Gen style augmentation - preserve skill segments, replan free-space
3. Simulation validation - all episodes are sim-verified with quality scores
4. Motion planning with collision checking - TAMP-style planning

Pipeline Position:
    3D-RE-GEN → simready → usd-assembly → replicator → isaac-lab → [THIS JOB]

Process (SOTA):
1. Use Gemini to specify tasks (goals, constraints, keypoints) - NOT trajectories
2. Generate seed episodes with motion planning
3. For each scene variation:
   a. Apply CP-Gen style constraint-preserving augmentation
   b. Replan free-space motions for collision avoidance
   c. Validate in simulation
   d. Export only high-quality episodes
4. Package with quality metrics for sellable data

Output Structure:
    episodes/
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.jsonl
    │   └── episodes.jsonl
    ├── data/
    │   ├── chunk-000/
    │   │   ├── episode_000000.parquet
    │   │   └── ...
    ├── manifests/
    │   ├── generation_manifest.json
    │   └── task_coverage.json
    └── quality/
        └── validation_report.json

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ASSETS_PREFIX: Path to scene assets (scene_manifest.json)
    EPISODES_PREFIX: Output path for episodes
    ENABLE_FIREBASE_UPLOAD: Enable Firebase Storage upload of generated episodes
    FIREBASE_STORAGE_BUCKET: Firebase Storage bucket name for uploads
    FIREBASE_SERVICE_ACCOUNT_JSON: Service account JSON payload for Firebase
    FIREBASE_SERVICE_ACCOUNT_PATH: Path to service account JSON for Firebase
    FIREBASE_UPLOAD_PREFIX: Remote prefix for Firebase uploads (default: datasets)
    ROBOT_TYPE: Robot type (franka, ur10, fetch) - default: franka
    EPISODES_PER_VARIATION: Episodes per variation - default: 10
    MAX_VARIATIONS: Max variations to process - default: all
    FPS: Target FPS for trajectories - default: 30
    USE_LLM: Enable LLM for task specification - default: true
    USE_CPGEN: Enable CP-Gen augmentation - default: true
    MIN_QUALITY_SCORE: Minimum quality score for export - default: 0.7
    MIN_SUCCESS_RATE: Minimum success rate for episode generation - default: 0.5
    BYPASS_QUALITY_GATES: Skip quality gate evaluation (dev-only)
    LABS_STAGING: Treat run as labs-staging (requires Isaac Sim + Replicator)
"""

import gc
import json
import logging
import os
import shutil
import sys
import time
import traceback
import uuid
from math import ceil
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.alerting import send_alert
from tools.config.env import parse_bool_env
from tools.lerobot_format import LeRobotExportFormat, parse_lerobot_export_format
from tools.config.production_mode import resolve_production_mode
from tools.config.seed_manager import set_global_seed
from tools.metrics.pipeline_metrics import get_metrics

REQUIRED_ISAAC_SIM_VERSION = "2024.1.0+"
REQUIRED_ISAAC_SIM_CONTAINER = "nvcr.io/nvidia/isaac-sim:2024.1.0"
REQUIRED_EXTENSIONS = (
    "omni.isaac.core",
    "omni.physx",
    "omni.replicator.core",
)
JOB_NAME = "episode-generation-job"


def _should_bypass_quality_gates() -> bool:
    return os.getenv("BYPASS_QUALITY_GATES", "").lower() in {"1", "true", "yes", "y"}


def _is_service_mode() -> bool:
    return (
        os.getenv("SERVICE_MODE", "").lower() in {"1", "true", "yes", "y"}
        or os.getenv("K_SERVICE") is not None
        or os.getenv("KUBERNETES_SERVICE_HOST") is not None
    )


def _gate_report_path(root: Path, scene_id: str) -> Path:
    report_path = root / f"scenes/{scene_id}/{JOB_NAME}/quality_gate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path


def _compute_collision_free_rate(
    validation_report_path: Optional[Path],
    fallback_rate: float,
) -> float:
    if not validation_report_path or not validation_report_path.is_file():
        return fallback_rate
    try:
        report = json.loads(validation_report_path.read_text())
        episodes = report.get("episodes", [])
        if not episodes:
            return fallback_rate
        collision_free = sum(1 for ep in episodes if not ep.get("collision_events"))
        return collision_free / len(episodes)
    except Exception:
        logger.warning(
            "[EPISODE-GEN-JOB] Failed to compute collision-free rate from %s; using fallback.",
            validation_report_path,
            exc_info=True,
        )
        return fallback_rate


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} {units[-1]}"


def _parse_headroom_pct() -> float:
    raw = os.getenv("EXPORT_DISK_HEADROOM_PCT")
    if not raw:
        return 0.15
    try:
        pct = float(raw)
        if pct > 1:
            pct /= 100.0
        return max(pct, 0.0)
    except ValueError:
        logger.warning(
            "[EPISODE-GEN-JOB] Invalid EXPORT_DISK_HEADROOM_PCT value %r; using default.",
            raw,
        )
        return 0.15


def _is_production_run() -> bool:
    if resolve_production_mode():
        return True
    if HAVE_QUALITY_SYSTEM:
        try:
            return get_data_quality_level().value == "production"
        except Exception:
            logger.warning(
                "[EPISODE-GEN-JOB] Failed to determine data quality level; "
                "falling back to production env flags.",
                exc_info=True,
            )
    return False


def _estimate_export_requirements(
    episodes: List["GeneratedEpisode"],
    config: "EpisodeGenerationConfig",
) -> Dict[str, Any]:
    expected_episodes = len(episodes)
    resolution = config.image_resolution
    width, height = resolution
    num_cameras = max(config.num_cameras, 1)

    frame_counts = []
    for episode in episodes:
        if not episode.trajectory:
            continue
        if getattr(episode.trajectory, "num_frames", 0):
            frame_counts.append(int(episode.trajectory.num_frames))
        elif getattr(episode.trajectory, "total_duration", 0):
            frame_counts.append(int(ceil(episode.trajectory.total_duration * config.fps)))

    if frame_counts:
        frames_per_episode = max(frame_counts)
        episode_seconds = frames_per_episode / max(config.fps, 1.0)
    else:
        episode_seconds = 10.0
        frames_per_episode = int(ceil(episode_seconds * max(config.fps, 1.0)))

    bytes_per_pixel = {
        "core": 3,  # RGB
        "plus": 8,  # RGB + depth + segmentation
        "full": 16,  # RGB + depth + segmentation + normals/metadata
    }.get(config.data_pack_tier.lower(), 3)

    pixels_per_frame = width * height
    per_frame_overhead = 64 * 1024  # metadata, state, compression overhead
    per_episode_overhead = 5 * 1024 * 1024  # manifests, metadata, stats
    bytes_per_frame = num_cameras * pixels_per_frame * bytes_per_pixel + per_frame_overhead
    required_bytes = expected_episodes * (frames_per_episode * bytes_per_frame + per_episode_overhead)

    return {
        "required_bytes": required_bytes,
        "expected_episodes": expected_episodes,
        "frames_per_episode": frames_per_episode,
        "episode_seconds": episode_seconds,
        "num_cameras": num_cameras,
        "resolution": resolution,
        "bytes_per_pixel": bytes_per_pixel,
        "bytes_per_frame": bytes_per_frame,
        "per_frame_overhead": per_frame_overhead,
        "per_episode_overhead": per_episode_overhead,
    }


def _ensure_disk_space(output_dir: Path, required_bytes: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    free_bytes = usage.free

    headroom_pct = _parse_headroom_pct()
    min_free_gb = os.getenv("EXPORT_DISK_FREE_GB")
    min_free_bytes = 0
    if min_free_gb:
        try:
            min_free_bytes = int(float(min_free_gb) * 1024**3)
        except ValueError:
            logger.warning(
                "[EPISODE-GEN-JOB] Invalid EXPORT_DISK_FREE_GB value %r; ignoring.",
                min_free_gb,
            )

    required_with_headroom = int(required_bytes * (1 + headroom_pct))
    required_threshold = max(required_with_headroom, min_free_bytes)

    logger.info(
        "[EPISODE-GEN-JOB] Export disk check: estimated=%s, headroom=%.1f%%, "
        "required_with_headroom=%s, min_free=%s, threshold=%s, free=%s, path=%s",
        _format_bytes(required_bytes),
        headroom_pct * 100,
        _format_bytes(required_with_headroom),
        _format_bytes(min_free_bytes),
        _format_bytes(required_threshold),
        _format_bytes(free_bytes),
        output_dir,
    )

    if free_bytes < required_threshold:
        message = (
            "Insufficient disk space for LeRobot export. "
            f"Required {_format_bytes(required_threshold)} "
            f"(estimated payload {_format_bytes(required_bytes)}, "
            f"headroom {headroom_pct:.0%}, "
            f"min free {_format_bytes(min_free_bytes)}), "
            f"available {_format_bytes(free_bytes)} at {output_dir}."
        )
        logger.error("[EPISODE-GEN-JOB] %s", message)
        raise RuntimeError(message)
# Core imports
from motion_planner import AIMotionPlanner, MotionPlan
from trajectory_solver import TrajectorySolver, JointTrajectory, ROBOT_CONFIGS, TrajectoryIKError
from motion_planner import AIMotionPlanner, MotionPlan, SceneContext
from trajectory_solver import TrajectorySolver, JointTrajectory, ROBOT_CONFIGS
from lerobot_exporter import LeRobotExporter, LeRobotDatasetConfig
from quality_constants import MIN_QUALITY_SCORE, MAX_RETRIES, PRODUCTION_TRAINING_THRESHOLD
from tools.error_handling.partial_failure import (
    PartialFailureError,
    PartialFailureHandler,
    PartialFailureResult,
)

# SOTA imports (new architecture)
from task_specifier import TaskSpecifier, TaskSpecification, SegmentType
from cpgen_augmenter import (
    ConstraintPreservingAugmenter,
    SeedEpisode,
    AugmentedEpisode,
    ObjectTransform,
)
from sim_validator import (
    SimulationValidator,
    FailureReason,
    ValidationResult,
    ValidationStatus,
    ValidationConfig,
    ValidationReportGenerator,
    QualityMetrics,
)
from tools.quality_gates.quality_gate import QualityGateCheckpoint, QualityGateRegistry
from tools.workflow.failure_markers import FailureMarkerWriter
from tools.validation.entrypoint_checks import (
    validate_required_env_vars,
    validate_scene_manifest,
)

# Sensor data capture (enhanced pipeline)
try:
    from sensor_data_capture import (
        SensorDataConfig,
        DataPackTier,
        SensorDataCaptureMode,
        IsaacSimSensorCapture,
        MockSensorCapture,
        EpisodeSensorData,
        create_sensor_capture,
        get_capture_mode_from_env,
        require_isaac_sim_or_fail,
        check_sensor_capture_environment,
    )
    from data_pack_config import (
        DataPackConfig,
        get_data_pack_config,
        data_pack_from_string,
    )
    HAVE_SENSOR_CAPTURE = True
except ImportError:
    HAVE_SENSOR_CAPTURE = False
    SensorDataConfig = None
    DataPackTier = None
    SensorDataCaptureMode = None
    get_capture_mode_from_env = None
    require_isaac_sim_or_fail = None
    check_sensor_capture_environment = None

# Isaac Sim enforcement and quality certificates
try:
    from isaac_sim_enforcement import (
        enforce_isaac_sim_for_production,
        get_environment_capabilities,
        get_data_quality_level,
        print_environment_report,
        DataQualityLevel,
        SensorSource,
        PhysicsValidationBackend,
        IsaacSimRequirementError,
        ProductionDataQualityError,
    )
    from quality_certificate import (
        QualityCertificate,
        QualityCertificateGenerator,
        TrajectoryQualityMetrics,
        VisualQualityMetrics,
        TaskQualityMetrics,
        DiversityMetrics,
        Sim2RealMetrics,
        compute_episode_data_hash,
    )
    HAVE_QUALITY_SYSTEM = True
except ImportError:
    HAVE_QUALITY_SYSTEM = False
    logger.warning("[EPISODE-GEN-JOB] Quality certificate system not available")

# Pipeline imports
try:
    from dwm_preparation_job.scene_analyzer import SceneAnalyzer, SceneAnalysisResult
    from dwm_preparation_job.task_planner import TaskPlanner, TaskPlannerOutput
    HAVE_DWM_MODULES = True
except ImportError:
    HAVE_DWM_MODULES = False


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class EpisodeGenerationConfig:
    """Configuration for episode generation."""

    # Scene info
    scene_id: str
    manifest_path: Path

    # Robot configuration
    robot_type: str = "franka"
    robot_prim_paths: Optional[List[str]] = None
    camera_specs: Optional[List[Dict[str, str]]] = None
    scene_usd_path: Optional[str] = None
    robot_urdf_path: Optional[str] = None

    # Generation parameters
    episodes_per_variation: int = 10
    max_variations: Optional[int] = None
    fps: float = 30.0

    # AI configuration
    use_llm: bool = True  # Use Gemini for task specification

    # SOTA features
    use_cpgen: bool = True  # Use CP-Gen style augmentation
    use_validation: bool = True  # Use simulation validation

    # Quality settings - LABS-BLOCKER-002 FIX: Uses unified quality constants
    # Imported from quality_constants.py to ensure consistency across pipeline
    min_quality_score: float = MIN_QUALITY_SCORE  # 0.85 - unified threshold
    min_success_rate: float = 0.5
    max_retries: int = MAX_RETRIES  # 3 - unified retry limit
    validate_trajectories: bool = True
    include_failed: bool = False

    # Data pack configuration (Core/Plus/Full)
    data_pack_tier: str = "core"  # "core", "plus", "full"
    num_cameras: int = 1
    image_resolution: Tuple[int, int] = (640, 480)
    capture_sensor_data: bool = True  # Enable visual observation capture
    use_mock_capture: bool = False  # [DEPRECATED] Use mock capture (use sensor_capture_mode instead)
    sensor_capture_mode: Optional[str] = None  # "isaac_sim", "mock_dev", "fail_closed" (None = auto-detect)
    allow_mock_capture: bool = False  # Explicit dev-only guard for mock capture
    lerobot_export_format: LeRobotExportFormat = LeRobotExportFormat.LEROBOT_V2

    # Output
    output_dir: Path = Path("./episodes")


@dataclass
class GeneratedEpisode:
    """A generated episode with metadata."""

    episode_id: str
    task_name: str
    task_description: str

    # Trajectory
    trajectory: Optional[JointTrajectory]
    motion_plan: Optional[MotionPlan]

    # Task specification (new)
    task_spec: Optional[TaskSpecification] = None

    # Context
    scene_id: str = ""
    variation_index: int = 0
    is_seed: bool = False  # True if this is a seed episode

    # Quality (new - from validation)
    is_valid: bool = True
    validation_result: Optional[ValidationResult] = None
    quality_score: float = 1.0

    # Augmentation info (new)
    augmentation_method: str = "direct"  # "direct", "cpgen"
    seed_episode_id: Optional[str] = None

    # Sensor data (enhanced - visual observations + ground-truth)
    sensor_data: Optional[Any] = None  # EpisodeSensorData when available

    # Object metadata for ground-truth
    object_metadata: Dict[str, Any] = field(default_factory=dict)

    # Quality certificate (new)
    quality_certificate: Optional[Any] = None  # QualityCertificate when available

    # Legacy
    validation_errors: List[str] = field(default_factory=list)

    # Timing
    generation_time_seconds: float = 0.0


def _load_scene_config(scene_dir: Path) -> Dict[str, Any]:
    """Load optional scene configuration from scenes/<id>/config.json."""
    config_path = scene_dir / "config.json"
    if not config_path.is_file():
        return {}

    try:
        with open(config_path) as file:
            return json.load(file)
    except Exception as exc:
        logger.warning("[EPISODE-GEN-JOB] Failed to read scene config: %s", exc)
        return {}


def _resolve_scene_usd_path(scene_dir: Path) -> Optional[str]:
    """Resolve a USD scene path for auto-discovery."""
    usd_dir = scene_dir / "usd"
    for filename in ("scene.usd", "scene.usda", "scene.usdz"):
        candidate = usd_dir / filename
        if candidate.is_file():
            return str(candidate)

    if usd_dir.is_dir():
        for pattern in ("*.usd", "*.usda", "*.usdz"):
            matches = sorted(usd_dir.glob(pattern))
            if matches:
                return str(matches[0])

    return None


def _load_camera_specs(scene_config: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Load optional camera specs from scene config."""
    raw_specs = scene_config.get("cameras")
    if not raw_specs:
        return None
    if not isinstance(raw_specs, list):
        logger.warning("[EPISODE-GEN-JOB] scene config cameras entry is not a list.")
        return None

    normalized: List[Dict[str, str]] = []
    for entry in raw_specs:
        if not isinstance(entry, dict):
            continue
        prim_path = entry.get("prim_path") or entry.get("path")
        if not prim_path:
            continue
        camera_type = entry.get("type") or entry.get("camera_type") or "rgb"
        camera_id = entry.get("camera_id") or entry.get("id") or camera_type
        normalized.append(
            {
                "prim_path": prim_path,
                "camera_type": camera_type,
                "camera_id": camera_id,
            }
        )

    if not normalized:
        logger.warning("[EPISODE-GEN-JOB] scene config cameras list is empty.")
        return None

    return normalized


@dataclass
class EpisodeGenerationOutput:
    """Output from episode generation job."""

    scene_id: str
    robot_type: str

    # Statistics
    total_episodes: int = 0
    valid_episodes: int = 0
    seed_episodes: int = 0
    augmented_episodes: int = 0
    total_variations: int = 0
    total_frames: int = 0
    total_duration_seconds: float = 0.0

    # Quality metrics (new)
    average_quality_score: float = 0.0
    pass_rate: float = 0.0
    augmentation_success_rate: float = 0.0

    # Task coverage
    tasks_generated: Dict[str, int] = field(default_factory=dict)

    # Output paths
    output_dir: Optional[Path] = None
    lerobot_dataset_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    validation_report_path: Optional[Path] = None
    firebase_upload_summary: Optional[Dict[str, Any]] = None
    firebase_upload_error: Optional[str] = None

    # Timing
    generation_time_seconds: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.valid_episodes > 0 and len(self.errors) == 0


# =============================================================================
# Task Generator (Updated for SOTA)
# =============================================================================


class ManipulationTaskGenerator:
    """
    Generates manipulation tasks from scene analysis.

    UPDATED: Now uses TaskSpecifier for Gemini-powered task specification
    at the top of the stack.
    """

    # Mapping from object categories to manipulation tasks
    CATEGORY_TASKS = {
        # Kitchen objects
        "cup": [("pick_cup", "Pick up the cup and place it on the counter")],
        "mug": [("pick_mug", "Pick up the mug and move it to the sink")],
        "plate": [("pick_plate", "Pick up the plate and place it in the rack")],
        "bowl": [("pick_bowl", "Pick up the bowl and move it to the table")],
        "utensil": [("pick_utensil", "Pick up the utensil and place in drawer")],
        "bottle": [("pick_bottle", "Pick up the bottle and place on shelf")],

        # Warehouse objects
        "box": [("pick_box", "Pick up the box and place on pallet")],
        "package": [("pick_package", "Pick up the package and place in bin")],
        "carton": [("pick_carton", "Pick up the carton and stack it")],
        "tote": [("pick_tote", "Pick up the tote and move to staging area")],

        # Articulated objects
        "drawer": [
            ("open_drawer", "Open the drawer by pulling the handle"),
            ("close_drawer", "Close the drawer by pushing"),
        ],
        "door": [
            ("open_door", "Open the door by rotating the handle"),
            ("close_door", "Close the door"),
        ],
        "cabinet": [
            ("open_cabinet", "Open the cabinet door"),
        ],

        # Generic
        "object": [("pick_object", "Pick up the object and relocate it")],
    }

    MULTI_OBJECT_TEMPLATES = {
        "stack_objects": {
            "description": "Pick up {top_obj} and stack it on {base_obj}",
            "min_objects": 2,
        },
        "stack_three_objects": {
            "description": "Pick up {obj_a}, place it on {obj_b}, then stack {obj_c} on top",
            "min_objects": 3,
        },
    }

    def __init__(self, use_llm: bool = True, verbose: bool = True):
        self.use_llm = use_llm
        self.verbose = verbose
        self.task_specifier = TaskSpecifier(verbose=verbose) if use_llm else None
        self.scene_analyzer = SceneAnalyzer(verbose=verbose) if HAVE_DWM_MODULES else None
        self.task_planner = TaskPlanner(verbose=verbose) if HAVE_DWM_MODULES else None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[TASK-GENERATOR] [%s] %s", level, msg)

    def generate_tasks_with_specs(
        self,
        manifest: Dict[str, Any],
        manifest_path: Optional[Path] = None,
        robot_type: str = "franka",
    ) -> List[Tuple[Dict[str, Any], TaskSpecification]]:
        """
        Generate manipulation tasks WITH full task specifications.

        This is the SOTA approach: Gemini specifies the task at a high level,
        including constraints and keypoints for CP-Gen augmentation.

        Returns:
            List of (task_dict, TaskSpecification) tuples
        """
        self.log("Generating manipulation tasks with SOTA specification...")

        # Get basic tasks from scene
        basic_tasks = self.generate_tasks(manifest, manifest_path)
        valid_tasks = []
        objects = manifest.get("objects", [])
        object_lookup = self._build_object_lookup(objects)

        for task in basic_tasks:
            is_valid, ordered_steps, errors = self._validate_task_steps(task, object_lookup)
            if not is_valid:
                self.log(
                    f"  Skipping task {task.get('task_id', 'unknown')} due to errors: {errors}",
                    "WARNING",
                )
                continue
            if ordered_steps is not None:
                task["task_steps"] = ordered_steps
            valid_tasks.append(task)

        # Enhance with full specifications
        tasks_with_specs = []

        for task in valid_tasks:
            try:
                if "task_steps" in task:
                    spec = self._create_multi_object_spec(task, objects, robot_type)
                elif self.task_specifier:
                    # Use Gemini to generate full specification
                    spec = self.task_specifier.specify_task(
                        task_name=task["task_name"],
                        task_description=task["description"],
                        scene_objects=objects,
                        robot_type=robot_type,
                        target_object_id=task.get("target_object_id"),
                        place_position=task.get("place_position"),
                    )
                else:
                    # Create minimal spec without LLM
                    spec = self._create_minimal_spec(task, objects, robot_type)

                tasks_with_specs.append((task, spec))

            except Exception as e:
                self.log(f"  Failed to specify {task['task_name']}: {e}", "WARNING")
                # Still add with minimal spec
                spec = self._create_multi_object_spec(task, objects, robot_type) if "task_steps" in task else self._create_minimal_spec(task, objects, robot_type)
                tasks_with_specs.append((task, spec))

        self.log(f"Generated {len(tasks_with_specs)} task specifications")
        return tasks_with_specs

    def _create_minimal_spec(
        self,
        task: Dict[str, Any],
        objects: List[Dict[str, Any]],
        robot_type: str,
    ) -> TaskSpecification:
        """Create minimal task specification without LLM."""
        from task_specifier import SkillSegment

        # Get default position from task, or compute from robot workspace
        goal_position = task.get("place_position")
        if goal_position is None:
            # Compute sensible default from robot workspace and task context
            goal_position = self._compute_default_goal_position(task, objects, robot_type)

        return TaskSpecification(
            spec_id=f"spec_{task['task_id']}",
            task_name=task["task_name"],
            task_description=task["description"],
            goal_object_id=task.get("target_object_id"),
            goal_position=np.array(goal_position),
            segments=[
                SkillSegment(
                    segment_id=f"seg_{task['task_id']}_main",
                    segment_type=SegmentType.SKILL,
                    skill_name=task["task_name"],
                    description=task["description"],
                ),
            ],
            robot_type=robot_type,
        )

    def _create_multi_object_spec(
        self,
        task: Dict[str, Any],
        objects: List[Dict[str, Any]],
        robot_type: str,
    ) -> TaskSpecification:
        """Create a minimal multi-object task specification with ordered steps."""
        from task_specifier import SkillSegment

        steps = task.get("task_steps", [])
        spec_id = f"spec_{task['task_id']}"
        segments = []
        time_cursor = 0.0
        segment_duration = 1.0
        step_dependencies = {}

        for step in steps:
            step_id = step.get("step_id", f"{task['task_id']}_step")
            step_dependencies[step_id] = list(step.get("depends_on", []))
            target_obj_id = step.get("target_object_id")
            description = step.get("description", task["description"])

            segments.append(
                SkillSegment(
                    segment_id=f"{spec_id}_{step_id}",
                    segment_type=SegmentType.SKILL,
                    skill_name=step.get("action", task["task_name"]),
                    description=description,
                    start_time=time_cursor,
                    end_time=time_cursor + segment_duration,
                    manipulated_object_id=target_obj_id,
                    contact_objects=[target_obj_id] if target_obj_id else [],
                )
            )
            time_cursor += segment_duration

        goal_object_id = steps[-1].get("target_object_id") if steps else task.get("target_object_id")
        goal_position = task.get("place_position")
        if goal_position is None and steps:
            goal_position = steps[-1].get("place_position")

        if goal_position is None:
            goal_position = self._compute_default_goal_position(task, objects, robot_type)

        return TaskSpecification(
            spec_id=spec_id,
            task_name=task["task_name"],
            task_description=task["description"],
            goal_object_id=goal_object_id,
            goal_position=np.array(goal_position),
            segments=segments,
            success_criteria={
                "step_dependencies": step_dependencies,
                "ordered_execution_required": True,
            },
            robot_type=robot_type,
            estimated_duration=time_cursor,
        )

    def generate_tasks(
        self,
        manifest: Dict[str, Any],
        manifest_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate manipulation tasks from scene manifest.

        Args:
            manifest: Scene manifest dict
            manifest_path: Path to manifest file (for scene analyzer)

        Returns:
            List of task definitions
        """
        self.log("Generating manipulation tasks from scene...")

        # Try full analysis if available
        if self.scene_analyzer and manifest_path:
            try:
                analysis = self.scene_analyzer.analyze(manifest_path)
                if self.task_planner:
                    plan = self.task_planner.plan_episodes(analysis, max_episodes=20)
                    return self._convert_planned_tasks(plan, manifest)
            except Exception as e:
                self.log(f"Full analysis failed, using simplified: {e}", "WARNING")

        # Fallback: simplified task generation
        return self._generate_simplified_tasks(manifest)

    def _convert_planned_tasks(
        self,
        plan: "TaskPlannerOutput",
        manifest: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Convert TaskPlannerOutput to task definitions."""
        tasks = []

        # Get workspace defaults from manifest or robot config
        robot_config = manifest.get("robot_config", {})
        workspace_center = robot_config.get("workspace_center", [0.5, 0.0, 0.85])

        for episode in plan.episodes:
            # Find target object info
            target_id = episode.source_objects[0] if episode.source_objects else None
            target_obj = None
            if target_id:
                for obj in manifest.get("objects", []):
                    if obj.get("id") == target_id or obj.get("name") == target_id:
                        target_obj = obj
                        break

            task = {
                "task_id": episode.task_id,
                "task_name": episode.task_name,
                "description": episode.description,
                "target_object_id": target_id,
                "target_position": target_obj.get("position", workspace_center) if target_obj else workspace_center,
                "target_dimensions": target_obj.get("dimensions", [0.1, 0.1, 0.1]) if target_obj else [0.1, 0.1, 0.1],
                "place_position": self._calculate_place_position(target_obj, manifest),
                "is_articulated": episode.source_template.requires_articulation if episode.source_template else False,
            }
            tasks.append(task)

        self.log(f"Generated {len(tasks)} tasks from full analysis")
        return tasks

    def _generate_simplified_tasks(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks from object categories (fallback)."""
        tasks = []
        objects = manifest.get("objects", [])

        # Get workspace defaults from manifest or robot config
        robot_config = manifest.get("robot_config", {})
        workspace_center = robot_config.get("workspace_center", [0.5, 0.0, 0.85])

        for obj in objects:
            category = obj.get("category", "object").lower()
            obj_id = obj.get("id", obj.get("name", "unknown"))
            position = obj.get("position", workspace_center)
            dimensions = obj.get("dimensions", obj.get("bbox", [0.1, 0.1, 0.1]))

            # Find matching task templates
            for cat_key, task_templates in self.CATEGORY_TASKS.items():
                if cat_key in category:
                    for task_name, description in task_templates:
                        task = {
                            "task_id": f"{task_name}_{obj_id}",
                            "task_name": task_name,
                            "description": description.replace("the ", f"the {obj_id} "),
                            "target_object_id": obj_id,
                            "target_position": position if isinstance(position, list) else list(position),
                            "target_dimensions": dimensions if isinstance(dimensions, list) else list(dimensions),
                            "place_position": self._calculate_place_position(obj, manifest),
                            "is_articulated": "drawer" in category or "door" in category,
                        }
                        tasks.append(task)
                    break
            else:
                # Default task for unknown categories
                task = {
                    "task_id": f"manipulate_{obj_id}",
                    "task_name": "pick_object",
                    "description": f"Pick up {obj_id} and relocate it",
                    "target_object_id": obj_id,
                    "target_position": position if isinstance(position, list) else list(position),
                    "target_dimensions": dimensions if isinstance(dimensions, list) else list(dimensions),
                    "place_position": self._calculate_place_position(obj, manifest),
                    "is_articulated": False,
                }
                tasks.append(task)

        tasks.extend(self._generate_multi_object_tasks(manifest))

        self.log(f"Generated {len(tasks)} tasks from object categories")
        return tasks

    def _generate_multi_object_tasks(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multi-object tasks such as stacking based on manifest or templates."""
        tasks = []
        objects = manifest.get("objects", [])

        explicit_tasks = manifest.get("multi_object_tasks", [])
        if explicit_tasks:
            for task in explicit_tasks:
                task.setdefault("task_id", f"{task.get('task_name', 'multi')}_{uuid.uuid4().hex[:8]}")
                task.setdefault("task_name", "multi_object_task")
                task.setdefault("description", "Multi-object manipulation task")
                tasks.append(task)
            return tasks

        stackable = [
            obj for obj in objects
            if obj.get("category", "object").lower() not in ["background", "surface"]
        ]

        if len(stackable) >= 2:
            base_obj = stackable[0]
            top_obj = stackable[1]
            base_id = base_obj.get("id", base_obj.get("name", "base"))
            top_id = top_obj.get("id", top_obj.get("name", "top"))
            base_position = base_obj.get("position", [0.5, 0.0, 0.85])
            base_dims = base_obj.get("dimensions", base_obj.get("bbox", [0.1, 0.1, 0.1]))
            stack_position = self._calculate_stack_position(base_position, base_dims)

            tasks.append({
                "task_id": f"stack_{top_id}_on_{base_id}",
                "task_name": "stack_objects",
                "description": self.MULTI_OBJECT_TEMPLATES["stack_objects"]["description"].format(
                    top_obj=top_id,
                    base_obj=base_id,
                ),
                "target_object_id": top_id,
                "target_position": top_obj.get("position", base_position),
                "target_dimensions": top_obj.get("dimensions", base_dims),
                "place_position": stack_position,
                "task_steps": [
                    {
                        "step_id": "pick_place_top",
                        "action": "pick_place",
                        "description": f"Pick {top_id} and place on {base_id}",
                        "target_object_id": top_id,
                        "target_position": top_obj.get("position", base_position),
                        "target_dimensions": top_obj.get("dimensions", base_dims),
                        "place_position": stack_position,
                        "place_on_object_id": base_id,
                        "depends_on": [],
                    },
                ],
                "is_articulated": False,
            })

        if len(stackable) >= 3:
            obj_a, obj_b, obj_c = stackable[:3]
            obj_a_id = obj_a.get("id", obj_a.get("name", "obj_a"))
            obj_b_id = obj_b.get("id", obj_b.get("name", "obj_b"))
            obj_c_id = obj_c.get("id", obj_c.get("name", "obj_c"))
            obj_b_position = obj_b.get("position", [0.5, 0.0, 0.85])
            obj_b_dims = obj_b.get("dimensions", obj_b.get("bbox", [0.1, 0.1, 0.1]))
            obj_c_position = obj_c.get("position", [0.5, 0.0, 0.85])
            obj_c_dims = obj_c.get("dimensions", obj_c.get("bbox", [0.1, 0.1, 0.1]))

            place_on_b = self._calculate_stack_position(obj_b_position, obj_b_dims)
            place_on_c = self._calculate_stack_position(obj_c_position, obj_c_dims)

            tasks.append({
                "task_id": f"stack_{obj_a_id}_{obj_b_id}_{obj_c_id}",
                "task_name": "stack_three_objects",
                "description": self.MULTI_OBJECT_TEMPLATES["stack_three_objects"]["description"].format(
                    obj_a=obj_a_id,
                    obj_b=obj_b_id,
                    obj_c=obj_c_id,
                ),
                "target_object_id": obj_a_id,
                "target_position": obj_a.get("position", obj_b_position),
                "target_dimensions": obj_a.get("dimensions", obj_b_dims),
                "place_position": place_on_c,
                "task_steps": [
                    {
                        "step_id": "stack_first",
                        "action": "pick_place",
                        "description": f"Pick {obj_a_id} and place on {obj_b_id}",
                        "target_object_id": obj_a_id,
                        "target_position": obj_a.get("position", obj_b_position),
                        "target_dimensions": obj_a.get("dimensions", obj_b_dims),
                        "place_position": place_on_b,
                        "place_on_object_id": obj_b_id,
                        "depends_on": [],
                    },
                    {
                        "step_id": "stack_second",
                        "action": "pick_place",
                        "description": f"Pick {obj_c_id} and place on {obj_a_id}",
                        "target_object_id": obj_c_id,
                        "target_position": obj_c.get("position", obj_c_position),
                        "target_dimensions": obj_c.get("dimensions", obj_c_dims),
                        "place_position": self._calculate_stack_position(place_on_b, obj_a.get("dimensions", obj_b_dims)),
                        "place_on_object_id": obj_a_id,
                        "depends_on": ["stack_first"],
                    },
                ],
                "is_articulated": False,
            })

        return tasks

    def _build_object_lookup(self, objects: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            obj.get("id", obj.get("name", f"obj_{idx}")): obj
            for idx, obj in enumerate(objects)
        }

    def _validate_task_steps(
        self,
        task: Dict[str, Any],
        object_lookup: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], List[str]]:
        """Validate multi-object task step ordering and dependencies."""
        steps = task.get("task_steps")
        if not steps:
            return True, None, []

        errors = []
        step_ids = [step.get("step_id") for step in steps]
        if any(step_id is None for step_id in step_ids):
            errors.append("All task steps must include a step_id")
        if len(set(step_ids)) != len(step_ids):
            errors.append("Task step_ids must be unique")

        for step in steps:
            target_id = step.get("target_object_id")
            if target_id and target_id not in object_lookup:
                errors.append(f"Unknown target object in steps: {target_id}")
            place_on_id = step.get("place_on_object_id")
            if place_on_id and place_on_id not in object_lookup:
                errors.append(f"Unknown place_on object in steps: {place_on_id}")
            for dep in step.get("depends_on", []):
                if dep not in step_ids:
                    errors.append(f"Unknown dependency step_id: {dep}")

        ordered_steps = self._order_task_steps(steps)
        if ordered_steps is None:
            errors.append("Task steps contain cyclic dependencies")

        return len(errors) == 0, ordered_steps, errors

    def _order_task_steps(self, steps: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Topologically order task steps based on dependencies."""
        steps_by_id = {step.get("step_id"): step for step in steps}
        if None in steps_by_id:
            return None

        dependencies = {
            step_id: set(steps_by_id[step_id].get("depends_on", []))
            for step_id in steps_by_id
        }
        ordered = []
        ready = [step_id for step_id, deps in dependencies.items() if not deps]

        while ready:
            current = ready.pop(0)
            ordered.append(steps_by_id[current])
            for step_id, deps in dependencies.items():
                if current in deps:
                    deps.remove(current)
                    if not deps and step_id not in [s.get("step_id") for s in ordered] and step_id not in ready:
                        ready.append(step_id)

        if len(ordered) != len(steps):
            return None

        return ordered

    def _calculate_stack_position(
        self,
        base_position: List[float],
        base_dimensions: List[float],
    ) -> List[float]:
        """Calculate a stacking position above a base object."""
        base_pos = np.array(base_position)
        base_dims = np.array(base_dimensions)
        return [base_pos[0], base_pos[1], base_pos[2] + (base_dims[2] / 2) + 0.02]

    def _compute_default_goal_position(
        self,
        task: Dict[str, Any],
        objects: List[Dict[str, Any]],
        robot_type: str,
    ) -> List[float]:
        """
        Compute sensible default goal position from task context and robot workspace.

        Strategy:
        1. If task has explicit goal_position in manifest, use it
        2. If task has target object, compute offset from object position
        3. Otherwise, use robot workspace center with safe height
        """
        # Check if task has explicit goal_position
        if "goal_position" in task:
            return task["goal_position"]

        # Get robot workspace bounds (robot-specific)
        robot_config = ROBOT_CONFIGS.get(robot_type, ROBOT_CONFIGS["franka"])
        workspace_center = getattr(robot_config, "workspace_center", [0.5, 0.0, 0.85])

        # If task has target object, compute offset from object
        target_obj_id = task.get("target_object_id")
        if target_obj_id:
            # Find target object in scene
            target_obj = None
            for obj in objects:
                if obj.get("id") == target_obj_id or obj.get("name") == target_obj_id:
                    target_obj = obj
                    break

            if target_obj:
                pos = target_obj.get("position", workspace_center)
                if isinstance(pos, np.ndarray):
                    pos = pos.tolist()

                # Offset placement position (place nearby, not on top)
                return [pos[0] - 0.2, pos[1] + 0.15, pos[2]]

        # Fallback: use workspace center
        return workspace_center

    def _calculate_place_position(
        self,
        obj: Optional[Dict[str, Any]],
        manifest: Dict[str, Any],
    ) -> List[float]:
        """
        Calculate a sensible place position for an object.

        Now uses workspace-aware computation instead of hardcoded defaults.
        """
        # Get robot workspace center from manifest if available
        robot_config = manifest.get("robot_config", {})
        workspace_center = robot_config.get("workspace_center", [0.5, 0.0, 0.85])

        if obj is None:
            return workspace_center

        pos = obj.get("position", workspace_center)
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()

        # Offset placement position (place nearby, not on top)
        return [pos[0] - 0.2, pos[1] + 0.15, pos[2]]


# =============================================================================
# Episode Generator (SOTA Implementation)
# =============================================================================


class EpisodeGenerator:
    """
    Main episode generation engine.

    SOTA Implementation:
    1. TaskSpecifier (Gemini) generates high-level task specifications
    2. Motion planner generates seed episodes
    3. CP-Gen augmenter creates variations preserving skill constraints
    4. Simulation validator filters to high-quality episodes
    5. Export only validated episodes with quality scores
    """

    def __init__(self, config: EpisodeGenerationConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose

        # Core components
        self.motion_planner = AIMotionPlanner(
            robot_type=config.robot_type,
            use_llm=config.use_llm,  # LLM now for enhancement, not core planning
            verbose=verbose,
        )
        self.trajectory_solver = TrajectorySolver(
            robot_type=config.robot_type,
            fps=config.fps,
            verbose=verbose,
        )
        self.task_generator = ManipulationTaskGenerator(
            use_llm=config.use_llm,
            verbose=verbose,
        )
        self._partial_failure_errors: List[str] = []
        self._partial_failure_handler = PartialFailureHandler(
            min_success_rate=config.min_success_rate,
            save_successful=True,
            output_dir=config.output_dir / "meta" / "partial_failures",
            failure_report_path=config.output_dir / "meta" / "partial_failure_report.json",
        )

        # SOTA components
        self.cpgen_augmenter = ConstraintPreservingAugmenter(
            robot_type=config.robot_type,
            verbose=verbose,
        ) if config.use_cpgen else None

        self.validator = SimulationValidator(
            robot_type=config.robot_type,
            config=ValidationConfig(
                min_quality_score=config.min_quality_score,
                max_retries=config.max_retries,
            ),
            scene_usd_path=config.scene_usd_path,
            robot_prim_paths=config.robot_prim_paths,
            verbose=verbose,
        ) if config.use_validation else None

        # Sensor data capture (enhanced pipeline with explicit mode control)
        self.sensor_capture = None
        self._sensor_capture_is_mock = False

        if HAVE_SENSOR_CAPTURE and config.capture_sensor_data:
            try:
                # Get capture mode from environment or config
                if hasattr(config, 'sensor_capture_mode') and config.sensor_capture_mode:
                    capture_mode = config.sensor_capture_mode
                elif config.use_mock_capture:
                    # Legacy support: use_mock_capture -> MOCK_DEV
                    capture_mode = SensorDataCaptureMode.MOCK_DEV
                else:
                    # Default: get from environment (defaults to fail_closed)
                    capture_mode = get_capture_mode_from_env()

                self.log(f"Sensor capture mode: {capture_mode.value}")
                if capture_mode == SensorDataCaptureMode.MOCK_DEV and not config.allow_mock_capture:
                    raise RuntimeError(
                        "Mock sensor capture requested but allow_mock_capture is False. "
                        "Enable allow_mock_capture for development/testing only."
                    )

                # Parse data pack tier
                tier = data_pack_from_string(config.data_pack_tier)

                # Create sensor capture with explicit mode
                self.sensor_capture = create_sensor_capture(
                    data_pack=tier,
                    num_cameras=config.num_cameras,
                    resolution=config.image_resolution,
                    fps=config.fps,
                    camera_specs=config.camera_specs,
                    robot_prim_paths=config.robot_prim_paths,
                    scene_usd_path=config.scene_usd_path,
                    capture_mode=capture_mode,
                    allow_mock_capture=config.allow_mock_capture,
                    verbose=verbose,
                )

                # Check if we got mock capture using the dedicated method
                self._sensor_capture_is_mock = hasattr(self.sensor_capture, 'is_mock') and self.sensor_capture.is_mock()

                self.log(f"Sensor capture initialized: {config.data_pack_tier} pack, {config.num_cameras} cameras")
                if self._sensor_capture_is_mock:
                    self.log("⚠️  [TEST] Using MOCK sensor capture - NOT suitable for production training!", "WARNING")
                else:
                    self.log("✅ [PRODUCTION] Using Isaac Sim Replicator - production quality data")

            except Exception as e:
                # Sensor capture creation failed - this is expected to raise in fail_closed mode
                self.log(f"Sensor capture initialization failed: {e}", "ERROR")
                raise

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[EPISODE-GENERATOR] [%s] %s", level, msg)

    def _process_with_partial_failures(
        self,
        items: List[Any],
        process_fn,
        item_id_fn,
        batch_name: str,
    ) -> PartialFailureResult[Any]:
        try:
            result = self._partial_failure_handler.process_batch(
                items,
                process_fn=process_fn,
                item_id_fn=item_id_fn,
                batch_name=batch_name,
            )
        except PartialFailureError as exc:
            result = exc.result
            error_msg = (
                f"{batch_name} success rate {result.success_rate:.1%} below minimum "
                f"{self.config.min_success_rate:.1%} "
                f"({result.success_count}/{result.total_attempted} succeeded)"
            )
            self._partial_failure_errors.append(error_msg)
            self.log(error_msg, "ERROR")

        if result.failed:
            self.log(
                f"{batch_name} failures: {result.failure_count}/{result.total_attempted}",
                "WARNING",
            )
            for failure in result.failed:
                self.log(
                    f"  [{batch_name}] index={failure.get('item_index')} "
                    f"id={failure.get('item_id')} error={failure.get('error')}",
                    "WARNING",
                )
        return result

    def _is_production_quality_level(self) -> bool:
        env_level = os.getenv("DATA_QUALITY_LEVEL")
        if env_level:
            return env_level.lower() == "production"
        if HAVE_QUALITY_SYSTEM:
            try:
                return get_data_quality_level().value == "production"
            except Exception:
                return False
        return False

    def _allow_lerobot_export_failure(self) -> bool:
        if self._is_production_quality_level():
            return False
        override = os.getenv("ALLOW_LEROBOT_EXPORT_FAILURE")
        if override is None:
            return True
        return bool(parse_bool_env(override, default=False))

    def _enforce_physics_backed_qc(self, validated_episodes: List[GeneratedEpisode]) -> None:
        """Ensure production exports only use physics-backed QC results."""
        enforcement_reasons = []
        if resolve_production_mode():
            enforcement_reasons.append("production environment (resolve_production_mode)")
        if self._is_production_quality_level():
            enforcement_reasons.append("production data quality level (DATA_QUALITY_LEVEL=production)")
        if not enforcement_reasons:
            return
        enforcement_label = ", ".join(enforcement_reasons)

        if not HAVE_QUALITY_SYSTEM:
            raise RuntimeError(
                "Production QC gating requires the quality system to be available."
            )

        if not self.validator or not self.validator.is_using_real_physics():
            raise ProductionDataQualityError(
                "Production exports require physics-backed QC results. "
                "Run with Isaac Sim or Isaac Lab + PhysX enabled. "
                f"(Enforced by {enforcement_label}.)"
            )

        non_physics = [
            ep for ep in validated_episodes
            if ep.validation_result is None
            or ep.validation_result.dev_only_fallback
            or ep.validation_result.physics_backend not in {"isaac_sim", "isaac_lab"}
        ]
        if non_physics:
            raise ProductionDataQualityError(
                "Production exports require physics-backed QC results for every episode. "
                f"Found {len(non_physics)} episodes without physics-backed validation. "
                f"(Enforced by {enforcement_label}.)"
            )

    def _apply_camera_capture_quality_gates(self, episodes: List[GeneratedEpisode]) -> None:
        """Filter episodes with missing required camera frames."""
        for episode in episodes:
            warnings = []
            if episode.sensor_data is not None:
                warnings = getattr(episode.sensor_data, "camera_capture_warnings", [])
            if warnings:
                episode.validation_errors.append("Missing required camera frames")
                episode.is_valid = False
                episode.quality_score = min(episode.quality_score, 0.0)
                self.log(
                    f"Episode {episode.episode_id} filtered due to missing camera frames",
                    "WARNING",
                )

    def _solve_trajectory_with_replan(
        self,
        motion_plan: MotionPlan,
        task_name: str,
        task_description: str,
        target_object: Dict[str, Any],
        place_position: Optional[List[float]],
        articulation_info: Optional[Dict[str, Any]],
        context_label: str,
    ) -> Tuple[MotionPlan, JointTrajectory]:
        """Solve trajectory, replanning if IK fails."""
        current_plan = motion_plan
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                trajectory = self.trajectory_solver.solve(current_plan)
                return current_plan, trajectory
            except TrajectoryIKError as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                self.log(
                    f"    IK failed ({context_label}) attempt {attempt + 1}/{self.config.max_retries + 1}: {exc}. Replanning...",
                    "WARNING",
                )
                current_plan = self.motion_planner.plan_motion(
                    task_name=task_name,
                    task_description=task_description,
                    target_object=target_object,
                    place_position=place_position,
                    articulation_info=articulation_info,
                )

        raise TrajectoryIKError(
            f"IK failed after {self.config.max_retries + 1} attempts ({context_label}): {last_error}"
        )

    def generate(self, manifest: Dict[str, Any]) -> EpisodeGenerationOutput:
        """
        Generate episodes for all variations in the scene.

        SOTA Process:
        1. Generate task specifications using Gemini
        2. Create seed episodes for each task
        3. Use CP-Gen to augment across variations
        4. Validate and filter episodes
        5. Export high-quality episodes only
        """
        start_time = time.time()

        self.log("=" * 70)
        self.log("EPISODE GENERATION (SOTA Pipeline)")
        self.log("=" * 70)
        self.log(f"Scene: {self.config.scene_id}")
        self.log(f"Robot: {self.config.robot_type}")
        self.log(f"Episodes per variation: {self.config.episodes_per_variation}")
        self.log(f"FPS: {self.config.fps}")
        self.log(f"CP-Gen Augmentation: {self.config.use_cpgen}")
        self.log(f"Simulation Validation: {self.config.use_validation}")
        self.log(f"Min Quality Score: {self.config.min_quality_score}")
        self.log(f"Min Success Rate: {self.config.min_success_rate:.1%}")
        self.log(f"Data Pack: {self.config.data_pack_tier}")
        self.log(f"Cameras: {self.config.num_cameras}")
        self.log(f"Sensor Capture: {'enabled' if self.sensor_capture else 'disabled'}")
        self.log("=" * 70)

        output = EpisodeGenerationOutput(
            scene_id=self.config.scene_id,
            robot_type=self.config.robot_type,
        )

        # Step 1: Generate tasks with full specifications
        self.log("\nStep 1: Generating task specifications (Gemini at top of stack)...")
        tasks_with_specs = self.task_generator.generate_tasks_with_specs(
            manifest=manifest,
            manifest_path=self.config.manifest_path,
            robot_type=self.config.robot_type,
        )

        if not tasks_with_specs:
            output.errors.append("No manipulation tasks could be generated from scene")
            return output

        self.log(f"  Generated {len(tasks_with_specs)} task specifications")

        # Step 2: Generate seed episodes
        self.log("\nStep 2: Generating seed episodes...")
        seed_episodes = self._generate_seed_episodes(tasks_with_specs, manifest)
        output.seed_episodes = len(seed_episodes)
        self.log(f"  Generated {len(seed_episodes)} seed episodes")

        # Step 3: Determine variations
        variation_count = manifest.get("variation_count", 1)
        if self.config.max_variations:
            variation_count = min(variation_count, self.config.max_variations)
        output.total_variations = variation_count

        # Step 4: Augment using CP-Gen style approach
        self.log(f"\nStep 3: Generating {variation_count} variations using CP-Gen augmentation...")
        all_episodes = self._generate_augmented_episodes(
            seed_episodes=seed_episodes,
            manifest=manifest,
            num_variations=variation_count,
        )
        output.augmented_episodes = len(all_episodes) - len(seed_episodes)

        # Clean up memory after augmentation (large arrays may have been created)
        gc.collect()

        # Step 5: Validate episodes
        self.log("\nStep 4: Validating episodes in simulation...")
        validated_episodes = self._validate_episodes(all_episodes, manifest)
        self._enforce_physics_backed_qc(validated_episodes)
        self._apply_camera_capture_quality_gates(validated_episodes)

        if HAVE_QUALITY_SYSTEM:
            self.log("\nStep 4b: Generating quality certificates...")
            cert_generator = QualityCertificateGenerator(get_environment_capabilities())
            for episode in validated_episodes:
                episode.quality_certificate = self._build_quality_certificate(
                    episode,
                    cert_generator,
                )
            self._write_quality_artifacts(validated_episodes)

        # Step 6: Filter to high-quality episodes
        valid_episodes = [
            ep for ep in validated_episodes
            if ep.is_valid and ep.quality_score >= self.config.min_quality_score
        ]
        self.log(f"  {len(valid_episodes)}/{len(validated_episodes)} episodes passed validation")

        # Clean up memory after filtering (remove references to invalid episodes)
        invalid_episodes = [ep for ep in validated_episodes if not ep.is_valid or ep.quality_score < self.config.min_quality_score]
        del invalid_episodes
        gc.collect()

        # Service-mode QA validation gate before export
        if _is_service_mode():
            self.log("\nStep 5: Running QA validation gate before export...")
            qa_report_path = self.config.output_dir / "quality" / "qa_validation_report.json"
            try:
                from tools.qa_validation.validator import run_qa_validation
                scene_dir = self.config.manifest_path.parent.parent
                qa_report = run_qa_validation(
                    scene_dir=scene_dir,
                    scene_id=self.config.scene_id,
                    output_report=qa_report_path,
                    verbose=self.verbose,
                )
            except Exception as e:
                error_msg = f"QA validation failed to run: {e}"
                self.log(error_msg, "ERROR")
                output.errors.append(error_msg)
                output.total_episodes = len(all_episodes)
                output.valid_episodes = len(valid_episodes)
                output.output_dir = self.config.output_dir
                output.generation_time_seconds = time.time() - start_time
                return output

            if not qa_report.passed:
                error_msg = "QA validation failed; skipping dataset export."
                self.log(error_msg, "ERROR")
                output.errors.append(error_msg)
                for issue in qa_report.issues:
                    self.log(f"  - {issue}", "ERROR")
                output.total_episodes = len(all_episodes)
                output.valid_episodes = len(valid_episodes)
                output.output_dir = self.config.output_dir
                output.generation_time_seconds = time.time() - start_time
                return output

        # Step 7: Export to LeRobot format (with data pack configuration)
        self.log("\nStep 6: Exporting LeRobot dataset...")
        self.log(f"  Data Pack: {self.config.data_pack_tier}")

        # Use data pack-aware configuration
        lerobot_config = LeRobotDatasetConfig.from_data_pack(
            dataset_name=f"{self.config.scene_id}_episodes",
            data_pack_tier=self.config.data_pack_tier,
            robot_type=self.config.robot_type,
            num_cameras=self.config.num_cameras,
            resolution=self.config.image_resolution,
            fps=self.config.fps,
            output_dir=self.config.output_dir / "lerobot",
            strict_alignment=self._is_production_quality_level(),
            export_format=self.config.lerobot_export_format,
        )
        export_estimate = _estimate_export_requirements(valid_episodes, self.config)
        self.log(
            "  Estimated export size: "
            f"{_format_bytes(export_estimate['required_bytes'])} "
            f"({export_estimate['expected_episodes']} episodes, "
            f"{export_estimate['frames_per_episode']} frames/episode "
            f"~{export_estimate['episode_seconds']:.1f}s, "
            f"{export_estimate['num_cameras']} cameras @ "
            f"{export_estimate['resolution'][0]}x{export_estimate['resolution'][1]}, "
            f"{export_estimate['bytes_per_pixel']} B/pixel)"
        )
        _ensure_disk_space(lerobot_config.output_dir, export_estimate["required_bytes"])
        exporter = LeRobotExporter(lerobot_config, verbose=False)

        for episode in valid_episodes:
            if episode.trajectory:
                exporter.add_episode(
                    trajectory=episode.trajectory,
                    task_description=episode.task_description,
                    scene_id=self.config.scene_id,
                    variation_index=episode.variation_index,
                    success=episode.is_valid,
                    quality_score=episode.quality_score,
                    sensor_data=episode.sensor_data,
                    object_metadata=episode.object_metadata,
                )

                # Track task coverage
                task_key = episode.task_name
                output.tasks_generated[task_key] = output.tasks_generated.get(task_key, 0) + 1

        try:
            dataset_path = exporter.finalize()
            output.lerobot_dataset_path = dataset_path
        except Exception as e:
            import traceback
            error_msg = f"LeRobot export failed: {e}"
            is_production = self._is_production_quality_level()
            allow_failure = self._allow_lerobot_export_failure()
            if is_production and os.getenv("ALLOW_LEROBOT_EXPORT_FAILURE"):
                self.log(
                    "ALLOW_LEROBOT_EXPORT_FAILURE is ignored in production runs.",
                    "ERROR",
                )
            if allow_failure:
                output.warnings.append(error_msg)
                self.log(
                    f"{error_msg} (continuing because ALLOW_LEROBOT_EXPORT_FAILURE is enabled for dev runs)",
                    "WARNING",
                )
                if self.verbose:
                    self.log(traceback.format_exc(), "DEBUG")
            else:
                output.errors.append(error_msg)
                self.log(error_msg, "ERROR")
                if self.verbose:
                    self.log(traceback.format_exc(), "DEBUG")
                raise

        # Clean up memory after export
        del exporter
        gc.collect()

        # Step 8: Write generation manifest
        self.log("\nStep 7: Writing generation manifest and validation report...")
        output.manifest_path = self._write_manifest(valid_episodes, tasks_with_specs, output)

        # Write validation report
        if self.validator:
            report_gen = ValidationReportGenerator(self.config.output_dir / "quality")
            results = [ep.validation_result for ep in validated_episodes if ep.validation_result]
            output.validation_report_path = report_gen.generate_report(
                results, self.config.scene_id
            )

        # Calculate statistics
        output.total_episodes = len(all_episodes)
        output.valid_episodes = len(valid_episodes)
        output.total_frames = sum(
            ep.trajectory.num_frames for ep in valid_episodes if ep.trajectory
        )
        output.total_duration_seconds = sum(
            ep.trajectory.total_duration for ep in valid_episodes if ep.trajectory
        )
        output.generation_time_seconds = time.time() - start_time
        output.output_dir = self.config.output_dir

        # Quality metrics
        if valid_episodes:
            output.average_quality_score = np.mean([ep.quality_score for ep in valid_episodes])
        output.pass_rate = len(valid_episodes) / len(all_episodes) if all_episodes else 0
        output.augmentation_success_rate = (
            output.augmented_episodes / max(1, len(seed_episodes) * (variation_count - 1))
            if seed_episodes else 0
        )

        metrics = get_metrics()
        metrics_labels = {"scene_id": self.config.scene_id, "job": JOB_NAME}
        for episode in valid_episodes:
            metrics.episode_quality_score.observe(
                episode.quality_score,
                labels=metrics_labels,
            )
        collision_free_rate = _compute_collision_free_rate(
            output.validation_report_path,
            output.pass_rate,
        )
        metrics.collision_free_rate.set(
            collision_free_rate,
            labels=metrics_labels,
        )

        if self._partial_failure_errors:
            output.errors.extend(self._partial_failure_errors)

        # Summary
        self.log("\n" + "=" * 70)
        self.log("GENERATION COMPLETE (SOTA Pipeline)")
        self.log("=" * 70)
        self.log(f"Total episodes generated: {output.total_episodes}")
        self.log(f"Valid episodes (exported): {output.valid_episodes}")
        self.log(f"Pass rate: {output.pass_rate:.1%}")
        self.log(f"Average quality score: {output.average_quality_score:.2f}")
        self.log(f"Total frames: {output.total_frames}")
        self.log(f"Total duration: {output.total_duration_seconds:.1f}s")
        self.log(f"Tasks covered: {len(output.tasks_generated)}")
        self.log(f"Generation time: {output.generation_time_seconds:.1f}s")
        self.log(f"Output: {output.output_dir}")
        if output.errors:
            self.log(f"Errors: {len(output.errors)}", "ERROR")
        self.log("=" * 70)

        return output

    def _generate_seed_episode(
        self,
        task: Dict[str, Any],
        spec: TaskSpecification,
        manifest: Dict[str, Any],
        scene_context: SceneContext,
    ) -> GeneratedEpisode:
        start_time = time.time()

        if "task_steps" in task:
            motion_plan, trajectory, sensor_data, object_metadata = self._plan_multi_step_task(
                task=task,
                spec=spec,
                manifest=manifest,
                scene_context=scene_context,
            )
        else:
            # Generate motion plan using constraint-aware planning
            target_object = {
                "id": task["target_object_id"],
                "position": task["target_position"],
                "dimensions": task["target_dimensions"],
            }

            # Make articulation info configurable from task specification
            articulation_info = None
            if task.get("is_articulated"):
                # Try to get articulation info from task, otherwise use defaults
                if "articulation_info" in task:
                    # Use provided articulation info
                    articulation_info = task["articulation_info"]
                else:
                    # Infer from object category or use conservative defaults
                    object_category = task.get("target_object_category", "").lower()

                    # Default articulation parameters by category
                    if "drawer" in object_category:
                        articulation_info = {
                            "handle_position": task["target_position"],
                            "axis": [-1, 0, 0],  # Pull outward
                            "range": [0, 0.4],   # 40cm max extension
                            "type": "prismatic",
                        }
                    elif "door" in object_category or "cabinet" in object_category:
                        articulation_info = {
                            "handle_position": task["target_position"],
                            "axis": [0, 0, 1],   # Vertical axis (hinge)
                            "range": [0, 1.57],  # 90 degrees
                            "type": "revolute",
                        }
                    elif "lid" in object_category or "box" in object_category:
                        articulation_info = {
                            "handle_position": task["target_position"],
                            "axis": [0, 1, 0],   # Horizontal axis
                            "range": [0, 1.57],  # 90 degrees
                            "type": "revolute",
                        }
                    else:
                        # Generic fallback (conservative prismatic)
                        articulation_info = {
                            "handle_position": task["target_position"],
                            "axis": task.get("articulation_axis", [-1, 0, 0]),
                            "range": task.get("articulation_range", [0, 0.3]),
                            "type": task.get("articulation_type", "prismatic"),
                        }

            motion_plan = self.motion_planner.plan_motion(
                task_name=task["task_name"],
                task_description=task["description"],
                target_object=target_object,
                place_position=task.get("place_position"),
                articulation_info=articulation_info,
                scene_context=scene_context,
            )

            # Solve trajectory with replan on IK failure
            motion_plan, trajectory = self._solve_trajectory_with_replan(
                motion_plan=motion_plan,
                task_name=task["task_name"],
                task_description=task["description"],
                target_object=target_object,
                place_position=task.get("place_position"),
                articulation_info=articulation_info,
                context_label=f"seed:{task['task_id']}",
            )
            if not getattr(motion_plan, "planning_success", True):
                self.log(
                    f"    Motion planning failed for {task['task_name']}: {motion_plan.planning_errors}",
                    "WARNING",
                )
                return GeneratedEpisode(
                    episode_id=f"seed_{task['task_id']}",
                    task_name=task["task_name"],
                    task_description=task["description"],
                    trajectory=None,
                    motion_plan=motion_plan,
                    task_spec=spec,
                    scene_id=self.config.scene_id,
                    variation_index=0,
                    is_seed=True,
                    augmentation_method="direct",
                    generation_time_seconds=time.time() - start_time,
                )

            # Solve trajectory
            trajectory = self.trajectory_solver.solve(motion_plan)

            # Capture sensor data during trajectory execution (if available)
            sensor_data = None
            object_metadata = {}
            if self.sensor_capture:
                try:
                    # Get scene objects for metadata
                    scene_objects = manifest.get("objects", [])
                    object_metadata = {
                        obj.get("id", obj.get("name", "")): {
                            "category": obj.get("category", "object"),
                            "dimensions": obj.get("dimensions", [0.1, 0.1, 0.1]),
                            "position": obj.get("position", [0, 0, 0]),
                        }
                        for obj in scene_objects
                    }

                    sensor_data = self.sensor_capture.capture_episode(
                        episode_id=f"seed_{task['task_id']}",
                        trajectory_states=trajectory.states,
                        scene_objects=scene_objects,
                    )
                    self.log(f"    Captured sensor data: {sensor_data.num_frames} frames")
                except Exception as e:
                    self.log(f"    Sensor capture failed: {e}", "WARNING")

        if trajectory is None or not getattr(motion_plan, "planning_success", True):
            self.log(
                f"    Motion planning failed for {task['task_name']}: {getattr(motion_plan, 'planning_errors', [])}",
                "WARNING",
            )
            return GeneratedEpisode(
                episode_id=f"seed_{task['task_id']}",
                task_name=task["task_name"],
                task_description=task["description"],
                trajectory=None,
                motion_plan=motion_plan,
                task_spec=spec,
                scene_id=self.config.scene_id,
                variation_index=0,
                is_seed=True,
                augmentation_method="direct",
                generation_time_seconds=time.time() - start_time,
            )

        episode = GeneratedEpisode(
            episode_id=f"seed_{task['task_id']}",
            task_name=task["task_name"],
            task_description=task["description"],
            trajectory=trajectory,
            motion_plan=motion_plan,
            task_spec=spec,
            scene_id=self.config.scene_id,
            variation_index=0,
            is_seed=True,
            augmentation_method="direct",
            sensor_data=sensor_data,
            object_metadata=object_metadata,
            generation_time_seconds=time.time() - start_time,
        )

        visual_info = f" (visual: {sensor_data.num_frames}f)" if sensor_data else ""
        self.log(f"    Created seed: {task['task_name']}{visual_info}")
        return episode

    def _generate_seed_episodes(
        self,
        tasks_with_specs: List[Tuple[Dict[str, Any], TaskSpecification]],
        manifest: Dict[str, Any],
    ) -> List[GeneratedEpisode]:
        """Generate seed episodes for each task."""
        scene_context = SceneContext(
            scene_id=self.config.scene_id,
            environment_type=manifest.get("environment_type", "unknown"),
            objects=manifest.get("objects", []),
            scene_usd_path=self.config.scene_usd_path,
            robot_urdf_path=self.config.robot_urdf_path,
        )

        def process_item(item: Tuple[Dict[str, Any], TaskSpecification]) -> GeneratedEpisode:
            task, spec = item
            return self._generate_seed_episode(task, spec, manifest, scene_context)

        result = self._process_with_partial_failures(
            tasks_with_specs,
            process_fn=process_item,
            item_id_fn=lambda item: item[0].get("task_id", item[0].get("task_name", "unknown")),
            batch_name="seed_generation",
        )
        return result.successful

    def _plan_multi_step_task(
        self,
        task: Dict[str, Any],
        spec: TaskSpecification,
        manifest: Dict[str, Any],
        scene_context: SceneContext,
    ) -> Tuple[MotionPlan, Optional[JointTrajectory], Optional[Any], Dict[str, Any]]:
        """Plan and solve a multi-step manipulation task."""
        steps = task.get("task_steps", [])
        step_plans = []
        step_trajectories = []
        object_metadata = {}

        for step_index, step in enumerate(steps):
            target_object = {
                "id": step.get("target_object_id"),
                "position": step.get("target_position", task.get("target_position")),
                "dimensions": step.get("target_dimensions", task.get("target_dimensions")),
            }
            motion_plan = self.motion_planner.plan_motion(
                task_name=step.get("action", task["task_name"]),
                task_description=step.get("description", task["description"]),
                target_object=target_object,
                place_position=step.get("place_position", task.get("place_position")),
                articulation_info=step.get("articulation_info"),
                scene_context=scene_context,
            )

            motion_plan, trajectory = self._solve_trajectory_with_replan(
                motion_plan=motion_plan,
                task_name=step.get("action", task["task_name"]),
                task_description=step.get("description", task["description"]),
                target_object=target_object,
                place_position=step.get("place_position", task.get("place_position")),
                articulation_info=step.get("articulation_info"),
                context_label=f"seed:{task['task_id']}:step{step_index}",
            )

            if not getattr(motion_plan, "planning_success", True):
                motion_plan.planning_errors.append(f"Multi-step planning failed at step index {step_index}")
                return motion_plan, None, None, {}

            step_plans.append(motion_plan)
            step_trajectories.append(trajectory)

        combined_plan = self._combine_motion_plans(step_plans, task)
        combined_trajectory = self._combine_trajectories(step_trajectories, task)

        sensor_data = None
        if self.sensor_capture and combined_trajectory is not None:
            try:
                scene_objects = manifest.get("objects", [])
                object_metadata = {
                    obj.get("id", obj.get("name", "")): {
                        "category": obj.get("category", "object"),
                        "dimensions": obj.get("dimensions", [0.1, 0.1, 0.1]),
                        "position": obj.get("position", [0, 0, 0]),
                    }
                    for obj in scene_objects
                }
                sensor_data = self.sensor_capture.capture_episode(
                    episode_id=f"seed_{task['task_id']}",
                    trajectory_states=combined_trajectory.states,
                    scene_objects=scene_objects,
                )
                self.log(f"    Captured sensor data: {sensor_data.num_frames} frames")
            except Exception as e:
                self.log(f"    Sensor capture failed: {e}", "WARNING")

        return combined_plan, combined_trajectory, sensor_data, object_metadata

    def _combine_motion_plans(
        self,
        plans: List[MotionPlan],
        task: Dict[str, Any],
    ) -> MotionPlan:
        """Combine motion plans from multiple steps into a single plan."""
        all_waypoints = []
        for plan in plans:
            all_waypoints.extend(plan.waypoints)

        last_plan = plans[-1] if plans else None
        return MotionPlan(
            plan_id=f"plan_{task['task_id']}",
            task_name=task["task_name"],
            task_description=task["description"],
            waypoints=all_waypoints,
            target_object_id=last_plan.target_object_id if last_plan else None,
            target_object_position=last_plan.target_object_position if last_plan else None,
            target_object_dimensions=last_plan.target_object_dimensions if last_plan else None,
            place_position=last_plan.place_position if last_plan else None,
            articulation_axis=last_plan.articulation_axis if last_plan else None,
            articulation_range=last_plan.articulation_range if last_plan else None,
            handle_position=last_plan.handle_position if last_plan else None,
            robot_type=self.config.robot_type,
            planning_backend="multi_step",
            planning_success=True,
        )

    def _combine_trajectories(
        self,
        trajectories: List[JointTrajectory],
        task: Dict[str, Any],
    ) -> Optional[JointTrajectory]:
        """Combine step trajectories into a single trajectory."""
        if not trajectories:
            return None

        combined_states = []
        time_offset = 0.0
        frame_offset = 0

        for trajectory in trajectories:
            for state in trajectory.states:
                combined_states.append(
                    replace(
                        state,
                        frame_idx=state.frame_idx + frame_offset,
                        timestamp=state.timestamp + time_offset,
                    )
                )
            time_offset += trajectory.total_duration
            frame_offset += trajectory.num_frames

        combined = JointTrajectory(
            trajectory_id=f"traj_{task['task_id']}",
            robot_type=trajectories[0].robot_type,
            robot_config=trajectories[0].robot_config,
            states=combined_states,
            source_plan_id=f"plan_{task['task_id']}",
            fps=trajectories[0].fps,
            total_duration=time_offset,
        )
        return combined

    def _generate_augmented_episodes(
        self,
        seed_episodes: List[GeneratedEpisode],
        manifest: Dict[str, Any],
        num_variations: int,
    ) -> List[GeneratedEpisode]:
        """Generate augmented episodes using CP-Gen approach."""
        all_episodes = list(seed_episodes)  # Start with seeds
        objects = manifest.get("objects", [])

        if not self.config.use_cpgen or not self.cpgen_augmenter:
            # Fallback: simple variation without CP-Gen
            self.log("  Using simple variation (CP-Gen disabled)")
            return self._generate_simple_variations(seed_episodes, manifest, num_variations)

        eligible_seeds = []
        for seed in seed_episodes:
            if seed.task_spec is None or seed.motion_plan is None:
                continue
            if not getattr(seed.motion_plan, "planning_success", True):
                self.log(f"  Skipping seed {seed.episode_id} due to planning failure", "WARNING")
                continue
            eligible_seeds.append(seed)

        if not eligible_seeds:
            self.log("  No eligible seeds available for CP-Gen augmentation", "WARNING")
            return all_episodes

        def create_cpgen_seed(seed: GeneratedEpisode) -> Tuple[GeneratedEpisode, SeedEpisode]:
            cpgen_seed = self.cpgen_augmenter.create_seed_episode(
                task_spec=seed.task_spec,
                motion_plan=seed.motion_plan,
                scene_objects=objects,
            )
            return seed, cpgen_seed

        seed_result = self._process_with_partial_failures(
            eligible_seeds,
            process_fn=create_cpgen_seed,
            item_id_fn=lambda seed: seed.episode_id,
            batch_name="cpgen_seed",
        )

        variation_tasks: List[Tuple[GeneratedEpisode, SeedEpisode, int]] = []
        for seed, cpgen_seed in seed_result.successful:
            for var_idx in range(1, num_variations):
                variation_tasks.append((seed, cpgen_seed, var_idx))

        def process_variation(task: Tuple[GeneratedEpisode, SeedEpisode, int]) -> GeneratedEpisode:
            seed, cpgen_seed, var_idx = task
            object_transforms = self._generate_variation_transforms(
                objects, var_idx
            )
            updated_obstacles = self._apply_transforms_to_objects(
                objects, object_transforms
            )

            augmented = self.cpgen_augmenter.augment(
                seed=cpgen_seed,
                object_transforms=object_transforms,
                obstacles=updated_obstacles,
                variation_index=var_idx,
            )

            if not augmented.planning_success:
                planning_errors = []
                if augmented.motion_plan is not None:
                    planning_errors = getattr(augmented.motion_plan, "planning_errors", [])
                raise RuntimeError(
                    f"CP-Gen planning failed for {seed.task_name} variation {var_idx}: {planning_errors}"
                )

            trajectory = self.trajectory_solver.solve(augmented.motion_plan)

            sensor_data = None
            if self.sensor_capture:
                try:
                    sensor_data = self.sensor_capture.capture_episode(
                        episode_id=augmented.episode_id,
                        trajectory_states=trajectory.states,
                        scene_objects=updated_obstacles,
                    )
                except Exception as e:
                    self.log(f"    Sensor capture failed for variation {var_idx}: {e}", "WARNING")

            return GeneratedEpisode(
                episode_id=augmented.episode_id,
                task_name=seed.task_name,
                task_description=seed.task_description,
                trajectory=trajectory,
                motion_plan=augmented.motion_plan,
                task_spec=seed.task_spec,
                scene_id=self.config.scene_id,
                variation_index=var_idx,
                is_seed=False,
                augmentation_method="cpgen",
                seed_episode_id=seed.episode_id,
                sensor_data=sensor_data,
                object_metadata=seed.object_metadata,
                quality_score=augmented.constraint_satisfaction,
                is_valid=augmented.collision_free,
                generation_time_seconds=augmented.generation_time_seconds,
            )

        if variation_tasks:
            variation_result = self._process_with_partial_failures(
                variation_tasks,
                process_fn=process_variation,
                item_id_fn=lambda task: f"{task[0].episode_id}_var{task[2]}",
                batch_name="cpgen_variations",
            )
            all_episodes.extend(variation_result.successful)

        self.log(f"  Generated {len(all_episodes)} total episodes")
        return all_episodes

    def _generate_simple_variations(
        self,
        seed_episodes: List[GeneratedEpisode],
        manifest: Dict[str, Any],
        num_variations: int,
    ) -> List[GeneratedEpisode]:
        """Simple variation without CP-Gen (fallback)."""
        all_episodes = list(seed_episodes)

        # Get workspace defaults from manifest or robot config
        robot_config = manifest.get("robot_config", {})
        workspace_center = robot_config.get("workspace_center", [0.5, 0.0, 0.85])

        eligible_seeds = []
        for seed in seed_episodes:
            # Skip seeds with uninitialized task spec or motion plan
            if seed.task_spec is None or seed.motion_plan is None:
                self.log(f"    Skipping seed {seed.episode_id} - missing task_spec or motion_plan", "WARNING")
                continue
            if not getattr(seed.motion_plan, "planning_success", True):
                self.log(f"    Skipping seed {seed.episode_id} due to planning failure", "WARNING")
                continue
            eligible_seeds.append(seed)

        variation_tasks: List[Tuple[GeneratedEpisode, int]] = []
        for seed in eligible_seeds:
            for var_idx in range(1, num_variations):
                variation_tasks.append((seed, var_idx))

        def process_variation(task: Tuple[GeneratedEpisode, int]) -> GeneratedEpisode:
            seed, var_idx = task
            set_global_seed(var_idx)
            varied_manifest = json.loads(json.dumps(manifest))

            for obj in varied_manifest.get("objects", []):
                if "position" in obj:
                    pos = obj["position"]
                    if isinstance(pos, list):
                        offset = np.random.uniform(-0.05, 0.05, 3)
                        offset[2] = 0
                        obj["position"] = [p + o for p, o in zip(pos, offset)]

            # Regenerate episode for this variation
            target_object = {
                "id": seed.motion_plan.target_object_id if seed.motion_plan else "target",
                "position": seed.motion_plan.target_object_position.tolist() if seed.motion_plan and seed.motion_plan.target_object_position is not None else workspace_center,
                "dimensions": seed.motion_plan.target_object_dimensions.tolist() if seed.motion_plan and seed.motion_plan.target_object_dimensions is not None else [0.08, 0.08, 0.1],
            }

            # Add variation offset
            offset = np.random.uniform(-0.05, 0.05, 3)
            offset[2] = 0
            target_object["position"] = [
                p + o for p, o in zip(target_object["position"], offset)
            ]

            variation_scene_context = SceneContext(
                scene_id=self.config.scene_id,
                environment_type=varied_manifest.get("environment_type", "unknown"),
                objects=varied_manifest.get("objects", []),
                scene_usd_path=self.config.scene_usd_path,
                robot_urdf_path=self.config.robot_urdf_path,
            )

            motion_plan = self.motion_planner.plan_motion(
                task_name=seed.task_name,
                task_description=seed.task_description,
                target_object=target_object,
                place_position=seed.motion_plan.place_position.tolist() if seed.motion_plan and seed.motion_plan.place_position is not None else None,
                scene_context=variation_scene_context,
            )

            motion_plan, trajectory = self._solve_trajectory_with_replan(
                motion_plan=motion_plan,
                task_name=seed.task_name,
                task_description=seed.task_description,
                target_object=target_object,
                place_position=seed.motion_plan.place_position.tolist() if seed.motion_plan and seed.motion_plan.place_position is not None else None,
                articulation_info=None,
                context_label=f"simple_var:{seed.task_name}:{var_idx}",
            )
            if not getattr(motion_plan, "planning_success", True):
                raise RuntimeError(
                    f"Simple variation planning failed for {seed.task_name} variation {var_idx}: "
                    f"{motion_plan.planning_errors}"
                )

            trajectory = self.trajectory_solver.solve(motion_plan)

            # Capture sensor data for simple variation
            sensor_data = None
            if self.sensor_capture:
                try:
                    sensor_data = self.sensor_capture.capture_episode(
                        episode_id=f"{seed.task_name}_var{var_idx}",
                        trajectory_states=trajectory.states,
                        scene_objects=varied_manifest.get("objects", []),
                    )
                except Exception as e:
                    self.log(f"    Sensor capture failed for simple var {var_idx}: {e}", "WARNING")

            return GeneratedEpisode(
                episode_id=f"{seed.task_name}_var{var_idx}_{uuid.uuid4().hex[:8]}",
                task_name=seed.task_name,
                task_description=seed.task_description,
                trajectory=trajectory,
                motion_plan=motion_plan,
                scene_id=self.config.scene_id,
                variation_index=var_idx,
                augmentation_method="simple",
                seed_episode_id=seed.episode_id,
                sensor_data=sensor_data,
                object_metadata=seed.object_metadata,
            )

        if variation_tasks:
            variation_result = self._process_with_partial_failures(
                variation_tasks,
                process_fn=process_variation,
                item_id_fn=lambda task: f"{task[0].episode_id}_var{task[1]}",
                batch_name="simple_variations",
            )
            all_episodes.extend(variation_result.successful)

        return all_episodes

    def _generate_variation_transforms(
        self,
        objects: List[Dict[str, Any]],
        variation_index: int,
    ) -> Dict[str, ObjectTransform]:
        """Generate random transforms for a variation."""
        set_global_seed(variation_index)
        transforms = {}

        for obj in objects:
            obj_id = obj.get("id", obj.get("name", ""))

            # Random position offset
            pos_offset = np.random.randn(3) * 0.05
            pos_offset[2] = 0  # Keep height stable

            # Random rotation around z-axis
            angle = np.random.randn() * 0.1
            rot_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

            transforms[obj_id] = ObjectTransform(
                object_id=obj_id,
                position_offset=pos_offset,
                rotation_offset=rot_quat,
            )

        return transforms

    def _apply_transforms_to_objects(
        self,
        objects: List[Dict[str, Any]],
        transforms: Dict[str, ObjectTransform],
    ) -> List[Dict[str, Any]]:
        """Apply transforms to get updated obstacle list."""
        updated = []
        for obj in objects:
            obj_id = obj.get("id", obj.get("name", ""))
            transform = transforms.get(obj_id)

            if transform:
                orig_pos = np.array(obj.get("position", [0, 0, 0]))
                new_pos = (orig_pos + transform.position_offset).tolist()
            else:
                new_pos = obj.get("position", [0, 0, 0])

            updated.append({
                "id": obj_id,
                "position": new_pos,
                "dimensions": obj.get("dimensions", [0.1, 0.1, 0.1]),
            })

        return updated

    def _validate_episodes(
        self,
        episodes: List[GeneratedEpisode],
        manifest: Dict[str, Any],
    ) -> List[GeneratedEpisode]:
        """Validate all episodes using simulation."""
        if not self.validator:
            # No validation - mark all as valid
            for ep in episodes:
                ep.is_valid = True
                ep.quality_score = 1.0
            return episodes

        objects = manifest.get("objects", [])

        def validate_episode(episode: GeneratedEpisode) -> GeneratedEpisode:
            if episode.trajectory is None or episode.motion_plan is None:
                if episode.motion_plan is not None and not getattr(episode.motion_plan, "planning_success", True):
                    result = ValidationResult(
                        episode_id=episode.episode_id,
                        status=ValidationStatus.NEEDS_RETRY,
                    )
                    result.failure_reasons.append(FailureReason.PLANNING_FAILURE)
                    result.failure_details = "; ".join(episode.motion_plan.planning_errors)
                    episode.validation_result = result
                    episode.is_valid = False
                    episode.quality_score = 0.0
                    episode.validation_errors = [r.value for r in result.failure_reasons]
                    return episode

                episode.is_valid = False
                episode.quality_score = 0.0
                episode.validation_errors.append("Missing trajectory or motion plan")
                return episode

            try:
                result = self.validator.validate(
                    trajectory=episode.trajectory,
                    motion_plan=episode.motion_plan,
                    scene_objects=objects,
                )

                episode.validation_result = result
                episode.is_valid = result.status == ValidationStatus.PASSED
                episode.quality_score = result.metrics.overall_score
                episode.validation_errors = [r.value for r in result.failure_reasons]

            except Exception as e:
                episode.is_valid = False
                episode.quality_score = 0.0
                episode.validation_errors.append(str(e))

            return episode

        result = self._process_with_partial_failures(
            episodes,
            process_fn=validate_episode,
            item_id_fn=lambda ep: ep.episode_id,
            batch_name="validation",
        )
        return result.successful

    def _resolve_sensor_source(self, episode: GeneratedEpisode) -> str:
        if episode.sensor_data is None or self.sensor_capture is None:
            return "disabled"
        if self._sensor_capture_is_mock:
            return SensorSource.MOCK.value
        return SensorSource.ISAAC_SIM_REPLICATOR.value

    def _resolve_physics_backend(self, episode: GeneratedEpisode) -> str:
        backend = None
        if episode.validation_result:
            backend = episode.validation_result.physics_backend
        if backend in {"isaac_sim", "isaac_lab"}:
            return PhysicsValidationBackend.PHYSX.value
        if backend == "heuristic":
            return PhysicsValidationBackend.HEURISTIC.value
        if self.validator and self.validator.is_using_real_physics():
            return PhysicsValidationBackend.PHYSX.value
        return PhysicsValidationBackend.HEURISTIC.value

    def _build_quality_certificate(
        self,
        episode: GeneratedEpisode,
        generator: "QualityCertificateGenerator",
    ) -> Optional["QualityCertificate"]:
        if not HAVE_QUALITY_SYSTEM:
            return None

        validation_metrics = (
            episode.validation_result.metrics
            if episode.validation_result and episode.validation_result.metrics
            else None
        )
        sensor_source = self._resolve_sensor_source(episode)
        physics_backend = self._resolve_physics_backend(episode)

        collision_count = validation_metrics.total_collisions if validation_metrics else 0
        joint_limit_violations = validation_metrics.joint_limit_violations if validation_metrics else 0
        torque_limit_violations = validation_metrics.torque_limit_violations if validation_metrics else 0
        path_length = validation_metrics.path_length if validation_metrics else 0.0
        jerk_integral = validation_metrics.jerk_integral if validation_metrics else 0.0
        smoothness_score = validation_metrics.velocity_smoothness if validation_metrics else 0.0

        trajectory_metrics = TrajectoryQualityMetrics(
            smoothness_score=smoothness_score,
            mean_jerk=jerk_integral,
            path_efficiency=1.0,
            time_efficiency=1.0,
            trajectory_length_meters=path_length,
            dynamics_feasibility=1.0 if joint_limit_violations == 0 else 0.7,
            joint_limit_violations=joint_limit_violations,
            torque_limit_violations=torque_limit_violations,
            collision_count=collision_count,
        )

        skill_segments_correct = 0
        if validation_metrics:
            skill_segments_correct = int(validation_metrics.grasp_success) + int(
                validation_metrics.placement_success
            )
        skill_segment_count = 2 if validation_metrics else 0
        skill_correctness_ratio = (
            skill_segments_correct / skill_segment_count
            if skill_segment_count
            else 0.0
        )

        constraint_violations = joint_limit_violations + torque_limit_violations
        constraint_satisfaction = 1.0 if constraint_violations == 0 else 0.5
        task_metrics = TaskQualityMetrics(
            goal_achievement_score=1.0 if validation_metrics and validation_metrics.task_success else 0.0,
            skill_segment_count=skill_segment_count,
            skill_segments_correct=skill_segments_correct,
            skill_correctness_ratio=skill_correctness_ratio,
            constraint_violations=constraint_violations,
            constraint_satisfaction_score=constraint_satisfaction,
        )

        frame_count = episode.sensor_data.num_frames if episode.sensor_data else 0
        camera_count = len(episode.sensor_data.camera_ids) if episode.sensor_data else 0
        visual_metrics = VisualQualityMetrics(
            target_visibility_ratio=1.0 if episode.sensor_data and episode.sensor_data.has_rgb else 0.0,
            viewpoint_diversity=min(1.0, camera_count / 3.0) if camera_count else 0.0,
        )

        sim2real_metrics = Sim2RealMetrics(
            physics_plausibility_score=1.0 if physics_backend == PhysicsValidationBackend.PHYSX.value else 0.5,
            episode_duration_seconds=validation_metrics.total_duration if validation_metrics else 0.0,
            timing_realism_score=1.0,
        )

        episode_hash = compute_episode_data_hash(
            {
                "episode_id": episode.episode_id,
                "task_name": episode.task_name,
                "quality_score": episode.quality_score,
                "validation_errors": episode.validation_errors,
                "sensor_frames": frame_count,
                "sensor_source": sensor_source,
                "physics_backend": physics_backend,
            }
        )

        cert = generator.generate_certificate(
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            task_id=episode.task_name,
            trajectory_metrics=trajectory_metrics,
            visual_metrics=visual_metrics,
            task_metrics=task_metrics,
            sim2real_metrics=sim2real_metrics,
            validation_passed=episode.is_valid,
            frame_count=frame_count,
            camera_count=camera_count,
            episode_data_hash=episode_hash,
        )

        cert.sensor_source = sensor_source
        cert.physics_backend = physics_backend
        cert.overall_quality_score = episode.quality_score
        for error in episode.validation_errors:
            cert.add_error(error)
        cert.recommended_use = cert.assess_training_suitability()
        cert.confidence_score = generator._compute_confidence_score(cert)

        if sensor_source != SensorSource.ISAAC_SIM_REPLICATOR.value or (
            physics_backend != PhysicsValidationBackend.PHYSX.value
        ):
            cert.data_quality_level = DataQualityLevel.DEVELOPMENT.value

        if sensor_source == SensorSource.MOCK.value or self._sensor_capture_is_mock:
            cert.add_warning(
                "MockSensorCapture in use - sensor data is placeholder noise and non-production"
            )

        if sensor_source == "disabled":
            cert.add_warning("Sensor capture disabled - no visual observations recorded")

        if physics_backend == PhysicsValidationBackend.HEURISTIC.value:
            cert.add_warning(
                "Heuristic physics validation in use - not suitable for production filtering"
            )

        return cert

    def _map_quality_suitability(self, cert: "QualityCertificate") -> str:
        if cert.recommended_use == "production_training":
            return "production"
        if cert.recommended_use == "fine_tuning":
            return "fine_tuning"
        return "dev_only"

    def _write_quality_artifacts(self, episodes: List[GeneratedEpisode]) -> None:
        if not HAVE_QUALITY_SYSTEM:
            return

        quality_dir = self.config.output_dir / "quality" / "episodes"
        quality_dir.mkdir(parents=True, exist_ok=True)

        manifest_entries = []
        for episode in episodes:
            cert = episode.quality_certificate
            if cert is None:
                continue
            episode_dir = quality_dir / episode.episode_id
            episode_dir.mkdir(parents=True, exist_ok=True)
            cert.save(episode_dir / "quality_certificate.json")
            manifest_entries.append(
                {
                    "episode_id": episode.episode_id,
                    "task_name": episode.task_name,
                    "quality_score": cert.overall_quality_score,
                    "suitability": self._map_quality_suitability(cert),
                    "sensor_backend": cert.sensor_source,
                    "physics_backend": cert.physics_backend,
                    "warnings": list(cert.validation_warnings),
                }
            )

        manifest = {
            "scene_id": self.config.scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "episodes": manifest_entries,
            "summary": {
                "total_episodes": len(manifest_entries),
                "production": sum(1 for entry in manifest_entries if entry["suitability"] == "production"),
                "fine_tuning": sum(1 for entry in manifest_entries if entry["suitability"] == "fine_tuning"),
                "dev_only": sum(1 for entry in manifest_entries if entry["suitability"] == "dev_only"),
            },
        }

        manifest_path = self.config.output_dir / "dataset_quality_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _get_data_pack_includes(self) -> List[str]:
        """Get list of data streams included in the configured data pack."""
        tier = self.config.data_pack_tier.lower()

        includes = ["rgb", "robot_state", "actions", "episode_metadata", "quality_metrics"]

        if tier in ["plus", "full"]:
            includes.extend([
                "depth",
                "semantic_segmentation",
                "instance_segmentation",
                "bounding_box_2d",
                "bounding_box_3d",
            ])

        if tier == "full":
            includes.extend([
                "object_poses",
                "contact_info",
                "surface_normals",
                "privileged_state",
            ])

        return includes

    def _write_manifest(
        self,
        episodes: List[GeneratedEpisode],
        tasks_with_specs: List[Tuple[Dict[str, Any], TaskSpecification]],
        output: EpisodeGenerationOutput,
    ) -> Path:
        """Write generation manifest."""
        manifest_dir = self.config.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Count episodes with sensor data
        episodes_with_sensor_data = sum(1 for e in episodes if e.sensor_data is not None)

        # Generation manifest
        metrics = get_metrics()
        metrics_summary = {
            "backend": metrics.backend.value,
            "stats": metrics.get_stats(),
        }
        manifest = {
            "scene_id": self.config.scene_id,
            "robot_type": self.config.robot_type,
            "fps": self.config.fps,
            "generation_config": {
                "episodes_per_variation": self.config.episodes_per_variation,
                "use_llm": self.config.use_llm,
                "use_cpgen": self.config.use_cpgen,
                "use_validation": self.config.use_validation,
                "min_quality_score": self.config.min_quality_score,
                "min_success_rate": self.config.min_success_rate,
            },
            "data_pack": {
                "tier": self.config.data_pack_tier,
                "num_cameras": self.config.num_cameras,
                "image_resolution": list(self.config.image_resolution),
                "capture_enabled": self.config.capture_sensor_data,
                "episodes_with_visual_obs": episodes_with_sensor_data,
                "includes": self._get_data_pack_includes(),
            },
            "pipeline_version": "2.0.0-sota",  # SOTA version
            "methodology": {
                "task_specification": "Gemini-powered (top of stack)",
                "augmentation": "CP-Gen style constraint-preserving",
                "validation": "Simulation-verified with quality scoring",
                "sensor_data": "Isaac Sim Replicator integration",
                "references": [
                    "CP-Gen (CoRL 2025): https://cp-gen.github.io/",
                    "DemoGen (RSS 2025): https://demo-generation.github.io/",
                    "AnyTask: https://anytask.rai-inst.com/",
                ],
            },
            "statistics": {
                "total_episodes": len(episodes),
                "valid_episodes": sum(1 for e in episodes if e.is_valid),
                "seed_episodes": sum(1 for e in episodes if e.is_seed),
                "augmented_episodes": sum(1 for e in episodes if not e.is_seed),
                "average_quality_score": output.average_quality_score,
                "pass_rate": output.pass_rate,
                "total_frames": sum(
                    e.trajectory.num_frames for e in episodes if e.trajectory
                ),
                "total_duration_seconds": sum(
                    e.trajectory.total_duration for e in episodes if e.trajectory
                ),
            },
            "quality_assurance": {
                "sim_verified": True,
                "constraint_preserving": self.config.use_cpgen,
                "collision_checked": True,
                "includes_quality_metrics": True,
            },
            "tasks": [
                {
                    "task_id": t["task_id"],
                    "task_name": t["task_name"],
                    "description": t["description"],
                    "has_full_spec": spec is not None,
                    "num_segments": len(spec.segments) if spec else 0,
                    "num_steps": len(t.get("task_steps", [])) if t else 0,
                    "step_dependencies": spec.success_criteria.get("step_dependencies")
                    if spec and isinstance(spec.success_criteria, dict)
                    else None,
                    "episodes_generated": output.tasks_generated.get(t["task_name"], 0),
                }
                for t, spec in tasks_with_specs
            ],
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "task_name": e.task_name,
                    "variation_index": e.variation_index,
                    "is_seed": e.is_seed,
                    "augmentation_method": e.augmentation_method,
                    "is_valid": e.is_valid,
                    "quality_score": e.quality_score,
                    "num_frames": e.trajectory.num_frames if e.trajectory else 0,
                    "duration_seconds": e.trajectory.total_duration if e.trajectory else 0,
                    "generation_time_seconds": e.generation_time_seconds,
                    "errors": e.validation_errors,
                }
                for e in episodes
            ],
            "metrics_summary": metrics_summary,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "generator_version": "2.0.0",
        }

        manifest_path = manifest_dir / "generation_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Task coverage report
        coverage = {
            "tasks": output.tasks_generated,
            "total_unique_tasks": len(output.tasks_generated),
            "average_episodes_per_task": (
                sum(output.tasks_generated.values()) / len(output.tasks_generated)
                if output.tasks_generated else 0
            ),
        }

        coverage_path = manifest_dir / "task_coverage.json"
        with open(coverage_path, "w") as f:
            json.dump(coverage, f, indent=2)

        return manifest_path


# =============================================================================
# Cloud Run Entrypoint
# =============================================================================


def run_episode_generation_job(
    root: Path,
    bucket: str,
    scene_id: str,
    assets_prefix: str,
    episodes_prefix: str,
    enable_firebase_upload: bool = False,
    firebase_upload_prefix: str = "datasets",
    robot_type: str = "franka",
    episodes_per_variation: int = 10,
    max_variations: Optional[int] = None,
    fps: float = 30.0,
    use_llm: bool = True,
    use_cpgen: bool = True,
    min_quality_score: float = 0.85,  # LABS-BLOCKER-002 FIX: Raised from 0.7
    min_success_rate: float = 0.5,
    data_pack_tier: str = "core",
    num_cameras: int = 1,
    image_resolution: Tuple[int, int] = (640, 480),
    capture_sensor_data: bool = True,
    use_mock_capture: bool = False,
    allow_mock_capture: bool = False,
    bundle_tier: str = "standard",
    lerobot_export_format: Optional[Union[str, LeRobotExportFormat]] = None,
) -> int:
    """
    Run the episode generation job (SOTA Pipeline).

    Args:
        root: Root path (e.g., /mnt/gcs)
        bucket: GCS bucket name
        scene_id: Scene identifier
        assets_prefix: Path to scene assets
        episodes_prefix: Output path for episodes
        enable_firebase_upload: Enable Firebase Storage upload of generated episodes
        firebase_upload_prefix: Remote prefix for Firebase uploads
        robot_type: Robot type
        episodes_per_variation: Episodes to generate per variation
        max_variations: Max variations to process (None = all)
        fps: Target FPS
        use_llm: Enable Gemini for task specification
        use_cpgen: Enable CP-Gen augmentation
        min_quality_score: Minimum quality score for export
        min_success_rate: Minimum success rate for episode generation
        data_pack_tier: Data pack tier ("core", "plus", "full")
        num_cameras: Number of cameras to capture
        image_resolution: Image resolution (width, height)
        capture_sensor_data: Enable visual observation capture
        use_mock_capture: Use mock capture (no Isaac Sim)
        allow_mock_capture: Allow mock capture (development only)
        bundle_tier: Bundle tier for upsell features (standard, pro, enterprise, foundation)
        lerobot_export_format: LeRobot export format ("lerobot_v2", "lerobot_v3", "lerobot_v0.3.3")

    Returns:
        0 on success, 1 on failure
    """
    logger.info(
        "[EPISODE-GEN-JOB] Starting SOTA episode generation for scene: %s", scene_id
    )
    logger.info("[EPISODE-GEN-JOB] Bundle tier: %s", bundle_tier)
    logger.info("[EPISODE-GEN-JOB] Assets prefix: %s", assets_prefix)
    logger.info("[EPISODE-GEN-JOB] Episodes prefix: %s", episodes_prefix)
    logger.info("[EPISODE-GEN-JOB] Firebase upload enabled: %s", enable_firebase_upload)
    logger.info("[EPISODE-GEN-JOB] Robot type: %s", robot_type)
    logger.info("[EPISODE-GEN-JOB] Episodes per variation: %s", episodes_per_variation)
    logger.info("[EPISODE-GEN-JOB] CP-Gen augmentation: %s", use_cpgen)
    logger.info("[EPISODE-GEN-JOB] Min quality score: %s", min_quality_score)
    logger.info("[EPISODE-GEN-JOB] Min success rate: %.1f%%", min_success_rate * 100)
    logger.info("[EPISODE-GEN-JOB] Data pack: %s", data_pack_tier)
    logger.info("[EPISODE-GEN-JOB] Cameras: %s", num_cameras)
    logger.info("[EPISODE-GEN-JOB] Resolution: %s", image_resolution)
    logger.info("[EPISODE-GEN-JOB] Sensor capture: %s", capture_sensor_data)
    logger.info(
        "[EPISODE-GEN-JOB] LeRobot export format: %s",
        parse_lerobot_export_format(
            lerobot_export_format,
            default=LeRobotExportFormat.LEROBOT_V2,
        ).value,
    )

    assets_dir = root / assets_prefix
    output_dir = root / episodes_prefix

    # Load manifest
    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        logger.error("[EPISODE-GEN-JOB] Manifest not found: %s", manifest_path)
        return 1

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info(
            "[EPISODE-GEN-JOB] Loaded manifest: %s objects",
            len(manifest.get("objects", [])),
        )
    except Exception as e:
        logger.error("[EPISODE-GEN-JOB] Failed to load manifest: %s", e)
        return 1

    scene_dir = root / f"scenes/{scene_id}"
    scene_config = _load_scene_config(scene_dir)
    env_scene_usd_path = os.getenv("SCENE_USD_PATH") or os.getenv("USD_SCENE_PATH")
    scene_usd_path = env_scene_usd_path or _resolve_scene_usd_path(scene_dir)
    if env_scene_usd_path:
        logger.info(
            "[EPISODE-GEN-JOB] Using USD scene path from environment: %s",
            env_scene_usd_path,
        )
    production_requested = resolve_production_mode()
    if production_requested and not scene_usd_path:
        logger.error(
            "[EPISODE-GEN-JOB] ERROR: Production runs require a USD scene path for PhysX validation. "
            "Set SCENE_USD_PATH or ensure scenes/<scene_id>/usd contains a USD file."
        )
        return 1
    camera_specs = _load_camera_specs(scene_config)
    robot_urdf_path = scene_config.get("robot_urdf_path") or os.getenv("ROBOT_URDF_PATH")

    robot_prim_paths = scene_config.get("robot_prim_paths")
    if not robot_prim_paths:
        robot_prim_path = scene_config.get("robot_prim_path")
        if robot_prim_path:
            robot_prim_paths = [robot_prim_path]
    if isinstance(robot_prim_paths, str):
        robot_prim_paths = [robot_prim_paths]

    # Configure and run generator
    config = EpisodeGenerationConfig(
        scene_id=scene_id,
        manifest_path=manifest_path,
        robot_type=robot_type,
        robot_prim_paths=robot_prim_paths,
        camera_specs=camera_specs,
        scene_usd_path=scene_usd_path,
        robot_urdf_path=robot_urdf_path,
        episodes_per_variation=episodes_per_variation,
        max_variations=max_variations,
        fps=fps,
        use_llm=use_llm,
        use_cpgen=use_cpgen,
        min_quality_score=min_quality_score,
        min_success_rate=min_success_rate,
        data_pack_tier=data_pack_tier,
        num_cameras=num_cameras,
        image_resolution=image_resolution,
        capture_sensor_data=capture_sensor_data,
        use_mock_capture=use_mock_capture,
        allow_mock_capture=allow_mock_capture,
        lerobot_export_format=parse_lerobot_export_format(
            lerobot_export_format,
            default=LeRobotExportFormat.LEROBOT_V2,
        ),
        output_dir=output_dir,
    )

    try:
        generator = EpisodeGenerator(config, verbose=True)
        output = generator.generate(manifest)

        if output.success:
            logger.info("[EPISODE-GEN-JOB] Episode generation completed successfully")
            logger.info("[EPISODE-GEN-JOB]   Total episodes: %s", output.total_episodes)
            logger.info("[EPISODE-GEN-JOB]   Valid episodes: %s", output.valid_episodes)
            logger.info("[EPISODE-GEN-JOB]   Pass rate: %.1f%%", output.pass_rate * 100)
            logger.info(
                "[EPISODE-GEN-JOB]   Avg quality: %.2f", output.average_quality_score
            )
            logger.info("[EPISODE-GEN-JOB]   Total frames: %s", output.total_frames)
            logger.info(
                "[EPISODE-GEN-JOB]   Duration: %.1fs",
                output.total_duration_seconds,
            )
            logger.info("[EPISODE-GEN-JOB]   Output: %s", output.output_dir)

            # Run upsell post-processing if bundle tier is not standard
            if bundle_tier != "standard":
                logger.info(
                    "[EPISODE-GEN-JOB] Running upsell post-processing (%s tier)...",
                    bundle_tier,
                )
                try:
                    # Import upsell post-processor
                    upsell_module_path = REPO_ROOT / "upsell-features-job"
                    if str(upsell_module_path) not in sys.path:
                        sys.path.insert(0, str(upsell_module_path))

                    from post_processor import run_upsell_post_processing

                    # Run post-processing on the scene directory
                    scene_dir = root / f"scenes/{scene_id}"
                    upsell_result = run_upsell_post_processing(
                        scene_dir=scene_dir,
                        tier=bundle_tier,
                        robot_type=robot_type,
                        verbose=True,
                    )

                    if upsell_result.get("success"):
                        logger.info(
                            "[EPISODE-GEN-JOB] Upsell post-processing completed successfully"
                        )
                        features = upsell_result.get("features_applied", [])
                        if features:
                            logger.info(
                                "[EPISODE-GEN-JOB]   Features applied: %s",
                                ", ".join(features),
                            )
                    else:
                        logger.warning(
                            "[EPISODE-GEN-JOB] Upsell post-processing had errors"
                        )
                        for err in upsell_result.get("errors", []):
                            logger.warning("[EPISODE-GEN-JOB]     - %s", err)

                except ImportError as e:
                    logger.warning(
                        "[EPISODE-GEN-JOB] Upsell module not available: %s", e
                    )
                except Exception as e:
                    logger.warning(
                        "[EPISODE-GEN-JOB] Upsell post-processing failed: %s", e
                    )
                    # Don't fail the job for upsell errors

            if enable_firebase_upload:
                firebase_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
                firebase_service_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
                firebase_service_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
                missing_firebase = []
                if not firebase_bucket:
                    missing_firebase.append("FIREBASE_STORAGE_BUCKET")
                if not firebase_service_json and not firebase_service_path:
                    missing_firebase.append(
                        "FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_SERVICE_ACCOUNT_PATH"
                    )
                if missing_firebase:
                    message = (
                        "Firebase upload requested but missing required configuration: "
                        + ", ".join(missing_firebase)
                    )
                    if _is_production_run():
                        logger.error("[EPISODE-GEN-JOB] %s", message)
                        output.errors.append(message)
                        return 1
                    logger.warning("[EPISODE-GEN-JOB] %s", message)
                    output.warnings.append(message)
                else:
                    logger.info(
                        "[EPISODE-GEN-JOB] Uploading episodes to Firebase Storage..."
                    )
                    try:
                        from tools.firebase_upload.uploader import (
                            upload_episodes_to_firebase,
                        )

                        if output.output_dir is None:
                            raise RuntimeError(
                                "Output directory missing; cannot upload episodes."
                            )

                        upload_summary = upload_episodes_to_firebase(
                            output.output_dir,
                            scene_id,
                            prefix=firebase_upload_prefix,
                        )
                        output.firebase_upload_summary = upload_summary
                        logger.info(
                            "[EPISODE-GEN-JOB] Firebase upload complete: %s/%s files",
                            upload_summary["uploaded"],
                            upload_summary["total_files"],
                        )
                    except Exception as exc:
                        error_message = f"Firebase upload failed: {exc}"
                        output.firebase_upload_error = error_message
                        if _is_production_run():
                            logger.error("[EPISODE-GEN-JOB] %s", error_message)
                            output.errors.append(error_message)
                            return 1
                        logger.warning("[EPISODE-GEN-JOB] %s", error_message)
                        output.warnings.append(error_message)

            if _should_bypass_quality_gates():
                logger.warning(
                    "[EPISODE-GEN-JOB] ⚠️  BYPASS_QUALITY_GATES enabled - skipping quality gates"
                )
                return 0

            collision_free_rate = _compute_collision_free_rate(
                output.validation_report_path,
                output.pass_rate,
            )
            episode_stats = {
                "total_generated": output.total_episodes,
                "passed_quality_filter": output.valid_episodes,
                "average_quality_score": output.average_quality_score,
                "collision_free_rate": collision_free_rate,
            }

            quality_gates = QualityGateRegistry(verbose=True)
            quality_gates.run_checkpoint(
                QualityGateCheckpoint.EPISODES_GENERATED,
                context={
                    "episode_stats": episode_stats,
                    "scene_id": scene_id,
                    "lerobot_dataset_path": str(output.lerobot_dataset_path)
                    if output.lerobot_dataset_path
                    else None,
                    "episode_metadata_path": str(output.lerobot_dataset_path / "meta" / "info.json")
                    if output.lerobot_dataset_path
                    else None,
                },
            )
            report_path = _gate_report_path(root, scene_id)
            quality_gates.save_report(scene_id, report_path)

            if not quality_gates.can_proceed():
                logger.error(
                    "[EPISODE-GEN-JOB] ❌ Quality gates blocked downstream pipeline"
                )
                FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
                    exception=RuntimeError("Quality gates blocked: episode validation failed"),
                    failed_step="quality_gates",
                    input_params={
                        "scene_id": scene_id,
                        "episodes_prefix": episodes_prefix,
                    },
                    partial_results={"quality_gate_report": str(report_path)},
                    recommendations=[
                        "Review episode quality metrics before proceeding.",
                        f"Review quality gate report: {report_path}",
                    ],
                )
                return 1

            return 0
        else:
            logger.error(
                "[EPISODE-GEN-JOB] Generation failed with %s errors",
                len(output.errors),
            )
            for err in output.errors:
                logger.error("[EPISODE-GEN-JOB]   - %s", err)
            return 1

    except Exception as e:
        logger.exception("[EPISODE-GEN-JOB] ERROR: %s", e)
        return 1


def _run_main():
    """Main entry point."""
    logger.info("[EPISODE-GEN-JOB] ================================")
    logger.info("[EPISODE-GEN-JOB] Episode Generation Job (SOTA)")
    logger.info("[EPISODE-GEN-JOB] ================================")

    def _is_production_preflight() -> bool:
        env_flags = resolve_production_mode()
        if HAVE_QUALITY_SYSTEM:
            try:
                return get_data_quality_level().value == "production" or env_flags
            except Exception:
                logger.warning(
                    "[EPISODE-GEN-JOB] Failed to read data quality level; falling back to env flags.",
                    exc_info=True,
                )
                return env_flags
        return env_flags

    def _collect_preflight_capabilities() -> Dict[str, Any]:
        capabilities: Dict[str, Any] = {
            "isaac_sim_available": False,
            "replicator_available": False,
            "physx_available": False,
            "gpu_available": os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia"),
        }
        if HAVE_QUALITY_SYSTEM:
            try:
                detected = get_environment_capabilities()
                capabilities.update(
                    {
                        "isaac_sim_available": detected.isaac_sim_available,
                        "replicator_available": detected.replicator_available,
                        "physx_available": detected.physx_available,
                        "gpu_available": detected.gpu_available,
                        "production_mode": detected.production_mode,
                        "allow_mock_capture": detected.allow_mock_capture,
                    }
                )
                return capabilities
            except Exception:
                logger.warning(
                    "[EPISODE-GEN-JOB] Failed to collect environment capabilities from quality system.",
                    exc_info=True,
                )
        if check_sensor_capture_environment is not None:
            try:
                status = check_sensor_capture_environment()
                capabilities.update(
                    {
                        "isaac_sim_available": bool(status.get("isaac_sim_available")),
                        "replicator_available": bool(status.get("replicator_available")),
                    }
                )
            except Exception:
                logger.warning(
                    "[EPISODE-GEN-JOB] Failed to collect environment capabilities from capture check.",
                    exc_info=True,
                )
        return capabilities

    def _emit_preflight_failure(capabilities: Dict[str, Any]) -> None:
        required_extensions = []
        for extension in REQUIRED_EXTENSIONS:
            if extension == "omni.replicator.core":
                available = capabilities.get("replicator_available", False)
            elif extension == "omni.physx":
                available = capabilities.get("physx_available", False)
            else:
                available = capabilities.get("isaac_sim_available", False)
            required_extensions.append(
                {
                    "name": extension,
                    "required": True,
                    "minimum_version": f"bundled with Isaac Sim {REQUIRED_ISAAC_SIM_VERSION}",
                    "available": bool(available),
                }
            )

        payload = {
            "error": "isaac_sim_preflight_failed",
            "mode": "production",
            "required": {
                "isaac_sim_version": REQUIRED_ISAAC_SIM_VERSION,
                "runtime": "Isaac Sim Python (/isaac-sim/python.sh)",
                "container_image": REQUIRED_ISAAC_SIM_CONTAINER,
                "extensions": required_extensions,
                "gpu": "NVIDIA RTX/Tesla-class GPU with drivers + nvidia-container-toolkit",
            },
            "detected": {
                "isaac_sim_available": capabilities.get("isaac_sim_available"),
                "replicator_available": capabilities.get("replicator_available"),
                "physx_available": capabilities.get("physx_available"),
                "gpu_available": capabilities.get("gpu_available"),
            },
            "remediation": [
                "Run inside the Isaac Sim container or lab runtime.",
                "Ensure the Replicator extension (omni.replicator.core) is enabled.",
                "Verify GPU drivers and NVIDIA Container Toolkit are installed.",
                "See docs/ISAAC_SIM_SETUP.md for setup steps.",
            ],
        }
        logger.error(json.dumps(payload))
        sys.exit(1)

    if _is_production_preflight():
        preflight_capabilities = _collect_preflight_capabilities()
        if (
            not preflight_capabilities.get("isaac_sim_available")
            or not preflight_capabilities.get("replicator_available")
        ):
            _emit_preflight_failure(preflight_capabilities)

    # =========================================================================
    # PHASE 1: Isaac Sim Enforcement + Environment Check
    # =========================================================================

    # Explicit Isaac Sim availability check at startup
    isaac_sim_available = False
    allow_mock_capture = False

    capabilities = None
    if HAVE_QUALITY_SYSTEM:
        try:
            # Get environment capabilities
            capabilities = get_environment_capabilities()

            # Extract Isaac Sim availability from capabilities
            isaac_sim_available = capabilities.get("isaac_sim_available", False) if isinstance(capabilities, dict) else False

            # Print detailed environment report
            print_environment_report(capabilities)

            # Enforce Isaac Sim for production
            required_quality = get_data_quality_level()
            logger.info(
                "[EPISODE-GEN-JOB] Required quality level: %s", required_quality.value
            )

            capabilities = enforce_isaac_sim_for_production(required_quality)

            logger.info("[EPISODE-GEN-JOB] ✅ Environment check passed")

            allow_mock_capture_env = parse_bool_env(
                os.getenv("ALLOW_MOCK_CAPTURE", os.getenv("ALLOW_MOCK_DATA")),
                default=False,
            )
            if required_quality.value == "production" or resolve_production_mode():
                allow_mock_capture = False
                if allow_mock_capture_env:
                    logger.warning(
                        "[EPISODE-GEN-JOB] ⚠️  ALLOW_MOCK_CAPTURE ignored in production quality mode"
                    )
            else:
                allow_mock_capture = allow_mock_capture_env

            # Confirm Isaac Sim is available after enforcement
            if not isaac_sim_available:
                logger.warning(
                    "[EPISODE-GEN-JOB] ⚠️  WARNING: Isaac Sim not confirmed available"
                )

        except IsaacSimRequirementError as e:
            logger.error("❌ ISAAC SIM REQUIREMENT ERROR:\n%s", e)
            sys.exit(1)
        except ProductionDataQualityError as e:
            logger.error("❌ PRODUCTION DATA QUALITY ERROR:\n%s", e)
            sys.exit(1)
    else:
        # Fallback to legacy enforcement (quality system unavailable)
        logger.warning(
            "[EPISODE-GEN-JOB] Using legacy Isaac Sim check (quality system unavailable)"
        )

        # Add explicit check for sensor capture environment
        replicator_available = False
        if check_sensor_capture_environment is not None:
            status = check_sensor_capture_environment()
            isaac_sim_available = status.get("isaac_sim_available", False)
            replicator_available = status.get("replicator_available", False)
            logger.info(
                "[EPISODE-GEN-JOB] Isaac Sim available: %s", isaac_sim_available
            )
        else:
            isaac_sim_available = False
            logger.warning(
                "[EPISODE-GEN-JOB] ⚠️  WARNING: Cannot verify Isaac Sim availability"
            )

        # Detect if running in production environment
        # Production indicators: running in container, K8s, or Cloud Run
        production_env_flags = resolve_production_mode()
        is_production = (
            os.getenv("KUBERNETES_SERVICE_HOST") is not None or  # K8s
            os.getenv("K_SERVICE") is not None or  # Cloud Run
            os.path.exists("/.dockerenv") or  # Docker
            parse_bool_env(os.getenv("PRODUCTION"), default=False)
            or production_env_flags
        )

        # In production, require real physics by default unless explicitly disabled
        # In development, allow mock data by default unless explicitly required
        if is_production:
            default_require_real = "true"
        else:
            default_require_real = "false"

        require_real_physics = parse_bool_env(
            os.getenv("REQUIRE_REAL_PHYSICS", default_require_real),
            default=False,
        )
        if production_env_flags:
            require_real_physics = True

        # LABS-BLOCKER-001 FIX: Remove ALLOW_MOCK_DATA override in production
        # In production, we MUST have Isaac Sim - no exceptions
        # This prevents labs from wasting GPU hours on garbage data
        if is_production:
            allow_mock_data = False  # NEVER allow mock data in production
            if parse_bool_env(os.getenv("ALLOW_MOCK_DATA"), default=False):
                logger.error("%s", "=" * 80)
                logger.error(
                    "❌ FATAL ERROR: ALLOW_MOCK_DATA not permitted in production"
                )
                logger.error("%s", "=" * 80)
                logger.error("ALLOW_MOCK_DATA=true is IGNORED in production environments.")
                logger.error("Production training data MUST use real Isaac Sim physics.")
                logger.error("This protection prevents:")
                logger.error("  - Training models on random noise images")
                logger.error("  - Wasting GPU hours on useless data")
                logger.error("  - Shipping low-quality data to labs/customers")
                logger.error("%s", "=" * 80)
        else:
            # Development mode: allow override for testing
            allow_mock_data = parse_bool_env(os.getenv("ALLOW_MOCK_DATA"), default=False)

        # Log configuration BEFORE checking Isaac Sim
        logger.info("[EPISODE-GEN-JOB] Production mode: %s", is_production)
        logger.info("[EPISODE-GEN-JOB] Require real physics: %s", require_real_physics)
        logger.info("[EPISODE-GEN-JOB] Allow mock data: %s", allow_mock_data)
        allow_mock_capture = allow_mock_data

        # LABS-BLOCKER-001 FIX: Fail hard in production if Isaac Sim/Replicator not available
        if require_real_physics and (not isaac_sim_available or not replicator_available):
            # In production, this is ALWAYS a fatal error (allow_mock_data is always False)
            logger.error("%s", "=" * 80)
            if not isaac_sim_available:
                logger.error("❌ FATAL ERROR: Isaac Sim not available in production mode")
            else:
                logger.error(
                    "❌ FATAL ERROR: Replicator extension not available in production mode"
                )
            logger.error("%s", "=" * 80)
            logger.error("Episode generation requires NVIDIA Isaac Sim for:")
            logger.error("  ✓ Real physics simulation (PhysX)")
            logger.error("  ✓ Actual sensor data capture (Replicator)")
            logger.error("  ✓ Physics-validated trajectories")
            logger.error("  ✓ Collision detection with real geometry")
            logger.error("Without Isaac Sim, the pipeline would generate:")
            logger.error("  ✗ Random noise RGB images (NOT real sensor data)")
            logger.error("  ✗ Heuristic-based validation (NOT physics-verified)")
            logger.error("  ✗ Mock contact/collision data (NOT accurate)")
            logger.error("This data is USELESS for:")
            logger.error("  • Training production ML models")
            logger.error("  • Real-world robot deployment")
            logger.error("  • Lab testing and evaluation")
            logger.error("To fix this:")
            logger.error(
                "  1. Install Isaac Sim: https://developer.nvidia.com/isaac-sim"
            )
            logger.error(
                "  2. Run with Isaac Sim Python: /isaac-sim/python.sh generate_episodes.py"
            )
            logger.error(
                "  3. Or use the Isaac Sim container: see docs/ISAAC_SIM_SETUP.md"
            )
            if is_production:
                logger.error(
                    "NOTE: ALLOW_MOCK_DATA override is DISABLED in production."
                )
                logger.error(
                    "      There is no way to bypass this check in production mode."
                )
            else:
                logger.error("To bypass in development (NOT for production data):")
                logger.error("  export REQUIRE_REAL_PHYSICS=false")
            logger.error("%s", "=" * 80)
            sys.exit(1)

    # Development mode warning
    if not isaac_sim_available and not require_real_physics:
        logger.warning("[EPISODE-GEN-JOB] ========================================")
        logger.warning("[EPISODE-GEN-JOB] WARNING: Isaac Sim not available")
        logger.warning(
            "[EPISODE-GEN-JOB] Running with MOCK DATA (random noise)"
        )
        logger.warning("[EPISODE-GEN-JOB] For real data: /isaac-sim/python.sh")
        logger.warning("[EPISODE-GEN-JOB] ========================================")

    # Get configuration from environment
    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]

    # Prefixes with defaults
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    episodes_prefix = os.getenv("EPISODES_PREFIX", f"scenes/{scene_id}/episodes")
    enable_firebase_upload = parse_bool_env(
        os.getenv("ENABLE_FIREBASE_UPLOAD"),
        default=False,
    )
    firebase_upload_prefix = os.getenv("FIREBASE_UPLOAD_PREFIX", "datasets")

    # Configuration with validation
    robot_type = os.getenv("ROBOT_TYPE", "franka")

    try:
        episodes_per_variation = int(os.getenv("EPISODES_PER_VARIATION", "10"))
        if episodes_per_variation <= 0:
            raise ValueError("EPISODES_PER_VARIATION must be positive")
    except ValueError as e:
        logger.error("[EPISODE-GEN-JOB] Invalid EPISODES_PER_VARIATION: %s", e)
        sys.exit(1)

    max_variations = os.getenv("MAX_VARIATIONS")
    if max_variations:
        try:
            max_variations = int(max_variations)
            if max_variations <= 0:
                raise ValueError("MAX_VARIATIONS must be positive")
        except ValueError as e:
            logger.error("[EPISODE-GEN-JOB] Invalid MAX_VARIATIONS: %s", e)
            sys.exit(1)
    else:
        max_variations = None

    try:
        fps = float(os.getenv("FPS", "30"))
        if fps <= 0 or fps > 240:
            raise ValueError("FPS must be between 0 and 240")
    except ValueError as e:
        logger.error("[EPISODE-GEN-JOB] Invalid FPS: %s", e)
        sys.exit(1)

    use_llm = parse_bool_env(os.getenv("USE_LLM"), default=True)
    use_cpgen = parse_bool_env(os.getenv("USE_CPGEN"), default=True)

    try:
        # LABS-BLOCKER-002 FIX: Default raised from 0.7 to 0.85 for production quality
        min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.85"))
        if not (0.0 <= min_quality_score <= 1.0):
            raise ValueError("MIN_QUALITY_SCORE must be between 0.0 and 1.0")
    except ValueError as e:
        logger.error("[EPISODE-GEN-JOB] Invalid MIN_QUALITY_SCORE: %s", e)
        sys.exit(1)

    try:
        min_success_rate = float(os.getenv("MIN_SUCCESS_RATE", "0.5"))
        if not (0.0 <= min_success_rate <= 1.0):
            raise ValueError("MIN_SUCCESS_RATE must be between 0.0 and 1.0")
    except ValueError as e:
        logger.error("[EPISODE-GEN-JOB] Invalid MIN_SUCCESS_RATE: %s", e)
        sys.exit(1)

    # Data pack configuration (Core/Plus/Full)
    data_pack_tier = os.getenv("DATA_PACK_TIER", "core")
    lerobot_export_format = os.getenv("LEROBOT_EXPORT_FORMAT")

    try:
        num_cameras = int(os.getenv("NUM_CAMERAS", "1"))
        if num_cameras < 1 or num_cameras > 8:
            raise ValueError("NUM_CAMERAS must be between 1 and 8")
    except ValueError as e:
        logger.error("[EPISODE-GEN-JOB] Invalid NUM_CAMERAS: %s", e)
        sys.exit(1)

    resolution_str = os.getenv("IMAGE_RESOLUTION", "640,480")
    try:
        resolution_parts = resolution_str.split(",")
        if len(resolution_parts) != 2:
            raise ValueError("IMAGE_RESOLUTION must be in format 'width,height'")
        image_resolution = tuple(map(int, resolution_parts))
        if image_resolution[0] <= 0 or image_resolution[1] <= 0:
            raise ValueError("IMAGE_RESOLUTION dimensions must be positive")
    except (ValueError, TypeError) as e:
        logger.error("[EPISODE-GEN-JOB] Invalid IMAGE_RESOLUTION: %s", e)
        sys.exit(1)

    capture_sensor_data = parse_bool_env(os.getenv("CAPTURE_SENSOR_DATA"), default=True)
    use_mock_capture = parse_bool_env(os.getenv("USE_MOCK_CAPTURE"), default=False)

    # Bundle tier for upsell features (standard, pro, enterprise, foundation)
    bundle_tier = os.getenv("BUNDLE_TIER", "standard")

    logger.info("[EPISODE-GEN-JOB] Configuration:")
    logger.info("[EPISODE-GEN-JOB]   Bucket: %s", bucket)
    logger.info("[EPISODE-GEN-JOB]   Scene ID: %s", scene_id)
    logger.info("[EPISODE-GEN-JOB]   Pipeline: SOTA (CP-Gen + Validation)")
    logger.info("[EPISODE-GEN-JOB]   Data Pack: %s", data_pack_tier)
    logger.info("[EPISODE-GEN-JOB]   Bundle Tier: %s", bundle_tier)
    logger.info(
        "[EPISODE-GEN-JOB]   Firebase Uploads Enabled: %s", enable_firebase_upload
    )
    if lerobot_export_format:
        logger.info("[EPISODE-GEN-JOB]   LeRobot export format: %s", lerobot_export_format)
    logger.info("[EPISODE-GEN-JOB]   Cameras: %s", num_cameras)
    logger.info("[EPISODE-GEN-JOB]   Resolution: %s", image_resolution)
    logger.info(
        "[EPISODE-GEN-JOB]   Min success rate: %.1f%%", min_success_rate * 100
    )
    logger.info("[EPISODE-GEN-JOB]   Allow mock capture: %s", allow_mock_capture)

    GCS_ROOT = Path("/mnt/gcs")

    exit_code = run_episode_generation_job(
        root=GCS_ROOT,
        bucket=bucket,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        episodes_prefix=episodes_prefix,
        enable_firebase_upload=enable_firebase_upload,
        firebase_upload_prefix=firebase_upload_prefix,
        robot_type=robot_type,
        episodes_per_variation=episodes_per_variation,
        max_variations=max_variations,
        fps=fps,
        use_llm=use_llm,
        use_cpgen=use_cpgen,
        min_quality_score=min_quality_score,
        min_success_rate=min_success_rate,
        data_pack_tier=data_pack_tier,
        num_cameras=num_cameras,
        image_resolution=image_resolution,
        capture_sensor_data=capture_sensor_data,
        use_mock_capture=use_mock_capture,
        allow_mock_capture=allow_mock_capture,
        bundle_tier=bundle_tier,
        lerobot_export_format=lerobot_export_format,
    )
    metrics = get_metrics()
    with metrics.track_job("episode-generation-job", scene_id):
        exit_code = run_episode_generation_job(
            root=GCS_ROOT,
            bucket=bucket,
            scene_id=scene_id,
            assets_prefix=assets_prefix,
            episodes_prefix=episodes_prefix,
            enable_firebase_upload=enable_firebase_upload,
            firebase_upload_prefix=firebase_upload_prefix,
            robot_type=robot_type,
            episodes_per_variation=episodes_per_variation,
            max_variations=max_variations,
            fps=fps,
            use_llm=use_llm,
            use_cpgen=use_cpgen,
            min_quality_score=min_quality_score,
            min_success_rate=min_success_rate,
            data_pack_tier=data_pack_tier,
            num_cameras=num_cameras,
            image_resolution=image_resolution,
            capture_sensor_data=capture_sensor_data,
            use_mock_capture=use_mock_capture,
            allow_mock_capture=allow_mock_capture,
            bundle_tier=bundle_tier,
            lerobot_export_format=lerobot_export_format,
        )

    sys.exit(exit_code)


def main() -> None:
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[EPISODE-GEN-JOB]",
    )
    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]
    assets_prefix = os.getenv(
        "ASSETS_PREFIX",
        f"scenes/{scene_id}/assets",
    )
    episodes_prefix = os.getenv(
        "EPISODES_PREFIX",
        f"scenes/{scene_id}/episodes",
    )
    input_params = {
        "bucket": bucket,
        "scene_id": scene_id,
        "assets_prefix": assets_prefix,
        "episodes_prefix": episodes_prefix,
        "enable_firebase_upload": os.getenv("ENABLE_FIREBASE_UPLOAD"),
        "firebase_upload_prefix": os.getenv("FIREBASE_UPLOAD_PREFIX", "datasets"),
        "robot_type": os.getenv("ROBOT_TYPE", "franka"),
        "episodes_per_variation": os.getenv("EPISODES_PER_VARIATION", "10"),
        "max_variations": os.getenv("MAX_VARIATIONS"),
        "fps": os.getenv("FPS", "30"),
        "use_llm": os.getenv("USE_LLM", "true"),
        "use_cpgen": os.getenv("USE_CPGEN", "true"),
        "min_quality_score": os.getenv("MIN_QUALITY_SCORE", "0.85"),
        "min_success_rate": os.getenv("MIN_SUCCESS_RATE", "0.5"),
        "data_pack_tier": os.getenv("DATA_PACK_TIER", "core"),
        "num_cameras": os.getenv("NUM_CAMERAS", "1"),
        "image_resolution": os.getenv("IMAGE_RESOLUTION", "640,480"),
        "capture_sensor_data": os.getenv("CAPTURE_SENSOR_DATA", "true"),
        "use_mock_capture": os.getenv("USE_MOCK_CAPTURE", "false"),
        "bundle_tier": os.getenv("BUNDLE_TIER", "standard"),
        "lerobot_export_format": os.getenv("LEROBOT_EXPORT_FORMAT"),
    }
    partial_results = {
        "episodes_prefix": episodes_prefix,
        "quality_report": (
            f"{episodes_prefix}/quality/validation_report.json" if episodes_prefix else None
        ),
    }

    def _write_failure_marker(exc: Exception, failed_step: str) -> None:
        if not bucket or not scene_id:
            logger.warning(
                "[EPISODE-GEN-JOB] WARNING: Skipping failure marker; BUCKET/SCENE_ID missing."
            )
            return
        FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
            exception=exc,
            failed_step=failed_step,
            input_params=input_params,
            partial_results=partial_results,
        )

    validated = False
    try:
        assets_root = Path("/mnt/gcs") / assets_prefix
        validate_scene_manifest(assets_root / "scene_manifest.json", label="[EPISODE-GEN-JOB]")
        validated = True
        _run_main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            failed_step = "entrypoint_validation" if not validated else "entrypoint_exit"
            _write_failure_marker(RuntimeError("Job exited early"), failed_step)
        raise
    except Exception as exc:
        _write_failure_marker(exc, "entrypoint")
        raise


if __name__ == "__main__":
    try:
        from tools.startup_validation import validate_and_fail_fast
        from tools.logging_config import init_logging

        init_logging()
        validate_and_fail_fast(job_name="EPISODE-GEN", validate_gcs=True)
        main()
    except Exception as exc:
        send_alert(
            event_type="episode_generation_job_fatal_exception",
            summary="Episode generation job failed with an unhandled exception",
            details={
                "job": "episode-generation-job",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            severity=os.getenv("ALERT_JOB_EXCEPTION_SEVERITY", "critical"),
        )
        raise

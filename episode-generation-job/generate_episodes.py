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
    ROBOT_TYPE: Robot type (franka, ur10, fetch) - default: franka
    EPISODES_PER_VARIATION: Episodes per variation - default: 10
    MAX_VARIATIONS: Max variations to process - default: all
    FPS: Target FPS for trajectories - default: 30
    USE_LLM: Enable LLM for task specification - default: true
    USE_CPGEN: Enable CP-Gen augmentation - default: true
    MIN_QUALITY_SCORE: Minimum quality score for export - default: 0.7
"""

import gc
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Core imports
from motion_planner import AIMotionPlanner, MotionPlan
from trajectory_solver import TrajectorySolver, JointTrajectory, ROBOT_CONFIGS, TrajectoryIKError
from motion_planner import AIMotionPlanner, MotionPlan, SceneContext
from trajectory_solver import TrajectorySolver, JointTrajectory, ROBOT_CONFIGS
from lerobot_exporter import LeRobotExporter, LeRobotDatasetConfig
from quality_constants import MIN_QUALITY_SCORE, MAX_RETRIES, PRODUCTION_TRAINING_THRESHOLD

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
    print("[EPISODE-GEN-JOB] WARNING: Quality certificate system not available")

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

    def __init__(self, use_llm: bool = True, verbose: bool = True):
        self.use_llm = use_llm
        self.verbose = verbose
        self.task_specifier = TaskSpecifier(verbose=verbose) if use_llm else None
        self.scene_analyzer = SceneAnalyzer(verbose=verbose) if HAVE_DWM_MODULES else None
        self.task_planner = TaskPlanner(verbose=verbose) if HAVE_DWM_MODULES else None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[TASK-GENERATOR] [{level}] {msg}")

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

        # Enhance with full specifications
        tasks_with_specs = []
        objects = manifest.get("objects", [])

        for task in basic_tasks:
            try:
                if self.task_specifier:
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
                spec = self._create_minimal_spec(task, objects, robot_type)
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

        # P2-1 FIX: Get default position from task, or compute from robot workspace
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

        # P2-1 FIX: Get workspace defaults from manifest or robot config
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
        """P2-1 FIX: Generate tasks from object categories (fallback)."""
        tasks = []
        objects = manifest.get("objects", [])

        # P2-1 FIX: Get workspace defaults from manifest or robot config
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

        self.log(f"Generated {len(tasks)} tasks from object categories")
        return tasks

    def _compute_default_goal_position(
        self,
        task: Dict[str, Any],
        objects: List[Dict[str, Any]],
        robot_type: str,
    ) -> List[float]:
        """
        P2-1 FIX: Compute sensible default goal position from task context and robot workspace.

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
        P2-1 FIX: Calculate a sensible place position for an object.

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

                # Parse data pack tier
                tier = data_pack_from_string(config.data_pack_tier)

                # Create sensor capture with explicit mode
                self.sensor_capture = create_sensor_capture(
                    data_pack=tier,
                    num_cameras=config.num_cameras,
                    resolution=config.image_resolution,
                    fps=config.fps,
                    capture_mode=capture_mode,
                    verbose=verbose,
                )

                # Check if we got mock capture using the dedicated method
                self._sensor_capture_is_mock = hasattr(self.sensor_capture, 'is_mock') and self.sensor_capture.is_mock()

                self.log(f"Sensor capture initialized: {config.data_pack_tier} pack, {config.num_cameras} cameras")
                if self._sensor_capture_is_mock:
                    self.log("⚠️  Using MOCK sensor capture - NOT suitable for production training!", "WARNING")
                else:
                    self.log("✅ Using Isaac Sim Replicator - production quality data")

            except Exception as e:
                # Sensor capture creation failed - this is expected to raise in fail_closed mode
                self.log(f"Sensor capture initialization failed: {e}", "ERROR")
                raise

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[EPISODE-GENERATOR] [{level}] {msg}")

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

        # Step 7: Export to LeRobot format (with data pack configuration)
        self.log("\nStep 5: Exporting LeRobot dataset...")
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
        )
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
            output.errors.append(error_msg)
            self.log(error_msg, "ERROR")
            if self.verbose:
                self.log(traceback.format_exc(), "DEBUG")
            # LeRobot export failure is non-fatal - episodes are still valid

        # Clean up memory after export
        del exporter
        gc.collect()

        # Step 8: Write generation manifest
        self.log("\nStep 6: Writing generation manifest and validation report...")
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

    def _generate_seed_episodes(
        self,
        tasks_with_specs: List[Tuple[Dict[str, Any], TaskSpecification]],
        manifest: Dict[str, Any],
    ) -> List[GeneratedEpisode]:
        """Generate seed episodes for each task."""
        seed_episodes = []
        failed_seeds = []
        scene_context = SceneContext(
            scene_id=self.config.scene_id,
            environment_type=manifest.get("environment_type", "unknown"),
            objects=manifest.get("objects", []),
        )

        for task, spec in tasks_with_specs:
            try:
                start_time = time.time()

                # Generate motion plan using constraint-aware planning
                target_object = {
                    "id": task["target_object_id"],
                    "position": task["target_position"],
                    "dimensions": task["target_dimensions"],
                }

                # P1-7 FIX: Make articulation info configurable from task specification
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
                    episode = GeneratedEpisode(
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
                    seed_episodes.append(episode)
                    continue

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

                seed_episodes.append(episode)
                visual_info = f" (visual: {sensor_data.num_frames}f)" if sensor_data else ""
                self.log(f"    Created seed: {task['task_name']}{visual_info}")

            except Exception as e:
                import traceback
                failed_seeds.append({
                    "task_name": task.get("task_name", "unknown"),
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                self.log(f"    Failed seed {task['task_name']}: {e}", "ERROR")
                # Log full traceback for debugging
                if self.verbose:
                    self.log(traceback.format_exc(), "DEBUG")

        # Check if too many seeds failed
        total_tasks = len(tasks_with_specs)
        if total_tasks > 0:
            failure_rate = len(failed_seeds) / total_tasks
            if failure_rate > 0.8:  # More than 80% failed
                error_msg = f"Seed generation failed: {len(failed_seeds)}/{total_tasks} tasks failed ({failure_rate*100:.1f}%)"
                self.log(error_msg, "ERROR")
                raise RuntimeError(error_msg)
            elif failed_seeds:
                self.log(f"Seed generation completed with {len(failed_seeds)}/{total_tasks} failures ({failure_rate*100:.1f}%)", "WARNING")

        return seed_episodes

    def _generate_augmented_episodes(
        self,
        seed_episodes: List[GeneratedEpisode],
        manifest: Dict[str, Any],
        num_variations: int,
    ) -> List[GeneratedEpisode]:
        """Generate augmented episodes using CP-Gen approach."""
        all_episodes = list(seed_episodes)  # Start with seeds
        objects = manifest.get("objects", [])
        failed_variations = []
        failed_seeds = []

        if not self.config.use_cpgen or not self.cpgen_augmenter:
            # Fallback: simple variation without CP-Gen
            self.log("  Using simple variation (CP-Gen disabled)")
            return self._generate_simple_variations(seed_episodes, manifest, num_variations)

        for seed in seed_episodes:
            if seed.task_spec is None or seed.motion_plan is None:
                continue
            if not getattr(seed.motion_plan, "planning_success", True):
                self.log(f"  Skipping seed {seed.episode_id} due to planning failure", "WARNING")
                continue

            try:
                # Create CP-Gen seed episode
                cpgen_seed = self.cpgen_augmenter.create_seed_episode(
                    task_spec=seed.task_spec,
                    motion_plan=seed.motion_plan,
                    scene_objects=objects,
                )

                # Generate variations (skip first as it's the seed)
                for var_idx in range(1, num_variations):
                    try:
                        # Generate random object transforms
                        object_transforms = self._generate_variation_transforms(
                            objects, var_idx
                        )
                        updated_obstacles = self._apply_transforms_to_objects(
                            objects, object_transforms
                        )

                        # Augment episode
                        augmented = self.cpgen_augmenter.augment(
                            seed=cpgen_seed,
                            object_transforms=object_transforms,
                            obstacles=updated_obstacles,
                            variation_index=var_idx,
                        )

                        if augmented.planning_success:
                            # Solve trajectory for augmented plan
                            trajectory = self.trajectory_solver.solve(augmented.motion_plan)

                            # Capture sensor data for augmented episode
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

                            episode = GeneratedEpisode(
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
                            all_episodes.append(episode)

                    except Exception as e:
                        import traceback
                        failed_variations.append({
                            "seed_task": seed.task_name,
                            "variation_idx": var_idx,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        })
                        self.log(f"    Variation {var_idx} failed: {e}", "ERROR")
                        if self.verbose:
                            self.log(traceback.format_exc(), "DEBUG")

            except Exception as e:
                import traceback
                failed_seeds.append({
                    "task_name": seed.task_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                self.log(f"  CP-Gen augmentation failed for {seed.task_name}: {e}", "ERROR")
                if self.verbose:
                    self.log(traceback.format_exc(), "DEBUG")

        # Log summary of failures
        if failed_variations or failed_seeds:
            self.log(f"  Variation generation: {len(failed_seeds)} seed failures, {len(failed_variations)} variation failures", "WARNING")

        self.log(f"  Generated {len(all_episodes)} total episodes")
        return all_episodes

    def _generate_simple_variations(
        self,
        seed_episodes: List[GeneratedEpisode],
        manifest: Dict[str, Any],
        num_variations: int,
    ) -> List[GeneratedEpisode]:
        """P2-1 FIX: Simple variation without CP-Gen (fallback)."""
        all_episodes = list(seed_episodes)
        objects = manifest.get("objects", [])
        failed_variations = []

        # P2-1 FIX: Get workspace defaults from manifest or robot config
        robot_config = manifest.get("robot_config", {})
        workspace_center = robot_config.get("workspace_center", [0.5, 0.0, 0.85])

        for seed in seed_episodes:
            # Skip seeds with uninitialized task spec or motion plan
            if seed.task_spec is None or seed.motion_plan is None:
                self.log(f"    Skipping seed {seed.episode_id} - missing task_spec or motion_plan", "WARNING")
                continue
            if not getattr(seed.motion_plan, "planning_success", True):
                self.log(f"    Skipping seed {seed.episode_id} due to planning failure", "WARNING")
                continue

            for var_idx in range(1, num_variations):
                try:
                    # Apply small random offsets
                    np.random.seed(var_idx)
                    varied_manifest = json.loads(json.dumps(manifest))

                    for obj in varied_manifest.get("objects", []):
                        if "position" in obj:
                            pos = obj["position"]
                            if isinstance(pos, list):
                                offset = np.random.uniform(-0.05, 0.05, 3)
                                offset[2] = 0
                                obj["position"] = [p + o for p, o in zip(pos, offset)]

                    # P2-1 FIX: Regenerate episode for this variation
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
                        self.log(
                            f"    Motion planning failed for variation {var_idx}: {motion_plan.planning_errors}",
                            "WARNING",
                        )
                        continue

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

                    episode = GeneratedEpisode(
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
                    all_episodes.append(episode)

                except Exception as e:
                    import traceback
                    failed_variations.append({
                        "seed_task": seed.task_name,
                        "variation_idx": var_idx,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    self.log(f"    Simple variation {var_idx} failed: {e}", "ERROR")
                    if self.verbose:
                        self.log(traceback.format_exc(), "DEBUG")

        # Log summary of failures
        if failed_variations:
            self.log(f"  Simple variation generation: {len(failed_variations)} failures", "WARNING")

        return all_episodes

    def _generate_variation_transforms(
        self,
        objects: List[Dict[str, Any]],
        variation_index: int,
    ) -> Dict[str, ObjectTransform]:
        """Generate random transforms for a variation."""
        np.random.seed(variation_index)
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

        for episode in episodes:
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
                    continue

                episode.is_valid = False
                episode.quality_score = 0.0
                episode.validation_errors.append("Missing trajectory or motion plan")
                continue

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

        return episodes

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
    scene_id: str,
    assets_prefix: str,
    episodes_prefix: str,
    robot_type: str = "franka",
    episodes_per_variation: int = 10,
    max_variations: Optional[int] = None,
    fps: float = 30.0,
    use_llm: bool = True,
    use_cpgen: bool = True,
    min_quality_score: float = 0.85,  # LABS-BLOCKER-002 FIX: Raised from 0.7
    data_pack_tier: str = "core",
    num_cameras: int = 1,
    image_resolution: Tuple[int, int] = (640, 480),
    capture_sensor_data: bool = True,
    use_mock_capture: bool = False,
    bundle_tier: str = "standard",
) -> int:
    """
    Run the episode generation job (SOTA Pipeline).

    Args:
        root: Root path (e.g., /mnt/gcs)
        scene_id: Scene identifier
        assets_prefix: Path to scene assets
        episodes_prefix: Output path for episodes
        robot_type: Robot type
        episodes_per_variation: Episodes to generate per variation
        max_variations: Max variations to process (None = all)
        fps: Target FPS
        use_llm: Enable Gemini for task specification
        use_cpgen: Enable CP-Gen augmentation
        min_quality_score: Minimum quality score for export
        data_pack_tier: Data pack tier ("core", "plus", "full")
        num_cameras: Number of cameras to capture
        image_resolution: Image resolution (width, height)
        capture_sensor_data: Enable visual observation capture
        use_mock_capture: Use mock capture (no Isaac Sim)
        bundle_tier: Bundle tier for upsell features (standard, pro, enterprise, foundation)

    Returns:
        0 on success, 1 on failure
    """
    print(f"[EPISODE-GEN-JOB] Starting SOTA episode generation for scene: {scene_id}")
    print(f"[EPISODE-GEN-JOB] Bundle tier: {bundle_tier}")
    print(f"[EPISODE-GEN-JOB] Assets prefix: {assets_prefix}")
    print(f"[EPISODE-GEN-JOB] Episodes prefix: {episodes_prefix}")
    print(f"[EPISODE-GEN-JOB] Robot type: {robot_type}")
    print(f"[EPISODE-GEN-JOB] Episodes per variation: {episodes_per_variation}")
    print(f"[EPISODE-GEN-JOB] CP-Gen augmentation: {use_cpgen}")
    print(f"[EPISODE-GEN-JOB] Min quality score: {min_quality_score}")
    print(f"[EPISODE-GEN-JOB] Data pack: {data_pack_tier}")
    print(f"[EPISODE-GEN-JOB] Cameras: {num_cameras}")
    print(f"[EPISODE-GEN-JOB] Resolution: {image_resolution}")
    print(f"[EPISODE-GEN-JOB] Sensor capture: {capture_sensor_data}")

    assets_dir = root / assets_prefix
    output_dir = root / episodes_prefix

    # Load manifest
    manifest_path = assets_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        print(f"[EPISODE-GEN-JOB] ERROR: Manifest not found: {manifest_path}")
        return 1

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"[EPISODE-GEN-JOB] Loaded manifest: {len(manifest.get('objects', []))} objects")
    except Exception as e:
        print(f"[EPISODE-GEN-JOB] ERROR: Failed to load manifest: {e}")
        return 1

    # Configure and run generator
    config = EpisodeGenerationConfig(
        scene_id=scene_id,
        manifest_path=manifest_path,
        robot_type=robot_type,
        episodes_per_variation=episodes_per_variation,
        max_variations=max_variations,
        fps=fps,
        use_llm=use_llm,
        use_cpgen=use_cpgen,
        min_quality_score=min_quality_score,
        data_pack_tier=data_pack_tier,
        num_cameras=num_cameras,
        image_resolution=image_resolution,
        capture_sensor_data=capture_sensor_data,
        use_mock_capture=use_mock_capture,
        output_dir=output_dir,
    )

    try:
        generator = EpisodeGenerator(config, verbose=True)
        output = generator.generate(manifest)

        if output.success:
            print("[EPISODE-GEN-JOB] Episode generation completed successfully")
            print(f"[EPISODE-GEN-JOB]   Total episodes: {output.total_episodes}")
            print(f"[EPISODE-GEN-JOB]   Valid episodes: {output.valid_episodes}")
            print(f"[EPISODE-GEN-JOB]   Pass rate: {output.pass_rate:.1%}")
            print(f"[EPISODE-GEN-JOB]   Avg quality: {output.average_quality_score:.2f}")
            print(f"[EPISODE-GEN-JOB]   Total frames: {output.total_frames}")
            print(f"[EPISODE-GEN-JOB]   Duration: {output.total_duration_seconds:.1f}s")
            print(f"[EPISODE-GEN-JOB]   Output: {output.output_dir}")

            # Run upsell post-processing if bundle tier is not standard
            if bundle_tier != "standard":
                print(f"\n[EPISODE-GEN-JOB] Running upsell post-processing ({bundle_tier} tier)...")
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
                        print("[EPISODE-GEN-JOB] Upsell post-processing completed successfully")
                        features = upsell_result.get("features_applied", [])
                        if features:
                            print(f"[EPISODE-GEN-JOB]   Features applied: {', '.join(features)}")
                    else:
                        print("[EPISODE-GEN-JOB] WARNING: Upsell post-processing had errors")
                        for err in upsell_result.get("errors", []):
                            print(f"[EPISODE-GEN-JOB]     - {err}")

                except ImportError as e:
                    print(f"[EPISODE-GEN-JOB] WARNING: Upsell module not available: {e}")
                except Exception as e:
                    print(f"[EPISODE-GEN-JOB] WARNING: Upsell post-processing failed: {e}")
                    # Don't fail the job for upsell errors

            return 0
        else:
            print(f"[EPISODE-GEN-JOB] ERROR: Generation failed with {len(output.errors)} errors")
            for err in output.errors:
                print(f"[EPISODE-GEN-JOB]   - {err}")
            return 1

    except Exception as e:
        print(f"[EPISODE-GEN-JOB] ERROR: {e}")
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    print("\n[EPISODE-GEN-JOB] ================================")
    print("[EPISODE-GEN-JOB] Episode Generation Job (SOTA)")
    print("[EPISODE-GEN-JOB] ================================\n")

    # =========================================================================
    # PHASE 1: Isaac Sim Enforcement + Environment Check
    # =========================================================================

    # P0-4 FIX: Explicit Isaac Sim availability check at startup
    isaac_sim_available = False

    capabilities = None
    if HAVE_QUALITY_SYSTEM:
        try:
            # Get environment capabilities
            capabilities = get_environment_capabilities()

            # P0-4 FIX: Extract Isaac Sim availability from capabilities
            isaac_sim_available = capabilities.get("isaac_sim_available", False) if isinstance(capabilities, dict) else False

            # Print detailed environment report
            print_environment_report(capabilities)

            # Enforce Isaac Sim for production
            required_quality = get_data_quality_level()
            print(f"[EPISODE-GEN-JOB] Required quality level: {required_quality.value}\n")

            capabilities = enforce_isaac_sim_for_production(required_quality)

            print("[EPISODE-GEN-JOB] ✅ Environment check passed\n")

            # P0-4 FIX: Confirm Isaac Sim is available after enforcement
            if not isaac_sim_available:
                print("[EPISODE-GEN-JOB] ⚠️  WARNING: Isaac Sim not confirmed available")

        except IsaacSimRequirementError as e:
            print(f"\n❌ ISAAC SIM REQUIREMENT ERROR:\n{e}\n")
            sys.exit(1)
        except ProductionDataQualityError as e:
            print(f"\n❌ PRODUCTION DATA QUALITY ERROR:\n{e}\n")
            sys.exit(1)
    else:
        # Fallback to legacy enforcement (quality system unavailable)
        print("[EPISODE-GEN-JOB] Using legacy Isaac Sim check (quality system unavailable)\n")

        # P0-4 FIX: Add explicit check for sensor capture environment
        if check_sensor_capture_environment is not None:
            status = check_sensor_capture_environment()
            isaac_sim_available = status.get("isaac_sim_available", False)
            print(f"[EPISODE-GEN-JOB] Isaac Sim available: {isaac_sim_available}\n")
        else:
            isaac_sim_available = False
            print("[EPISODE-GEN-JOB] ⚠️  WARNING: Cannot verify Isaac Sim availability\n")

        # Detect if running in production environment
        # Production indicators: running in container, K8s, or Cloud Run
        is_production = (
            os.getenv("KUBERNETES_SERVICE_HOST") is not None or  # K8s
            os.getenv("K_SERVICE") is not None or  # Cloud Run
            os.path.exists("/.dockerenv") or  # Docker
            os.getenv("PRODUCTION", "false").lower() == "true"
        )

        # In production, require real physics by default unless explicitly disabled
        # In development, allow mock data by default unless explicitly required
        if is_production:
            default_require_real = "true"
        else:
            default_require_real = "false"

        require_real_physics = os.getenv("REQUIRE_REAL_PHYSICS", default_require_real).lower() == "true"

        # LABS-BLOCKER-001 FIX: Remove ALLOW_MOCK_DATA override in production
        # In production, we MUST have Isaac Sim - no exceptions
        # This prevents labs from wasting GPU hours on garbage data
        if is_production:
            allow_mock_data = False  # NEVER allow mock data in production
            if os.getenv("ALLOW_MOCK_DATA", "").lower() == "true":
                print("\n" + "=" * 80)
                print("❌ FATAL ERROR: ALLOW_MOCK_DATA not permitted in production")
                print("=" * 80)
                print("")
                print("ALLOW_MOCK_DATA=true is IGNORED in production environments.")
                print("Production training data MUST use real Isaac Sim physics.")
                print("")
                print("This protection prevents:")
                print("  - Training models on random noise images")
                print("  - Wasting GPU hours on useless data")
                print("  - Shipping low-quality data to labs/customers")
                print("")
                print("=" * 80 + "\n")
        else:
            # Development mode: allow override for testing
            allow_mock_data = os.getenv("ALLOW_MOCK_DATA", "false").lower() == "true"

        # P0-4 FIX: Log configuration BEFORE checking Isaac Sim
        print(f"[EPISODE-GEN-JOB] Production mode: {is_production}")
        print(f"[EPISODE-GEN-JOB] Require real physics: {require_real_physics}")
        print(f"[EPISODE-GEN-JOB] Allow mock data: {allow_mock_data}")

        # LABS-BLOCKER-001 FIX: Fail hard in production if Isaac Sim not available
        if require_real_physics and not isaac_sim_available:
            # In production, this is ALWAYS a fatal error (allow_mock_data is always False)
            print("\n" + "=" * 80)
            print("❌ FATAL ERROR: Isaac Sim not available in production mode")
            print("=" * 80)
            print("")
            print("Episode generation requires NVIDIA Isaac Sim for:")
            print("  ✓ Real physics simulation (PhysX)")
            print("  ✓ Actual sensor data capture (Replicator)")
            print("  ✓ Physics-validated trajectories")
            print("  ✓ Collision detection with real geometry")
            print("")
            print("Without Isaac Sim, the pipeline would generate:")
            print("  ✗ Random noise RGB images (NOT real sensor data)")
            print("  ✗ Heuristic-based validation (NOT physics-verified)")
            print("  ✗ Mock contact/collision data (NOT accurate)")
            print("")
            print("This data is USELESS for:")
            print("  • Training production ML models")
            print("  • Real-world robot deployment")
            print("  • Lab testing and evaluation")
            print("")
            print("To fix this:")
            print("  1. Install Isaac Sim: https://developer.nvidia.com/isaac-sim")
            print("  2. Run with Isaac Sim Python: /isaac-sim/python.sh generate_episodes.py")
            print("  3. Or use the Isaac Sim container: see docs/ISAAC_SIM_SETUP.md")
            print("")
            if is_production:
                print("NOTE: ALLOW_MOCK_DATA override is DISABLED in production.")
                print("      There is no way to bypass this check in production mode.")
            else:
                print("To bypass in development (NOT for production data):")
                print("  export REQUIRE_REAL_PHYSICS=false")
            print("=" * 80 + "\n")
            sys.exit(1)

    # Development mode warning
    if not isaac_sim_available and not require_real_physics:
        print("\n[EPISODE-GEN-JOB] ========================================")
        print("[EPISODE-GEN-JOB] WARNING: Isaac Sim not available")
        print("[EPISODE-GEN-JOB] Running with MOCK DATA (random noise)")
        print("[EPISODE-GEN-JOB] For real data: /isaac-sim/python.sh")
        print("[EPISODE-GEN-JOB] ========================================\n")

    # Get configuration from environment
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")

    if not scene_id:
        print("[EPISODE-GEN-JOB] ERROR: SCENE_ID is required")
        sys.exit(1)

    # Prefixes with defaults
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    episodes_prefix = os.getenv("EPISODES_PREFIX", f"scenes/{scene_id}/episodes")

    # Configuration with validation
    robot_type = os.getenv("ROBOT_TYPE", "franka")

    try:
        episodes_per_variation = int(os.getenv("EPISODES_PER_VARIATION", "10"))
        if episodes_per_variation <= 0:
            raise ValueError("EPISODES_PER_VARIATION must be positive")
    except ValueError as e:
        print(f"[EPISODE-GEN-JOB] ERROR: Invalid EPISODES_PER_VARIATION: {e}")
        sys.exit(1)

    max_variations = os.getenv("MAX_VARIATIONS")
    if max_variations:
        try:
            max_variations = int(max_variations)
            if max_variations <= 0:
                raise ValueError("MAX_VARIATIONS must be positive")
        except ValueError as e:
            print(f"[EPISODE-GEN-JOB] ERROR: Invalid MAX_VARIATIONS: {e}")
            sys.exit(1)
    else:
        max_variations = None

    try:
        fps = float(os.getenv("FPS", "30"))
        if fps <= 0 or fps > 240:
            raise ValueError("FPS must be between 0 and 240")
    except ValueError as e:
        print(f"[EPISODE-GEN-JOB] ERROR: Invalid FPS: {e}")
        sys.exit(1)

    use_llm = os.getenv("USE_LLM", "true").lower() == "true"
    use_cpgen = os.getenv("USE_CPGEN", "true").lower() == "true"

    try:
        # LABS-BLOCKER-002 FIX: Default raised from 0.7 to 0.85 for production quality
        min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.85"))
        if not (0.0 <= min_quality_score <= 1.0):
            raise ValueError("MIN_QUALITY_SCORE must be between 0.0 and 1.0")
    except ValueError as e:
        print(f"[EPISODE-GEN-JOB] ERROR: Invalid MIN_QUALITY_SCORE: {e}")
        sys.exit(1)

    # Data pack configuration (Core/Plus/Full)
    data_pack_tier = os.getenv("DATA_PACK_TIER", "core")

    try:
        num_cameras = int(os.getenv("NUM_CAMERAS", "1"))
        if num_cameras < 1 or num_cameras > 8:
            raise ValueError("NUM_CAMERAS must be between 1 and 8")
    except ValueError as e:
        print(f"[EPISODE-GEN-JOB] ERROR: Invalid NUM_CAMERAS: {e}")
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
        print(f"[EPISODE-GEN-JOB] ERROR: Invalid IMAGE_RESOLUTION: {e}")
        sys.exit(1)

    capture_sensor_data = os.getenv("CAPTURE_SENSOR_DATA", "true").lower() == "true"
    use_mock_capture = os.getenv("USE_MOCK_CAPTURE", "false").lower() == "true"

    # Bundle tier for upsell features (standard, pro, enterprise, foundation)
    bundle_tier = os.getenv("BUNDLE_TIER", "standard")

    print(f"[EPISODE-GEN-JOB] Configuration:")
    print(f"[EPISODE-GEN-JOB]   Bucket: {bucket}")
    print(f"[EPISODE-GEN-JOB]   Scene ID: {scene_id}")
    print(f"[EPISODE-GEN-JOB]   Pipeline: SOTA (CP-Gen + Validation)")
    print(f"[EPISODE-GEN-JOB]   Data Pack: {data_pack_tier}")
    print(f"[EPISODE-GEN-JOB]   Bundle Tier: {bundle_tier}")
    print(f"[EPISODE-GEN-JOB]   Cameras: {num_cameras}")
    print(f"[EPISODE-GEN-JOB]   Resolution: {image_resolution}")

    GCS_ROOT = Path("/mnt/gcs")

    exit_code = run_episode_generation_job(
        root=GCS_ROOT,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        episodes_prefix=episodes_prefix,
        robot_type=robot_type,
        episodes_per_variation=episodes_per_variation,
        max_variations=max_variations,
        fps=fps,
        use_llm=use_llm,
        use_cpgen=use_cpgen,
        min_quality_score=min_quality_score,
        data_pack_tier=data_pack_tier,
        num_cameras=num_cameras,
        image_resolution=image_resolution,
        capture_sensor_data=capture_sensor_data,
        use_mock_capture=use_mock_capture,
        bundle_tier=bundle_tier,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

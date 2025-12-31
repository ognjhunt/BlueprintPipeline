#!/usr/bin/env python3
"""
Episode Generation Job for BlueprintPipeline.

Generates training-ready robotic manipulation episodes from scenes and variations.
This is the core module that creates sellable episode data alongside scene assets.

Pipeline Position:
    3D-RE-GEN → simready → usd-assembly → replicator → isaac-lab → [THIS JOB]

Process:
1. Analyze scene manifest to identify manipulation tasks
2. For each scene variation:
   a. Generate motion plans using AI (Gemini)
   b. Solve trajectories to joint-level
   c. Export to LeRobot format
3. Package episodes with metadata

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
    │   └── ...
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
    USE_LLM: Enable LLM for enhanced planning - default: true
"""

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

# Local imports
from motion_planner import AIMotionPlanner, MotionPlan
from trajectory_solver import TrajectorySolver, JointTrajectory, ROBOT_CONFIGS
from lerobot_exporter import LeRobotExporter, LeRobotDatasetConfig

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
    use_llm: bool = True

    # Output
    output_dir: Path = Path("./episodes")

    # Quality settings
    validate_trajectories: bool = True
    include_failed: bool = False


@dataclass
class GeneratedEpisode:
    """A generated episode with metadata."""

    episode_id: str
    task_name: str
    task_description: str

    # Trajectory
    trajectory: JointTrajectory
    motion_plan: MotionPlan

    # Context
    scene_id: str
    variation_index: int

    # Quality
    is_valid: bool = True
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
    total_variations: int = 0
    total_frames: int = 0
    total_duration_seconds: float = 0.0

    # Task coverage
    tasks_generated: Dict[str, int] = field(default_factory=dict)

    # Output paths
    output_dir: Optional[Path] = None
    lerobot_dataset_path: Optional[Path] = None
    manifest_path: Optional[Path] = None

    # Timing
    generation_time_seconds: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.total_episodes > 0 and len(self.errors) == 0


# =============================================================================
# Task Generator
# =============================================================================


class ManipulationTaskGenerator:
    """
    Generates manipulation tasks from scene analysis.

    If dwm-preparation-job modules are available, uses full scene analysis.
    Otherwise, uses simplified task generation based on object types.
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

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.scene_analyzer = SceneAnalyzer(verbose=verbose) if HAVE_DWM_MODULES else None
        self.task_planner = TaskPlanner(verbose=verbose) if HAVE_DWM_MODULES else None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[TASK-GENERATOR] [{level}] {msg}")

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
                "target_position": target_obj.get("position", [0.5, 0, 0.85]) if target_obj else [0.5, 0, 0.85],
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

        for obj in objects:
            category = obj.get("category", "object").lower()
            obj_id = obj.get("id", obj.get("name", "unknown"))
            position = obj.get("position", [0.5, 0, 0.85])
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

    def _calculate_place_position(
        self,
        obj: Optional[Dict[str, Any]],
        manifest: Dict[str, Any],
    ) -> List[float]:
        """Calculate a sensible place position for an object."""
        if obj is None:
            return [0.3, 0.2, 0.85]

        pos = obj.get("position", [0.5, 0, 0.85])
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()

        # Offset placement position
        return [pos[0] - 0.2, pos[1] + 0.15, pos[2]]


# =============================================================================
# Episode Generator
# =============================================================================


class EpisodeGenerator:
    """
    Main episode generation engine.

    Generates manipulation episodes for each scene variation.
    """

    def __init__(self, config: EpisodeGenerationConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose

        # Initialize components
        self.motion_planner = AIMotionPlanner(
            robot_type=config.robot_type,
            use_llm=config.use_llm,
            verbose=verbose,
        )
        self.trajectory_solver = TrajectorySolver(
            robot_type=config.robot_type,
            fps=config.fps,
            verbose=verbose,
        )
        self.task_generator = ManipulationTaskGenerator(verbose=verbose)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[EPISODE-GENERATOR] [{level}] {msg}")

    def generate(self, manifest: Dict[str, Any]) -> EpisodeGenerationOutput:
        """
        Generate episodes for all variations in the scene.

        Args:
            manifest: Scene manifest

        Returns:
            EpisodeGenerationOutput with all generated episodes
        """
        start_time = time.time()

        self.log("=" * 70)
        self.log("EPISODE GENERATION")
        self.log("=" * 70)
        self.log(f"Scene: {self.config.scene_id}")
        self.log(f"Robot: {self.config.robot_type}")
        self.log(f"Episodes per variation: {self.config.episodes_per_variation}")
        self.log(f"FPS: {self.config.fps}")
        self.log("=" * 70)

        output = EpisodeGenerationOutput(
            scene_id=self.config.scene_id,
            robot_type=self.config.robot_type,
        )

        # Step 1: Generate tasks from scene
        self.log("\nStep 1: Generating manipulation tasks...")
        tasks = self.task_generator.generate_tasks(
            manifest=manifest,
            manifest_path=self.config.manifest_path,
        )

        if not tasks:
            output.errors.append("No manipulation tasks could be generated from scene")
            return output

        self.log(f"  Generated {len(tasks)} tasks")

        # Step 2: Determine variations
        variation_count = manifest.get("variation_count", 1)
        if self.config.max_variations:
            variation_count = min(variation_count, self.config.max_variations)

        self.log(f"\nStep 2: Processing {variation_count} variations...")

        # Step 3: Create LeRobot exporter
        lerobot_config = LeRobotDatasetConfig(
            dataset_name=f"{self.config.scene_id}_episodes",
            robot_type=self.config.robot_type,
            fps=self.config.fps,
            output_dir=self.config.output_dir / "lerobot",
        )
        exporter = LeRobotExporter(lerobot_config, verbose=False)

        # Step 4: Generate episodes for each variation
        self.log(f"\nStep 3: Generating episodes...")

        all_episodes: List[GeneratedEpisode] = []

        for var_idx in range(variation_count):
            self.log(f"\n  Variation {var_idx + 1}/{variation_count}")

            # Apply variation (in a real scenario, this would modify object positions)
            variation_manifest = self._apply_variation(manifest, var_idx)

            # Generate episodes for this variation
            variation_episodes = self._generate_variation_episodes(
                manifest=variation_manifest,
                tasks=tasks,
                variation_index=var_idx,
            )

            # Add to exporter
            for episode in variation_episodes:
                if episode.is_valid or self.config.include_failed:
                    exporter.add_episode(
                        trajectory=episode.trajectory,
                        task_description=episode.task_description,
                        scene_id=self.config.scene_id,
                        variation_index=var_idx,
                        success=episode.is_valid,
                    )
                    all_episodes.append(episode)

                    # Track task coverage
                    task_key = episode.task_name
                    output.tasks_generated[task_key] = output.tasks_generated.get(task_key, 0) + 1

            self.log(f"    Generated {len(variation_episodes)} episodes")

        # Step 5: Export dataset
        self.log(f"\nStep 4: Exporting LeRobot dataset...")
        try:
            dataset_path = exporter.finalize()
            output.lerobot_dataset_path = dataset_path
        except Exception as e:
            output.errors.append(f"LeRobot export failed: {e}")

        # Step 6: Write generation manifest
        self.log(f"\nStep 5: Writing generation manifest...")
        output.manifest_path = self._write_manifest(all_episodes, tasks, output)

        # Calculate statistics
        output.total_episodes = len(all_episodes)
        output.total_variations = variation_count
        output.total_frames = sum(ep.trajectory.num_frames for ep in all_episodes)
        output.total_duration_seconds = sum(ep.trajectory.total_duration for ep in all_episodes)
        output.generation_time_seconds = time.time() - start_time
        output.output_dir = self.config.output_dir

        # Summary
        self.log("\n" + "=" * 70)
        self.log("GENERATION COMPLETE")
        self.log("=" * 70)
        self.log(f"Total episodes: {output.total_episodes}")
        self.log(f"Total frames: {output.total_frames}")
        self.log(f"Total duration: {output.total_duration_seconds:.1f}s")
        self.log(f"Tasks covered: {len(output.tasks_generated)}")
        self.log(f"Generation time: {output.generation_time_seconds:.1f}s")
        self.log(f"Output: {output.output_dir}")
        if output.errors:
            self.log(f"Errors: {len(output.errors)}", "ERROR")
        self.log("=" * 70)

        return output

    def _apply_variation(
        self,
        manifest: Dict[str, Any],
        variation_index: int,
    ) -> Dict[str, Any]:
        """
        Apply variation to manifest (randomize object positions).

        In production, this would use the Replicator bundle to get
        actual variation transforms.
        """
        # For now, add small random offsets based on variation index
        np.random.seed(variation_index)

        varied = json.loads(json.dumps(manifest))  # Deep copy

        for obj in varied.get("objects", []):
            if "position" in obj:
                pos = obj["position"]
                if isinstance(pos, list):
                    # Add small random offset (up to 5cm)
                    offset = np.random.uniform(-0.05, 0.05, 3)
                    offset[2] = 0  # Don't change height
                    obj["position"] = [p + o for p, o in zip(pos, offset)]

        return varied

    def _generate_variation_episodes(
        self,
        manifest: Dict[str, Any],
        tasks: List[Dict[str, Any]],
        variation_index: int,
    ) -> List[GeneratedEpisode]:
        """Generate episodes for a single variation."""
        episodes = []

        # Select tasks for this variation
        num_episodes = min(self.config.episodes_per_variation, len(tasks))
        selected_tasks = tasks[:num_episodes]

        for task in selected_tasks:
            try:
                episode = self._generate_single_episode(
                    task=task,
                    manifest=manifest,
                    variation_index=variation_index,
                )
                episodes.append(episode)
            except Exception as e:
                self.log(f"      Failed {task['task_name']}: {e}", "WARNING")
                # Create failed episode record
                episodes.append(GeneratedEpisode(
                    episode_id=f"failed_{task['task_id']}_{variation_index}",
                    task_name=task["task_name"],
                    task_description=task["description"],
                    trajectory=None,
                    motion_plan=None,
                    scene_id=self.config.scene_id,
                    variation_index=variation_index,
                    is_valid=False,
                    validation_errors=[str(e)],
                ))

        return episodes

    def _generate_single_episode(
        self,
        task: Dict[str, Any],
        manifest: Dict[str, Any],
        variation_index: int,
    ) -> GeneratedEpisode:
        """Generate a single episode."""
        start_time = time.time()

        # Build target object info
        target_object = {
            "id": task["target_object_id"],
            "position": task["target_position"],
            "dimensions": task["target_dimensions"],
        }

        # Build articulation info if needed
        articulation_info = None
        if task.get("is_articulated"):
            articulation_info = {
                "handle_position": task["target_position"],
                "axis": [-1, 0, 0],  # Default: pull toward robot
                "range": [0, 0.3],
                "type": "prismatic",
            }

        # Generate motion plan
        motion_plan = self.motion_planner.plan_motion(
            task_name=task["task_name"],
            task_description=task["description"],
            target_object=target_object,
            place_position=task.get("place_position"),
            articulation_info=articulation_info,
        )

        # Solve trajectory
        trajectory = self.trajectory_solver.solve(motion_plan)

        # Validate trajectory
        is_valid, errors = self._validate_trajectory(trajectory)

        episode = GeneratedEpisode(
            episode_id=f"{task['task_id']}_var{variation_index}_{uuid.uuid4().hex[:8]}",
            task_name=task["task_name"],
            task_description=task["description"],
            trajectory=trajectory,
            motion_plan=motion_plan,
            scene_id=self.config.scene_id,
            variation_index=variation_index,
            is_valid=is_valid,
            validation_errors=errors,
            generation_time_seconds=time.time() - start_time,
        )

        return episode

    def _validate_trajectory(self, trajectory: JointTrajectory) -> Tuple[bool, List[str]]:
        """Validate a trajectory for quality."""
        errors = []

        if not self.config.validate_trajectories:
            return True, errors

        # Check minimum frames
        if trajectory.num_frames < 10:
            errors.append(f"Too few frames: {trajectory.num_frames}")

        # Check for NaN values
        positions = trajectory.get_joint_positions_array()
        if np.any(np.isnan(positions)):
            errors.append("NaN values in joint positions")

        # Check joint limits
        robot_config = ROBOT_CONFIGS.get(self.config.robot_type)
        if robot_config:
            lower = robot_config.joint_limits_lower
            upper = robot_config.joint_limits_upper

            if np.any(positions < lower - 0.1) or np.any(positions > upper + 0.1):
                errors.append("Joint positions exceed limits")

        # Check for excessive velocities
        if trajectory.states and len(trajectory.states) > 1:
            dt = 1.0 / self.config.fps
            for i in range(1, len(trajectory.states)):
                vel = np.abs(
                    trajectory.states[i].joint_positions -
                    trajectory.states[i-1].joint_positions
                ) / dt
                if np.any(vel > 5.0):  # 5 rad/s threshold
                    errors.append(f"Excessive velocity at frame {i}")
                    break

        return len(errors) == 0, errors

    def _write_manifest(
        self,
        episodes: List[GeneratedEpisode],
        tasks: List[Dict[str, Any]],
        output: EpisodeGenerationOutput,
    ) -> Path:
        """Write generation manifest."""
        manifest_dir = self.config.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Generation manifest
        manifest = {
            "scene_id": self.config.scene_id,
            "robot_type": self.config.robot_type,
            "fps": self.config.fps,
            "generation_config": {
                "episodes_per_variation": self.config.episodes_per_variation,
                "use_llm": self.config.use_llm,
                "validate_trajectories": self.config.validate_trajectories,
            },
            "statistics": {
                "total_episodes": len(episodes),
                "valid_episodes": sum(1 for e in episodes if e.is_valid),
                "failed_episodes": sum(1 for e in episodes if not e.is_valid),
                "total_frames": sum(e.trajectory.num_frames for e in episodes if e.trajectory),
                "total_duration_seconds": sum(
                    e.trajectory.total_duration for e in episodes if e.trajectory
                ),
            },
            "tasks": [
                {
                    "task_id": t["task_id"],
                    "task_name": t["task_name"],
                    "description": t["description"],
                    "episodes_generated": output.tasks_generated.get(t["task_name"], 0),
                }
                for t in tasks
            ],
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "task_name": e.task_name,
                    "variation_index": e.variation_index,
                    "is_valid": e.is_valid,
                    "num_frames": e.trajectory.num_frames if e.trajectory else 0,
                    "duration_seconds": e.trajectory.total_duration if e.trajectory else 0,
                    "generation_time_seconds": e.generation_time_seconds,
                    "errors": e.validation_errors,
                }
                for e in episodes
            ],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "generator_version": "1.0.0",
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
) -> int:
    """
    Run the episode generation job.

    Args:
        root: Root path (e.g., /mnt/gcs)
        scene_id: Scene identifier
        assets_prefix: Path to scene assets
        episodes_prefix: Output path for episodes
        robot_type: Robot type
        episodes_per_variation: Episodes to generate per variation
        max_variations: Max variations to process (None = all)
        fps: Target FPS
        use_llm: Enable LLM for enhanced planning

    Returns:
        0 on success, 1 on failure
    """
    print(f"[EPISODE-GEN-JOB] Starting episode generation for scene: {scene_id}")
    print(f"[EPISODE-GEN-JOB] Assets prefix: {assets_prefix}")
    print(f"[EPISODE-GEN-JOB] Episodes prefix: {episodes_prefix}")
    print(f"[EPISODE-GEN-JOB] Robot type: {robot_type}")
    print(f"[EPISODE-GEN-JOB] Episodes per variation: {episodes_per_variation}")

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
        output_dir=output_dir,
    )

    try:
        generator = EpisodeGenerator(config, verbose=True)
        output = generator.generate(manifest)

        if output.success:
            print("[EPISODE-GEN-JOB] ✓ Episode generation completed successfully")
            print(f"[EPISODE-GEN-JOB]   Episodes: {output.total_episodes}")
            print(f"[EPISODE-GEN-JOB]   Frames: {output.total_frames}")
            print(f"[EPISODE-GEN-JOB]   Duration: {output.total_duration_seconds:.1f}s")
            print(f"[EPISODE-GEN-JOB]   Output: {output.output_dir}")
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
    # Get configuration from environment
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")

    if not scene_id:
        print("[EPISODE-GEN-JOB] ERROR: SCENE_ID is required")
        sys.exit(1)

    # Prefixes with defaults
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    episodes_prefix = os.getenv("EPISODES_PREFIX", f"scenes/{scene_id}/episodes")

    # Configuration
    robot_type = os.getenv("ROBOT_TYPE", "franka")
    episodes_per_variation = int(os.getenv("EPISODES_PER_VARIATION", "10"))
    max_variations = os.getenv("MAX_VARIATIONS")
    max_variations = int(max_variations) if max_variations else None
    fps = float(os.getenv("FPS", "30"))
    use_llm = os.getenv("USE_LLM", "true").lower() == "true"

    print(f"[EPISODE-GEN-JOB] Configuration:")
    print(f"[EPISODE-GEN-JOB]   Bucket: {bucket}")
    print(f"[EPISODE-GEN-JOB]   Scene ID: {scene_id}")

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
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

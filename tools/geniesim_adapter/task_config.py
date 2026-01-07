"""
Task Configuration Generator for Genie Sim 3.0.

Generates task hints and configuration for Genie Sim's LLM-based task generation,
based on the scene content and object affordances.

Genie Sim uses LLMs to automatically generate task instructions from scenes.
This module provides hints to guide that generation based on BlueprintPipeline's
rich object metadata (affordances, articulation, placement regions).

References:
    - Genie Sim 3.0 Paper: https://arxiv.org/html/2601.02078v1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RobotConfig:
    """Robot configuration for Genie Sim."""

    robot_type: str = "franka"  # franka, g2, ur10, custom
    urdf_path: Optional[str] = None
    base_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    workspace_bounds: List[List[float]] = field(
        default_factory=lambda: [[-0.5, -0.5, 0.0], [1.0, 1.0, 1.5]]
    )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.robot_type,
            "base_position": self.base_position,
            "workspace_bounds": self.workspace_bounds,
        }
        if self.urdf_path:
            result["urdf_path"] = self.urdf_path
        return result


@dataclass
class SuggestedTask:
    """A suggested task for Genie Sim to generate data for."""

    task_type: str  # pick_place, open_close, pour, stack, etc.
    target_object: str  # Object ID
    goal_region: Optional[str] = None  # Placement region or goal object
    difficulty: str = "medium"  # easy, medium, hard
    priority: int = 1  # Higher = more important
    description_hint: Optional[str] = None  # Hint for LLM task description

    # Constraints from BlueprintPipeline affordances
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "task_type": self.task_type,
            "target_object": self.target_object,
            "difficulty": self.difficulty,
            "priority": self.priority,
        }
        if self.goal_region:
            result["goal_region"] = self.goal_region
        if self.description_hint:
            result["description_hint"] = self.description_hint
        if self.constraints:
            result["constraints"] = self.constraints
        return result


@dataclass
class GenieSimTaskConfig:
    """Complete task configuration for Genie Sim."""

    scene_id: str
    environment_type: str
    suggested_tasks: List[SuggestedTask]
    robot_config: RobotConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "environment_type": self.environment_type,
            "suggested_tasks": [t.to_dict() for t in self.suggested_tasks],
            "robot_config": self.robot_config.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save task config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


# =============================================================================
# Task Type Mapping
# =============================================================================

# Maps (sim_role, affordance) -> task_type
TASK_TYPE_MAPPING = {
    # Pick and place tasks
    ("manipulable_object", "Graspable"): "pick_place",
    ("manipulable_object", "Stackable"): "stack",
    ("manipulable_object", "Pourable"): "pour",
    ("manipulable_object", "Insertable"): "insert",
    ("manipulable_object", "Hangable"): "hang",

    # Articulated tasks
    ("articulated_furniture", "Openable"): "open_close",
    ("articulated_furniture", "Slidable"): "slide",
    ("articulated_appliance", "Openable"): "open_close",
    ("articulated_appliance", "Turnable"): "turn",
    ("articulated_appliance", "Pressable"): "press",

    # Container tasks
    ("manipulable_object", "Fillable"): "fill",
    ("manipulable_object", "Containable"): "organize",
}

# Default task types by sim_role
DEFAULT_TASK_TYPE = {
    "manipulable_object": "pick_place",
    "articulated_furniture": "open_close",
    "articulated_appliance": "interact",
    "interactive": "interact",
    "clutter": "pick_place",
}

# Difficulty estimation based on task type and properties
DIFFICULTY_FACTORS = {
    # Task type base difficulty
    "pick_place": 0.3,
    "stack": 0.5,
    "pour": 0.7,
    "insert": 0.6,
    "hang": 0.5,
    "open_close": 0.4,
    "slide": 0.3,
    "turn": 0.4,
    "press": 0.2,
    "fill": 0.6,
    "organize": 0.5,
    "interact": 0.4,
}


# =============================================================================
# Task Config Generator
# =============================================================================


class TaskConfigGenerator:
    """
    Generates Genie Sim task configuration from BlueprintPipeline scene.

    Usage:
        generator = TaskConfigGenerator()
        config = generator.generate(manifest_dict, robot_type="franka")
        config.save(Path("output/task_config.json"))
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[TASK-CONFIG-GENERATOR] [{level}] {msg}")

    def generate(
        self,
        manifest: Dict[str, Any],
        robot_type: str = "franka",
        urdf_path: Optional[str] = None,
        max_tasks: int = 50,
    ) -> GenieSimTaskConfig:
        """
        Generate Genie Sim task configuration from manifest.

        Args:
            manifest: BlueprintPipeline scene_manifest.json as dict
            robot_type: Robot type (franka, g2, ur10)
            urdf_path: Optional custom URDF path
            max_tasks: Maximum number of suggested tasks

        Returns:
            GenieSimTaskConfig ready for Genie Sim
        """
        self.log("Generating Genie Sim task configuration")

        scene_id = manifest.get("scene_id", "unknown")
        scene_config = manifest.get("scene", {})
        objects = manifest.get("objects", [])

        environment_type = scene_config.get("environment_type", "general")
        self.log(f"Scene: {scene_id}, environment: {environment_type}")

        # Generate suggested tasks
        tasks = self._generate_tasks(objects, environment_type)
        self.log(f"Generated {len(tasks)} task suggestions")

        # Prioritize and limit tasks
        tasks = self._prioritize_tasks(tasks, max_tasks)

        # Create robot config
        robot_config = self._create_robot_config(
            robot_type=robot_type,
            urdf_path=urdf_path,
            scene_config=scene_config,
            objects=objects,
        )

        # Build metadata
        metadata = {
            "total_objects": len(objects),
            "manipulable_objects": sum(
                1 for o in objects if o.get("sim_role") == "manipulable_object"
            ),
            "articulated_objects": sum(
                1 for o in objects
                if o.get("sim_role") in ["articulated_furniture", "articulated_appliance"]
            ),
            "task_types": list(set(t.task_type for t in tasks)),
        }

        return GenieSimTaskConfig(
            scene_id=scene_id,
            environment_type=environment_type,
            suggested_tasks=tasks,
            robot_config=robot_config,
            metadata=metadata,
        )

    def _generate_tasks(
        self,
        objects: List[Dict[str, Any]],
        environment_type: str,
    ) -> List[SuggestedTask]:
        """Generate task suggestions from objects."""
        tasks = []

        # Find placement regions (static surfaces)
        placement_regions = self._find_placement_regions(objects)

        for obj in objects:
            obj_id = str(obj.get("id", ""))
            sim_role = obj.get("sim_role", "unknown")

            # Skip non-interactive objects
            if sim_role in ["background", "scene_shell", "static"]:
                continue

            # Get affordances
            semantics = obj.get("semantics", {})
            affordances = semantics.get("affordances", [])

            # Generate tasks based on affordances
            obj_tasks = self._generate_tasks_for_object(
                obj=obj,
                affordances=affordances,
                placement_regions=placement_regions,
                environment_type=environment_type,
            )
            tasks.extend(obj_tasks)

        return tasks

    def _generate_tasks_for_object(
        self,
        obj: Dict[str, Any],
        affordances: List,
        placement_regions: List[str],
        environment_type: str,
    ) -> List[SuggestedTask]:
        """Generate tasks for a single object."""
        tasks = []
        obj_id = str(obj.get("id", ""))
        sim_role = obj.get("sim_role", "unknown")
        category = obj.get("category", "object")

        # Process each affordance
        for aff in affordances:
            if isinstance(aff, str):
                aff_name = aff
                aff_params = {}
            elif isinstance(aff, dict):
                aff_name = aff.get("type", "")
                aff_params = aff
            else:
                continue

            # Get task type
            key = (sim_role, aff_name)
            task_type = TASK_TYPE_MAPPING.get(key)

            if not task_type:
                continue

            # Build constraints from affordance params
            constraints = self._build_constraints(aff_name, aff_params, obj)

            # Determine goal region
            goal_region = None
            if task_type in ["pick_place", "stack", "insert"]:
                # Use placement region from object or find nearby
                goal_region = obj.get("placement_region")
                if not goal_region and placement_regions:
                    goal_region = placement_regions[0]

            # Estimate difficulty
            difficulty = self._estimate_difficulty(
                task_type=task_type,
                obj=obj,
                constraints=constraints,
            )

            # Generate description hint
            description_hint = self._generate_description_hint(
                task_type=task_type,
                obj=obj,
                aff_name=aff_name,
                environment_type=environment_type,
            )

            task = SuggestedTask(
                task_type=task_type,
                target_object=obj_id,
                goal_region=goal_region,
                difficulty=difficulty,
                priority=self._calculate_priority(task_type, obj),
                description_hint=description_hint,
                constraints=constraints,
            )
            tasks.append(task)

        # If no affordance-based tasks, add default task
        if not tasks and sim_role in DEFAULT_TASK_TYPE:
            default_type = DEFAULT_TASK_TYPE[sim_role]
            tasks.append(SuggestedTask(
                task_type=default_type,
                target_object=obj_id,
                goal_region=placement_regions[0] if placement_regions else None,
                difficulty="medium",
                priority=1,
                description_hint=f"{default_type.replace('_', ' ')} the {category}",
            ))

        return tasks

    def _build_constraints(
        self,
        aff_name: str,
        aff_params: Dict[str, Any],
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build task constraints from affordance parameters."""
        constraints = {}

        # Articulation constraints
        if aff_name == "Openable":
            articulation = obj.get("articulation", {})
            if "limits" in articulation:
                limits = articulation["limits"]
                constraints["angle_range"] = [limits.get("lower", 0), limits.get("upper", 90)]
            if "open_angle" in aff_params:
                constraints["target_angle"] = aff_params["open_angle"]

        elif aff_name == "Slidable":
            if "open_distance" in aff_params:
                constraints["slide_distance"] = aff_params["open_distance"]

        # Grasp constraints
        elif aff_name == "Graspable":
            if "grasp_width_range" in aff_params:
                constraints["grasp_width"] = aff_params["grasp_width_range"]
            if "approach_direction" in aff_params:
                constraints["approach_direction"] = aff_params["approach_direction"]

        # Stack constraints
        elif aff_name == "Stackable":
            if "max_stack_height" in aff_params:
                constraints["max_stack"] = aff_params["max_stack_height"]
            if "stack_axis" in aff_params:
                constraints["stack_axis"] = aff_params["stack_axis"]

        # Pour constraints
        elif aff_name == "Pourable":
            if "pour_axis" in aff_params:
                constraints["pour_axis"] = aff_params["pour_axis"]
            if "capacity_liters" in aff_params:
                constraints["capacity"] = aff_params["capacity_liters"]

        return constraints

    def _find_placement_regions(
        self,
        objects: List[Dict[str, Any]],
    ) -> List[str]:
        """Find objects that can serve as placement regions."""
        regions = []

        for obj in objects:
            sim_role = obj.get("sim_role", "")
            semantics = obj.get("semantics", {})
            affordances = semantics.get("affordances", [])

            # Static objects with Supportable affordance
            if sim_role == "static":
                for aff in affordances:
                    aff_name = aff if isinstance(aff, str) else aff.get("type", "")
                    if aff_name in ["Supportable", "Placeable"]:
                        regions.append(str(obj.get("id", "")))
                        break

            # Also consider countertops, tables, shelves
            category = obj.get("category", "").lower()
            if category in ["countertop", "table", "shelf", "desk", "surface"]:
                obj_id = str(obj.get("id", ""))
                if obj_id not in regions:
                    regions.append(obj_id)

        return regions

    def _estimate_difficulty(
        self,
        task_type: str,
        obj: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Estimate task difficulty."""
        base_difficulty = DIFFICULTY_FACTORS.get(task_type, 0.5)

        # Adjust based on object size
        dimensions = obj.get("dimensions_est", {})
        if isinstance(dimensions, dict):
            size = max(
                dimensions.get("width", 0.1),
                dimensions.get("depth", 0.1),
                dimensions.get("height", 0.1),
            )
            if size < 0.05:
                base_difficulty += 0.2  # Small objects are harder
            elif size > 0.3:
                base_difficulty += 0.1  # Large objects are harder

        # Adjust based on constraints
        if constraints:
            base_difficulty += 0.1 * len(constraints)

        # Map to difficulty string
        if base_difficulty < 0.3:
            return "easy"
        elif base_difficulty < 0.6:
            return "medium"
        else:
            return "hard"

    def _calculate_priority(
        self,
        task_type: str,
        obj: Dict[str, Any],
    ) -> int:
        """Calculate task priority (higher = more important)."""
        priority = 1

        # Pick and place tasks are fundamental
        if task_type == "pick_place":
            priority += 2

        # Objects with explicit affordances are higher priority
        semantics = obj.get("semantics", {})
        if semantics.get("affordances"):
            priority += 1

        # Confidence-based priority
        for aff in semantics.get("affordances", []):
            if isinstance(aff, dict):
                confidence = aff.get("confidence", 0.5)
                if confidence > 0.8:
                    priority += 1

        return min(priority, 5)  # Cap at 5

    def _generate_description_hint(
        self,
        task_type: str,
        obj: Dict[str, Any],
        aff_name: str,
        environment_type: str,
    ) -> str:
        """Generate a description hint for the LLM task generator."""
        category = obj.get("category", "object")
        name = obj.get("name", obj.get("id", ""))

        hints = {
            "pick_place": f"Pick up the {category} ({name}) and place it on a nearby surface",
            "stack": f"Stack the {category} ({name}) on top of another similar object",
            "pour": f"Pour the contents of the {category} ({name}) into a container",
            "insert": f"Insert the {category} ({name}) into its designated location",
            "hang": f"Hang the {category} ({name}) on a hook or rack",
            "open_close": f"Open and then close the {category} ({name})",
            "slide": f"Slide the {category} ({name}) open",
            "turn": f"Turn the {category} ({name}) to adjust its setting",
            "press": f"Press the button on the {category} ({name})",
            "fill": f"Fill the {category} ({name}) with an appropriate substance",
            "organize": f"Organize items into the {category} ({name})",
            "interact": f"Interact with the {category} ({name})",
        }

        base_hint = hints.get(task_type, f"Manipulate the {category} ({name})")

        # Add environment context
        if environment_type == "kitchen":
            base_hint += " in a kitchen environment"
        elif environment_type == "warehouse":
            base_hint += " in a warehouse setting"
        elif environment_type == "office":
            base_hint += " in an office workspace"

        return base_hint

    def _prioritize_tasks(
        self,
        tasks: List[SuggestedTask],
        max_tasks: int,
    ) -> List[SuggestedTask]:
        """Sort and limit tasks by priority."""
        # Sort by priority (descending), then by task_type for consistency
        tasks.sort(key=lambda t: (-t.priority, t.task_type))

        # Ensure diversity: limit tasks per object
        seen_objects = {}
        diverse_tasks = []

        for task in tasks:
            obj_count = seen_objects.get(task.target_object, 0)
            if obj_count < 3:  # Max 3 tasks per object
                diverse_tasks.append(task)
                seen_objects[task.target_object] = obj_count + 1

            if len(diverse_tasks) >= max_tasks:
                break

        return diverse_tasks

    def _create_robot_config(
        self,
        robot_type: str,
        urdf_path: Optional[str],
        scene_config: Dict[str, Any],
        objects: List[Dict[str, Any]],
    ) -> RobotConfig:
        """Create robot configuration for the scene."""
        # Calculate workspace from scene bounds
        room = scene_config.get("room", {})
        bounds = room.get("bounds", {})

        if bounds:
            workspace_bounds = [
                [0.0, -bounds.get("width", 1.0) / 2, 0.0],
                [bounds.get("depth", 1.0), bounds.get("width", 1.0) / 2, bounds.get("height", 2.0)],
            ]
        else:
            # Default workspace
            workspace_bounds = [[-0.5, -0.5, 0.0], [1.0, 1.0, 1.5]]

        # Robot base position (center of workspace, on floor)
        base_position = [
            (workspace_bounds[0][0] + workspace_bounds[1][0]) / 2,
            (workspace_bounds[0][1] + workspace_bounds[1][1]) / 2,
            0.0,
        ]

        return RobotConfig(
            robot_type=robot_type,
            urdf_path=urdf_path,
            base_position=base_position,
            workspace_bounds=workspace_bounds,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_task_config(
    manifest_path: Path,
    output_path: Optional[Path] = None,
    robot_type: str = "franka",
    verbose: bool = True,
) -> GenieSimTaskConfig:
    """
    Convenience function to generate task config from manifest file.

    Args:
        manifest_path: Path to scene_manifest.json
        output_path: Optional path to save task_config.json
        robot_type: Robot type (franka, g2, ur10)
        verbose: Print progress

    Returns:
        GenieSimTaskConfig
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    generator = TaskConfigGenerator(verbose=verbose)
    config = generator.generate(manifest, robot_type=robot_type)

    if output_path:
        config.save(output_path)

    return config

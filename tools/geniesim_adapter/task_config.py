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

import importlib
import importlib.util
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from json import JSONDecodeError, loads as json_loads
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode
from tools.geniesim_adapter.config import (
    get_geniesim_task_confidence_threshold,
    get_geniesim_task_max_per_object,
    get_geniesim_task_size_large_threshold,
    get_geniesim_task_size_small_threshold,
)
from tools.geniesim_adapter.multi_robot_config import ROBOT_SPECS, RobotType

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODE_JOB_ROOT = REPO_ROOT / "episode-generation-job"


def _resolve_task_config_strictness(env: Optional[Mapping[str, str]] = None) -> bool:
    strict_override = parse_bool_env(
        (env or os.environ).get("BP_TASK_CONFIG_STRICT"),
        default=False,
    )
    return resolve_production_mode(env) or bool(strict_override)


def _resolve_scene_id_from_manifest(manifest: Dict[str, Any]) -> str:
    metadata = manifest.get("metadata")
    if isinstance(metadata, dict):
        scene_id = metadata.get("scene_id")
        if scene_id:
            return str(scene_id)
    return str(manifest.get("scene_id", "unknown"))


def _namespaced_asset_id(scene_id: Optional[str], obj_id: str) -> str:
    if not scene_id or scene_id == "unknown":
        return obj_id
    scene_prefix = f"{scene_id}_obj_"
    if obj_id.startswith(scene_prefix) or obj_id.startswith(f"{scene_id}:"):
        return obj_id
    return f"{scene_id}_obj_{obj_id}"


CANONICAL_REQUIRED_CAMERA_IDS: List[str] = ["left", "right", "wrist"]
_CAMERA_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "left": "left",
    "left_wrist": "left",
    "side": "left",
    "right": "right",
    "head": "right",
    "overhead": "right",
    "wrist": "wrist",
    "hand": "wrist",
    "eye_in_hand": "wrist",
}
_COMPOSITE_TASK_TYPES = {"organize", "fill", "stack"}


def _sanitize_token(value: Optional[str], *, fallback: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
    return token or fallback


def _normalize_required_camera_ids(camera_ids: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for camera_id in (camera_ids or []):
        alias = str(camera_id).strip().lower()
        if not alias:
            continue
        canonical = _CAMERA_ALIAS_TO_CANONICAL.get(alias, alias)
        if canonical in CANONICAL_REQUIRED_CAMERA_IDS and canonical not in normalized:
            normalized.append(canonical)
    for required in CANONICAL_REQUIRED_CAMERA_IDS:
        if required not in normalized:
            normalized.append(required)
    return normalized


def _derive_task_complexity(task_type: str, description_hint: Optional[str] = None) -> str:
    task_name = (task_type or "").strip().lower()
    hint = (description_hint or "").strip().lower()
    if task_name in _COMPOSITE_TASK_TYPES:
        return "composite"
    if " then " in hint or " and then " in hint:
        return "composite"
    return "atomic"


def _derive_curriculum_split(task_complexity: str) -> str:
    return "target" if str(task_complexity).strip().lower() == "composite" else "pretrain"

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RobotConfig:
    """Robot configuration for Genie Sim."""

    robot_type: str = "franka"  # franka, g2, ur10, custom
    robot_cfg_file: Optional[str] = None  # Optional explicit server robot config JSON
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
        if self.robot_cfg_file:
            result["robot_cfg_file"] = self.robot_cfg_file
        if self.urdf_path:
            result["urdf_path"] = self.urdf_path
        return result


@dataclass
class SuggestedTask:
    """A suggested task for Genie Sim to generate data for."""

    task_type: str  # pick_place, open_close, pour, stack, etc.
    target_object: str  # Object ID
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    task_complexity: Optional[str] = None  # atomic|composite
    curriculum_split: Optional[str] = None  # pretrain|target
    required_camera_ids: List[str] = field(default_factory=lambda: list(CANONICAL_REQUIRED_CAMERA_IDS))
    goal_region: Optional[str] = None  # Placement region or goal object
    difficulty: str = "medium"  # easy, medium, hard
    priority: int = 1  # Higher = more important
    description_hint: Optional[str] = None  # Hint for LLM task description
    requires_object_motion: Optional[bool] = None

    # Constraints from BlueprintPipeline affordances
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, scene_id: Optional[str] = None) -> Dict[str, Any]:
        target_object = _namespaced_asset_id(scene_id, self.target_object)
        requires_motion = (
            self.requires_object_motion
            if self.requires_object_motion is not None
            else _infer_requires_object_motion(self.task_type, self.description_hint)
        )
        complexity = (
            str(self.task_complexity).strip().lower()
            if self.task_complexity
            else _derive_task_complexity(self.task_type, self.description_hint)
        )
        if complexity not in {"atomic", "composite"}:
            complexity = _derive_task_complexity(self.task_type, self.description_hint)
        curriculum_split = (
            str(self.curriculum_split).strip().lower()
            if self.curriculum_split
            else _derive_curriculum_split(complexity)
        )
        if curriculum_split not in {"pretrain", "target"}:
            curriculum_split = _derive_curriculum_split(complexity)
        normalized_camera_ids = _normalize_required_camera_ids(self.required_camera_ids)
        task_name = self.task_name or self.description_hint or f"{self.task_type}_{self.target_object}"
        resolved_task_name = str(task_name).strip() or f"{self.task_type}_{self.target_object}"
        resolved_task_id = self.task_id or f"{_sanitize_token(scene_id, fallback='scene')}::{_sanitize_token(self.task_type, fallback='task')}::{_sanitize_token(self.target_object, fallback='object')}"
        result = {
            "task_id": resolved_task_id,
            "task_name": resolved_task_name,
            "task_type": self.task_type,
            "target_object": target_object,
            "task_complexity": complexity,
            "curriculum_split": curriculum_split,
            "required_camera_ids": normalized_camera_ids,
            "difficulty": self.difficulty,
            "priority": self.priority,
            "requires_object_motion": requires_motion,
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
            "schema_version": "3.0",
            "scene_id": self.scene_id,
            "environment_type": self.environment_type,
            "suggested_tasks": [t.to_dict(scene_id=self.scene_id) for t in self.suggested_tasks],
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

def _infer_requires_object_motion(
    task_type: str,
    description_hint: Optional[str] = None,
) -> bool:
    """Infer whether a task expects object motion (fallback when explicit flag missing)."""
    name = (task_type or "").lower()
    hint = (description_hint or "").lower()
    combined = f"{name} {hint}"

    non_motion_keywords = {
        "inspect", "observe", "scan", "look", "view", "perceive", "detect",
        "classify", "segment", "navigate", "reach", "point", "pose", "calibrate",
        "idle", "monitor",
    }
    motion_keywords = {
        "pick", "place", "grasp", "lift", "stack", "insert", "open", "close",
        "pour", "push", "pull", "transport", "handover", "rotate", "turn", "twist",
        "slide", "wipe", "clean", "sweep", "press", "toggle", "assemble", "disassemble",
        "hang", "fill", "organize",
    }

    if any(key in combined for key in non_motion_keywords):
        return False
    if any(key in combined for key in motion_keywords):
        return True

    # Known task types default to motion (manipulation tasks)
    if name in DIFFICULTY_FACTORS:
        return True

    # Conservative fallback: unknown tasks default to no motion required
    return False

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
        self.task_confidence_threshold = get_geniesim_task_confidence_threshold()
        self.task_size_small_threshold = get_geniesim_task_size_small_threshold()
        self.task_size_large_threshold = get_geniesim_task_size_large_threshold()
        self.task_max_per_object = get_geniesim_task_max_per_object()

    def log(self, msg: str, level: str = "INFO", extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.verbose:
            return
        level_name = level.upper()
        if level_name in {"WARN", "WARNING"}:
            logger.warning("[TASK-CONFIG-GENERATOR] %s", msg, extra=extra)
        elif level_name == "ERROR":
            logger.error("[TASK-CONFIG-GENERATOR] %s", msg, extra=extra)
        else:
            logger.info("[TASK-CONFIG-GENERATOR] %s", msg, extra=extra)

    def generate(
        self,
        manifest: Dict[str, Any],
        robot_type: str = "franka",
        urdf_path: Optional[str] = None,
        max_tasks: int = 50,
        strict_reachability: bool = False,
    ) -> GenieSimTaskConfig:
        """
        Generate Genie Sim task configuration from manifest.

        Args:
            manifest: BlueprintPipeline scene_manifest.json as dict
            robot_type: Robot type (franka, g2, ur10)
            urdf_path: Optional custom URDF path
            max_tasks: Maximum number of suggested tasks
            strict_reachability: Drop tasks if target positions are missing

        Returns:
            GenieSimTaskConfig ready for Genie Sim
        """
        self.log("Generating Genie Sim task configuration")

        scene_id = _resolve_scene_id_from_manifest(manifest)
        strict_validation = _resolve_task_config_strictness()
        if scene_id == "unknown" and strict_validation:
            raise RuntimeError(
                "Scene ID is required for task configuration in strict mode. "
                "Set scene_id in the manifest metadata or top-level scene_id field."
            )
        scene_config = manifest.get("scene", {})
        objects = manifest.get("objects", [])

        environment_type = scene_config.get("environment_type", "general")
        self.log(
            f"Scene: {scene_id}, environment: {environment_type}",
            extra={"scene_id": scene_id},
        )

        # Generate tasks and robot config
        use_llm = os.environ.get("GENIESIM_USE_LLM_TASK_PLANNING", "0") == "1"
        llm_robot_config = None
        if use_llm:
            try:
                tasks, llm_robot_config = self._plan_scene_with_llm(
                    manifest, environment_type, robot_type, max_tasks,
                )
                self.log(
                    f"LLM planned scene: {len(tasks)} tasks, "
                    f"bounds={llm_robot_config.workspace_bounds}, "
                    f"base={llm_robot_config.base_position}",
                    extra={"scene_id": scene_id},
                )
            except Exception as exc:
                self.log(
                    f"LLM scene planning failed, falling back to deterministic: {exc}",
                    level="WARNING",
                    extra={"scene_id": scene_id},
                )
                tasks = self._generate_tasks(objects, environment_type)
        else:
            tasks = self._generate_tasks(objects, environment_type)
        self.log(
            f"Generated {len(tasks)} task suggestions",
            extra={"scene_id": scene_id},
        )

        # Prioritize and limit tasks
        tasks = self._prioritize_tasks(tasks, max_tasks)

        # Create robot config (use LLM-derived if available)
        if llm_robot_config is not None:
            robot_config = llm_robot_config
            if urdf_path:
                robot_config.urdf_path = urdf_path
        else:
            robot_config = self._create_robot_config(
                robot_type=robot_type,
                urdf_path=urdf_path,
                scene_config=scene_config,
                objects=objects,
            )

        tasks, reachability_metadata = self._filter_tasks_by_reachability(
            tasks=tasks,
            manifest=manifest,
            robot_config=robot_config,
            strict_reachability=strict_reachability,
        )
        tasks, curobo_metadata = self._filter_tasks_by_curobo(
            tasks=tasks,
            manifest=manifest,
            robot_config=robot_config,
        )
        if not tasks and strict_validation:
            raise RuntimeError(
                "No suggested tasks available after filtering. "
                "Provide objects with task affordances or adjust filtering settings."
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
            **reachability_metadata,
            **curobo_metadata,
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

            affordance_confidence = None
            if isinstance(aff, dict) and "confidence" in aff:
                raw_confidence = aff.get("confidence")
                if isinstance(raw_confidence, (list, tuple)):
                    affordance_confidence = max(raw_confidence) if raw_confidence else None
                elif isinstance(raw_confidence, dict):
                    affordance_confidence = (
                        max(raw_confidence.values()) if raw_confidence else None
                    )
                else:
                    try:
                        affordance_confidence = float(raw_confidence)
                    except (TypeError, ValueError):
                        affordance_confidence = None

            if (
                affordance_confidence is not None
                and affordance_confidence < self.task_confidence_threshold
            ):
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
                requires_object_motion=_infer_requires_object_motion(task_type, description_hint),
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
                requires_object_motion=_infer_requires_object_motion(
                    default_type,
                    f"{default_type.replace('_', ' ')} the {category}",
                ),
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
            category = (obj.get("category") or "").lower()
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
            if size < self.task_size_small_threshold:
                base_difficulty += 0.2  # Small objects are harder
            elif size > self.task_size_large_threshold:
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
                if confidence > self.task_confidence_threshold:
                    priority += 1

        return min(priority, 5)  # Cap at 5

    # =========================================================================
    # LLM-based Scene Planning (Gemini 3.1 Pro Preview)
    # =========================================================================

    def _plan_scene_with_llm(
        self,
        manifest: Dict[str, Any],
        environment_type: str,
        robot_type: str,
        max_tasks: int = 50,
    ) -> Tuple[List[SuggestedTask], RobotConfig]:
        """
        Unified LLM scene planning: derives workspace bounds, robot placement,
        and manipulation tasks in a single call to Gemini 3.1 Pro Preview.

        Returns:
            Tuple of (tasks, robot_config) where the robot_config has
            LLM-derived workspace bounds and base position.
        """
        from tools.llm_client import create_llm_client, LLMProvider

        client = create_llm_client(provider=LLMProvider.GEMINI)
        prompt = self._build_scene_planning_prompt(manifest, environment_type, robot_type, max_tasks)

        self.log("Requesting unified scene planning from Gemini 3.1 Pro Preview")
        response = client.generate(
            prompt=prompt,
            json_output=True,
            temperature=0.7,
        )

        if response.error_message:
            raise RuntimeError(f"LLM scene planning failed: {response.error_message}")

        data = response.data
        if data is None:
            data = json.loads(response.text)

        # Parse robot config
        llm_robot = data.get("robot_config", {})
        workspace_bounds = llm_robot.get("workspace_bounds", [[-0.5, -0.5, 0.0], [1.0, 1.0, 1.5]])
        base_position = llm_robot.get("base_position")
        if not base_position:
            base_position = [
                (workspace_bounds[0][0] + workspace_bounds[1][0]) / 2,
                (workspace_bounds[0][1] + workspace_bounds[1][1]) / 2,
                (workspace_bounds[0][2] + workspace_bounds[1][2]) / 2,
            ]
        robot_config = RobotConfig(
            robot_type=robot_type,
            base_position=base_position,
            workspace_bounds=workspace_bounds,
        )
        self.log(
            f"LLM derived workspace_bounds={workspace_bounds}, "
            f"base_position={base_position}"
        )

        # Parse tasks
        raw_tasks = data.get("suggested_tasks", [])
        self.log(f"LLM generated {len(raw_tasks)} task suggestions")

        tasks = []
        seen_task_keys = set()  # Deduplication: (task_type, target_object)
        valid_task_types = set(DIFFICULTY_FACTORS.keys())
        for t in raw_tasks:
            task_type = t.get("task_type", "")
            if task_type not in valid_task_types:
                self.log(f"Skipping unknown task_type from LLM: {task_type}", level="WARNING")
                continue
            target_obj = str(t.get("target_object", ""))
            dedup_key = (task_type, target_obj.lower())
            if dedup_key in seen_task_keys:
                self.log(f"Skipping duplicate LLM task: {task_type} -> {target_obj}", level="INFO")
                continue
            seen_task_keys.add(dedup_key)
            tasks.append(SuggestedTask(
                task_type=task_type,
                target_object=target_obj,
                task_id=t.get("task_id"),
                task_name=t.get("task_name"),
                task_complexity=t.get("task_complexity"),
                curriculum_split=t.get("curriculum_split"),
                required_camera_ids=t.get("required_camera_ids") or list(CANONICAL_REQUIRED_CAMERA_IDS),
                goal_region=t.get("goal_region"),
                difficulty=t.get("difficulty", "medium"),
                priority=min(int(t.get("priority", 1)), 5),
                description_hint=t.get("description_hint"),
                requires_object_motion=(
                    t.get("requires_object_motion")
                    if t.get("requires_object_motion") is not None
                    else _infer_requires_object_motion(task_type, t.get("description_hint"))
                ),
                constraints=t.get("constraints", {}),
            ))

        self.log(f"LLM tasks after deduplication: {len(tasks)} (from {len(raw_tasks)} raw)")
        return tasks, robot_config

    def _build_scene_planning_prompt(
        self,
        manifest: Dict[str, Any],
        environment_type: str,
        robot_type: str,
        max_tasks: int,
    ) -> str:
        """Build a unified prompt for scene planning: bounds, robot placement, and tasks."""
        objects = manifest.get("objects", [])

        # Build object descriptions
        object_descriptions = []
        for obj in objects:
            sim_role = obj.get("sim_role", "unknown")
            if sim_role in ("background", "scene_shell"):
                continue
            semantics = obj.get("semantics", {})
            affordances = semantics.get("affordances", [])
            aff_names = []
            for a in affordances:
                if isinstance(a, str):
                    aff_names.append(a)
                elif isinstance(a, dict):
                    aff_names.append(a.get("name", str(a)))
            pos = obj.get("transform", {}).get("position", {})
            mass = obj.get("physics", {}).get("mass")
            desc = {
                "id": obj.get("id"),
                "name": obj.get("name"),
                "category": obj.get("category"),
                "sim_role": sim_role,
                "affordances": aff_names,
            }
            if pos:
                desc["position"] = pos
            if mass is not None:
                desc["mass_kg"] = mass
            object_descriptions.append(desc)

        objects_json = json.dumps(object_descriptions, indent=2)
        valid_types = sorted(DIFFICULTY_FACTORS.keys())

        return f"""You are a robot scene planner for a simulated {environment_type} environment.

Given the objects below, you must:
1. Determine the workspace bounding box that contains all manipulable/interactive objects
2. Choose an optimal robot base position within the workspace where the robot can reach the most objects
3. Generate diverse manipulation tasks, ensuring all tasks target objects within the workspace bounds

## Scene Objects
{objects_json}

## Robot Type
{robot_type} (typical reach radius ~0.85m)

## Valid Task Types
{json.dumps(valid_types)}

## Instructions

### Workspace Bounds
- Analyze all object positions and compute a bounding box [min_xyz, max_xyz] that encompasses all manipulable and interactive objects with ~0.3m padding
- The bounds must contain every object referenced in suggested_tasks

### Robot Base Position
- Place the robot where it can reach the most manipulable objects
- Position should be within the workspace bounds
- Consider the robot's reach radius (~0.85m) — objects beyond this distance from the base cannot be manipulated

### Tasks
- Generate up to {max_tasks} tasks — MAXIMIZE DIVERSITY
- Each task MUST reference a real object ID from the scene
- ONLY create tasks for objects within the robot's reach from the base position
- Prefer objects with relevant affordances (e.g., Graspable for pick_place)
- IMPORTANT: Use ALL applicable task types, not just pick_place. Include interact, open_close, slide, turn, press, pour, stack, organize, etc.
- Generate MULTI-STEP tasks where possible (e.g., "pick up mug, place on tray, then pick up plate, place on tray")
- For articulated objects (drawers, doors, knobs), prefer open_close, slide, or turn tasks
- Vary difficulty levels: at least 20% easy, 50% medium, 30% hard
- Set priority 1-5 (5 = most important)
- Write unique, specific natural language description_hints (avoid generic descriptions)
- goal_region should be another object ID (a surface or container) when applicable
- Do NOT generate duplicate tasks targeting the same object with the same task_type

## Output Format
Return JSON with this exact structure:
{{
  "robot_config": {{
    "workspace_bounds": [[min_x, min_y, min_z], [max_x, max_y, max_z]],
    "base_position": [x, y, z]
  }},
  "suggested_tasks": [
    {{
      "task_id": "stable_task_id",
      "task_name": "human_readable_task_name",
      "task_type": "pick_place",
      "target_object": "ObjectId",
      "task_complexity": "atomic|composite",
      "curriculum_split": "pretrain|target",
      "required_camera_ids": ["left", "right", "wrist"],
      "goal_region": "SurfaceId",
      "difficulty": "easy|medium|hard",
      "priority": 1,
      "description_hint": "Natural language task description",
      "constraints": {{}}
    }}
  ]
}}"""

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
            if obj_count < self.task_max_per_object:
                diverse_tasks.append(task)
                seen_objects[task.target_object] = obj_count + 1

            if len(diverse_tasks) >= max_tasks:
                break

        return diverse_tasks

    def _filter_tasks_by_reachability(
        self,
        tasks: List[SuggestedTask],
        manifest: Dict[str, Any],
        robot_config: RobotConfig,
        strict_reachability: bool,
    ) -> Tuple[List[SuggestedTask], Dict[str, int]]:
        """Filter tasks by robot reachability and workspace bounds."""
        import os as _os

        # Allow bypassing reachability filter entirely
        if _os.environ.get("GENIESIM_SKIP_REACHABILITY_FILTER", "0") == "1":
            self.log(
                "Reachability filter skipped (GENIESIM_SKIP_REACHABILITY_FILTER=1)",
                level="WARNING",
            )
            return tasks, {
                "tasks_total_before_reachability_filter": len(tasks),
                "tasks_filtered_unreachable": 0,
                "tasks_filtered_missing_position": 0,
                "tasks_missing_position": 0,
                "tasks_total_after_reachability_filter": len(tasks),
            }

        def _pos_to_list(pos):
            """Convert position dict {"x":..,"y":..,"z":..} to [x,y,z] list."""
            if isinstance(pos, dict):
                return [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
            return pos

        objects = manifest.get("objects", [])
        object_positions = {
            str(obj.get("id", "")): _pos_to_list(obj.get("transform", {}).get("position"))
            for obj in objects
        }

        reach_radius = None
        try:
            robot_spec = ROBOT_SPECS[RobotType(robot_config.robot_type)]
            reach_radius = robot_spec.reach_radius
        except (KeyError, ValueError):
            self.log(
                f"Unknown robot type '{robot_config.robot_type}' for reachability filter",
                level="WARNING",
            )

        base_position = robot_config.base_position
        workspace_bounds = robot_config.workspace_bounds

        def within_bounds(position: List[float]) -> bool:
            if len(position) < 3 or len(workspace_bounds) != 2:
                return False
            min_bounds, max_bounds = workspace_bounds
            return all(
                min_bounds[i] <= position[i] <= max_bounds[i]
                for i in range(3)
            )

        filtered_tasks = []
        filtered_unreachable = 0
        filtered_missing_position = 0
        missing_positions = 0

        for task in tasks:
            position = object_positions.get(task.target_object)
            if not position or len(position) < 3:
                missing_positions += 1
                self.log(
                    f"Missing position for object '{task.target_object}' in reachability filter",
                    level="WARNING",
                )
                if strict_reachability:
                    filtered_missing_position += 1
                    continue
                filtered_tasks.append(task)
                continue

            if not within_bounds(position):
                filtered_unreachable += 1
                continue

            if reach_radius is not None:
                distance = math.dist(base_position[:3], position[:3])
                if distance > reach_radius:
                    filtered_unreachable += 1
                    continue

            filtered_tasks.append(task)

        # Safety net: if ALL tasks were filtered out, keep them with a warning
        # so the pipeline can still generate data (trajectory planner will use
        # fallback targets).
        if not filtered_tasks and tasks:
            self.log(
                f"Reachability filter rejected ALL {len(tasks)} tasks — keeping them "
                f"anyway (workspace bounds may not match scene coordinate system). "
                f"Set GENIESIM_SKIP_REACHABILITY_FILTER=1 to suppress this warning.",
                level="WARNING",
            )
            filtered_tasks = list(tasks)
            filtered_unreachable = 0

        return filtered_tasks, {
            "tasks_total_before_reachability_filter": len(tasks),
            "tasks_filtered_unreachable": filtered_unreachable,
            "tasks_filtered_missing_position": filtered_missing_position,
            "tasks_missing_position": missing_positions,
            "tasks_total_after_reachability_filter": len(filtered_tasks),
        }

    def _filter_tasks_by_curobo(
        self,
        tasks: List[SuggestedTask],
        manifest: Dict[str, Any],
        robot_config: RobotConfig,
    ) -> Tuple[List[SuggestedTask], Dict[str, Any]]:
        use_curobo = parse_bool_env(os.getenv("GENIESIM_USE_CUROBO"), default=False)
        metadata = {
            "curobo_reachability_enabled": bool(use_curobo),
            "tasks_total_before_curobo_filter": len(tasks),
            "tasks_filtered_by_curobo": 0,
            "tasks_total_after_curobo_filter": len(tasks),
        }

        if not use_curobo or not tasks:
            return tasks, metadata

        if EPISODE_JOB_ROOT.exists() and str(EPISODE_JOB_ROOT) not in sys.path:
            sys.path.insert(0, str(EPISODE_JOB_ROOT))

        if importlib.util.find_spec("curobo_planner") is None:
            self.log(
                "GENIESIM_USE_CUROBO enabled, but curobo_planner module not found; "
                "skipping cuRobo task filtering.",
                level="WARNING",
            )
            return tasks, metadata

        curobo_planner = importlib.import_module("curobo_planner")
        if not curobo_planner.is_curobo_available():
            self.log(
                "GENIESIM_USE_CUROBO enabled, but cuRobo is not installed; skipping cuRobo checks.",
                level="WARNING",
            )
            return tasks, metadata

        planner = curobo_planner.create_curobo_planner(robot_config.robot_type)
        if planner is None:
            self.log(
                "GENIESIM_USE_CUROBO enabled, but cuRobo planner initialization failed; "
                "skipping cuRobo checks.",
                level="WARNING",
            )
            return tasks, metadata

        def _pos_to_list_2(pos):
            if isinstance(pos, dict):
                return [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
            return pos

        objects = manifest.get("objects", [])
        object_positions = {
            str(obj.get("id", "")): _pos_to_list_2(obj.get("transform", {}).get("position"))
            for obj in objects
        }
        filtered_tasks = []
        filtered_by_curobo = 0

        for task in tasks:
            position = object_positions.get(task.target_object)
            if not position or len(position) < 3:
                filtered_tasks.append(task)
                continue

            obstacles = self._build_curobo_obstacles(
                objects=objects,
                exclude_object_id=task.target_object,
                collision_object_cls=curobo_planner.CollisionObject,
                geometry_type_cls=curobo_planner.CollisionGeometryType,
            )
            goal_pose = np.array(
                [position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0],
                dtype=float,
            )
            default_joints = getattr(planner.robot_config, "default_joint_positions", None)
            if default_joints is None:
                default_joints = np.zeros(getattr(planner.robot_config, "num_joints", 0))
            start_joints = np.asarray(default_joints, dtype=float)
            request = curobo_planner.CuRoboPlanRequest(
                start_joint_positions=start_joints,
                goal_pose=goal_pose,
                obstacles=obstacles,
                max_iterations=10,
                parallel_finetune=False,
                batch_size=1,
            )

            try:
                result = planner.plan_to_pose(request)
            except Exception as exc:
                self.log(
                    f"cuRobo validation failed for task {task.target_object}: {exc}",
                    level="WARNING",
                )
                filtered_tasks.append(task)
                continue

            if result.success and result.is_collision_free:
                filtered_tasks.append(task)
            else:
                filtered_by_curobo += 1

        metadata.update(
            {
                "tasks_filtered_by_curobo": filtered_by_curobo,
                "tasks_total_after_curobo_filter": len(filtered_tasks),
            }
        )
        return filtered_tasks, metadata

    def _build_curobo_obstacles(
        self,
        objects: List[Dict[str, Any]],
        exclude_object_id: str,
        collision_object_cls: Any,
        geometry_type_cls: Any,
    ) -> List[Any]:
        obstacles = []
        for obj in objects:
            obj_id = str(obj.get("id", ""))
            if obj_id == exclude_object_id:
                continue
            sim_role = obj.get("sim_role", "")
            if sim_role not in {"static", "scene_shell", "articulated_furniture", "articulated_appliance"}:
                continue

            position = obj.get("transform", {}).get("position")
            if not position or len(position) < 3:
                continue

            dimensions = obj.get("dimensions_est", {})
            if not isinstance(dimensions, dict):
                continue

            width = float(dimensions.get("width", 0.0) or 0.0)
            depth = float(dimensions.get("depth", 0.0) or 0.0)
            height = float(dimensions.get("height", 0.0) or 0.0)
            if min(width, depth, height) <= 0.0:
                continue

            rotation = obj.get("transform", {}).get("rotation")
            if isinstance(rotation, (list, tuple)) and len(rotation) == 4:
                orientation = np.array(rotation, dtype=float)
            else:
                orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

            obstacles.append(
                collision_object_cls(
                    object_id=obj_id,
                    geometry_type=geometry_type_cls.CUBOID,
                    position=np.array(position[:3], dtype=float),
                    orientation=orientation,
                    dimensions=np.array([width, depth, height], dtype=float),
                    is_static=True,
                )
            )
        return obstacles

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
        workspace_bounds: List[List[float]]

        if bounds:
            workspace_bounds = [
                [0.0, -bounds.get("width", 1.0) / 2, 0.0],
                [bounds.get("depth", 1.0), bounds.get("width", 1.0) / 2, bounds.get("height", 2.0)],
            ]
        else:
            bounds_override = os.getenv("GENIESIM_WORKSPACE_BOUNDS_JSON")
            if bounds_override:
                workspace_bounds = self._parse_workspace_bounds_override(bounds_override)
            elif resolve_production_mode():
                raise ValueError(
                    "Workspace bounds are required in production. Provide room.bounds in the "
                    "scene manifest or set GENIESIM_WORKSPACE_BOUNDS_JSON to "
                    "[[min_x, min_y, min_z], [max_x, max_y, max_z]]."
                )
            else:
                # Default workspace — will be expanded below if objects exist
                workspace_bounds = [[-0.5, -0.5, 0.0], [1.0, 1.0, 1.5]]

        # Robot base position: prefer centroid of manipulable object positions
        # (so the robot is within reach of most objects), fall back to
        # workspace center if no object positions are available.
        _obj_positions = []
        for _obj in objects:
            _pos = _obj.get("transform", {}).get("position")
            if _pos and isinstance(_pos, dict):
                _obj_positions.append([
                    float(_pos.get("x", 0)), float(_pos.get("y", 0)), float(_pos.get("z", 0)),
                ])
            elif _pos and isinstance(_pos, (list, tuple)) and len(_pos) >= 3:
                _obj_positions.append([float(_pos[0]), float(_pos[1]), float(_pos[2])])
        if _obj_positions:
            _centroid = [
                sum(p[0] for p in _obj_positions) / len(_obj_positions),
                sum(p[1] for p in _obj_positions) / len(_obj_positions),
                0.0,  # Robot base at floor level
            ]
            # Offset 0.55m back from centroid (so objects are within reach)
            base_position = [_centroid[0] - 0.55, _centroid[1], 0.0]
            self.log(
                f"Robot base from object centroid: {[round(v, 3) for v in base_position]} "
                f"({len(_obj_positions)} objects)",
            )
            # Expand workspace_bounds to encompass all object positions + 1m margin
            _xs = [p[0] for p in _obj_positions]
            _ys = [p[1] for p in _obj_positions]
            _zs = [p[2] for p in _obj_positions]
            workspace_bounds = [
                [min(_xs) - 1.0, min(_ys) - 1.0, 0.0],
                [max(_xs) + 1.0, max(_ys) + 1.0, max(_zs) + 1.0],
            ]
            self.log(
                f"Workspace bounds expanded from objects: {[[round(v, 2) for v in b] for b in workspace_bounds]}",
            )
        else:
            base_position = [
                (workspace_bounds[0][0] + workspace_bounds[1][0]) / 2,
                (workspace_bounds[0][1] + workspace_bounds[1][1]) / 2,
                0.0,  # Floor level, not workspace center z
            ]

        return RobotConfig(
            robot_type=robot_type,
            urdf_path=urdf_path,
            base_position=base_position,
            workspace_bounds=workspace_bounds,
        )

    @staticmethod
    def _parse_workspace_bounds_override(bounds_override: str) -> List[List[float]]:
        try:
            parsed = json_loads(bounds_override)
        except JSONDecodeError as exc:
            raise ValueError(
                "GENIESIM_WORKSPACE_BOUNDS_JSON must be valid JSON formatted as "
                "[[min_x, min_y, min_z], [max_x, max_y, max_z]]."
            ) from exc

        if not isinstance(parsed, list) or len(parsed) != 2:
            raise ValueError(
                "GENIESIM_WORKSPACE_BOUNDS_JSON must be a JSON array with two 3-element lists."
            )

        bounds: List[List[float]] = []
        for index, point in enumerate(parsed):
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(
                    "GENIESIM_WORKSPACE_BOUNDS_JSON must contain two 3-element lists of numbers."
                )
            try:
                bounds.append([float(coord) for coord in point])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "GENIESIM_WORKSPACE_BOUNDS_JSON must contain numeric bounds values."
                ) from exc

        return bounds


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_task_config(
    manifest_path: Path,
    output_path: Optional[Path] = None,
    robot_type: str = "franka",
    verbose: bool = True,
    strict_reachability: bool = False,
) -> GenieSimTaskConfig:
    """
    Convenience function to generate task config from manifest file.

    Args:
        manifest_path: Path to scene_manifest.json
        output_path: Optional path to save task_config.json
        robot_type: Robot type (franka, g2, ur10)
        verbose: Print progress
        strict_reachability: Drop tasks if target positions are missing

    Returns:
        GenieSimTaskConfig
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    generator = TaskConfigGenerator(verbose=verbose)
    config = generator.generate(
        manifest,
        robot_type=robot_type,
        strict_reachability=strict_reachability,
    )

    if output_path:
        config.save(output_path)

    return config

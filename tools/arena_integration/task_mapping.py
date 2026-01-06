"""
Task-Affordance Mapping for Isaac Lab-Arena Integration.

This module maps Blueprint Pipeline policies to Arena-compatible tasks
via the affordance system. It enables automatic task generation based
on detected object affordances.

Key Concepts:
- Blueprint policies (e.g., "dish_loading") map to specific affordances
- Arena tasks are generated based on object affordances
- A single affordance can map to multiple task variants
- Tasks are composable: complex tasks chain multiple affordances
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .affordances import AffordanceType, AffordanceParams


class ArenaTaskType(str, Enum):
    """
    Standard Arena task types that can be auto-generated from affordances.

    These map directly to Isaac Lab-Arena task templates.
    """
    # Articulation tasks
    OPEN_DOOR = "OpenDoorTask"
    CLOSE_DOOR = "CloseDoorTask"
    OPEN_DRAWER = "OpenDrawerTask"
    CLOSE_DRAWER = "CloseDrawerTask"
    TURN_KNOB = "TurnKnobTask"
    PRESS_BUTTON = "PressButtonTask"
    TOGGLE_SWITCH = "ToggleSwitchTask"
    SLIDE_CONTROL = "SlideControlTask"

    # Pick-and-place tasks
    PICK_OBJECT = "PickObjectTask"
    PLACE_OBJECT = "PlaceObjectTask"
    PICK_AND_PLACE = "PickAndPlaceTask"
    STACK_OBJECTS = "StackObjectsTask"
    UNSTACK_OBJECTS = "UnstackObjectsTask"

    # Insertion tasks
    INSERT_OBJECT = "InsertObjectTask"
    REMOVE_OBJECT = "RemoveObjectTask"
    PEG_INSERTION = "PegInsertionTask"

    # Container tasks
    POUR_LIQUID = "PourLiquidTask"
    FILL_CONTAINER = "FillContainerTask"
    EMPTY_CONTAINER = "EmptyContainerTask"

    # Composite tasks
    OPEN_AND_RETRIEVE = "OpenAndRetrieveTask"
    PLACE_IN_CONTAINER = "PlaceInContainerTask"
    LOAD_DISHWASHER = "LoadDishwasherTask"
    CLEAR_TABLE = "ClearTableTask"

    # Deformable tasks
    FOLD_CLOTH = "FoldClothTask"
    HANG_OBJECT = "HangObjectTask"

    # Custom/generic
    CUSTOM = "CustomTask"


@dataclass
class ArenaTaskSpec:
    """
    Specification for an Arena-compatible task.

    This encodes everything needed to instantiate an Arena task:
    - Task type and parameters
    - Required affordances
    - Success criteria
    - Default configurations
    """
    task_type: ArenaTaskType
    display_name: str
    description: str

    # Required affordances (object must have these)
    required_affordances: list[AffordanceType] = field(default_factory=list)

    # Optional affordances (enhance task if present)
    optional_affordances: list[AffordanceType] = field(default_factory=list)

    # Task parameters (passed to Arena task constructor)
    default_params: dict[str, Any] = field(default_factory=dict)

    # Success criteria
    success_threshold: float = 0.9  # Threshold for task completion
    max_steps: int = 500            # Maximum episode steps

    # Observation keys required
    required_observations: list[str] = field(default_factory=list)

    # Compatible robot embodiments
    compatible_embodiments: list[str] = field(default_factory=lambda: [
        "franka", "ur10", "fetch", "gr1", "g1"
    ])


# Mapping from AffordanceType to ArenaTaskSpecs
AFFORDANCE_TO_TASKS: dict[AffordanceType, list[ArenaTaskSpec]] = {
    AffordanceType.OPENABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.OPEN_DOOR,
            display_name="Open Door",
            description="Open a door/lid to a target angle",
            required_affordances=[AffordanceType.OPENABLE],
            default_params={
                "target_openness": 0.9,
                "reset_openness": 0.1,
            },
            required_observations=["joint_pos", "joint_vel", "handle_pos"],
        ),
        ArenaTaskSpec(
            task_type=ArenaTaskType.CLOSE_DOOR,
            display_name="Close Door",
            description="Close a door/lid from open position",
            required_affordances=[AffordanceType.OPENABLE],
            default_params={
                "target_openness": 0.1,
                "reset_openness": 0.8,
            },
            required_observations=["joint_pos", "joint_vel", "handle_pos"],
        ),
        ArenaTaskSpec(
            task_type=ArenaTaskType.OPEN_DRAWER,
            display_name="Open Drawer",
            description="Pull open a drawer",
            required_affordances=[AffordanceType.OPENABLE],
            default_params={
                "target_extension": 0.8,
                "reset_extension": 0.1,
            },
            required_observations=["joint_pos", "joint_vel", "handle_pos"],
        ),
        ArenaTaskSpec(
            task_type=ArenaTaskType.CLOSE_DRAWER,
            display_name="Close Drawer",
            description="Push closed a drawer",
            required_affordances=[AffordanceType.OPENABLE],
            default_params={
                "target_extension": 0.1,
                "reset_extension": 0.7,
            },
            required_observations=["joint_pos", "joint_vel", "handle_pos"],
        ),
    ],

    AffordanceType.TURNABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.TURN_KNOB,
            display_name="Turn Knob",
            description="Rotate a knob to target position",
            required_affordances=[AffordanceType.TURNABLE],
            default_params={
                "target_rotation": 1.57,  # 90 degrees
                "rotation_tolerance": 0.1,
            },
            required_observations=["joint_pos", "joint_vel"],
        ),
    ],

    AffordanceType.PRESSABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.PRESS_BUTTON,
            display_name="Press Button",
            description="Press a button",
            required_affordances=[AffordanceType.PRESSABLE],
            default_params={
                "press_force_threshold": 5.0,
                "hold_duration": 0.1,
            },
            required_observations=["button_state", "contact_force"],
            max_steps=200,
        ),
        ArenaTaskSpec(
            task_type=ArenaTaskType.TOGGLE_SWITCH,
            display_name="Toggle Switch",
            description="Toggle a switch between states",
            required_affordances=[AffordanceType.PRESSABLE],
            default_params={
                "target_state": True,
            },
            required_observations=["switch_state"],
            max_steps=200,
        ),
    ],

    AffordanceType.SLIDABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.SLIDE_CONTROL,
            display_name="Slide Control",
            description="Move a slider to target position",
            required_affordances=[AffordanceType.SLIDABLE],
            default_params={
                "target_position": 0.5,
                "position_tolerance": 0.05,
            },
            required_observations=["slider_pos"],
        ),
    ],

    AffordanceType.GRASPABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.PICK_OBJECT,
            display_name="Pick Object",
            description="Grasp and lift an object",
            required_affordances=[AffordanceType.GRASPABLE],
            default_params={
                "lift_height": 0.15,
                "grasp_force_range": (5.0, 50.0),
            },
            required_observations=["object_pos", "object_quat", "gripper_state"],
        ),
        ArenaTaskSpec(
            task_type=ArenaTaskType.PICK_AND_PLACE,
            display_name="Pick and Place",
            description="Pick object and place at target location",
            required_affordances=[AffordanceType.GRASPABLE],
            optional_affordances=[AffordanceType.PLACEABLE],
            default_params={
                "placement_tolerance": 0.03,
            },
            required_observations=["object_pos", "object_quat", "target_pos", "gripper_state"],
        ),
    ],

    AffordanceType.STACKABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.STACK_OBJECTS,
            display_name="Stack Objects",
            description="Stack objects vertically",
            required_affordances=[AffordanceType.GRASPABLE, AffordanceType.STACKABLE],
            default_params={
                "stack_height": 3,
                "alignment_tolerance": 0.02,
            },
            required_observations=["object_poses", "stack_state"],
        ),
        ArenaTaskSpec(
            task_type=ArenaTaskType.UNSTACK_OBJECTS,
            display_name="Unstack Objects",
            description="Remove top object from stack",
            required_affordances=[AffordanceType.GRASPABLE, AffordanceType.STACKABLE],
            default_params={
                "target_location": [0.3, 0.0, 0.1],
            },
            required_observations=["object_poses", "stack_state"],
        ),
    ],

    AffordanceType.INSERTABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.INSERT_OBJECT,
            display_name="Insert Object",
            description="Insert object into receptacle",
            required_affordances=[AffordanceType.GRASPABLE, AffordanceType.INSERTABLE],
            default_params={
                "insertion_force_limit": 20.0,
                "alignment_tolerance": 0.005,
            },
            required_observations=["object_pos", "object_quat", "receptacle_pos", "insertion_depth"],
            max_steps=300,
        ),
        ArenaTaskSpec(
            task_type=ArenaTaskType.PEG_INSERTION,
            display_name="Peg Insertion",
            description="Insert peg into hole (precision task)",
            required_affordances=[AffordanceType.GRASPABLE, AffordanceType.INSERTABLE],
            default_params={
                "clearance_tolerance": 0.001,
                "force_limit": 30.0,
            },
            required_observations=["peg_pos", "peg_quat", "hole_pos", "contact_force"],
            max_steps=400,
        ),
    ],

    AffordanceType.POURABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.POUR_LIQUID,
            display_name="Pour Liquid",
            description="Pour contents from container",
            required_affordances=[AffordanceType.GRASPABLE, AffordanceType.POURABLE],
            optional_affordances=[AffordanceType.FILLABLE],
            default_params={
                "pour_amount": 0.5,  # Fraction to pour
                "spill_tolerance": 0.1,
            },
            required_observations=["container_quat", "fill_level", "target_fill"],
        ),
    ],

    AffordanceType.FILLABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.FILL_CONTAINER,
            display_name="Fill Container",
            description="Fill container to target level",
            required_affordances=[AffordanceType.FILLABLE],
            optional_affordances=[AffordanceType.POURABLE],
            default_params={
                "target_fill_level": 0.8,
                "overflow_penalty": True,
            },
            required_observations=["fill_level", "target_level"],
        ),
    ],

    AffordanceType.CONTAINABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.PLACE_IN_CONTAINER,
            display_name="Place in Container",
            description="Place object inside container",
            required_affordances=[AffordanceType.CONTAINABLE],
            optional_affordances=[AffordanceType.GRASPABLE, AffordanceType.OPENABLE],
            default_params={
                "object_count": 1,
            },
            required_observations=["object_pos", "container_bounds", "containment_state"],
        ),
    ],

    AffordanceType.FOLDABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.FOLD_CLOTH,
            display_name="Fold Cloth",
            description="Fold deformable cloth item",
            required_affordances=[AffordanceType.GRASPABLE, AffordanceType.FOLDABLE],
            default_params={
                "fold_type": "half",  # half, quarter, thirds
                "alignment_tolerance": 0.05,
            },
            required_observations=["cloth_keypoints", "fold_state"],
            max_steps=600,
        ),
    ],

    AffordanceType.HANGABLE: [
        ArenaTaskSpec(
            task_type=ArenaTaskType.HANG_OBJECT,
            display_name="Hang Object",
            description="Hang object on hook/rack",
            required_affordances=[AffordanceType.GRASPABLE, AffordanceType.HANGABLE],
            default_params={
                "hook_tolerance": 0.02,
            },
            required_observations=["object_pos", "hook_pos", "hang_state"],
        ),
    ],
}

# Mapping from Blueprint policies to primary affordances and Arena tasks
BLUEPRINT_TO_ARENA_MAPPING: dict[str, dict[str, Any]] = {
    "dexterous_pick_place": {
        "primary_affordances": [AffordanceType.GRASPABLE],
        "arena_tasks": [ArenaTaskType.PICK_AND_PLACE, ArenaTaskType.PICK_OBJECT],
        "description": "General pick and place with various grasp strategies",
    },
    "articulated_access": {
        "primary_affordances": [AffordanceType.OPENABLE],
        "arena_tasks": [ArenaTaskType.OPEN_DOOR, ArenaTaskType.OPEN_DRAWER],
        "description": "Opening articulated mechanisms",
    },
    "drawer_manipulation": {
        "primary_affordances": [AffordanceType.OPENABLE],
        "arena_tasks": [ArenaTaskType.OPEN_DRAWER, ArenaTaskType.CLOSE_DRAWER],
        "description": "Opening and closing drawers",
        "filter": {"joint_type": "prismatic"},
    },
    "door_manipulation": {
        "primary_affordances": [AffordanceType.OPENABLE],
        "arena_tasks": [ArenaTaskType.OPEN_DOOR, ArenaTaskType.CLOSE_DOOR],
        "description": "Opening and closing doors",
        "filter": {"joint_type": "revolute"},
    },
    "knob_manipulation": {
        "primary_affordances": [AffordanceType.TURNABLE],
        "arena_tasks": [ArenaTaskType.TURN_KNOB],
        "description": "Rotating knobs, dials, valves",
    },
    "panel_interaction": {
        "primary_affordances": [AffordanceType.PRESSABLE],
        "arena_tasks": [ArenaTaskType.PRESS_BUTTON, ArenaTaskType.TOGGLE_SWITCH],
        "description": "Interacting with control panels",
    },
    "precision_insertion": {
        "primary_affordances": [AffordanceType.INSERTABLE, AffordanceType.GRASPABLE],
        "arena_tasks": [ArenaTaskType.INSERT_OBJECT, ArenaTaskType.PEG_INSERTION],
        "description": "Precision insertion tasks",
    },
    "dish_loading": {
        "primary_affordances": [AffordanceType.GRASPABLE, AffordanceType.OPENABLE, AffordanceType.CONTAINABLE],
        "arena_tasks": [ArenaTaskType.LOAD_DISHWASHER, ArenaTaskType.PLACE_IN_CONTAINER],
        "description": "Loading dishes into dishwasher",
        "composite": True,
    },
    "table_clearing": {
        "primary_affordances": [AffordanceType.GRASPABLE, AffordanceType.PLACEABLE],
        "arena_tasks": [ArenaTaskType.CLEAR_TABLE, ArenaTaskType.PICK_AND_PLACE],
        "description": "Clearing items from table surfaces",
        "composite": True,
    },
    "laundry_sorting": {
        "primary_affordances": [AffordanceType.GRASPABLE, AffordanceType.FOLDABLE],
        "arena_tasks": [ArenaTaskType.FOLD_CLOTH, ArenaTaskType.PICK_AND_PLACE],
        "description": "Sorting and folding laundry",
    },
    "mixed_sku_logistics": {
        "primary_affordances": [AffordanceType.GRASPABLE, AffordanceType.STACKABLE],
        "arena_tasks": [ArenaTaskType.PICK_AND_PLACE, ArenaTaskType.STACK_OBJECTS],
        "description": "Warehouse logistics operations",
    },
    "grocery_stocking": {
        "primary_affordances": [AffordanceType.GRASPABLE, AffordanceType.PLACEABLE],
        "arena_tasks": [ArenaTaskType.PICK_AND_PLACE],
        "description": "Restocking shelves",
    },
    "general_manipulation": {
        "primary_affordances": [AffordanceType.GRASPABLE],
        "arena_tasks": [ArenaTaskType.PICK_OBJECT, ArenaTaskType.PICK_AND_PLACE],
        "description": "Generic manipulation tasks",
    },
}


class TaskAffordanceMapper:
    """
    Maps between Blueprint policies, affordances, and Arena tasks.

    This class provides the intelligence to:
    1. Given a Blueprint policy, find relevant Arena tasks
    2. Given object affordances, generate applicable Arena task specs
    3. Filter tasks based on object-specific affordance parameters
    """

    def __init__(self, policy_config: Optional[dict[str, Any]] = None):
        """
        Initialize mapper.

        Args:
            policy_config: Optional policy configuration from environment_policies.json
        """
        self.policy_config = policy_config or {}
        self.policies = self.policy_config.get("policies", {})

    def get_tasks_for_policy(
        self,
        policy_id: str,
        object_affordances: Optional[list[AffordanceParams]] = None
    ) -> list[ArenaTaskSpec]:
        """
        Get Arena tasks compatible with a Blueprint policy.

        Args:
            policy_id: Blueprint policy ID (e.g., "drawer_manipulation")
            object_affordances: Optional object affordances to filter by

        Returns:
            List of compatible ArenaTaskSpec
        """
        mapping = BLUEPRINT_TO_ARENA_MAPPING.get(policy_id, {})
        if not mapping:
            return []

        primary_affordances = mapping.get("primary_affordances", [])
        task_types = mapping.get("arena_tasks", [])
        task_filter = mapping.get("filter", {})

        tasks: list[ArenaTaskSpec] = []

        # Get task specs for each primary affordance
        for aff_type in primary_affordances:
            aff_tasks = AFFORDANCE_TO_TASKS.get(aff_type, [])
            for task_spec in aff_tasks:
                if task_spec.task_type in task_types:
                    # Apply filter if specified
                    if task_filter and object_affordances:
                        if not self._matches_filter(task_filter, object_affordances):
                            continue
                    tasks.append(task_spec)

        return tasks

    def get_tasks_for_affordances(
        self,
        affordances: list[AffordanceParams]
    ) -> list[ArenaTaskSpec]:
        """
        Get all Arena tasks compatible with given affordances.

        Args:
            affordances: List of detected affordances

        Returns:
            List of compatible ArenaTaskSpec
        """
        tasks: list[ArenaTaskSpec] = []
        aff_types = {aff.affordance_type for aff in affordances}

        for aff_type in aff_types:
            aff_tasks = AFFORDANCE_TO_TASKS.get(aff_type, [])
            for task_spec in aff_tasks:
                # Check if all required affordances are present
                if all(req in aff_types for req in task_spec.required_affordances):
                    tasks.append(task_spec)

        return tasks

    def _matches_filter(
        self,
        filter_spec: dict[str, Any],
        affordances: list[AffordanceParams]
    ) -> bool:
        """Check if affordances match filter specification."""
        for aff in affordances:
            for key, value in filter_spec.items():
                if hasattr(aff, key):
                    if getattr(aff, key) != value:
                        return False
        return True

    def generate_task_config(
        self,
        task_spec: ArenaTaskSpec,
        obj: dict[str, Any],
        affordance_params: AffordanceParams
    ) -> dict[str, Any]:
        """
        Generate Arena task configuration for a specific object.

        Args:
            task_spec: Arena task specification
            obj: Object dictionary from manifest
            affordance_params: Object's affordance parameters

        Returns:
            Dict with complete Arena task configuration
        """
        config = {
            "task_type": task_spec.task_type.value,
            "display_name": task_spec.display_name,
            "description": task_spec.description,
            "object_id": obj.get("id"),
            "object_name": obj.get("name") or obj.get("category"),
            "max_steps": task_spec.max_steps,
            "success_threshold": task_spec.success_threshold,
            "compatible_embodiments": task_spec.compatible_embodiments,
            "params": dict(task_spec.default_params),
        }

        # Merge affordance-specific parameters
        if task_spec.task_type in (ArenaTaskType.OPEN_DOOR, ArenaTaskType.CLOSE_DOOR):
            if affordance_params.joint_type == "revolute":
                config["params"]["target_angle"] = affordance_params.open_angle
                config["params"]["joint_type"] = "revolute"
            else:
                config["params"]["target_distance"] = affordance_params.open_distance
                config["params"]["joint_type"] = "prismatic"

            if affordance_params.joint_name:
                config["params"]["joint_prim"] = affordance_params.joint_name

        elif task_spec.task_type in (ArenaTaskType.OPEN_DRAWER, ArenaTaskType.CLOSE_DRAWER):
            config["params"]["target_extension"] = affordance_params.open_distance
            if affordance_params.joint_name:
                config["params"]["joint_prim"] = affordance_params.joint_name

        elif task_spec.task_type == ArenaTaskType.TURN_KNOB:
            config["params"]["rotation_axis"] = affordance_params.rotation_axis
            config["params"]["rotation_range"] = list(affordance_params.rotation_range)

        elif task_spec.task_type == ArenaTaskType.PRESS_BUTTON:
            config["params"]["press_depth"] = affordance_params.press_depth
            config["params"]["is_toggle"] = affordance_params.toggle
            if affordance_params.button_ids:
                config["params"]["button_prims"] = affordance_params.button_ids

        elif task_spec.task_type in (ArenaTaskType.PICK_OBJECT, ArenaTaskType.PICK_AND_PLACE):
            config["params"]["grasp_width_range"] = list(affordance_params.grasp_width_range)
            config["params"]["approach_direction"] = affordance_params.grasp_approach_direction

        elif task_spec.task_type in (ArenaTaskType.INSERT_OBJECT, ArenaTaskType.PEG_INSERTION):
            config["params"]["insertion_axis"] = affordance_params.insertion_axis
            config["params"]["insertion_depth"] = affordance_params.insertion_depth
            config["params"]["tolerance"] = affordance_params.receptacle_tolerance

        return config


def get_arena_tasks_for_affordances(
    affordances: list[AffordanceParams]
) -> list[ArenaTaskSpec]:
    """
    Convenience function to get Arena tasks for a set of affordances.

    Args:
        affordances: List of detected affordance parameters

    Returns:
        List of compatible Arena task specifications
    """
    mapper = TaskAffordanceMapper()
    return mapper.get_tasks_for_affordances(affordances)

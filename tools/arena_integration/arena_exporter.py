"""
Arena Scene Exporter - Convert Blueprint scenes to Isaac Lab-Arena format.

This module exports Blueprint Pipeline scenes to the Isaac Lab-Arena format,
enabling standardized policy evaluation and benchmark creation.

Output Structure:
    scenes/{scene_id}/arena/
        ├── scene_module.py         # Arena Scene class definition
        ├── tasks/                   # Generated task definitions
        │   ├── open_microwave.py
        │   ├── pick_cup.py
        │   └── ...
        ├── arena_manifest.json      # Arena-specific manifest
        ├── asset_registry.json      # Asset registry for Arena
        └── hub_config.yaml          # LeRobot Hub configuration
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .affordances import (
    AffordanceType,
    AffordanceDetector,
    AffordanceParams,
)
from .task_mapping import (
    ArenaTaskSpec,
    ArenaTaskType,
    TaskAffordanceMapper,
    AFFORDANCE_TO_TASKS,
)

LOGGER = logging.getLogger(__name__)
FALLBACK_AFFORDANCE_VERSION = "0.1.0"
FALLBACK_AFFORDANCES = [
    "Openable",
    "Turnable",
    "Pressable",
    "Graspable",
    "Insertable",
    "Stackable",
    "Pourable",
    "Fillable",
    "Containable",
    "Foldable",
    "Hangable",
]


@dataclass
class ArenaExportConfig:
    """Configuration for Arena export."""
    scene_id: str
    scene_path: str                           # Path to scene.usda
    output_dir: Path
    environment_type: str = "generic"
    use_llm_affordances: bool = True          # Use Gemini for affordance detection
    supported_embodiments: list[str] = field(default_factory=lambda: [
        "franka", "ur10", "fetch", "gr1", "g1"
    ])
    generate_hub_config: bool = True
    hub_namespace: str = "blueprint-robotics"


@dataclass
class ArenaExportResult:
    """Result of Arena export operation."""
    success: bool
    scene_id: str
    output_dir: Path
    generated_files: list[str]
    task_count: int
    affordance_count: int
    errors: list[str] = field(default_factory=list)


class ArenaSceneExporter:
    """
    Exports Blueprint scenes to Isaac Lab-Arena compatible format.

    This exporter:
    1. Detects affordances for all scene objects
    2. Generates Arena Scene class with asset registry
    3. Creates task definitions based on affordances
    4. Produces hub configuration for LeRobot registration
    """

    def __init__(self, config: ArenaExportConfig):
        self.config = config
        self.affordance_detector = AffordanceDetector(use_llm=config.use_llm_affordances)
        self.task_mapper = TaskAffordanceMapper()

    def export(self, manifest: dict[str, Any]) -> ArenaExportResult:
        """
        Export scene to Arena format.

        Args:
            manifest: Scene manifest dictionary

        Returns:
            ArenaExportResult with export status and details
        """
        errors: list[str] = []
        generated_files: list[str] = []

        # Create output directory
        arena_dir = self.config.output_dir / "arena"
        arena_dir.mkdir(parents=True, exist_ok=True)
        tasks_dir = arena_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)

        # Step 1: Detect affordances for all objects
        objects = manifest.get("objects", [])
        object_affordances: dict[str, list[AffordanceParams]] = {}
        all_affordances: list[AffordanceParams] = []

        for obj in objects:
            obj_id = obj.get("id", "unknown")
            affordances = self.affordance_detector.detect(obj)
            object_affordances[obj_id] = affordances
            all_affordances.extend(affordances)

            # Update object in manifest with affordances
            manifest_format = self.affordance_detector.to_manifest_format(affordances)
            obj.setdefault("semantics", {}).update(manifest_format)

        # Step 2: Validate affordances against Arena enum (or fallback)
        validation = self._validate_affordances(all_affordances)
        if validation["status"] == "error":
            LOGGER.error(validation["message"])
            errors.append(validation["message"])
        elif validation["status"] == "warning":
            LOGGER.warning(validation["message"])

        # Step 3: Generate Arena manifest
        arena_manifest = self._generate_arena_manifest(
            manifest,
            object_affordances,
            validation["metadata"],
        )
        manifest_path = arena_dir / "arena_manifest.json"
        manifest_path.write_text(json.dumps(arena_manifest, indent=2))
        generated_files.append(str(manifest_path))

        # Step 4: Generate asset registry
        asset_registry = self._generate_asset_registry(manifest, object_affordances)
        registry_path = arena_dir / "asset_registry.json"
        registry_path.write_text(json.dumps(asset_registry, indent=2))
        generated_files.append(str(registry_path))

        # Step 5: Generate Scene class module
        scene_module = self._generate_scene_module(manifest, object_affordances)
        scene_path = arena_dir / "scene_module.py"
        scene_path.write_text(scene_module)
        generated_files.append(str(scene_path))

        # Step 6: Generate task definitions
        task_count = 0
        for obj in objects:
            obj_id = obj.get("id")
            affordances = object_affordances.get(obj_id, [])

            for aff in affordances:
                task_specs = AFFORDANCE_TO_TASKS.get(aff.affordance_type, [])
                for task_spec in task_specs:
                    try:
                        task_code = self._generate_task_module(obj, aff, task_spec)
                        task_name = self._make_task_filename(obj, task_spec)
                        task_path = tasks_dir / f"{task_name}.py"
                        task_path.write_text(task_code)
                        generated_files.append(str(task_path))
                        task_count += 1
                    except Exception as e:
                        errors.append(f"Failed to generate task {task_spec.task_type} for {obj_id}: {e}")

        # Step 7: Generate tasks __init__.py
        init_code = self._generate_tasks_init(tasks_dir)
        init_path = tasks_dir / "__init__.py"
        init_path.write_text(init_code)
        generated_files.append(str(init_path))

        # Step 8: Generate hub config if requested
        if self.config.generate_hub_config:
            hub_config = self._generate_hub_config(manifest, task_count)
            hub_path = arena_dir / "hub_config.yaml"
            hub_path.write_text(hub_config)
            generated_files.append(str(hub_path))

        return ArenaExportResult(
            success=len(errors) == 0,
            scene_id=self.config.scene_id,
            output_dir=arena_dir,
            generated_files=generated_files,
            task_count=task_count,
            affordance_count=len(all_affordances),
            errors=errors,
        )

    def _generate_arena_manifest(
        self,
        manifest: dict[str, Any],
        object_affordances: dict[str, list[AffordanceParams]],
        affordance_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate Arena-specific manifest."""
        scene_meta = manifest.get("scene", {})

        arena_objects = []
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id")
            affordances = object_affordances.get(obj_id, [])

            arena_obj = {
                "id": obj_id,
                "name": obj.get("name") or obj.get("category"),
                "category": obj.get("category"),
                "sim_role": obj.get("sim_role"),
                "usd_path": obj.get("asset", {}).get("path"),
                "transform": obj.get("transform"),
                "affordances": [aff.affordance_type.value for aff in affordances],
                "affordance_params": {},
            }

            # Add affordance-specific params
            for aff in affordances:
                params = self._affordance_to_dict(aff)
                arena_obj["affordance_params"][aff.affordance_type.value] = params

            arena_objects.append(arena_obj)

        return {
            "version": "0.1.0",
            "arena_version": "0.1.0",
            "scene_id": self.config.scene_id,
            "environment_type": self.config.environment_type,
            "source": "BlueprintPipeline",
            "generated_at": datetime.utcnow().isoformat(),
            "scene": {
                "usd_path": self.config.scene_path,
                "coordinate_frame": scene_meta.get("coordinate_frame", "z_up"),
                "meters_per_unit": scene_meta.get("meters_per_unit", 1.0),
                "metadata": affordance_metadata,
            },
            "objects": arena_objects,
            "supported_embodiments": self.config.supported_embodiments,
            "task_count": sum(len(affs) for affs in object_affordances.values()),
        }

    def _validate_affordances(self, affordances: list[AffordanceParams]) -> dict[str, Any]:
        """Validate detected affordances against Arena's official enum or fallback list."""
        official_affordances, reference = self._get_official_affordances()
        used_fallback = reference["source"] == "fallback"

        if used_fallback:
            LOGGER.info(
                "Arena not installed; using fallback affordance list version %s.",
                FALLBACK_AFFORDANCE_VERSION,
            )

        seen_affordances = {aff.affordance_type.value for aff in affordances}
        unknown_affordances = sorted(seen_affordances - official_affordances)

        metadata = {
            "affordance_reference": reference,
            "unknown_affordances": unknown_affordances,
            "known_affordance_count": len(official_affordances),
            "used_fallback": used_fallback,
        }

        if unknown_affordances:
            message = (
                "Unknown affordances detected in scene manifest: "
                f"{unknown_affordances}. "
                f"Reference source: {reference['source']}."
            )
            status = "warning" if used_fallback else "error"
            return {"status": status, "message": message, "metadata": metadata}

        return {
            "status": "ok",
            "message": "All affordances validated against reference.",
            "metadata": metadata,
        }

    def _get_official_affordances(self) -> tuple[set[str], dict[str, Any]]:
        """Fetch the official affordance list from Isaac Lab-Arena, with fallback."""
        try:
            from enum import Enum
            from isaaclab_arena.core import AffordanceType as ArenaAffordanceType

            if isinstance(ArenaAffordanceType, type) and issubclass(ArenaAffordanceType, Enum):
                values = {member.value for member in ArenaAffordanceType}
            else:
                values = {
                    value for name in dir(ArenaAffordanceType)
                    if not name.startswith("_")
                    for value in [getattr(ArenaAffordanceType, name)]
                    if isinstance(value, str)
                }

            return values, {
                "source": "isaaclab_arena.core.AffordanceType",
                "version": getattr(ArenaAffordanceType, "__version__", None),
                "values": sorted(values),
            }
        except ImportError:
            return set(FALLBACK_AFFORDANCES), {
                "source": "fallback",
                "version": FALLBACK_AFFORDANCE_VERSION,
                "values": FALLBACK_AFFORDANCES,
            }

    def _generate_asset_registry(
        self,
        manifest: dict[str, Any],
        object_affordances: dict[str, list[AffordanceParams]]
    ) -> dict[str, Any]:
        """Generate Arena asset registry."""
        assets = {}

        for obj in manifest.get("objects", []):
            obj_id = obj.get("id")
            obj_name = obj.get("name") or obj.get("category") or obj_id
            affordances = object_affordances.get(obj_id, [])

            # Create registry entry
            assets[obj_name] = {
                "id": obj_id,
                "usd_path": obj.get("asset", {}).get("path"),
                "category": obj.get("category"),
                "sim_role": obj.get("sim_role"),
                "affordances": [aff.affordance_type.value for aff in affordances],
                "physics": obj.get("physics", {}),
                "dimensions": obj.get("dimensions_est", {}),
            }

        return {
            "version": "0.1.0",
            "source": "BlueprintPipeline",
            "scene_id": self.config.scene_id,
            "assets": assets,
        }

    def _generate_scene_module(
        self,
        manifest: dict[str, Any],
        object_affordances: dict[str, list[AffordanceParams]]
    ) -> str:
        """Generate Python module defining the Arena Scene class."""
        scene_id = self.config.scene_id
        class_name = self._to_class_name(scene_id) + "Scene"
        env_type = self.config.environment_type.title()

        # Build asset list
        asset_lines = []
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id")
            obj_name = obj.get("name") or obj.get("category") or obj_id
            usd_path = obj.get("asset", {}).get("path", "")
            affordances = object_affordances.get(obj_id, [])
            aff_str = ", ".join([f'"{a.affordance_type.value}"' for a in affordances])

            asset_lines.append(f'''
        # {obj_name}
        self.register_asset(
            name="{obj_name}",
            usd_path="{usd_path}",
            affordances=[{aff_str}],
            object_id="{obj_id}",
        )''')

        assets_code = "\n".join(asset_lines)

        return f'''"""
Arena Scene Module - {scene_id}
Auto-generated by BlueprintPipeline Arena Exporter

This module defines the Arena-compatible Scene class for this environment.
It can be used directly with Isaac Lab-Arena for policy evaluation.

Usage:
    from isaaclab_arena import ArenaEnvBuilder
    from .scene_module import {class_name}

    scene = {class_name}()
    builder = ArenaEnvBuilder(scene=scene, embodiment=my_robot)
    env = builder.build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Isaac Lab-Arena imports (available when running in Arena environment)
try:
    from isaaclab_arena import Scene, AssetRegistry
    from isaaclab_arena.core import AffordanceType
    ARENA_AVAILABLE = True
except ImportError:
    # Fallback for development/testing without Arena
    ARENA_AVAILABLE = False
    Scene = object
    AssetRegistry = None

    class AffordanceType:
        OPENABLE = "Openable"
        TURNABLE = "Turnable"
        PRESSABLE = "Pressable"
        GRASPABLE = "Graspable"
        INSERTABLE = "Insertable"
        STACKABLE = "Stackable"
        POURABLE = "Pourable"
        FILLABLE = "Fillable"
        CONTAINABLE = "Containable"
        FOLDABLE = "Foldable"
        HANGABLE = "Hangable"


# Scene metadata
SCENE_ID = "{scene_id}"
SCENE_USD_PATH = "{self.config.scene_path}"
ENVIRONMENT_TYPE = "{env_type}"
SUPPORTED_EMBODIMENTS = {self.config.supported_embodiments}


@dataclass
class {class_name}AssetEntry:
    """Asset entry for the scene registry."""
    name: str
    usd_path: str
    affordances: list[str]
    object_id: str
    transform: Optional[dict] = None


class {class_name}(Scene if ARENA_AVAILABLE else object):
    """
    Arena Scene: {scene_id}
    Environment Type: {env_type}

    This scene was auto-generated from a BlueprintPipeline scene manifest.
    It registers all assets with their detected affordances for use in
    Arena task generation and policy evaluation.
    """

    def __init__(self):
        if ARENA_AVAILABLE:
            super().__init__()

        self.scene_id = SCENE_ID
        self.usd_path = SCENE_USD_PATH
        self.environment_type = ENVIRONMENT_TYPE
        self.supported_embodiments = SUPPORTED_EMBODIMENTS

        # Asset registry
        self._assets: dict[str, {class_name}AssetEntry] = {{}}

        # Register all scene assets
        self._register_assets()

    def _register_assets(self):
        """Register all assets from the Blueprint scene."""
{assets_code}

    def register_asset(
        self,
        name: str,
        usd_path: str,
        affordances: list[str],
        object_id: str,
        transform: Optional[dict] = None,
    ):
        """Register an asset with the scene."""
        entry = {class_name}AssetEntry(
            name=name,
            usd_path=usd_path,
            affordances=affordances,
            object_id=object_id,
            transform=transform,
        )
        self._assets[name] = entry

        # Register with Arena if available
        if ARENA_AVAILABLE and hasattr(self, 'asset_registry'):
            self.asset_registry.register(
                name=name,
                usd_path=usd_path,
                affordances=[getattr(AffordanceType, a.upper(), a) for a in affordances],
            )

    def get_asset(self, name: str) -> Optional[{class_name}AssetEntry]:
        """Get an asset by name."""
        return self._assets.get(name)

    def get_assets_by_affordance(self, affordance: str) -> list[{class_name}AssetEntry]:
        """Get all assets with a specific affordance."""
        return [
            asset for asset in self._assets.values()
            if affordance in asset.affordances
        ]

    @property
    def asset_names(self) -> list[str]:
        """Get all asset names."""
        return list(self._assets.keys())

    @property
    def all_affordances(self) -> set[str]:
        """Get all unique affordances in the scene."""
        affordances = set()
        for asset in self._assets.values():
            affordances.update(asset.affordances)
        return affordances

    def to_dict(self) -> dict[str, Any]:
        """Export scene configuration as dictionary."""
        return {{
            "scene_id": self.scene_id,
            "usd_path": self.usd_path,
            "environment_type": self.environment_type,
            "supported_embodiments": self.supported_embodiments,
            "assets": {{
                name: {{
                    "usd_path": asset.usd_path,
                    "affordances": asset.affordances,
                    "object_id": asset.object_id,
                }}
                for name, asset in self._assets.items()
            }},
        }}


# Convenience function for quick scene instantiation
def get_scene() -> {class_name}:
    """Get an instance of this scene."""
    return {class_name}()


# Export for Arena discovery
__all__ = ["{class_name}", "get_scene", "SCENE_ID", "SCENE_USD_PATH"]
'''

    def _generate_task_module(
        self,
        obj: dict[str, Any],
        affordance: AffordanceParams,
        task_spec: ArenaTaskSpec
    ) -> str:
        """Generate a task module for a specific object-affordance combination."""
        obj_id = obj.get("id", "unknown")
        obj_name = obj.get("name") or obj.get("category") or obj_id
        task_class = task_spec.task_type.value
        task_name = f"{obj_name}_{task_spec.display_name}".replace(" ", "")

        # Get task-specific parameters
        task_config = self.task_mapper.generate_task_config(task_spec, obj, affordance)
        params_json = json.dumps(task_config["params"], indent=8)

        return f'''"""
Arena Task: {task_spec.display_name} on {obj_name}
Auto-generated by BlueprintPipeline Arena Exporter

Object: {obj_name} (ID: {obj_id})
Affordance: {affordance.affordance_type.value}
Task Type: {task_class}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Isaac Lab-Arena imports
try:
    from isaaclab_arena.tasks import {task_class}
    from isaaclab_arena.core import TaskConfig
    ARENA_AVAILABLE = True
except ImportError:
    ARENA_AVAILABLE = False
    {task_class} = object
    TaskConfig = object


# Task configuration
TASK_NAME = "{task_name}"
OBJECT_ID = "{obj_id}"
OBJECT_NAME = "{obj_name}"
AFFORDANCE = "{affordance.affordance_type.value}"
MAX_STEPS = {task_spec.max_steps}
SUCCESS_THRESHOLD = {task_spec.success_threshold}

# Task parameters
TASK_PARAMS = {params_json}


@dataclass
class {task_name}Config:
    """Configuration for {task_name} task."""
    object_id: str = OBJECT_ID
    object_name: str = OBJECT_NAME
    max_steps: int = MAX_STEPS
    success_threshold: float = SUCCESS_THRESHOLD
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = dict(TASK_PARAMS)


class {task_name}Task({task_class} if ARENA_AVAILABLE else object):
    """
    {task_spec.display_name} task for {obj_name}.

    Description: {task_spec.description}
    Required Affordances: {[a.value for a in task_spec.required_affordances]}
    """

    def __init__(self, config: Optional[{task_name}Config] = None):
        self.config = config or {task_name}Config()

        if ARENA_AVAILABLE:
            super().__init__(
                object_name=self.config.object_name,
                **self.config.params
            )

    @classmethod
    def get_task_spec(cls) -> dict[str, Any]:
        """Get task specification for Arena registration."""
        return {{
            "name": TASK_NAME,
            "task_class": "{task_class}",
            "object_id": OBJECT_ID,
            "object_name": OBJECT_NAME,
            "affordance": AFFORDANCE,
            "max_steps": MAX_STEPS,
            "success_threshold": SUCCESS_THRESHOLD,
            "params": TASK_PARAMS,
            "required_affordances": {[a.value for a in task_spec.required_affordances]},
            "compatible_embodiments": {task_spec.compatible_embodiments},
        }}


def create_task(config: Optional[{task_name}Config] = None) -> {task_name}Task:
    """Factory function to create task instance."""
    return {task_name}Task(config)


# Export for Arena discovery
__all__ = ["{task_name}Task", "{task_name}Config", "create_task", "TASK_NAME"]
'''

    def _generate_tasks_init(self, tasks_dir: Path) -> str:
        """Generate __init__.py for tasks package."""
        task_files = [f.stem for f in tasks_dir.glob("*.py") if f.stem != "__init__"]

        imports = []
        exports = []
        for task_file in sorted(task_files):
            # Assuming task file names map to task classes
            class_name = self._to_class_name(task_file) + "Task"
            imports.append(f"from .{task_file} import {class_name}, create_task as create_{task_file}")
            exports.append(f'"{class_name}"')
            exports.append(f'"create_{task_file}"')

        imports_code = "\n".join(imports)
        exports_code = ",\n    ".join(exports)

        return f'''"""
Arena Tasks Package
Auto-generated by BlueprintPipeline Arena Exporter

This package contains all auto-generated tasks for the scene.
Tasks are generated based on detected object affordances.
"""

{imports_code}

__all__ = [
    {exports_code}
]


def get_all_tasks() -> list:
    """Get all task classes in this package."""
    return [
        {", ".join([self._to_class_name(f) + "Task" for f in sorted(task_files)])}
    ]


def get_task_registry() -> dict:
    """Get task registry mapping names to classes."""
    return {{
        task.__name__: task for task in get_all_tasks()
    }}
'''

    def _generate_hub_config(self, manifest: dict[str, Any], task_count: int) -> str:
        """Generate LeRobot Hub configuration YAML."""
        env_type = self.config.environment_type

        return f'''# LeRobot Environment Hub Configuration
# Auto-generated by BlueprintPipeline Arena Exporter
# Register this environment at: https://huggingface.co/spaces/lerobot/environment-hub

name: blueprint-{self.config.scene_id}
display_name: "Blueprint {env_type.title()} - {self.config.scene_id}"
source: BlueprintPipeline
version: "1.0.0"

description: |
  Auto-generated environment from BlueprintPipeline.
  Environment Type: {env_type}
  Task Count: {task_count}

# Scene configuration
scene:
  id: {self.config.scene_id}
  usd_path: {self.config.scene_path}
  environment_type: {env_type}

# Supported robot embodiments
embodiments:
{self._format_yaml_list(self.config.supported_embodiments, indent=2)}

# Available tasks (auto-detected from affordances)
tasks:
  count: {task_count}
  categories:
    - manipulation
    - articulation
    - pick_place

# Hugging Face repository configuration
huggingface:
  namespace: {self.config.hub_namespace}
  repo_id: {self.config.hub_namespace}/{self.config.scene_id}
  visibility: public

# Metadata
metadata:
  generated_at: "{datetime.utcnow().isoformat()}"
  generator: "BlueprintPipeline Arena Exporter"
  arena_version: "0.1.0"
'''

    def _affordance_to_dict(self, aff: AffordanceParams) -> dict[str, Any]:
        """Convert AffordanceParams to dictionary."""
        result = {
            "confidence": aff.confidence,
            "source": aff.source,
        }

        if aff.affordance_type == AffordanceType.OPENABLE:
            result["joint_type"] = aff.joint_type
            result["open_angle"] = aff.open_angle
            result["open_distance"] = aff.open_distance
            if aff.joint_name:
                result["joint_name"] = aff.joint_name

        elif aff.affordance_type == AffordanceType.GRASPABLE:
            result["grasp_width_range"] = list(aff.grasp_width_range)
            result["approach_direction"] = aff.grasp_approach_direction

        elif aff.affordance_type == AffordanceType.TURNABLE:
            result["rotation_axis"] = aff.rotation_axis
            result["rotation_range"] = list(aff.rotation_range)

        elif aff.affordance_type == AffordanceType.PRESSABLE:
            result["press_depth"] = aff.press_depth
            result["toggle"] = aff.toggle

        return result

    def _make_task_filename(self, obj: dict[str, Any], task_spec: ArenaTaskSpec) -> str:
        """Create a valid filename for a task module."""
        obj_name = obj.get("name") or obj.get("category") or obj.get("id", "unknown")
        task_name = task_spec.display_name.lower().replace(" ", "_")
        return f"{obj_name.lower().replace(' ', '_')}_{task_name}"

    def _to_class_name(self, name: str) -> str:
        """Convert string to PascalCase class name."""
        words = name.replace("-", "_").replace(".", "_").split("_")
        return "".join(word.capitalize() for word in words if word)

    def _format_yaml_list(self, items: list[str], indent: int = 0) -> str:
        """Format a list for YAML output."""
        prefix = " " * indent
        return "\n".join(f"{prefix}- {item}" for item in items)


def export_scene_to_arena(
    manifest: dict[str, Any],
    scene_id: str,
    output_dir: Path,
    scene_path: str = "scene.usda",
    environment_type: str = "generic",
    use_llm: bool = True,
) -> ArenaExportResult:
    """
    Convenience function to export a scene to Arena format.

    Args:
        manifest: Scene manifest dictionary
        scene_id: Scene identifier
        output_dir: Output directory path
        scene_path: Path to scene USD file
        environment_type: Environment classification
        use_llm: Whether to use Gemini for affordance detection

    Returns:
        ArenaExportResult with export status
    """
    config = ArenaExportConfig(
        scene_id=scene_id,
        scene_path=scene_path,
        output_dir=output_dir,
        environment_type=environment_type,
        use_llm_affordances=use_llm,
    )

    exporter = ArenaSceneExporter(config)
    return exporter.export(manifest)

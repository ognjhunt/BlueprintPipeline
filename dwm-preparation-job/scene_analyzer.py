#!/usr/bin/env python3
"""
Scene Analyzer for DWM - Extracts semantic meaning from scene manifests.

Uses Gemini 3.1 Pro with Grounded Search to analyze scenes and extract:
1. Object affordances (what actions can be performed on each object)
2. Semantic relationships between objects
3. Valid task sequences for the environment
4. Placement regions and interaction zones

This module provides the intelligence layer for dynamic, scene-aware DWM video generation.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.llm_client import create_llm_client, LLMResponse
    HAVE_LLM_CLIENT = True
except ImportError:
    HAVE_LLM_CLIENT = False
    create_llm_client = None


# =============================================================================
# Data Models
# =============================================================================


class EnvironmentType(str, Enum):
    """Types of environments for scene analysis."""
    KITCHEN = "kitchen"
    WAREHOUSE = "warehouse"
    GROCERY = "grocery"
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    OFFICE = "office"
    LAUNDRY = "laundry"
    LAB = "lab"
    LOADING_DOCK = "loading_dock"
    GENERIC = "generic"


class ObjectAffordance(str, Enum):
    """Actions that can be performed on objects."""
    # Manipulation affordances
    GRASP = "grasp"
    LIFT = "lift"
    PLACE = "place"
    PUSH = "push"
    PULL = "pull"
    ROTATE = "rotate"
    SLIDE = "slide"
    FLIP = "flip"
    POUR = "pour"

    # Articulation affordances
    OPEN = "open"
    CLOSE = "close"
    EXTEND = "extend"  # For drawers
    RETRACT = "retract"

    # Container affordances
    INSERT = "insert"
    REMOVE = "remove"
    FILL = "fill"
    EMPTY = "empty"

    # Specialized affordances
    STACK = "stack"
    HANG = "hang"
    FOLD = "fold"
    UNFOLD = "unfold"


@dataclass
class ObjectSemantics:
    """Semantic information about a scene object."""
    object_id: str
    category: str
    description: str
    sim_role: str

    # Extracted semantics
    affordances: List[ObjectAffordance] = field(default_factory=list)
    typical_locations: List[str] = field(default_factory=list)
    interaction_zones: List[str] = field(default_factory=list)
    related_objects: List[str] = field(default_factory=list)

    # Physical properties
    is_articulated: bool = False
    articulation_type: Optional[str] = None  # "revolute", "prismatic"
    is_container: bool = False
    is_surface: bool = False
    typical_height_m: Optional[float] = None

    # Task relevance
    relevant_tasks: List[str] = field(default_factory=list)
    priority_for_dwm: int = 0  # 0=low, 1=medium, 2=high


@dataclass
class InteractionZone:
    """A zone where interactions can occur."""
    zone_id: str
    zone_type: str  # "surface", "container_interior", "approach_area"
    parent_object_id: str
    description: str

    # Spatial info
    position: Optional[List[float]] = None
    size: Optional[List[float]] = None
    approach_direction: Optional[List[float]] = None

    # Affordances available in this zone
    available_affordances: List[ObjectAffordance] = field(default_factory=list)
    suitable_objects: List[str] = field(default_factory=list)


@dataclass
class TaskTemplate:
    """A template for a manipulation task in this scene."""
    task_id: str
    task_name: str
    description: str

    # Objects involved
    source_objects: List[str] = field(default_factory=list)
    target_objects: List[str] = field(default_factory=list)
    tool_objects: List[str] = field(default_factory=list)

    # Action sequence (high-level)
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)

    # Requirements
    requires_articulation: bool = False
    estimated_duration_seconds: float = 5.0
    difficulty: str = "medium"  # "easy", "medium", "hard"

    # DWM relevance
    dwm_clip_count: int = 1  # How many 49-frame clips needed
    priority: int = 1


@dataclass
class SceneAnalysisResult:
    """Complete analysis result for a scene."""
    scene_id: str
    environment_type: EnvironmentType

    # Analyzed content
    object_semantics: List[ObjectSemantics] = field(default_factory=list)
    interaction_zones: List[InteractionZone] = field(default_factory=list)
    task_templates: List[TaskTemplate] = field(default_factory=list)

    # Scene-level insights
    scene_summary: str = ""
    key_objects: List[str] = field(default_factory=list)
    recommended_policies: List[str] = field(default_factory=list)

    # Metadata
    analysis_confidence: float = 0.0
    llm_sources: List[Dict[str, str]] = field(default_factory=list)
    object_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_objects_by_affordance(self, affordance: ObjectAffordance) -> List[ObjectSemantics]:
        """Get objects that support a specific affordance."""
        return [obj for obj in self.object_semantics if affordance in obj.affordances]

    def get_articulated_objects(self) -> List[ObjectSemantics]:
        """Get all articulated objects."""
        return [obj for obj in self.object_semantics if obj.is_articulated]

    def get_container_objects(self) -> List[ObjectSemantics]:
        """Get all container objects."""
        return [obj for obj in self.object_semantics if obj.is_container]

    def get_surface_objects(self) -> List[ObjectSemantics]:
        """Get all surface objects."""
        return [obj for obj in self.object_semantics if obj.is_surface]


# =============================================================================
# Scene Analyzer
# =============================================================================


class SceneAnalyzer:
    """
    Analyzes scenes using Gemini 3.1 Pro with Grounded Search.

    Extracts semantic information to enable dynamic, scene-aware DWM video generation.

    Usage:
        analyzer = SceneAnalyzer()
        result = analyzer.analyze(manifest_path)

        # Get objects that can be grasped
        graspable = result.get_objects_by_affordance(ObjectAffordance.GRASP)

        # Get task templates for this scene
        tasks = result.task_templates
    """

    def __init__(self, verbose: bool = True):
        """Initialize the scene analyzer."""
        self.verbose = verbose
        self._client = None

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.verbose:
            print(f"[SCENE-ANALYZER] [{level}] {msg}")

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is None:
            if not HAVE_LLM_CLIENT:
                raise ImportError(
                    "LLM client not available. Install google-genai or openai package."
                )
            self._client = create_llm_client()
        return self._client

    def analyze(
        self,
        manifest_path: Path,
        policy_configs_path: Optional[Path] = None,
    ) -> SceneAnalysisResult:
        """
        Analyze a scene manifest to extract semantic information.

        Args:
            manifest_path: Path to scene_manifest.json
            policy_configs_path: Optional path to environment_policies.json

        Returns:
            SceneAnalysisResult with complete semantic analysis
        """
        manifest_path = Path(manifest_path)
        self.log(f"Analyzing scene: {manifest_path}")

        # Load manifest
        manifest = json.loads(manifest_path.read_text())
        scene_id = manifest.get("scene_id", "unknown")
        object_states = self._extract_object_states(manifest)

        # Load policy configs if available
        policy_configs = None
        if policy_configs_path is None:
            policy_configs_path = REPO_ROOT / "policy_configs" / "environment_policies.json"
        if policy_configs_path.exists():
            policy_configs = json.loads(policy_configs_path.read_text())

        # Detect environment type
        env_type = self._detect_environment_type(manifest)
        self.log(f"Detected environment: {env_type.value}")

        # Build analysis prompt
        prompt = self._build_analysis_prompt(manifest, env_type, policy_configs)

        # Call LLM
        try:
            client = self._get_client()
            response = client.generate(
                prompt=prompt,
                json_output=True,
                use_web_search=True,  # Enable grounded search
                temperature=0.3,
                max_tokens=16000,
            )

            analysis_data = response.parse_json()
            sources = response.sources

        except Exception as e:
            self.log(f"LLM analysis failed: {e}", "ERROR")
            # Return basic analysis without LLM
            return self._basic_analysis(manifest, scene_id, env_type)

        # Parse LLM response into result
        result = self._parse_analysis_response(
            analysis_data, manifest, scene_id, env_type, sources, object_states
        )

        self.log(f"Analysis complete: {len(result.object_semantics)} objects, "
                 f"{len(result.task_templates)} tasks")

        return result

    def _detect_environment_type(self, manifest: dict) -> EnvironmentType:
        """Detect environment type from manifest."""
        # Check scene metadata
        scene_type = manifest.get("scene", {}).get("type", "").lower()
        environment_hint = manifest.get("scene", {}).get("environment_type", "").lower()

        # Direct mapping
        type_map = {
            "kitchen": EnvironmentType.KITCHEN,
            "warehouse": EnvironmentType.WAREHOUSE,
            "grocery": EnvironmentType.GROCERY,
            "living_room": EnvironmentType.LIVING_ROOM,
            "bedroom": EnvironmentType.BEDROOM,
            "bathroom": EnvironmentType.BATHROOM,
            "office": EnvironmentType.OFFICE,
            "laundry": EnvironmentType.LAUNDRY,
            "lab": EnvironmentType.LAB,
            "loading_dock": EnvironmentType.LOADING_DOCK,
        }

        for key, env_type in type_map.items():
            if key in scene_type or key in environment_hint:
                return env_type

        # Infer from objects
        objects = manifest.get("objects", [])
        object_categories = set()
        for obj in objects:
            cat = obj.get("category", "").lower()
            object_categories.add(cat)

        # Kitchen indicators
        kitchen_indicators = {"refrigerator", "oven", "dishwasher", "sink", "counter", "stove"}
        if len(object_categories & kitchen_indicators) >= 2:
            return EnvironmentType.KITCHEN

        # Warehouse indicators
        warehouse_indicators = {"racking", "pallet", "forklift", "tote", "conveyor"}
        if len(object_categories & warehouse_indicators) >= 2:
            return EnvironmentType.WAREHOUSE

        # Bedroom indicators
        bedroom_indicators = {"bed", "dresser", "nightstand", "wardrobe", "closet"}
        if len(object_categories & bedroom_indicators) >= 2:
            return EnvironmentType.BEDROOM

        return EnvironmentType.GENERIC

    def _build_analysis_prompt(
        self,
        manifest: dict,
        env_type: EnvironmentType,
        policy_configs: Optional[dict]
    ) -> str:
        """Build the LLM prompt for scene analysis."""

        # Extract object summary
        objects_summary = []
        for obj in manifest.get("objects", []):
            obj_summary = {
                "id": obj.get("id"),
                "category": obj.get("category"),
                "sim_role": obj.get("sim_role"),
                "description": obj.get("semantics", {}).get("description", ""),
                "articulation_hint": obj.get("articulation_hint"),
                "position": obj.get("transform", {}).get("position"),
            }
            objects_summary.append(obj_summary)

        # Get environment-specific info
        env_info = ""
        valid_policies = []
        if policy_configs:
            env_config = policy_configs.get("environments", {}).get(env_type.value, {})
            env_info = env_config.get("description", "")
            valid_policies = env_config.get("default_policies", [])

        prompt = f"""You are an expert in robotics manipulation and scene understanding for synthetic data generation.

Analyze this scene and extract detailed semantic information for generating Dexterous World Model (DWM) training videos.

## Scene Information

Environment Type: {env_type.value}
{f"Environment Description: {env_info}" if env_info else ""}

Objects in Scene:
```json
{json.dumps(objects_summary, indent=2)}
```

{f"Valid Policies for this environment: {valid_policies}" if valid_policies else ""}

## Your Task

Analyze each object and the scene as a whole to extract:

1. **Object Affordances**: What actions can be performed on each object?
   - Consider: grasp, lift, place, push, pull, rotate, open, close, insert, remove, pour, stack, etc.
   - Be specific to the object type (e.g., dishwasher has open/close, drawers have extend/retract)

2. **Interaction Zones**: Where can manipulations occur?
   - Surfaces (counters, tables, shelves)
   - Container interiors (dishwasher rack, drawer, cabinet)
   - Approach areas (space in front of appliances)

3. **Task Templates**: What manipulation tasks make sense in this scene?
   - Consider the environment type and available objects
   - Include multi-step tasks (e.g., "load dishwasher" involves multiple actions)
   - Estimate difficulty and duration

4. **Object Relationships**: How do objects relate to each other?
   - What objects are typically used together?
   - What objects can contain other objects?
   - What are source/destination pairs for manipulation?

## Output Format

Return ONLY valid JSON in this exact structure:

{{
  "scene_summary": "Brief description of the scene and its manipulation potential",
  "key_objects": ["list", "of", "most", "important", "objects"],
  "recommended_policies": ["policy1", "policy2"],

  "object_semantics": [
    {{
      "object_id": "obj_0",
      "category": "dishwasher",
      "description": "Kitchen dishwasher with door and racks",
      "affordances": ["open", "close", "insert", "remove"],
      "typical_locations": ["kitchen", "near_sink"],
      "interaction_zones": ["door_handle", "upper_rack", "lower_rack"],
      "related_objects": ["dishes", "utensils", "sink"],
      "is_articulated": true,
      "articulation_type": "revolute",
      "is_container": true,
      "is_surface": false,
      "relevant_tasks": ["dish_loading", "articulated_access"],
      "priority_for_dwm": 2
    }}
  ],

  "interaction_zones": [
    {{
      "zone_id": "counter_surface_1",
      "zone_type": "surface",
      "parent_object_id": "obj_1",
      "description": "Main counter surface for dish staging",
      "available_affordances": ["place", "slide", "grasp"],
      "suitable_objects": ["dishes", "utensils", "containers"]
    }}
  ],

  "task_templates": [
    {{
      "task_id": "dish_loading_01",
      "task_name": "Load Dirty Dish into Dishwasher",
      "description": "Pick a dirty dish from counter and place in dishwasher rack",
      "source_objects": ["counter", "sink"],
      "target_objects": ["dishwasher_rack"],
      "action_sequence": [
        {{"action": "approach", "target": "counter", "description": "Walk toward dirty dishes"}},
        {{"action": "grasp", "target": "dish", "description": "Pick up dirty dish"}},
        {{"action": "lift", "target": "dish", "description": "Lift dish from counter"}},
        {{"action": "approach", "target": "dishwasher", "description": "Move to dishwasher"}},
        {{"action": "open", "target": "dishwasher_door", "description": "Open dishwasher door"}},
        {{"action": "insert", "target": "dishwasher_rack", "description": "Place dish on rack"}},
        {{"action": "release", "target": "dish", "description": "Release grip on dish"}}
      ],
      "requires_articulation": true,
      "estimated_duration_seconds": 8.0,
      "difficulty": "medium",
      "dwm_clip_count": 4
    }}
  ],

  "analysis_confidence": 0.85
}}

## Guidelines

1. **Be Comprehensive**: Include all objects that could be manipulated
2. **Be Specific**: Use concrete affordances, not vague descriptions
3. **Consider Physics**: Think about what's physically plausible
4. **Multi-Step Tasks**: Break complex tasks into action sequences
5. **DWM Relevance**: Prioritize objects/tasks useful for egocentric manipulation videos
6. **Clip Count**: Estimate how many 49-frame (~2 second) clips a task needs

Return ONLY the JSON, no additional text or explanation.
"""
        return prompt

    def _parse_analysis_response(
        self,
        data: dict,
        manifest: dict,
        scene_id: str,
        env_type: EnvironmentType,
        sources: List[Dict[str, str]],
        object_states: Dict[str, Dict[str, Any]],
    ) -> SceneAnalysisResult:
        """Parse LLM response into SceneAnalysisResult."""

        # Parse object semantics
        object_semantics = []
        for obj_data in data.get("object_semantics", []):
            affordances = []
            for aff_str in obj_data.get("affordances", []):
                try:
                    affordances.append(ObjectAffordance(aff_str))
                except ValueError:
                    pass  # Skip unknown affordances

            obj_sem = ObjectSemantics(
                object_id=obj_data.get("object_id", ""),
                category=obj_data.get("category", ""),
                description=obj_data.get("description", ""),
                sim_role=obj_data.get("sim_role", "static"),
                affordances=affordances,
                typical_locations=obj_data.get("typical_locations", []),
                interaction_zones=obj_data.get("interaction_zones", []),
                related_objects=obj_data.get("related_objects", []),
                is_articulated=obj_data.get("is_articulated", False),
                articulation_type=obj_data.get("articulation_type"),
                is_container=obj_data.get("is_container", False),
                is_surface=obj_data.get("is_surface", False),
                typical_height_m=obj_data.get("typical_height_m"),
                relevant_tasks=obj_data.get("relevant_tasks", []),
                priority_for_dwm=obj_data.get("priority_for_dwm", 0),
            )
            object_semantics.append(obj_sem)

        # Parse interaction zones
        interaction_zones = []
        for zone_data in data.get("interaction_zones", []):
            affordances = []
            for aff_str in zone_data.get("available_affordances", []):
                try:
                    affordances.append(ObjectAffordance(aff_str))
                except ValueError:
                    pass

            zone = InteractionZone(
                zone_id=zone_data.get("zone_id", ""),
                zone_type=zone_data.get("zone_type", "surface"),
                parent_object_id=zone_data.get("parent_object_id", ""),
                description=zone_data.get("description", ""),
                position=zone_data.get("position"),
                size=zone_data.get("size"),
                approach_direction=zone_data.get("approach_direction"),
                available_affordances=affordances,
                suitable_objects=zone_data.get("suitable_objects", []),
            )
            interaction_zones.append(zone)

        # Parse task templates
        task_templates = []
        for task_data in data.get("task_templates", []):
            task = TaskTemplate(
                task_id=task_data.get("task_id", ""),
                task_name=task_data.get("task_name", ""),
                description=task_data.get("description", ""),
                source_objects=task_data.get("source_objects", []),
                target_objects=task_data.get("target_objects", []),
                tool_objects=task_data.get("tool_objects", []),
                action_sequence=task_data.get("action_sequence", []),
                requires_articulation=task_data.get("requires_articulation", False),
                estimated_duration_seconds=task_data.get("estimated_duration_seconds", 5.0),
                difficulty=task_data.get("difficulty", "medium"),
                dwm_clip_count=task_data.get("dwm_clip_count", 1),
                priority=task_data.get("priority", 1),
            )
            task_templates.append(task)

        return SceneAnalysisResult(
            scene_id=scene_id,
            environment_type=env_type,
            object_semantics=object_semantics,
            interaction_zones=interaction_zones,
            task_templates=task_templates,
            scene_summary=data.get("scene_summary", ""),
            key_objects=data.get("key_objects", []),
            recommended_policies=data.get("recommended_policies", []),
            analysis_confidence=data.get("analysis_confidence", 0.0),
            llm_sources=sources,
            object_states=object_states,
        )

    def _basic_analysis(
        self,
        manifest: dict,
        scene_id: str,
        env_type: EnvironmentType
    ) -> SceneAnalysisResult:
        """Provide basic analysis without LLM."""

        object_semantics = []
        for obj in manifest.get("objects", []):
            sim_role = obj.get("sim_role", "static")
            category = obj.get("category", "").lower()

            # Basic affordance mapping
            affordances = []
            if sim_role in ["manipulable_object", "clutter"]:
                affordances = [
                    ObjectAffordance.GRASP,
                    ObjectAffordance.LIFT,
                    ObjectAffordance.PLACE,
                ]
            elif sim_role in ["articulated_furniture", "articulated_appliance"]:
                affordances = [
                    ObjectAffordance.OPEN,
                    ObjectAffordance.CLOSE,
                ]
                if "drawer" in category:
                    affordances.extend([ObjectAffordance.EXTEND, ObjectAffordance.RETRACT])

            obj_sem = ObjectSemantics(
                object_id=obj.get("id", ""),
                category=category,
                description=obj.get("semantics", {}).get("description", ""),
                sim_role=sim_role,
                affordances=affordances,
                is_articulated=sim_role in ["articulated_furniture", "articulated_appliance"],
                is_container="container" in category or "cabinet" in category,
                is_surface="counter" in category or "table" in category or "shelf" in category,
                priority_for_dwm=2 if sim_role != "static" else 0,
            )
            object_semantics.append(obj_sem)

        return SceneAnalysisResult(
            scene_id=scene_id,
            environment_type=env_type,
            object_semantics=object_semantics,
            scene_summary="Basic analysis (LLM unavailable)",
            analysis_confidence=0.3,
            object_states=self._extract_object_states(manifest),
        )

    def _extract_object_states(self, manifest: dict) -> Dict[str, Dict[str, Any]]:
        """Extract static object poses and articulation hints from manifest."""
        states: Dict[str, Dict[str, Any]] = {}

        for obj in manifest.get("objects", []):
            obj_id = obj.get("id") or obj.get("object_id")
            if not obj_id:
                continue

            transform = obj.get("transform", {})
            pos = transform.get("position", {})
            rot = transform.get("rotation", {})

            position = [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
            rotation = rot if rot else None

            state: Dict[str, Any] = {"position": position}
            if rotation:
                state["rotation"] = rotation

            articulation = obj.get("articulation_state") or obj.get("articulation")
            if articulation is not None:
                state["articulation"] = articulation

            states[obj_id] = state

        return states


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_scene(manifest_path: Path, verbose: bool = True) -> SceneAnalysisResult:
    """Convenience function to analyze a scene."""
    analyzer = SceneAnalyzer(verbose=verbose)
    return analyzer.analyze(manifest_path)


def get_graspable_objects(result: SceneAnalysisResult) -> List[ObjectSemantics]:
    """Get all objects that can be grasped."""
    return result.get_objects_by_affordance(ObjectAffordance.GRASP)


def get_articulated_objects(result: SceneAnalysisResult) -> List[ObjectSemantics]:
    """Get all articulated objects."""
    return result.get_articulated_objects()


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze scene for DWM video generation")
    parser.add_argument("manifest_path", type=Path, help="Path to scene_manifest.json")
    parser.add_argument("--output", "-o", type=Path, help="Output path for analysis JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output")

    args = parser.parse_args()

    result = analyze_scene(args.manifest_path, verbose=not args.quiet)

    # Convert to JSON-serializable dict
    output_data = {
        "scene_id": result.scene_id,
        "environment_type": result.environment_type.value,
        "scene_summary": result.scene_summary,
        "key_objects": result.key_objects,
        "recommended_policies": result.recommended_policies,
        "analysis_confidence": result.analysis_confidence,
        "object_count": len(result.object_semantics),
        "task_count": len(result.task_templates),
        "zone_count": len(result.interaction_zones),
        "object_semantics": [
            {
                "object_id": obj.object_id,
                "category": obj.category,
                "affordances": [a.value for a in obj.affordances],
                "is_articulated": obj.is_articulated,
                "is_container": obj.is_container,
                "priority_for_dwm": obj.priority_for_dwm,
            }
            for obj in result.object_semantics
        ],
        "task_templates": [
            {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "dwm_clip_count": task.dwm_clip_count,
                "action_count": len(task.action_sequence),
            }
            for task in result.task_templates
        ],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"Analysis saved to: {args.output}")
    else:
        print(json.dumps(output_data, indent=2))

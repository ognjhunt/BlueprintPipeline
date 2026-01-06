"""
Affordance Detection and Tagging System for Isaac Lab-Arena Integration.

Affordances represent standardized object interactions that enable task generalization.
Instead of writing task-specific code for each object, Arena uses affordance labels
to auto-generate compatible tasks.

This module provides:
1. AffordanceType enum with all standard Arena affordances
2. AffordanceDetector that infers affordances from object metadata
3. AffordanceParams that encode interaction-specific parameters
4. Integration with Gemini for intelligent affordance detection

Arena Affordance Standard (v0.1):
- Openable: Doors, drawers, lids, etc. (rotational or linear)
- Pressable: Buttons, switches, touchscreens
- Turnable: Knobs, dials, valves
- Graspable: Objects that can be picked up
- Insertable: Objects that fit into receptacles
- Stackable: Objects that can be stacked
- Pourable: Containers with liquid/granular contents
- Slidable: Objects that slide along surfaces
"""

import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# LLM client for intelligent affordance detection
try:
    from google import genai
    from google.genai import types
    HAVE_GEMINI = True
except ImportError:
    genai = None
    types = None
    HAVE_GEMINI = False


class AffordanceType(str, Enum):
    """
    Standard affordance types compatible with Isaac Lab-Arena.

    Each affordance type maps to a set of Arena tasks that can
    automatically be generated for objects with that affordance.
    """
    # Articulation affordances
    OPENABLE = "Openable"           # Doors, drawers, lids (rotational or linear)
    TURNABLE = "Turnable"           # Knobs, dials, valves (continuous rotation)
    PRESSABLE = "Pressable"         # Buttons, switches, keypads
    SLIDABLE = "Slidable"           # Sliders, handles that slide

    # Manipulation affordances
    GRASPABLE = "Graspable"         # Can be picked up by gripper
    INSERTABLE = "Insertable"       # Can be inserted into receptacles
    STACKABLE = "Stackable"         # Can be stacked on similar objects
    HANGABLE = "Hangable"           # Can be hung on hooks/racks

    # Container affordances
    POURABLE = "Pourable"           # Containers with pourable contents
    FILLABLE = "Fillable"           # Can receive poured contents
    CONTAINABLE = "Containable"     # Can hold other objects

    # Surface affordances
    PLACEABLE = "Placeable"         # Surfaces that accept placed objects
    SUPPORTABLE = "Supportable"     # Can support weight of other objects

    # Deformable affordances
    FOLDABLE = "Foldable"           # Cloth, paper, flexible items
    SQUEEZABLE = "Squeezable"       # Soft, deformable objects

    # Tool affordances
    CUTTABLE = "Cuttable"           # Can be cut (food, paper)
    WRITABLE = "Writable"           # Writing surfaces/implements


@dataclass
class AffordanceParams:
    """
    Parameters that define how an affordance interaction works.

    These parameters are used by Arena to auto-generate task objectives,
    success criteria, and observation/action spaces.
    """
    affordance_type: AffordanceType

    # For Openable affordances
    joint_name: Optional[str] = None          # USD joint prim path
    joint_type: str = "revolute"              # revolute, prismatic
    open_angle: float = 1.57                  # radians for revolute
    open_distance: float = 0.3                # meters for prismatic
    close_angle: float = 0.0
    close_distance: float = 0.0

    # For Turnable affordances
    rotation_axis: str = "z"                  # x, y, z
    rotation_range: tuple[float, float] = (0.0, 6.28)  # min/max radians
    discrete_positions: Optional[list[float]] = None   # for dials with stops

    # For Pressable affordances
    button_ids: list[str] = field(default_factory=list)  # prim paths
    press_depth: float = 0.01                            # meters
    toggle: bool = False                                 # momentary vs toggle

    # For Graspable affordances
    grasp_width_range: tuple[float, float] = (0.02, 0.10)  # gripper opening
    grasp_approach_direction: str = "top"                   # top, side, any
    preferred_grasp_points: list[dict] = field(default_factory=list)

    # For Insertable affordances
    insertion_axis: str = "z"                 # axis of insertion
    insertion_depth: float = 0.05             # required depth
    receptacle_tolerance: float = 0.005       # position tolerance

    # For Stackable affordances
    stack_axis: str = "z"                     # vertical stacking axis
    max_stack_height: int = 5
    requires_alignment: bool = True

    # For Pourable/Fillable
    pour_axis: str = "-z"                     # direction of pour
    capacity_liters: float = 0.5
    current_fill_ratio: float = 0.0

    # Metadata
    confidence: float = 1.0                   # detection confidence
    source: str = "detected"                  # detected, manual, llm
    notes: str = ""


# Registry mapping sim_role + category patterns to likely affordances
AFFORDANCE_REGISTRY: dict[str, list[dict[str, Any]]] = {
    # Articulated appliances
    "articulated_appliance": [
        {"pattern": ["microwave", "oven", "dishwasher", "washer", "dryer", "refrigerator", "fridge"],
         "affordances": [AffordanceType.OPENABLE, AffordanceType.PRESSABLE]},
        {"pattern": ["stove", "range", "cooktop"],
         "affordances": [AffordanceType.TURNABLE, AffordanceType.PRESSABLE]},
    ],
    # Articulated furniture
    "articulated_furniture": [
        {"pattern": ["cabinet", "cupboard", "wardrobe", "closet"],
         "affordances": [AffordanceType.OPENABLE, AffordanceType.CONTAINABLE]},
        {"pattern": ["drawer", "dresser", "nightstand"],
         "affordances": [AffordanceType.OPENABLE, AffordanceType.CONTAINABLE]},
        {"pattern": ["door"],
         "affordances": [AffordanceType.OPENABLE]},
    ],
    # Manipulable objects
    "manipulable_object": [
        {"pattern": ["cup", "mug", "glass", "bottle", "can", "jar"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.POURABLE, AffordanceType.FILLABLE]},
        {"pattern": ["plate", "bowl", "dish", "pan", "pot"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.CONTAINABLE, AffordanceType.STACKABLE]},
        {"pattern": ["box", "carton", "package", "container"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.STACKABLE, AffordanceType.CONTAINABLE]},
        {"pattern": ["book", "magazine", "folder"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.STACKABLE, AffordanceType.OPENABLE]},
        {"pattern": ["knife", "fork", "spoon", "utensil", "spatula"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.INSERTABLE]},
        {"pattern": ["pen", "pencil", "marker"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.INSERTABLE, AffordanceType.WRITABLE]},
        {"pattern": ["shirt", "pants", "cloth", "towel", "fabric"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.FOLDABLE, AffordanceType.HANGABLE]},
        {"pattern": ["fruit", "vegetable", "food"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.CUTTABLE]},
        {"pattern": ["remote", "phone", "device"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.PRESSABLE]},
        {"pattern": ["tube", "test_tube", "vial", "pipette"],
         "affordances": [AffordanceType.GRASPABLE, AffordanceType.INSERTABLE]},
    ],
    # Interactive objects (non-movable but interactable)
    "interactive": [
        {"pattern": ["switch", "button", "panel"],
         "affordances": [AffordanceType.PRESSABLE]},
        {"pattern": ["knob", "dial", "valve"],
         "affordances": [AffordanceType.TURNABLE]},
        {"pattern": ["lever", "handle"],
         "affordances": [AffordanceType.OPENABLE, AffordanceType.TURNABLE]},
        {"pattern": ["slider", "fader"],
         "affordances": [AffordanceType.SLIDABLE]},
    ],
    # Static surfaces
    "static": [
        {"pattern": ["table", "desk", "counter", "shelf", "bench"],
         "affordances": [AffordanceType.PLACEABLE, AffordanceType.SUPPORTABLE]},
        {"pattern": ["hook", "rack", "hanger"],
         "affordances": [AffordanceType.HANGABLE]},
        {"pattern": ["slot", "receptacle", "holder"],
         "affordances": [AffordanceType.INSERTABLE]},
    ],
}


def _match_pattern(text: str, patterns: list[str]) -> bool:
    """Check if text matches any pattern (case-insensitive substring)."""
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)


def detect_affordances_heuristic(obj: dict[str, Any]) -> list[AffordanceParams]:
    """
    Detect affordances using rule-based heuristics.

    This is the fast path when Gemini is unavailable. It uses the
    AFFORDANCE_REGISTRY to match object properties to likely affordances.
    """
    affordances: list[AffordanceParams] = []

    sim_role = obj.get("sim_role", "unknown")
    category = obj.get("category", "") or obj.get("name", "") or obj.get("id", "")
    description = obj.get("description", "")
    search_text = f"{category} {description}"

    # Check registry for sim_role
    role_rules = AFFORDANCE_REGISTRY.get(sim_role, [])
    for rule in role_rules:
        if _match_pattern(search_text, rule["pattern"]):
            for aff_type in rule["affordances"]:
                params = AffordanceParams(
                    affordance_type=aff_type,
                    confidence=0.7,
                    source="heuristic",
                    notes=f"Matched pattern in {sim_role}"
                )

                # Populate type-specific defaults
                if aff_type == AffordanceType.OPENABLE:
                    # Infer joint type from category
                    if any(kw in search_text.lower() for kw in ["drawer", "slide"]):
                        params.joint_type = "prismatic"
                        params.open_distance = 0.4
                    else:
                        params.joint_type = "revolute"
                        params.open_angle = 1.57

                    # Try to find joint name from articulation data
                    articulation = obj.get("articulation", {})
                    if articulation.get("physx_endpoint"):
                        params.joint_name = articulation["physx_endpoint"]

                elif aff_type == AffordanceType.GRASPABLE:
                    # Estimate grasp width from dimensions
                    dims = obj.get("dimensions_est", {})
                    width = dims.get("width", 0.1)
                    depth = dims.get("depth", 0.1)
                    min_dim = min(width, depth)
                    params.grasp_width_range = (
                        max(0.01, min_dim * 0.5),
                        min(0.15, min_dim * 1.2)
                    )

                affordances.append(params)
            break  # Use first matching rule

    # Fallback: if manipulable but no affordances found, assume graspable
    if not affordances and sim_role in ["manipulable_object", "clutter"]:
        affordances.append(AffordanceParams(
            affordance_type=AffordanceType.GRASPABLE,
            confidence=0.5,
            source="fallback",
            notes="Default graspable for manipulable objects"
        ))

    return affordances


def _make_affordance_prompt(obj: dict[str, Any]) -> str:
    """Create prompt for Gemini affordance detection."""

    # Compact object description
    obj_info = {
        "id": obj.get("id"),
        "name": obj.get("name") or obj.get("category"),
        "category": obj.get("category"),
        "description": obj.get("description"),
        "sim_role": obj.get("sim_role"),
        "dimensions": obj.get("dimensions_est"),
        "articulation": obj.get("articulation"),
    }

    affordance_list = "\n".join([f"- {a.value}: {a.name}" for a in AffordanceType])

    return f"""
You are analyzing an object for robot manipulation in NVIDIA Isaac Lab-Arena.
Your task is to identify which AFFORDANCES this object has.

An affordance describes HOW a robot can interact with the object.

Available affordance types:
{affordance_list}

Object information:
{json.dumps(obj_info, indent=2)}

For each applicable affordance, provide:
1. The affordance type (exact name from list above)
2. Confidence (0.0-1.0)
3. Type-specific parameters

Return ONLY valid JSON (no markdown, no comments) with this structure:
{{
    "affordances": [
        {{
            "type": "Openable",
            "confidence": 0.9,
            "joint_type": "revolute|prismatic",
            "open_angle": 1.57,
            "open_distance": 0.3,
            "notes": "Door opens outward"
        }},
        {{
            "type": "Graspable",
            "confidence": 0.8,
            "grasp_width_min": 0.02,
            "grasp_width_max": 0.08,
            "approach_direction": "top|side|any",
            "notes": "Handle is graspable"
        }}
    ]
}}

Be specific and accurate. Only include affordances that genuinely apply.
"""


def detect_affordances_llm(
    obj: dict[str, Any],
    client: Optional["genai.Client"] = None
) -> list[AffordanceParams]:
    """
    Detect affordances using Gemini LLM for more accurate inference.

    This path provides higher-quality affordance detection by understanding
    context and object semantics beyond simple pattern matching.
    """
    if client is None or not HAVE_GEMINI:
        return detect_affordances_heuristic(obj)

    try:
        prompt = _make_affordance_prompt(obj)

        model_name = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=config,
        )

        raw = (response.text or "").strip()
        # Strip code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        data = json.loads(raw)
        if not isinstance(data, dict) or "affordances" not in data:
            raise ValueError("Invalid response structure")

        affordances: list[AffordanceParams] = []
        for aff_data in data["affordances"]:
            aff_type_str = aff_data.get("type", "")
            try:
                aff_type = AffordanceType(aff_type_str)
            except ValueError:
                continue  # Skip unknown affordance types

            params = AffordanceParams(
                affordance_type=aff_type,
                confidence=float(aff_data.get("confidence", 0.8)),
                source="llm",
                notes=aff_data.get("notes", "")
            )

            # Populate type-specific params from LLM response
            if aff_type == AffordanceType.OPENABLE:
                params.joint_type = aff_data.get("joint_type", "revolute")
                params.open_angle = float(aff_data.get("open_angle", 1.57))
                params.open_distance = float(aff_data.get("open_distance", 0.3))

            elif aff_type == AffordanceType.GRASPABLE:
                params.grasp_width_range = (
                    float(aff_data.get("grasp_width_min", 0.02)),
                    float(aff_data.get("grasp_width_max", 0.10))
                )
                params.grasp_approach_direction = aff_data.get("approach_direction", "any")

            elif aff_type == AffordanceType.TURNABLE:
                params.rotation_axis = aff_data.get("rotation_axis", "z")
                rot_min = float(aff_data.get("rotation_min", 0.0))
                rot_max = float(aff_data.get("rotation_max", 6.28))
                params.rotation_range = (rot_min, rot_max)

            elif aff_type == AffordanceType.PRESSABLE:
                params.press_depth = float(aff_data.get("press_depth", 0.01))
                params.toggle = bool(aff_data.get("toggle", False))

            elif aff_type == AffordanceType.INSERTABLE:
                params.insertion_axis = aff_data.get("insertion_axis", "z")
                params.insertion_depth = float(aff_data.get("insertion_depth", 0.05))

            affordances.append(params)

        return affordances if affordances else detect_affordances_heuristic(obj)

    except Exception as e:
        print(f"[AFFORDANCES] LLM detection failed: {e}, falling back to heuristics", file=sys.stderr)
        return detect_affordances_heuristic(obj)


class AffordanceDetector:
    """
    Main affordance detection class that manages detection strategy.

    Usage:
        detector = AffordanceDetector()
        affordances = detector.detect(obj_dict)

        # Or with Gemini for higher accuracy
        detector = AffordanceDetector(use_llm=True)
        affordances = detector.detect(obj_dict)
    """

    def __init__(self, use_llm: bool = False):
        """
        Initialize affordance detector.

        Args:
            use_llm: Whether to use Gemini for intelligent detection.
                     Falls back to heuristics if Gemini unavailable.
        """
        self.use_llm = use_llm
        self.client = None

        if use_llm and HAVE_GEMINI:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)

    def detect(self, obj: dict[str, Any]) -> list[AffordanceParams]:
        """
        Detect affordances for an object.

        Args:
            obj: Object dictionary from scene manifest

        Returns:
            List of detected affordances with parameters
        """
        if self.use_llm and self.client:
            return detect_affordances_llm(obj, self.client)
        return detect_affordances_heuristic(obj)

    def detect_batch(self, objects: list[dict[str, Any]]) -> dict[str, list[AffordanceParams]]:
        """
        Detect affordances for multiple objects.

        Args:
            objects: List of object dictionaries

        Returns:
            Dict mapping object IDs to their affordances
        """
        results = {}
        for obj in objects:
            obj_id = obj.get("id", "unknown")
            results[obj_id] = self.detect(obj)
        return results

    def to_manifest_format(self, affordances: list[AffordanceParams]) -> dict[str, Any]:
        """
        Convert affordances to scene manifest format.

        Returns format suitable for scene_manifest.json objects:
        {
            "affordances": ["Openable", "Pressable"],
            "affordance_params": {
                "Openable": {"joint_type": "revolute", "open_angle": 1.57, ...},
                "Pressable": {"button_ids": [...], ...}
            }
        }
        """
        affordance_types = []
        affordance_params = {}

        for aff in affordances:
            aff_name = aff.affordance_type.value
            affordance_types.append(aff_name)

            # Build params dict based on affordance type
            params: dict[str, Any] = {
                "confidence": aff.confidence,
                "source": aff.source,
            }

            if aff.affordance_type == AffordanceType.OPENABLE:
                params["joint_type"] = aff.joint_type
                if aff.joint_type == "revolute":
                    params["open_angle"] = aff.open_angle
                    params["close_angle"] = aff.close_angle
                else:
                    params["open_distance"] = aff.open_distance
                    params["close_distance"] = aff.close_distance
                if aff.joint_name:
                    params["joint_name"] = aff.joint_name

            elif aff.affordance_type == AffordanceType.GRASPABLE:
                params["grasp_width_range"] = list(aff.grasp_width_range)
                params["approach_direction"] = aff.grasp_approach_direction
                if aff.preferred_grasp_points:
                    params["preferred_grasp_points"] = aff.preferred_grasp_points

            elif aff.affordance_type == AffordanceType.TURNABLE:
                params["rotation_axis"] = aff.rotation_axis
                params["rotation_range"] = list(aff.rotation_range)
                if aff.discrete_positions:
                    params["discrete_positions"] = aff.discrete_positions

            elif aff.affordance_type == AffordanceType.PRESSABLE:
                params["press_depth"] = aff.press_depth
                params["toggle"] = aff.toggle
                if aff.button_ids:
                    params["button_ids"] = aff.button_ids

            elif aff.affordance_type == AffordanceType.INSERTABLE:
                params["insertion_axis"] = aff.insertion_axis
                params["insertion_depth"] = aff.insertion_depth
                params["receptacle_tolerance"] = aff.receptacle_tolerance

            elif aff.affordance_type == AffordanceType.STACKABLE:
                params["stack_axis"] = aff.stack_axis
                params["max_stack_height"] = aff.max_stack_height
                params["requires_alignment"] = aff.requires_alignment

            elif aff.affordance_type in (AffordanceType.POURABLE, AffordanceType.FILLABLE):
                params["pour_axis"] = aff.pour_axis
                params["capacity_liters"] = aff.capacity_liters

            if aff.notes:
                params["notes"] = aff.notes

            affordance_params[aff_name] = params

        return {
            "affordances": affordance_types,
            "affordance_params": affordance_params,
        }


def detect_affordances(
    obj: dict[str, Any],
    use_llm: bool = False
) -> dict[str, Any]:
    """
    Convenience function to detect affordances and return manifest format.

    Args:
        obj: Object dictionary from scene manifest
        use_llm: Whether to use Gemini for detection

    Returns:
        Dict with 'affordances' and 'affordance_params' keys
    """
    detector = AffordanceDetector(use_llm=use_llm)
    affordances = detector.detect(obj)
    return detector.to_manifest_format(affordances)

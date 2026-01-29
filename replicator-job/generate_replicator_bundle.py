#!/usr/bin/env python3
"""
Replicator Bundle Generator for Isaac Sim Synthetic Data Generation.

This script analyzes a completed scene and generates:
1. Placement regions as USD layers (sink_region, counter_region, etc.)
2. Policy-specific Replicator Python scripts
3. Variation asset manifests (dirty dishes, groceries, clothes, etc.)
4. Configuration files for different training policies

The output bundle is ready to be loaded into Isaac Sim for domain randomization
and synthetic data generation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Unified LLM client supporting Gemini + OpenAI
try:
    from tools.llm_client import create_llm_client, LLMProvider, LLMClient
    HAVE_LLM_CLIENT = True
except ImportError:
    HAVE_LLM_CLIENT = False
    create_llm_client = None
    LLMProvider = None
    LLMClient = None

from tools.asset_catalog import AssetCatalogClient
from tools.scene_manifest.loader import load_manifest_or_scene_assets

# ============================================================================
# Constants and Configuration
# ============================================================================

GCS_ROOT = Path("/mnt/gcs")

# Environment archetype definitions
class EnvironmentType(str, Enum):
    KITCHEN = "kitchen"
    GROCERY = "grocery"
    WAREHOUSE = "warehouse"
    LOADING_DOCK = "loading_dock"
    LAB = "lab"
    OFFICE = "office"
    UTILITY_ROOM = "utility_room"
    HOME_LAUNDRY = "home_laundry"
    BEDROOM = "bedroom"
    LIVING_ROOM = "living_room"
    BATHROOM = "bathroom"
    GENERIC = "generic"


# Policy target definitions
class PolicyTarget(str, Enum):
    DEXTEROUS_PICK_PLACE = "dexterous_pick_place"
    ARTICULATED_ACCESS = "articulated_access"
    PANEL_INTERACTION = "panel_interaction"
    MIXED_SKU_LOGISTICS = "mixed_sku_logistics"
    PRECISION_INSERTION = "precision_insertion"
    LAUNDRY_SORTING = "laundry_sorting"
    DISH_LOADING = "dish_loading"
    GROCERY_STOCKING = "grocery_stocking"
    TABLE_CLEARING = "table_clearing"
    DRAWER_MANIPULATION = "drawer_manipulation"
    DOOR_MANIPULATION = "door_manipulation"
    KNOB_MANIPULATION = "knob_manipulation"
    GENERAL_MANIPULATION = "general_manipulation"


# Mapping from scene_type to environment type
SCENE_TYPE_TO_ENVIRONMENT = {
    "kitchen": EnvironmentType.KITCHEN,
    "grocery": EnvironmentType.GROCERY,
    "warehouse": EnvironmentType.WAREHOUSE,
    "loading_dock": EnvironmentType.LOADING_DOCK,
    "lab": EnvironmentType.LAB,
    "office": EnvironmentType.OFFICE,
    "utility_room": EnvironmentType.UTILITY_ROOM,
    "laundry": EnvironmentType.HOME_LAUNDRY,
    "laundry_room": EnvironmentType.HOME_LAUNDRY,
    "bedroom": EnvironmentType.BEDROOM,
    "living_room": EnvironmentType.LIVING_ROOM,
    "bathroom": EnvironmentType.BATHROOM,
}

# Default policies per environment type
ENVIRONMENT_DEFAULT_POLICIES = {
    EnvironmentType.KITCHEN: [
        PolicyTarget.DISH_LOADING,
        PolicyTarget.TABLE_CLEARING,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.ARTICULATED_ACCESS,
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
    ],
    EnvironmentType.GROCERY: [
        PolicyTarget.GROCERY_STOCKING,
        PolicyTarget.MIXED_SKU_LOGISTICS,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.ARTICULATED_ACCESS,
    ],
    EnvironmentType.WAREHOUSE: [
        PolicyTarget.MIXED_SKU_LOGISTICS,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.GENERAL_MANIPULATION,
    ],
    EnvironmentType.LOADING_DOCK: [
        PolicyTarget.MIXED_SKU_LOGISTICS,
        PolicyTarget.GENERAL_MANIPULATION,
    ],
    EnvironmentType.LAB: [
        PolicyTarget.PRECISION_INSERTION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.ARTICULATED_ACCESS,
        PolicyTarget.DRAWER_MANIPULATION,
    ],
    EnvironmentType.OFFICE: [
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.PANEL_INTERACTION,
    ],
    EnvironmentType.UTILITY_ROOM: [
        PolicyTarget.PANEL_INTERACTION,
        PolicyTarget.KNOB_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.ARTICULATED_ACCESS,
    ],
    EnvironmentType.HOME_LAUNDRY: [
        PolicyTarget.LAUNDRY_SORTING,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.KNOB_MANIPULATION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
    ],
    EnvironmentType.BEDROOM: [
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.LAUNDRY_SORTING,
    ],
    EnvironmentType.LIVING_ROOM: [
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.GENERAL_MANIPULATION,
    ],
    EnvironmentType.BATHROOM: [
        PolicyTarget.DRAWER_MANIPULATION,
        PolicyTarget.DOOR_MANIPULATION,
        PolicyTarget.DEXTEROUS_PICK_PLACE,
    ],
    EnvironmentType.GENERIC: [
        PolicyTarget.DEXTEROUS_PICK_PLACE,
        PolicyTarget.GENERAL_MANIPULATION,
    ],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlacementRegion:
    """Defines a placement region for object scattering."""
    name: str
    description: str
    surface_type: str  # "horizontal", "vertical", "volume"
    parent_object_id: Optional[str] = None
    position: Optional[List[float]] = None  # [x, y, z] in meters
    size: Optional[List[float]] = None  # [width, depth, height] in meters
    rotation: Optional[List[float]] = None  # [rx, ry, rz] in degrees
    semantic_tags: List[str] = field(default_factory=list)
    suitable_for: List[str] = field(default_factory=list)  # Object categories


@dataclass
class VariationAsset:
    """Defines a variation asset needed for domain randomization."""
    name: str
    category: str
    description: str
    semantic_class: str
    priority: str  # "required", "recommended", "optional"
    source_hint: Optional[str] = None  # "generate", "library", "simready"
    example_variants: List[str] = field(default_factory=list)
    physics_hints: Dict[str, Any] = field(default_factory=dict)
    asset_id: Optional[str] = None
    asset_path: Optional[str] = None
    thumbnail_uri: Optional[str] = None


@dataclass
class RandomizerConfig:
    """Configuration for a specific randomizer."""
    name: str
    enabled: bool = True
    frequency: str = "per_frame"  # "per_frame", "per_episode", "once"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    """Complete configuration for a training policy."""
    policy_id: str
    policy_name: str
    policy_target: PolicyTarget
    description: str
    placement_regions: List[PlacementRegion] = field(default_factory=list)
    variation_assets: List[VariationAsset] = field(default_factory=list)
    randomizers: List[RandomizerConfig] = field(default_factory=list)
    capture_config: Dict[str, Any] = field(default_factory=dict)
    scene_modifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplicatorBundle:
    """Complete bundle of Replicator configurations for a scene."""
    scene_id: str
    environment_type: EnvironmentType
    scene_type: str
    policies: List[PolicyConfig] = field(default_factory=list)
    global_placement_regions: List[PlacementRegion] = field(default_factory=list)
    global_variation_assets: List[VariationAsset] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Helper Functions
# ============================================================================

def load_json(path: Path) -> dict:
    """Load JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def create_gemini_client():
    """Create Gemini client."""
    if genai is None:
        raise ImportError("google-genai package not installed")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


def create_unified_llm_client():
    """Create a unified LLM client supporting Gemini and OpenAI.

    Returns the unified client if available, falls back to Gemini.
    """
    if HAVE_LLM_CLIENT and create_llm_client is not None:
        try:
            return create_llm_client()
        except Exception as e:
            print(f"[REPLICATOR] Unified LLM client failed: {e}, falling back to Gemini", file=sys.stderr)

    # Fallback to direct Gemini
    return create_gemini_client()


def get_llm_provider_name() -> str:
    """Get the name of the current LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return "openai"
    if provider == "mock":
        return "mock"
    return "gemini"


def create_catalog_client() -> Optional[AssetCatalogClient]:
    """Create an asset catalog client if dependencies are available."""

    try:
        return AssetCatalogClient()
    except Exception as exc:  # pragma: no cover - optional dependency may be missing
        print(f"[REPLICATOR] WARNING: catalog unavailable: {exc}", file=sys.stderr)
        return None


def enrich_variation_assets_from_catalog(
    assets: List[VariationAsset], catalog_client: Optional[AssetCatalogClient]
) -> List[VariationAsset]:
    """Augment variation assets with catalog paths/physics when possible."""

    if catalog_client is None:
        return assets

    enriched: List[VariationAsset] = []
    for asset in assets:
        match = None
        try:
            matches = catalog_client.query_assets(
                sim_roles={"variation_asset", "background", "decor"},
                class_name=asset.semantic_class,
                limit=1,
            )
            match = matches[0] if matches else None
        except Exception as exc:  # pragma: no cover - network errors
            print(f"[REPLICATOR] WARNING: catalog query failed: {exc}", file=sys.stderr)

        if match is None:
            enriched.append(asset)
            continue

        asset.asset_id = asset.asset_id or match.asset_id
        asset.asset_path = asset.asset_path or match.usd_path or match.gcs_uri
        asset.thumbnail_uri = asset.thumbnail_uri or match.thumbnail_uri
        if not asset.physics_hints and match.physics_profile:
            asset.physics_hints = match.physics_profile
        if match.source:
            asset.source_hint = asset.source_hint or match.source

        enriched.append(asset)

    return enriched


def detect_environment_type(scene_type: str, inventory: dict) -> EnvironmentType:
    """Detect environment type from scene data."""
    # First check direct mapping
    scene_type_lower = scene_type.lower().strip()
    if scene_type_lower in SCENE_TYPE_TO_ENVIRONMENT:
        return SCENE_TYPE_TO_ENVIRONMENT[scene_type_lower]

    # Try to infer from objects
    objects = inventory.get("objects", [])
    object_categories = set()
    object_ids = set()

    for obj in objects:
        category = obj.get("category", "").lower()
        obj_id = obj.get("id", "").lower()
        object_categories.add(category)
        object_ids.add(obj_id)

    # Inference rules
    if any("refrigerator" in oid or "oven" in oid or "dishwasher" in oid for oid in object_ids):
        return EnvironmentType.KITCHEN
    if any("washer" in oid or "dryer" in oid or "hamper" in oid for oid in object_ids):
        return EnvironmentType.HOME_LAUNDRY
    if any("pallet" in oid or "racking" in oid or "forklift" in oid for oid in object_ids):
        return EnvironmentType.WAREHOUSE
    if any("shelf" in oid and "grocery" in scene_type_lower for oid in object_ids):
        return EnvironmentType.GROCERY
    if any("bed" in oid or "dresser" in oid or "nightstand" in oid for oid in object_ids):
        return EnvironmentType.BEDROOM
    if any("lab" in oid or "bench" in oid or "microscope" in oid for oid in object_ids):
        return EnvironmentType.LAB

    return EnvironmentType.GENERIC


def get_object_by_categories(objects: List[dict], categories: List[str]) -> List[dict]:
    """Filter objects by category."""
    return [obj for obj in objects if obj.get("category", "").lower() in categories]


def get_objects_by_sim_role(objects: List[dict], sim_roles: List[str]) -> List[dict]:
    """Filter objects by sim_role."""
    return [obj for obj in objects if obj.get("sim_role", "").lower() in sim_roles]


def get_articulated_objects(objects: List[dict]) -> List[dict]:
    """Get objects with articulation hints."""
    return [obj for obj in objects if obj.get("articulation_hint")]


def get_manipulable_objects(objects: List[dict]) -> List[dict]:
    """Get manipulable objects."""
    return get_objects_by_sim_role(objects, ["manipulable_object"])


# ============================================================================
# Gemini-Based Analysis
# ============================================================================

def build_scene_analysis_prompt(
    scene_type: str,
    environment_type: EnvironmentType,
    inventory: dict,
    scene_assets: dict,
    requested_policies: Optional[List[str]] = None
) -> str:
    """Build prompt for Gemini to analyze scene and generate Replicator config."""

    objects_summary = []
    for obj in inventory.get("objects", []):
        obj_summary = {
            "id": obj.get("id"),
            "category": obj.get("category"),
            "sim_role": obj.get("sim_role"),
            "description": obj.get("short_description"),
            "articulation": obj.get("articulation_hint"),
            "location": obj.get("approx_location"),
        }
        objects_summary.append(obj_summary)

    # Get available policies for this environment
    available_policies = ENVIRONMENT_DEFAULT_POLICIES.get(
        environment_type,
        ENVIRONMENT_DEFAULT_POLICIES[EnvironmentType.GENERIC]
    )

    policy_list = "\n".join([f"  - {p.value}" for p in available_policies])

    if requested_policies:
        policy_filter = f"\nUser has specifically requested these policies: {requested_policies}"
    else:
        policy_filter = "\nGenerate configurations for ALL applicable policies from the list above."

    prompt = f"""You are an expert in NVIDIA Isaac Sim Replicator for robotics synthetic data generation.

Analyze this scene and generate a comprehensive Replicator configuration for domain randomization.

## Scene Information

Scene Type: {scene_type}
Environment Type: {environment_type.value}

Objects in Scene:
```json
{json.dumps(objects_summary, indent=2)}
```

## Available Policies for {environment_type.value} environments:
{policy_list}
{policy_filter}

## Your Task

Generate a JSON configuration with:

1. **placement_regions**: Define surfaces/volumes where objects can be placed
   - For each region: name, description, surface_type, position estimate, size estimate
   - Link to parent objects when applicable (e.g., "counter_region" linked to "counter_1")
   - Include semantic tags and suitable object categories

2. **variation_assets**: Additional assets needed for domain randomization
   - For each asset: name, category, description, semantic_class, priority
   - Include example variants (e.g., ["dirty_plate", "clean_plate", "chipped_plate"])
   - Physics hints (mass range, friction, etc.)

3. **policy_configs**: For each applicable policy:
   - Which placement regions to use
   - Which variation assets to spawn
   - Randomizer settings (how many objects, distribution, etc.)
   - What to randomize (positions, materials, lighting, etc.)

## Output Format

Return ONLY valid JSON in this exact structure:

{{
  "analysis": {{
    "scene_summary": "Brief description of the scene",
    "key_surfaces": ["list of key surfaces for placement"],
    "key_articulated": ["list of articulated objects"],
    "key_manipulable": ["list of manipulable objects"],
    "recommended_policies": ["list of recommended policy IDs"]
  }},
  "placement_regions": [
    {{
      "name": "region_name",
      "description": "Human readable description",
      "surface_type": "horizontal|vertical|volume",
      "parent_object_id": "object_id or null",
      "position": [x, y, z],
      "size": [width, depth, height],
      "rotation": [rx, ry, rz],
      "semantic_tags": ["tag1", "tag2"],
      "suitable_for": ["dishes", "groceries", "tools"]
    }}
  ],
  "variation_assets": [
    {{
      "name": "asset_name",
      "category": "category",
      "description": "Description",
      "semantic_class": "dish|grocery|clothing|tool|container",
      "priority": "required|recommended|optional",
      "source_hint": "generate|library|simready",
      "example_variants": ["variant1", "variant2"],
      "physics_hints": {{
        "mass_range_kg": [min, max],
        "friction": 0.5,
        "collision_shape": "convex|box|sphere"
      }}
    }}
  ],
  "policy_configs": [
    {{
      "policy_id": "policy_target_value",
      "policy_name": "Human Readable Name",
      "description": "What this policy trains for",
      "placement_regions_used": ["region1", "region2"],
      "variation_assets_used": ["asset1", "asset2"],
      "randomizers": [
        {{
          "name": "object_scatter",
          "enabled": true,
          "frequency": "per_frame",
          "parameters": {{
            "min_objects": 5,
            "max_objects": 15,
            "distribution": "uniform|clustered|sparse"
          }}
        }},
        {{
          "name": "material_variation",
          "enabled": true,
          "frequency": "per_frame",
          "parameters": {{
            "vary_color": true,
            "vary_roughness": true,
            "roughness_range": [0.2, 0.8]
          }}
        }},
        {{
          "name": "lighting_variation",
          "enabled": true,
          "frequency": "per_episode",
          "parameters": {{
            "intensity_range": [0.5, 1.5],
            "color_temperature_range": [4000, 6500]
          }}
        }}
      ],
      "capture_config": {{
        "resolution": [1280, 720],
        "annotations": ["rgb", "depth", "semantic_segmentation", "instance_segmentation", "bounding_box_2d", "bounding_box_3d"],
        "frames_per_episode": 100
      }},
      "scene_modifications": {{
        "hide_objects": [],
        "spawn_additional": true,
        "modify_existing": false
      }}
    }}
  ]
}}

## Guidelines

1. **Placement Regions**:
   - Create regions on ALL horizontal surfaces (counters, tables, shelves, floors)
   - Create regions inside articulated containers (dishwasher racks, drawers, cabinets)
   - Estimate positions based on object locations in the inventory
   - Size should be realistic for the surface type

2. **Variation Assets**:
   - Include assets appropriate for the scene type
   - Kitchen: dishes, utensils, food items, containers
   - Grocery: packaged goods, produce, bottles, cans
   - Warehouse: boxes, totes, pallets, tools
   - Laundry: clothes, towels, detergent bottles
   - Lab: test tubes, beakers, pipettes, sample containers
   - Mark "required" assets that are essential for the policy
   - Mark "optional" assets for additional variety

3. **Policy Configs**:
   - Each policy should have realistic randomization parameters
   - Include material/texture variation for sim-to-real transfer
   - Include lighting variation
   - Specify which annotations to capture

4. **Be Specific**:
   - Use actual object IDs from the inventory
   - Estimate realistic sizes in meters
   - Include physics-plausible parameters

Return ONLY the JSON, no additional text or explanation.
"""
    return prompt


def analyze_scene_with_gemini(
    client,
    scene_type: str,
    environment_type: EnvironmentType,
    inventory: dict,
    scene_assets: dict,
    requested_policies: Optional[List[str]] = None
) -> dict:
    """Use Gemini to analyze scene and generate Replicator configuration."""

    prompt = build_scene_analysis_prompt(
        scene_type=scene_type,
        environment_type=environment_type,
        inventory=inventory,
        scene_assets=scene_assets,
        requested_policies=requested_policies
    )

    print("[REPLICATOR] Calling Gemini for scene analysis...")

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=16000,
            response_mime_type="application/json",
        ),
    )

    response_text = response.text.strip()

    # Clean up response if needed
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    try:
        result = json.loads(response_text)
        print("[REPLICATOR] Successfully parsed Gemini response")
        return result
    except json.JSONDecodeError as e:
        print(f"[REPLICATOR] WARNING: Failed to parse Gemini response: {e}", file=sys.stderr)
        print(f"[REPLICATOR] Response was: {response_text[:500]}...", file=sys.stderr)
        # Return minimal valid structure
        return build_minimal_analysis_result("Analysis failed")


def build_minimal_analysis_result(scene_summary: str) -> dict:
    """Return a minimal analysis result compatible with downstream parsing."""
    return {
        "analysis": {"scene_summary": scene_summary, "recommended_policies": []},
        "placement_regions": [],
        "variation_assets": [],
        "policy_configs": []
    }


def analyze_scene_with_llm(
    client: LLMClient,
    scene_type: str,
    environment_type: EnvironmentType,
    inventory: dict,
    scene_assets: dict,
    requested_policies: Optional[List[str]] = None
) -> dict:
    """Use the unified LLM client to analyze a scene."""

    prompt = build_scene_analysis_prompt(
        scene_type=scene_type,
        environment_type=environment_type,
        inventory=inventory,
        scene_assets=scene_assets,
        requested_policies=requested_policies
    )

    print("[REPLICATOR] Calling unified LLM for scene analysis...")

    response = client.generate(
        prompt=prompt,
        json_output=True,
        temperature=0.3,
        max_tokens=16000,
    )

    try:
        result = response.parse_json() if hasattr(response, "parse_json") else json.loads(response.text)
        print("[REPLICATOR] Successfully parsed unified LLM response")
        return result
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        print(f"[REPLICATOR] WARNING: Failed to parse LLM response: {exc}", file=sys.stderr)
        response_preview = getattr(response, "text", "")[:500]
        print(f"[REPLICATOR] Response was: {response_preview}...", file=sys.stderr)
        return build_minimal_analysis_result("Analysis failed")


# ============================================================================
# USD Layer Generation
# ============================================================================

def generate_placement_regions_usda(
    regions: List[PlacementRegion],
    scene_id: str
) -> str:
    """Generate USDA content for placement regions layer."""

    usda_content = f'''#usda 1.0
(
    defaultPrim = "PlacementRegions"
    metersPerUnit = 1.0
    upAxis = "Y"
    doc = "Placement regions for Replicator domain randomization - Scene: {scene_id}"
)

def Xform "PlacementRegions" (
    doc = "Container for all placement regions used by Replicator scripts"
)
{{
'''

    for region in regions:
        # Default values if not provided
        pos = region.position or [0, 0, 0]
        size = region.size or [1, 1, 0.01]  # Default thin plane
        rot = region.rotation or [0, 0, 0]

        # Sanitize name for USD
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', region.name)

        # Determine geometry type
        if region.surface_type == "volume":
            geom_type = "Cube"
            extent_attr = f'float3[] extent = [({-size[0]/2}, {-size[1]/2}, {-size[2]/2}), ({size[0]/2}, {size[1]/2}, {size[2]/2})]'
            size_attr = f'double size = 1.0'
            scale_attr = f'double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})'
        else:
            # Horizontal or vertical plane
            geom_type = "Plane"
            extent_attr = f'float3[] extent = [({-size[0]/2}, 0, {-size[1]/2}), ({size[0]/2}, 0, {size[1]/2})]'
            size_attr = f'double length = {size[0]}\n        double width = {size[1]}'
            scale_attr = f'double3 xformOp:scale = (1, 1, 1)'

        # Semantic tags as custom attributes
        tags_str = ", ".join([f'"{t}"' for t in region.semantic_tags])
        suitable_str = ", ".join([f'"{s}"' for s in region.suitable_for])

        usda_content += f'''
    def Xform "{safe_name}" (
        doc = "{region.description}"
        customData = {{
            string replicator:region_type = "{region.surface_type}"
            string replicator:parent_object = "{region.parent_object_id or ''}"
        }}
    )
    {{
        double3 xformOp:translate = ({pos[0]}, {pos[1]}, {pos[2]})
        double3 xformOp:rotateXYZ = ({rot[0]}, {rot[1]}, {rot[2]})
        {scale_attr}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]

        # Custom attributes for Replicator
        custom string[] replicator:semantic_tags = [{tags_str}]
        custom string[] replicator:suitable_for = [{suitable_str}]
        custom string replicator:surface_type = "{region.surface_type}"

        def {geom_type} "Surface" (
            purpose = "guide"
        )
        {{
            {size_attr}
            {extent_attr}

            # Make invisible in render but visible in viewport
            token visibility = "invisible"
            bool doubleSided = true

            # Guide display color
            color3f[] primvars:displayColor = [(0.2, 0.8, 0.2)]
            float[] primvars:displayOpacity = [0.3]
        }}
    }}
'''

    usda_content += "}\n"
    return usda_content


# ============================================================================
# Replicator Script Generation
# ============================================================================

def generate_replicator_script(
    policy_config: PolicyConfig,
    all_regions: List[PlacementRegion],
    all_assets: List[VariationAsset],
    scene_id: str
) -> str:
    """Generate a complete Replicator Python script for a policy."""

    # Get regions and assets used by this policy
    region_names = policy_config.placement_regions if hasattr(policy_config, 'placement_regions') else []
    if isinstance(region_names, list) and len(region_names) > 0 and isinstance(region_names[0], PlacementRegion):
        regions_used = region_names
    else:
        # It's a list of names, find matching regions
        region_name_set = set(region_names) if region_names else set()
        regions_used = [r for r in all_regions if r.name in region_name_set] if region_name_set else all_regions

    asset_names = policy_config.variation_assets if hasattr(policy_config, 'variation_assets') else []
    if isinstance(asset_names, list) and len(asset_names) > 0 and isinstance(asset_names[0], VariationAsset):
        assets_used = asset_names
    else:
        asset_name_set = set(asset_names) if asset_names else set()
        assets_used = [a for a in all_assets if a.name in asset_name_set] if asset_name_set else all_assets

    # Build region paths dict
    region_paths = {}
    for r in regions_used:
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', r.name)
        region_paths[r.name] = f"/PlacementRegions/{safe_name}/Surface"

    # Build asset paths dict (assuming they'll be in variation_assets folder)
    asset_paths = {}
    for a in assets_used:
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', a.name)
        target_path = a.asset_path or f"./variation_assets/{safe_name}.usdz"
        asset_paths[a.name] = target_path

    # Build randomizer configs
    randomizers = policy_config.randomizers if policy_config.randomizers else []

    # Capture config
    capture = policy_config.capture_config if policy_config.capture_config else {
        "resolution": [1280, 720],
        "annotations": ["rgb", "depth", "semantic_segmentation", "bounding_box_2d"],
        "frames_per_episode": 100
    }

    script = f'''#!/usr/bin/env python3
"""
Replicator Script: {policy_config.policy_name}
Scene: {scene_id}
Policy: {policy_config.policy_target.value if isinstance(policy_config.policy_target, PolicyTarget) else policy_config.policy_target}

{policy_config.description}

This script is auto-generated by the BlueprintPipeline replicator-job.
To use: Open scene.usda in Isaac Sim, then run this script in the Script Editor.
"""

import omni.replicator.core as rep
from typing import List, Dict, Any, Optional
import random
import json
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

SCENE_ROOT = "/World"
PLACEMENT_REGIONS_ROOT = "/PlacementRegions"

# Placement regions for this policy
PLACEMENT_REGIONS = {json.dumps(region_paths, indent=4)}

# Variation assets for this policy
VARIATION_ASSETS = {json.dumps(asset_paths, indent=4)}

# Capture configuration
CAPTURE_CONFIG = {json.dumps(capture, indent=4)}

# Randomization parameters
RANDOMIZER_CONFIGS = {json.dumps([asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in randomizers], indent=4)}

# Tracks objects spawned by scatter randomizer
SPAWNED_OBJECTS = []


# ============================================================================
# Utility Functions
# ============================================================================

def get_placement_surface(region_name: str):
    """Get a placement surface prim by region name."""
    path = PLACEMENT_REGIONS.get(region_name)
    if path:
        return rep.get.prim_at_path(PLACEMENT_REGIONS_ROOT + "/" + region_name.replace("_region", "") + "/Surface")
    return None


def get_all_placement_surfaces():
    """Get all placement surfaces."""
    surfaces = []
    for name in PLACEMENT_REGIONS.keys():
        surface = get_placement_surface(name)
        if surface:
            surfaces.append(surface)
    return surfaces


def load_variation_assets() -> List[str]:
    """Load all variation asset paths."""
    # In production, these would be actual USD paths
    # For now, return the configured paths
    return list(VARIATION_ASSETS.values())


def load_variation_metadata() -> Dict[str, Any]:
    """Load variation asset metadata if available."""
    candidate_paths = [
        Path("./variation_assets/variation_assets.json"),
        Path("./variation_assets.json"),
        Path("./variation_assets/manifest.json"),
    ]
    for path in candidate_paths:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                assets = data.get("assets") or data.get("variation_assets") or []
                return {{
                    "assets": assets,
                    "raw": data,
                }}
            except json.JSONDecodeError:
                print(f"[REPLICATOR] Warning: Failed to parse variation metadata at {{path}}")
                return {{"assets": [], "raw": {{}}}}
    return {{"assets": [], "raw": {{}}}}


def _material_ranges_from_hint(material_hint: str) -> Dict[str, Any]:
    hint = (material_hint or "").lower()
    ranges = {{
        "base_color_min": (0.4, 0.4, 0.4),
        "base_color_max": (1.0, 1.0, 1.0),
        "roughness": (0.2, 0.8),
        "metallic": (0.0, 0.2),
    }}
    if any(token in hint for token in ["metal", "aluminum", "steel", "stainless"]):
        ranges.update({{
            "roughness": (0.05, 0.4),
            "metallic": (0.6, 1.0),
        }})
    elif any(token in hint for token in ["glass", "ceramic", "porcelain", "stoneware"]):
        ranges.update({{
            "roughness": (0.05, 0.3),
            "metallic": (0.0, 0.1),
        }})
    elif any(token in hint for token in ["fabric", "cloth", "cotton", "textile"]):
        ranges.update({{
            "roughness": (0.6, 1.0),
            "metallic": (0.0, 0.05),
        }})
    elif any(token in hint for token in ["plastic", "polymer", "rubber"]):
        ranges.update({{
            "roughness": (0.3, 0.7),
            "metallic": (0.0, 0.1),
        }})
    elif "wood" in hint:
        ranges.update({{
            "roughness": (0.4, 0.85),
            "metallic": (0.0, 0.05),
        }})
    return ranges


def _resolve_target_prims(targets: Optional[List[str]], label: str):
    if not targets:
        raise RuntimeError(f"[REPLICATOR] Missing target prims for {{label}}: no targets configured")
    prims = []
    for target in targets:
        prims.extend(rep.get.prims(path_pattern=target))
    if not prims:
        raise RuntimeError(f"[REPLICATOR] Missing target prims for {{label}}: {{targets}}")
    return prims


def _collect_texture_variants(asset_metadata: Dict[str, Any]) -> List[str]:
    for key in ["texture_variants", "textures", "texture_paths", "material_textures"]:
        textures = asset_metadata.get(key)
        if isinstance(textures, list):
            return [t for t in textures if isinstance(t, str)]
    return []


def _build_material_metadata_index(metadata: Dict[str, Any]) -> Dict[str, Any]:
    by_name = {{}}
    by_semantic = {{}}
    textures_by_name = {{}}
    textures_by_semantic = {{}}
    for asset in metadata.get("assets", []):
        name = asset.get("name")
        semantic = asset.get("semantic_class")
        if name:
            by_name[name] = asset
            textures = _collect_texture_variants(asset)
            if textures:
                textures_by_name[name] = textures
        if semantic:
            by_semantic.setdefault(semantic, []).append(asset)
            textures = _collect_texture_variants(asset)
            if textures:
                textures_by_semantic.setdefault(semantic, []).extend(textures)
    return {{
        "by_name": by_name,
        "by_semantic": by_semantic,
        "textures_by_name": textures_by_name,
        "textures_by_semantic": textures_by_semantic,
    }}


def _collect_assets_by_category(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    assets_by_category: Dict[str, List[str]] = {{}}
    for asset in metadata.get("assets", []):
        category = asset.get("category")
        asset_path = asset.get("asset_path") or asset.get("path") or asset.get("usd_path")
        if category and asset_path:
            assets_by_category.setdefault(category, []).append(asset_path)
    return assets_by_category


def _extract_physics_hints(params: Dict[str, Any]) -> Dict[str, Any]:
    hints: Dict[str, Any] = {{}}
    if isinstance(params.get("physics_hints"), dict):
        hints.update(params["physics_hints"])
    for key in (
        "dynamic_friction_range",
        "static_friction_range",
        "restitution_range",
        "mass_scale_range",
        "density_scale_range",
    ):
        if key in params and key not in hints:
            hints[key] = params[key]
    return hints


def _apply_physics_hints(target_prims, physics_hints: Dict[str, Any]) -> None:
    if not physics_hints or target_prims is None:
        return

    dynamic_friction_range = physics_hints.get("dynamic_friction_range")
    static_friction_range = physics_hints.get("static_friction_range")
    restitution_range = physics_hints.get("restitution_range")
    mass_scale_range = physics_hints.get("mass_scale_range")
    density_scale_range = physics_hints.get("density_scale_range")

    if not any(
        [
            dynamic_friction_range,
            static_friction_range,
            restitution_range,
            mass_scale_range,
            density_scale_range,
        ]
    ):
        return

    with target_prims:
        if dynamic_friction_range:
            rep.modify.attribute(
                "physxMaterial:dynamicFriction",
                rep.distribution.uniform(dynamic_friction_range[0], dynamic_friction_range[1]),
            )
        if static_friction_range:
            rep.modify.attribute(
                "physxMaterial:staticFriction",
                rep.distribution.uniform(static_friction_range[0], static_friction_range[1]),
            )
        if restitution_range:
            rep.modify.attribute(
                "physxMaterial:restitution",
                rep.distribution.uniform(restitution_range[0], restitution_range[1]),
            )
        if mass_scale_range:
            rep.modify.attribute(
                "physxMassProperties:massScale",
                rep.distribution.uniform(mass_scale_range[0], mass_scale_range[1]),
            )
        if density_scale_range:
            rep.modify.attribute(
                "physxMassProperties:densityScale",
                rep.distribution.uniform(density_scale_range[0], density_scale_range[1]),
            )


# ============================================================================
# Randomizers
# ============================================================================

def create_object_scatter_randomizer(
    surfaces,
    asset_paths: List[str],
    min_objects: int = 5,
    max_objects: int = 15,
    semantic_class: str = "object",
    collision_check: bool = True,
    physics_hints: Optional[Dict[str, Any]] = None,
):
    """Create a randomizer that scatters objects on surfaces."""
    physics_hints = physics_hints or {{}}

    def randomize_objects():
        # Determine number of objects to spawn
        num_objects = random.randint(min_objects, max_objects)

        if not asset_paths:
            print("[REPLICATOR] Warning: No asset paths provided for scatter")
            return None

        # Create objects from random asset selection
        objects = rep.create.from_usd(
            rep.distribution.choice(asset_paths, num_objects),
            semantics=[("class", semantic_class)],
            count=num_objects
        )
        SPAWNED_OBJECTS.append(objects)

        with objects:
            # Random rotation
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, -15, 0), (0, 15, 360))
            )

            # Scatter on surfaces
            if surfaces:
                rep.randomizer.scatter_2d(
                    surface_prims=surfaces,
                    check_for_collisions=collision_check,
                    seed=random.randint(0, 999999)
                )
        _apply_physics_hints(objects, physics_hints)

        return objects

    return randomize_objects


def create_material_variation_randomizer(
    variation_metadata: Dict[str, Any],
    target_semantic_class: str = "object",
    allow_textures: bool = True,
    physics_hints: Optional[Dict[str, Any]] = None,
):
    """Create a randomizer for material properties on spawned variant objects."""
    metadata_index = _build_material_metadata_index(variation_metadata)
    physics_hints = physics_hints or {{}}

    def _select_target_prims():
        if SPAWNED_OBJECTS:
            return SPAWNED_OBJECTS
        prims = rep.get.prims(semantics=[("class", target_semantic_class)])
        if prims:
            return [prims]
        prims = rep.get.prims(semantics=[("variant", "true")])
        if prims:
            return [prims]
        prims = rep.get.prims(semantics=[("variant", "variant")])
        if prims:
            return [prims]
        return []

    def _resolve_material_ranges(semantic_class: str) -> Dict[str, Any]:
        assets = metadata_index["by_semantic"].get(semantic_class, [])
        hints = [a.get("material_hint") for a in assets if a.get("material_hint")]
        if hints:
            return _material_ranges_from_hint(random.choice(hints))
        return _material_ranges_from_hint("")

    def _resolve_textures(semantic_class: str) -> List[str]:
        textures = metadata_index["textures_by_semantic"].get(semantic_class, [])
        return list(set(t for t in textures if isinstance(t, str)))

    def randomize_materials():
        target_groups = _select_target_prims()
        if not target_groups:
            return

        material_ranges = _resolve_material_ranges(target_semantic_class)
        textures = _resolve_textures(target_semantic_class) if allow_textures else []
        for target_prims in target_groups:
            with target_prims:
                rep.modify.attribute(
                    "inputs:base_color",
                    rep.distribution.uniform(
                        material_ranges["base_color_min"],
                        material_ranges["base_color_max"],
                    ),
                )
                rep.modify.attribute(
                    "inputs:roughness",
                    rep.distribution.uniform(
                        material_ranges["roughness"][0],
                        material_ranges["roughness"][1],
                    ),
                )
                rep.modify.attribute(
                    "inputs:metallic",
                    rep.distribution.uniform(
                        material_ranges["metallic"][0],
                        material_ranges["metallic"][1],
                    ),
                )
                if textures:
                    texture_choice = rep.distribution.choice(textures)
                    rep.modify.attribute("inputs:diffuse_texture", texture_choice)
                    rep.modify.attribute("inputs:diffuseTexture", texture_choice)
                    rep.modify.attribute("inputs:base_color_texture", texture_choice)
            _apply_physics_hints(target_prims, physics_hints)

    return randomize_materials


def create_lighting_randomizer(
    intensity_range: tuple = (0.5, 1.5),
    color_temp_range: tuple = (4000, 6500)
):
    """Create a randomizer for scene lighting."""

    def randomize_lighting():
        lights = rep.get.prims(path_pattern="/World/.*[Ll]ight.*")

        with lights:
            rep.modify.attribute(
                "inputs:intensity",
                rep.distribution.uniform(intensity_range[0], intensity_range[1])
            )

    return randomize_lighting


def create_camera_randomizer(
    camera_path: str = "/World/Camera",
    position_noise: tuple = (0.05, 0.05, 0.05),
    rotation_noise: tuple = (2, 2, 2)
):
    """Create a randomizer for camera pose with small perturbations."""

    def randomize_camera():
        camera = rep.get.prim_at_path(camera_path)

        if camera:
            with camera:
                # Small position noise
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-position_noise[0], -position_noise[1], -position_noise[2]),
                        (position_noise[0], position_noise[1], position_noise[2])
                    ),
                    rotation=rep.distribution.uniform(
                        (-rotation_noise[0], -rotation_noise[1], -rotation_noise[2]),
                        (rotation_noise[0], rotation_noise[1], rotation_noise[2])
                    )
                )

    return randomize_camera


def create_articulation_state_randomizer(
    targets: List[str],
    open_probability: float = 0.5,
    normalize_range: bool = True,
):
    """Randomize articulated joints to open/closed states."""
    prims = _resolve_target_prims(targets, "articulation_state")

    def randomize_articulation_state():
        target_position = rep.distribution.choice(
            [0.0, 1.0],
            weights=[1.0 - open_probability, open_probability],
        )
        if not normalize_range:
            target_position = rep.distribution.choice(
                [0.0, 0.5, 1.0],
                weights=[1.0 - open_probability, open_probability * 0.5, open_probability * 0.5],
            )
        for prim in prims:
            with prim:
                rep.modify.attribute("drive:targetPosition", target_position)

    return randomize_articulation_state


def create_object_placement_randomizer(
    targets: List[str],
    position_noise: float = 0.05,
    rotation_noise: float = 5.0,
    maintain_surface_contact: bool = True,
    surfaces: Optional[List[Any]] = None,
    collision_check: bool = True,
):
    """Randomize poses for existing objects."""
    prims = _resolve_target_prims(targets, "object_placement")

    def randomize_object_placement():
        if maintain_surface_contact and surfaces:
            rep.randomizer.scatter_2d(
                prims=prims,
                surface_prims=surfaces,
                check_for_collisions=collision_check,
                seed=random.randint(0, 999999),
            )
        for prim in prims:
            with prim:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-position_noise, -position_noise, 0.0),
                        (position_noise, position_noise, position_noise),
                    ),
                    rotation=rep.distribution.uniform(
                        (-rotation_noise, -rotation_noise, -rotation_noise),
                        (rotation_noise, rotation_noise, rotation_noise),
                    ),
                )

    return randomize_object_placement


def create_drawer_state_randomizer(
    targets: List[str],
    open_range: List[float],
):
    """Randomize drawer joint positions."""
    prims = _resolve_target_prims(targets, "drawer_state")

    def randomize_drawer_state():
        target_position = rep.distribution.uniform(open_range[0], open_range[1])
        for prim in prims:
            with prim:
                rep.modify.attribute("drive:targetPosition", target_position)

    return randomize_drawer_state


def create_drawer_contents_randomizer(
    targets: List[str],
    asset_paths: List[str],
    fill_ratio: List[float],
    max_items: int = 8,
):
    """Scatter assets into drawer interiors."""
    prims = _resolve_target_prims(targets, "drawer_contents")

    def randomize_drawer_contents():
        if not asset_paths:
            raise RuntimeError("[REPLICATOR] Missing asset paths for drawer contents scatter")
        count = max(1, int(random.uniform(fill_ratio[0], fill_ratio[1]) * max_items))
        items = rep.create.from_usd(
            rep.distribution.choice(asset_paths, count),
            semantics=[("class", "drawer_contents")],
            count=count,
        )
        SPAWNED_OBJECTS.append(items)
        rep.randomizer.scatter_2d(
            prims=items,
            surface_prims=prims,
            check_for_collisions=True,
            seed=random.randint(0, 999999),
        )

    return randomize_drawer_contents


def create_door_state_randomizer(
    targets: List[str],
    open_range: List[float],
):
    """Randomize door joint positions."""
    prims = _resolve_target_prims(targets, "door_state")

    def randomize_door_state():
        target_position = rep.distribution.uniform(open_range[0], open_range[1])
        for prim in prims:
            with prim:
                rep.modify.attribute("drive:targetPosition", target_position)

    return randomize_door_state


def create_knob_state_randomizer(
    targets: List[str],
    rotation_range: List[float],
):
    """Randomize knob rotation."""
    prims = _resolve_target_prims(targets, "knob_state")

    def randomize_knob_state():
        target_position = rep.distribution.uniform(rotation_range[0], rotation_range[1])
        for prim in prims:
            with prim:
                rep.modify.attribute("drive:targetPosition", target_position)

    return randomize_knob_state


def create_cloth_scatter_randomizer(
    targets: List[str],
    asset_paths: List[str],
    min_items: int = 5,
    max_items: int = 20,
):
    """Scatter cloth assets across hamper/basket surfaces."""
    prims = _resolve_target_prims(targets, "cloth_scatter")

    def randomize_cloth_scatter():
        if not asset_paths:
            raise RuntimeError("[REPLICATOR] Missing asset paths for cloth scatter")
        count = random.randint(min_items, max_items)
        items = rep.create.from_usd(
            rep.distribution.choice(asset_paths, count),
            semantics=[("class", "cloth")],
            count=count,
        )
        SPAWNED_OBJECTS.append(items)
        rep.randomizer.scatter_2d(
            prims=items,
            surface_prims=prims,
            check_for_collisions=True,
            seed=random.randint(0, 999999),
        )

    return randomize_cloth_scatter


def create_cloth_deformation_randomizer(
    targets: List[str],
    simulation_steps: int = 10,
    gravity_variation: List[float] = None,
    wind_enabled: bool = False,
):
    """Randomize cloth simulation parameters."""
    prims = _resolve_target_prims(targets, "cloth_deformation")
    gravity_variation = gravity_variation or [-0.2, 0.2]

    def randomize_cloth_deformation():
        gravity_scale = rep.distribution.uniform(
            1.0 + gravity_variation[0],
            1.0 + gravity_variation[1],
        )
        for prim in prims:
            with prim:
                rep.modify.attribute("physxCloth:gravityScale", gravity_scale)
                rep.modify.attribute("physxCloth:solverIterations", simulation_steps)
                if wind_enabled:
                    rep.modify.attribute("physxCloth:windDrag", rep.distribution.uniform(0.1, 1.0))

    return randomize_cloth_deformation


def create_shelf_population_randomizer(
    targets: List[str],
    asset_paths: List[str],
    fill_ratio_range: List[float],
):
    """Populate shelves with random assets."""
    prims = _resolve_target_prims(targets, "shelf_population")

    def randomize_shelf_population():
        if not asset_paths:
            raise RuntimeError("[REPLICATOR] Missing asset paths for shelf population")
        count = max(1, int(random.uniform(fill_ratio_range[0], fill_ratio_range[1]) * 20))
        items = rep.create.from_usd(
            rep.distribution.choice(asset_paths, count),
            semantics=[("class", "shelf_item")],
            count=count,
        )
        SPAWNED_OBJECTS.append(items)
        rep.randomizer.scatter_2d(
            prims=items,
            surface_prims=prims,
            check_for_collisions=True,
            seed=random.randint(0, 999999),
        )

    return randomize_shelf_population


def create_table_setup_randomizer(
    targets: List[str],
    asset_paths: List[str],
    place_settings: List[int],
    include_centerpiece: bool,
):
    """Set up table settings with dishes and utensils."""
    prims = _resolve_target_prims(targets, "table_setup")

    def randomize_table_setup():
        if not asset_paths:
            raise RuntimeError("[REPLICATOR] Missing asset paths for table setup")
        count = random.randint(place_settings[0], place_settings[1])
        items = rep.create.from_usd(
            rep.distribution.choice(asset_paths, count),
            semantics=[("class", "table_setting")],
            count=count,
        )
        SPAWNED_OBJECTS.append(items)
        rep.randomizer.scatter_2d(
            prims=items,
            surface_prims=prims,
            check_for_collisions=True,
            seed=random.randint(0, 999999),
        )
        if include_centerpiece:
            centerpiece = rep.create.from_usd(
                rep.distribution.choice(asset_paths),
                semantics=[("class", "centerpiece")],
                count=1,
            )
            SPAWNED_OBJECTS.append(centerpiece)
            rep.randomizer.scatter_2d(
                prims=centerpiece,
                surface_prims=prims,
                check_for_collisions=True,
                seed=random.randint(0, 999999),
            )

    return randomize_table_setup


def create_dirty_state_randomizer(
    targets: List[str],
    dirty_probability: float,
    intensity_range: List[float],
):
    """Apply dirty material variations."""
    prims = _resolve_target_prims(targets, "dirty_state")

    def randomize_dirty_state():
        if random.random() > dirty_probability:
            return
        intensity = rep.distribution.uniform(intensity_range[0], intensity_range[1])
        for prim in prims:
            with prim:
                rep.modify.attribute(
                    "inputs:base_color",
                    rep.distribution.uniform((0.4, 0.3, 0.2), (0.9, 0.85, 0.8)),
                )
                rep.modify.attribute("inputs:roughness", intensity)

    return randomize_dirty_state


def create_dishwasher_state_randomizer(
    targets: List[str],
    door_state: str,
    loaded_probability: float,
    asset_paths: List[str],
):
    """Randomize dishwasher door and load contents."""
    prims = _resolve_target_prims(targets, "dishwasher_state")

    def randomize_dishwasher_state():
        open_range = (0.0, 1.0) if door_state == "variable" else (0.0, 0.0)
        if door_state == "open":
            open_range = (1.0, 1.0)
        target_position = rep.distribution.uniform(open_range[0], open_range[1])
        for prim in prims:
            with prim:
                rep.modify.attribute("drive:targetPosition", target_position)
        if asset_paths and random.random() < loaded_probability:
            count = random.randint(4, 12)
            items = rep.create.from_usd(
                rep.distribution.choice(asset_paths, count),
                semantics=[("class", "dishwasher_load")],
                count=count,
            )
            SPAWNED_OBJECTS.append(items)
            rep.randomizer.scatter_2d(
                prims=items,
                surface_prims=prims,
                check_for_collisions=True,
                seed=random.randint(0, 999999),
            )

    return randomize_dishwasher_state


def create_switch_states_randomizer(
    targets: List[str],
    on_probability: float,
):
    """Randomize binary switch states."""
    prims = _resolve_target_prims(targets, "switch_states")

    def randomize_switch_states():
        state = rep.distribution.choice([0, 1], weights=[1.0 - on_probability, on_probability])
        for prim in prims:
            with prim:
                rep.modify.attribute("inputs:state", state)

    return randomize_switch_states


def create_label_variation_randomizer(
    targets: List[str],
    variation_metadata: Dict[str, Any],
    texture_library: str,
):
    """Swap label textures on target assets."""
    prims = _resolve_target_prims(targets, "label_variation")
    metadata_index = _build_material_metadata_index(variation_metadata)
    textures = metadata_index["textures_by_semantic"].get(texture_library, [])

    if not textures:
        raise RuntimeError(f"[REPLICATOR] Missing label textures for library '{{texture_library}}'")

    def randomize_label_variation():
        texture_choice = rep.distribution.choice(textures)
        for prim in prims:
            with prim:
                rep.modify.attribute("inputs:diffuse_texture", texture_choice)
                rep.modify.attribute("inputs:diffuseTexture", texture_choice)
                rep.modify.attribute("inputs:base_color_texture", texture_choice)

    return randomize_label_variation


def create_pallet_placement_randomizer(
    targets: List[str],
    position_noise: float,
    rotation_noise: float,
):
    """Randomize pallet poses."""
    prims = _resolve_target_prims(targets, "pallet_placement")

    def randomize_pallet_placement():
        for prim in prims:
            with prim:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (-position_noise, -position_noise, 0.0),
                        (position_noise, position_noise, position_noise),
                    ),
                    rotation=rep.distribution.uniform(
                        (-rotation_noise, -rotation_noise, -rotation_noise),
                        (rotation_noise, rotation_noise, rotation_noise),
                    ),
                )

    return randomize_pallet_placement


def create_load_variation_randomizer(
    targets: List[str],
    asset_paths: List[str],
    stack_height_range: List[int],
):
    """Randomize pallet load stacking."""
    prims = _resolve_target_prims(targets, "load_variation")

    def randomize_load_variation():
        if not asset_paths:
            raise RuntimeError("[REPLICATOR] Missing asset paths for load variation")
        count = random.randint(stack_height_range[0], stack_height_range[1])
        items = rep.create.from_usd(
            rep.distribution.choice(asset_paths, count),
            semantics=[("class", "pallet_load")],
            count=count,
        )
        SPAWNED_OBJECTS.append(items)
        rep.randomizer.scatter_2d(
            prims=items,
            surface_prims=prims,
            check_for_collisions=True,
            seed=random.randint(0, 999999),
        )

    return randomize_load_variation


# ============================================================================
# Main Replicator Setup
# ============================================================================

def setup_replicator():
    """Set up the complete Replicator pipeline for {policy_config.policy_name}."""

    print("[REPLICATOR] Setting up {policy_config.policy_name}...")

    # Create a new layer for replicator modifications
    with rep.new_layer():

        # Get placement surfaces
        surfaces = get_all_placement_surfaces()
        if not surfaces:
            print("[REPLICATOR] Warning: No placement surfaces found")
            surfaces = None

        # Load variation assets
        asset_paths = load_variation_assets()
        variation_metadata = load_variation_metadata()

        # Create render product
        resolution = tuple(CAPTURE_CONFIG.get("resolution", [1280, 720]))
        render_product = rep.create.render_product("/World/Camera", resolution)

        # Set up randomizers based on config
        registered_randomizers = []

        for config in RANDOMIZER_CONFIGS:
            if not config.get("enabled", True):
                continue

            name = config.get("name", "")
            params = config.get("parameters", {{}})
            targets = config.get("targets") or params.get("targets") or []
            metadata_assets_by_category = _collect_assets_by_category(variation_metadata)
            asset_categories = params.get("asset_categories", [])
            categorized_assets = []
            for category in asset_categories:
                categorized_assets.extend(metadata_assets_by_category.get(category, []))
            resolved_assets = categorized_assets or asset_paths

            if name == "object_scatter":
                physics_hints = _extract_physics_hints(params)
                randomizer = create_object_scatter_randomizer(
                    surfaces=surfaces,
                    asset_paths=asset_paths,
                    min_objects=params.get("min_objects", 5),
                    max_objects=params.get("max_objects", 15),
                    semantic_class=params.get("semantic_class", "object"),
                    collision_check=params.get("collision_check", True),
                    physics_hints=physics_hints,
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("object_scatter", randomizer, config.get("frequency", "per_frame")))

            elif name == "material_variation":
                physics_hints = _extract_physics_hints(params)
                randomizer = create_material_variation_randomizer(
                    variation_metadata=variation_metadata,
                    target_semantic_class=params.get("semantic_class", "object"),
                    allow_textures=params.get("allow_textures", True),
                    physics_hints=physics_hints,
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append((
                    "material_variation",
                    randomizer,
                    config.get("frequency", "per_frame")
                ))

            elif name == "lighting_variation":
                randomizer = create_lighting_randomizer(
                    intensity_range=tuple(params.get("intensity_range", [0.5, 1.5])),
                    color_temp_range=tuple(params.get("color_temperature_range", [4000, 6500]))
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("lighting", randomizer, config.get("frequency", "per_episode")))

            elif name == "camera_variation":
                randomizer = create_camera_randomizer(
                    position_noise=tuple(params.get("position_noise", [0.05, 0.05, 0.05])),
                    rotation_noise=tuple(params.get("rotation_noise", [2, 2, 2]))
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("camera", randomizer, config.get("frequency", "per_frame")))

            elif name == "articulation_state":
                randomizer = create_articulation_state_randomizer(
                    targets=targets,
                    open_probability=params.get("open_probability", 0.5),
                    normalize_range=params.get("normalize_range", True),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("articulation_state", randomizer, config.get("frequency", "per_frame")))

            elif name == "object_placement":
                randomizer = create_object_placement_randomizer(
                    targets=targets,
                    position_noise=params.get("position_noise", 0.05),
                    rotation_noise=params.get("rotation_noise", 5),
                    maintain_surface_contact=params.get("maintain_surface_contact", True),
                    surfaces=surfaces,
                    collision_check=params.get("collision_check", True),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("object_placement", randomizer, config.get("frequency", "per_frame")))

            elif name == "drawer_state":
                randomizer = create_drawer_state_randomizer(
                    targets=targets,
                    open_range=params.get("open_range", [0.0, 1.0]),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("drawer_state", randomizer, config.get("frequency", "per_frame")))

            elif name == "drawer_contents":
                randomizer = create_drawer_contents_randomizer(
                    targets=targets,
                    asset_paths=resolved_assets,
                    fill_ratio=params.get("fill_ratio", [0.2, 0.8]),
                    max_items=params.get("max_items", 8),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("drawer_contents", randomizer, config.get("frequency", "per_frame")))

            elif name == "door_state":
                randomizer = create_door_state_randomizer(
                    targets=targets,
                    open_range=params.get("open_range", [0.0, 1.57]),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("door_state", randomizer, config.get("frequency", "per_frame")))

            elif name == "knob_state":
                randomizer = create_knob_state_randomizer(
                    targets=targets,
                    rotation_range=params.get("rotation_range", [0.0, 6.28]),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("knob_state", randomizer, config.get("frequency", "per_frame")))

            elif name == "cloth_scatter":
                randomizer = create_cloth_scatter_randomizer(
                    targets=targets,
                    asset_paths=resolved_assets,
                    min_items=params.get("min_items", 5),
                    max_items=params.get("max_items", 20),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("cloth_scatter", randomizer, config.get("frequency", "per_frame")))

            elif name == "cloth_deformation":
                randomizer = create_cloth_deformation_randomizer(
                    targets=targets,
                    simulation_steps=params.get("simulation_steps", 10),
                    gravity_variation=params.get("gravity_variation", [-0.2, 0.2]),
                    wind_enabled=params.get("wind_enabled", False),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("cloth_deformation", randomizer, config.get("frequency", "per_frame")))

            elif name == "shelf_population":
                randomizer = create_shelf_population_randomizer(
                    targets=targets,
                    asset_paths=resolved_assets,
                    fill_ratio_range=params.get("fill_ratio_range", [0.3, 0.9]),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("shelf_population", randomizer, config.get("frequency", "per_frame")))

            elif name == "table_setup":
                randomizer = create_table_setup_randomizer(
                    targets=targets,
                    asset_paths=resolved_assets,
                    place_settings=params.get("place_settings", [1, 6]),
                    include_centerpiece=params.get("include_centerpiece", True),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("table_setup", randomizer, config.get("frequency", "per_frame")))

            elif name == "dirty_state":
                randomizer = create_dirty_state_randomizer(
                    targets=targets,
                    dirty_probability=params.get("dirty_probability", 0.7),
                    intensity_range=params.get("intensity_range", [0.1, 0.8]),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("dirty_state", randomizer, config.get("frequency", "per_frame")))

            elif name == "dishwasher_state":
                randomizer = create_dishwasher_state_randomizer(
                    targets=targets,
                    door_state=params.get("door_state", "variable"),
                    loaded_probability=params.get("loaded_probability", 0.3),
                    asset_paths=resolved_assets,
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("dishwasher_state", randomizer, config.get("frequency", "per_frame")))

            elif name == "switch_states":
                randomizer = create_switch_states_randomizer(
                    targets=targets,
                    on_probability=params.get("on_probability", 0.5),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("switch_states", randomizer, config.get("frequency", "per_frame")))

            elif name == "label_variation":
                randomizer = create_label_variation_randomizer(
                    targets=targets,
                    variation_metadata=variation_metadata,
                    texture_library=params.get("texture_library", "shipping_labels"),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("label_variation", randomizer, config.get("frequency", "per_frame")))

            elif name == "pallet_placement":
                randomizer = create_pallet_placement_randomizer(
                    targets=targets,
                    position_noise=params.get("position_noise", 0.1),
                    rotation_noise=params.get("rotation_noise", 10),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("pallet_placement", randomizer, config.get("frequency", "per_frame")))

            elif name == "load_variation":
                randomizer = create_load_variation_randomizer(
                    targets=targets,
                    asset_paths=resolved_assets,
                    stack_height_range=params.get("stack_height_range", [1, 4]),
                )
                rep.randomizer.register(randomizer)
                registered_randomizers.append(("load_variation", randomizer, config.get("frequency", "per_frame")))

        # Set up writer for annotations
        annotations = CAPTURE_CONFIG.get("annotations", ["rgb"])

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=f"./synthetic_data/{scene_id}/{policy_config.policy_id}",
            rgb=("rgb" in annotations),
            distance_to_camera=("depth" in annotations),
            semantic_segmentation=("semantic_segmentation" in annotations),
            instance_segmentation=("instance_segmentation" in annotations),
            bounding_box_2d_tight=("bounding_box_2d" in annotations),
            bounding_box_3d=("bounding_box_3d" in annotations),
        )
        writer.attach([render_product])

        print(f"[REPLICATOR] Registered {{len(registered_randomizers)}} randomizers")
        print(f"[REPLICATOR] Output annotations: {{annotations}}")

        return registered_randomizers, render_product


def run_replicator(num_frames: int = None):
    """Run the Replicator data generation."""

    if num_frames is None:
        num_frames = CAPTURE_CONFIG.get("frames_per_episode", 100)

    print(f"[REPLICATOR] Starting data generation: {{num_frames}} frames")

    randomizers, render_product = setup_replicator()

    # Trigger randomization on each frame
    with rep.trigger.on_frame(num_frames=num_frames):
        for name, randomizer, frequency in randomizers:
            if frequency == "per_frame":
                randomizer()

    print("[REPLICATOR] Data generation complete!")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Default: run with configured frame count
    run_replicator()
else:
    # When imported in Isaac Sim Script Editor
    print("[REPLICATOR] Script loaded. Call run_replicator() to start.")
    print(f"[REPLICATOR] Policy: {policy_config.policy_name}")
    print(f"[REPLICATOR] Regions: {{list(PLACEMENT_REGIONS.keys())}}")
    print(f"[REPLICATOR] Assets: {{len(VARIATION_ASSETS)}} variation assets configured")
'''

    return script


def generate_master_replicator_script(
    bundle: ReplicatorBundle,
    policies: List[PolicyConfig]
) -> str:
    """Generate a master script that can run any policy."""

    policy_list = []
    for p in policies:
        policy_id = p.policy_id if hasattr(p, 'policy_id') else p.get('policy_id', 'unknown')
        policy_name = p.policy_name if hasattr(p, 'policy_name') else p.get('policy_name', 'Unknown')
        policy_list.append(f'    "{policy_id}": "{policy_name}"')

    policies_dict_str = ",\n".join(policy_list)

    script = f'''#!/usr/bin/env python3
"""
Master Replicator Script for Scene: {bundle.scene_id}
Environment Type: {bundle.environment_type.value}

This script provides a unified interface to run any of the available
policy-specific Replicator configurations for this scene.

Usage in Isaac Sim Script Editor:
    from replicator_master import ReplicatorManager

    manager = ReplicatorManager()
    manager.list_policies()
    manager.run_policy("dish_loading", num_frames=500)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional, List

# Available policies for this scene
AVAILABLE_POLICIES = {{
{policies_dict_str}
}}

SCENE_ID = "{bundle.scene_id}"
ENVIRONMENT_TYPE = "{bundle.environment_type.value}"


class ReplicatorManager:
    """Manager for running Replicator policies on this scene."""

    def __init__(self, scripts_dir: Optional[str] = None):
        """Initialize the manager."""
        if scripts_dir is None:
            # Assume scripts are in the same directory
            self.scripts_dir = Path(__file__).parent / "policies"
        else:
            self.scripts_dir = Path(scripts_dir)

    def list_policies(self) -> List[str]:
        """List all available policies."""
        print(f"\\nAvailable policies for {{SCENE_ID}} ({{ENVIRONMENT_TYPE}}):")
        print("-" * 50)
        for policy_id, policy_name in AVAILABLE_POLICIES.items():
            print(f"  {{policy_id}}: {{policy_name}}")
        print("-" * 50)
        return list(AVAILABLE_POLICIES.keys())

    def load_policy_module(self, policy_id: str):
        """Dynamically load a policy script module."""
        script_path = self.scripts_dir / f"{{policy_id}}.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Policy script not found: {{script_path}}")

        spec = importlib.util.spec_from_file_location(policy_id, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[policy_id] = module
        spec.loader.exec_module(module)

        return module

    def run_policy(self, policy_id: str, num_frames: int = 100, **kwargs):
        """Run a specific policy."""
        if policy_id not in AVAILABLE_POLICIES:
            print(f"Error: Unknown policy '{{policy_id}}'")
            self.list_policies()
            return

        print(f"\\n[REPLICATOR] Loading policy: {{AVAILABLE_POLICIES[policy_id]}}")

        module = self.load_policy_module(policy_id)

        if hasattr(module, 'run_replicator'):
            module.run_replicator(num_frames=num_frames)
        else:
            print(f"Error: Policy module does not have run_replicator function")

    def run_all_policies(self, num_frames_each: int = 100):
        """Run all available policies sequentially."""
        for policy_id in AVAILABLE_POLICIES.keys():
            print(f"\\n{'='*60}")
            print(f"Running policy: {{policy_id}}")
            print(f"{'='*60}")
            self.run_policy(policy_id, num_frames=num_frames_each)


# Quick access functions
def list_policies():
    """List all available policies."""
    manager = ReplicatorManager()
    return manager.list_policies()


def run_policy(policy_id: str, num_frames: int = 100):
    """Run a specific policy."""
    manager = ReplicatorManager()
    manager.run_policy(policy_id, num_frames=num_frames)


def run_all(num_frames_each: int = 100):
    """Run all policies."""
    manager = ReplicatorManager()
    manager.run_all_policies(num_frames_each)


if __name__ == "__main__":
    print("[REPLICATOR] Master script loaded")
    list_policies()
'''

    return script


# ============================================================================
# Variation Asset Manifest Generation
# ============================================================================

def generate_asset_manifest(
    assets: List[VariationAsset],
    scene_id: str,
    scene_type: str = "generic",
    environment_type: Optional[EnvironmentType] = None,
    policies: Optional[List[str]] = None
) -> dict:
    """
    Generate a manifest of variation assets needed for domain randomization.

    This manifest is consumed by variation-gen-job to generate reference images
    using Gemini 3.0 Pro Image (Nano Banana Pro), which are then processed by
    downstream jobs for 3D conversion.

    Args:
        assets: List of VariationAsset specifications
        scene_id: Scene identifier
        scene_type: Type of scene (kitchen, grocery, etc.)
        environment_type: EnvironmentType enum value
        policies: List of policy IDs this manifest supports
    """

    manifest = {
        "scene_id": scene_id,
        "scene_type": scene_type,
        "environment_type": environment_type.value if environment_type else "generic",
        "generated_at": None,  # Will be filled in
        "total_assets": len(assets),
        "policies": policies or [],
        "by_priority": {
            "required": [],
            "recommended": [],
            "optional": []
        },
        "by_category": {},
        "assets": [],
        # Generation hints for variation-gen-job
        "generation_config": {
            "image_model": "gemini-3-pro-image-preview",
            "default_style": "photorealistic product photography",
            "background": "white studio background",
            "lighting": "soft 3-point studio lighting"
        }
    }

    for asset in assets:
        asset_dict = asdict(asset) if hasattr(asset, '__dataclass_fields__') else asset

        # Add to priority groups
        priority = asset_dict.get("priority", "optional")
        if priority in manifest["by_priority"]:
            manifest["by_priority"][priority].append(asset_dict["name"])

        # Add to category groups
        category = asset_dict.get("category", "other")
        if category not in manifest["by_category"]:
            manifest["by_category"][category] = []
        manifest["by_category"][category].append(asset_dict["name"])

        # Enrich asset with generation hints based on category and semantic class
        enriched_asset = _enrich_asset_for_generation(asset_dict, scene_type)

        # Add full asset details
        manifest["assets"].append(enriched_asset)

    return manifest


def _enrich_asset_for_generation(asset_dict: dict, scene_type: str) -> dict:
    """
    Enrich an asset dictionary with additional hints for image generation.

    This adds material hints, style suggestions, and physics defaults based
    on the asset's category and semantic class.
    """
    enriched = asset_dict.copy()

    category = asset_dict.get("category", "").lower()
    semantic_class = asset_dict.get("semantic_class", "").lower()

    # Material hints by category
    material_hints = {
        "dishes": "ceramic, porcelain, or stoneware",
        "utensils": "stainless steel or silver-plated metal",
        "food": "realistic food textures and colors",
        "groceries": "plastic packaging, cardboard boxes, or metal cans",
        "produce": "natural organic textures with realistic imperfections",
        "bottles": "glass or plastic with appropriate transparency",
        "cans": "aluminum or tin with printed labels",
        "boxes": "cardboard with printed packaging graphics",
        "clothing": "cotton, polyester, or mixed fabric textures",
        "towels": "cotton terry cloth or microfiber texture",
        "tools": "metal with rubber or plastic grips",
        "containers": "plastic, glass, or metal storage containers",
        "electronics": "plastic housing with metal accents",
        "office_supplies": "plastic, metal, or paper materials",
        "lab_equipment": "borosilicate glass, stainless steel, or medical-grade plastic",
    }

    # Style hints by category
    style_hints = {
        "dishes": "clean dinnerware, may have subtle patterns or solid colors",
        "utensils": "polished cutlery, professional quality",
        "food": "appetizing presentation, realistic textures",
        "groceries": "retail packaging, brand-appropriate design",
        "produce": "fresh market quality, natural variations",
        "bottles": "consumer beverage or household product",
        "cans": "retail food packaging with labels",
        "boxes": "shipping or retail packaging",
        "clothing": "casual or household garments",
        "towels": "household linens, folded or crumpled",
        "tools": "hand tools or small equipment",
        "containers": "storage or organization items",
        "electronics": "consumer electronics or devices",
        "office_supplies": "desk accessories and supplies",
        "lab_equipment": "scientific or medical instruments",
    }

    # Default physics hints if not provided
    default_physics = {
        "dishes": {"mass_range_kg": [0.2, 0.8], "friction": 0.4, "collision_shape": "convex"},
        "utensils": {"mass_range_kg": [0.02, 0.15], "friction": 0.3, "collision_shape": "convex"},
        "food": {"mass_range_kg": [0.05, 1.0], "friction": 0.5, "collision_shape": "convex"},
        "groceries": {"mass_range_kg": [0.1, 2.0], "friction": 0.4, "collision_shape": "box"},
        "produce": {"mass_range_kg": [0.05, 0.5], "friction": 0.5, "collision_shape": "convex"},
        "bottles": {"mass_range_kg": [0.3, 1.5], "friction": 0.3, "collision_shape": "convex"},
        "cans": {"mass_range_kg": [0.2, 0.8], "friction": 0.4, "collision_shape": "convex"},
        "boxes": {"mass_range_kg": [0.1, 5.0], "friction": 0.5, "collision_shape": "box"},
        "clothing": {"mass_range_kg": [0.1, 1.0], "friction": 0.6, "collision_shape": "convex"},
        "towels": {"mass_range_kg": [0.2, 0.8], "friction": 0.6, "collision_shape": "convex"},
        "tools": {"mass_range_kg": [0.1, 2.0], "friction": 0.5, "collision_shape": "convex"},
        "containers": {"mass_range_kg": [0.1, 1.0], "friction": 0.4, "collision_shape": "box"},
        "electronics": {"mass_range_kg": [0.1, 2.0], "friction": 0.4, "collision_shape": "box"},
        "office_supplies": {"mass_range_kg": [0.01, 0.5], "friction": 0.4, "collision_shape": "box"},
        "lab_equipment": {"mass_range_kg": [0.05, 1.0], "friction": 0.3, "collision_shape": "convex"},
    }

    # Add material hint
    if category in material_hints:
        enriched["material_hint"] = material_hints[category]

    # Add style hint
    if category in style_hints:
        enriched["style_hint"] = style_hints[category]

    # Add/merge physics hints
    if not enriched.get("physics_hints") or not enriched["physics_hints"]:
        enriched["physics_hints"] = default_physics.get(category, {
            "mass_range_kg": [0.1, 1.0],
            "friction": 0.5,
            "collision_shape": "convex"
        })
    else:
        # Merge with defaults for any missing fields
        defaults = default_physics.get(category, {})
        for key, value in defaults.items():
            if key not in enriched["physics_hints"]:
                enriched["physics_hints"][key] = value

    # Add image generation prompt hint
    description = enriched.get("description", "")
    material = enriched.get("material_hint", "appropriate materials")
    style = enriched.get("style_hint", "")

    enriched["generation_prompt_hint"] = (
        f"A {description}, made of {material}. "
        f"{style}. Photorealistic product photography style."
    )

    return enriched


# ============================================================================
# Main Processing
# ============================================================================

def process_scene(
    root: Path,
    scene_id: str,
    seg_prefix: str,
    assets_prefix: str,
    usd_prefix: str,
    replicator_prefix: str,
    requested_policies: Optional[List[str]] = None
) -> ReplicatorBundle:
    """Process a scene and generate Replicator bundle."""

    print(f"[REPLICATOR] Processing scene: {scene_id}")

    # Load scene data
    inventory_path = root / seg_prefix / "inventory_enriched.json"
    if not inventory_path.is_file():
        inventory_path = root / seg_prefix / "inventory.json"
    assets_root = root / assets_prefix

    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing inventory.json at {inventory_path}")

    inventory = load_json(inventory_path)

    scene_assets = load_manifest_or_scene_assets(assets_root)
    if scene_assets is None:
        print(
            f"[REPLICATOR] Warning: no scene manifest found in {assets_root}, using inventory only"
        )
        scene_assets = {"objects": inventory.get("objects", [])}

    # Detect environment type
    scene_type = inventory.get("scene_type", "generic")
    environment_type = detect_environment_type(scene_type, inventory)

    print(f"[REPLICATOR] Detected environment: {environment_type.value}")

    # Create LLM client and analyze scene
    llm_provider = os.getenv("LLM_PROVIDER", "auto").lower()
    mock_mode = llm_provider == "mock"

    client = create_unified_llm_client()
    catalog_client = None if mock_mode else create_catalog_client()

    if mock_mode and not (HAVE_LLM_CLIENT and isinstance(client, LLMClient)):
        print("[REPLICATOR] Mock mode enabled without unified LLM client; using minimal analysis.")
        analysis_result = build_minimal_analysis_result("Mock analysis (client unavailable)")
    elif HAVE_LLM_CLIENT and isinstance(client, LLMClient):
        analysis_result = analyze_scene_with_llm(
            client=client,
            scene_type=scene_type,
            environment_type=environment_type,
            inventory=inventory,
            scene_assets=scene_assets,
            requested_policies=requested_policies
        )
    else:
        analysis_result = analyze_scene_with_gemini(
            client=client,
            scene_type=scene_type,
            environment_type=environment_type,
            inventory=inventory,
            scene_assets=scene_assets,
            requested_policies=requested_policies
        )

    # Parse results into data structures
    placement_regions = []
    for r in analysis_result.get("placement_regions", []):
        region = PlacementRegion(
            name=r.get("name", "unknown_region"),
            description=r.get("description", ""),
            surface_type=r.get("surface_type", "horizontal"),
            parent_object_id=r.get("parent_object_id"),
            position=r.get("position"),
            size=r.get("size"),
            rotation=r.get("rotation"),
            semantic_tags=r.get("semantic_tags", []),
            suitable_for=r.get("suitable_for", [])
        )
        placement_regions.append(region)

    variation_assets = []
    for a in analysis_result.get("variation_assets", []):
        asset = VariationAsset(
            name=a.get("name", "unknown_asset"),
            category=a.get("category", "other"),
            description=a.get("description", ""),
            semantic_class=a.get("semantic_class", "object"),
            priority=a.get("priority", "optional"),
            source_hint=a.get("source_hint"),
            example_variants=a.get("example_variants", []),
            physics_hints=a.get("physics_hints", {})
        )
        variation_assets.append(asset)

    variation_assets = enrich_variation_assets_from_catalog(variation_assets, catalog_client)

    policy_configs = []
    for p in analysis_result.get("policy_configs", []):
        # Map policy_id to PolicyTarget enum
        policy_id = p.get("policy_id", "general_manipulation")
        try:
            policy_target = PolicyTarget(policy_id)
        except ValueError:
            policy_target = PolicyTarget.GENERAL_MANIPULATION

        config = PolicyConfig(
            policy_id=policy_id,
            policy_name=p.get("policy_name", policy_id.replace("_", " ").title()),
            policy_target=policy_target,
            description=p.get("description", ""),
            placement_regions=[PlacementRegion(
                name=r, description="", surface_type="horizontal"
            ) for r in p.get("placement_regions_used", [])],
            variation_assets=[VariationAsset(
                name=a, category="", description="", semantic_class="object", priority="optional"
            ) for a in p.get("variation_assets_used", [])],
            randomizers=[RandomizerConfig(
                name=r.get("name", "unknown"),
                enabled=r.get("enabled", True),
                frequency=r.get("frequency", "per_frame"),
                parameters=r.get("parameters", {})
            ) for r in p.get("randomizers", [])],
            capture_config=p.get("capture_config", {}),
            scene_modifications=p.get("scene_modifications", {})
        )
        policy_configs.append(config)

    # Create bundle
    bundle = ReplicatorBundle(
        scene_id=scene_id,
        environment_type=environment_type,
        scene_type=scene_type,
        policies=policy_configs,
        global_placement_regions=placement_regions,
        global_variation_assets=variation_assets,
        metadata={
            "analysis": analysis_result.get("analysis", {}),
            "source_inventory": str(inventory_path),
            "source_assets": str(assets_root),
        }
    )

    return bundle, analysis_result


def write_replicator_bundle(
    bundle: ReplicatorBundle,
    analysis_result: dict,
    output_dir: Path
) -> None:
    """Write all Replicator bundle files to output directory."""

    print(f"[REPLICATOR] Writing bundle to {output_dir}")

    # Create directories
    ensure_dir(output_dir)
    ensure_dir(output_dir / "policies")
    ensure_dir(output_dir / "configs")
    ensure_dir(output_dir / "variation_assets")
    ensure_dir(output_dir / "placement_regions")

    # 1. Write placement regions USD layer
    usda_content = generate_placement_regions_usda(
        regions=bundle.global_placement_regions,
        scene_id=bundle.scene_id
    )
    (output_dir / "placement_regions.usda").write_text(usda_content)
    (output_dir / "placement_regions" / "placement_regions.usda").write_text(usda_content)
    print(f"[REPLICATOR] Written: placement_regions.usda ({len(bundle.global_placement_regions)} regions)")

    # 2. Write policy-specific Replicator scripts
    for policy in bundle.policies:
        script_content = generate_replicator_script(
            policy_config=policy,
            all_regions=bundle.global_placement_regions,
            all_assets=bundle.global_variation_assets,
            scene_id=bundle.scene_id
        )
        script_path = output_dir / "policies" / f"{policy.policy_id}.py"
        script_path.write_text(script_content)
        print(f"[REPLICATOR] Written: policies/{policy.policy_id}.py")

    # 3. Write master Replicator script
    master_script = generate_master_replicator_script(bundle, bundle.policies)
    (output_dir / "replicator_master.py").write_text(master_script)
    print(f"[REPLICATOR] Written: replicator_master.py")

    # 4. Write policy configurations as YAML/JSON
    for policy in bundle.policies:
        config_dict = {
            "policy_id": policy.policy_id,
            "policy_name": policy.policy_name,
            "policy_target": policy.policy_target.value if isinstance(policy.policy_target, PolicyTarget) else str(policy.policy_target),
            "description": policy.description,
            "placement_regions": [r.name for r in policy.placement_regions] if policy.placement_regions else [],
            "variation_assets": [a.name for a in policy.variation_assets] if policy.variation_assets else [],
            "randomizers": [asdict(r) for r in policy.randomizers] if policy.randomizers else [],
            "capture_config": policy.capture_config,
            "scene_modifications": policy.scene_modifications
        }
        config_path = output_dir / "configs" / f"{policy.policy_id}.json"
        save_json(config_path, config_dict)
    print(f"[REPLICATOR] Written: configs/*.json ({len(bundle.policies)} configs)")

    # 5. Write variation asset manifest (enhanced for variation-gen-job)
    asset_manifest = generate_asset_manifest(
        assets=bundle.global_variation_assets,
        scene_id=bundle.scene_id,
        scene_type=bundle.scene_type,
        environment_type=bundle.environment_type,
        policies=[p.policy_id for p in bundle.policies]
    )
    import datetime
    asset_manifest["generated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    save_json(output_dir / "variation_assets" / "manifest.json", asset_manifest)
    print(f"[REPLICATOR] Written: variation_assets/manifest.json ({len(bundle.global_variation_assets)} assets)")

    # 6. Write bundle metadata
    bundle_meta = {
        "scene_id": bundle.scene_id,
        "environment_type": bundle.environment_type.value,
        "scene_type": bundle.scene_type,
        "policies": [p.policy_id for p in bundle.policies],
        "placement_regions_count": len(bundle.global_placement_regions),
        "variation_assets_count": len(bundle.global_variation_assets),
        "analysis_summary": analysis_result.get("analysis", {}),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    save_json(output_dir / "bundle_metadata.json", bundle_meta)
    print(f"[REPLICATOR] Written: bundle_metadata.json")

    # 7. Write README
    readme_content = generate_readme(bundle)
    (output_dir / "README.md").write_text(readme_content)
    print(f"[REPLICATOR] Written: README.md")


def generate_readme(bundle: ReplicatorBundle) -> str:
    """Generate README for the Replicator bundle."""

    policy_list = "\n".join([
        f"- **{p.policy_name}** (`{p.policy_id}`): {p.description}"
        for p in bundle.policies
    ])

    region_list = "\n".join([
        f"- `{r.name}`: {r.description} ({r.surface_type})"
        for r in bundle.global_placement_regions
    ])

    asset_categories = {}
    for a in bundle.global_variation_assets:
        cat = a.category
        if cat not in asset_categories:
            asset_categories[cat] = []
        asset_categories[cat].append(a.name)

    asset_list = "\n".join([
        f"- **{cat}**: {', '.join(assets)}"
        for cat, assets in asset_categories.items()
    ])

    readme = f'''# Replicator Bundle: {bundle.scene_id}

**Environment Type**: {bundle.environment_type.value}
**Scene Type**: {bundle.scene_type}

This bundle contains everything needed to run domain randomization and synthetic data generation for this scene in NVIDIA Isaac Sim.

## Quick Start

1. Open `scene.usda` in Isaac Sim
2. Load the placement regions layer: `placement_regions.usda`
3. Open the Script Editor and run:

```python
from replicator_master import ReplicatorManager

manager = ReplicatorManager()
manager.list_policies()  # See available policies
manager.run_policy("dish_loading", num_frames=500)  # Run specific policy
```

## Available Policies

{policy_list}

## Placement Regions

{region_list}

## Variation Assets

{asset_list}

See `variation_assets/manifest.json` for full asset details and generation instructions.

## Directory Structure

```
replicator/
├── replicator_master.py      # Main entry point
├── placement_regions.usda    # USD layer with placement surfaces
├── bundle_metadata.json      # Bundle information
├── README.md                 # This file
├── policies/                 # Policy-specific scripts
│   ├── dish_loading.py
│   ├── table_clearing.py
│   └── ...
├── configs/                  # Policy configurations
│   ├── dish_loading.json
│   └── ...
└── variation_assets/         # Assets for domain randomization
    ├── manifest.json         # Asset requirements
    └── *.usdz                # Asset files (to be added)
```

## Adding Variation Assets

The `variation_assets/manifest.json` file lists all assets needed for domain randomization.
Assets can be:
- Generated using the BlueprintPipeline asset generation
- Downloaded from NVIDIA SimReady asset library
- Created manually in USD format

Place USDZ files in the `variation_assets/` directory with names matching the manifest.

## Customization

Edit `configs/<policy_id>.json` to customize:
- Number of objects to spawn
- Randomization parameters
- Capture resolution and annotations
- Which regions to use

## Generated by BlueprintPipeline

This bundle was auto-generated by the BlueprintPipeline replicator-job.
For issues or improvements, see the pipeline documentation.
'''

    return readme


# ============================================================================
# Entry Point
# ============================================================================

def generate_replicator_bundle_job(
    bucket: str,
    scene_id: str,
    seg_prefix: str,
    assets_prefix: str,
    usd_prefix: str,
    replicator_prefix: str,
    requested_policies: Optional[List[str]] = None,
    root: Path = GCS_ROOT,
) -> int:
    try:
        bundle, analysis_result = process_scene(
            root=root,
            scene_id=scene_id,
            seg_prefix=seg_prefix,
            assets_prefix=assets_prefix,
            usd_prefix=usd_prefix,
            replicator_prefix=replicator_prefix,
            requested_policies=requested_policies,
        )
    except Exception as exc:
        print(f"[REPLICATOR] ERROR: {exc}", file=sys.stderr)
        raise

    # GAP-REPLICATOR-001 FIX: Use correct function name write_replicator_bundle
    output_dir = root / replicator_prefix
    write_replicator_bundle(bundle, analysis_result, output_dir)
    print("[REPLICATOR] Bundle generation complete")
    return 0


def main():
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))

    from blueprint_sim.replicator import run_from_env

    return run_from_env(root=GCS_ROOT)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="REPLICATOR", validate_gcs=True)
    sys.exit(main())

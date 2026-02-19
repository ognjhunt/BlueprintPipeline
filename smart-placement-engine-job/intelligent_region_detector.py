"""Intelligent Region Detector using Gemini 3.1 Pro Preview.

This module provides AI-powered detection and understanding of placement regions
within scenes. It uses Gemini to:
- Analyze scene structure and identify placement surfaces
- Understand spatial relationships between objects
- Detect articulation states and their implications for placement
- Recommend optimal placement regions for specific asset types
"""

import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
    HAVE_GENAI = True
except ImportError:
    HAVE_GENAI = False

# Add repo root to path for config imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.config import PIPELINE_CONFIG_PATH, load_pipeline_config
    HAVE_CONFIG = True
except ImportError:
    HAVE_CONFIG = False

from tools.config.production_mode import resolve_production_mode

from .compatibility_matrix import (
    ArticulationState,
    AssetCategory,
    PlacementContext,
    RegionType,
    SceneArchetype,
)


class DetectionConfidence(str, Enum):
    """Confidence level for region detection."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DetectedRegion:
    """A placement region detected by the AI."""
    id: str
    name: str
    region_type: RegionType
    position: List[float]  # [x, y, z] in meters
    size: List[float]  # [width, depth, height] in meters
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    parent_object_id: Optional[str] = None
    parent_object_name: Optional[str] = None
    surface_type: str = "horizontal"  # horizontal, vertical, volume
    articulation_state: Optional[ArticulationState] = None
    semantic_tags: List[str] = field(default_factory=list)
    suitable_for: List[str] = field(default_factory=list)
    confidence: DetectionConfidence = DetectionConfidence.MEDIUM
    reasoning: str = ""  # AI's reasoning for this detection
    clearance_above_m: Optional[float] = None
    accessibility_score: float = 1.0  # 0-1, how easy to access
    occlusion_factor: float = 0.0  # 0-1, how occluded by other objects


@dataclass
class RegionDetectionResult:
    """Result of region detection for a scene."""
    scene_id: str
    scene_archetype: SceneArchetype
    detected_regions: List[DetectedRegion]
    scene_understanding: Dict[str, Any]
    placement_recommendations: List[Dict[str, Any]]
    raw_ai_response: Optional[str] = None
    tokens_used: int = 0


class IntelligentRegionDetector:
    """AI-powered region detection using Gemini 3.1 Pro Preview.

    This detector analyzes scene manifests, USD files, and optionally images
    to understand the spatial layout and identify optimal placement regions.
    """

    @staticmethod
    def _get_default_model() -> str:
        """Get default model from config or fallback to hardcoded value."""
        if HAVE_CONFIG:
            try:
                config = load_pipeline_config(validate=False)
                model_config = config.models.get_model("intelligent_region_detector")
                if model_config:
                    return model_config.default_model
            except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
                logging.warning(
                    "Failed to load pipeline config from %s; using default model. Error: %s",
                    PIPELINE_CONFIG_PATH,
                    exc,
                )
                if resolve_production_mode():
                    raise
        return "gemini-3-flash-preview"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_web_search: bool = False,
    ):
        """Initialize the detector.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model to use (defaults from config if not provided)
            enable_web_search: Whether to enable web search grounding
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or self._get_default_model()
        self.enable_web_search = enable_web_search
        self._client: Optional["genai.Client"] = None

    def _get_client(self) -> "genai.Client":
        """Get or create the Gemini client."""
        if not HAVE_GENAI:
            raise RuntimeError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )

        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Please set it or pass api_key to the constructor."
                )
            self._client = genai.Client(api_key=self.api_key)

        return self._client

    def _build_region_detection_prompt(
        self,
        scene_manifest: Dict[str, Any],
        scene_archetype: SceneArchetype,
        target_assets: Optional[List[AssetCategory]] = None,
    ) -> str:
        """Build the prompt for region detection.

        Args:
            scene_manifest: The scene manifest JSON
            scene_archetype: The type of scene
            target_assets: Optional list of asset categories to place

        Returns:
            The formatted prompt string
        """
        target_assets_str = ""
        if target_assets:
            target_assets_str = f"""
Target Assets to Place:
{json.dumps([a.value for a in target_assets], indent=2)}

For each target asset, identify the best placement regions considering:
- Contextual appropriateness (dishes go in dishwashers, not on floors)
- Articulation states (open drawers/cabinets can receive items)
- Surface suitability (horizontal for stacking, vertical for hanging)
- Accessibility for robot manipulation
"""

        prompt = f"""You are an expert in robotic simulation environments and spatial reasoning.
Analyze this scene manifest and identify all viable placement regions for robotic manipulation tasks.

Scene Type: {scene_archetype.value}

Scene Manifest:
```json
{json.dumps(scene_manifest, indent=2)}
```
{target_assets_str}
For each placement region, provide:
1. A unique ID (e.g., "counter_surface_01", "dishwasher_rack_upper")
2. The region type (counter, shelf, drawer, dishwasher, cabinet, sink, table, floor, rack_level, etc.)
3. Position [x, y, z] in meters (center of the region)
4. Size [width, depth, height] in meters
5. Parent object ID (the furniture/appliance this region belongs to)
6. Surface type: "horizontal", "vertical", or "volume"
7. Articulation state if applicable: "open", "closed", "partially_open"
8. Semantic tags describing the region (e.g., ["prep_area", "food_surface", "wet"])
9. What asset types are suitable (e.g., ["dishes", "utensils", "food_items"])
10. Clearance above the surface in meters
11. Accessibility score (0-1, how easy for a robot to reach)
12. Your reasoning for identifying this region

Important considerations:
- Identify surfaces on counters, tables, shelves, and inside open containers
- For articulated objects (drawers, dishwashers, cabinets), only create regions if they are OPEN
- Consider robot reachability and manipulation constraints
- Prioritize surfaces that are unoccluded and at manipulable heights (0.5m - 1.5m preferred)
- For dishwashers and appliances with racks, identify upper/lower rack regions separately
- For drawers, consider the drawer's extended position, not its stored position
- Kitchen sinks and dish pits are valid placement regions for dirty dishes

Respond with a JSON object in this exact format:
{{
  "scene_understanding": {{
    "primary_workspace_area": "description of main work area",
    "articulated_objects_detected": ["list of articulated objects found"],
    "open_containers": ["list of open containers/drawers/dishwashers"],
    "surface_heights_m": {{"lowest": 0.0, "highest": 1.5}},
    "estimated_floor_area_m2": 10.0,
    "key_observations": ["observation 1", "observation 2"]
  }},
  "detected_regions": [
    {{
      "id": "region_id",
      "name": "Human readable name",
      "region_type": "counter",
      "position": [x, y, z],
      "size": [width, depth, height],
      "rotation": [rx, ry, rz],
      "parent_object_id": "object_id",
      "parent_object_name": "Object Name",
      "surface_type": "horizontal",
      "articulation_state": "open",
      "semantic_tags": ["tag1", "tag2"],
      "suitable_for": ["dishes", "utensils"],
      "clearance_above_m": 0.5,
      "accessibility_score": 0.9,
      "occlusion_factor": 0.1,
      "reasoning": "Why this region was identified"
    }}
  ],
  "placement_recommendations": [
    {{
      "asset_type": "dishes",
      "recommended_regions": ["region_id_1", "region_id_2"],
      "priority_order_reasoning": "Why this order"
    }}
  ]
}}
"""
        return prompt

    def _build_image_analysis_prompt(
        self,
        scene_archetype: SceneArchetype,
        existing_regions: Optional[List[DetectedRegion]] = None,
    ) -> str:
        """Build prompt for image-based region analysis.

        Args:
            scene_archetype: The type of scene
            existing_regions: Previously detected regions to validate/enhance

        Returns:
            The formatted prompt string
        """
        existing_str = ""
        if existing_regions:
            existing_str = f"""
Previously Detected Regions (validate and enhance these):
```json
{json.dumps([{
    "id": r.id,
    "region_type": r.region_type.value,
    "position": r.position,
    "suitable_for": r.suitable_for
} for r in existing_regions], indent=2)}
```
"""

        prompt = f"""Analyze this image of a {scene_archetype.value} environment.
Identify all surfaces and regions where objects could be placed for robotic manipulation.
{existing_str}
Look for:
1. Horizontal surfaces (counters, tables, shelves)
2. Open containers (drawers, dishwashers, cabinets)
3. Storage areas (racks, shelving units)
4. Specialized areas (sinks, drying racks)

For each region, estimate:
- Position relative to scene center
- Approximate dimensions in meters
- What types of objects would naturally be placed there
- How accessible it is for robotic manipulation

Provide your analysis as JSON with the same format as the previous analysis.
Focus on visual details that may not be in the manifest:
- Actual articulation states visible in the image
- Clutter or occlusions
- Precise surface boundaries
"""
        return prompt

    def _parse_region_type(self, type_str: str) -> RegionType:
        """Parse a region type string into the enum."""
        type_str = type_str.lower().replace(" ", "_").replace("-", "_")

        # Direct mapping
        try:
            return RegionType(type_str)
        except ValueError:
            pass

        # Fuzzy matching
        mappings = {
            "countertop": RegionType.COUNTER,
            "work_surface": RegionType.COUNTER,
            "prep_table": RegionType.PREP_SURFACE,
            "preparation_surface": RegionType.PREP_SURFACE,
            "serving_area": RegionType.SERVING_SURFACE,
            "dishwasher_rack": RegionType.DISHWASHER,
            "dish_rack": RegionType.DRYING_RACK,
            "storage_shelf": RegionType.SHELF,
            "warehouse_rack": RegionType.RACK_LEVEL,
            "pallet_spot": RegionType.PALLET_POSITION,
            "lab_bench": RegionType.BENCH,
            "workbench": RegionType.BENCH,
            "office_desk": RegionType.DESK,
            "work_desk": RegionType.DESK,
        }

        return mappings.get(type_str, RegionType.SHELF)  # Default to shelf

    def _parse_articulation_state(
        self,
        state_str: Optional[str],
    ) -> Optional[ArticulationState]:
        """Parse articulation state string."""
        if not state_str:
            return None

        state_str = state_str.lower()
        if "open" in state_str:
            if "partial" in state_str:
                return ArticulationState.PARTIALLY_OPEN
            return ArticulationState.OPEN
        if "closed" in state_str:
            return ArticulationState.CLOSED
        if "extended" in state_str:
            return ArticulationState.EXTENDED
        if "retracted" in state_str:
            return ArticulationState.RETRACTED

        return None

    def _parse_confidence(self, reasoning: str) -> DetectionConfidence:
        """Estimate confidence from reasoning text."""
        reasoning_lower = reasoning.lower()
        if any(word in reasoning_lower for word in ["clearly", "definitely", "visible", "explicit"]):
            return DetectionConfidence.HIGH
        if any(word in reasoning_lower for word in ["likely", "probably", "appears", "seems"]):
            return DetectionConfidence.MEDIUM
        return DetectionConfidence.LOW

    def _parse_ai_response(
        self,
        response_text: str,
        scene_id: str,
        scene_archetype: SceneArchetype,
    ) -> RegionDetectionResult:
        """Parse the AI response into structured data.

        Args:
            response_text: Raw JSON response from AI
            scene_id: The scene identifier
            scene_archetype: The scene type

        Returns:
            Parsed RegionDetectionResult
        """
        # Clean up response - handle markdown code blocks
        text = response_text.strip()
        if text.startswith("```"):
            # Remove markdown code block markers
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse AI response as JSON: {e}")
            else:
                raise ValueError(f"No valid JSON found in AI response: {e}")

        # Parse detected regions
        detected_regions: List[DetectedRegion] = []
        for region_data in data.get("detected_regions", []):
            try:
                region = DetectedRegion(
                    id=region_data.get("id", f"region_{len(detected_regions)}"),
                    name=region_data.get("name", "Unnamed Region"),
                    region_type=self._parse_region_type(
                        region_data.get("region_type", "shelf")
                    ),
                    position=region_data.get("position", [0.0, 0.0, 0.0]),
                    size=region_data.get("size", [0.5, 0.5, 0.1]),
                    rotation=region_data.get("rotation", [0.0, 0.0, 0.0]),
                    parent_object_id=region_data.get("parent_object_id"),
                    parent_object_name=region_data.get("parent_object_name"),
                    surface_type=region_data.get("surface_type", "horizontal"),
                    articulation_state=self._parse_articulation_state(
                        region_data.get("articulation_state")
                    ),
                    semantic_tags=region_data.get("semantic_tags", []),
                    suitable_for=region_data.get("suitable_for", []),
                    clearance_above_m=region_data.get("clearance_above_m"),
                    accessibility_score=region_data.get("accessibility_score", 1.0),
                    occlusion_factor=region_data.get("occlusion_factor", 0.0),
                    reasoning=region_data.get("reasoning", ""),
                    confidence=self._parse_confidence(
                        region_data.get("reasoning", "")
                    ),
                )
                detected_regions.append(region)
            except Exception as e:
                print(f"[REGION_DETECTOR] Warning: Failed to parse region: {e}")
                continue

        return RegionDetectionResult(
            scene_id=scene_id,
            scene_archetype=scene_archetype,
            detected_regions=detected_regions,
            scene_understanding=data.get("scene_understanding", {}),
            placement_recommendations=data.get("placement_recommendations", []),
            raw_ai_response=response_text,
        )

    def detect_regions_from_manifest(
        self,
        scene_manifest: Dict[str, Any],
        scene_id: str,
        scene_archetype: Optional[SceneArchetype] = None,
        target_assets: Optional[List[AssetCategory]] = None,
    ) -> RegionDetectionResult:
        """Detect placement regions from a scene manifest.

        Args:
            scene_manifest: The scene manifest JSON
            scene_id: Unique identifier for the scene
            scene_archetype: Scene type (auto-detected if not provided)
            target_assets: Optional list of asset categories to optimize for

        Returns:
            RegionDetectionResult with detected regions and recommendations
        """
        # Auto-detect archetype if not provided
        if scene_archetype is None:
            env_type = scene_manifest.get("environment", {}).get("type", "generic")
            try:
                scene_archetype = SceneArchetype(env_type.lower())
            except ValueError:
                scene_archetype = SceneArchetype.GENERIC

        client = self._get_client()

        # Build the prompt
        prompt = self._build_region_detection_prompt(
            scene_manifest,
            scene_archetype,
            target_assets,
        )

        # Configure generation
        tools = [
            types.Tool(url_context=types.UrlContext()),
            types.Tool(googleSearch=types.GoogleSearch()),
        ]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            response_mime_type="application/json",
            tools=tools,
        )

        # Generate response
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                response_text += chunk.text

        # Parse and return result
        result = self._parse_ai_response(response_text, scene_id, scene_archetype)
        return result

    def detect_regions_from_image(
        self,
        image_path: str,
        scene_id: str,
        scene_archetype: SceneArchetype,
        existing_regions: Optional[List[DetectedRegion]] = None,
    ) -> RegionDetectionResult:
        """Detect/enhance placement regions using an image.

        Args:
            image_path: Path to scene image
            scene_id: Unique identifier for the scene
            scene_archetype: Scene type
            existing_regions: Previously detected regions to enhance

        Returns:
            RegionDetectionResult with detected/enhanced regions
        """
        client = self._get_client()

        # Read and encode image
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Determine mime type
        suffix = image_path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        # Build prompt
        prompt = self._build_image_analysis_prompt(scene_archetype, existing_regions)

        # Configure generation
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            response_mime_type="application/json",
            tools=[
                types.Tool(url_context=types.UrlContext()),
                types.Tool(googleSearch=types.GoogleSearch()),
            ],
        )

        # Generate response with image
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        data=base64.standard_b64decode(image_data),
                        mime_type=mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                response_text += chunk.text

        # Parse and return result
        result = self._parse_ai_response(response_text, scene_id, scene_archetype)
        return result

    def enhance_regions_with_context(
        self,
        regions: List[DetectedRegion],
        scene_manifest: Dict[str, Any],
        scene_archetype: SceneArchetype,
    ) -> List[DetectedRegion]:
        """Enhance detected regions with additional contextual intelligence.

        Uses AI to add semantic understanding to detected regions.

        Args:
            regions: Previously detected regions
            scene_manifest: The scene manifest
            scene_archetype: The scene type

        Returns:
            Enhanced list of regions with better semantic tags and suitability
        """
        if not regions:
            return regions

        client = self._get_client()

        prompt = f"""Given these detected placement regions in a {scene_archetype.value} scene,
enhance each region with:
1. More specific semantic tags based on context
2. Better suitable_for lists based on robotics best practices
3. Adjusted accessibility scores based on robot reachability
4. Occlusion factors based on scene layout

Current Regions:
```json
{json.dumps([{
    "id": r.id,
    "name": r.name,
    "region_type": r.region_type.value,
    "position": r.position,
    "size": r.size,
    "parent_object_name": r.parent_object_name,
    "articulation_state": r.articulation_state.value if r.articulation_state else None,
    "semantic_tags": r.semantic_tags,
    "suitable_for": r.suitable_for,
} for r in regions], indent=2)}
```

Scene Context:
```json
{json.dumps(scene_manifest.get("environment", {}), indent=2)}
```

Consider:
- Kitchen counters near sinks are "wet" surfaces
- Surfaces near stoves may be "heated"
- Upper cabinets have lower accessibility
- Dishwasher racks have specific dish placement patterns
- Warehouse rack levels have weight constraints

Return a JSON array with enhanced region data in the same format.
"""

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            response_mime_type="application/json",
            tools=[
                types.Tool(url_context=types.UrlContext()),
                types.Tool(googleSearch=types.GoogleSearch()),
            ],
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                response_text += chunk.text

        # Parse enhanced regions
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            enhanced_data = json.loads(text)

            # Update original regions with enhanced data
            region_map = {r.id: r for r in regions}
            for enh in enhanced_data:
                region_id = enh.get("id")
                if region_id in region_map:
                    region = region_map[region_id]
                    if "semantic_tags" in enh:
                        region.semantic_tags = enh["semantic_tags"]
                    if "suitable_for" in enh:
                        region.suitable_for = enh["suitable_for"]
                    if "accessibility_score" in enh:
                        region.accessibility_score = enh["accessibility_score"]
                    if "occlusion_factor" in enh:
                        region.occlusion_factor = enh["occlusion_factor"]

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[REGION_DETECTOR] Warning: Failed to parse enhancement: {e}")

        return regions

    def get_best_regions_for_asset(
        self,
        regions: List[DetectedRegion],
        asset_category: AssetCategory,
        max_results: int = 5,
    ) -> List[Tuple[DetectedRegion, float]]:
        """Get the best regions for placing a specific asset type.

        Args:
            regions: Available placement regions
            asset_category: The asset category to place
            max_results: Maximum number of results to return

        Returns:
            List of (region, score) tuples, sorted by score descending
        """
        scored_regions: List[Tuple[DetectedRegion, float]] = []

        for region in regions:
            score = 0.0

            # Check if asset is suitable for this region
            asset_str = asset_category.value.lower()
            if asset_str in [s.lower() for s in region.suitable_for]:
                score += 50.0

            # Accessibility bonus
            score += region.accessibility_score * 20.0

            # Occlusion penalty
            score -= region.occlusion_factor * 15.0

            # Confidence bonus
            confidence_bonus = {
                DetectionConfidence.HIGH: 10.0,
                DetectionConfidence.MEDIUM: 5.0,
                DetectionConfidence.LOW: 0.0,
            }
            score += confidence_bonus.get(region.confidence, 0.0)

            # Articulation state bonus (open containers are more accessible)
            if region.articulation_state == ArticulationState.OPEN:
                score += 10.0

            scored_regions.append((region, score))

        # Sort by score descending
        scored_regions.sort(key=lambda x: x[1], reverse=True)

        return scored_regions[:max_results]


def create_region_detector(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> IntelligentRegionDetector:
    """Factory function to create a region detector.

    Args:
        api_key: Optional Gemini API key
        model: Model to use (defaults from config if not provided)

    Returns:
        Configured IntelligentRegionDetector instance
    """
    return IntelligentRegionDetector(api_key=api_key, model=model)


if __name__ == "__main__":
    # Test with a sample manifest
    sample_manifest = {
        "scene_id": "kitchen_001",
        "environment": {
            "type": "kitchen",
            "sub_type": "commercial_prep"
        },
        "objects": [
            {
                "id": "counter_01",
                "name": "Prep Counter",
                "category": "furniture",
                "dimensions": {"width": 2.0, "depth": 0.6, "height": 0.9}
            },
            {
                "id": "dishwasher_01",
                "name": "Commercial Dishwasher",
                "category": "appliance",
                "articulation": {"door": {"state": "open", "angle": 90}}
            },
            {
                "id": "cabinet_01",
                "name": "Upper Cabinet",
                "category": "storage",
                "articulation": {"door_left": {"state": "closed"}}
            }
        ]
    }

    # Note: This will only work with a valid GEMINI_API_KEY
    if os.environ.get("GEMINI_API_KEY"):
        detector = create_region_detector()
        result = detector.detect_regions_from_manifest(
            sample_manifest,
            scene_id="kitchen_001",
            scene_archetype=SceneArchetype.KITCHEN,
            target_assets=[AssetCategory.DISHES, AssetCategory.UTENSILS],
        )
        print(f"Detected {len(result.detected_regions)} regions")
        for region in result.detected_regions:
            print(f"  - {region.name}: {region.region_type.value}")
    else:
        print("Set GEMINI_API_KEY to test region detection")

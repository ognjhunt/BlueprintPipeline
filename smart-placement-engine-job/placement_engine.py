"""Smart Placement Engine with Collision Awareness.

This module provides intelligent placement of assets in scenes with:
- Collision detection and avoidance
- Physics-aware placement (stability, weight constraints)
- Contextual placement rules from compatibility matrix
- Optimal distribution within regions
"""

import json
import logging
import math
import os
import random
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
    from tools.config import load_pipeline_config
    HAVE_CONFIG = True
except ImportError:
    HAVE_CONFIG = False

from .compatibility_matrix import (
    ArticulationState,
    AssetCategory,
    AssetPlacementRule,
    CompatibilityMatrix,
    PlacementContext,
    RegionType,
    SceneArchetype,
    get_compatibility_matrix,
)

logger = logging.getLogger(__name__)
from .intelligent_region_detector import (
    DetectedRegion,
    IntelligentRegionDetector,
    RegionDetectionResult,
)


class PlacementStatus(str, Enum):
    """Status of a placement attempt."""
    SUCCESS = "success"
    COLLISION = "collision"
    OUT_OF_BOUNDS = "out_of_bounds"
    WEIGHT_EXCEEDED = "weight_exceeded"
    STABILITY_FAILED = "stability_failed"
    INCOMPATIBLE = "incompatible"
    REGION_FULL = "region_full"


@dataclass
class BoundingBox:
    """3D bounding box for collision detection."""
    center: List[float]  # [x, y, z]
    size: List[float]  # [width, depth, height]
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @property
    def min_corner(self) -> List[float]:
        """Get minimum corner of AABB."""
        return [
            self.center[0] - self.size[0] / 2,
            self.center[1] - self.size[1] / 2,
            self.center[2] - self.size[2] / 2,
        ]

    @property
    def max_corner(self) -> List[float]:
        """Get maximum corner of AABB."""
        return [
            self.center[0] + self.size[0] / 2,
            self.center[1] + self.size[1] / 2,
            self.center[2] + self.size[2] / 2,
        ]

    def intersects(self, other: "BoundingBox", margin: float = 0.01) -> bool:
        """Check if this box intersects another (AABB test with margin)."""
        for i in range(3):
            if (self.max_corner[i] + margin < other.min_corner[i] or
                self.min_corner[i] - margin > other.max_corner[i]):
                return False
        return True

    def contains_point(self, point: List[float], margin: float = 0.0) -> bool:
        """Check if a point is inside this box."""
        for i in range(3):
            if (point[i] < self.min_corner[i] - margin or
                point[i] > self.max_corner[i] + margin):
                return False
        return True

    def volume(self) -> float:
        """Calculate volume of the box."""
        return self.size[0] * self.size[1] * self.size[2]


@dataclass
class AssetInstance:
    """An instance of an asset to be placed."""
    asset_id: str
    asset_name: str
    category: AssetCategory
    bounding_box: BoundingBox
    mass_kg: float = 1.0
    semantic_class: str = ""
    stackable: bool = True
    graspable: bool = True
    fragile: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacementCandidate:
    """A candidate placement position."""
    position: List[float]  # [x, y, z]
    rotation: List[float]  # [rx, ry, rz] in degrees
    region_id: str
    score: float = 0.0
    is_stacked: bool = False
    stack_level: int = 0
    supporting_asset_id: Optional[str] = None


@dataclass
class CollisionCheckResult:
    """Result of a collision check."""
    has_collision: bool
    colliding_object_ids: List[str] = field(default_factory=list)
    penetration_depth_m: float = 0.0
    nearest_object_distance_m: float = float("inf")


@dataclass
class PlacementResult:
    """Result of a placement attempt."""
    asset: AssetInstance
    status: PlacementStatus
    final_position: Optional[List[float]] = None
    final_rotation: Optional[List[float]] = None
    region_id: Optional[str] = None
    collision_check: Optional[CollisionCheckResult] = None
    candidates_evaluated: int = 0
    reasoning: str = ""


@dataclass
class PlacementPlan:
    """A complete plan for placing multiple assets."""
    scene_id: str
    placements: List[PlacementResult]
    regions_used: Dict[str, int]  # region_id -> count of placements
    total_assets_placed: int = 0
    total_collisions_avoided: int = 0
    ai_reasoning: str = ""


class SmartPlacementEngine:
    """Intelligent placement engine with collision awareness.

    This engine combines:
    - Compatibility matrix for contextual rules
    - AI-powered region detection
    - Physics-based collision detection
    - Optimal distribution algorithms
    """

    @staticmethod
    def _get_default_model() -> str:
        """Get default model from config or fallback to hardcoded value."""
        if HAVE_CONFIG:
            try:
                config = load_pipeline_config(validate=False)
                model_config = config.models.get_model("placement_engine")
                if model_config:
                    return model_config.default_model
            except Exception:
                logger.warning(
                    "[PLACEMENT-ENGINE] Failed to load pipeline config for default model; using fallback.",
                    exc_info=True,
                )
        return "gemini-3-pro-preview"

    def __init__(
        self,
        compatibility_matrix: Optional[CompatibilityMatrix] = None,
        region_detector: Optional[IntelligentRegionDetector] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        collision_margin_m: float = 0.01,
        enable_stacking: bool = True,
        randomize_placement: bool = True,
    ):
        """Initialize the placement engine.

        Args:
            compatibility_matrix: Matrix for scene-to-asset rules
            region_detector: AI-powered region detector
            api_key: Gemini API key for AI features
            model: LLM model to use (defaults from config if not provided)
            collision_margin_m: Margin for collision detection
            enable_stacking: Whether to allow stacking objects
            randomize_placement: Whether to add randomness to placement
        """
        self.matrix = compatibility_matrix or get_compatibility_matrix()
        self.region_detector = region_detector
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or self._get_default_model()
        self.collision_margin = collision_margin_m
        self.enable_stacking = enable_stacking
        self.randomize_placement = randomize_placement

        # Placed objects for collision tracking
        self._placed_objects: Dict[str, Tuple[AssetInstance, PlacementCandidate]] = {}

        # Region capacity tracking
        self._region_usage: Dict[str, List[BoundingBox]] = {}

        # AI client for complex decisions
        self._client: Optional["genai.Client"] = None

    def reset(self) -> None:
        """Reset the engine state for a new scene."""
        self._placed_objects.clear()
        self._region_usage.clear()

    def _get_client(self) -> Optional["genai.Client"]:
        """Get or create the Gemini client."""
        if not HAVE_GENAI:
            return None

        if self._client is None and self.api_key:
            self._client = genai.Client(api_key=self.api_key)

        return self._client

    def check_collision(
        self,
        asset: AssetInstance,
        position: List[float],
        rotation: List[float],
    ) -> CollisionCheckResult:
        """Check for collisions at a given position.

        Args:
            asset: The asset to place
            position: Target position [x, y, z]
            rotation: Target rotation [rx, ry, rz]

        Returns:
            CollisionCheckResult with collision details
        """
        # Create bounding box at target position
        target_box = BoundingBox(
            center=position,
            size=asset.bounding_box.size,
            rotation=rotation,
        )

        colliding_ids: List[str] = []
        min_distance = float("inf")
        max_penetration = 0.0

        for obj_id, (placed_asset, placement) in self._placed_objects.items():
            placed_box = BoundingBox(
                center=placement.position,
                size=placed_asset.bounding_box.size,
                rotation=placement.rotation,
            )

            if target_box.intersects(placed_box, margin=self.collision_margin):
                colliding_ids.append(obj_id)

                # Calculate penetration depth (approximate)
                for i in range(3):
                    overlap = min(
                        target_box.max_corner[i] - placed_box.min_corner[i],
                        placed_box.max_corner[i] - target_box.min_corner[i],
                    )
                    if overlap > 0:
                        max_penetration = max(max_penetration, overlap)
            else:
                # Calculate distance to nearest placed object
                distance = self._calculate_distance(target_box, placed_box)
                min_distance = min(min_distance, distance)

        return CollisionCheckResult(
            has_collision=len(colliding_ids) > 0,
            colliding_object_ids=colliding_ids,
            penetration_depth_m=max_penetration,
            nearest_object_distance_m=min_distance,
        )

    def _calculate_distance(
        self,
        box1: BoundingBox,
        box2: BoundingBox,
    ) -> float:
        """Calculate approximate distance between two boxes."""
        # Use center-to-center distance minus radii
        dx = box1.center[0] - box2.center[0]
        dy = box1.center[1] - box2.center[1]
        dz = box1.center[2] - box2.center[2]

        center_distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Approximate radius as half diagonal of larger dimension
        r1 = max(box1.size) / 2
        r2 = max(box2.size) / 2

        return max(0.0, center_distance - r1 - r2)

    def _check_region_bounds(
        self,
        asset: AssetInstance,
        position: List[float],
        region: DetectedRegion,
    ) -> bool:
        """Check if asset placement is within region bounds.

        Args:
            asset: The asset to place
            position: Target position
            region: The placement region

        Returns:
            True if within bounds
        """
        asset_half_size = [s / 2 for s in asset.bounding_box.size]

        region_min = [
            region.position[0] - region.size[0] / 2,
            region.position[1] - region.size[1] / 2,
            region.position[2],  # Bottom of region
        ]
        region_max = [
            region.position[0] + region.size[0] / 2,
            region.position[1] + region.size[1] / 2,
            region.position[2] + (region.clearance_above_m or 1.0),
        ]

        # Check each dimension
        for i in range(3):
            asset_min = position[i] - asset_half_size[i]
            asset_max = position[i] + asset_half_size[i]

            if asset_min < region_min[i] - self.collision_margin:
                return False
            if asset_max > region_max[i] + self.collision_margin:
                return False

        return True

    def generate_placement_candidates(
        self,
        asset: AssetInstance,
        region: DetectedRegion,
        rule: AssetPlacementRule,
        num_candidates: int = 20,
    ) -> List[PlacementCandidate]:
        """Generate candidate placement positions within a region.

        Args:
            asset: The asset to place
            region: The placement region
            rule: The placement rule governing this combination
            num_candidates: Number of candidates to generate

        Returns:
            List of placement candidates
        """
        candidates: List[PlacementCandidate] = []

        # Calculate available area
        region_width = region.size[0]
        region_depth = region.size[1]
        asset_width = asset.bounding_box.size[0]
        asset_depth = asset.bounding_box.size[1]
        asset_height = asset.bounding_box.size[2]

        # Minimum spacing between objects
        spacing = max(self.collision_margin * 2, 0.02)

        # Calculate base height (top of region surface)
        base_z = region.position[2] + region.size[2] / 2

        # Grid-based placement for clustering
        if rule.clustering_enabled:
            # Calculate grid dimensions
            cols = max(1, int((region_width - asset_width) / (asset_width + spacing)) + 1)
            rows = max(1, int((region_depth - asset_depth) / (asset_depth + spacing)) + 1)

            for row in range(rows):
                for col in range(cols):
                    if len(candidates) >= num_candidates:
                        break

                    # Calculate position
                    x = (region.position[0] - region_width / 2 +
                         asset_width / 2 + col * (asset_width + spacing))
                    y = (region.position[1] - region_depth / 2 +
                         asset_depth / 2 + row * (asset_depth + spacing))
                    z = base_z + asset_height / 2

                    # Add some randomness
                    if self.randomize_placement:
                        x += random.uniform(-spacing / 2, spacing / 2)
                        y += random.uniform(-spacing / 2, spacing / 2)

                    position = [x, y, z]
                    rotation = [0.0, 0.0, random.uniform(-5, 5) if self.randomize_placement else 0.0]

                    candidates.append(PlacementCandidate(
                        position=position,
                        rotation=rotation,
                        region_id=region.id,
                        score=0.0,
                    ))

        else:
            # Random placement for non-clustered items
            for _ in range(num_candidates):
                x = random.uniform(
                    region.position[0] - region_width / 2 + asset_width / 2,
                    region.position[0] + region_width / 2 - asset_width / 2,
                )
                y = random.uniform(
                    region.position[1] - region_depth / 2 + asset_depth / 2,
                    region.position[1] + region_depth / 2 - asset_depth / 2,
                )
                z = base_z + asset_height / 2

                position = [x, y, z]
                rotation = [0.0, 0.0, random.uniform(0, 360)]

                candidates.append(PlacementCandidate(
                    position=position,
                    rotation=rotation,
                    region_id=region.id,
                    score=0.0,
                ))

        # Generate stacking candidates if enabled
        if self.enable_stacking and rule.max_stack_height > 1 and asset.stackable:
            stacking_candidates = self._generate_stacking_candidates(
                asset, region, rule
            )
            candidates.extend(stacking_candidates)

        return candidates

    def _generate_stacking_candidates(
        self,
        asset: AssetInstance,
        region: DetectedRegion,
        rule: AssetPlacementRule,
    ) -> List[PlacementCandidate]:
        """Generate candidates for stacking on existing objects.

        Args:
            asset: The asset to stack
            region: The placement region
            rule: The placement rule

        Returns:
            List of stacking candidates
        """
        candidates: List[PlacementCandidate] = []

        # Find stackable objects in this region
        for obj_id, (placed_asset, placement) in self._placed_objects.items():
            if placement.region_id != region.id:
                continue

            # Check if we can stack on this object
            current_stack = placement.stack_level + 1
            if current_stack >= rule.max_stack_height:
                continue

            # Check if categories are compatible for stacking
            if placed_asset.category != asset.category:
                continue

            # Calculate stacking position
            stack_z = (placement.position[2] +
                       placed_asset.bounding_box.size[2] / 2 +
                       asset.bounding_box.size[2] / 2)

            # Check clearance
            clearance = region.clearance_above_m or 1.0
            if stack_z + asset.bounding_box.size[2] / 2 > region.position[2] + clearance:
                continue

            position = [
                placement.position[0],
                placement.position[1],
                stack_z,
            ]

            # Slight rotation variation for realism
            rotation = [
                0.0,
                0.0,
                placement.rotation[2] + random.uniform(-5, 5) if self.randomize_placement else 0.0,
            ]

            candidates.append(PlacementCandidate(
                position=position,
                rotation=rotation,
                region_id=region.id,
                score=5.0,  # Bonus for stacking (fills vertical space)
                is_stacked=True,
                stack_level=current_stack,
                supporting_asset_id=obj_id,
            ))

        return candidates

    def score_candidate(
        self,
        candidate: PlacementCandidate,
        asset: AssetInstance,
        region: DetectedRegion,
        collision_result: CollisionCheckResult,
    ) -> float:
        """Score a placement candidate.

        Args:
            candidate: The candidate to score
            asset: The asset being placed
            region: The placement region
            collision_result: Result of collision check

        Returns:
            Score (higher is better, negative means invalid)
        """
        if collision_result.has_collision:
            return -100.0

        score = candidate.score

        # Accessibility bonus
        score += region.accessibility_score * 10.0

        # Distance from other objects (prefer some spacing)
        if collision_result.nearest_object_distance_m < 0.05:
            score -= 5.0
        elif collision_result.nearest_object_distance_m > 0.1:
            score += 2.0

        # Prefer center of region
        region_center = region.position[:2]
        candidate_pos = candidate.position[:2]
        distance_to_center = math.sqrt(
            (candidate_pos[0] - region_center[0]) ** 2 +
            (candidate_pos[1] - region_center[1]) ** 2
        )
        max_distance = math.sqrt(region.size[0] ** 2 + region.size[1] ** 2) / 2
        if max_distance > 0:
            center_score = (1 - distance_to_center / max_distance) * 5.0
            score += center_score

        # Stacking bonus (efficient use of space)
        if candidate.is_stacked:
            score += 3.0

        return score

    def place_asset(
        self,
        asset: AssetInstance,
        regions: List[DetectedRegion],
        scene_archetype: SceneArchetype,
    ) -> PlacementResult:
        """Place an asset in the best available location.

        Args:
            asset: The asset to place
            regions: Available placement regions
            scene_archetype: The scene archetype

        Returns:
            PlacementResult with placement details
        """
        best_candidate: Optional[PlacementCandidate] = None
        best_score = float("-inf")
        best_region: Optional[DetectedRegion] = None
        best_collision: Optional[CollisionCheckResult] = None
        total_candidates = 0

        # Get compatible regions for this asset
        compatible_regions = []
        for region in regions:
            context = PlacementContext(
                scene_archetype=scene_archetype,
                region_type=region.region_type,
                region_id=region.id,
                articulation_state=region.articulation_state,
                semantic_tags=region.semantic_tags,
            )

            compatible = self.matrix.get_compatible_assets(context)
            for cat, rule in compatible:
                if cat == asset.category:
                    compatible_regions.append((region, rule))
                    break

        if not compatible_regions:
            return PlacementResult(
                asset=asset,
                status=PlacementStatus.INCOMPATIBLE,
                reasoning=f"No compatible regions for {asset.category.value}",
            )

        # Sort regions by rule priority
        compatible_regions.sort(key=lambda x: x[1].priority, reverse=True)

        # Try each region
        for region, rule in compatible_regions:
            # Check region capacity
            density = rule.density_per_m2
            max_items = int(region.size[0] * region.size[1] * density)
            current_count = len(self._region_usage.get(region.id, []))

            if current_count >= max_items:
                continue

            # Generate candidates
            candidates = self.generate_placement_candidates(
                asset, region, rule
            )
            total_candidates += len(candidates)

            # Evaluate candidates
            for candidate in candidates:
                # Check bounds
                if not self._check_region_bounds(asset, candidate.position, region):
                    continue

                # Check collisions
                collision = self.check_collision(
                    asset, candidate.position, candidate.rotation
                )

                # Score candidate
                score = self.score_candidate(candidate, asset, region, collision)

                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_region = region
                    best_collision = collision

        # Return result
        if best_candidate and best_score > 0:
            # Register the placement
            self._placed_objects[asset.asset_id] = (asset, best_candidate)

            if best_region.id not in self._region_usage:
                self._region_usage[best_region.id] = []
            self._region_usage[best_region.id].append(BoundingBox(
                center=best_candidate.position,
                size=asset.bounding_box.size,
                rotation=best_candidate.rotation,
            ))

            return PlacementResult(
                asset=asset,
                status=PlacementStatus.SUCCESS,
                final_position=best_candidate.position,
                final_rotation=best_candidate.rotation,
                region_id=best_region.id,
                collision_check=best_collision,
                candidates_evaluated=total_candidates,
                reasoning=f"Placed in {best_region.name} with score {best_score:.2f}",
            )

        # Determine failure reason
        if best_collision and best_collision.has_collision:
            status = PlacementStatus.COLLISION
            reasoning = "All candidates had collisions"
        elif total_candidates == 0:
            status = PlacementStatus.REGION_FULL
            reasoning = "All compatible regions are at capacity"
        else:
            status = PlacementStatus.OUT_OF_BOUNDS
            reasoning = "No valid placement found within bounds"

        return PlacementResult(
            asset=asset,
            status=status,
            candidates_evaluated=total_candidates,
            reasoning=reasoning,
        )

    def plan_placements(
        self,
        assets: List[AssetInstance],
        regions: List[DetectedRegion],
        scene_id: str,
        scene_archetype: SceneArchetype,
    ) -> PlacementPlan:
        """Create a complete placement plan for multiple assets.

        Args:
            assets: List of assets to place
            regions: Available placement regions
            scene_id: Scene identifier
            scene_archetype: Scene type

        Returns:
            PlacementPlan with all placement results
        """
        self.reset()

        placements: List[PlacementResult] = []
        collisions_avoided = 0

        # Sort assets by priority (required first, then by size)
        sorted_assets = sorted(
            assets,
            key=lambda a: (
                -a.bounding_box.volume(),  # Larger first
                a.category.value,
            ),
        )

        for asset in sorted_assets:
            result = self.place_asset(asset, regions, scene_archetype)
            placements.append(result)

            if result.status == PlacementStatus.SUCCESS:
                if result.collision_check and result.candidates_evaluated > 1:
                    collisions_avoided += result.candidates_evaluated - 1

        # Generate AI reasoning for the plan
        ai_reasoning = self._generate_plan_reasoning(placements, scene_archetype)

        return PlacementPlan(
            scene_id=scene_id,
            placements=placements,
            regions_used=dict(self._region_usage),
            total_assets_placed=sum(
                1 for p in placements if p.status == PlacementStatus.SUCCESS
            ),
            total_collisions_avoided=collisions_avoided,
            ai_reasoning=ai_reasoning,
        )

    def _generate_plan_reasoning(
        self,
        placements: List[PlacementResult],
        scene_archetype: SceneArchetype,
    ) -> str:
        """Generate AI reasoning for placement decisions.

        Args:
            placements: The placement results
            scene_archetype: Scene type

        Returns:
            Natural language explanation of placement decisions
        """
        client = self._get_client()
        if not client:
            return "AI reasoning not available (no API key)"

        # Summarize placements
        success_count = sum(1 for p in placements if p.status == PlacementStatus.SUCCESS)
        failed_count = len(placements) - success_count

        placement_summary = []
        for p in placements[:10]:  # Limit for prompt size
            placement_summary.append({
                "asset": p.asset.asset_name,
                "category": p.asset.category.value,
                "status": p.status.value,
                "region": p.region_id,
                "reasoning": p.reasoning,
            })

        prompt = f"""Summarize this placement plan for a {scene_archetype.value} scene in 2-3 sentences.

Placements: {success_count} successful, {failed_count} failed

Details:
{json.dumps(placement_summary, indent=2)}

Focus on:
- Overall placement strategy
- Any interesting spatial decisions
- Suggestions for failed placements
"""

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]

            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            )

            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    response_text += chunk.text

            return response_text.strip()

        except Exception as e:
            return f"AI reasoning failed: {e}"

    def get_placement_statistics(self) -> Dict[str, Any]:
        """Get statistics about current placements.

        Returns:
            Dictionary with placement statistics
        """
        total_volume = sum(
            asset.bounding_box.volume()
            for asset, _ in self._placed_objects.values()
        )

        region_stats = {}
        for region_id, boxes in self._region_usage.items():
            region_stats[region_id] = {
                "count": len(boxes),
                "total_volume": sum(b.volume() for b in boxes),
            }

        return {
            "total_objects_placed": len(self._placed_objects),
            "total_volume_m3": total_volume,
            "regions_used": len(self._region_usage),
            "region_details": region_stats,
        }

    def export_placements_to_usd(self) -> str:
        """Export placements as USD layer content.

        Returns:
            USDA formatted string
        """
        lines = [
            '#usda 1.0',
            '(',
            '    defaultPrim = "PlacedObjects"',
            ')',
            '',
            'def Xform "PlacedObjects"',
            '{',
        ]

        for obj_id, (asset, placement) in self._placed_objects.items():
            safe_id = obj_id.replace("-", "_").replace(" ", "_")
            pos = placement.position
            rot = placement.rotation

            lines.extend([
                f'    def Xform "{safe_id}"',
                '    {',
                f'        double3 xformOp:translate = ({pos[0]}, {pos[1]}, {pos[2]})',
                f'        float3 xformOp:rotateXYZ = ({rot[0]}, {rot[1]}, {rot[2]})',
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]',
                '',
                f'        string asset_id = "{asset.asset_id}"',
                f'        string asset_name = "{asset.asset_name}"',
                f'        string category = "{asset.category.value}"',
                f'        string region_id = "{placement.region_id}"',
                f'        int stack_level = {placement.stack_level}',
                '    }',
            ])

        lines.append('}')
        return '\n'.join(lines)


def create_placement_engine(
    api_key: Optional[str] = None,
    enable_stacking: bool = True,
) -> SmartPlacementEngine:
    """Factory function to create a placement engine.

    Args:
        api_key: Optional Gemini API key
        enable_stacking: Whether to allow stacking

    Returns:
        Configured SmartPlacementEngine instance
    """
    return SmartPlacementEngine(
        api_key=api_key,
        enable_stacking=enable_stacking,
    )


if __name__ == "__main__":
    # Test the placement engine
    engine = create_placement_engine()

    # Create test region
    test_region = DetectedRegion(
        id="counter_01",
        name="Kitchen Counter",
        region_type=RegionType.COUNTER,
        position=[0.0, 0.0, 0.9],
        size=[2.0, 0.6, 0.05],
        surface_type="horizontal",
        clearance_above_m=0.5,
        accessibility_score=0.9,
        suitable_for=["dishes", "utensils"],
    )

    # Create test assets
    test_assets = [
        AssetInstance(
            asset_id=f"plate_{i}",
            asset_name=f"Dinner Plate {i}",
            category=AssetCategory.DISHES,
            bounding_box=BoundingBox(
                center=[0, 0, 0],
                size=[0.25, 0.25, 0.02],
            ),
            mass_kg=0.4,
        )
        for i in range(5)
    ]

    # Plan placements
    plan = engine.plan_placements(
        assets=test_assets,
        regions=[test_region],
        scene_id="test_kitchen",
        scene_archetype=SceneArchetype.KITCHEN,
    )

    print(f"Placed {plan.total_assets_placed}/{len(test_assets)} assets")
    for result in plan.placements:
        print(f"  - {result.asset.asset_name}: {result.status.value}")
        if result.final_position:
            print(f"    Position: {result.final_position}")

#!/usr/bin/env python3
"""Smart Placement Engine - Main Entry Point.

This script orchestrates the complete smart placement workflow:
1. Load scene manifest and detect regions using AI
2. Load asset specifications to place
3. Use compatibility matrix to match assets to regions
4. Run smart placement with collision awareness
5. Validate physics plausibility
6. Output placement USD layer and metadata

Usage:
    python run_smart_placement.py --scene-id <scene_id> --bucket <bucket>

Environment Variables:
    GEMINI_API_KEY: Required for AI-powered features
    BUCKET: GCS bucket for scene data
    SCENE_ID: Scene identifier
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports
from compatibility_matrix import (
    AssetCategory,
    CompatibilityMatrix,
    SceneArchetype,
    get_compatibility_matrix,
)
from intelligent_region_detector import (
    IntelligentRegionDetector,
    DetectedRegion,
    create_region_detector,
)
from placement_engine import (
    AssetInstance,
    BoundingBox,
    PlacementPlan,
    SmartPlacementEngine,
    create_placement_engine,
)
from physics_validator import (
    PhysicsValidator,
    ValidationResult,
    create_physics_validator,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class PlacementConfig:
    """Configuration for the smart placement job."""

    def __init__(
        self,
        scene_id: str,
        bucket: Optional[str] = None,
        manifest_path: Optional[str] = None,
        assets_manifest_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        enable_ai: bool = True,
        enable_stacking: bool = True,
        validate_physics: bool = True,
        strict_physics: bool = False,
        max_assets: Optional[int] = None,
        dry_run: bool = False,
    ):
        self.scene_id = scene_id
        self.bucket = bucket
        self.manifest_path = manifest_path
        self.assets_manifest_path = assets_manifest_path
        self.output_dir = output_dir or f"/tmp/placement/{scene_id}"
        self.enable_ai = enable_ai
        self.enable_stacking = enable_stacking
        self.validate_physics = validate_physics
        self.strict_physics = strict_physics
        self.max_assets = max_assets
        self.dry_run = dry_run


# =============================================================================
# ASSET LOADING
# =============================================================================

def load_assets_from_manifest(
    manifest: Dict[str, Any],
) -> List[AssetInstance]:
    """Load asset instances from a variation assets manifest.

    Args:
        manifest: The variation_assets/manifest.json content

    Returns:
        List of AssetInstance objects
    """
    assets: List[AssetInstance] = []

    for item in manifest.get("assets", []):
        # Parse category
        category_str = item.get("category", "misc_objects").lower()
        try:
            category = AssetCategory(category_str)
        except ValueError:
            category = AssetCategory.MISC_OBJECTS

        # Parse dimensions
        dims = item.get("dimensions", {})
        size = [
            dims.get("width", 0.1),
            dims.get("depth", 0.1),
            dims.get("height", 0.1),
        ]

        # Create asset instance
        asset = AssetInstance(
            asset_id=item.get("id", f"asset_{len(assets)}"),
            asset_name=item.get("name", "Unknown Asset"),
            category=category,
            bounding_box=BoundingBox(
                center=[0, 0, 0],
                size=size,
            ),
            mass_kg=item.get("mass_kg", 0.5),
            semantic_class=item.get("semantic_class", ""),
            stackable=item.get("stackable", True),
            graspable=item.get("graspable", True),
            fragile=item.get("fragile", False),
            metadata=item.get("metadata", {}),
        )
        assets.append(asset)

    return assets


def create_sample_assets(
    scene_archetype: SceneArchetype,
    count: int = 10,
) -> List[AssetInstance]:
    """Create sample assets for testing based on scene archetype.

    Args:
        scene_archetype: The scene type
        count: Number of assets to create

    Returns:
        List of sample AssetInstance objects
    """
    # Asset templates by archetype
    templates = {
        SceneArchetype.KITCHEN: [
            ("dinner_plate", AssetCategory.DISHES, [0.25, 0.25, 0.02], 0.4),
            ("salad_plate", AssetCategory.DISHES, [0.20, 0.20, 0.02], 0.3),
            ("bowl", AssetCategory.DISHES, [0.15, 0.15, 0.08], 0.25),
            ("mug", AssetCategory.DISHES, [0.08, 0.08, 0.10], 0.3),
            ("fork", AssetCategory.UTENSILS, [0.02, 0.18, 0.01], 0.05),
            ("knife", AssetCategory.UTENSILS, [0.02, 0.22, 0.01], 0.08),
            ("spoon", AssetCategory.UTENSILS, [0.03, 0.16, 0.01], 0.04),
            ("pan", AssetCategory.COOKWARE, [0.28, 0.28, 0.05], 0.8),
            ("pot", AssetCategory.COOKWARE, [0.22, 0.22, 0.15], 1.2),
        ],
        SceneArchetype.GROCERY: [
            ("cereal_box", AssetCategory.BOXES, [0.08, 0.20, 0.28], 0.4),
            ("soup_can", AssetCategory.CANS, [0.07, 0.07, 0.11], 0.4),
            ("soda_bottle", AssetCategory.BOTTLES, [0.07, 0.07, 0.30], 0.6),
            ("milk_carton", AssetCategory.GROCERIES, [0.08, 0.08, 0.24], 1.0),
            ("pasta_box", AssetCategory.BOXES, [0.06, 0.15, 0.22], 0.5),
        ],
        SceneArchetype.WAREHOUSE: [
            ("small_box", AssetCategory.SHIPPING_BOXES, [0.3, 0.3, 0.2], 2.0),
            ("medium_box", AssetCategory.SHIPPING_BOXES, [0.4, 0.4, 0.3], 5.0),
            ("tote_bin", AssetCategory.TOTES, [0.6, 0.4, 0.3], 1.5),
            ("carton", AssetCategory.CARTONS, [0.35, 0.25, 0.25], 3.0),
        ],
        SceneArchetype.LAB: [
            ("beaker", AssetCategory.LAB_EQUIPMENT, [0.08, 0.08, 0.12], 0.15),
            ("flask", AssetCategory.LAB_EQUIPMENT, [0.10, 0.10, 0.15], 0.2),
            ("sample_tube", AssetCategory.SAMPLE_CONTAINERS, [0.02, 0.02, 0.10], 0.02),
            ("pipette", AssetCategory.LAB_TOOLS, [0.02, 0.02, 0.25], 0.03),
        ],
        SceneArchetype.HOME_LAUNDRY: [
            ("tshirt", AssetCategory.CLOTHING, [0.40, 0.30, 0.02], 0.2),
            ("pants", AssetCategory.CLOTHING, [0.35, 0.50, 0.03], 0.4),
            ("towel", AssetCategory.LINENS, [0.60, 0.40, 0.02], 0.3),
            ("sock", AssetCategory.CLOTHING, [0.10, 0.20, 0.02], 0.05),
        ],
    }

    # Get templates for archetype (fallback to kitchen)
    archetype_templates = templates.get(
        scene_archetype,
        templates[SceneArchetype.KITCHEN]
    )

    assets: List[AssetInstance] = []
    for i in range(count):
        template = archetype_templates[i % len(archetype_templates)]
        name, category, size, mass = template

        asset = AssetInstance(
            asset_id=f"{name}_{i:03d}",
            asset_name=f"{name.replace('_', ' ').title()} {i+1}",
            category=category,
            bounding_box=BoundingBox(
                center=[0, 0, 0],
                size=size,
            ),
            mass_kg=mass,
            stackable=category in [
                AssetCategory.DISHES,
                AssetCategory.BOXES,
                AssetCategory.SHIPPING_BOXES,
            ],
        )
        assets.append(asset)

    return assets


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_placement_usd(
    plan: PlacementPlan,
    output_path: str,
) -> str:
    """Generate USD layer for placements.

    Args:
        plan: The placement plan
        output_path: Path to write the USD file

    Returns:
        Path to generated file
    """
    lines = [
        '#usda 1.0',
        '(',
        '    doc = "Smart Placement Engine - Generated Placements"',
        f'    metersPerUnit = 1',
        '    upAxis = "Z"',
        ')',
        '',
        'def Xform "Placements" (',
        '    kind = "group"',
        ')',
        '{',
    ]

    for result in plan.placements:
        if not result.final_position:
            continue

        safe_id = result.asset.asset_id.replace("-", "_").replace(" ", "_")
        pos = result.final_position
        rot = result.final_rotation or [0, 0, 0]
        size = result.asset.bounding_box.size

        lines.extend([
            f'    def Xform "{safe_id}" (',
            '        kind = "component"',
            '    )',
            '    {',
            f'        double3 xformOp:translate = ({pos[0]}, {pos[1]}, {pos[2]})',
            f'        float3 xformOp:rotateXYZ = ({rot[0]}, {rot[1]}, {rot[2]})',
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]',
            '',
            '        # Placement metadata',
            f'        string smartPlacement:assetId = "{result.asset.asset_id}"',
            f'        string smartPlacement:assetName = "{result.asset.asset_name}"',
            f'        string smartPlacement:category = "{result.asset.category.value}"',
            f'        string smartPlacement:regionId = "{result.region_id}"',
            f'        string smartPlacement:status = "{result.status.value}"',
            '',
            '        # Bounding box for collision reference',
            f'        float3 smartPlacement:boundingBox = ({size[0]}, {size[1]}, {size[2]})',
            f'        float smartPlacement:massKg = {result.asset.mass_kg}',
            '    }',
        ])

    lines.append('}')

    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines))

    return str(output_path)


def generate_placement_report(
    plan: PlacementPlan,
    validation: Optional[ValidationResult],
    output_path: str,
) -> str:
    """Generate JSON report of placement results.

    Args:
        plan: The placement plan
        validation: Physics validation result
        output_path: Path to write the report

    Returns:
        Path to generated file
    """
    report = {
        "scene_id": plan.scene_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_assets": len(plan.placements),
            "successfully_placed": plan.total_assets_placed,
            "collisions_avoided": plan.total_collisions_avoided,
            "regions_used": len(plan.regions_used),
        },
        "ai_reasoning": plan.ai_reasoning,
        "placements": [],
    }

    for result in plan.placements:
        placement_data = {
            "asset_id": result.asset.asset_id,
            "asset_name": result.asset.asset_name,
            "category": result.asset.category.value,
            "status": result.status.value,
            "position": result.final_position,
            "rotation": result.final_rotation,
            "region_id": result.region_id,
            "candidates_evaluated": result.candidates_evaluated,
            "reasoning": result.reasoning,
        }
        report["placements"].append(placement_data)

    if validation:
        report["physics_validation"] = {
            "status": validation.status.value,
            "score": validation.score,
            "warnings": validation.warnings,
            "errors": validation.errors,
            "ai_analysis": validation.ai_analysis,
        }

    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    return str(output_path)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_smart_placement(config: PlacementConfig) -> Dict[str, Any]:
    """Run the complete smart placement workflow.

    Args:
        config: Placement configuration

    Returns:
        Dictionary with results and output paths
    """
    print(f"[SMART_PLACEMENT] Starting for scene: {config.scene_id}")
    start_time = time.time()

    results: Dict[str, Any] = {
        "scene_id": config.scene_id,
        "success": False,
        "outputs": {},
        "statistics": {},
    }

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and config.enable_ai:
        print("[SMART_PLACEMENT] Warning: GEMINI_API_KEY not set, AI features disabled")
        config.enable_ai = False

    # Load scene manifest
    print("[SMART_PLACEMENT] Loading scene manifest...")
    if config.manifest_path:
        manifest_path = Path(config.manifest_path)
    else:
        manifest_path = Path(f"/mnt/gcs/{config.scene_id}/manifest.json")

    if manifest_path.exists():
        scene_manifest = json.loads(manifest_path.read_text())
    else:
        # Create sample manifest for testing
        print("[SMART_PLACEMENT] No manifest found, using sample data")
        scene_manifest = {
            "scene_id": config.scene_id,
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
            ]
        }

    # Detect scene archetype
    env_type = scene_manifest.get("environment", {}).get("type", "kitchen")
    try:
        scene_archetype = SceneArchetype(env_type.lower())
    except ValueError:
        scene_archetype = SceneArchetype.KITCHEN

    print(f"[SMART_PLACEMENT] Scene archetype: {scene_archetype.value}")

    # Step 1: Detect placement regions
    print("[SMART_PLACEMENT] Detecting placement regions...")
    if config.enable_ai:
        detector = create_region_detector(api_key=api_key)
        detection_result = detector.detect_regions_from_manifest(
            scene_manifest,
            scene_id=config.scene_id,
            scene_archetype=scene_archetype,
        )
        regions = detection_result.detected_regions
        results["scene_understanding"] = detection_result.scene_understanding
    else:
        # Create fallback regions based on manifest
        regions = _create_fallback_regions(scene_manifest, scene_archetype)

    print(f"[SMART_PLACEMENT] Detected {len(regions)} placement regions")
    for region in regions[:5]:
        print(f"  - {region.name}: {region.region_type.value}")

    # Step 2: Load assets to place
    print("[SMART_PLACEMENT] Loading assets...")
    if config.assets_manifest_path:
        assets_path = Path(config.assets_manifest_path)
        if assets_path.exists():
            assets_manifest = json.loads(assets_path.read_text())
            assets = load_assets_from_manifest(assets_manifest)
        else:
            assets = create_sample_assets(scene_archetype, count=15)
    else:
        # Create sample assets for testing
        assets = create_sample_assets(scene_archetype, count=15)

    if config.max_assets:
        assets = assets[:config.max_assets]

    print(f"[SMART_PLACEMENT] Loaded {len(assets)} assets to place")

    # Step 3: Run smart placement
    print("[SMART_PLACEMENT] Running placement engine...")
    engine = create_placement_engine(
        api_key=api_key if config.enable_ai else None,
        enable_stacking=config.enable_stacking,
    )

    plan = engine.plan_placements(
        assets=assets,
        regions=regions,
        scene_id=config.scene_id,
        scene_archetype=scene_archetype,
    )

    print(f"[SMART_PLACEMENT] Placed {plan.total_assets_placed}/{len(assets)} assets")
    print(f"[SMART_PLACEMENT] Avoided {plan.total_collisions_avoided} collisions")

    results["statistics"]["total_assets"] = len(assets)
    results["statistics"]["placed"] = plan.total_assets_placed
    results["statistics"]["collisions_avoided"] = plan.total_collisions_avoided

    # Step 4: Validate physics
    validation: Optional[ValidationResult] = None
    if config.validate_physics:
        print("[SMART_PLACEMENT] Validating physics...")
        validator = create_physics_validator(
            api_key=api_key if config.enable_ai else None,
            strict_mode=config.strict_physics,
        )

        validation = validator.validate_plan(plan, regions)
        print(f"[SMART_PLACEMENT] Physics validation: {validation.status.value}")
        print(f"[SMART_PLACEMENT] Physics score: {validation.score:.2f}")

        if validation.warnings:
            print(f"[SMART_PLACEMENT] Warnings: {len(validation.warnings)}")
        if validation.errors:
            print(f"[SMART_PLACEMENT] Errors: {len(validation.errors)}")

        results["physics"] = {
            "status": validation.status.value,
            "score": validation.score,
            "warnings": len(validation.warnings),
            "errors": len(validation.errors),
        }

    # Step 5: Generate outputs
    if not config.dry_run:
        print("[SMART_PLACEMENT] Generating outputs...")
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # USD layer
        usd_path = generate_placement_usd(
            plan,
            output_dir / "placements.usda",
        )
        results["outputs"]["usd"] = usd_path
        print(f"[SMART_PLACEMENT] Generated USD: {usd_path}")

        # JSON report
        report_path = generate_placement_report(
            plan,
            validation,
            output_dir / "placement_report.json",
        )
        results["outputs"]["report"] = report_path
        print(f"[SMART_PLACEMENT] Generated report: {report_path}")

    # Done
    elapsed = time.time() - start_time
    results["success"] = plan.total_assets_placed > 0
    results["elapsed_seconds"] = elapsed
    print(f"[SMART_PLACEMENT] Completed in {elapsed:.2f}s")

    return results


def _create_fallback_regions(
    manifest: Dict[str, Any],
    archetype: SceneArchetype,
) -> List[DetectedRegion]:
    """Create fallback regions when AI is not available.

    Args:
        manifest: Scene manifest
        archetype: Scene archetype

    Returns:
        List of basic DetectedRegion objects
    """
    from compatibility_matrix import RegionType

    regions: List[DetectedRegion] = []

    for obj in manifest.get("objects", []):
        obj_id = obj.get("id", "")
        obj_name = obj.get("name", "")
        category = obj.get("category", "").lower()
        dims = obj.get("dimensions", {})

        # Determine region type based on category/name
        region_type = RegionType.SHELF  # Default
        suitable_for: List[str] = []

        if "counter" in obj_name.lower() or category == "counter":
            region_type = RegionType.COUNTER
            suitable_for = ["dishes", "utensils", "food_items"]
        elif "dishwasher" in obj_name.lower():
            region_type = RegionType.DISHWASHER
            suitable_for = ["dishes", "utensils"]
        elif "cabinet" in obj_name.lower():
            region_type = RegionType.CABINET
            suitable_for = ["dishes", "containers"]
        elif "shelf" in obj_name.lower() or "rack" in obj_name.lower():
            region_type = RegionType.SHELF
            suitable_for = ["groceries", "boxes"]
        elif "table" in obj_name.lower():
            region_type = RegionType.TABLE
            suitable_for = ["dishes", "food_items"]

        # Check articulation state
        articulation = obj.get("articulation", {})
        articulation_state = None
        for joint_name, joint_data in articulation.items():
            if joint_data.get("state") == "open":
                from intelligent_region_detector import ArticulationState
                articulation_state = ArticulationState.OPEN
                break

        regions.append(DetectedRegion(
            id=f"{obj_id}_surface",
            name=f"{obj_name} Surface",
            region_type=region_type,
            position=[0.0, 0.0, dims.get("height", 0.9)],
            size=[
                dims.get("width", 1.0),
                dims.get("depth", 0.5),
                0.05,
            ],
            parent_object_id=obj_id,
            parent_object_name=obj_name,
            surface_type="horizontal",
            articulation_state=articulation_state,
            suitable_for=suitable_for,
            accessibility_score=0.8,
        ))

    return regions


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Smart Placement Engine - Intelligent asset placement with collision awareness"
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=os.environ.get("SCENE_ID", "test_scene"),
        help="Scene identifier",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("BUCKET", ""),
        help="GCS bucket for scene data",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=os.environ.get("MANIFEST_PATH", ""),
        help="Path to scene manifest JSON",
    )
    parser.add_argument(
        "--assets-manifest-path",
        type=str,
        default=os.environ.get("ASSETS_MANIFEST_PATH", ""),
        help="Path to assets manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", ""),
        help="Output directory for results",
    )
    parser.add_argument(
        "--disable-ai",
        action="store_true",
        help="Disable AI-powered features",
    )
    parser.add_argument(
        "--disable-stacking",
        action="store_true",
        help="Disable object stacking",
    )
    parser.add_argument(
        "--skip-physics",
        action="store_true",
        help="Skip physics validation",
    )
    parser.add_argument(
        "--strict-physics",
        action="store_true",
        help="Treat physics warnings as errors",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Maximum number of assets to place",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without generating output files",
    )

    args = parser.parse_args()

    config = PlacementConfig(
        scene_id=args.scene_id,
        bucket=args.bucket or None,
        manifest_path=args.manifest_path or None,
        assets_manifest_path=args.assets_manifest_path or None,
        output_dir=args.output_dir or None,
        enable_ai=not args.disable_ai,
        enable_stacking=not args.disable_stacking,
        validate_physics=not args.skip_physics,
        strict_physics=args.strict_physics,
        max_assets=args.max_assets,
        dry_run=args.dry_run,
    )

    try:
        results = run_smart_placement(config)
        print("\n" + "=" * 60)
        print("RESULTS:")
        print(json.dumps(results, indent=2))

        if results["success"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"[SMART_PLACEMENT] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="SMART-PLACEMENT", validate_gcs=True)
    main()

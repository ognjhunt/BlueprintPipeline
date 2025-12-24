# Smart Placement Engine

Intelligent asset-to-placement-region matching with collision awareness for SimReady scene population.

## Overview

The Smart Placement Engine solves three critical problems in scene randomization:

1. **Intelligent Placement Regions** - AI-powered detection and understanding of where objects can be placed
2. **Smart Placement Logic** - Collision-aware placement with physics validation
3. **Scene-to-Asset Mapping** - Compatibility matrix defining what assets go where

## Components

### 1. Compatibility Matrix (`compatibility_matrix.py`)

Defines the complete mapping between:
- Scene archetypes (kitchen, warehouse, grocery, lab, etc.)
- Placement regions (counters, shelves, dishwashers, etc.)
- Asset categories (dishes, utensils, groceries, etc.)
- Contextual constraints (articulation states, semantic tags)

```python
from compatibility_matrix import (
    CompatibilityMatrix,
    PlacementContext,
    SceneArchetype,
    RegionType,
    ArticulationState,
)

matrix = CompatibilityMatrix()

# What can go in an open dishwasher?
context = PlacementContext(
    scene_archetype=SceneArchetype.KITCHEN,
    region_type=RegionType.DISHWASHER,
    region_id="dishwasher_01",
    articulation_state=ArticulationState.OPEN,
)

compatible = matrix.get_compatible_assets(context)
# Returns: [(AssetCategory.DISHES, rule), (AssetCategory.UTENSILS, rule), ...]
```

### 2. Intelligent Region Detector (`intelligent_region_detector.py`)

Uses Gemini 3.0 Pro Preview to analyze scenes and detect placement regions:

```python
from intelligent_region_detector import create_region_detector

detector = create_region_detector()

result = detector.detect_regions_from_manifest(
    scene_manifest=manifest_json,
    scene_id="kitchen_001",
    scene_archetype=SceneArchetype.KITCHEN,
    target_assets=[AssetCategory.DISHES, AssetCategory.UTENSILS],
)

for region in result.detected_regions:
    print(f"Region: {region.name}")
    print(f"  Type: {region.region_type}")
    print(f"  Suitable for: {region.suitable_for}")
    print(f"  Accessibility: {region.accessibility_score}")
```

### 3. Smart Placement Engine (`placement_engine.py`)

Collision-aware placement with optimal distribution:

```python
from placement_engine import create_placement_engine, AssetInstance, BoundingBox

engine = create_placement_engine(enable_stacking=True)

assets = [
    AssetInstance(
        asset_id="plate_001",
        asset_name="Dinner Plate",
        category=AssetCategory.DISHES,
        bounding_box=BoundingBox(center=[0,0,0], size=[0.25, 0.25, 0.02]),
        mass_kg=0.4,
    ),
    # ... more assets
]

plan = engine.plan_placements(
    assets=assets,
    regions=detected_regions,
    scene_id="kitchen_001",
    scene_archetype=SceneArchetype.KITCHEN,
)

print(f"Placed {plan.total_assets_placed}/{len(assets)} assets")
print(f"Avoided {plan.total_collisions_avoided} collisions")
```

### 4. Physics Validator (`physics_validator.py`)

Validates placement physics plausibility:

```python
from physics_validator import create_physics_validator

validator = create_physics_validator(strict_mode=False)

validation = validator.validate_plan(plan, regions)

print(f"Physics score: {validation.score:.2f}")
print(f"Warnings: {validation.warnings}")
print(f"Errors: {validation.errors}")

# Get improvement suggestions
suggestions = validator.suggest_improvements(validation)
```

## Usage

### Command Line

```bash
# Basic usage
python run_smart_placement.py --scene-id kitchen_001

# With custom paths
python run_smart_placement.py \
    --scene-id kitchen_001 \
    --manifest-path /path/to/manifest.json \
    --assets-manifest-path /path/to/assets.json \
    --output-dir /path/to/output

# Disable AI features
python run_smart_placement.py --scene-id test --disable-ai

# Strict physics validation
python run_smart_placement.py --scene-id test --strict-physics
```

### Docker

```bash
# Build
docker build -t smart-placement-engine .

# Run with GCS bucket
docker run -e GEMINI_API_KEY=$GEMINI_API_KEY \
           -e BUCKET=my-bucket \
           -e SCENE_ID=kitchen_001 \
           --privileged \
           smart-placement-engine
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Gemini API key for AI features | Yes (for AI) |
| `BUCKET` | GCS bucket for scene data | No |
| `SCENE_ID` | Scene identifier | Yes |
| `OUTPUT_DIR` | Output directory | No |

## Output Files

1. **placements.usda** - USD layer with all placement transforms and metadata
2. **placement_report.json** - Detailed JSON report with statistics and validation

## Integration with Pipeline

The Smart Placement Engine integrates with the existing pipeline:

```
scene-generation-job → 3D-RE-GEN → smart-placement-engine → replicator-job → isaac-lab
                                           ↓
                                  variation-asset-pipeline-job
```

1. Scene manifest is generated
2. Smart Placement Engine analyzes the scene
3. Detects placement regions using AI
4. Places variation assets with collision awareness
5. Validates physics plausibility
6. Outputs placement USD layer for Replicator

## Compatibility Matrix Coverage

### Scene Archetypes
- Kitchen (commercial prep, dish pit, quick-serve)
- Grocery/Retail (aisles, refrigeration, checkout)
- Warehouse (racks, staging, conveyors)
- Lab (benches, gloveboxes, fume hoods)
- Office (desks, filing, meeting rooms)
- Utility Rooms (panels, valves, HVAC)
- Home Laundry (washer/dryer, folding, closets)

### Asset Categories
- Dishes, Utensils, Cookware
- Groceries, Bottles, Cans, Boxes
- Pallets, Totes, Cartons
- Lab Equipment, Sample Containers
- Clothing, Linens
- Tools, Maintenance Supplies

### Region Types
- Horizontal surfaces (counters, tables, shelves)
- Storage (drawers, cabinets, refrigerators)
- Appliances (dishwashers, ovens, washers)
- Industrial (racks, pallets, conveyors)
- Specialized (sinks, gloveboxes, panels)

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Type checking
mypy .
```

## Architecture

```
smart-placement-engine-job/
├── __init__.py                    # Package exports
├── compatibility_matrix.py        # Scene-to-asset mapping rules
├── intelligent_region_detector.py # AI-powered region detection
├── placement_engine.py            # Collision-aware placement
├── physics_validator.py           # Physics plausibility checks
├── run_smart_placement.py         # Main orchestrator
├── Dockerfile
├── requirements.txt
└── README.md
```

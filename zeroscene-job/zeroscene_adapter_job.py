#!/usr/bin/env python3
"""
ZeroScene Adapter Job - Converts ZeroScene outputs to BlueprintPipeline format.

This job is the integration layer between ZeroScene reconstruction and the
BlueprintPipeline downstream jobs (simready, usd-assembly, replicator, etc.).

Pipeline Position:
    image upload → ZeroScene reconstruction → [THIS JOB] → simready → usd-assembly → replicator

Inputs:
    - ZeroScene output directory with:
        - objects/obj_*/mesh.glb + pose.json + bounds.json
        - background/mesh.glb
        - camera/ (optional)
        - depth/ (optional)

Outputs:
    - assets/scene_manifest.json (canonical manifest)
    - layout/scene_layout_scaled.json (layout for usd-assembly)
    - assets/obj_*/ (copied/structured assets)
    - seg/inventory.json (semantic inventory for downstream)

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ZEROSCENE_PREFIX: Path to ZeroScene outputs (default: scenes/{scene_id}/zeroscene)
    ASSETS_PREFIX: Path to output assets (default: scenes/{scene_id}/assets)
    LAYOUT_PREFIX: Path to output layout (default: scenes/{scene_id}/layout)
    ENVIRONMENT_TYPE: Environment type hint (kitchen, office, etc.)
    SCALE_FACTOR: Optional scale factor for metric calibration (default: 1.0)
    SKIP_INVENTORY: If "true", skip semantic inventory generation
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.zeroscene_adapter import (
    ZeroSceneAdapter,
    manifest_from_zeroscene,
    layout_from_zeroscene,
    ZeroSceneOutput,
)

# Optional: Gemini for semantic inventory
try:
    from google import genai
    from google.genai import types
    HAVE_GEMINI = True
except ImportError:
    HAVE_GEMINI = False

GCS_ROOT = Path("/mnt/gcs")


# =============================================================================
# Semantic Inventory Generation
# =============================================================================

def generate_semantic_inventory(
    zeroscene_output: ZeroSceneOutput,
    environment_type: str,
    source_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate semantic inventory from ZeroScene objects.

    Even though ZeroScene provides segmentation, downstream jobs need:
    - Object categories and descriptions
    - sim_role classification
    - articulation_hint for interactive objects
    - manipulable vs static vs background labeling

    This can be enhanced with Gemini vision if available.
    """
    objects = []

    for obj in zeroscene_output.objects:
        inventory_obj = {
            "id": obj.id,
            "category": obj.category or "object",
            "short_description": obj.description or f"Object {obj.id}",
            "sim_role": infer_sim_role(obj, environment_type),
            "articulation_hint": infer_articulation_hint(obj),
            "approx_location": None,  # Will be filled from layout
            "bounds": obj.bounds,
        }

        # Add material-based hints
        if obj.material:
            inventory_obj["material_type"] = obj.material.material_type

        objects.append(inventory_obj)

    # Add background
    if zeroscene_output.background:
        objects.append({
            "id": "scene_background",
            "category": "scene_background",
            "short_description": "Room background mesh",
            "sim_role": "background",
            "articulation_hint": None,
        })

    inventory = {
        "scene_id": zeroscene_output.scene_id,
        "environment_type": environment_type,
        "total_objects": len(objects),
        "objects": objects,
        "metadata": {
            "source": "zeroscene",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    }

    return inventory


def infer_sim_role(obj, environment_type: str) -> str:
    """Infer sim_role from object properties and environment context.

    sim_role values:
    - static: Non-moving scene elements
    - interactive: Objects robot can interact with (doors, drawers)
    - manipulable_object: Objects robot can pick up
    - articulated_furniture: Furniture with joints (cabinets, drawers)
    - articulated_appliance: Appliances with joints (dishwasher, oven)
    - clutter: Small objects for variation
    - background: Scene shell/background mesh
    """
    category = (obj.category or "").lower()
    bounds_size = obj.bounds.get("size", [1, 1, 1])
    volume = bounds_size[0] * bounds_size[1] * bounds_size[2]

    # Large furniture is typically static or articulated
    if volume > 0.5:  # > 0.5 m^3
        if any(k in category for k in ["cabinet", "drawer", "door", "closet"]):
            return "articulated_furniture"
        if any(k in category for k in ["oven", "dishwasher", "fridge", "refrigerator", "washer", "dryer"]):
            return "articulated_appliance"
        if any(k in category for k in ["table", "counter", "shelf", "desk", "bed", "sofa", "couch"]):
            return "static"

    # Small objects are typically manipulable
    if volume < 0.01:  # < 10 liters
        if any(k in category for k in ["dish", "plate", "bowl", "cup", "mug", "glass", "utensil"]):
            return "manipulable_object"
        if any(k in category for k in ["bottle", "can", "box", "food", "grocery"]):
            return "manipulable_object"
        return "clutter"

    # Medium objects - context dependent
    if any(k in category for k in ["chair", "stool"]):
        return "interactive"

    return "static"


def infer_articulation_hint(obj) -> Optional[str]:
    """Infer articulation type from object category."""
    category = (obj.category or "").lower()

    if any(k in category for k in ["drawer"]):
        return "prismatic"
    if any(k in category for k in ["door", "cabinet", "oven", "microwave", "dishwasher", "fridge"]):
        return "revolute"
    if any(k in category for k in ["knob", "dial"]):
        return "revolute"
    if any(k in category for k in ["lid", "hatch"]):
        return "revolute"

    return None


def enrich_inventory_with_gemini(
    inventory: Dict[str, Any],
    source_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Use Gemini to enrich inventory with better categories and descriptions.

    This is optional but improves downstream policy targeting.
    """
    if not HAVE_GEMINI or not os.getenv("GEMINI_API_KEY"):
        print("[ZEROSCENE-JOB] Gemini not available, skipping enrichment")
        return inventory

    # Only enrich if we have a source image
    if not source_image_path or not Path(source_image_path).is_file():
        print("[ZEROSCENE-JOB] No source image for enrichment")
        return inventory

    # This is a placeholder - in production, you'd call Gemini with the
    # source image and ask it to categorize and describe each object
    # based on their approximate locations and the scene context.

    print("[ZEROSCENE-JOB] Gemini enrichment available (placeholder)")
    return inventory


# =============================================================================
# Main Job
# =============================================================================

def run_zeroscene_adapter_job(
    root: Path,
    scene_id: str,
    zeroscene_prefix: str,
    assets_prefix: str,
    layout_prefix: str,
    environment_type: str = "generic",
    scale_factor: float = 1.0,
    skip_inventory: bool = False,
) -> int:
    """Run the ZeroScene adapter job.

    Returns:
        0 on success, 1 on failure
    """
    print(f"[ZEROSCENE-JOB] Starting ZeroScene adapter for scene: {scene_id}")
    print(f"[ZEROSCENE-JOB] ZeroScene prefix: {zeroscene_prefix}")
    print(f"[ZEROSCENE-JOB] Assets prefix: {assets_prefix}")
    print(f"[ZEROSCENE-JOB] Layout prefix: {layout_prefix}")
    print(f"[ZEROSCENE-JOB] Environment type: {environment_type}")
    print(f"[ZEROSCENE-JOB] Scale factor: {scale_factor}")

    zeroscene_dir = root / zeroscene_prefix
    assets_dir = root / assets_prefix
    layout_dir = root / layout_prefix

    # Check ZeroScene outputs exist
    if not zeroscene_dir.is_dir():
        print(f"[ZEROSCENE-JOB] ERROR: ZeroScene output not found at {zeroscene_dir}")
        return 1

    # Initialize adapter
    adapter = ZeroSceneAdapter(verbose=True)

    # Load ZeroScene outputs
    try:
        zeroscene_output = adapter.load_zeroscene_output(zeroscene_dir)
        print(f"[ZEROSCENE-JOB] Loaded {len(zeroscene_output.objects)} objects from ZeroScene")
    except Exception as e:
        print(f"[ZEROSCENE-JOB] ERROR: Failed to load ZeroScene outputs: {e}")
        return 1

    if not zeroscene_output.objects:
        print("[ZEROSCENE-JOB] WARNING: No objects found in ZeroScene output")

    # Create output directories
    assets_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy assets to expected structure
    print("[ZEROSCENE-JOB] Copying assets...")
    try:
        asset_paths = adapter.copy_assets(
            zeroscene_output,
            root / assets_prefix.rsplit("/", 1)[0],  # Parent of assets dir
            assets_prefix.rsplit("/", 1)[-1],  # Just "assets" part
        )
        print(f"[ZEROSCENE-JOB] Copied {len(asset_paths)} assets")
    except Exception as e:
        print(f"[ZEROSCENE-JOB] ERROR: Failed to copy assets: {e}")
        return 1

    # 2. Generate canonical manifest
    print("[ZEROSCENE-JOB] Generating scene manifest...")
    try:
        manifest = adapter.create_manifest(
            zeroscene_output,
            scene_id=scene_id,
            environment_type=environment_type,
        )
        manifest_path = assets_dir / "scene_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[ZEROSCENE-JOB] Wrote manifest: {manifest_path}")
    except Exception as e:
        print(f"[ZEROSCENE-JOB] ERROR: Failed to generate manifest: {e}")
        return 1

    # 3. Generate layout
    print("[ZEROSCENE-JOB] Generating scene layout...")
    try:
        layout = adapter.create_layout(zeroscene_output, scale_factor)
        layout_path = layout_dir / "scene_layout_scaled.json"
        layout_path.write_text(json.dumps(layout, indent=2))
        print(f"[ZEROSCENE-JOB] Wrote layout: {layout_path}")
    except Exception as e:
        print(f"[ZEROSCENE-JOB] ERROR: Failed to generate layout: {e}")
        return 1

    # 4. Generate semantic inventory (for replicator/policy targeting)
    if not skip_inventory:
        print("[ZEROSCENE-JOB] Generating semantic inventory...")
        try:
            seg_dir = root / f"scenes/{scene_id}/seg"
            seg_dir.mkdir(parents=True, exist_ok=True)

            inventory = generate_semantic_inventory(
                zeroscene_output,
                environment_type,
                zeroscene_output.source_image_path,
            )

            # Optional Gemini enrichment
            if os.getenv("GEMINI_API_KEY"):
                inventory = enrich_inventory_with_gemini(
                    inventory,
                    zeroscene_output.source_image_path,
                )

            inventory_path = seg_dir / "inventory.json"
            inventory_path.write_text(json.dumps(inventory, indent=2))
            print(f"[ZEROSCENE-JOB] Wrote inventory: {inventory_path}")
        except Exception as e:
            print(f"[ZEROSCENE-JOB] WARNING: Failed to generate inventory: {e}")
            # Non-fatal - continue

    # 5. Write completion marker
    marker_content = {
        "status": "complete",
        "scene_id": scene_id,
        "objects_count": len(zeroscene_output.objects),
        "has_background": zeroscene_output.background is not None,
        "scale_factor": scale_factor,
        "environment_type": environment_type,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    marker_path = assets_dir / ".zeroscene_adapter_complete"
    marker_path.write_text(json.dumps(marker_content, indent=2))

    print("[ZEROSCENE-JOB] ✓ ZeroScene adapter completed successfully")
    print(f"[ZEROSCENE-JOB]   Objects: {len(zeroscene_output.objects)}")
    print(f"[ZEROSCENE-JOB]   Background: {'yes' if zeroscene_output.background else 'no'}")
    print(f"[ZEROSCENE-JOB]   Manifest: {manifest_path}")
    print(f"[ZEROSCENE-JOB]   Layout: {layout_path}")

    return 0


def main():
    """Main entry point."""
    # Get configuration from environment
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")

    if not scene_id:
        print("[ZEROSCENE-JOB] ERROR: SCENE_ID is required")
        sys.exit(1)

    # Prefixes with defaults
    zeroscene_prefix = os.getenv("ZEROSCENE_PREFIX", f"scenes/{scene_id}/zeroscene")
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    layout_prefix = os.getenv("LAYOUT_PREFIX", f"scenes/{scene_id}/layout")

    # Optional configuration
    environment_type = os.getenv("ENVIRONMENT_TYPE", "generic")
    scale_factor = float(os.getenv("SCALE_FACTOR", "1.0"))
    skip_inventory = os.getenv("SKIP_INVENTORY", "").lower() in ("true", "1", "yes")

    print(f"[ZEROSCENE-JOB] Configuration:")
    print(f"[ZEROSCENE-JOB]   Bucket: {bucket}")
    print(f"[ZEROSCENE-JOB]   Scene ID: {scene_id}")

    exit_code = run_zeroscene_adapter_job(
        root=GCS_ROOT,
        scene_id=scene_id,
        zeroscene_prefix=zeroscene_prefix,
        assets_prefix=assets_prefix,
        layout_prefix=layout_prefix,
        environment_type=environment_type,
        scale_factor=scale_factor,
        skip_inventory=skip_inventory,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

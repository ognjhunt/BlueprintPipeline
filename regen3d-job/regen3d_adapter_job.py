#!/usr/bin/env python3
"""
3D-RE-GEN Adapter Job - Converts 3D-RE-GEN outputs to BlueprintPipeline format.

This job is the integration layer between 3D-RE-GEN reconstruction and the
BlueprintPipeline downstream jobs (simready, usd-assembly, replicator, etc.).

3D-RE-GEN (arXiv:2512.17459) is a modular, compositional pipeline for
"image → sim-ready 3D reconstruction" with explicit physical constraints:
- 4-DoF ground-alignment for floor-contact objects
- Background bounding constraint for anti-penetration
- A-Q (Application-Querying) for scene-aware occlusion completion

Pipeline Position:
    image upload → 3D-RE-GEN reconstruction → [THIS JOB] → simready → usd-assembly → replicator

Inputs:
    - 3D-RE-GEN output directory with:
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
    REGEN3D_PREFIX: Path to 3D-RE-GEN outputs (default: scenes/{scene_id}/regen3d)
    ASSETS_PREFIX: Path to output assets (default: scenes/{scene_id}/assets)
    LAYOUT_PREFIX: Path to output layout (default: scenes/{scene_id}/layout)
    ENVIRONMENT_TYPE: Environment type hint (kitchen, office, etc.)
    SCALE_FACTOR: Optional scale factor for metric calibration (default: 1.0)
    SKIP_INVENTORY: If "true", skip semantic inventory generation

Reference:
    - Paper: https://arxiv.org/abs/2512.17459
    - Project: https://3dregen.jdihlmann.com/
    - GitHub: https://github.com/cgtuebingen/3D-RE-GEN
"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.regen3d_adapter import (
    Regen3DAdapter,
    manifest_from_regen3d,
    layout_from_regen3d,
    Regen3DOutput,
)
from tools.scale_authority.authority import REFERENCE_DIMENSIONS
from tools.scene_manifest.validate_manifest import validate_manifest
from tools.metrics.pipeline_metrics import get_metrics

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
    regen3d_output: Regen3DOutput,
    environment_type: str,
    source_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate semantic inventory from 3D-RE-GEN objects.

    Even though 3D-RE-GEN provides segmentation (via Grounded-SAM), downstream jobs need:
    - Object categories and descriptions
    - sim_role classification
    - articulation_hint for interactive objects
    - manipulable vs static vs background labeling
    - is_floor_contact from 3D-RE-GEN's 4-DoF optimization

    This can be enhanced with Gemini vision if available.
    """
    objects = []

    for obj in regen3d_output.objects:
        inventory_obj = {
            "id": obj.id,
            "category": obj.category or "object",
            "short_description": obj.description or f"Object {obj.id}",
            "sim_role": infer_sim_role(obj, environment_type),
            "articulation_hint": infer_articulation_hint(obj),
            "approx_location": None,  # Will be filled from layout
            "bounds": obj.bounds,
            "is_floor_contact": obj.pose.is_floor_contact,
        }

        # Add material-based hints
        if obj.material:
            inventory_obj["material_type"] = obj.material.material_type

        objects.append(inventory_obj)

    # Add background
    if regen3d_output.background:
        objects.append({
            "id": "scene_background",
            "category": "scene_background",
            "short_description": "Room background mesh (comprehensive for collision/lighting)",
            "sim_role": "background",
            "articulation_hint": None,
        })

    inventory = {
        "scene_id": regen3d_output.scene_id,
        "environment_type": environment_type,
        "total_objects": len(objects),
        "objects": objects,
        "metadata": {
            "source": "3d-re-gen",
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
    """
    Infer articulation type from object category.

    UPDATED: Now uses enhanced ArticulationDetector for better accuracy.
    Falls back to simple keyword matching if detector unavailable.
    """
    # Try to use enhanced detector
    try:
        from tools.articulation.detector import ArticulationDetector, ArticulationType

        detector = ArticulationDetector(use_llm=False, verbose=False)  # No LLM for speed
        obj_dict = {
            "id": getattr(obj, "id", "unknown"),
            "category": getattr(obj, "category", ""),
            "dimensions": getattr(obj, "dimensions", None),
        }

        result = detector.detect(obj_dict)

        if result.has_articulation and result.confidence >= 0.5:
            return result.articulation_type.value

        return None

    except ImportError:
        # Fallback to legacy keyword matching if detector unavailable
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

    This uses Gemini's vision capabilities to analyze the source image and
    provide more accurate:
    - Object categories (e.g., "microwave" instead of "appliance")
    - Detailed descriptions for downstream policy targeting
    - sim_role classifications (static, interactive, manipulable_object, etc.)
    - articulation_hint values (revolute, prismatic, etc.)

    Returns the enriched inventory with updated object metadata.
    """
    if not HAVE_GEMINI or not os.getenv("GEMINI_API_KEY"):
        print("[REGEN3D-JOB] Gemini not available, skipping enrichment")
        return inventory

    # Only enrich if we have a source image
    if not source_image_path or not Path(source_image_path).is_file():
        print("[REGEN3D-JOB] No source image for enrichment")
        return inventory

    print(f"[REGEN3D-JOB] Enriching inventory with Gemini vision...")

    try:
        # Import LLM client
        from tools.llm_client.client import create_llm_client, LLMProvider

        # Create Gemini client
        client = create_llm_client(provider=LLMProvider.GEMINI)

        # Build the object list for the prompt
        objects_for_prompt = []
        for obj in inventory.get("objects", []):
            if obj.get("id") == "scene_background":
                continue
            objects_for_prompt.append({
                "id": obj["id"],
                "current_category": obj.get("category", "unknown"),
                "approx_location": obj.get("approx_location"),
                "bounds_size": obj.get("bounds", {}).get("size", [1, 1, 1]),
                "is_floor_contact": obj.get("is_floor_contact", False),
            })

        if not objects_for_prompt:
            print("[REGEN3D-JOB] No objects to enrich")
            return inventory

        # Build the enrichment prompt
        environment_type = inventory.get("environment_type", "generic")
        prompt = _build_gemini_enrichment_prompt(objects_for_prompt, environment_type)

        # Call Gemini with the image
        metrics = get_metrics()
        scene_id = inventory.get("scene_id", "")
        with metrics.track_api_call("gemini", "enrich_inventory", scene_id):
            response = client.generate(
                prompt=prompt,
                image=source_image_path,
                json_output=True,
                temperature=0.3,  # Lower temperature for more consistent categorization
            )

        # Parse response
        enrichment_data = response.parse_json()
        enriched_objects = enrichment_data.get("objects", [])

        # Create lookup by ID
        enrichment_by_id = {obj["id"]: obj for obj in enriched_objects}

        # Merge enrichments into inventory
        enriched_count = 0
        for obj in inventory.get("objects", []):
            obj_id = obj.get("id")
            if obj_id in enrichment_by_id:
                enrichment = enrichment_by_id[obj_id]

                # Update category if provided and more specific
                if enrichment.get("category") and enrichment["category"] != "unknown":
                    obj["category"] = enrichment["category"]

                # Update description if provided
                if enrichment.get("description"):
                    obj["short_description"] = enrichment["description"]

                # Update sim_role if provided
                if enrichment.get("sim_role"):
                    obj["sim_role"] = enrichment["sim_role"]

                # Update articulation_hint if provided
                if enrichment.get("articulation_hint"):
                    obj["articulation_hint"] = enrichment["articulation_hint"]

                # Add affordances if provided
                if enrichment.get("affordances"):
                    obj["affordances"] = enrichment["affordances"]

                # Add confidence score from Gemini
                if enrichment.get("confidence"):
                    obj["gemini_confidence"] = enrichment["confidence"]

                enriched_count += 1

        # Update metadata
        inventory.setdefault("metadata", {})
        inventory["metadata"]["gemini_enriched"] = True
        inventory["metadata"]["gemini_enriched_count"] = enriched_count
        inventory["metadata"]["gemini_model"] = response.model

        print(f"[REGEN3D-JOB] Enriched {enriched_count}/{len(objects_for_prompt)} objects with Gemini")
        return inventory

    except ImportError as e:
        print(f"[REGEN3D-JOB] LLM client not available: {e}")
        return inventory
    except Exception as e:
        print(f"[REGEN3D-JOB] Gemini enrichment failed: {e}")
        # Return original inventory on failure (non-fatal)
        return inventory


def _build_gemini_enrichment_prompt(objects: List[Dict], environment_type: str) -> str:
    """Build the prompt for Gemini object enrichment.

    The prompt instructs Gemini to analyze the image and provide enriched
    metadata for each detected object.
    """
    objects_json = json.dumps(objects, indent=2)

    return f'''Analyze this image of a {environment_type} scene and enrich the metadata for each object.

The scene contains the following detected objects:
{objects_json}

For each object, provide:
1. **category**: The most specific object category (e.g., "microwave_oven" not just "appliance")
2. **description**: A concise description (1-2 sentences) useful for robotics simulation
3. **sim_role**: One of:
   - "static": Non-moving scene elements (tables, counters, walls)
   - "interactive": Objects that can be interacted with but not picked up (doors, drawers, switches)
   - "manipulable_object": Objects a robot can pick up and move (dishes, bottles, food items)
   - "articulated_furniture": Furniture with joints (cabinets with doors, drawers)
   - "articulated_appliance": Appliances with joints (ovens, dishwashers, refrigerators)
   - "clutter": Small objects for scene variation
   - "background": Scene shell/background mesh
4. **articulation_hint**: If articulated, one of:
   - "revolute": Rotating joints (doors, lids, hinges)
   - "prismatic": Sliding joints (drawers, sliding doors)
   - null: Not articulated
5. **affordances**: List of possible interactions (e.g., ["openable", "graspable", "heatable"])
6. **confidence**: Your confidence in this classification (0.0 to 1.0)

Respond with JSON in this exact format:
{{
  "objects": [
    {{
      "id": "obj_0",
      "category": "microwave_oven",
      "description": "Countertop microwave oven with digital display and rotating turntable",
      "sim_role": "articulated_appliance",
      "articulation_hint": "revolute",
      "affordances": ["openable", "heatable"],
      "confidence": 0.95
    }}
  ],
  "scene_analysis": {{
    "environment_confirmed": "{environment_type}",
    "notable_features": ["brief notes about the scene"]
  }}
}}

Only include objects from the provided list. Match object IDs exactly.'''


# =============================================================================
# Main Job
# =============================================================================

@dataclass
class ScaleCalibrationResult:
    scale_factor: float
    reference_types: List[str]
    reference_objects: List[str]
    method: str


def _extract_object_height(obj) -> Optional[float]:
    bounds = obj.bounds or {}
    size = bounds.get("size")
    if size and len(size) == 3:
        height = size[1]
        if height and height > 0:
            return height

    min_pt = bounds.get("min")
    max_pt = bounds.get("max")
    if min_pt and max_pt and len(min_pt) == 3 and len(max_pt) == 3:
        height = max_pt[1] - min_pt[1]
        if height > 0:
            return height

    if size and len(size) == 3:
        largest = max(size)
        if largest and largest > 0:
            return largest

    return None


def _auto_calibrate_scale(
    regen3d_output: Regen3DOutput,
    min_samples: int = 1,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
) -> Optional[ScaleCalibrationResult]:
    scale_estimates = []

    for obj in regen3d_output.objects:
        text_parts = [obj.category, obj.description, obj.id]
        text = " ".join([part for part in text_parts if part]).lower()
        if not text:
            continue

        measured_height = _extract_object_height(obj)
        if not measured_height:
            continue

        for ref_key, ref_dims in REFERENCE_DIMENSIONS.items():
            ref_key_text = ref_key.replace("_", " ")
            if ref_key in text or ref_key_text in text:
                expected = ref_dims.get("height") or max(ref_dims.values())
                if not expected or measured_height <= 0:
                    break
                scale = expected / measured_height
                if min_scale <= scale <= max_scale:
                    scale_estimates.append({
                        "scale": scale,
                        "object_id": obj.id,
                        "reference_type": ref_key,
                        "expected_m": expected,
                        "measured_m": measured_height,
                    })
                break

    if len(scale_estimates) < min_samples:
        return None

    scales = [estimate["scale"] for estimate in scale_estimates]
    calibrated_scale = median(scales)
    reference_types = sorted({estimate["reference_type"] for estimate in scale_estimates})
    reference_objects = sorted({estimate["object_id"] for estimate in scale_estimates})

    return ScaleCalibrationResult(
        scale_factor=calibrated_scale,
        reference_types=reference_types,
        reference_objects=reference_objects,
        method="scene_priors",
    )


def run_regen3d_adapter_job(
    root: Path,
    scene_id: str,
    regen3d_prefix: str,
    assets_prefix: str,
    layout_prefix: str,
    environment_type: str = "generic",
    scale_factor: float = 1.0,
    skip_inventory: bool = False,
) -> int:
    """Run the 3D-RE-GEN adapter job.

    Returns:
        0 on success, 1 on failure
    """
    print(f"[REGEN3D-JOB] Starting 3D-RE-GEN adapter for scene: {scene_id}")
    print(f"[REGEN3D-JOB] 3D-RE-GEN prefix: {regen3d_prefix}")
    print(f"[REGEN3D-JOB] Assets prefix: {assets_prefix}")
    print(f"[REGEN3D-JOB] Layout prefix: {layout_prefix}")
    print(f"[REGEN3D-JOB] Environment type: {environment_type}")
    print(f"[REGEN3D-JOB] Scale factor (env/default): {scale_factor}")

    regen3d_dir = root / regen3d_prefix
    assets_dir = root / assets_prefix
    layout_dir = root / layout_prefix

    # Check 3D-RE-GEN outputs exist
    if not regen3d_dir.is_dir():
        print(f"[REGEN3D-JOB] ERROR: 3D-RE-GEN output not found at {regen3d_dir}")
        return 1

    # Initialize adapter
    adapter = Regen3DAdapter(verbose=True)

    # Load 3D-RE-GEN outputs
    try:
        regen3d_output = adapter.load_regen3d_output(regen3d_dir)
        print(f"[REGEN3D-JOB] Loaded {len(regen3d_output.objects)} objects from 3D-RE-GEN")
    except Exception as e:
        print(f"[REGEN3D-JOB] ERROR: Failed to load 3D-RE-GEN outputs: {e}")
        return 1

    if not regen3d_output.objects:
        print("[REGEN3D-JOB] WARNING: No objects found in 3D-RE-GEN output")
    else:
        try:
            calibration = _auto_calibrate_scale(regen3d_output)
        except Exception as e:
            calibration = None
            print(f"[REGEN3D-JOB] WARNING: Auto scale calibration failed: {e}")

        if calibration:
            scale_factor = calibration.scale_factor
            print(
                "[REGEN3D-JOB] Auto scale calibration selected "
                f"scale={scale_factor:.4f} using {calibration.method} "
                f"(refs={calibration.reference_types}, objects={calibration.reference_objects})"
            )
        else:
            print(
                "[REGEN3D-JOB] Auto scale calibration unavailable; "
                f"falling back to SCALE_FACTOR={scale_factor}"
            )

    # Create output directories
    assets_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy assets to expected structure
    print("[REGEN3D-JOB] Copying assets...")
    try:
        asset_paths = adapter.copy_assets(
            regen3d_output,
            root / assets_prefix.rsplit("/", 1)[0],  # Parent of assets dir
            assets_prefix.rsplit("/", 1)[-1],  # Just "assets" part
        )
        print(f"[REGEN3D-JOB] Copied {len(asset_paths)} assets")
    except Exception as e:
        print(f"[REGEN3D-JOB] ERROR: Failed to copy assets: {e}")
        return 1

    # 2. Generate canonical manifest
    print("[REGEN3D-JOB] Generating scene manifest...")
    try:
        manifest = adapter.create_manifest(
            regen3d_output,
            scene_id=scene_id,
            environment_type=environment_type,
            scale_factor=scale_factor,  # (P1-23) Apply scale_factor consistently
        )

        # Validate manifest against schema (P1-21)
        print("[REGEN3D-JOB] Validating manifest against schema...")
        try:
            validate_manifest(manifest)
            print("[REGEN3D-JOB] Manifest validation passed")
        except Exception as validation_error:
            print(f"[REGEN3D-JOB] ERROR: Manifest validation failed: {validation_error}")
            return 1

        manifest_path = assets_dir / "scene_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[REGEN3D-JOB] Wrote manifest: {manifest_path}")
    except Exception as e:
        print(f"[REGEN3D-JOB] ERROR: Failed to generate manifest: {e}")
        return 1

    # 3. Generate layout
    print("[REGEN3D-JOB] Generating scene layout...")
    try:
        layout = adapter.create_layout(regen3d_output, scale_factor)
        layout_path = layout_dir / "scene_layout_scaled.json"
        layout_path.write_text(json.dumps(layout, indent=2))
        print(f"[REGEN3D-JOB] Wrote layout: {layout_path}")
    except Exception as e:
        print(f"[REGEN3D-JOB] ERROR: Failed to generate layout: {e}")
        return 1

    # 4. Generate semantic inventory (for replicator/policy targeting)
    # (P1-22) Make inventory generation failures fatal - required for downstream jobs
    if not skip_inventory:
        print("[REGEN3D-JOB] Generating semantic inventory...")
        try:
            seg_dir = root / f"scenes/{scene_id}/seg"
            seg_dir.mkdir(parents=True, exist_ok=True)

            inventory = generate_semantic_inventory(
                regen3d_output,
                environment_type,
                regen3d_output.source_image_path,
            )

            # Optional Gemini enrichment
            if os.getenv("GEMINI_API_KEY"):
                inventory = enrich_inventory_with_gemini(
                    inventory,
                    regen3d_output.source_image_path,
                )

            inventory_path = seg_dir / "inventory.json"
            inventory_path.write_text(json.dumps(inventory, indent=2))
            print(f"[REGEN3D-JOB] Wrote inventory: {inventory_path}")
        except Exception as e:
            print(f"[REGEN3D-JOB] ERROR: Failed to generate inventory (required for downstream jobs): {e}")
            import traceback
            traceback.print_exc()
            return 1  # Fail the job - inventory is critical for replicator/policy targeting

    # 5. Write completion marker
    # (P2-14) Use only .regen3d_complete as the primary marker.
    # This marker signals that the 3D-RE-GEN adapter job has completed
    # and all outputs (manifest, layout, inventory) are ready for downstream jobs.
    # All downstream workflows should trigger on .regen3d_complete.
    metrics = get_metrics()
    metrics_summary = {
        "backend": metrics.backend.value,
        "stats": metrics.get_stats(),
    }
    marker_content = {
        "status": "complete",
        "scene_id": scene_id,
        "objects_count": len(regen3d_output.objects),
        "has_background": regen3d_output.background is not None,
        "scale_factor": scale_factor,
        "environment_type": environment_type,
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "metrics_summary": metrics_summary,
    }

    # Primary marker: .regen3d_complete
    # This replaces the legacy .regen3d_adapter_complete marker.
    # Downstream jobs should watch for this marker:
    #   - simready-pipeline
    #   - usd-assembly-pipeline (via simready)
    #   - replicator-pipeline
    #   - isaac-lab-pipeline
    primary_marker_path = assets_dir / ".regen3d_complete"
    primary_marker_path.write_text(json.dumps(marker_content, indent=2))
    print(f"[REGEN3D-JOB] Wrote completion marker: {primary_marker_path}")

    print("[REGEN3D-JOB] 3D-RE-GEN adapter completed successfully")
    print(f"[REGEN3D-JOB]   Objects: {len(regen3d_output.objects)}")
    print(f"[REGEN3D-JOB]   Background: {'yes' if regen3d_output.background else 'no'}")
    print(f"[REGEN3D-JOB]   Manifest: {manifest_path}")
    print(f"[REGEN3D-JOB]   Layout: {layout_path}")

    return 0


def main():
    """Main entry point."""
    # Get configuration from environment
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")

    if not scene_id:
        print("[REGEN3D-JOB] ERROR: SCENE_ID is required")
        sys.exit(1)

    # Prefixes with defaults
    regen3d_prefix = os.getenv("REGEN3D_PREFIX", f"scenes/{scene_id}/regen3d")
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    layout_prefix = os.getenv("LAYOUT_PREFIX", f"scenes/{scene_id}/layout")

    # Optional configuration
    environment_type = os.getenv("ENVIRONMENT_TYPE", "generic")
    scale_factor = float(os.getenv("SCALE_FACTOR", "1.0"))
    skip_inventory = os.getenv("SKIP_INVENTORY", "").lower() in ("true", "1", "yes")

    print(f"[REGEN3D-JOB] Configuration:")
    print(f"[REGEN3D-JOB]   Bucket: {bucket}")
    print(f"[REGEN3D-JOB]   Scene ID: {scene_id}")

    metrics = get_metrics()
    with metrics.track_job("regen3d-job", scene_id):
        exit_code = run_regen3d_adapter_job(
            root=GCS_ROOT,
            scene_id=scene_id,
            regen3d_prefix=regen3d_prefix,
            assets_prefix=assets_prefix,
            layout_prefix=layout_prefix,
            environment_type=environment_type,
            scale_factor=scale_factor,
            skip_inventory=skip_inventory,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

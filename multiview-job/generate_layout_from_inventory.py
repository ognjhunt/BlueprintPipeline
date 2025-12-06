#!/usr/bin/env python3
"""
Generate a simple layout JSON from inventory for SAM3D processing.

Since the Gemini pipeline skips the layout-job, we need to create a minimal
layout JSON that SAM3D can use to discover objects.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def create_layout_from_inventory(
    inventory: Dict[str, Any],
    scene_id: str
) -> Dict[str, Any]:
    """
    Create a minimal layout JSON from inventory.

    This layout is used by SAM3D to discover which objects to process.
    """
    objects = []

    # Add all separate asset objects
    for obj in inventory.get("objects", []):
        if obj.get("must_be_separate_asset", False):
            obj_id = obj.get("id")
            category = obj.get("category", "object")

            entry = {
                "id": obj_id,
                "class_name": category,
                "class_id": 0,  # Generic class ID
                "short_description": obj.get("short_description", ""),
                "sim_role": obj.get("sim_role", ""),
            }
            # Carry forward approx_location for synthetic position generation
            approx_loc = obj.get("approx_location")
            if approx_loc:
                entry["approx_location"] = approx_loc
            objects.append(entry)

    # Add scene background as a special object
    background_objects = [
        obj for obj in inventory.get("objects", [])
        if not obj.get("must_be_separate_asset", False)
    ]

    if background_objects:
        objects.append({
            "id": "scene_background",
            "class_name": "scene_background",
            "class_id": 999,  # Special class ID for background
            "short_description": "Static scene background (walls, floor, ceiling, built-in furniture)",
            "sim_role": "scene_shell",
        })

    layout = {
        "scene_id": scene_id,
        "scene_type": inventory.get("scene_type", "interior"),
        "objects": objects,
        "metadata": {
            "generated_from": "inventory",
            "pipeline": "gemini",
            "total_objects": len(objects),
        }
    }

    return layout


def main():
    """Main entry point."""
    seg_prefix = os.getenv("SEG_PREFIX", "")
    scene_id = os.getenv("SCENE_ID", "unknown")

    if not seg_prefix:
        print("[LAYOUT-GEN] ERROR: SEG_PREFIX env var is required", file=sys.stderr)
        sys.exit(1)

    root = Path("/mnt/gcs")
    seg_dir = root / seg_prefix
    inventory_path = seg_dir / "inventory.json"
    layout_output_path = seg_dir / "scene_layout_scaled.json"

    # Check if layout already exists and has objects
    if layout_output_path.exists():
        try:
            with layout_output_path.open("r") as f:
                existing_layout = json.load(f)
            existing_objects = existing_layout.get("objects", [])
            if existing_objects:
                print(f"[LAYOUT-GEN] Layout already exists at {layout_output_path} with {len(existing_objects)} objects, skipping generation")
                return
            else:
                print(f"[LAYOUT-GEN] Layout exists but has no objects, regenerating from inventory")
        except (json.JSONDecodeError, Exception) as e:
            print(f"[LAYOUT-GEN] Error reading existing layout: {e}, regenerating from inventory")

    # Load inventory
    if not inventory_path.exists():
        print(f"[LAYOUT-GEN] ERROR: Inventory not found at {inventory_path}", file=sys.stderr)
        sys.exit(1)

    with inventory_path.open("r") as f:
        inventory = json.load(f)

    print(f"[LAYOUT-GEN] Creating layout from inventory for scene {scene_id}")
    print(f"[LAYOUT-GEN] Inventory has {len(inventory.get('objects', []))} total objects")

    # Create layout
    layout = create_layout_from_inventory(inventory, scene_id)

    print(f"[LAYOUT-GEN] Generated layout with {len(layout.get('objects', []))} objects")

    # Save layout
    layout_output_path.parent.mkdir(parents=True, exist_ok=True)
    with layout_output_path.open("w") as f:
        json.dump(layout, f, indent=2)

    print(f"[LAYOUT-GEN] ✓ Saved layout to {layout_output_path}")


if __name__ == "__main__":
    main()

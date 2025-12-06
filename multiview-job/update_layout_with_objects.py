#!/usr/bin/env python3
"""
Update the layout/scene_layout_scaled.json file with objects from inventory.

This script runs AFTER multiview generation completes to ensure the layout file
in the standard location (layout/) has the correct objects list.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any


def main():
    """Update layout/scene_layout_scaled.json with objects from inventory."""
    seg_prefix = os.getenv("SEG_PREFIX", "")
    scene_id = os.getenv("SCENE_ID", "unknown")

    if not seg_prefix:
        print("[UPDATE-LAYOUT] ERROR: SEG_PREFIX env var is required", file=sys.stderr)
        sys.exit(1)

    # Derive layout prefix from seg prefix (seg -> layout)
    # e.g., "scenes/ChIJ.../seg" -> "scenes/ChIJ.../layout"
    layout_prefix = seg_prefix.replace("/seg", "/layout")

    root = Path("/mnt/gcs")
    seg_dir = root / seg_prefix
    layout_dir = root / layout_prefix

    inventory_path = seg_dir / "inventory.json"
    layout_path = layout_dir / "scene_layout_scaled.json"

    print(f"[UPDATE-LAYOUT] Updating {layout_path} with objects from inventory")
    print(f"[UPDATE-LAYOUT] Reading inventory from: {inventory_path}")

    # Load inventory
    if not inventory_path.exists():
        print(f"[UPDATE-LAYOUT] ERROR: Inventory not found at {inventory_path}", file=sys.stderr)
        sys.exit(1)

    with inventory_path.open("r") as f:
        inventory = json.load(f)

    # Load or create layout structure FIRST to preserve existing spatial data
    if layout_path.exists():
        try:
            with layout_path.open("r") as f:
                layout = json.load(f)
            print(f"[UPDATE-LAYOUT] Loaded existing layout from {layout_path}")
        except (json.JSONDecodeError, Exception) as e:
            print(f"[UPDATE-LAYOUT] Error reading existing layout: {e}, creating new structure")
            layout = {"scene_id": scene_id}
    else:
        print(f"[UPDATE-LAYOUT] Creating new layout structure")
        layout = {"scene_id": scene_id}

    # Build a lookup of existing objects by ID to preserve spatial data (obb, center3d, etc.)
    existing_objects_by_id: Dict[Any, Dict] = {}
    for existing_obj in layout.get("objects", []):
        oid = existing_obj.get("id")
        if oid is not None:
            existing_objects_by_id[oid] = existing_obj
            # Also index by string version for flexibility
            existing_objects_by_id[str(oid)] = existing_obj

    print(f"[UPDATE-LAYOUT] Found {len(existing_objects_by_id) // 2} existing objects with spatial data")

    # Extract objects from inventory, MERGING with existing spatial data
    objects = []
    for obj in inventory.get("objects", []):
        if obj.get("must_be_separate_asset", False):
            obj_id = obj.get("id")
            category = obj.get("category", "object")

            # Start with inventory metadata
            new_obj = {
                "id": obj_id,
                "class_name": category,
                "class_id": 0,
                "short_description": obj.get("short_description", ""),
                "sim_role": obj.get("sim_role", ""),
            }

            # Carry forward approx_location for synthetic position generation
            approx_loc = obj.get("approx_location")
            if approx_loc:
                new_obj["approx_location"] = approx_loc

            # MERGE: Preserve spatial data from existing layout object
            existing_obj = existing_objects_by_id.get(obj_id) or existing_objects_by_id.get(str(obj_id))
            if existing_obj:
                # Preserve critical spatial keys
                spatial_keys = ["obb", "center3d", "center", "bounds", "scale", "approx_location"]
                for key in spatial_keys:
                    if key in existing_obj:
                        new_obj[key] = existing_obj[key]
                        print(f"[UPDATE-LAYOUT]   obj_{obj_id}: preserved '{key}' from layout")

            objects.append(new_obj)

    # Add scene background
    background_objects = [
        obj for obj in inventory.get("objects", [])
        if not obj.get("must_be_separate_asset", False)
    ]

    if background_objects:
        scene_bg_obj = {
            "id": "scene_background",
            "class_name": "scene_background",
            "class_id": 999,
            "short_description": "Static scene background (walls, floor, ceiling, built-in furniture)",
            "sim_role": "scene_shell",
        }
        # Also preserve any existing spatial data for scene_background
        existing_bg = existing_objects_by_id.get("scene_background")
        if existing_bg:
            for key in ["obb", "center3d", "center", "bounds", "scale", "approx_location"]:
                if key in existing_bg:
                    scene_bg_obj[key] = existing_bg[key]
        objects.append(scene_bg_obj)

    print(f"[UPDATE-LAYOUT] Found {len(objects)} objects to add to layout")

    # Update objects (now with merged spatial data preserved)
    layout["objects"] = objects

    # Add metadata
    if "metadata" not in layout:
        layout["metadata"] = {}
    layout["metadata"].update({
        "updated_from": "inventory",
        "pipeline": "gemini",
        "total_objects": len(objects),
    })

    # Save updated layout
    layout_dir.mkdir(parents=True, exist_ok=True)
    with layout_path.open("w") as f:
        json.dump(layout, f, indent=2)

    print(f"[UPDATE-LAYOUT] ✓ Updated layout with {len(objects)} objects at {layout_path}")


if __name__ == "__main__":
    main()

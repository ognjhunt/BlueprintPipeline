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

    root = Path("/mnt/gcs")
    seg_dir = root / seg_prefix
    inventory_path = seg_dir / "inventory.json"

    # Derive layout path from seg_prefix
    # e.g., scenes/ChIJ.../seg -> scenes/ChIJ.../layout
    layout_prefix = seg_prefix.replace("/seg", "/layout")
    layout_dir = root / layout_prefix
    layout_path = layout_dir / "scene_layout_scaled.json"

    print(f"[UPDATE-LAYOUT] Scene ID: {scene_id}")
    print(f"[UPDATE-LAYOUT] Inventory path: {inventory_path}")
    print(f"[UPDATE-LAYOUT] Layout path: {layout_path}")

    # Load inventory
    if not inventory_path.exists():
        print(f"[UPDATE-LAYOUT] ERROR: Inventory not found at {inventory_path}", file=sys.stderr)
        sys.exit(1)

    with inventory_path.open("r") as f:
        inventory = json.load(f)

    print(f"[UPDATE-LAYOUT] Loaded inventory with {len(inventory.get('objects', []))} objects")

    # Load existing layout or create minimal structure
    if layout_path.exists():
        print(f"[UPDATE-LAYOUT] Loading existing layout from {layout_path}")
        with layout_path.open("r") as f:
            layout = json.load(f)
    else:
        print(f"[UPDATE-LAYOUT] Layout file doesn't exist, creating new structure")
        # Create minimal layout structure
        layout = {
            "scene_id": scene_id,
            "objects": [],
        }

    # Build objects list from inventory
    objects = []

    # Add all separate asset objects
    for obj in inventory.get("objects", []):
        if obj.get("must_be_separate_asset", False):
            obj_id = obj.get("id")
            category = obj.get("category", "object")

            objects.append({
                "id": obj_id,
                "class_name": category,
                "class_id": 0,  # Generic class ID
                "short_description": obj.get("short_description", ""),
                "sim_role": obj.get("sim_role", ""),
            })

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

    # Update layout with objects
    layout["objects"] = objects
    layout["scene_id"] = scene_id

    print(f"[UPDATE-LAYOUT] Updating layout with {len(objects)} objects:")
    for obj in objects:
        print(f"[UPDATE-LAYOUT]   - {obj['id']}: {obj.get('short_description', obj['class_name'])}")

    # Ensure layout directory exists
    layout_path.parent.mkdir(parents=True, exist_ok=True)

    # Write updated layout
    with layout_path.open("w") as f:
        json.dump(layout, f, indent=2)

    print(f"[UPDATE-LAYOUT] ✓ Successfully updated {layout_path}")
    print(f"[UPDATE-LAYOUT] Total objects in layout: {len(layout['objects'])}")


if __name__ == "__main__":
    main()

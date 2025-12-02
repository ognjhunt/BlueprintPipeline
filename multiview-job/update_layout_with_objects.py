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

    # Extract objects from inventory
    objects = []
    for obj in inventory.get("objects", []):
        if obj.get("must_be_separate_asset", False):
            obj_id = obj.get("id")
            category = obj.get("category", "object")

            objects.append({
                "id": obj_id,
                "class_name": category,
                "class_id": 0,
                "short_description": obj.get("short_description", ""),
                "sim_role": obj.get("sim_role", ""),
            })

    # Add scene background
    background_objects = [
        obj for obj in inventory.get("objects", [])
        if not obj.get("must_be_separate_asset", False)
    ]

    if background_objects:
        objects.append({
            "id": "scene_background",
            "class_name": "scene_background",
            "class_id": 999,
            "short_description": "Static scene background (walls, floor, ceiling, built-in furniture)",
            "sim_role": "scene_shell",
        })

    print(f"[UPDATE-LAYOUT] Found {len(objects)} objects to add to layout")

    # Load or create layout structure
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

    # Update objects
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

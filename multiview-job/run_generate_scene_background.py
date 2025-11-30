#!/usr/bin/env python3
"""
Generate scene background mesh for static/non-separate-asset elements.

This script creates a scene background image by removing all separate asset objects,
leaving only the static scene elements (walls, floor, ceiling, built-in furniture).
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

from PIL import Image
from google import genai
from google.genai import types


def create_gemini_client():
    """Create Gemini client using API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


def load_inventory(inventory_path: Path) -> Dict[str, Any]:
    """Load scene inventory JSON."""
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")

    with inventory_path.open("r") as f:
        inventory = json.load(f)

    print(f"[BG-GEN] Loaded inventory with {len(inventory.get('objects', []))} objects")
    return inventory


def get_separate_and_background_objects(inventory: Dict[str, Any]) -> tuple[List[Dict], List[Dict]]:
    """
    Split objects into separate assets and background elements.

    Returns:
        (separate_assets, background_objects)
    """
    all_objects = inventory.get("objects", [])

    separate_assets = [
        obj for obj in all_objects
        if obj.get("must_be_separate_asset", False)
    ]

    background_objects = [
        obj for obj in all_objects
        if not obj.get("must_be_separate_asset", False)
    ]

    return separate_assets, background_objects


def build_background_generation_prompt(
    separate_assets: List[Dict[str, Any]],
    background_objects: List[Dict[str, Any]],
    inventory: Dict[str, Any]
) -> str:
    """Generate prompt for scene background generation."""

    # Build list of objects to REMOVE
    objects_to_remove = []
    for obj in separate_assets:
        obj_id = obj.get("id", "unknown")
        desc = obj.get("short_description", "")
        category = obj.get("category", "")
        objects_to_remove.append(f"  - {obj_id}: {desc} ({category})")

    remove_list = "\n".join(objects_to_remove) if objects_to_remove else "  (none)"

    # Build list of objects to KEEP
    objects_to_keep = []
    for obj in background_objects:
        obj_id = obj.get("id", "unknown")
        desc = obj.get("short_description", "")
        sim_role = obj.get("sim_role", "")
        objects_to_keep.append(f"  - {obj_id}: {desc} (sim_role: {sim_role})")

    keep_list = "\n".join(objects_to_keep) if objects_to_keep else "  (none)"

    scene_type = inventory.get("scene_type", "interior")

    prompt = f"""You are a specialized AI for generating clean scene backgrounds for 3D reconstruction.

## Task

You are given an interior scene photograph. Your task is to generate a clean background image that shows ONLY the static scene elements (walls, floor, ceiling, built-in furniture), with all movable/interactive objects removed.

---

## Scene Information

Scene type: {scene_type}
Total objects in scene: {len(separate_assets) + len(background_objects)}

---

## OBJECTS TO REMOVE (paint out / inpaint)

These are separate movable/interactive objects that should be COMPLETELY REMOVED from the scene:

{remove_list}

For each object above:
- Remove it completely from the scene
- Inpaint the space it occupied with appropriate background
- Match the surrounding materials and textures
- Ensure seamless transitions

---

## OBJECTS TO KEEP (preserve in output)

These are the static background elements that should REMAIN in the scene:

{keep_list}

Preserve these elements exactly as they appear in the original image.

---

## Background Generation Requirements

### 1. **Object Removal** (CRITICAL)
   - Remove ALL separate assets listed above completely
   - No traces, shadows, or remnants of removed objects
   - Inpaint removed areas naturally to match surroundings
   - Preserve holes/gaps where objects were attached (e.g., cabinet door hinges should show the cabinet interior)

### 2. **Static Elements Preservation**
   - Keep walls, floor, ceiling exactly as shown
   - Preserve built-in furniture, countertops, cabinet frames
   - Maintain architectural features (windows, doors, trim)
   - Keep lighting fixtures, outlets, switches

### 3. **Inpainting Quality**
   - Match materials: wood grain, tile patterns, wall texture
   - Maintain consistent lighting and shadows
   - Preserve perspective and depth
   - Natural transitions between preserved and inpainted areas

### 4. **Completeness**
   - Show complete background where objects were removed
   - Reveal surfaces that were occluded (counter under kettle, wall behind plant)
   - Maintain structural integrity (no floating elements)

### 5. **Lighting & Color**
   - Match the original scene lighting
   - Preserve color temperature and exposure
   - Keep shadows consistent with light sources
   - No artificial lighting changes

### 6. **Output Format**
   - Same resolution as input image
   - Same camera view/perspective
   - RGB image (no alpha channel needed)
   - High quality, no compression artifacts

---

## Critical Reminders

❌ **REMOVE (paint out):**
- All movable objects (appliances, dishes, plants, kettles, etc.)
- Articulated parts that are separate assets (fridge doors, oven doors, etc.)
- Decorative items marked as separate assets
- Any object listed in "OBJECTS TO REMOVE" above

✅ **KEEP (preserve):**
- Walls, floor, ceiling
- Built-in furniture frames/boxes
- Countertops, backsplash
- Architectural elements
- Any object listed in "OBJECTS TO KEEP" above

---

## Output Specification

Generate **ONE** high-quality image with:
- **Resolution**: Same as input (maintain original dimensions)
- **Format**: RGB image
- **Content**: Clean scene background with separate assets removed
- **View**: Same camera perspective as input
- **Quality**: Seamless inpainting, natural appearance
"""

    return prompt


def extract_image_from_response(response):
    """Extract a single image from Gemini response."""
    # Try candidates first
    for cand in getattr(response, "candidates", []):
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []):
            if getattr(part, "inline_data", None) is not None:
                try:
                    return part.as_image()
                except (AttributeError, Exception):
                    pass

    # Fallback to response.parts
    for part in getattr(response, "parts", []):
        if getattr(part, "inline_data", None) is not None:
            try:
                return part.as_image()
            except (AttributeError, Exception):
                pass

    return None


def generate_scene_background(
    client,
    scene_image: Image.Image,
    separate_assets: List[Dict[str, Any]],
    background_objects: List[Dict[str, Any]],
    inventory: Dict[str, Any],
    output_dir: Path
) -> bool:
    """Generate scene background image with separate assets removed."""

    print(f"[BG-GEN] Generating scene background...")
    print(f"[BG-GEN]   Removing {len(separate_assets)} separate asset objects")
    print(f"[BG-GEN]   Preserving {len(background_objects)} background elements")

    # Build prompt
    prompt = build_background_generation_prompt(separate_assets, background_objects, inventory)

    # Call Gemini
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, scene_image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(image_size="2K"),
                temperature=0.2,  # Lower temperature for more accurate removal/preservation
                tools=[
                    types.Tool(googleSearch=types.GoogleSearch()),
                ],
            ),
        )

        # Extract generated image
        gen_img = extract_image_from_response(response)

        if gen_img is None:
            print("[BG-GEN] ERROR: No image returned from Gemini", file=sys.stderr)
            return False

        # Save background image
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "view_0.png"
        gen_img.save(str(output_path))

        print(f"[BG-GEN] ✓ Saved scene background to {output_path}")

        # Save metadata
        meta = {
            "object_type": "scene_background",
            "generation_method": "gemini-inpainting",
            "model": "gemini-3-pro-image-preview",
            "removed_objects": len(separate_assets),
            "preserved_objects": len(background_objects),
            "scene_type": inventory.get("scene_type", "unknown"),
            "background_object_ids": [obj.get("id") for obj in background_objects]
        }

        meta_path = output_dir / "generation_meta.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        print(f"[BG-GEN] ✓ Saved metadata to {meta_path}")

        return True

    except Exception as e:
        print(f"[BG-GEN] ERROR: Failed to generate background: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""

    # Get environment variables
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    seg_prefix = os.getenv("SEG_PREFIX", "")
    multiview_prefix = os.getenv("MULTIVIEW_PREFIX", "")

    if not seg_prefix or not multiview_prefix:
        print(
            "[BG-GEN] ERROR: SEG_PREFIX and MULTIVIEW_PREFIX env vars are required",
            file=sys.stderr
        )
        sys.exit(1)

    # Setup paths
    root = Path("/mnt/gcs")
    seg_dir = root / seg_prefix
    multiview_root = root / multiview_prefix

    inventory_path = seg_dir / "inventory.json"
    scene_image_path = seg_dir / "dataset" / "valid" / "images" / "room.jpg"

    print(f"[BG-GEN] Bucket: {bucket}")
    print(f"[BG-GEN] Scene ID: {scene_id}")
    print(f"[BG-GEN] Inventory: {inventory_path}")
    print(f"[BG-GEN] Scene image: {scene_image_path}")
    print(f"[BG-GEN] Output root: {multiview_root}")

    # Validate inputs
    if not inventory_path.exists():
        print(f"[BG-GEN] ERROR: Inventory not found at {inventory_path}", file=sys.stderr)
        sys.exit(1)

    if not scene_image_path.exists():
        print(f"[BG-GEN] ERROR: Scene image not found at {scene_image_path}", file=sys.stderr)
        sys.exit(1)

    # Check if background already exists
    # Use obj_ prefix so SAM3D job picks it up automatically
    bg_output_dir = multiview_root / "obj_scene_background"
    if (bg_output_dir / "view_0.png").exists():
        print(f"[BG-GEN] Background already exists at {bg_output_dir}, skipping generation")
        return

    # Load inventory
    inventory = load_inventory(inventory_path)
    separate_assets, background_objects = get_separate_and_background_objects(inventory)

    if not background_objects:
        print("[BG-GEN] WARNING: No background objects found in inventory")
        print("[BG-GEN] This might indicate all objects are separate assets")

    # Load scene image
    print(f"[BG-GEN] Loading scene image: {scene_image_path}")
    scene_image = Image.open(scene_image_path).convert("RGB")
    print(f"[BG-GEN] Scene image size: {scene_image.size[0]}x{scene_image.size[1]}")

    # Create Gemini client
    client = create_gemini_client()

    # Generate background
    success = generate_scene_background(
        client=client,
        scene_image=scene_image,
        separate_assets=separate_assets,
        background_objects=background_objects,
        inventory=inventory,
        output_dir=bg_output_dir
    )

    if success:
        print("[BG-GEN] ✓ Scene background generation complete")
    else:
        print("[BG-GEN] ✗ Scene background generation failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

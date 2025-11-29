#!/usr/bin/env python3
"""
Gemini generative multiview generation (replaces crop-based approach).

This script uses Gemini 3.0 Pro Image to generate isolated object renders
directly from the full scene image, without any cropping or segmentation.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

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

    print(f"[MULTIVIEW-GEN] Loaded inventory with {len(inventory.get('objects', []))} objects")
    return inventory


def build_object_details_block(obj: Dict[str, Any]) -> str:
    """Format object details for the prompt."""
    lines = []

    # Use proper JSON formatting for the object details
    obj_json = {
        "id": obj.get("id"),
        "category": obj.get("category"),
        "short_description": obj.get("short_description"),
        "approx_location": obj.get("approx_location"),
        "relationships": obj.get("relationships", [])
    }

    return json.dumps(obj_json, indent=4)


def build_scene_objects_list(inventory: Dict[str, Any], exclude_id: str) -> str:
    """Build a formatted list of all scene objects (excluding the target)."""
    objects = inventory.get("objects", [])
    other_objects = [obj for obj in objects if obj.get("id") != exclude_id]

    if not other_objects:
        return "No other objects in scene."

    # Format as a simple bulleted list
    lines = []
    for obj in other_objects:
        obj_id = obj.get("id", "unknown")
        desc = obj.get("short_description", "")
        category = obj.get("category", "")
        lines.append(f"  - {obj_id}: {desc} ({category})")

    return "\n".join(lines)


def generate_object_reconstruction_prompt(
    target_obj: Dict[str, Any],
    inventory: Dict[str, Any]
) -> str:
    """Generate the full prompt for object reconstruction."""

    object_details = build_object_details_block(target_obj)
    scene_objects = build_scene_objects_list(inventory, target_obj.get("id", ""))
    total_objects = len(inventory.get("objects", []))

    prompt = f"""You are a specialized AI for generating isolated 3D-ready object renders from interior scene photographs.

## Task
You are given a scene reference image containing multiple objects. Your task is to generate an isolated, high-quality render of ONLY the specified target object below, excluding all other objects and scene context.

---

## TARGET OBJECT (Generate ONLY this object)

{object_details}

---

## SCENE CONTEXT - ALL OTHER OBJECTS TO EXCLUDE

The scene contains {total_objects} total objects. **You must EXCLUDE all of these except the target object:**

{scene_objects}

---

## Reconstruction Requirements

### 1. **Object Isolation** (CRITICAL)
   - Render **ONLY** the target object specified above
   - **EXCLUDE** every other object from the scene
   - **REMOVE** all context: walls, floors, ceilings, shelves, tables, countertops, adjacent objects, decorative elements
   - The object must appear as if photographed in a professional product studio

### 2. **Shape & Geometry**
   - Infer the complete 3D form based on visible portions in the scene
   - Maintain proportions consistent with the visible silhouette
   - Reconstruct occluded/hidden parts plausibly (back, sides, bottom)
   - Keep realistic proportions - do not exaggerate or distort

### 3. **Materials & Surface**
   - Preserve the exact material type visible in the scene (wood, metal, ceramic, plastic, fabric, glass, etc.)
   - Match the color palette precisely
   - Carry over surface details: texture, grain, scratches, wear patterns, glaze, finish
   - Maintain realistic material properties

### 4. **Lighting & Rendering**
   - Use soft, neutral, studio-quality lighting (3-point lighting style)
   - Lighting should reveal form and surface detail clearly
   - Avoid harsh shadows or dramatic lighting
   - Maintain even, professional illumination

### 5. **Camera & Framing**
   - Render a **single front-facing orthographic view** (straight-on, centered)
   - Center the object in the frame with small margin (~5-10%)
   - Show the complete object - do not crop any part
   - Camera positioned at object's vertical center

### 6. **Background**
   - **Transparent background (alpha channel)** - mandatory
   - No shadows, reflections, or ground plane
   - Object should float in empty space

### 7. **Consistency**
   - The reconstructed object must be recognizable as the same item from the source scene
   - Do not add accessories or embellishments not visible in the original
   - Do not change style, era, or design
   - Complete hidden areas conservatively to match visible style

---

## Output Specification

Generate **ONE** high-resolution PNG image with:
- **Resolution**: 2048px or higher (largest dimension)
- **Format**: PNG with alpha transparency
- **Content**: The isolated target object ONLY
- **View**: Front-facing orthographic
- **Background**: Fully transparent

---

## Critical Reminders

❌ **DO NOT INCLUDE:**
- Other objects from the scene
- Walls, floors, ceilings, architectural elements
- Supporting surfaces (tables, shelves, countertops, cabinets)
- Shadows or reflections from original scene
- Background context or environment

✅ **DO INCLUDE:**
- Only the specified target object
- Complete object geometry
- Accurate materials and surface details
- Professional studio lighting
- Transparent background
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


def generate_isolated_object(
    client,
    scene_image: Image.Image,
    target_obj: Dict[str, Any],
    inventory: Dict[str, Any],
    output_dir: Path
):
    """Generate an isolated object render using Gemini."""

    obj_id = target_obj.get("id", "unknown")
    obj_desc = target_obj.get("short_description", "object")

    print(f"[MULTIVIEW-GEN] Generating isolated render for '{obj_id}': {obj_desc}")

    # Build prompt
    prompt = generate_object_reconstruction_prompt(target_obj, inventory)

    # Call Gemini
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, scene_image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(image_size="2K"),
                temperature=0.3,  # Slightly higher for creative reconstruction
                grounding=types.GroundingConfig(
                    google_search=types.GoogleSearch()
                ),
            ),
        )

        # Extract generated image
        gen_img = extract_image_from_response(response)

        if gen_img is None:
            print(
                f"[MULTIVIEW-GEN] WARNING: No image returned for object '{obj_id}'",
                file=sys.stderr
            )
            return False

        # Save to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "view_0.png"
        gen_img.save(str(output_path))

        print(f"[MULTIVIEW-GEN] ✓ Saved isolated render to {output_path}")

        # Save metadata
        meta = {
            "object_id": obj_id,
            "short_description": obj_desc,
            "category": target_obj.get("category"),
            "generation_method": "gemini-generative",
            "model": "gemini-3-pro-image-preview"
        }
        meta_path = output_dir / "generation_meta.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        return True

    except Exception as e:
        print(
            f"[MULTIVIEW-GEN] ERROR: Failed to generate object '{obj_id}': {e}",
            file=sys.stderr
        )
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
            "[MULTIVIEW-GEN] ERROR: SEG_PREFIX and MULTIVIEW_PREFIX env vars are required",
            file=sys.stderr
        )
        sys.exit(1)

    # Setup paths
    root = Path("/mnt/gcs")
    seg_dir = root / seg_prefix
    multiview_root = root / multiview_prefix

    inventory_path = seg_dir / "inventory.json"
    scene_image_path = seg_dir / "dataset" / "valid" / "images" / "room.jpg"

    print(f"[MULTIVIEW-GEN] Bucket: {bucket}")
    print(f"[MULTIVIEW-GEN] Scene ID: {scene_id}")
    print(f"[MULTIVIEW-GEN] Inventory: {inventory_path}")
    print(f"[MULTIVIEW-GEN] Scene image: {scene_image_path}")
    print(f"[MULTIVIEW-GEN] Output root: {multiview_root}")

    # Validate inputs
    if not inventory_path.exists():
        print(
            f"[MULTIVIEW-GEN] ERROR: Inventory not found at {inventory_path}",
            file=sys.stderr
        )
        sys.exit(1)

    if not scene_image_path.exists():
        print(
            f"[MULTIVIEW-GEN] ERROR: Scene image not found at {scene_image_path}",
            file=sys.stderr
        )
        sys.exit(1)

    # Load inventory
    inventory = load_inventory(inventory_path)
    objects = inventory.get("objects", [])

    if not objects:
        print("[MULTIVIEW-GEN] No objects in inventory; nothing to do.")
        return

    # Load scene image
    print(f"[MULTIVIEW-GEN] Loading scene image: {scene_image_path}")
    scene_image = Image.open(scene_image_path).convert("RGB")
    print(f"[MULTIVIEW-GEN] Scene image size: {scene_image.size[0]}x{scene_image.size[1]}")

    # Create Gemini client
    client = create_gemini_client()

    # Process each object
    multiview_root.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    for obj in objects:
        obj_id = obj.get("id", "unknown")
        output_dir = multiview_root / f"obj_{obj_id}"

        success = generate_isolated_object(
            client=client,
            scene_image=scene_image,
            target_obj=obj,
            inventory=inventory,
            output_dir=output_dir
        )

        if success:
            success_count += 1
        else:
            failure_count += 1

    print(f"\n[MULTIVIEW-GEN] ===== Summary =====")
    print(f"[MULTIVIEW-GEN] Total objects: {len(objects)}")
    print(f"[MULTIVIEW-GEN] Successful: {success_count}")
    print(f"[MULTIVIEW-GEN] Failed: {failure_count}")
    print(f"[MULTIVIEW-GEN] Done.")


if __name__ == "__main__":
    main()

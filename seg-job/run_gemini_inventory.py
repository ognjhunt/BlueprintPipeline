#!/usr/bin/env python3
"""
Gemini-based scene inventory generation (replaces SAM3 segmentation).

This script analyzes a scene image using Gemini 3.0 Pro and generates
a structured inventory of all objects in the scene.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from google import genai
from google.genai import types


def create_gemini_client():
    """Create Gemini client using API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


def load_scene_image(images_dir: Path) -> tuple[Path, Image.Image]:
    """Load the first valid image from the images directory."""
    supported_exts = {".jpg", ".jpeg", ".png"}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() in supported_exts:
            print(f"[GEMINI-INV] Loading image: {img_path}")
            try:
                img = Image.open(img_path).convert("RGB")
                return img_path, img
            except Exception as e:
                print(f"[GEMINI-INV] WARNING: Failed to load {img_path}: {e}", file=sys.stderr)
                continue

    raise FileNotFoundError(f"No valid images found in {images_dir}")


def generate_inventory_prompt() -> str:
    """Generate the Gemini prompt for scene inventory analysis (simulation-aware)."""
    return """You are analyzing an interior scene photograph to create an object inventory for a ROBOTICS SIMULATION (Isaac Sim / USD-style).

We do NOT want every tiny part. We want a compact set of SIMULATION-LEVEL objects.

## Task

Analyze the image and identify a SMALL set of objects that matter for simulation:

1. Large static structures the robot may collide with:
   - walls, floor, ceiling, room shell
   - large built-in cabinet runs
   - continuous countertops
   - door + frame, window + frame + blinds

2. Large appliances:
   - refrigerator, oven, stove, dishwasher, microwave, etc.

3. Articulated parts that may move:
   - fridge door, oven door, hinged doors, drawers, cabinet doors the robot might open.

4. Manipulable objects:
   - items a robot arm could pick up, move, or push:
     dishes, glasses, bowls, mugs, plants, kettles, jars, utensils, tools, boxes, etc.

Do NOT split small sub-parts (handles, baseboards, individual hinges, trim pieces) into separate objects unless they must be separately simulated.

## Output Format

Return ONLY this JSON:

{
  "scene_type": "kitchen" | "bedroom" | "living_room" | "bathroom" | "office" | "other",
  "objects": [
    {
      "id": "string_snake_case_id",
      "category": "furniture" | "appliance" | "fixture" | "kitchenware" | "decor" | "plant" | "architectural_element" | "other",
      "sim_role": "scene_shell" | "static_furniture_block" | "appliance" | "articulated_base" | "articulated_part" | "manipulable_object" | "ignore_for_sim",
      "must_be_separate_asset": true | false,
      "short_description": "5-15 word description",
      "approx_location": "top left" | "middle center" | "bottom right" | etc.,
      "relationships": ["above X", "on Y", "left of Z", ...],
      "parent_id": "id_of_parent_if_articulated_part_or_nested",
      "grouping_hint": "optional free-text note about what visual parts are included in this object"
    }
  ]
}

## Granularity Rules

- Aim for roughly **10–30 objects total**, not dozens of tiny parts.
- Merge visually connected cabinetry and counters into 1–3 blocks:
  - e.g., "base_cabinets_and_counter", "upper_cabinets_main".
- Treat the following as part of their parent object **not separate assets**:
  - baseboards, toe kicks, small trim, backsplash tiles
  - door and cabinet handles
  - faucet knobs, hinges, small mounting hardware
- For a window with a blind, you may represent it as one object: "window_block".
- For sets of small identical items (3 wine glasses on a shelf), do this:
  - If the robot might pick them individually: create several objects with ids glass_1, glass_2, etc., sim_role="manipulable_object".
  - If they are just decorative clutter: create ONE object (e.g., "glass_set_1") with sim_role="ignore_for_sim" or "decor" and must_be_separate_asset=false.

## Articulation

If an object obviously has a moving part (door, drawer):

- Create one object for the main body:
  - sim_role = "articulated_base"
- Create one object per moving part:
  - sim_role = "articulated_part"
  - parent_id = id of the base object
- Only do this when the movement is reasonably clear from the image (e.g., cabinet doors, oven door).

## Simulation Asset Flag

Set "must_be_separate_asset" to:

- true:
  - manipulable_object
  - appliance the robot might collide with directly
  - articulated_base and articulated_part
- false:
  - scene_shell
  - most static_furniture_block
  - ignore_for_sim / clutter-only decor

## sim_role Values

Use these sim_role values appropriately:

- **scene_shell**: walls, floor, ceiling, baseboard, large architectural bits
- **static_furniture_block**: base-cabinet+counter block, upper-cabinet block, big shelf walls, etc.
- **appliance**: fridge, oven, microwave, dishwasher body
- **articulated_base**: the main body of an articulated object (fridge, oven, door frame)
- **articulated_part**: doors, drawers, handles you actually want to simulate as joints (with parent_id)
- **manipulable_object**: anything likely to be picked/placed or pushed (kettle, plant, dishes, utensils, jars)
- **ignore_for_sim**: things you don't care about at all (ceiling lights, tiny trim, text labels)

## ID Assignment

- Use descriptive lowercase snake_case IDs: "refrigerator_1", "base_cabinets_and_counter", "plant_pot_1"
- For multiples of the same type, use sequential numbers
- IDs should be stable and meaningful

## Descriptions

- Keep descriptions concise but informative (5-15 words)
- Include material, color, and distinguishing features
- Examples:
  - "L-shaped white base cabinets with wooden countertop"
  - "Tall white built-in refrigerator with flat panel doors"
  - "Stainless steel electric kettle"

## Approximate Location

Use a simple grid system:
- Horizontal: left, center, right
- Vertical: top, middle, bottom
- Examples: "top left", "middle center", "bottom right"

## Relationships

List spatial relationships to other objects using their IDs:
- Position: "above X", "below Y", "left of Z", "right of W"
- Containment: "inside X", "on Y", "in Z"
- Attachment: "attached_to X"
- Examples: ["above refrigerator_1", "below upper_cabinets_main", "on base_cabinets_and_counter"]

## Quality Requirements

- Prefer simulation-level objects over tiny parts
- Be consistent with ids and sim_role values
- Keep descriptions informative but concise
- Aim for 10-30 objects total (not 60+)

## Output

Return ONLY the JSON object, with no additional text or markdown formatting.
"""


def parse_gemini_inventory(response) -> Dict[str, Any]:
    """Parse Gemini response and extract inventory JSON."""
    # Get text from response
    text = getattr(response, "text", "")
    if not text:
        raise ValueError("Empty response from Gemini")

    # Clean up markdown code blocks if present
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Parse JSON
    try:
        inventory = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[GEMINI-INV] ERROR: Failed to parse JSON response: {e}", file=sys.stderr)
        print(f"[GEMINI-INV] Raw response:\n{text}", file=sys.stderr)
        raise

    # Validate structure
    if not isinstance(inventory, dict):
        raise ValueError("Inventory must be a JSON object")
    if "objects" not in inventory:
        raise ValueError("Inventory missing 'objects' field")
    if not isinstance(inventory["objects"], list):
        raise ValueError("'objects' field must be a list")

    # Validate and log new simulation fields
    objects = inventory.get("objects", [])
    objects_with_sim_fields = 0
    separate_assets = 0

    valid_sim_roles = {
        "scene_shell", "static_furniture_block", "appliance",
        "articulated_base", "articulated_part", "manipulable_object", "ignore_for_sim"
    }

    for obj in objects:
        # Check for new sim_role field
        if "sim_role" in obj:
            objects_with_sim_fields += 1
            sim_role = obj.get("sim_role")
            if sim_role not in valid_sim_roles:
                print(
                    f"[GEMINI-INV] WARNING: Object '{obj.get('id', 'unknown')}' has invalid sim_role: '{sim_role}'",
                    file=sys.stderr
                )

        # Count separate assets
        if obj.get("must_be_separate_asset", False):
            separate_assets += 1

        # Validate articulated_part has parent_id
        if obj.get("sim_role") == "articulated_part" and not obj.get("parent_id"):
            print(
                f"[GEMINI-INV] WARNING: Articulated part '{obj.get('id', 'unknown')}' missing parent_id",
                file=sys.stderr
            )

    print(f"[GEMINI-INV] Objects with sim_role field: {objects_with_sim_fields}/{len(objects)}")
    print(f"[GEMINI-INV] Objects marked as separate assets: {separate_assets}/{len(objects)}")

    return inventory


def analyze_scene_with_gemini(client, image: Image.Image) -> Dict[str, Any]:
    """Analyze scene image with Gemini and return structured inventory."""
    print("[GEMINI-INV] Calling Gemini 3.0 Pro for scene analysis...")

    prompt = generate_inventory_prompt()

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                # Using default temperature=1.0 as recommended for Gemini 3
                response_mime_type="application/json",
                tools=[
                    types.Tool(googleSearch=types.GoogleSearch()),
                ],
            ),
        )

        inventory = parse_gemini_inventory(response)

        num_objects = len(inventory.get("objects", []))
        scene_type = inventory.get("scene_type", "unknown")
        print(f"[GEMINI-INV] Analysis complete: {num_objects} objects found in {scene_type} scene")

        return inventory

    except Exception as e:
        print(f"[GEMINI-INV] ERROR: Gemini analysis failed: {e}", file=sys.stderr)
        raise


def save_inventory(inventory: Dict[str, Any], output_path: Path):
    """Save inventory JSON to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(inventory, f, indent=2)

    print(f"[GEMINI-INV] Saved inventory to {output_path}")


def setup_dataset_structure(images_dir: Path, seg_dir: Path, source_image_path: Path):
    """Create minimal dataset structure for compatibility with downstream jobs."""
    dataset_dir = seg_dir / "dataset" / "valid"
    images_out_dir = dataset_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # Copy source image to expected location
    dest_image = images_out_dir / "room.jpg"
    shutil.copy2(source_image_path, dest_image)
    print(f"[GEMINI-INV] Copied source image to {dest_image}")

    # Create minimal data.yaml for compatibility
    # (not used by new pipeline, but keeps old code from breaking)
    data_yaml = seg_dir / "dataset" / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = """# Gemini-based inventory (no YOLO segmentation)
path: .
train:
val: valid/images
names:
  0: object
"""

    with data_yaml.open("w") as f:
        f.write(yaml_content)

    print(f"[GEMINI-INV] Created minimal dataset structure at {seg_dir}/dataset")


def main():
    """Main entry point."""
    # Get environment variables
    bucket = os.getenv("BUCKET", "")
    images_prefix = os.getenv("IMAGES_PREFIX", "")
    seg_prefix = os.getenv("SEG_PREFIX", "")

    if not images_prefix or not seg_prefix:
        print("[GEMINI-INV] ERROR: IMAGES_PREFIX and SEG_PREFIX env vars are required", file=sys.stderr)
        sys.exit(1)

    # Setup paths
    root = Path("/mnt/gcs")
    images_dir = root / images_prefix
    seg_dir = root / seg_prefix

    print(f"[GEMINI-INV] Bucket: {bucket}")
    print(f"[GEMINI-INV] Images directory: {images_dir}")
    print(f"[GEMINI-INV] Output directory: {seg_dir}")

    if not images_dir.exists():
        print(f"[GEMINI-INV] ERROR: Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    # Load image
    source_image_path, image = load_scene_image(images_dir)
    print(f"[GEMINI-INV] Loaded image: {image.size[0]}x{image.size[1]}")

    # Create Gemini client
    client = create_gemini_client()

    # Analyze scene
    inventory = analyze_scene_with_gemini(client, image)

    # Save inventory
    inventory_path = seg_dir / "inventory.json"
    save_inventory(inventory, inventory_path)

    # Setup dataset structure for compatibility
    setup_dataset_structure(images_dir, seg_dir, source_image_path)

    print("[GEMINI-INV] Done.")


if __name__ == "__main__":
    main()

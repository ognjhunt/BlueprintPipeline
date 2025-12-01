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

This inventory will be used to generate 3D assets for robotic training, where robots need to:
- Navigate through the space
- Open doors and drawers
- Manipulate objects
- Interact with articulated furniture and appliances

## CRITICAL REQUIREMENT FOR ROBOTIC TRAINING

**ALWAYS SEPARATE ARTICULATED COMPONENTS:**
- Every cabinet door, drawer, appliance door MUST be a separate object
- Cabinet frames and doors are NEVER merged together
- Each moving part gets its own object for physics simulation

## Task

Analyze the image and identify objects that matter for simulation:

1. **Large static structures** (stay in background shell):
   - walls, floor, ceiling, room shell
   - countertops (if not articulated)
   - fixed shelving
   - backsplash, baseboards

2. **Articulated furniture** (EACH COMPONENT SEPARATE):
   - Cabinet frame → articulated_base
   - Each cabinet door → articulated_part
   - Each drawer front → articulated_part
   - Refrigerator body → articulated_base
   - Refrigerator door(s) → articulated_part
   - Oven body → articulated_base
   - Oven door → articulated_part

3. **Appliances** (without visible articulation):
   - stovetop, dishwasher body, microwave body (if doors not visible)

4. **Manipulable objects** (robot can pick up):
   - dishes, glasses, bowls, mugs, plants, kettles, jars, utensils, tools, boxes, etc.

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

- Aim for roughly **10–40 objects total**, not dozens of tiny parts.
- **CRITICAL FOR ARTICULATED FURNITURE:**
  - **DO NOT merge cabinets, drawers, or furniture with visible doors/drawers into blocks**
  - Each cabinet frame, door, and drawer MUST be a separate object for articulation
  - Examples:
    - Upper cabinet with door → TWO objects: cabinet frame (articulated_base) + door (articulated_part)
    - Drawer unit with 4 drawers → FIVE objects: frame (articulated_base) + 4 drawer fronts (articulated_part)
    - Base cabinets with 2 doors and 3 drawers → SIX objects: frame + 2 doors + 3 drawers
  - **ONLY merge cabinetry if it has NO visible doors/drawers** (e.g., solid toe kick, fixed panels)
- Countertops:
  - If countertop is separate from cabinets → separate object with sim_role="static_furniture_block", must_be_separate_asset=false
  - If countertop is integrated with closed cabinets → can merge with cabinet block
- Treat the following as part of their parent object **not separate assets**:
  - baseboards, toe kicks, small trim, backsplash tiles
  - door and cabinet handles (the handle itself, but the door IS a separate asset)
  - faucet knobs, hinges, small mounting hardware
- For a window with a blind, you may represent it as one object: "window_block".
- For sets of small identical items (3 wine glasses on a shelf), do this:
  - If the robot might pick them individually: create several objects with ids glass_1, glass_2, etc., sim_role="manipulable_object".
  - If they are just decorative clutter: create ONE object (e.g., "glass_set_1") with sim_role="ignore_for_sim" or "decor" and must_be_separate_asset=false.

## Articulation (CRITICAL FOR ROBOTIC TRAINING)

**ALWAYS identify and separate articulated components for robotic training:**

For ANY furniture or appliance with visible doors, drawers, or moving parts:

1. **Create the base/frame object:**
   - sim_role = "articulated_base"
   - must_be_separate_asset = true
   - This is the fixed part (cabinet box, refrigerator body, oven body, etc.)

2. **Create separate objects for EACH moving part:**
   - sim_role = "articulated_part"
   - must_be_separate_asset = true
   - parent_id = id of the base object
   - One object per door, drawer, or moving component

**Common articulated objects to identify:**
- **Cabinet doors**: Each hinged door is a separate articulated_part
- **Drawers**: Each drawer front is a separate articulated_part
- **Refrigerator**: Body is articulated_base, each door/freezer door is articulated_part
- **Oven**: Body is articulated_base, oven door is articulated_part
- **Dishwasher**: Body is articulated_base, door is articulated_part
- **Microwave**: Body is articulated_base, door is articulated_part

**Examples:**
- Kitchen with upper cabinet (2 doors), lower cabinet (1 door + 2 drawers):
  - upper_cabinet_frame (articulated_base, must_be_separate_asset=true)
  - upper_cabinet_door_left (articulated_part, must_be_separate_asset=true, parent_id="upper_cabinet_frame")
  - upper_cabinet_door_right (articulated_part, must_be_separate_asset=true, parent_id="upper_cabinet_frame")
  - lower_cabinet_frame (articulated_base, must_be_separate_asset=true)
  - lower_cabinet_door (articulated_part, must_be_separate_asset=true, parent_id="lower_cabinet_frame")
  - drawer_1 (articulated_part, must_be_separate_asset=true, parent_id="lower_cabinet_frame")
  - drawer_2 (articulated_part, must_be_separate_asset=true, parent_id="lower_cabinet_frame")

**Do NOT create articulated objects for:**
- Handles, knobs, hinges (these are part of the door/drawer object)
- Fixed panels or closed cabinetry with no visible access

## Simulation Asset Flag (CRITICAL)

Set "must_be_separate_asset" based on whether the object needs individual 3D model generation:

**must_be_separate_asset = true (will be removed from scene shell and get individual 3D model):**
- ALL manipulable_object (items robots can pick up)
- ALL appliance bodies (refrigerator, oven, microwave, dishwasher)
- ALL articulated_base (cabinet frames, appliance bodies with doors)
- ALL articulated_part (every door, drawer, moving component)

**must_be_separate_asset = false (will remain in static scene shell background):**
- ALL scene_shell (walls, floor, ceiling, baseboards)
- MOST static_furniture_block (countertops, fixed shelving, solid panels)
- ALL ignore_for_sim / decorative clutter

**REMEMBER:** If a cabinet has doors or drawers visible, the frame AND each door/drawer MUST have must_be_separate_asset=true

## sim_role Values

Use these sim_role values appropriately:

- **scene_shell**: walls, floor, ceiling, baseboards, large architectural elements
  - Always must_be_separate_asset=false (stays in background shell)

- **static_furniture_block**: ONLY for furniture with NO moving parts
  - Examples: solid countertops, fixed shelving, solid panels, open shelves
  - Usually must_be_separate_asset=false (stays in background shell)
  - DO NOT use for cabinets with doors/drawers

- **appliance**: Large appliance bodies (when they DON'T have visible doors/drawers)
  - Examples: stovetop, built-in microwave (if door not visible)
  - Usually must_be_separate_asset=true

- **articulated_base**: Main body/frame of furniture or appliances WITH moving parts
  - Examples: cabinet frame (when it has doors/drawers), refrigerator body, oven body
  - ALWAYS must_be_separate_asset=true
  - Child objects will reference this as parent_id

- **articulated_part**: Every door, drawer, or moving component
  - Examples: cabinet door, drawer front, refrigerator door, oven door, freezer door
  - ALWAYS must_be_separate_asset=true
  - ALWAYS has parent_id pointing to the articulated_base

- **manipulable_object**: Items robots can pick up, move, or push
  - Examples: kettle, plant, dishes, utensils, jars, bowls, mugs, boxes
  - ALWAYS must_be_separate_asset=true

- **ignore_for_sim**: Decorative elements not needed for simulation
  - Examples: ceiling lights, tiny trim, text labels, small decorative items
  - Always must_be_separate_asset=false (stays in background shell)

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
- Aim for 10-40 objects total for typical rooms
- ALWAYS separate articulated components (cabinet frames, doors, drawers)

## Example Output for Kitchen Scene

For a kitchen with upper cabinets (2 doors), lower cabinets (2 doors + 3 drawers), refrigerator, oven, and countertop:

```json
{
  "scene_type": "kitchen",
  "objects": [
    {
      "id": "walls",
      "sim_role": "scene_shell",
      "must_be_separate_asset": false,
      "short_description": "Kitchen walls with white paint"
    },
    {
      "id": "floor",
      "sim_role": "scene_shell",
      "must_be_separate_asset": false,
      "short_description": "Light wood laminate flooring"
    },
    {
      "id": "countertop",
      "sim_role": "static_furniture_block",
      "must_be_separate_asset": false,
      "short_description": "L-shaped wooden countertop surface"
    },
    {
      "id": "upper_cabinet_left_frame",
      "sim_role": "articulated_base",
      "must_be_separate_asset": true,
      "short_description": "White upper cabinet frame with two door openings"
    },
    {
      "id": "upper_cabinet_left_door_1",
      "sim_role": "articulated_part",
      "must_be_separate_asset": true,
      "parent_id": "upper_cabinet_left_frame",
      "short_description": "Left hinged white cabinet door with black handle"
    },
    {
      "id": "upper_cabinet_left_door_2",
      "sim_role": "articulated_part",
      "must_be_separate_asset": true,
      "parent_id": "upper_cabinet_left_frame",
      "short_description": "Right hinged white cabinet door with black handle"
    },
    {
      "id": "lower_cabinet_frame",
      "sim_role": "articulated_base",
      "must_be_separate_asset": true,
      "short_description": "White lower cabinet frame with drawer and door openings"
    },
    {
      "id": "drawer_1",
      "sim_role": "articulated_part",
      "must_be_separate_asset": true,
      "parent_id": "lower_cabinet_frame",
      "short_description": "Top drawer with white front and black handle"
    },
    {
      "id": "drawer_2",
      "sim_role": "articulated_part",
      "must_be_separate_asset": true,
      "parent_id": "lower_cabinet_frame",
      "short_description": "Middle drawer with white front and black handle"
    },
    {
      "id": "drawer_3",
      "sim_role": "articulated_part",
      "must_be_separate_asset": true,
      "parent_id": "lower_cabinet_frame",
      "short_description": "Bottom drawer with white front and black handle"
    },
    {
      "id": "refrigerator_body",
      "sim_role": "articulated_base",
      "must_be_separate_asset": true,
      "short_description": "Tall white refrigerator body frame"
    },
    {
      "id": "refrigerator_door",
      "sim_role": "articulated_part",
      "must_be_separate_asset": true,
      "parent_id": "refrigerator_body",
      "short_description": "White refrigerator door with flat panel"
    },
    {
      "id": "oven_body",
      "sim_role": "articulated_base",
      "must_be_separate_asset": true,
      "short_description": "Black built-in oven body"
    },
    {
      "id": "oven_door",
      "sim_role": "articulated_part",
      "must_be_separate_asset": true,
      "parent_id": "oven_body",
      "short_description": "Black oven door with glass window"
    },
    {
      "id": "kettle_1",
      "sim_role": "manipulable_object",
      "must_be_separate_asset": true,
      "short_description": "Stainless steel electric kettle on countertop"
    }
  ]
}
```

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

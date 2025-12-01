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
    """Generate the Gemini prompt for scene inventory analysis (PhysX-Anything compatible)."""
    return """You are analyzing an interior scene photograph to create an object inventory for ROBOTICS SIMULATION using PhysX-Anything.

This inventory will be used to generate 3D assets for robotic training, where robots need to:
- Navigate through the space
- Open doors and drawers
- Manipulate objects
- Interact with articulated furniture and appliances

## Important: PhysX-Anything Automatic Articulation

**PhysX-Anything automatically detects and segments articulated parts from single images.**

This means:
- Keep articulated furniture/appliances as SINGLE complete objects (don't manually separate doors/drawers)
- The system will automatically identify hinges, joints, and moving parts
- No need to pre-segment cabinet frames from doors or appliance bodies from doors
- Just describe what articulation exists (e.g., "2 hinged doors", "3 drawers")

## Task

Analyze the image and identify objects that matter for simulation:

1. **Large static structures** (stay in background shell):
   - walls, floor, ceiling, room shell
   - countertops (if separate from cabinets)
   - fixed shelving, backsplash, baseboards

2. **Articulated furniture** (keep as COMPLETE objects):
   - Cabinets with doors/drawers → single object with articulation_hint
   - Dressers, wardrobes → single object describing movable parts
   - Tables with drawers → single object

3. **Articulated appliances** (keep as COMPLETE objects):
   - Refrigerator → single object (system will detect doors)
   - Oven → single object (system will detect door)
   - Dishwasher, microwave → single objects

4. **Non-articulated appliances**:
   - stovetop, range hood, fixed microwave (no visible doors)

5. **Manipulable objects** (robot can pick up):
   - dishes, glasses, bowls, mugs, plants, kettles, jars, utensils, tools, boxes, etc.

Do NOT split small sub-parts (handles, baseboards, individual hinges, trim pieces) into separate objects.

## Output Format

Return ONLY this JSON:

{
  "scene_type": "kitchen" | "bedroom" | "living_room" | "bathroom" | "office" | "other",
  "objects": [
    {
      "id": "string_snake_case_id",
      "category": "furniture" | "appliance" | "fixture" | "kitchenware" | "decor" | "plant" | "architectural_element" | "other",
      "sim_role": "scene_shell" | "static_furniture" | "articulated_furniture" | "articulated_appliance" | "static_appliance" | "manipulable_object" | "ignore_for_sim",
      "must_be_separate_asset": true | false,
      "short_description": "5-15 word description",
      "articulation_hint": "description of moving parts (if applicable)",
      "approx_location": "top left" | "middle center" | "bottom right" | etc.,
      "relationships": ["above X", "on Y", "left of Z", ...],
      "grouping_hint": "optional free-text note about what visual parts are included in this object"
    }
  ]
}

## Granularity Rules

- Aim for roughly **10–40 objects total**, not dozens of tiny parts.
- **For articulated furniture/appliances:**
  - Keep as SINGLE complete objects (cabinet with doors = 1 object, not separate frame + doors)
  - Add "articulation_hint" describing movable parts (e.g., "2 hinged doors and 3 drawers")
  - PhysX-Anything will automatically segment and create joints
  - Examples:
    - Upper cabinet with 2 doors → ONE object with articulation_hint="2 hinged doors"
    - Drawer unit with 4 drawers → ONE object with articulation_hint="4 drawers"
    - Refrigerator → ONE object with articulation_hint="main door and freezer drawer"
- **Countertops:**
  - If clearly separate from cabinets → separate object with sim_role="static_furniture"
  - If integrated with closed cabinets → can merge with cabinet block
- **Small parts (NOT separate objects):**
  - Handles, knobs, hinges (part of parent object)
  - Baseboards, toe kicks, trim, backsplash tiles
  - Faucet knobs, small mounting hardware
- **Windows:** Represent as one object (e.g., "window_with_blind")
- **Sets of small items:**
  - If robot might pick individually: create multiple objects (glass_1, glass_2, etc.)
  - If decorative clutter: ONE object (e.g., "glass_set_1") with sim_role="ignore_for_sim"

## Articulation Hints (for PhysX-Anything)

For furniture/appliances with moving parts, add "articulation_hint" field describing the articulation:

**Format examples:**
- "2 hinged doors" (upper cabinet)
- "1 hinged door and 3 drawers" (lower cabinet)
- "4 drawers" (drawer unit)
- "main door and freezer drawer" (refrigerator)
- "oven door" (oven)
- "dishwasher door" (dishwasher)
- "2 wardrobe doors" (wardrobe)

**What to include:**
- Number of moving parts
- Type of motion (hinged, sliding, pull-out)
- Distinguishing features if multiple parts

**Do NOT include:**
- Handles, knobs (not separately articulated)
- Fixed panels or trim
- Internal shelves (unless they slide/articulate)

## Simulation Asset Flag

Set "must_be_separate_asset" based on whether the object needs individual 3D model generation:

**must_be_separate_asset = true (will be removed from scene shell and get individual 3D model):**
- ALL articulated_furniture (cabinets with doors/drawers, dressers, wardrobes, etc.)
- ALL articulated_appliance (refrigerator, oven, dishwasher, microwave with doors)
- ALL static_appliance (stovetop, range hood, built-in appliances)
- ALL manipulable_object (items robots can pick up)

**must_be_separate_asset = false (will remain in static scene shell background):**
- ALL scene_shell (walls, floor, ceiling, baseboards)
- MOST static_furniture (countertops, fixed shelving, solid panels)
- ALL ignore_for_sim (decorative clutter, small trim)

## sim_role Values

Use these sim_role values appropriately:

- **scene_shell**: Architectural elements that form the room
  - Examples: walls, floor, ceiling, baseboards, window frames
  - Always must_be_separate_asset=false (stays in background shell)

- **static_furniture**: Furniture with NO moving parts
  - Examples: solid countertops, fixed shelving, solid panels, open shelves, tables without drawers
  - Usually must_be_separate_asset=false (stays in background shell)

- **articulated_furniture**: Furniture WITH moving parts (doors, drawers, etc.)
  - Examples: cabinets with doors, dressers, wardrobes, desks with drawers
  - ALWAYS must_be_separate_asset=true
  - ALWAYS include articulation_hint describing the moving parts

- **articulated_appliance**: Appliances WITH visible moving parts
  - Examples: refrigerator (with doors), oven (with door), dishwasher, microwave
  - ALWAYS must_be_separate_asset=true
  - ALWAYS include articulation_hint (e.g., "main door", "oven door")

- **static_appliance**: Appliances WITHOUT visible moving parts
  - Examples: stovetop, range hood, built-in microwave (door not visible)
  - Usually must_be_separate_asset=true

- **manipulable_object**: Items robots can pick up, move, or push
  - Examples: kettle, plant, dishes, utensils, jars, bowls, mugs, boxes, tools
  - ALWAYS must_be_separate_asset=true

- **ignore_for_sim**: Decorative elements not needed for simulation
  - Examples: ceiling lights, tiny trim, text labels, small decorative items
  - Always must_be_separate_asset=false (stays in background shell)

## ID Assignment

- Use descriptive lowercase snake_case IDs: "refrigerator_1", "upper_cabinet_left", "plant_pot_1"
- For multiples of the same type, use sequential numbers
- IDs should be stable and meaningful

## Descriptions

- Keep descriptions concise but informative (5-15 words)
- Include material, color, and distinguishing features
- Examples:
  - "White upper cabinet with 2 hinged doors and black handles"
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
- Examples: ["above refrigerator_1", "below upper_cabinet_left", "on countertop"]

## Quality Requirements

- Prefer simulation-level objects over tiny parts
- Be consistent with ids and sim_role values
- Keep descriptions informative but concise
- Aim for 10-40 objects total for typical rooms
- Keep articulated furniture/appliances as complete single objects (let PhysX-Anything handle segmentation)

## Example Output for Kitchen Scene

For a kitchen with upper cabinets (2 doors), lower cabinets (2 doors + 3 drawers), refrigerator, oven, and countertop:

```json
{
  "scene_type": "kitchen",
  "objects": [
    {
      "id": "walls",
      "category": "architectural_element",
      "sim_role": "scene_shell",
      "must_be_separate_asset": false,
      "short_description": "Kitchen walls with white paint",
      "approx_location": "all around",
      "relationships": []
    },
    {
      "id": "floor",
      "category": "architectural_element",
      "sim_role": "scene_shell",
      "must_be_separate_asset": false,
      "short_description": "Light wood laminate flooring",
      "approx_location": "bottom",
      "relationships": []
    },
    {
      "id": "countertop",
      "category": "furniture",
      "sim_role": "static_furniture",
      "must_be_separate_asset": false,
      "short_description": "L-shaped wooden countertop surface",
      "approx_location": "middle center",
      "relationships": ["above lower_cabinet_left"]
    },
    {
      "id": "upper_cabinet_left",
      "category": "furniture",
      "sim_role": "articulated_furniture",
      "must_be_separate_asset": true,
      "short_description": "White upper cabinet with 2 hinged doors and black handles",
      "articulation_hint": "2 hinged doors",
      "approx_location": "top left",
      "relationships": ["above countertop"],
      "grouping_hint": "Complete cabinet unit including frame, doors, and hardware"
    },
    {
      "id": "lower_cabinet_left",
      "category": "furniture",
      "sim_role": "articulated_furniture",
      "must_be_separate_asset": true,
      "short_description": "White lower cabinet with 2 doors and 3 drawers with black handles",
      "articulation_hint": "2 hinged doors and 3 drawers",
      "approx_location": "middle left",
      "relationships": ["below countertop"],
      "grouping_hint": "Complete cabinet unit with frame, doors, drawer fronts, and hardware"
    },
    {
      "id": "refrigerator_1",
      "category": "appliance",
      "sim_role": "articulated_appliance",
      "must_be_separate_asset": true,
      "short_description": "Tall white built-in refrigerator with flat panel doors",
      "articulation_hint": "main door and freezer drawer",
      "approx_location": "middle right",
      "relationships": ["right of upper_cabinet_left"]
    },
    {
      "id": "oven_1",
      "category": "appliance",
      "sim_role": "articulated_appliance",
      "must_be_separate_asset": true,
      "short_description": "Black built-in oven with glass window door",
      "articulation_hint": "oven door",
      "approx_location": "middle center",
      "relationships": ["below countertop"]
    },
    {
      "id": "stovetop_1",
      "category": "appliance",
      "sim_role": "static_appliance",
      "must_be_separate_asset": true,
      "short_description": "Black 4-burner electric stovetop",
      "approx_location": "middle center",
      "relationships": ["on countertop", "above oven_1"]
    },
    {
      "id": "kettle_1",
      "category": "kitchenware",
      "sim_role": "manipulable_object",
      "must_be_separate_asset": true,
      "short_description": "Stainless steel electric kettle",
      "approx_location": "middle center",
      "relationships": ["on countertop"]
    },
    {
      "id": "plant_pot_1",
      "category": "decor",
      "sim_role": "manipulable_object",
      "must_be_separate_asset": true,
      "short_description": "Small potted succulent in white ceramic pot",
      "approx_location": "top right",
      "relationships": ["on upper_cabinet_left"]
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
    articulated_objects = 0

    valid_sim_roles = {
        "scene_shell", "static_furniture", "articulated_furniture",
        "articulated_appliance", "static_appliance", "manipulable_object", "ignore_for_sim"
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

        # Count and validate articulated objects
        if obj.get("sim_role") in ("articulated_furniture", "articulated_appliance"):
            articulated_objects += 1
            if not obj.get("articulation_hint"):
                print(
                    f"[GEMINI-INV] WARNING: Articulated object '{obj.get('id', 'unknown')}' missing articulation_hint",
                    file=sys.stderr
                )

    print(f"[GEMINI-INV] Objects with sim_role field: {objects_with_sim_fields}/{len(objects)}")
    print(f"[GEMINI-INV] Objects marked as separate assets: {separate_assets}/{len(objects)}")
    print(f"[GEMINI-INV] Articulated objects (for PhysX-Anything): {articulated_objects}/{len(objects)}")

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

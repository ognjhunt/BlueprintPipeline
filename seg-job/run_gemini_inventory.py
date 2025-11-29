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
    """Generate the Gemini prompt for scene inventory analysis."""
    return """You are analyzing an interior scene photograph to create a comprehensive object inventory for 3D scene reconstruction.

## Task
Analyze the provided image and identify EVERY distinct object, furniture piece, fixture, and architectural element visible in the scene.

## Output Format
Provide your response as a JSON object with the following structure:

```json
{
  "scene_type": "kitchen" | "bedroom" | "living_room" | "bathroom" | "office" | "other",
  "objects": [
    {
      "id": "unique_identifier",
      "category": "furniture" | "appliance" | "fixture" | "kitchenware" | "decor" | "plant" | "architectural_element" | "other",
      "short_description": "Brief description (5-15 words)",
      "approx_location": "front left" | "top center" | "bottom right" | etc.,
      "relationships": ["above refrigerator_1", "left of window_1", "on counter_1", ...]
    }
  ]
}
```

## Guidelines

### Object Identification
- Identify EVERY visible object, including:
  - Major furniture (cabinets, counters, tables, chairs, shelves)
  - Appliances (refrigerators, ovens, microwaves, dishwashers)
  - Fixtures (lights, faucets, sinks, windows, doors)
  - Smaller items (jars, plants, utensils, decorative objects)
  - Architectural elements (walls, floors, ceilings, windows, doors)
- Each distinct item gets its own entry (e.g., multiple mugs = mug_1, mug_2, etc.)

### ID Assignment
- Use descriptive lowercase IDs: "refrigerator_1", "cabinet_2", "plant_pot_1"
- For multiples of the same type, use sequential numbers
- IDs should be stable and meaningful

### Categories
Use these categories appropriately:
- **furniture**: Cabinets, shelves, counters, tables, chairs, drawers
- **appliance**: Refrigerators, ovens, stoves, dishwashers, kettles, toasters
- **fixture**: Lights, faucets, sinks, windows, doors, blinds, handles
- **kitchenware**: Pots, pans, utensils, dishes, jars, cutting boards
- **decor**: Decorative items, picture frames, vases, canisters
- **plant**: Living plants
- **architectural_element**: Walls, floors, ceilings, backsplashes
- **other**: Anything that doesn't fit the above

### Descriptions
- Keep descriptions concise but informative (5-15 words)
- Include material, color, and distinguishing features
- Examples:
  - "Tall white built-in refrigerator with flat panel doors"
  - "Round black metal faucet with single handle"
  - "Small green succulent in white ceramic pot"

### Approximate Location
Use a simple grid system:
- Horizontal: left, center, right (or "front left", "back right" for depth)
- Vertical: top, middle, bottom
- Examples: "top left", "middle center", "bottom right"

### Relationships
List spatial relationships to other objects using their IDs:
- Position: "above cabinet_1", "below shelf_2", "left of window_1", "right of door_1"
- Containment: "inside cabinet_3", "on counter_1", "in sink_1"
- Support: "supports plant_1", "holds utensils"
- Examples: ["above refrigerator_1", "below cabinet_5", "left of oven_1"]

## Quality Requirements
- **Completeness**: Include every visible object - nothing should be missed
- **Accuracy**: Descriptions must match what's actually visible
- **Consistency**: Use consistent naming and relationship format
- **Granularity**: Individual items get individual entries (not "3 mugs" → use "mug_1", "mug_2", "mug_3")

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

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
from google import genai  # Google GenAI SDK
from google.genai import types


def load_layout(layout_path: Path):
    if not layout_path.is_file():
        raise FileNotFoundError(f"scene_layout file not found: {layout_path}")
    with layout_path.open("r") as f:
        return json.load(f)


def crop_object(image_bgr, bbox2d, polygon=None, margin_frac=0.05):
    """
    image_bgr: H x W x 3 (BGR, as loaded by OpenCV)
    bbox2d: [cx, cy, w, h] in normalized coords [0,1]
    polygon: optional list[[x,y], ...] in normalized coords [0,1] on the FULL image.
    margin_frac: extra margin around bbox (fraction of image size)
    Returns: (cropped BGR image, (x0,y0,x1,y1))
    """
    H, W, _ = image_bgr.shape

    if polygon:
        # Use polygon to define tighter crop and mask out everything else.
        coords = np.array(polygon, dtype=np.float32)
        xs = coords[:, 0] * W
        ys = coords[:, 1] * H

        min_x = xs.min()
        max_x = xs.max()
        min_y = ys.min()
        max_y = ys.max()

        x0 = int(np.clip(min_x - margin_frac * W, 0, W - 1))
        x1 = int(np.clip(max_x + margin_frac * W, 0, W - 1))
        y0 = int(np.clip(min_y - margin_frac * H, 0, H - 1))
        y1 = int(np.clip(max_y + margin_frac * H, 0, H - 1))

        if x1 <= x0 or y1 <= y0:
            return None, (x0, y0, x1, y1)

        crop = image_bgr[y0:y1, x0:x1].copy()

        # Build a mask from the polygon in crop-local coordinates
        poly_local = np.stack(
            [xs - x0, ys - y0],
            axis=-1,
        ).astype(np.int32)

        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly_local], 255)

        # Put object on light gray background (so table/shelf/etc disappear)
        background_val = 240
        crop_masked = np.full_like(crop, background_val, dtype=np.uint8)
        crop_masked[mask == 255] = crop[mask == 255]

        return crop_masked, (x0, y0, x1, y1)

    # Fallback: bbox-based crop as before
    cx_n, cy_n, w_n, h_n = bbox2d

    x_center = cx_n * W
    y_center = cy_n * H
    bw_px = w_n * W
    bh_px = h_n * H

    x0 = int(np.clip(x_center - bw_px / 2 - margin_frac * W, 0, W - 1))
    x1 = int(np.clip(x_center + bw_px / 2 + margin_frac * W, 0, W - 1))
    y0 = int(np.clip(y_center - bh_px / 2 - margin_frac * H, 0, H - 1))
    y1 = int(np.clip(y_center + bh_px / 2 + margin_frac * H, 0, H - 1))

    if x1 <= x0 or y1 <= y0:
        return None, (x0, y0, x1, y1)

    crop = image_bgr[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1, y1)


def summarize_crop_strength(image_shape, bbox2d, crop_box):
    """Return a simple description of how aggressive the crop is.

    Strength is reported as:
      * multiplier vs. the raw bbox area (1.0 = exact bbox, >1 includes margin)
      * percentage of the full image covered by the crop
    """

    H, W = image_shape[:2]
    cx_n, cy_n, w_n, h_n = bbox2d

    bbox_w_px = w_n * W
    bbox_h_px = h_n * H
    bbox_area = max(bbox_w_px * bbox_h_px, 1.0)

    x0, y0, x1, y1 = crop_box
    crop_w = max(float(x1 - x0), 1.0)
    crop_h = max(float(y1 - y0), 1.0)
    crop_area = crop_w * crop_h

    strength = crop_area / bbox_area
    image_fraction = crop_area / float(max(W * H, 1))

    return {
        "bbox_px": [bbox_w_px, bbox_h_px],
        "crop_px": [crop_w, crop_h],
        "crop_box": [x0, y0, x1, y1],
        "strength_vs_bbox": strength,
        "image_fraction": image_fraction,
    }


def create_gemini_client():
    # GEMINI_API_KEY should be set in environment (or pass api_key explicitly)
    client = genai.Client()
    return client


def load_inventory_metadata(seg_dataset_dir: Path) -> Dict[str, Dict]:
    """Load Gemini inventory metadata saved by the segmentation step.

    The inventory.json file lives one directory above the dataset folder
    (seg/inventory.json). We return a mapping from object id -> metadata dict.
    """

    inventory_path = seg_dataset_dir.parent / "inventory.json"
    if not inventory_path.is_file():
        print(f"[MULTIVIEW] No inventory.json found at {inventory_path}; using basic prompts")
        return {}

    try:
        with inventory_path.open("r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[MULTIVIEW] WARNING: failed to read {inventory_path}: {e}", file=sys.stderr)
        return {}

    objects = data.get("objects")
    if not isinstance(objects, list):
        print(f"[MULTIVIEW] WARNING: inventory.json missing 'objects' list")
        return {}

    mapping: Dict[str, Dict] = {}
    for item in objects:
        oid = str(item.get("id") or "").strip()
        if not oid:
            continue
        mapping[oid] = item

    print(f"[MULTIVIEW] Loaded {len(mapping)} object entries from inventory.json")
    return mapping


def build_object_context(object_phrase: str, metadata: Optional[Dict]) -> str:
    """Return a rich, human-readable description block for prompts."""

    lines = []
    if metadata:
        oid = metadata.get("id")
        category = metadata.get("category")
        desc = metadata.get("short_description") or metadata.get("description")
        approx_loc = metadata.get("approx_location") or metadata.get("location")
        relationships = metadata.get("relationships") or []

        if oid:
            lines.append(f"* **ID:** `{oid}`")
        if category:
            lines.append(f"* **Category:** {category}")
        if desc:
            lines.append(f"* **Short description:** {desc}")
        if approx_loc:
            lines.append(f"* **Location in source image:** {approx_loc}")
        if relationships:
            rel_list = ", ".join(map(str, relationships))
            lines.append(f"* **Relationships:** {rel_list}")

    if not lines:
        lines.append(f"* **Object:** {object_phrase or 'object'}")

    return "\n".join(lines)


def extract_image_from_response(response):
    """
    Robustly extract a single image from a Gemini response.
    Prefer candidates[...].content.parts.inline_data, fall back to response.parts.
    """
    # Preferred: candidates[...].content.parts
    for cand in getattr(response, "candidates", []):
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []):
            if getattr(part, "inline_data", None) is not None:
                try:
                    return part.as_image()
                except AttributeError:
                    pass

    # Fallback: response.parts (older style)
    for part in getattr(response, "parts", []):
        if getattr(part, "inline_data", None) is not None:
            try:
                return part.as_image()
            except AttributeError:
                pass

    return None


def infer_object_phrase(client, crop_path: Path, class_name: str) -> str:
    """
    Ask Gemini to describe the *foreground movable object* in the crop as
    a short noun phrase (e.g., 'tan ceramic jar', 'round wooden cutting board').
    This lets us stop trusting raw YOLO class_name like 'table'.
    """
    image = Image.open(str(crop_path)).convert("RGB")

    prompt = (
        "You are labelling objects from indoor scenes for 3D modelling.\n"
        "Look at the reference image and identify the SINGLE foreground movable object "
        "(for example: 'tan ceramic jar', 'round wooden cutting board', "
        "'small potted plant', 'metal cooking pot').\n"
        "Prefer the smaller physical object (jar, pot, utensil, board, plant, lamp, etc.) "
        "over large surfaces such as tables, shelves, countertops, cabinets, walls, or panels.\n"
        "Respond with ONE short noun phrase (3–8 words) naming that object, and nothing else."
    )

    try:
        resp = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt, image],
        )
        phrase = getattr(resp, "text", "") or ""
    except Exception as e:
        print(f"[MULTIVIEW] WARNING: object phrase inference failed: {e}", file=sys.stderr)
        phrase = ""

    phrase = (phrase or "").strip()
    if "\n" in phrase:
        phrase = phrase.splitlines()[0].strip()
    # Strip surrounding quotes if present
    if len(phrase) > 2 and phrase[0] in "\"'" and phrase[-1] == phrase[0]:
        phrase = phrase[1:-1].strip()

    if not phrase:
        phrase = class_name or "object"

    print(f"[MULTIVIEW] Inferred object phrase: {phrase!r} (class_name={class_name!r})")
    return phrase


def generate_views_for_object(
    client,
    crop_path: Path,
    object_phrase: str,
    out_dir: Path,
    views_per_object: int = 4,
    object_metadata: Optional[Dict] = None,
):
    """
    Calls Nano Banana Pro (Gemini 3.0 Pro Image Preview) to generate N views of the object.
    Writes view_0.png ... view_(N-1).png into out_dir.
    Each call requests ONE view only (no grids).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Different view descriptions per index
    view_descriptions = [
        "tight, centered, straight-on front view close to the camera",
        "three-quarter front-left hero angle",
        "three-quarter front-right hero angle",
        "slightly top-down front view",
        "left side view",
        "right side view",
    ]

    image = Image.open(str(crop_path)).convert("RGB")
    context_block = build_object_context(object_phrase, object_metadata)

    front_view_prompt = (
        "You are given a cropped reference image (isolated object on neutral background).\n"
        "Use that crop as a guide to reconstruct the complete object as a standalone asset.\n\n"
        "Object details:\n"
        f"{context_block}\n\n"
        "Reconstruction requirements:\n"
        "1. Shape & proportions\n"
        "   * Infer a plausible full 3D form consistent with the visible silhouette.\n"
        "   * Keep the relative scale suggested by the crop (no exaggerated stretching).\n"
        "2. Material & surface\n"
        "   * Preserve the material type, color palette, and wear visible in the crop.\n"
        "   * Carry over fine surface details (texture, scuffs, glaze/finish).\n"
        "3. Lighting & realism\n"
        "   * Use soft, neutral lighting that reveals form without harsh shadows.\n"
        "4. Camera & framing\n"
        "   * Render ONE centered, straight-on front view (orthographic-style).\n"
        "   * Show the full object with a small margin; do not crop any part.\n"
        "5. Background & isolation\n"
        "   * The object must appear alone on a transparent background (alpha).\n"
        "   * Remove every shelf, table, wall, or prop from the scene.\n"
        "6. Consistency with source image\n"
        "   * Do not invent new accessories or patterns beyond what the crop implies.\n"
        "   * Complete hidden areas plausibly so the final render matches the same object.\n\n"
        "Output: one high-resolution PNG render of the reconstructed object (front view, transparent background)."
    )

    for i in range(views_per_object):
        desc = view_descriptions[i % len(view_descriptions)]

        if i == 0:
            prompt = front_view_prompt
        else:
            prompt = (
                "You are given a cropped reference image (isolated object on neutral background).\n"
                "Recreate only the described object as a standalone render from the specified view angle.\n\n"
                "Object details:\n"
                f"{context_block}\n\n"
                f"View to render: {desc}.\n"
                "Rules:\n"
                "- Keep the full object visible with a slight margin; do not crop.\n"
                "- Match material, color, and texture cues from the crop; do not add new parts.\n"
                "- Remove all background, shelves, floors, and shadows; use a transparent background (alpha).\n"
                "- Output exactly one PNG image (no grids).\n"
            )

        print(f"[MULTIVIEW] Calling Gemini for view {i} of {object_phrase!r} ...")
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(image_size="2K"),
            ),
        )

        gen_img = extract_image_from_response(response)
        if gen_img is None:
            print(
                f"[MULTIVIEW] WARNING: no image part returned for object '{object_phrase}', view {i}",
                file=sys.stderr,
            )
            continue

        out_path = out_dir / f"view_{i}.png"
        gen_img.save(str(out_path))
        print(f"[MULTIVIEW] Saved {out_path}")


def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    layout_prefix = os.getenv("LAYOUT_PREFIX")           # e.g. scenes/<sceneId>/layout
    seg_dataset_prefix = os.getenv("SEG_DATASET_PREFIX") # e.g. scenes/<sceneId>/seg/dataset
    multiview_prefix = os.getenv("MULTIVIEW_PREFIX")     # e.g. scenes/<sceneId>/multiview
    layout_file_name = os.getenv("LAYOUT_FILE_NAME", "scene_layout_scaled.json")
    views_per_object_env = os.getenv("VIEWS_PER_OBJECT", "1")
    enable_gemini_views_env = os.getenv("ENABLE_GEMINI_VIEWS", "true")
    crop_margin_env = os.getenv("CROP_MARGIN_FRAC", "0.05")

    def _parse_bool(val: str) -> bool:
        return val.strip().lower() in {"1", "true", "yes", "on"}

    try:
        views_per_object = int(views_per_object_env)
    except ValueError:
        views_per_object = 1

    enable_gemini_views = _parse_bool(enable_gemini_views_env)
    try:
        crop_margin_frac = float(crop_margin_env)
    except ValueError:
        crop_margin_frac = 0.05

    if not layout_prefix or not seg_dataset_prefix or not multiview_prefix:
        print(
            "[MULTIVIEW] LAYOUT_PREFIX, SEG_DATASET_PREFIX, and MULTIVIEW_PREFIX env vars are required",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path("/mnt/gcs")
    layout_dir = root / layout_prefix
    seg_dataset_dir = root / seg_dataset_prefix
    multiview_root = root / multiview_prefix

    layout_path = layout_dir / layout_file_name
    room_img_path = seg_dataset_dir / "valid" / "images" / "room.jpg"

    print(f"[MULTIVIEW] Bucket: {bucket}")
    print(f"[MULTIVIEW] Scene ID: {scene_id}")
    print(f"[MULTIVIEW] Layout path: {layout_path}")
    print(f"[MULTIVIEW] Room image path: {room_img_path}")
    print(f"[MULTIVIEW] Multiview root: {multiview_root}")
    print(f"[MULTIVIEW] Views per object: {views_per_object}")
    print(f"[MULTIVIEW] Gemini view generation enabled: {enable_gemini_views}")
    print(f"[MULTIVIEW] Crop margin fraction: {crop_margin_frac:.3f}")

    if not layout_path.is_file():
        print(f"[MULTIVIEW] ERROR: layout file not found: {layout_path}", file=sys.stderr)
        sys.exit(1)

    if not room_img_path.is_file():
        print(f"[MULTIVIEW] ERROR: room image not found: {room_img_path}", file=sys.stderr)
        sys.exit(1)

    multiview_root.mkdir(parents=True, exist_ok=True)

    layout = load_layout(layout_path)
    objects = layout.get("objects", [])
    print(f"[MULTIVIEW] Layout has {len(objects)} objects")

    inventory_metadata = load_inventory_metadata(seg_dataset_dir)

    if not objects:
        print("[MULTIVIEW] No objects found in layout; nothing to do.")
        return

    # Load original image once (BGR)
    image_bgr = cv2.imread(str(room_img_path))
    if image_bgr is None:
        print(f"[MULTIVIEW] ERROR: failed to load {room_img_path}", file=sys.stderr)
        sys.exit(1)

    H, W, _ = image_bgr.shape
    print(f"[MULTIVIEW] Room image shape: {H}x{W}")

    # Create Gemini client if view generation is enabled
    client = create_gemini_client() if enable_gemini_views else None

    for obj in objects:
        obj_id = obj.get("id")
        class_name = obj.get("class_name", f"class_{obj.get('class_id', 0)}")
        bbox2d = obj.get("bbox2d")
        polygon = obj.get("polygon")  # normalized coords [N, 2]

        if not bbox2d or len(bbox2d) != 4:
            print(f"[MULTIVIEW] Skipping object {obj_id}: invalid bbox2d", file=sys.stderr)
            continue

        print(f"[MULTIVIEW] Processing object id={obj_id}, class_name={class_name!r}")

        crop_bgr, box = crop_object(
            image_bgr, bbox2d, polygon=polygon, margin_frac=crop_margin_frac
        )
        if crop_bgr is None:
            print(f"[MULTIVIEW] Skipping object {obj_id}: empty crop", file=sys.stderr)
            continue

        obj_dir = multiview_root / f"obj_{obj_id}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        crop_path = obj_dir / "crop.png"
        cv2.imwrite(str(crop_path), crop_bgr)

        strength = summarize_crop_strength(image_bgr.shape, bbox2d, box)
        strength_msg = (
            f"[MULTIVIEW] Crop strength for obj {obj_id}: "
            f"{strength['strength_vs_bbox']:.2f}x bbox area, "
            f"{strength['image_fraction']*100:.2f}% of full image"
        )
        print(strength_msg)

        meta = {
            "object_id": obj_id,
            "class_name": class_name,
            "crop_margin_frac": crop_margin_frac,
            "crop_strength": strength,
            "source_bbox2d": bbox2d,
        }
        meta_path = obj_dir / "crop_meta.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        print(f"[MULTIVIEW] Saved crop to {crop_path}")

        if not enable_gemini_views:
            print(
                "[MULTIVIEW] Gemini view generation disabled; skipping view synthesis.",
                file=sys.stderr,
            )
            continue

        # Step 1: infer a good noun phrase for the foreground movable object
        object_phrase = infer_object_phrase(client, crop_path, class_name)

        # Optional: enrich prompt with Gemini inventory metadata if available
        object_meta = None
        if inventory_metadata:
            object_meta = inventory_metadata.get(str(obj_id)) or inventory_metadata.get(str(obj.get("id")))

        # Step 2: generate isolated single-view renders
        generate_views_for_object(
            client=client,
            crop_path=crop_path,
            object_phrase=object_phrase,
            out_dir=obj_dir,
            views_per_object=views_per_object,
            object_metadata=object_meta,
        )

    print("[MULTIVIEW] Done.")


if __name__ == "__main__":
    main()
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# SAM3 via Hugging Face Transformers
# -----------------------------------------------------------------------------

try:
    # Uses the official SAM3 integration in Hugging Face Transformers.
    from transformers import Sam3Model, Sam3Processor  # type: ignore[import]
except Exception as e:  # pragma: no cover - import-time failure only logged at runtime
    print(
        f"[SAM3] FATAL: could not import Transformers SAM3 classes: {e}",
        file=sys.stderr,
    )
    raise

# -----------------------------------------------------------------------------
# Prompt handling
# -----------------------------------------------------------------------------


def _read_max_prompts() -> int:
    """Maximum number of text prompts to apply per image."""
    raw = os.environ.get("SAM3_MAX_PROMPTS", "24").strip()
    try:
        value = int(raw)
        return max(1, min(value, 64))
    except Exception:
        return 24


def _read_max_masks_per_prompt() -> int:
    """Maximum number of masks to keep per prompt per image.

    Defaults to 1 (highest-scoring mask only). Values are clamped to the
    inclusive range [1, 32].
    """

    raw = os.environ.get("SAM3_MAX_MASKS_PER_PROMPT", "1").strip()
    try:
        value = int(raw)
        return max(1, min(value, 32))
    except Exception:
        return 1


def _read_nms_iou_threshold() -> float:
    """IoU threshold for optional mask-level NMS.

    A value <= 0 disables NMS; otherwise, overlapping masks with IoU greater
    than this threshold will be suppressed.
    """

    raw = os.environ.get("SAM3_MASK_NMS_IOU", "0.0").strip()
    try:
        value = float(raw)
        if value <= 0:
            return 0.0
        return min(value, 1.0)
    except Exception:
        return 0.0


def read_prompt_hints_from_env() -> List[str]:
    """
    Read optional prompt hints from SAM3_PROMPTS to seed Gemini.

    The final list is:
      * lowercased
      * de-duplicated while preserving order
      * truncated to SAM3_MAX_PROMPTS (default 24)
    """
    max_prompts = _read_max_prompts()
    raw = os.environ.get("SAM3_PROMPTS", "").strip()

    tokens: List[str] = []
    if raw:
        # User-provided prompts (comma-separated).
        tokens = [p.strip() for p in raw.split(",") if p.strip()]
        print(f"[SAM3] Using SAM3_PROMPTS from environment ({len(tokens)} raw items) as Gemini hints")

    seen = set()
    prompts: List[str] = []
    for t in tokens:
        name = t.lower()
        if not name or name in seen:
            continue
        seen.add(name)
        prompts.append(name)
        if len(prompts) >= max_prompts:
            break

    print(f"[SAM3] Prompt hints after normalization: {prompts}")
    return prompts


def _gemini_api_key() -> str:
    """
    Look for a Gemini API key in a couple of common env vars.
    We do NOT prompt interactively; Cloud Run / your orchestrator should
    inject this as a secret or env var.
    """
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
        or ""
    )


def _extract_json_blob(text: str) -> str:
    """
    Try to pull out the JSON object from a free-form LLM response.
    If we can't find a {...} block, just return the raw text.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_inventory_objects(raw_objects: Any) -> List[Dict[str, Any]]:
    """Validate and normalize Gemini's inventory list."""

    normalized: List[Dict[str, Any]] = []

    if not isinstance(raw_objects, list):
        return normalized

    seen_ids = set()
    for idx, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            continue

        category = str(item.get("category") or item.get("label") or "").strip()
        if not category:
            continue

        obj_id = str(item.get("id") or f"obj_{idx + 1}").strip()
        if not obj_id:
            obj_id = f"obj_{idx + 1}"
        if obj_id in seen_ids:
            obj_id = f"{obj_id}_{idx + 1}"
        seen_ids.add(obj_id)

        relationships_raw = item.get("relationships")
        relationships: List[str] = []
        if isinstance(relationships_raw, list):
            relationships = [str(r).strip() for r in relationships_raw if str(r).strip()]

        normalized.append(
            {
                "id": obj_id,
                "category": category,
                "short_description": str(item.get("short_description") or item.get("description") or "").strip(),
                "approx_location": str(item.get("approx_location") or item.get("location") or "").strip(),
                "relationships": relationships,
            }
        )

    return normalized


def build_scene_inventory_with_gemini(
    image: Image.Image, prompt_hints: List[str]
) -> List[Dict[str, Any]]:
    """
    Use Gemini to produce a full, structured object inventory for the scene.

    The prompt is intentionally strict: Gemini must enumerate every distinct
    visible object as structured JSON (id, category, short description,
    location, relationships). The returned list is normalized and filtered to
    include only entries with a category so we can pass those categories
    directly to SAM 3 for segmentation.
    """
    api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "No GEMINI_API_KEY / GOOGLE_API_KEY found; cannot build inventory without Gemini"
        )

    # Use Gemini 3 Pro Preview by default. The correct model ID is
    # "gemini-3-pro-preview" for the Gemini API.
    model_id = os.environ.get("GEMINI_MODEL_ID", "gemini-3-pro-preview")

    # Thinking level: "low" or "high". Gemini 3 Pro defaults to high if unset.
    thinking_level_env = os.environ.get("GEMINI_THINKING_LEVEL", "high").strip().lower()
    thinking_level = thinking_level_env if thinking_level_env in ("low", "high") else "high"

    # Optional grounding with Google Search (for more factual / up-to-date inventories).
    enable_search = os.environ.get("GEMINI_ENABLE_SEARCH", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    try:
        # New official SDK: pip install google-genai
        from google import genai  # type: ignore[import]
        from google.genai import types  # type: ignore[import]
    except Exception as e:
        raise RuntimeError(f"Gemini SDK (google-genai) not available: {e}")

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to create Gemini client: {e}")

    # Build GenerateContentConfig with thinking + optional Google Search tool.
    tools = None
    if enable_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]
        print("[SAM3] Gemini Google Search grounding: ENABLED", file=sys.stderr)
    else:
        print("[SAM3] Gemini Google Search grounding: DISABLED", file=sys.stderr)

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        tools=tools,
        response_mime_type="application/json",
    )

    # Craft a strict, machine-parsable instruction to force a complete inventory.
    base_hint = ", ".join(prompt_hints)
    if base_hint:
        hint_line = (
            "Optional candidate objects to double-check (only include if visible): "
            f"{base_hint}.\n"
        )
    else:
        hint_line = ""

    instruction = (
        "You are an expert visual scene annotator for robotics and 3D simulation.\n"
        "You will be given ONE RGB image.\n"
        "Return a complete inventory of every distinct physical object a human can reasonably identify in the image.\n"
        "Requirements:\n"
        "1. Treat each separate physical object as its own item (e.g., two sofas -> sofa_1, sofa_2).\n"
        "2. Do not skip clearly visible small objects (books, vases, remotes, decorative bowls, small boxes).\n"
        "3. Do not use 'etc.' or similar shortcuts; enumerate everything visible.\n"
        "4. For each object include fields: id, category, short_description, approx_location, relationships (list).\n"
        "5. Use lowercase snake_case ids with category + index starting at 1 (sofa_1, armchair_2, round_ottoman_3).\n"
        "6. Use coarse locations such as front left, middle center, back right.\n"
        "7. Only describe what is visible in the single image; do not invent hidden objects.\n"
        f"{hint_line}"
        "Output format (strict JSON only):\n"
        '{"objects": [{"id": "sofa_1", "category": "sofa", "short_description": "...", '
        '"approx_location": "...", "relationships": ["..."]}]}\n'
        "Use valid JSON with double quotes and no trailing commas."
    )

    try:
        print(
            f"[SAM3] Calling Gemini model '{model_id}' to build scene inventory "
            f"(thinking_level={thinking_level}, search_enabled={enable_search})..."
        )
        response = client.models.generate_content(
            model=model_id,
            contents=[image, instruction],
            config=generate_content_config,
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned empty response text")
    except Exception as e:
        raise RuntimeError(f"Gemini generate_content failed: {e}")

    try:
        blob = _extract_json_blob(text)
        data = json.loads(blob)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from Gemini response: {e}")

    objects = _normalize_inventory_objects(data.get("objects"))
    if not objects:
        raise RuntimeError("Gemini produced no usable object inventory")

    print(f"[SAM3] Gemini inventory objects: {len(objects)} detected")
    return objects


# -----------------------------------------------------------------------------
# Image listing + polygon helpers
# -----------------------------------------------------------------------------


def list_images(images_dir: Path) -> List[Path]:
    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    paths: List[Path] = []
    for pattern in exts:
        paths.extend(images_dir.glob(pattern))
    return sorted(set(paths))


def mask_to_polygons(
    mask: np.ndarray, min_area_px: int = 150, min_area_pct: float = 0.0005
) -> List[np.ndarray]:
    """Convert a binary mask to polygons with dynamic area filtering.

    The minimum area is the larger of a fixed pixel floor (``min_area_px``)
    and a percentage of the image area (``min_area_pct``).
    """

    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = mask_u8.shape[:2]
    image_area = img_h * img_w
    dynamic_min_area = int(math.ceil(min_area_pct * image_area)) if image_area else 0
    min_area = max(min_area_px, dynamic_min_area)

    polys: List[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        polys.append(c.reshape(-1, 2))
    return polys


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""

    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def filter_masks_with_scores(
    masks: torch.Tensor,
    scores: Optional[torch.Tensor],
    max_masks: int,
    nms_iou_threshold: float,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """Select the top masks per prompt with optional NMS suppression."""

    mask_np = masks.detach().cpu().numpy()
    if scores is not None:
        scores_np = scores.detach().cpu().numpy().reshape(-1)
    else:
        scores_np = np.ones((mask_np.shape[0],), dtype=np.float32)

    if scores_np.shape[0] != mask_np.shape[0]:
        # Fallback: align lengths defensively.
        min_len = min(scores_np.shape[0], mask_np.shape[0])
        scores_np = scores_np[:min_len]
        mask_np = mask_np[:min_len]

    order = np.argsort(-scores_np)

    kept_masks: List[np.ndarray] = []
    dropped_nms = 0
    dropped_overflow = 0

    for idx in order:
        mask = mask_np[idx]
        mask_bin = (mask > 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask

        if nms_iou_threshold > 0:
            overlaps = any(_mask_iou(mask_bin, km) > nms_iou_threshold for km in kept_masks)
            if overlaps:
                dropped_nms += 1
                continue

        if len(kept_masks) >= max_masks:
            dropped_overflow += 1
            continue

        kept_masks.append(mask_bin)

    stats = {
        "total": int(mask_np.shape[0]),
        "kept": len(kept_masks),
        "dropped_nms": dropped_nms,
        "dropped_overflow": dropped_overflow,
    }
    return kept_masks, stats


def polygon_to_yolo_line(class_id: int, poly: np.ndarray, width: int, height: int) -> str:
    xs = np.clip(poly[:, 0], 0, width - 1)
    ys = np.clip(poly[:, 1], 0, height - 1)
    xs_n = xs / float(width)
    ys_n = ys / float(height)
    coords: List[str] = []
    for x, y in zip(xs_n.tolist(), ys_n.tolist()):
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return " ".join([str(class_id)] + coords)


def write_labels(
    label_path: Path,
    detections: List[Tuple[int, List[np.ndarray]]],
    img_w: int,
    img_h: int,
) -> None:
    lines: List[str] = []
    for class_id, polys in detections:
        for poly in polys:
            if poly.shape[0] < 3:
                continue
            line = polygon_to_yolo_line(class_id, poly, img_w, img_h)
            lines.append(line)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        f.write("\n".join(lines))


def copy_images(image_paths: Iterable[Path], dest: Path) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for src in image_paths:
        dst = dest / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def export_room_alias(image_path: Path, label_path: Path, images_dir: Path, labels_dir: Path) -> None:
    room_img = images_dir / "room.jpg"
    room_label = labels_dir / "room.txt"
    try:
        image = Image.open(image_path).convert("RGB")
        image.save(room_img)
        shutil.copy2(label_path, room_label)
        print("[SAM3] Exported canonical room.jpg and room.txt")
    except Exception as e:
        print(f"[SAM3] WARNING: failed to export room alias: {e}", file=sys.stderr)


def build_sam3_model_and_processor():
    """
    Build SAM3 image model + processor via Hugging Face Transformers.

    This uses the gated `facebook/sam3` checkpoint, so you must:
      * Have been granted access to the model on Hugging Face
      * Provide a valid token in one of:
          - HUGGINGFACE_HUB_TOKEN
          - HF_TOKEN
          - HUGGINGFACE_TOKEN
    """
    model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")

    # Respect HF token env vars; Transformers + huggingface_hub will pick them up.
    token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    print(f"[SAM3] Loading SAM3 model from Hugging Face repo: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM3] Using device: {device}")

    # Follow the official Transformers example:
    #   model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    #   processor = Sam3Processor.from_pretrained("facebook/sam3")
    # which uses the default checkpoint dtype (often bfloat16 on GPU).
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device)
    model.eval()

    print("[SAM3] SAM3 model and processor loaded.")
    return model, processor, device


def main() -> None:
    images_prefix = os.environ.get("IMAGES_PREFIX")
    seg_prefix = os.environ.get("SEG_PREFIX")
    bucket = os.environ.get("BUCKET", "")
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    if hf_token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    if not images_prefix or not seg_prefix:
        print(
            "[SAM3] IMAGES_PREFIX and SEG_PREFIX env vars are required",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path("/mnt/gcs")
    images_dir = root / images_prefix
    out_dir = root / seg_prefix
    dataset_dir = out_dir / "dataset" / "valid"
    images_out_dir = dataset_dir / "images"
    labels_out_dir = dataset_dir / "labels"

    if images_dir.is_file():
        images_dir = images_dir.parent
        print(
            f"[SAM3] IMAGES_PREFIX pointed to a file; using parent directory: {images_dir}",
            file=sys.stderr,
        )

    print(f"[SAM3] Bucket: {bucket}")
    print(f"[SAM3] Images dir: {images_dir}")
    print(f"[SAM3] Output dir: {out_dir}")
    print(f"[SAM3] Dataset dir: {dataset_dir}")

    if not images_dir.is_dir():
        print(
            f"[SAM3] ERROR: images directory does not exist: {images_dir}",
            file=sys.stderr,
        )
        # Helpful debug listing
        try:
            for p in sorted(root.glob("**/*")):
                print(f"[SAM3] FS: {p}")
        except Exception as e:
            print(f"[SAM3] WARNING: failed to walk /mnt/gcs: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[SAM3] Listing images in {images_dir}")
    image_paths = list_images(images_dir)
    print(f"[SAM3] Found {len(image_paths)} image(s)")

    if image_paths:
        sample_names = [p.name for p in image_paths[:10]]
        print(f"[SAM3] Sample image names: {sample_names}")

    if not image_paths:
        print(f"[SAM3] ERROR: no .png/.jpg/.jpeg images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    copied_images = copy_images(image_paths, images_out_dir)
    print(f"[SAM3] Copied {len(copied_images)} images into {images_out_dir}")

    # ------------------------------------------------------------------
    # Build prompts from a Gemini-generated object inventory.
    # Segmentation will abort if Gemini fails.
    # ------------------------------------------------------------------
    max_prompts = _read_max_prompts()
    prompt_hints = read_prompt_hints_from_env()

    if not copied_images:
        print("[SAM3] ERROR: no images available to seed Gemini prompt generation", file=sys.stderr)
        sys.exit(1)

    try:
        # Use the first image as a representative scene for dynamic prompts.
        scene_image = Image.open(copied_images[0]).convert("RGB")
        inventory_objects = build_scene_inventory_with_gemini(scene_image, prompt_hints)
    except Exception as e:
        print(
            f"[SAM3] ERROR: Gemini-based inventory generation failed: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not inventory_objects:
        print("[SAM3] ERROR: Gemini returned an empty object inventory", file=sys.stderr)
        sys.exit(1)

    # Build prompts either per-object (default) or category-only (legacy mode).
    use_category_only = os.environ.get("SAM3_CATEGORY_ONLY_PROMPTS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    prompt_mode = "category" if use_category_only else "object"
    class_names: List[str] = []
    prompt_texts: List[str] = []

    if use_category_only:
        seen_categories = set()
        for obj in inventory_objects:
            name = str(obj.get("category") or "").strip().lower()
            if not name or name in seen_categories:
                continue
            seen_categories.add(name)
            class_names.append(name)
            prompt_texts.append(name)
            if len(class_names) >= max_prompts:
                print(
                    f"[SAM3] Reached SAM3_MAX_PROMPTS limit ({max_prompts}); ignoring remaining categories",
                    file=sys.stderr,
                )
                break
    else:
        for obj in inventory_objects:
            obj_id = str(obj.get("id") or "").strip()
            category = str(obj.get("category") or "").strip().lower()
            description = str(obj.get("short_description") or "").strip()
            location = str(obj.get("approx_location") or "").strip()

            if not obj_id or not category:
                continue

            prompt_parts = [f"object id: {obj_id}", f"category: {category}"]
            if description:
                prompt_parts.append(f"description: {description}")
            if location:
                prompt_parts.append(f"approximate location: {location}")

            prompt_text = ", ".join(prompt_parts)
            class_names.append(obj_id)
            prompt_texts.append(prompt_text)
            if len(class_names) >= max_prompts:
                print(
                    f"[SAM3] Reached SAM3_MAX_PROMPTS limit ({max_prompts}); ignoring remaining objects",
                    file=sys.stderr,
                )
                break

    if not prompt_texts or not class_names:
        print(
            "[SAM3] ERROR: Gemini inventory produced no usable prompts/classes",
            file=sys.stderr,
        )
        sys.exit(1)

    inventory_path = out_dir / "inventory.json"
    id_to_class_index = {name: idx for idx, name in enumerate(class_names)}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with inventory_path.open("w") as f:
            json.dump(
                {
                    "source_image": copied_images[0].name,
                    "objects": inventory_objects,
                    "prompt_mode": prompt_mode,
                    "class_names": class_names,
                    "prompts": prompt_texts,
                    "id_to_class_index": id_to_class_index,
                },
                f,
                indent=2,
            )
        print(f"[SAM3] Saved Gemini inventory to {inventory_path}")
    except Exception as e:
        print(f"[SAM3] WARNING: failed to save inventory JSON: {e}", file=sys.stderr)

    print(f"[SAM3] Prompt mode: {prompt_mode}")
    print(f"[SAM3] Final prompts for this run ({len(prompt_texts)}): {prompt_texts}")
    class_to_id = id_to_class_index

    print("[SAM3] Building SAM3 model (Transformers)...")
    model, processor, device = build_sam3_model_and_processor()

    # Whatever dtype the checkpoint uses (float32 or bfloat16), we will
    # cast our inputs to match, to avoid "FloatTensor vs CUDABFloat16Type"
    # runtime errors.
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    print(f"[SAM3] Model dtype: {model_dtype}")

    # Inspect the text encoder's maximum supported sequence length so we can
    # truncate prompts before they hit CLIP's positional embedding limit.
    max_text_positions: Optional[int] = None
    try:
        text_encoder = getattr(model, "text_encoder", None)
        if text_encoder is not None and hasattr(text_encoder, "config"):
            max_text_positions = getattr(text_encoder.config, "max_position_embeddings", None)
        if isinstance(max_text_positions, int) and max_text_positions > 0:
            print(f"[SAM3] Text encoder max_position_embeddings={max_text_positions}")
        else:
            max_text_positions = None
            print(
                "[SAM3] WARNING: could not determine text encoder max_position_embeddings; "
                "text prompts will not be truncated automatically."
            )
    except Exception as e:
        max_text_positions = None
        print(
            f"[SAM3] WARNING: failed to inspect text encoder max_position_embeddings: {e}",
            file=sys.stderr,
        )

    conf_thresh = float(os.environ.get("SAM3_CONFIDENCE", "0.30"))
    print(f"[SAM3] Confidence threshold set to {conf_thresh}")

    max_masks_per_prompt = _read_max_masks_per_prompt()
    nms_iou_threshold = _read_nms_iou_threshold()
    print(
        "[SAM3] Mask filtering:",
        f"max_masks_per_prompt={max_masks_per_prompt}",
        f"nms_iou_threshold={nms_iou_threshold}",
    )

    data_yaml = {
        "path": str(out_dir / "dataset"),
        "train": "valid/images",
        "val": "valid/images",
        "test": "valid/images",
        "nc": len(class_names),
        "names": class_names,
    }
    (out_dir / "dataset").mkdir(parents=True, exist_ok=True)
    with (out_dir / "dataset" / "data.yaml").open("w") as f:
        f.write(json.dumps(data_yaml, indent=2))
    print(f"[SAM3] Wrote data.yaml to {out_dir / 'dataset' / 'data.yaml'}")

    for img_path in copied_images:
        print(f"[SAM3] Processing {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        detections: List[Tuple[int, List[np.ndarray]]] = []

        for class_name, prompt in zip(class_names, prompt_texts):
            # Run SAM3 for a single text prompt on this image.
            inputs = processor(images=image, text=prompt, return_tensors="pt")

            # Truncate text token sequences so they respect the CLIP text encoder's
            # max_position_embeddings. Without this, long natural-language prompts
            # can produce sequences (e.g. length 34) that exceed the 32-position
            # embedding limit and trigger:
            #   ValueError: Sequence length must be less than max_position_embeddings
            if max_text_positions is not None and "input_ids" in inputs:
                input_ids = inputs["input_ids"]
                if hasattr(input_ids, "shape") and input_ids.shape[-1] > max_text_positions:
                    orig_len = int(input_ids.shape[-1])
                    inputs["input_ids"] = input_ids[..., :max_text_positions]
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = inputs["attention_mask"][..., :max_text_positions]
                    if "position_ids" in inputs:
                        inputs["position_ids"] = inputs["position_ids"][..., :max_text_positions]
                    print(
                        f"[SAM3] Truncated text tokens for class '{class_name}' "
                        f"from length {orig_len} to {max_text_positions}"
                    )

            inputs = inputs.to(device=device, dtype=model_dtype)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process to instance masks at the original image resolution.
            results_list = processor.post_process_instance_segmentation(
                outputs,
                threshold=conf_thresh,
                mask_threshold=conf_thresh,
                target_sizes=inputs.get("original_sizes").tolist(),
            )
            if not results_list:
                continue

            results = results_list[0]
            masks = results.get("masks")
            scores = results.get("scores")

            if masks is None or masks.shape[0] == 0:
                continue

            kept_masks, mask_stats = filter_masks_with_scores(
                masks=masks,
                scores=scores,
                max_masks=max_masks_per_prompt,
                nms_iou_threshold=nms_iou_threshold,
            )
            dropped_other = mask_stats["total"] - mask_stats["kept"] - mask_stats["dropped_nms"] - mask_stats["dropped_overflow"]
            print(
                f"[SAM3] Prompt '{prompt}': kept {mask_stats['kept']} / {mask_stats['total']}"
                f" masks (dropped_nms={mask_stats['dropped_nms']},"
                f" dropped_overflow={mask_stats['dropped_overflow']},"
                f" dropped_other={dropped_other})"
            )

            for mask_bin in kept_masks:
                polys = mask_to_polygons(mask_bin)
                if polys:
                    detections.append((class_to_id[class_name], polys))

        if not detections:
            print(f"[SAM3] WARNING: no detections for {img_path.name}", file=sys.stderr)

        label_path = labels_out_dir / f"{img_path.stem}.txt"
        write_labels(label_path, detections, img_w=img_w, img_h=img_h)
        print(f"[SAM3] Wrote labels to {label_path}")

    # Export canonical room.jpg / room.txt alias using the first image, if labels exist.
    first_image = copied_images[0]
    first_label = labels_out_dir / f"{first_image.stem}.txt"
    if first_label.is_file():
        export_room_alias(first_image, first_label, images_out_dir, labels_out_dir)

    for p in sorted(out_dir.rglob("*")):
        rel = p.relative_to(out_dir)
        print(f"[SAM3] OUT: {rel}")

    print("[SAM3] Completed segmentation with SAM 3.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover - top-level safety net
        import traceback

        print(f"[SAM3] FATAL: unhandled exception: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

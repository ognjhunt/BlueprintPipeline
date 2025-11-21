import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

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

# A general-purpose fallback vocabulary that works reasonably across
# living rooms, bedrooms, offices, and warehouses. This is only used
# if no SAM3_PROMPTS env var is provided and Gemini is unavailable.
DEFAULT_BASE_PROMPTS: List[str] = [
    # Room / structural surfaces
    "wall",
    "floor",
    "ceiling",
    "window",
    "door",
    # Seating & tables
    "sofa",
    "armchair",
    "chair",
    "office chair",
    "round ottoman",
    "stool",
    "bench",
    "coffee table",
    "side table",
    "desk",
    "dining table",
    "workbench",
    # Storage & surfaces
    "cabinet",
    "wardrobe",
    "bookshelf",
    "shelving unit",
    "sideboard cabinet",
    "countertop",
    "kitchen island",
    # Sleeping
    "bed",
    "nightstand",
    "dresser",
    # Decor & fixtures
    "lamp",
    "floor lamp",
    "table lamp",
    "rug",
    "carpet",
    "pillow",
    "blanket",
    "mirror",
    "wall art",
    "plant",
    # Warehouse / industrial
    "box",
    "crate",
    "pallet",
    "pallet rack",
    "warehouse rack",
    "forklift",
]


def _read_max_prompts() -> int:
    """Maximum number of text prompts to apply per image."""
    raw = os.environ.get("SAM3_MAX_PROMPTS", "24").strip()
    try:
        value = int(raw)
        return max(1, min(value, 64))
    except Exception:
        return 24


def read_base_prompts_from_env() -> List[str]:
    """
    Read the base prompt vocabulary from SAM3_PROMPTS, or fall back to a
    generic set that works across many scene types.

    The final list is:
      * lowercased
      * de-duplicated while preserving order
      * truncated to SAM3_MAX_PROMPTS (default 24)
    """
    max_prompts = _read_max_prompts()
    raw = os.environ.get("SAM3_PROMPTS", "").strip()

    if raw:
        # User-provided prompts (comma-separated).
        tokens = [p.strip() for p in raw.split(",") if p.strip()]
        print(f"[SAM3] Using SAM3_PROMPTS from environment ({len(tokens)} raw items)")
    else:
        tokens = list(DEFAULT_BASE_PROMPTS)
        print(f"[SAM3] Using built-in default prompts ({len(tokens)} candidates)")

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

    print(f"[SAM3] Base prompts after normalization: {prompts}")
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


def build_scene_prompts_with_gemini(
    image: Image.Image, base_prompts: List[str]
) -> List[str]:
    """
    Optional step: use Gemini to analyze the scene and propose a small set of
    segmentation categories that fit THIS particular image (room, warehouse,
    office, bedroom, etc.).

    If anything goes wrong (no key, import error, bad JSON, etc.), we just
    fall back to base_prompts.
    """
    api_key = _gemini_api_key()
    if not api_key:
        print(
            "[SAM3] No GEMINI_API_KEY / GOOGLE_API_KEY found; "
            "skipping Gemini-based prompt generation.",
            file=sys.stderr,
        )
        return base_prompts

    max_prompts = _read_max_prompts()
    model_id = os.environ.get("GEMINI_MODEL_ID", "gemini-3.0-pro")

    try:
        # New official SDK: pip install google-genai
        from google import genai  # type: ignore[import]
    except Exception as e:
        print(
            f"[SAM3] Gemini SDK (google-genai) not available: {e}; "
            "falling back to base prompts.",
            file=sys.stderr,
        )
        return base_prompts

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(
            f"[SAM3] Failed to create Gemini client: {e}; "
            "falling back to base prompts.",
            file=sys.stderr,
        )
        return base_prompts

    # Craft an instruction that keeps the response machine-parseable.
    base_hint = ", ".join(base_prompts)
    instruction = (
        "You are helping choose text labels for an open-vocabulary segmentation model.\n"
        "You will be shown exactly one RGB image of a scene (for example a living room, "
        "bedroom, office, meeting room, warehouse, factory floor, corridor, or lobby).\n\n"
        "Tasks:\n"
        "1. Look at all visible objects and large surfaces.\n"
        "2. Choose at most 25 distinct object / surface CATEGORY NAMES that would be "
        "useful to segment (for example: sofa, armchair, round ottoman, coffee table, "
        "office chair, desk, monitor, forklift, pallet rack, crate, box, wall, floor, ceiling).\n"
        "3. Focus on large / important items (furniture, machinery, doors, windows, "
        "walls, floors, ceiling, shelves, pallets, vehicles, big boxes). Ignore tiny "
        "decorations unless they are central to the scene.\n"
        "4. Prefer categories from this candidate list when they match the image: "
        f"{base_hint}. You may also add new categories if needed.\n"
        "5. Return ONLY a single JSON object with this exact shape and nothing else:\n"
        '{\"categories\": [\"label1\", \"label2\", ...]}\n\n'
        "Rules for labels:\n"
        "- Use lowercase English.\n"
        "- Use singular nouns or very short noun phrases (e.g. \"sofa\", \"coffee table\").\n"
        "- Do not include duplicates.\n"
        "- Do not include free-form explanations, Markdown, or extra keys."
    )

    try:
        print(f"[SAM3] Calling Gemini model '{model_id}' to build scene prompts...")
        response = client.models.generate_content(
            model=model_id,
            contents=[image, instruction],
        )
        text = getattr(response, "text", None)
        if not text:
            print(
                "[SAM3] Gemini returned empty response text; "
                "falling back to base prompts.",
                file=sys.stderr,
            )
            return base_prompts
    except Exception as e:
        print(
            f"[SAM3] Gemini generate_content failed: {e}; "
            "falling back to base prompts.",
            file=sys.stderr,
        )
        return base_prompts

    try:
        blob = _extract_json_blob(text)
        data = json.loads(blob)
        raw_categories = data.get("categories") or data.get("objects") or data.get("labels") or []
    except Exception as e:
        print(
            f"[SAM3] Failed to parse JSON from Gemini response: {e}; "
            "falling back to base prompts.",
            file=sys.stderr,
        )
        return base_prompts

    seen = set()
    prompts: List[str] = []
    for item in raw_categories:
        name = str(item).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        prompts.append(name)
        if len(prompts) >= max_prompts:
            break

    if not prompts:
        print(
            "[SAM3] Gemini produced no usable categories; using base prompts instead.",
            file=sys.stderr,
        )
        return base_prompts

    print(f"[SAM3] Gemini scene prompts: {prompts}")
    return prompts


# -----------------------------------------------------------------------------
# Image listing + polygon helpers
# -----------------------------------------------------------------------------


def list_images(images_dir: Path) -> List[Path]:
    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    paths: List[Path] = []
    for pattern in exts:
        paths.extend(images_dir.glob(pattern))
    return sorted(set(paths))


def mask_to_polygons(mask: np.ndarray, min_area: int = 25) -> List[np.ndarray]:
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        polys.append(c.reshape(-1, 2))
    return polys


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
    # Build prompts: env / defaults, optionally refined by Gemini.
    # ------------------------------------------------------------------
    base_prompts = read_base_prompts_from_env()
    prompts = base_prompts

    if copied_images:
        try:
            # Use the first image as a representative scene for dynamic prompts.
            scene_image = Image.open(copied_images[0]).convert("RGB")
            prompts = build_scene_prompts_with_gemini(scene_image, base_prompts)
        except Exception as e:
            print(
                f"[SAM3] WARNING: Gemini-based prompt generation failed unexpectedly: {e}; "
                "using base prompts.",
                file=sys.stderr,
            )
            prompts = base_prompts

    print(f"[SAM3] Final prompts for this run ({len(prompts)}): {prompts}")
    class_to_id = {c: i for i, c in enumerate(prompts)}

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

    conf_thresh = float(os.environ.get("SAM3_CONFIDENCE", "0.15"))
    print(f"[SAM3] Confidence threshold set to {conf_thresh}")

    data_yaml = {
        "path": str(out_dir / "dataset"),
        "train": "valid/images",
        "val": "valid/images",
        "test": "valid/images",
        "nc": len(prompts),
        "names": prompts,
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

        for prompt in prompts:
            # Run SAM3 for a single text prompt on this image.
            inputs = processor(images=image, text=prompt, return_tensors="pt")
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

            if masks is None or masks.shape[0] == 0:
                continue

            masks_np = masks.detach().cpu().numpy()
            for mask in masks_np:
                # Ensure binary mask
                if mask.dtype != np.uint8:
                    mask_bin = (mask > 0.5).astype(np.uint8)
                else:
                    mask_bin = mask
                polys = mask_to_polygons(mask_bin)
                if polys:
                    detections.append((class_to_id[prompt], polys))

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

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


def parse_prompts() -> List[str]:
    raw = os.environ.get("SAM3_PROMPTS", "").strip()
    if raw:
        prompts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        prompts = [
            "object",
            "furniture",
            "chair",
            "table",
            "sofa",
            "couch",
            "bed",
            "cabinet",
            "drawer",
            "door",
            "window",
            "appliance",
            "lamp",
            "plant",
            "robot",
            "person",
            "floor",
            "ceiling",
            "wall",
        ]
    return prompts


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
    # which uses the default (float32) dtype and avoids bf16/float32 mismatch.
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

    prompts = parse_prompts()
    print(f"[SAM3] Using prompts: {prompts}")
    class_to_id = {c: i for i, c in enumerate(prompts)}

    print("[SAM3] Building SAM3 model (Transformers)...")
    model, processor, device = build_sam3_model_and_processor()

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
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

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

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

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


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

    print(f"[SAM3] Bucket: {bucket}")
    print(f"[SAM3] Images dir: {images_dir}")
    print(f"[SAM3] Output dir: {out_dir}")
    print(f"[SAM3] Dataset dir: {dataset_dir}")

    if not images_dir.is_dir():
        print(
            f"[SAM3] ERROR: images directory does not exist: {images_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    image_paths = list_images(images_dir)
    if not image_paths:
        print(f"[SAM3] ERROR: no .png/.jpg/.jpeg images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    copied_images = copy_images(image_paths, images_out_dir)
    print(f"[SAM3] Copied {len(copied_images)} images into {images_out_dir}")

    prompts = parse_prompts()
    print(f"[SAM3] Using prompts: {prompts}")
    class_to_id = {c: i for i, c in enumerate(prompts)}

    model = build_sam3_image_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    processor = Sam3Processor(model, device=device)
    conf_thresh = float(os.environ.get("SAM3_CONFIDENCE", "0.15"))
    processor.set_confidence_threshold(conf_thresh)

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

    for img_path in copied_images:
        print(f"[SAM3] Processing {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        state = processor.set_image(image)

        detections: List[Tuple[int, List[np.ndarray]]] = []
        for prompt in prompts:
            state = processor.set_text_prompt(prompt, state)
            masks = state.get("masks")
            boxes = state.get("boxes")
            scores = state.get("scores")

            if masks is None or boxes is None or scores is None:
                continue

            masks_np = masks.squeeze(1).detach().cpu().numpy()
            for mask in masks_np:
                polys = mask_to_polygons(mask)
                detections.append((class_to_id[prompt], polys))

            processor.reset_all_prompts(state)

        if not detections:
            print(f"[SAM3] WARNING: no detections for {img_path.name}", file=sys.stderr)

        label_path = labels_out_dir / f"{img_path.stem}.txt"
        write_labels(label_path, detections, img_w=image.width, img_h=image.height)
        print(f"[SAM3] Wrote labels to {label_path}")

    first_image = copied_images[0]
    first_label = labels_out_dir / f"{first_image.stem}.txt"
    if first_label.is_file():
        export_room_alias(first_image, first_label, images_out_dir, labels_out_dir)

    for p in sorted(out_dir.rglob("*")):
        rel = p.relative_to(out_dir)
        print(f"[SAM3] OUT: {rel}")

    print("[SAM3] Completed segmentation with SAM 3.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SAM3 text-prompted segmentation for 3D-RE-GEN pipeline.

Replaces Step 1 (GroundingDINO + SAM1) with SAM3's unified text-prompted
segmentation. Produces the same output structure that Steps 2-7 expect.

Runs on the remote GPU VM in the venv_sam3 (Python 3.12) environment.

Usage:
    python segmentation_sam3.py \
        --image /path/to/input.jpg \
        --output /path/to/output_dir \
        --labels "chair,table,sofa,floor" \
        --threshold 0.4 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="[SAM3-SEG] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM3 segmentation
# ---------------------------------------------------------------------------


def load_sam3(
    model_id: str = "facebook/sam3",
    device: str = "cuda:0",
) -> Tuple[Any, Any]:
    """Load SAM3 model and processor from HuggingFace."""
    from transformers import Sam3Model, Sam3Processor

    log.info("Loading SAM3 model: %s", model_id)
    t0 = time.monotonic()
    # `facebook/sam3` is a gated repo. If we don't have a HF token, avoid
    # network calls and rely on the local HF cache.
    has_token = any(
        os.getenv(k) for k in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    )
    local_only = not has_token
    try:
        processor = Sam3Processor.from_pretrained(model_id, local_files_only=local_only)
        model = Sam3Model.from_pretrained(model_id, local_files_only=local_only).to(device)
    except Exception as exc:
        if local_only:
            raise RuntimeError(
                "SAM3 model is gated and no HF token is configured. "
                "Either set HF_TOKEN (or HF_HUB_TOKEN/HUGGINGFACE_HUB_TOKEN) "
                "or ensure the model is already cached on disk."
            ) from exc
        raise
    model.eval()
    log.info("SAM3 loaded in %.1fs", time.monotonic() - t0)
    return model, processor


def segment_with_sam3(
    image: Image.Image,
    labels: List[str],
    model: Any,
    processor: Any,
    device: str = "cuda:0",
    threshold: float = 0.4,
    mask_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Run SAM3 text-prompted segmentation for each label.

    Returns a list of detections, each with:
        label: str, instance_idx: int, mask: np.ndarray (H, W bool),
        bbox: [x_min, y_min, x_max, y_max], score: float
    """
    detections: List[Dict[str, Any]] = []
    w, h = image.size

    # Prepare image inputs once. Depending on the transformers/SAM3 version,
    # the model forward may accept either `pixel_values` or precomputed
    # `vision_embeds`. Prefer vision embeddings (faster), but fall back to
    # pixel_values for compatibility.
    log.info("Computing vision embeddings (%dx%d)...", w, h)
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = img_inputs.get("pixel_values")
    vision_embeds = None
    if pixel_values is not None and hasattr(model, "get_vision_features"):
        try:
            with torch.no_grad():
                vision_features = model.get_vision_features(pixel_values=pixel_values)
            # Common shapes across versions: dict with key, or object attribute.
            if isinstance(vision_features, dict) and "vision_embeds" in vision_features:
                vision_embeds = vision_features["vision_embeds"]
            elif hasattr(vision_features, "vision_embeds"):
                vision_embeds = vision_features.vision_embeds
            elif isinstance(vision_features, torch.Tensor):
                # Some versions may return the tensor directly.
                vision_embeds = vision_features
            else:
                # Unknown return shape; fall back to pixel_values path.
                vision_embeds = None
        except Exception as exc:
            log.warning("Failed to precompute vision embeddings; falling back to pixel_values (%s)", exc)
            vision_embeds = None

    for label in labels:
        log.info("Segmenting label: %s", label)
        text_inputs = processor(text=label, return_tensors="pt").to(device)

        with torch.no_grad():
            if vision_embeds is not None:
                outputs = model(vision_embeds=vision_embeds, **text_inputs)
            elif pixel_values is not None:
                outputs = model(pixel_values=pixel_values, **text_inputs)
            else:
                raise RuntimeError("SAM3 processor did not produce pixel_values for the input image")

        # Post-process to get instance masks
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[(h, w)],
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])

        instance_idx = 0
        for mask_tensor, score in zip(masks, scores):
            score_val = float(score)
            if score_val < threshold:
                continue

            mask_np = mask_tensor.cpu().numpy().astype(bool)

            # Compute bounding box from mask
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            if not rows.any() or not cols.any():
                continue
            y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
            x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

            detections.append({
                "label": label,
                "instance_idx": instance_idx,
                "mask": mask_np,
                "bbox": [x_min, y_min, x_max, y_max],
                "score": score_val,
            })
            instance_idx += 1
            log.info(
                "  Found %s_%d (score=%.3f, bbox=[%d,%d,%d,%d])",
                label, instance_idx - 1, score_val,
                x_min, y_min, x_max, y_max,
            )

    log.info("Total detections: %d across %d labels", len(detections), len(labels))
    return detections


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------


def estimate_depth(
    image: Image.Image,
    output_path: Path,
    device: str = "cuda:0",
) -> None:
    """Run monocular depth estimation using DepthAnythingV2."""
    from transformers import pipeline

    log.info("Running depth estimation...")
    t0 = time.monotonic()

    depth_pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
    )
    result = depth_pipe(image)
    depth_map = result["depth"]  # PIL Image

    # Save as 16-bit PNG for precision
    depth_np = np.array(depth_map)
    if depth_np.dtype == np.float32 or depth_np.dtype == np.float64:
        # Normalize to 16-bit range
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_np = ((depth_np - d_min) / (d_max - d_min) * 65535).astype(
                np.uint16
            )
        else:
            depth_np = np.zeros_like(depth_np, dtype=np.uint16)
    elif depth_np.dtype == np.uint8:
        depth_np = depth_np.astype(np.uint16) * 257  # Scale 0-255 to 0-65535

    depth_img = Image.fromarray(depth_np, mode="I;16")
    depth_img.save(str(output_path))
    log.info("Depth saved to %s (%.1fs)", output_path, time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_outputs(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    output_dir: Path,
    padding: int = 5,
) -> Dict[str, Any]:
    """Write segmentation outputs in 3D-RE-GEN Step 1 format.

    Creates:
        findings/fullSize/{label}_{idx}.png  — binary masks (0/255)
        findings/banana/{label}_{idx}/crop.png  — cropped object regions
        masks/{label}_{idx}.png  — copy of binary masks
    """
    findings_dir = output_dir / "findings"
    fullsize_dir = findings_dir / "fullSize"
    banana_dir = findings_dir / "banana"
    masks_dir = output_dir / "masks"

    for d in [fullsize_dir, banana_dir, masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    w, h = image.size
    img_np = np.array(image)
    metadata = {"objects": [], "image_size": [w, h]}

    for det in detections:
        label = det["label"]
        idx = det["instance_idx"]
        mask = det["mask"]
        bbox = det["bbox"]
        score = det["score"]

        name = f"{label}_{idx}"

        # Save binary mask (0/255 grayscale PNG)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_img.save(str(fullsize_dir / f"{name}.png"))
        mask_img.save(str(masks_dir / f"{name}.png"))

        # Save cropped object region
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        crop = img_np[y_min:y_max, x_min:x_max]
        crop_dir = banana_dir / name
        crop_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(str(crop_dir / "crop.png"))

        metadata["objects"].append({
            "name": name,
            "label": label,
            "instance_idx": idx,
            "score": round(score, 4),
            "bbox": [x_min, y_min, x_max, y_max],
        })

    # Write metadata
    meta_path = findings_dir / "segmentation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata written to %s", meta_path)

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM3 text-prompted segmentation for 3D-RE-GEN"
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated object labels to detect",
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--padding", type=int, default=5)
    parser.add_argument(
        "--depth", action="store_true", default=True,
        help="Also compute monocular depth",
    )
    parser.add_argument("--no-depth", dest="depth", action="store_false")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output)

    if not image_path.is_file():
        log.error("Input image not found: %s", image_path)
        return 1

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if not labels:
        log.error("No labels provided")
        return 1

    log.info("Input: %s", image_path)
    log.info("Output: %s", output_dir)
    log.info("Labels (%d): %s", len(labels), labels)
    log.info("Device: %s", args.device)

    output_dir.mkdir(parents=True, exist_ok=True)
    failed_marker = output_dir / "findings" / ".sam3_failed"

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load SAM3
        model, processor = load_sam3(
            model_id=args.model, device=args.device
        )

        # Run segmentation
        detections = segment_with_sam3(
            image=image,
            labels=labels,
            model=model,
            processor=processor,
            device=args.device,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
        )

        if not detections:
            log.warning("No objects detected! Check labels and threshold.")

        # Write outputs
        metadata = write_outputs(
            image=image,
            detections=detections,
            output_dir=output_dir,
            padding=args.padding,
        )

        # Free SAM3 VRAM before depth estimation
        del model, processor
        torch.cuda.empty_cache()

        # Depth estimation
        if args.depth:
            depth_path = output_dir / "findings" / "depth.png"
            estimate_depth(image, depth_path, device=args.device)

        log.info(
            "Segmentation complete: %d objects detected",
            len(detections),
        )
        return 0

    except Exception as exc:
        log.error("SAM3 segmentation failed: %s", exc, exc_info=True)
        # Write failure marker so runner can detect and fall back
        failed_marker.parent.mkdir(parents=True, exist_ok=True)
        failed_marker.write_text(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Class priors (approximate object heights in meters)
CLASS_PRIORS: Dict[str, float] = {
    "door": 2.0,
    "doorway": 2.0,
    "desk": 0.75,
    "table": 0.75,
    "dining table": 0.75,
    "coffee table": 0.45,
    "chair": 1.0,
    "stool": 0.5,
    "sofa": 0.9,
    "couch": 0.9,
    "bed": 0.6,        # mattress thickness-ish
    "cabinet": 1.8,
    "wardrobe": 1.8,
    "bookshelf": 1.8,
    "shelf": 1.8,
}


def normalize_class_name(name: str) -> str:
    return name.strip().lower()


def get_prior_for_class(name: str) -> Optional[float]:
    key = normalize_class_name(name)
    # Try exact match
    if key in CLASS_PRIORS:
        return CLASS_PRIORS[key]
    # Simple synonyms / contains checks
    for k, v in CLASS_PRIORS.items():
        if k in key:
            return v
    return None


def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    layout_prefix = os.getenv("LAYOUT_PREFIX")  # e.g. scenes/<sceneId>/layout

    if not layout_prefix:
        print("[SCALE] LAYOUT_PREFIX env var is required", file=sys.stderr)
        sys.exit(1)

    root = Path("/mnt/gcs")
    layout_dir = root / layout_prefix
    layout_path = layout_dir / "scene_layout.json"

    print(f"[SCALE] Bucket: {bucket}")
    print(f"[SCALE] Scene ID: {scene_id}")
    print(f"[SCALE] Layout dir: {layout_dir}")

    if not layout_path.is_file():
        print(f"[SCALE] ERROR: scene_layout.json not found at {layout_path}", file=sys.stderr)
        sys.exit(1)

    with layout_path.open("r") as f:
        layout = json.load(f)

    room_box = layout.get("room_box")
    objects: List[dict] = layout.get("objects", [])

    if room_box is None:
        print("[SCALE] WARNING: room_box missing; cannot scale room extents", file=sys.stderr)

    print(f"[SCALE] Found {len(objects)} objects in layout")

    # Collect scale samples from class priors
    scales: List[float] = []
    used_samples: List[dict] = []

    # We assume Y-axis (index 1) is "up" in world frame
    UP_AXIS = 1

    for obj in objects:
        name = obj.get("class_name", "")
        prior_h = get_prior_for_class(name)
        obb = obj.get("obb")

        if prior_h is None or obb is None:
            continue

        extents = obb.get("extents")
        if not isinstance(extents, list) or len(extents) != 3:
            continue

        # extents are half-lengths; height ~ 2 * extents[UP_AXIS]
        measured_h = 2.0 * float(extents[UP_AXIS])
        if measured_h <= 1e-6:
            continue

        s = prior_h / measured_h
        scales.append(s)
        used_samples.append(
            {
                "class_name": name,
                "prior_h_m": prior_h,
                "measured_h_units": measured_h,
                "scale_sample": s,
            }
        )

    if not scales:
        # No priors available; record unity scale
        S = 1.0
        layout.setdefault("scale", {})
        layout["scale"]["factor"] = S
        layout["scale"]["n_samples"] = 0
        layout["scale"]["priors_used"] = []
        layout["scale"]["source"] = "none"
        layout["scale"]["up_axis"] = "y"

        # Write a separate scaled layout file to avoid loops
        scaled_path = layout_dir / "scene_layout_scaled.json"
        with scaled_path.open("w") as f:
            json.dump(layout, f, indent=2)

        print("[SCALE] No class priors available; wrote scene_layout_scaled.json with factor=1.0")
        return

    # Compute global scale factor as median for robustness
    scales_arr = np.array(scales, dtype=np.float32)
    S = float(np.median(scales_arr))

    print(f"[SCALE] Using global scale factor S={S:.4f} from {len(scales)} samples")

    # Scale room_box
    if room_box is not None:
        mins = np.array(room_box.get("min", [0, 0, 0]), dtype=np.float32)
        maxs = np.array(room_box.get("max", [0, 0, 0]), dtype=np.float32)
        mins_scaled = (S * mins).tolist()
        maxs_scaled = (S * maxs).tolist()
        room_box["min"] = mins_scaled
        room_box["max"] = maxs_scaled
        layout["room_box"] = room_box

    # Scale camera extrinsics translations
    camera = layout.get("camera")
    if camera is not None:
        extr = camera.get("extrinsics")
        if isinstance(extr, list) and len(extr) == 3 and len(extr[0]) == 4:
            w2c = np.array(extr, dtype=np.float32)  # 3x4
            R = w2c[:, :3]
            t = w2c[:, 3]
            t_scaled = (S * t)
            w2c_scaled = np.concatenate([R, t_scaled[:, None]], axis=1)
            camera["extrinsics"] = w2c_scaled.tolist()
            layout["camera"] = camera

    # Scale objects centers + extents
    for obj in objects:
        center = obj.get("center3d")
        if isinstance(center, list) and len(center) == 3:
            c = np.array(center, dtype=np.float32)
            obj["center3d"] = (S * c).tolist()

        obb = obj.get("obb")
        if isinstance(obb, dict):
            c_obb = obb.get("center")
            e_obb = obb.get("extents")

            if isinstance(c_obb, list) and len(c_obb) == 3:
                cc = np.array(c_obb, dtype=np.float32)
                obb["center"] = (S * cc).tolist()

            if isinstance(e_obb, list) and len(e_obb) == 3:
                ee = np.array(e_obb, dtype=np.float32)
                obb["extents"] = (S * ee).tolist()

            obj["obb"] = obb

    layout["objects"] = objects

    # Record scale metadata
    layout.setdefault("scale", {})
    layout["scale"]["factor"] = S
    layout["scale"]["n_samples"] = len(scales)
    layout["scale"]["priors_used"] = used_samples
    layout["scale"]["source"] = "class_priors"
    layout["scale"]["up_axis"] = "y"

    scaled_path = layout_dir / "scene_layout_scaled.json"
    with scaled_path.open("w") as f:
        json.dump(layout, f, indent=2)

    print(f"[SCALE] Wrote scaled layout to {scaled_path}")
    print("[SCALE] Done.")


if __name__ == "__main__":
    main()

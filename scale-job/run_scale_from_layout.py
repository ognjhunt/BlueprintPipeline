import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.validation.entrypoint_checks import validate_required_env_vars

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


def load_metric_metadata(layout_dir: Path) -> Tuple[dict, Optional[Path]]:
    """
    Look for user-provided metric cues. Default path is layout_dir/metric_metadata.json,
    but METRIC_METADATA_PATH can override it.
    """
    override = os.getenv("METRIC_METADATA_PATH")
    candidate = Path(override) if override else layout_dir / "metric_metadata.json"
    if not candidate.is_file():
        return {}, None
    try:
        with candidate.open("r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}, candidate
    except Exception as e:
        print(f"[SCALE] WARNING: failed to read metric metadata at {candidate}: {e}", file=sys.stderr)
        return {}, candidate


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


def gather_reference_scales(metadata: dict, objects: List[dict]) -> Tuple[List[float], List[dict]]:
    """
    Turn explicit references into scale samples. Supports:
      - scale_factors: list[float]
      - known_heights: list[{object_id, height_m}]
      - reference_objects: list[{object_id?, class_name?, height_m}]
    """
    scales: List[float] = []
    samples: List[dict] = []

    for s in metadata.get("scale_factors", []):
        try:
            sf = float(s)
        except (TypeError, ValueError):
            continue
        if sf <= 0:
            continue
        scales.append(sf)
        samples.append({"source": "explicit_factor", "scale_sample": sf})

    for entry in metadata.get("known_heights", []):
        if not isinstance(entry, dict):
            continue
        obj_id = entry.get("object_id")
        height_m = entry.get("height_m")
        if obj_id is None or height_m is None:
            continue
        try:
            height_m = float(height_m)
        except (TypeError, ValueError):
            continue
        match = next((o for o in objects if o.get("id") == obj_id), None)
        if not match:
            continue
        obb = match.get("obb")
        if not isinstance(obb, dict):
            continue
        extents = obb.get("extents")
        if not isinstance(extents, list) or len(extents) != 3:
            continue
        measured_h = 2.0 * float(extents[1])
        if measured_h <= 1e-6:
            continue
        s = height_m / measured_h
        scales.append(s)
        samples.append(
            {
                "source": "known_height",
                "object_id": obj_id,
                "height_m": height_m,
                "measured_h_units": measured_h,
                "scale_sample": s,
            }
        )

    for entry in metadata.get("reference_objects", []):
        if not isinstance(entry, dict):
            continue
        height_m = entry.get("height_m")
        if height_m is None:
            continue
        try:
            height_m = float(height_m)
        except (TypeError, ValueError):
            continue
        match = None
        obj_id = entry.get("object_id")
        class_name = entry.get("class_name")
        if obj_id is not None:
            match = next((o for o in objects if o.get("id") == obj_id), None)
        elif class_name:
            normalized = normalize_class_name(str(class_name))
            match = next(
                (o for o in objects if normalize_class_name(o.get("class_name", "")) == normalized),
                None,
            )
        if not match:
            continue
        obb = match.get("obb")
        if not isinstance(obb, dict):
            continue
        extents = obb.get("extents")
        if not isinstance(extents, list) or len(extents) != 3:
            continue
        measured_h = 2.0 * float(extents[1])
        if measured_h <= 1e-6:
            continue
        s = height_m / measured_h
        scales.append(s)
        samples.append(
            {
                "source": "reference_object",
                "object_id": obj_id,
                "class_name": class_name,
                "height_m": height_m,
                "measured_h_units": measured_h,
                "scale_sample": s,
            }
        )

    return scales, samples


def gather_scene_metric_scales(metadata: dict, room_box: Optional[dict]) -> Tuple[List[float], List[dict]]:
    scales: List[float] = []
    samples: List[dict] = []

    if not isinstance(room_box, dict):
        return scales, samples

    mins = room_box.get("min")
    maxs = room_box.get("max")
    if not (isinstance(mins, list) and isinstance(maxs, list) and len(mins) == 3 and len(maxs) == 3):
        return scales, samples

    dims = np.array(maxs, dtype=np.float32) - np.array(mins, dtype=np.float32)
    if np.any(dims <= 1e-6):
        return scales, samples

    scene_metrics = metadata.get("scene_metrics", {})
    if isinstance(scene_metrics, dict):
        metric_map = {
            "room_width_m": 0,
            "room_height_m": 1,
            "room_depth_m": 2,
        }
        for key, axis in metric_map.items():
            value = scene_metrics.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            s = value / float(dims[axis])
            scales.append(s)
            samples.append(
                {
                    "source": "scene_metric",
                    "metric": key,
                    "metric_value_m": value,
                    "measured_units": float(dims[axis]),
                    "scale_sample": s,
                }
            )

        room_box_m = scene_metrics.get("room_box_m")
        if isinstance(room_box_m, dict):
            room_min = room_box_m.get("min")
            room_max = room_box_m.get("max")
            if (
                isinstance(room_min, list)
                and isinstance(room_max, list)
                and len(room_min) == 3
                and len(room_max) == 3
            ):
                known_dims = np.array(room_max, dtype=np.float32) - np.array(room_min, dtype=np.float32)
                for axis, axis_name in enumerate(["x", "y", "z"]):
                    if known_dims[axis] <= 1e-6:
                        continue
                    s = float(known_dims[axis]) / float(dims[axis])
                    scales.append(s)
                    samples.append(
                        {
                            "source": "scene_metric",
                            "metric": f"room_box_{axis_name}_m",
                            "metric_value_m": float(known_dims[axis]),
                            "measured_units": float(dims[axis]),
                            "scale_sample": s,
                        }
                    )

    return scales, samples


def main() -> None:
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
            "LAYOUT_PREFIX": "Path prefix for layout files (scenes/<sceneId>/layout)",
        },
        label="[SCALE]",
    )

    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    layout_prefix = os.getenv("LAYOUT_PREFIX")  # e.g. scenes/<sceneId>/layout

    root = Path("/mnt/gcs")
    layout_dir = root / layout_prefix
    layout_path = layout_dir / "scene_layout.json"

    print(f"[SCALE] Bucket: {bucket}")
    print(f"[SCALE] Scene ID: {scene_id}")
    print(f"[SCALE] Layout dir: {layout_dir}")

    scaled_path = layout_dir / "scene_layout_scaled.json"
    done_marker_path = layout_dir / "scene_layout_scaled.done"
    expected_outputs = [scaled_path, done_marker_path]
    print("[SCALE] Expected outputs:")
    for p in expected_outputs:
        print(f"  - {p}")

    existing_outputs = []
    if scaled_path.is_file():
        try:
            with scaled_path.open("r") as f:
                scaled_layout = json.load(f)
            factor = scaled_layout.get("scale", {}).get("factor")
            if factor is not None:
                existing_outputs.append(f"{scaled_path} (scale factor={factor})")
            else:
                existing_outputs.append(str(scaled_path))
        except Exception as e:
            print(
                f"[SCALE] WARNING: failed to inspect existing scaled layout at {scaled_path}: {e}",
                file=sys.stderr,
            )

    if done_marker_path.is_file():
        existing_outputs.append(str(done_marker_path))

    if len(existing_outputs) == len(expected_outputs):
        print("[SCALE] All expected outputs already exist; skipping scale step.")
        for entry in existing_outputs:
            print(f"[SCALE]   • {entry}")
        return

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

    metric_metadata, metadata_path = load_metric_metadata(layout_dir)
    if metadata_path:
        print(f"[SCALE] Found metric metadata at {metadata_path}")

    # Collect scale samples from explicit references then scene metrics and class priors
    scales: List[float] = []
    used_samples: List[dict] = []

    ref_scales, ref_samples = gather_reference_scales(metric_metadata, objects)
    scales.extend(ref_scales)
    used_samples.extend(ref_samples)

    metric_scales, metric_samples = gather_scene_metric_scales(metric_metadata, room_box)
    scales.extend(metric_scales)
    used_samples.extend(metric_samples)

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
                "source": "class_prior",
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
        layout["scale"]["reference_samples"] = []
        layout["scale"]["source"] = "none"
        layout["scale"]["up_axis"] = "y"

        # Write a separate scaled layout file to avoid loops
        scaled_path = layout_dir / "scene_layout_scaled.json"
        with scaled_path.open("w") as f:
            json.dump(layout, f, indent=2)

        try:
            done_marker_path.write_text("ok\n")
        except Exception as e:
            print(f"[SCALE] WARNING: failed to write done marker at {done_marker_path}: {e}", file=sys.stderr)

        print("[SCALE] No scale cues available; wrote scene_layout_scaled.json with factor=1.0")
        return

    # Compute global scale factor as median for robustness
    scales_arr = np.array(scales, dtype=np.float32)
    S = float(np.median(scales_arr))

    source_flags = {
        "reference": len(ref_scales) > 0,
        "scene_metric": len(metric_scales) > 0,
        "prior": len(scales) > (len(ref_scales) + len(metric_scales)),
    }
    if sum(source_flags.values()) > 1:
        source_label = "blend"
    elif source_flags["reference"]:
        source_label = "reference"
    elif source_flags["scene_metric"]:
        source_label = "scene_metric"
    else:
        source_label = "class_priors"

    print(f"[SCALE] Using global scale factor S={S:.4f} from {len(scales)} samples (source={source_label})")

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
    layout["scale"]["priors_used"] = [s for s in used_samples if s.get("source") == "class_prior"]
    layout["scale"]["reference_samples"] = [s for s in used_samples if s.get("source") != "class_prior"]
    layout["scale"]["source"] = source_label
    layout["scale"]["up_axis"] = "y"
    if metadata_path:
        layout["scale"]["metadata_path"] = str(metadata_path)

    scaled_path = layout_dir / "scene_layout_scaled.json"
    with scaled_path.open("w") as f:
        json.dump(layout, f, indent=2)

    try:
        done_marker_path.write_text("ok\n")
    except Exception as e:
        print(f"[SCALE] WARNING: failed to write done marker at {done_marker_path}: {e}", file=sys.stderr)

    print(f"[SCALE] Wrote scaled layout to {scaled_path}")
    print("[SCALE] Done.")


if __name__ == "__main__":
    main()

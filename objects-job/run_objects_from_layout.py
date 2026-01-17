import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.validation.entrypoint_checks import validate_required_env_vars
from monitoring.alerting import send_alert

logger = logging.getLogger(__name__)

def load_da3_geom(geom_path: Path):
    # Pickle loading intentionally disabled for security.
    try:
        data = np.load(str(geom_path), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Failed to load DA3 geom file as .npz: {geom_path}") from exc
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError(f"DA3 geom file must be a .npz archive: {geom_path}")
    required_keys = {"depth", "conf", "extrinsics", "intrinsics", "image_paths"}
    missing_keys = required_keys.difference(data.files)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"DA3 geom file missing required keys: {missing}")
    depth = np.asarray(data["depth"])        # [N, H, W]
    conf = np.asarray(data["conf"])         # [N, H, W]
    extrinsics = np.asarray(data["extrinsics"])  # [N, 3, 4] (w2c)
    intrinsics = np.asarray(data["intrinsics"])  # [N, 3, 3]
    image_paths = np.asarray(data["image_paths"])  # [N]
    return depth, conf, extrinsics, intrinsics, image_paths


def backproject_region_to_world(depth, conf, K, w2c, x0, y0, x1, y1, conf_thresh=0.6, max_points=100000):
    """
    depth, conf: H x W
    region: [x0, y0, x1, y1] in pixel coords (inclusive/exclusive)
    Returns: (P, 3) world-space points for this region.
    """
    H, W = depth.shape
    x0 = int(np.clip(x0, 0, W - 1))
    x1 = int(np.clip(x1, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))
    y1 = int(np.clip(y1, 0, H - 1))

    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3), dtype=np.float32)

    region_depth = depth[y0:y1, x0:x1]
    region_conf = conf[y0:y1, x0:x1]

    mask = (region_depth > 0.0) & (region_conf >= conf_thresh)
    ys, xs = np.nonzero(mask)
    num = xs.size
    if num == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if num > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(num, size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    z = region_depth[ys, xs]
    # Convert local xs,ys in region to global pixel coords
    xs_global = xs + x0
    ys_global = ys + y0

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x_cam = (xs_global.astype(np.float32) - cx) * z / fx
    y_cam = (ys_global.astype(np.float32) - cy) * z / fy
    z_cam = z

    X_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [P, 3]

    R = w2c[:, :3]  # 3x3
    t = w2c[:, 3]   # 3
    Rt = R.T

    X_w = (Rt @ (X_cam - t[None, :]).T).T  # [P, 3]

    return X_w.astype(np.float32)


def parse_yolo_labels(label_path: Path, class_names: Optional[List[str]] = None):
    """
    Parse YOLO *segmentation* label file (instance polygons):

      class_id x1 y1 x2 y2 ... xn yn

    All coords normalized [0,1]. We derive bbox from polygon and keep
    polygon coords as well.
    """
    objects = []
    if not label_path.is_file():
        return objects

    with label_path.open("r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            try:
                vals = list(map(float, parts))
            except ValueError:
                continue

            cid = int(vals[0])
            coords = np.array(vals[1:], dtype=np.float32).reshape(-1, 2)  # [N, 2]
            xs = coords[:, 0]
            ys = coords[:, 1]

            cx = float(xs.mean())
            cy = float(ys.mean())
            w = float(xs.max() - xs.min())
            h = float(ys.max() - ys.min())

            name = (
                class_names[cid]
                if class_names and 0 <= cid < len(class_names)
                else f"class_{cid}"
            )

            objects.append(
                {
                    "id": idx,
                    "class_id": cid,
                    "class_name": name,
                    "bbox2d": [cx, cy, w, h],
                    "polygon": coords.tolist(),
                }
            )
    return objects


def load_class_names(data_yaml_path: Path) -> Optional[List[str]]:
    if not data_yaml_path.is_file():
        return None
    try:
        with data_yaml_path.open("r") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, list):
            return [str(n) for n in names]
    except Exception as e:
        logger.warning(
            "[OBJECTS] Failed to read class names from %s: %s",
            data_yaml_path,
            e,
        )
    return None


def main() -> None:
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
            "DA3_PREFIX": "Path prefix for DA3 inputs (scenes/<sceneId>/da3)",
            "SEG_DATASET_PREFIX": "Path prefix for segmentation dataset (scenes/<sceneId>/seg/dataset)",
            "LAYOUT_PREFIX": "Path prefix for layout files (scenes/<sceneId>/layout)",
        },
        label="[OBJECTS]",
    )

    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    da3_prefix = os.getenv("DA3_PREFIX")          # e.g. scenes/<sceneId>/da3
    seg_dataset_prefix = os.getenv("SEG_DATASET_PREFIX")  # e.g. scenes/<sceneId>/seg/dataset
    layout_prefix = os.getenv("LAYOUT_PREFIX")    # e.g. scenes/<sceneId>/layout

    root = Path("/mnt/gcs")
    da3_dir = root / da3_prefix
    seg_dataset_dir = root / seg_dataset_prefix
    layout_dir = root / layout_prefix

    logger.info("[OBJECTS] Bucket: %s", bucket)
    logger.info("[OBJECTS] Scene ID: %s", scene_id)
    logger.info("[OBJECTS] DA3 dir: %s", da3_dir)
    logger.info("[OBJECTS] Seg dataset dir: %s", seg_dataset_dir)
    logger.info("[OBJECTS] Layout dir: %s", layout_dir)

    layout_path = layout_dir / "scene_layout.json"
    expected_outputs = [layout_path]
    logger.info("[OBJECTS] Expected outputs:")
    for p in expected_outputs:
        logger.info("[OBJECTS]   - %s", p)

    existing_outputs = []
    if layout_path.is_file():
        try:
            with layout_path.open("r") as f:
                existing_layout = json.load(f)
            objs = existing_layout.get("objects", [])
            if objs:
                existing_outputs.append(
                    f"{layout_path} (objects already present: {len(objs)})"
                )
        except Exception as e:
            logger.warning(
                "[OBJECTS] Failed to inspect existing layout at %s: %s",
                layout_path,
                e,
            )

    if len(existing_outputs) == len(expected_outputs):
        logger.info("[OBJECTS] All expected outputs already exist; skipping objects step.")
        for entry in existing_outputs:
            logger.info("[OBJECTS]   • %s", entry)
        return

    geom_path = da3_dir / "da3_geom.npz"
    data_yaml_path = seg_dataset_dir / "data.yaml"
    valid_images_dir = seg_dataset_dir / "valid" / "images"
    valid_labels_dir = seg_dataset_dir / "valid" / "labels"
    room_label_path = valid_labels_dir / "room.txt"

    if not geom_path.is_file():
        logger.error("[OBJECTS] da3_geom.npz not found at %s", geom_path)
        sys.exit(1)

    if not layout_path.is_file():
        logger.error("[OBJECTS] scene_layout.json not found at %s", layout_path)
        sys.exit(1)

    if not room_label_path.is_file():
        logger.warning("[OBJECTS] Room labels not found at %s", room_label_path)

    layout_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing layout JSON ----
    with layout_path.open("r") as f:
        layout = json.load(f)

    # ---- Load DA3 geometry ----
    depth_all, conf_all, extr_all, intr_all, image_paths = load_da3_geom(geom_path)
    if depth_all.shape[0] != 1:
        logger.warning(
            "[OBJECTS] Multiple frames in da3_geom (%s). Using first.",
            depth_all.shape[0],
        )

    depth = depth_all[0]
    conf = conf_all[0]
    w2c = extr_all[0]
    K = intr_all[0]
    H, W = depth.shape
    logger.info("[OBJECTS] Depth shape: %s", depth.shape)

    # ---- Load class names and YOLO segmentation labels ----
    class_names = load_class_names(data_yaml_path)
    objects = parse_yolo_labels(room_label_path, class_names=class_names)
    logger.info("[OBJECTS] Parsed %s YOLO objects", len(objects))

    if not objects:
        layout["objects"] = []
        with layout_path.open("w") as f:
            json.dump(layout, f, indent=2)
        logger.info("[OBJECTS] No objects found; wrote layout with empty objects[]")
        return

    # ---- For each object, build a per-object point cloud + proxy ----
    proxies = []
    for obj in objects:
        cid = obj["class_id"]
        name = obj["class_name"]
        cx_n, cy_n, bw_n, bh_n = obj["bbox2d"]

        # YOLO normalized -> pixel region
        x_center = cx_n * W
        y_center = cy_n * H
        bw_px = bw_n * W
        bh_px = bh_n * H

        # Define region as the bbox +/- optional margin
        margin = 0.05  # 5% of image dims
        x0 = x_center - bw_px / 2 - margin * W
        x1 = x_center + bw_px / 2 + margin * W
        y0 = y_center - bh_px / 2 - margin * H
        y1 = y_center + bh_px / 2 + margin * H

        pts_obj = backproject_region_to_world(
            depth,
            conf,
            K,
            w2c,
            x0,
            y0,
            x1,
            y1,
            conf_thresh=0.6,
            max_points=50000,
        )

        if pts_obj.shape[0] == 0:
            logger.info("[OBJECTS] Object %s has no valid depth points", name)
            center3d = None
            obb = None
        else:
            # Center as mean
            center = pts_obj.mean(axis=0)

            # Axis-aligned bbox extents (aligned to room/world axes)
            mins = pts_obj.min(axis=0)
            maxs = pts_obj.max(axis=0)
            extents = (maxs - mins) / 2.0

            # For simplicity, orientation = identity in world frame
            R = np.eye(3, dtype=np.float32)

            center3d = center.tolist()
            obb = {
                "center": center3d,
                "extents": extents.tolist(),
                "R": R.tolist(),
            }

        proxy = {
            "id": obj["id"],
            "class_id": cid,
            "class_name": name,
            "bbox2d": obj["bbox2d"],
            "polygon": obj.get("polygon"),
            "center3d": center3d,
            "obb": obb,
        }
        proxies.append(proxy)

    # ---- Update layout JSON ----
    layout["objects"] = proxies
    with layout_path.open("w") as f:
        json.dump(layout, f, indent=2)

    logger.info("[OBJECTS] Wrote %s objects to %s", len(proxies), layout_path)
    logger.info("[OBJECTS] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        send_alert(
            event_type="objects_job_fatal_exception",
            summary="Objects job failed with an unhandled exception",
            details={
                "job": "objects-job",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            severity=os.getenv("ALERT_JOB_EXCEPTION_SEVERITY", "critical"),
        )
        raise

import os
import sys
import json
from pathlib import Path

import numpy as np


def load_da3_geom(geom_path: Path):
    data = np.load(str(geom_path), allow_pickle=True)
    depth = np.asarray(data["depth"])        # [N, H, W]
    conf = np.asarray(data["conf"])         # [N, H, W]
    extrinsics = np.asarray(data["extrinsics"])  # [N, 3, 4] (w2c)
    intrinsics = np.asarray(data["intrinsics"])  # [N, 3, 3]
    image_paths = np.asarray(data["image_paths"])  # [N]
    return depth, conf, extrinsics, intrinsics, image_paths


def backproject_depth_to_world(depth, conf, K, w2c, conf_thresh=0.6, max_points=500000):
    """
    depth: H x W
    conf: H x W
    K: 3 x 3
    w2c: 3 x 4  (X_c = R X_w + t)
    Returns: (P, 3) world-space points.
    """
    H, W = depth.shape
    mask = (depth > 0.0) & (conf >= conf_thresh)

    ys, xs = np.nonzero(mask)
    num = xs.size
    if num == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if num > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(num, size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    z = depth[ys, xs]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x_cam = (xs.astype(np.float32) - cx) * z / fx
    y_cam = (ys.astype(np.float32) - cy) * z / fy
    z_cam = z

    X_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [P, 3]

    R = w2c[:, :3]  # 3x3
    t = w2c[:, 3]   # 3
    Rt = R.T

    X_w = (Rt @ (X_cam - t[None, :]).T).T  # [P, 3]

    return X_w.astype(np.float32)


def parse_yolo_labels(label_path: Path, class_names=None):
    """
    Parse YOLO *segmentation* label file:

      class_id x1 y1 x2 y2 ... xn yn

    All coords normalized [0,1]. We derive a bbox from the polygon and
    also keep the polygon coords on the object.
    """
    objects = []
    if not label_path.is_file():
        return objects

    with label_path.open("r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            # Need at least: class + 3 (x,y) pairs = 7 numbers
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

            # Derive YOLO-style bbox from polygon
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


def estimate_object_centers3d(objects, depth, conf, K, w2c, img_w, img_h):
    """
    For each bbox, sample depth at bbox center and backproject to world.
    """
    centers = []
    H, W = depth.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx0 = float(K[0, 2])
    cy0 = float(K[1, 2])

    R = w2c[:, :3]  # 3x3
    t = w2c[:, 3]   # 3
    Rt = R.T

    for obj in objects:
        cx_n, cy_n, bw_n, bh_n = obj["bbox2d"]
        # Convert normalized [0,1] to pixel coords
        px = cx_n * img_w
        py = cy_n * img_h

        # Clamp to valid indices
        ix = int(np.clip(px, 0, W - 1))
        iy = int(np.clip(py, 0, H - 1))

        z = float(depth[iy, ix])
        c = float(conf[iy, ix])

        if z <= 0.0 or c <= 0.0:
            obj_center = None
        else:
            x_cam = (px - cx0) * z / fx
            y_cam = (py - cy0) * z / fy
            X_cam = np.array([x_cam, y_cam, z], dtype=np.float32)

            # X_w = R^T (X_c - t)
            X_w = Rt @ (X_cam - t)
            obj_center = X_w.tolist()

        centers.append(obj_center)

    return centers


def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    da3_prefix = os.getenv("DA3_PREFIX")  # e.g. scenes/<sceneId>/da3
    seg_dataset_prefix = os.getenv("SEG_DATASET_PREFIX")  # e.g. scenes/<sceneId>/seg/dataset
    out_prefix = os.getenv("LAYOUT_PREFIX")  # e.g. scenes/<sceneId>/layout

    if not da3_prefix or not seg_dataset_prefix or not out_prefix:
        print(
            "[LAYOUT] DA3_PREFIX, SEG_DATASET_PREFIX, and LAYOUT_PREFIX env vars are required",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path("/mnt/gcs")
    da3_dir = root / da3_prefix
    seg_dataset_dir = root / seg_dataset_prefix
    out_dir = root / out_prefix

    print(f"[LAYOUT] Bucket: {bucket}")
    print(f"[LAYOUT] Scene ID: {scene_id}")
    print(f"[LAYOUT] DA3 dir: {da3_dir}")
    print(f"[LAYOUT] Seg dataset dir: {seg_dataset_dir}")
    print(f"[LAYOUT] Layout out dir: {out_dir}")

    geom_path = da3_dir / "da3_geom.npz"
    if not geom_path.is_file():
        print(f"[LAYOUT] ERROR: da3_geom.npz not found at {geom_path}", file=sys.stderr)
        sys.exit(1)

    # For labels, assume YOLO file is under seg/dataset/valid/labels/room.txt
    # and image is seg/dataset/valid/images/room.jpg
    valid_images_dir = seg_dataset_dir / "valid" / "images"
    valid_labels_dir = seg_dataset_dir / "valid" / "labels"
    room_img_path = valid_images_dir / "room.jpg"
    room_label_path = valid_labels_dir / "room.txt"

    if not room_img_path.is_file():
        print(f"[LAYOUT] WARNING: room image not found at {room_img_path}", file=sys.stderr)

    if not room_label_path.is_file():
        print(f"[LAYOUT] WARNING: room labels not found at {room_label_path}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load DA3 geometry ----
    depth_all, conf_all, extr_all, intr_all, image_paths = load_da3_geom(geom_path)

    if depth_all.shape[0] != 1:
        print(
            f"[LAYOUT] WARNING: multiple frames in da3_geom ({depth_all.shape[0]}). Using first.",
            file=sys.stderr,
        )

    depth = depth_all[0]       # H x W
    conf = conf_all[0]         # H x W
    w2c = extr_all[0]          # 3 x 4
    K = intr_all[0]            # 3 x 3

    H, W = depth.shape
    print(f"[LAYOUT] Depth shape: {depth.shape}, Intrinsics: {K}, w2c: {w2c}")

    # ---- Build room point cloud and fit AABB ----
    pts_world = backproject_depth_to_world(depth, conf, K, w2c, conf_thresh=0.6, max_points=500000)
    if pts_world.shape[0] == 0:
        print("[LAYOUT] ERROR: No valid depth points; cannot compute layout.", file=sys.stderr)
        sys.exit(1)

    mins = pts_world.min(axis=0).tolist()
    maxs = pts_world.max(axis=0).tolist()
    print(f"[LAYOUT] Room AABB min={mins}, max={maxs}")

    room_box = {"min": mins, "max": maxs}

    # ---- Load YOLO labels and estimate object centers ----
    # For now, keep class_id -> "class_<id>" (class_names=None)
    objects = parse_yolo_labels(room_label_path, class_names=None)
    centers = estimate_object_centers3d(objects, depth, conf, K, w2c, img_w=W, img_h=H)

    for obj, c in zip(objects, centers):
        obj["center3d"] = c

    # ---- Build layout JSON ----
    layout = {
        "scene_id": scene_id,
        "camera": {
            "intrinsics": K.tolist(),
            "extrinsics": w2c.tolist(),
            "image_paths": image_paths.tolist(),
        },
        "room_box": room_box,
        "objects": objects,
    }

    layout_path = out_dir / "scene_layout.json"
    with layout_path.open("w") as f:
        json.dump(layout, f, indent=2)

    print(f"[LAYOUT] Wrote layout JSON to {layout_path}")
    print("[LAYOUT] Done.")


if __name__ == "__main__":
    main()

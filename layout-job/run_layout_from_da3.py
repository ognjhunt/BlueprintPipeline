import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def fuse_multiview_point_clouds(depth_all, conf_all, intr_all, extr_all, conf_thresh=0.6, max_per_view=150000):
    """
    Backproject all DA3 frames and concatenate a fused cloud for layout/plane fitting.
    """
    fused = []
    num_views = depth_all.shape[0]
    for i in range(num_views):
        pts = backproject_depth_to_world(
            depth_all[i], conf_all[i], intr_all[i], extr_all[i], conf_thresh=conf_thresh, max_points=max_per_view
        )
        if pts.size:
            fused.append(pts)
            print(f"[LAYOUT] View {i}: kept {pts.shape[0]} pts")
    if not fused:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(fused, axis=0)


def fit_plane_ransac(points: np.ndarray, num_iters: int = 200, threshold: float = 0.02) -> Tuple[np.ndarray, float]:
    """
    Very lightweight RANSAC plane fit. Returns (normal, d) with plane: n^T x + d = 0.
    """
    if points.shape[0] < 3:
        return np.array([0, 1, 0], dtype=np.float32), 0.0

    best_inliers = -1
    best_plane = (np.array([0, 1, 0], dtype=np.float32), 0.0)

    rng = np.random.default_rng(123)
    for _ in range(num_iters):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        d = -float(np.dot(n, p0))
        # Count inliers
        dist = np.abs(points @ n + d)
        inliers = np.count_nonzero(dist < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (n.astype(np.float32), d)

    return best_plane


def plane_to_extent(normal: np.ndarray, d: float, points: np.ndarray) -> Dict:
    """
    Project points onto plane to get a loose extent polygon for debugging/export.
    """
    if points.shape[0] == 0:
        return {"equation": [float(x) for x in normal] + [float(d)], "bbox": None}

    # Build orthonormal basis on plane
    up = normal
    # Choose arbitrary vector not parallel to up
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if np.abs(np.dot(ref, up)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis1 = np.cross(up, ref)
    axis1 /= (np.linalg.norm(axis1) + 1e-8)
    axis2 = np.cross(up, axis1)

    projected = np.stack([points @ axis1, points @ axis2], axis=-1)
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)

    corners = np.array(
        [
            mins,
            [maxs[0], mins[1]],
            maxs,
            [mins[0], maxs[1]],
        ]
    )
    # Lift corners back to 3D plane
    center = (projected.mean(axis=0) @ np.stack([axis1, axis2], axis=0)).T
    origin = center - up * (np.dot(center, up) + d)
    verts = origin + corners @ np.stack([axis1, axis2], axis=0)

    return {
        "equation": [float(x) for x in normal] + [float(d)],
        "bbox": verts.tolist(),
    }


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


def parse_inventory_objects(inventory_path: Path) -> List[Dict]:
    """
    Parse Gemini inventory.json and convert bboxes to normalized polygons.

    Inventory bbox format: [ymin, xmin, ymax, xmax] with values 0-1000
    Output polygon format: [[x, y], ...] normalized to [0, 1]
    """
    objects = []
    if not inventory_path.is_file():
        print(f"[LAYOUT] WARNING: inventory.json not found at {inventory_path}", file=sys.stderr)
        return objects

    try:
        with inventory_path.open("r") as f:
            inventory = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"[LAYOUT] WARNING: Failed to parse inventory.json: {e}", file=sys.stderr)
        return objects

    for obj in inventory.get("objects", []):
        obj_id = obj.get("id")
        if not obj_id:
            continue

        # Skip objects that shouldn't be separate assets (scene shell elements)
        # We still want to compute OBB for them though
        bbox = obj.get("bbox")
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            print(f"[LAYOUT] WARNING: Object '{obj_id}' missing valid bbox, skipping")
            continue

        # bbox is [ymin, xmin, ymax, xmax] in 0-1000 range
        ymin, xmin, ymax, xmax = bbox

        # Normalize to [0, 1]
        ymin_norm = ymin / 1000.0
        xmin_norm = xmin / 1000.0
        ymax_norm = ymax / 1000.0
        xmax_norm = xmax / 1000.0

        # Create a rectangular polygon from the bbox (4 corners)
        # Format: [[x, y], ...] normalized
        polygon = [
            [xmin_norm, ymin_norm],  # top-left
            [xmax_norm, ymin_norm],  # top-right
            [xmax_norm, ymax_norm],  # bottom-right
            [xmin_norm, ymax_norm],  # bottom-left
        ]

        # Compute YOLO-style bbox2d [cx, cy, w, h]
        cx = (xmin_norm + xmax_norm) / 2.0
        cy = (ymin_norm + ymax_norm) / 2.0
        w = xmax_norm - xmin_norm
        h = ymax_norm - ymin_norm

        objects.append({
            "id": obj_id,
            "class_id": 0,  # Generic
            "class_name": obj.get("category", "object"),
            "short_description": obj.get("short_description", ""),
            "sim_role": obj.get("sim_role", ""),
            "must_be_separate_asset": obj.get("must_be_separate_asset", False),
            "bbox2d": [cx, cy, w, h],
            "polygon": polygon,
            "approx_location": obj.get("approx_location", ""),
        })

    return objects


def gather_object_points(
    polygon: List[List[float]],
    depth_all: np.ndarray,
    conf_all: np.ndarray,
    intr_all: np.ndarray,
    extr_all: np.ndarray,
    margin_frac: float = 0.02,
    conf_thresh: float = 0.6,
    max_points_per_view: int = 50000,
) -> np.ndarray:
    """
    Collect fused 3D points for an object polygon across all views.
    Uses polygon AABB per-view for speed; keeps points above conf_thresh.
    """
    pts_all = []
    num_views = depth_all.shape[0]
    coords = np.array(polygon, dtype=np.float32)
    for i in range(num_views):
        depth = depth_all[i]
        conf = conf_all[i]
        K = intr_all[i]
        w2c = extr_all[i]
        H, W = depth.shape
        xs = coords[:, 0] * W
        ys = coords[:, 1] * H
        x0 = max(0, int(xs.min() - margin_frac * W))
        x1 = min(W - 1, int(xs.max() + margin_frac * W))
        y0 = max(0, int(ys.min() - margin_frac * H))
        y1 = min(H - 1, int(ys.max() + margin_frac * H))

        region_depth = depth[y0:y1, x0:x1]
        region_conf = conf[y0:y1, x0:x1]
        mask = (region_depth > 0.0) & (region_conf >= conf_thresh)
        ys_mask, xs_mask = np.nonzero(mask)
        num = xs_mask.size
        if num == 0:
            continue
        if num > max_points_per_view:
            rng = np.random.default_rng(7)
            idx = rng.choice(num, size=max_points_per_view, replace=False)
            xs_mask = xs_mask[idx]
            ys_mask = ys_mask[idx]

        z = region_depth[ys_mask, xs_mask]
        xs_global = xs_mask + x0
        ys_global = ys_mask + y0

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        x_cam = (xs_global.astype(np.float32) - cx) * z / fx
        y_cam = (ys_global.astype(np.float32) - cy) * z / fy
        z_cam = z
        X_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        R = w2c[:, :3]
        t = w2c[:, 3]
        Rt = R.T
        X_w = (Rt @ (X_cam - t[None, :]).T).T
        pts_all.append(X_w.astype(np.float32))

    if not pts_all:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts_all, axis=0)


def compute_obb_from_points(points: np.ndarray) -> Dict:
    """Fit an oriented bounding box using PCA."""
    if points.shape[0] < 4:
        return None
    mean = points.mean(axis=0)
    centered = points - mean[None, :]
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    # Project points to eigenbasis
    proj = centered @ axes
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    extents = (maxs - mins) / 2.0
    center_local = (maxs + mins) / 2.0
    center_world = mean + axes @ center_local
    return {
        "center": center_world.astype(np.float32).tolist(),
        "extents": extents.astype(np.float32).tolist(),
        "R": axes.astype(np.float32).tolist(),
    }


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

    # Look for inventory.json first (Gemini pipeline), then fall back to YOLO labels
    # inventory.json is in seg/ directory (parent of seg/dataset)
    seg_dir = seg_dataset_dir.parent  # seg/dataset -> seg/
    inventory_path = seg_dir / "inventory.json"

    # For YOLO labels fallback
    valid_images_dir = seg_dataset_dir / "valid" / "images"
    valid_labels_dir = seg_dataset_dir / "valid" / "labels"
    room_img_path = valid_images_dir / "room.jpg"
    room_label_path = valid_labels_dir / "room.txt"

    if not room_img_path.is_file():
        print(f"[LAYOUT] WARNING: room image not found at {room_img_path}", file=sys.stderr)

    # Check for inventory.json (preferred) or YOLO labels (fallback)
    use_inventory = inventory_path.is_file()
    if use_inventory:
        print(f"[LAYOUT] Found inventory.json at {inventory_path}, will use Gemini bboxes")
    elif room_label_path.is_file():
        print(f"[LAYOUT] No inventory.json, falling back to YOLO labels at {room_label_path}")
    else:
        print(f"[LAYOUT] WARNING: Neither inventory.json nor room.txt found", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load DA3 geometry ----
    depth_all, conf_all, extr_all, intr_all, image_paths = load_da3_geom(geom_path)
    num_views = depth_all.shape[0]
    H, W = depth_all[0].shape
    print(f"[LAYOUT] Loaded {num_views} DA3 frames with shape {depth_all[0].shape}")

    # ---- Build fused room point cloud ----
    pts_world = fuse_multiview_point_clouds(depth_all, conf_all, intr_all, extr_all, conf_thresh=0.6, max_per_view=200000)
    if pts_world.shape[0] == 0:
        print("[LAYOUT] ERROR: No valid depth points; cannot compute layout.", file=sys.stderr)
        sys.exit(1)

    mins = pts_world.min(axis=0).tolist()
    maxs = pts_world.max(axis=0).tolist()
    print(f"[LAYOUT] Room fused AABB min={mins}, max={maxs}")

    room_box = {"min": mins, "max": maxs, "num_points": int(pts_world.shape[0])}

    # ---- Plane detection (floor/ceiling/walls) ----
    y_vals = pts_world[:, 1]
    floor_candidates = pts_world[y_vals <= np.percentile(y_vals, 20)]
    ceiling_candidates = pts_world[y_vals >= np.percentile(y_vals, 80)]

    floor_n, floor_d = fit_plane_ransac(floor_candidates, num_iters=300, threshold=0.03)
    if floor_n[1] < 0:
        floor_n = -floor_n
        floor_d = -floor_d

    ceiling_n, ceiling_d = fit_plane_ransac(ceiling_candidates, num_iters=300, threshold=0.03)
    if ceiling_n[1] < 0:
        ceiling_n = -ceiling_n
        ceiling_d = -ceiling_d

    off_floor = np.abs(pts_world @ floor_n + floor_d) > 0.2
    vertical_points = pts_world[off_floor]
    walls = []
    remaining = vertical_points
    for _ in range(2):
        if remaining.shape[0] < 200:
            break
        n, d = fit_plane_ransac(remaining, num_iters=250, threshold=0.05)
        if np.abs(np.dot(n, floor_n)) > 0.5:
            remaining = remaining[1:]
            continue
        dist = np.abs(remaining @ n + d)
        inliers = dist < 0.05
        walls.append(plane_to_extent(n, d, remaining[inliers]))
        remaining = remaining[~inliers]

    room_planes = {
        "floor": plane_to_extent(floor_n, floor_d, floor_candidates),
        "ceiling": plane_to_extent(ceiling_n, ceiling_d, ceiling_candidates),
        "walls": walls,
    }

    # ---- Load objects from inventory.json (preferred) or YOLO labels (fallback) ----
    if use_inventory:
        objects = parse_inventory_objects(inventory_path)
        print(f"[LAYOUT] Parsed {len(objects)} objects from inventory.json")
    else:
        objects = parse_yolo_labels(room_label_path, class_names=None)
        print(f"[LAYOUT] Parsed {len(objects)} objects from YOLO labels")

    objects_with_obb = 0
    objects_without_obb = 0
    for obj in objects:
        poly = obj.get("polygon") or []
        pts_obj = gather_object_points(poly, depth_all, conf_all, intr_all, extr_all)
        obb = compute_obb_from_points(pts_obj) if pts_obj.size else None
        center = obb.get("center") if obb else None
        obj["center3d"] = center
        obj["obb"] = obb
        obj["points_used"] = int(pts_obj.shape[0])

        obj_id = obj.get("id")
        if obb:
            objects_with_obb += 1
            print(f"[LAYOUT] obj_{obj_id}: OBB computed from {pts_obj.shape[0]} depth points, center={[round(c, 3) for c in center]}")
        else:
            objects_without_obb += 1
            print(f"[LAYOUT] obj_{obj_id}: No OBB (insufficient depth points: {pts_obj.shape[0]})")

    print(f"[LAYOUT] OBB summary: {objects_with_obb}/{len(objects)} objects have valid OBB data")

    # ---- Cameras ----
    cameras = []
    for i in range(num_views):
        cameras.append(
            {
                "id": i,
                "intrinsics": intr_all[i].tolist(),
                "extrinsics": extr_all[i].tolist(),
                "image_path": str(image_paths[i]),
            }
        )

    # ---- Build layout JSON ----
    layout = {
        "scene_id": scene_id,
        "camera": {  # legacy single-view fields kept for downstream compatibility
            "intrinsics": intr_all[0].tolist(),
            "extrinsics": extr_all[0].tolist(),
            "image_paths": image_paths.tolist(),
        },
        "camera_trajectory": cameras,
        "room_box": room_box,
        "room_planes": room_planes,
        "objects": objects,
    }

    layout_path = out_dir / "scene_layout.json"
    with layout_path.open("w") as f:
        json.dump(layout, f, indent=2)

    print(f"[LAYOUT] Wrote layout JSON to {layout_path}")
    print("[LAYOUT] Done.")


if __name__ == "__main__":
    main()

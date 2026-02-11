"""Harvest 3D-RE-GEN native outputs into the adapter-expected format.

Converts the native 3D-RE-GEN output directory structure into the format
expected by tools/regen3d_adapter/adapter.py:load_regen3d_output().

Native 3D-RE-GEN output:
    output/
        findings/fullSize/          # full-res segmentation masks
        findings/banana/prepped/    # inpainted object images
        3D/                         # Hunyuan3D GLB meshes per object
            object_name/textured_mesh.glb
        pointclouds/                # point clouds per object
            object_name.ply
        glb/                        # optimized scene GLBs with transforms
            object_name.glb
            scene/combined_scene.glb
        vggt/sparse/                # VGGT camera + point cloud
            0/
                cameras.txt
                points3D.txt
                images.txt
        pre_3D/camera.npz           # camera parameters
        masks/                      # binary masks per object

Adapter expected format:
    regen3d/
        scene_info.json
        objects/obj_0/mesh.glb, pose.json, bounds.json, material.json
        background/mesh.glb, pose.json, bounds.json
        camera/intrinsics.json, extrinsics.json
        depth/depth.exr
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HarvestResult:
    """Result of harvesting 3D-RE-GEN outputs."""
    target_dir: Path
    objects_count: int
    has_background: bool
    has_camera: bool
    has_depth: bool
    warnings: List[str]


def harvest_regen3d_native_output(
    native_dir: Path,
    target_dir: Path,
    scene_id: str,
    source_image_path: Optional[str] = None,
    environment_type: str = "generic",
) -> HarvestResult:
    """Convert 3D-RE-GEN native output to adapter-expected format.

    Args:
        native_dir: Path to 3D-RE-GEN's output directory.
        target_dir: Path to write adapter-format output (the regen3d/ dir).
        scene_id: Scene identifier.
        source_image_path: Path to the original input image.
        environment_type: Environment type hint.

    Returns:
        HarvestResult with statistics.
    """
    native_dir = Path(native_dir)
    target_dir = Path(target_dir)
    warnings: List[str] = []

    # Create target directory structure
    objects_dir = target_dir / "objects"
    background_dir = target_dir / "background"
    camera_dir = target_dir / "camera"
    depth_dir = target_dir / "depth"

    for d in [objects_dir, background_dir, camera_dir, depth_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # --- Harvest objects ---
    objects_count, object_labels = _harvest_objects(native_dir, objects_dir, warnings)

    # --- Harvest background ---
    has_background = _harvest_background(native_dir, background_dir, warnings)

    # --- Harvest camera ---
    has_camera = _harvest_camera(native_dir, camera_dir, warnings)

    # --- Harvest depth ---
    has_depth = _harvest_depth(native_dir, depth_dir, warnings)

    # --- Write scene_info.json ---
    image_size = _detect_image_size(native_dir, source_image_path)
    scene_info = {
        "scene_id": scene_id,
        "image_size": list(image_size),
        "coordinate_frame": "y_up",
        "meters_per_unit": 1.0,
        "confidence": 0.9,
        "version": "1.0",
        "environment_type": environment_type,
        "reconstruction_method": "3d-re-gen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_image_path:
        scene_info["source_image_path"] = str(source_image_path)
    if object_labels:
        scene_info["object_labels"] = object_labels

    (target_dir / "scene_info.json").write_text(json.dumps(scene_info, indent=2))

    for w in warnings:
        logger.warning(f"[HARVEST] {w}")

    return HarvestResult(
        target_dir=target_dir,
        objects_count=objects_count,
        has_background=has_background,
        has_camera=has_camera,
        has_depth=has_depth,
        warnings=warnings,
    )


def _parse_label_from_filename(filename: str) -> Optional[str]:
    """Extract semantic label from 3D-RE-GEN GLB filename.

    3D-RE-GEN names GLBs as '{label}__{pixel_coords}.glb',
    e.g. 'bed_frame__(619, 582).glb' → 'bed_frame'.
    """
    if "__" in filename:
        return filename.split("__")[0].strip()
    # No pixel-coord suffix — use the whole stem as label
    return filename


def _harvest_objects(
    native_dir: Path, objects_dir: Path, warnings: List[str]
) -> Tuple[int, Dict[str, str]]:
    """Harvest per-object GLBs and create metadata files.

    Returns:
        Tuple of (object_count, label_map) where label_map maps obj_id → label.
    """
    # 3D-RE-GEN stores meshes in output_folder_hy (3D/) or glb/
    # The optimized GLBs with correct poses are in glb/
    glb_dir = native_dir / "glb"
    hy_dir = native_dir / "3D"
    masks_dir = native_dir / "findings" / "fullSize"

    # Collect object names from the optimized GLB directory
    object_glbs: Dict[str, Path] = {}

    if glb_dir.is_dir():
        for item in sorted(glb_dir.iterdir()):
            if item.is_file() and item.suffix == ".glb" and item.stem != "scene":
                object_glbs[item.stem] = item
            elif item.is_dir() and item.name != "scene":
                # Check for GLB inside subdirectory
                for glb_file in item.glob("*.glb"):
                    object_glbs[item.name] = glb_file
                    break

    # Fall back to Hunyuan3D output directory
    if not object_glbs and hy_dir.is_dir():
        for item in sorted(hy_dir.iterdir()):
            if item.is_dir():
                # Support both legacy names and Hunyuan2.1 naming:
                #   {name}.glb, {name}_shape.glb
                hy_candidates = [
                    f"{item.name}.glb",
                    f"{item.name}_shape.glb",
                    "textured_mesh.glb",
                    "mesh.glb",
                    "model.glb",
                ]
                for name in hy_candidates:
                    candidate = item / name
                    if candidate.is_file():
                        object_glbs[item.name] = candidate
                        break

    if not object_glbs:
        warnings.append("No object GLBs found in glb/ or 3D/ directories")
        return 0, {}

    # Load transforms from the combined scene or individual optimization results
    transforms = _load_object_transforms(native_dir)

    missing_or_invalid: List[str] = []
    for obj_name in sorted(object_glbs):
        matrix = transforms.get(obj_name)
        if matrix is None:
            missing_or_invalid.append(obj_name)
            continue
        if np.asarray(matrix).shape != (4, 4):
            missing_or_invalid.append(obj_name)

    if missing_or_invalid:
        # 3D-RE-GEN bakes transforms directly into GLBs when no separate
        # transform files are emitted. Use identity transforms — the objects
        # in glb/ are already in world-space poses.
        if len(missing_or_invalid) == len(object_glbs):
            logger.warning(
                "[HARVEST] No transform files found — using identity transforms "
                "(3D-RE-GEN bakes poses into GLBs)"
            )
            for obj_name in missing_or_invalid:
                transforms[obj_name] = np.eye(4)
            missing_or_invalid.clear()
        else:
            joined = ", ".join(sorted(missing_or_invalid))
            raise ValueError(
                "Missing valid 4x4 transforms for reconstructed objects: "
                f"{joined}. Ensure 3D-RE-GEN emitted glb/scene/transforms.json "
                "or per-object transform.json files."
            )

    label_map: Dict[str, str] = {}

    for idx, (obj_name, glb_path) in enumerate(sorted(object_glbs.items())):
        obj_id = f"obj_{idx}"
        obj_dir = objects_dir / obj_id
        obj_dir.mkdir(parents=True, exist_ok=True)

        # Extract semantic label from GLB filename (e.g. "bed_frame__(619, 582)" → "bed_frame")
        label = _parse_label_from_filename(obj_name)
        if label:
            label_map[obj_id] = label
            label_data = {"label": label, "original_filename": glb_path.name}
            (obj_dir / "label.json").write_text(json.dumps(label_data, indent=2))

        # Copy mesh
        shutil.copy2(glb_path, obj_dir / "mesh.glb")

        # Write pose.json
        transform = transforms[obj_name]
        pose = _build_pose_json(transform, obj_name)
        (obj_dir / "pose.json").write_text(json.dumps(pose, indent=2))

        # Compute and write bounds.json
        bounds = _compute_bounds(glb_path, transform)
        (obj_dir / "bounds.json").write_text(json.dumps(bounds, indent=2))

        # Write material.json
        material = _extract_material(native_dir, obj_name)
        (obj_dir / "material.json").write_text(json.dumps(material, indent=2))

        # Copy segmentation mask if available
        if masks_dir.is_dir():
            for mask_name in [f"{obj_name}.png", f"{obj_name}_mask.png"]:
                mask_path = masks_dir / mask_name
                if mask_path.is_file():
                    shutil.copy2(mask_path, obj_dir / "segmentation.png")
                    break

        logger.info(
            f"[HARVEST] Object {obj_id} ({obj_name}): label={label}, "
            f"mesh={glb_path.name}, has_transform={transform is not None}"
        )

    return len(object_glbs), label_map


def _harvest_background(
    native_dir: Path, background_dir: Path, warnings: List[str]
) -> bool:
    """Harvest background mesh."""
    # Check for background mesh in various locations
    candidates = [
        native_dir / "glb" / "scene" / "combined_scene.glb",
        native_dir / "pointclouds" / "meshed" / "background.glb",
        native_dir / "pointclouds" / "meshed" / "background.obj",
    ]

    bg_mesh = None
    for candidate in candidates:
        if candidate.is_file():
            bg_mesh = candidate
            break

    if bg_mesh is None:
        warnings.append("No background mesh found")
        return False

    size_kb = bg_mesh.stat().st_size / 1024
    logger.info(
        f"[HARVEST] Background mesh source: {bg_mesh.name} ({size_kb:.1f} KB)"
    )

    # Copy mesh
    target_name = "mesh.glb" if bg_mesh.suffix == ".glb" else f"mesh{bg_mesh.suffix}"
    shutil.copy2(bg_mesh, background_dir / target_name)

    # Write identity pose for background
    pose = {
        "transform_matrix": np.eye(4).tolist(),
        "translation": [0.0, 0.0, 0.0],
        "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "confidence": 1.0,
        "is_floor_contact": True,
    }
    (background_dir / "pose.json").write_text(json.dumps(pose, indent=2))

    # Compute bounds from actual mesh geometry; fall back to scene envelope
    bounds = _compute_bounds(background_dir / target_name, transform=None)
    if bounds["min"] == [-0.5, -0.5, -0.5]:
        # trimesh unavailable or failed — use default scene envelope
        bounds = {
            "min": [-5.0, 0.0, -5.0],
            "max": [5.0, 3.0, 5.0],
            "center": [0.0, 1.5, 0.0],
            "size": [10.0, 3.0, 10.0],
        }
    (background_dir / "bounds.json").write_text(json.dumps(bounds, indent=2))

    return True


def _harvest_camera(
    native_dir: Path, camera_dir: Path, warnings: List[str]
) -> bool:
    """Harvest camera parameters from VGGT output."""
    has_camera = False

    # Try VGGT COLMAP-format cameras
    vggt_sparse = native_dir / "vggt" / "sparse" / "0"
    cameras_txt = vggt_sparse / "cameras.txt" if vggt_sparse.is_dir() else None

    if cameras_txt and cameras_txt.is_file():
        intrinsics = _parse_colmap_cameras(cameras_txt)
        if intrinsics:
            (camera_dir / "intrinsics.json").write_text(
                json.dumps(intrinsics, indent=2)
            )
            has_camera = True

    # Try camera.npz
    camera_npz = native_dir / "pre_3D" / "camera.npz"
    if camera_npz.is_file():
        try:
            data = np.load(str(camera_npz), allow_pickle=True)

            # VGGT format: focal + image_size + extrinsic (4x4)
            if "focal" in data and "image_size" in data and not has_camera:
                focal = float(data["focal"])
                img_size = data["image_size"]  # [width, height]
                w, h = int(img_size[0]), int(img_size[1])
                cx, cy = w / 2.0, h / 2.0
                intrinsics = {
                    "matrix": [
                        [focal, 0.0, cx],
                        [0.0, focal, cy],
                        [0.0, 0.0, 1.0],
                    ],
                    "width": w,
                    "height": h,
                }
                (camera_dir / "intrinsics.json").write_text(
                    json.dumps(intrinsics, indent=2)
                )
                has_camera = True

            if "extrinsic" in data:
                extr = np.array(data["extrinsic"])
                if extr.shape == (4, 4):
                    extrinsics = {"matrix": extr.tolist()}
                    (camera_dir / "extrinsics.json").write_text(
                        json.dumps(extrinsics, indent=2)
                    )

            # Legacy format: K + R + t
            if "K" in data and not has_camera:
                K = data["K"].tolist()
                intrinsics = {
                    "matrix": K if isinstance(K[0], list) else [K],
                    "width": 1920,
                    "height": 1080,
                }
                (camera_dir / "intrinsics.json").write_text(
                    json.dumps(intrinsics, indent=2)
                )
                has_camera = True

            if "R" in data and "t" in data:
                R = np.array(data["R"])
                t = np.array(data["t"])
                if R.ndim >= 2:
                    R_mat = R[:3, :3] if R.shape[0] >= 3 else R
                    t_vec = t.flatten()[:3] if t.size >= 3 else t.flatten()
                    extr_matrix = np.eye(4)
                    extr_matrix[:3, :3] = R_mat
                    extr_matrix[:3, 3] = t_vec
                    extrinsics = {"matrix": extr_matrix.tolist()}
                    (camera_dir / "extrinsics.json").write_text(
                        json.dumps(extrinsics, indent=2)
                    )
        except Exception as exc:
            warnings.append(f"Failed to parse camera.npz: {exc}")

    if not has_camera:
        # Write default camera intrinsics
        intrinsics = {
            "matrix": [
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "width": 1920,
            "height": 1080,
        }
        (camera_dir / "intrinsics.json").write_text(
            json.dumps(intrinsics, indent=2)
        )
        warnings.append("Using default camera intrinsics (no VGGT data found)")
        has_camera = True

    return has_camera


def _harvest_depth(
    native_dir: Path, depth_dir: Path, warnings: List[str]
) -> bool:
    """Harvest depth map."""
    candidates = [
        native_dir / "findings" / "depth.png",
        native_dir / "findings" / "depth.exr",
        native_dir / "pre_3D" / "depth.exr",
        native_dir / "pre_3D" / "depth.png",
    ]

    for candidate in candidates:
        if candidate.is_file():
            target = depth_dir / f"depth{candidate.suffix}"
            shutil.copy2(candidate, target)
            return True

    warnings.append("No depth map found")
    return False


def _load_object_transforms(
    native_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load per-object transform matrices from optimization outputs.

    3D-RE-GEN's scene optimization stores transforms as part of the
    differentiable rendering output. We try multiple sources:
    1. Per-object transform files in glb/ directory
    2. Combined scene file parsing
    3. Fall back to identity transforms
    """
    transforms: Dict[str, np.ndarray] = {}

    # Check for per-object transform JSON files
    glb_dir = native_dir / "glb"
    if glb_dir.is_dir():
        for item in glb_dir.iterdir():
            if item.is_dir() and item.name != "scene":
                transform_file = item / "transform.json"
                if transform_file.is_file():
                    try:
                        data = json.loads(transform_file.read_text())
                        matrix = np.array(data.get("matrix", np.eye(4).tolist()))
                        transforms[item.name] = matrix
                    except Exception:
                        pass

    # Check for a scene-level transforms file
    transforms_file = native_dir / "glb" / "scene" / "transforms.json"
    if transforms_file.is_file():
        try:
            all_transforms = json.loads(transforms_file.read_text())
            for obj_name, matrix_data in all_transforms.items():
                if obj_name not in transforms:
                    transforms[obj_name] = np.array(matrix_data)
        except Exception:
            pass

    # Try to extract from camera.npz (some configs store object poses)
    camera_npz = native_dir / "pre_3D" / "camera.npz"
    if camera_npz.is_file() and not transforms:
        try:
            data = np.load(str(camera_npz), allow_pickle=True)
            if "object_transforms" in data:
                obj_t = data["object_transforms"].item()
                if isinstance(obj_t, dict):
                    for k, v in obj_t.items():
                        if k not in transforms:
                            transforms[k] = np.array(v)
        except Exception:
            pass

    return transforms


def _build_pose_json(
    transform: Optional[np.ndarray], obj_name: str
) -> Dict[str, Any]:
    """Build a pose.json dict from a transform matrix."""
    if transform is not None:
        matrix = np.array(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            matrix = np.eye(4, dtype=np.float64)
    else:
        matrix = np.eye(4, dtype=np.float64)

    # Decompose transform
    translation = matrix[:3, 3].tolist()

    # Extract scale from rotation columns
    col_norms = [np.linalg.norm(matrix[:3, i]) for i in range(3)]
    scale = [float(n) for n in col_norms]

    # Extract rotation (normalize columns)
    rot = matrix[:3, :3].copy()
    for i in range(3):
        if col_norms[i] > 1e-8:
            rot[:, i] /= col_norms[i]

    # Convert rotation matrix to quaternion [w, x, y, z]
    quat = _rotation_matrix_to_quaternion(rot)

    # Detect floor contact (object near y=0)
    is_floor_contact = abs(translation[1]) < 0.1 if len(translation) >= 2 else False

    return {
        "transform_matrix": matrix.tolist(),
        "translation": translation,
        "rotation_quaternion": quat,
        "scale": scale,
        "confidence": 0.9,
        "is_floor_contact": is_floor_contact,
    }


def _rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    # Normalize
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 1e-8:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return [float(w), float(x), float(y), float(z)]


def _compute_bounds(
    glb_path: Path, transform: Optional[np.ndarray]
) -> Dict[str, Any]:
    """Compute axis-aligned bounding box for an object.

    Tries to use trimesh for accurate bounds. Falls back to default
    unit-cube bounds if trimesh is not available.
    """
    try:
        import trimesh
        mesh = trimesh.load(str(glb_path), force="mesh")
        if hasattr(mesh, "bounds") and mesh.bounds is not None:
            bounds_min = mesh.bounds[0].tolist()
            bounds_max = mesh.bounds[1].tolist()
            center = ((mesh.bounds[0] + mesh.bounds[1]) / 2).tolist()
            size = (mesh.bounds[1] - mesh.bounds[0]).tolist()

            # If we have a world transform, apply it to get world-space bounds
            if transform is not None:
                verts = mesh.vertices
                T = np.array(transform)
                ones = np.ones((verts.shape[0], 1))
                verts_h = np.hstack([verts, ones])
                verts_world = (T @ verts_h.T).T[:, :3]
                bounds_min = verts_world.min(axis=0).tolist()
                bounds_max = verts_world.max(axis=0).tolist()
                center = ((verts_world.min(axis=0) + verts_world.max(axis=0)) / 2).tolist()
                size = (verts_world.max(axis=0) - verts_world.min(axis=0)).tolist()

            return {
                "min": bounds_min,
                "max": bounds_max,
                "center": center,
                "size": size,
            }
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(f"[HARVEST] trimesh bounds failed for {glb_path}: {exc}")

    # Default unit-cube bounds
    return {
        "min": [-0.5, -0.5, -0.5],
        "max": [0.5, 0.5, 0.5],
        "center": [0.0, 0.0, 0.0],
        "size": [1.0, 1.0, 1.0],
    }


def _extract_material(
    native_dir: Path, obj_name: str
) -> Dict[str, Any]:
    """Extract PBR material parameters from Marigold appearance maps."""
    material = {
        "base_color": [0.8, 0.8, 0.8],
        "metallic": 0.0,
        "roughness": 0.5,
        "material_type": "generic",
    }

    # Check for Marigold material maps
    marigold_dir = native_dir / "findings" / "scene_marigold" / obj_name
    if not marigold_dir.is_dir():
        marigold_dir = native_dir / "images_marigold_base" / obj_name

    if marigold_dir.is_dir():
        albedo = marigold_dir / "albedo_map.png"
        roughness = marigold_dir / "roughness_map.png"
        metallic = marigold_dir / "metallic_map.png"
        normal = marigold_dir / "normal_map.png"

        if albedo.is_file():
            material["albedo_map_path"] = str(albedo)
        if roughness.is_file():
            # Could sample average roughness from the map, but for now use default
            material["roughness"] = 0.5
        if metallic.is_file():
            material["metallic"] = 0.1
        if normal.is_file():
            material["normal_map_path"] = str(normal)

    return material


def _parse_colmap_cameras(cameras_txt: Path) -> Optional[Dict[str, Any]]:
    """Parse COLMAP cameras.txt to extract intrinsics."""
    try:
        lines = cameras_txt.read_text().strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                # COLMAP format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])

                if model == "SIMPLE_PINHOLE" and len(parts) >= 7:
                    f = float(parts[4])
                    cx = float(parts[5])
                    cy = float(parts[6])
                    return {
                        "matrix": [
                            [f, 0.0, cx],
                            [0.0, f, cy],
                            [0.0, 0.0, 1.0],
                        ],
                        "width": width,
                        "height": height,
                    }
                elif model == "PINHOLE" and len(parts) >= 8:
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])
                    return {
                        "matrix": [
                            [fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0],
                        ],
                        "width": width,
                        "height": height,
                    }
    except Exception as exc:
        logger.warning(f"[HARVEST] Failed to parse cameras.txt: {exc}")

    return None


def _detect_image_size(
    native_dir: Path, source_image_path: Optional[str]
) -> Tuple[int, int]:
    """Detect image dimensions from source image or config."""
    if source_image_path:
        try:
            from PIL import Image
            img = Image.open(source_image_path)
            return img.size  # (width, height)
        except Exception:
            pass

    # Try to detect from segmentation output
    findings_dir = native_dir / "findings"
    if findings_dir.is_dir():
        for img_file in findings_dir.glob("*.png"):
            try:
                from PIL import Image
                img = Image.open(img_file)
                return img.size
            except Exception:
                continue

    return (1920, 1080)  # default

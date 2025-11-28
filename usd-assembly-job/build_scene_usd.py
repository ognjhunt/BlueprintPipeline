#!/usr/bin/env python3
"""
build_scene_usd.py - Build a USD scene from layout and assets JSON.

Creates a scene.usda file that:
  - Defines the world coordinate system
  - Contains camera definitions with extrinsics/intrinsics
  - Contains object definitions with transforms and references to converted USDZ assets
  - Includes room plane definitions for physics/collision

This script does NOT perform GLB->USDZ conversion; that's handled by assemble_scene.py
which orchestrates the full pipeline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
except ImportError:
    print("ERROR: usd-core is required. Install with: pip install usd-core")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str, prefix: str = "prim") -> str:
    """Convert a name to a valid USD identifier."""
    if not name:
        return prefix

    result = ""
    for c in name:
        if c.isalnum() or c == "_":
            result += c
        else:
            result += "_"

    if result and result[0].isdigit():
        result = prefix + "_" + result

    return result or prefix


def safe_path_join(root: Path, rel: str) -> Path:
    """Join a relative path against a root, stripping any leading '/'."""
    rel_path = rel.lstrip("/")
    return root / rel_path


# -----------------------------------------------------------------------------
# Transform Utilities
# -----------------------------------------------------------------------------


def matrix_from_obb(obb: Dict) -> Optional[np.ndarray]:
    """
    Build a 4x4 transform matrix from an OBB record.

    obb["R"]      -> 3x3 rotation matrix
    obb["center"] -> 3D center

    Returns a row-major 4x4 matrix.
    """
    center = obb.get("center")
    R = obb.get("R")
    if center is None or R is None:
        return None

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(R, dtype=np.float64)
    T[:3, 3] = np.array(center, dtype=np.float64)
    return T


def numpy_to_gf_matrix(m: np.ndarray) -> Gf.Matrix4d:
    """Convert a numpy 4x4 matrix to Gf.Matrix4d."""
    m = np.array(m, dtype=np.float64).reshape(4, 4)
    return Gf.Matrix4d(*m.flatten().tolist())


# -----------------------------------------------------------------------------
# Metadata Loading
# -----------------------------------------------------------------------------


def load_object_metadata(root: Path, obj: Dict, assets_prefix: str) -> Optional[dict]:
    """
    Load per-object metadata if present.

    Resolution order:
      1. obj["metadata_path"] if present (bucket-relative)
      2. metadata.json next to the asset_path
      3. assets_prefix/obj_{id}/metadata.json (standard layout)
    """
    metadata_rel = obj.get("metadata_path")
    if metadata_rel:
        candidate = safe_path_join(root, metadata_rel)
        if candidate.is_file():
            return json.loads(candidate.read_text())

    asset_path = obj.get("asset_path")
    if asset_path:
        asset_dir = safe_path_join(root, asset_path).parent
        candidate = asset_dir / "metadata.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())

    oid = obj.get("id")
    if oid is not None:
        obj_dir = safe_path_join(root, f"{assets_prefix}/obj_{oid}")
        candidate = obj_dir / "metadata.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())

    return None


def alignment_from_metadata(metadata: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract alignment information from metadata.

    Returns:
      - translation (3,) that recenters mesh to origin, if available
      - half_extents (3,) in mesh-local space, if available
    """
    if not metadata:
        return None, None

    mesh_bounds = metadata.get("mesh_bounds") or {}
    export_bounds = (
        mesh_bounds.get("export") or mesh_bounds.get("bounds") or mesh_bounds
    )
    center = export_bounds.get("center")
    size = export_bounds.get("size")

    translation = None
    half_extents = None

    if center is not None:
        translation = -np.array(center, dtype=np.float64)

    if size is not None:
        half_extents = 0.5 * np.array(size, dtype=np.float64)

    return translation, half_extents


# -----------------------------------------------------------------------------
# Asset Path Resolution
# -----------------------------------------------------------------------------


def resolve_usdz_asset_path(
    root: Path,
    assets_prefix: str,
    usd_prefix: str,
    oid: Any,
    asset_path: Optional[str] = None,
) -> Optional[str]:
    """
    Find the USDZ asset for an object and return a path relative to the USD output dir.

    Checks for:
      1. Existing USDZ next to specified asset_path
      2. Standard location: assets_prefix/obj_{id}/model.usdz
      3. Alternative names: asset.usdz

    Returns path relative to usd_prefix, or None if not found.
    """
    stage_dir = root / usd_prefix

    # Check 1: USDZ next to specified asset_path
    if asset_path:
        glb_path = safe_path_join(root, asset_path)
        usdz_path = glb_path.with_suffix(".usdz")
        if usdz_path.is_file():
            rel = os.path.relpath(usdz_path, stage_dir)
            return rel.replace("\\", "/")

    # Check 2: Standard obj_N directory
    obj_dir = root / assets_prefix / f"obj_{oid}"

    candidate_names = [
        "model.usdz",
        "asset.usdz",
        "model.usda",
        "model.usd",
        "model.usdc",
    ]

    for name in candidate_names:
        candidate = obj_dir / name
        if candidate.is_file():
            rel = os.path.relpath(candidate, stage_dir)
            return rel.replace("\\", "/")

    return None


# -----------------------------------------------------------------------------
# Scene Building with USD API
# -----------------------------------------------------------------------------


class SceneBuilder:
    """Builds a USD scene from layout and assets data."""

    def __init__(
        self,
        stage: Usd.Stage,
        root: Path,
        assets_prefix: str,
        usd_prefix: str,
    ):
        self.stage = stage
        self.root = root
        self.assets_prefix = assets_prefix
        self.usd_prefix = usd_prefix

        # Configure stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Create world root
        self.world = UsdGeom.Xform.Define(stage, "/World")

    def add_room_planes(self, room_planes: Dict) -> None:
        """Add room plane definitions for physics."""
        room_scope = UsdGeom.Scope.Define(self.stage, "/World/Room")
        prim = room_scope.GetPrim()

        # Floor equation
        floor = room_planes.get("floor") or {}
        floor_eq = floor.get("equation", [0, 1, 0, 0])
        prim.CreateAttribute("floorEquation", Sdf.ValueTypeNames.Double4).Set(
            Gf.Vec4d(*floor_eq)
        )

        # Ceiling equation
        ceiling = room_planes.get("ceiling") or {}
        ceiling_eq = ceiling.get("equation", [0, -1, 0, 0])
        prim.CreateAttribute("ceilingEquation", Sdf.ValueTypeNames.Double4).Set(
            Gf.Vec4d(*ceiling_eq)
        )

        # Wall equations
        walls = room_planes.get("walls", [])
        for idx, wall in enumerate(walls):
            wall_eq = wall.get("equation", [1, 0, 0, 0])
            prim.CreateAttribute(f"wall{idx}Equation", Sdf.ValueTypeNames.Double4).Set(
                Gf.Vec4d(*wall_eq)
            )

    def add_cameras(self, cameras: List[Dict]) -> None:
        """Add camera definitions."""
        cameras_scope = UsdGeom.Scope.Define(self.stage, "/World/Cameras")

        for cam in cameras:
            cam_id = cam.get("id", 0)
            cam_path = f"/World/Cameras/cam_{cam_id}"
            cam_xform = UsdGeom.Xform.Define(self.stage, cam_path)
            prim = cam_xform.GetPrim()

            # Camera extrinsics
            extr = cam.get("extrinsics")
            if extr:
                mat = np.eye(4, dtype=np.float64)
                ext = np.array(extr, dtype=np.float64)
                if ext.shape == (3, 4):
                    mat[:3, :4] = ext
                prim.CreateAttribute("cameraExtrinsics", Sdf.ValueTypeNames.Matrix4d).Set(
                    numpy_to_gf_matrix(mat)
                )

            # Camera intrinsics
            intr = cam.get("intrinsics")
            if intr:
                fx = intr[0][0]
                fy = intr[1][1]
                cx = intr[0][2]
                prim.CreateAttribute("intrinsics", Sdf.ValueTypeNames.Double3).Set(
                    Gf.Vec3d(fx, fy, cx)
                )

            # Image path
            if cam.get("image_path"):
                prim.CreateAttribute("imagePath", Sdf.ValueTypeNames.String).Set(
                    cam["image_path"]
                )

    def add_object(self, obj: Dict, layout_objects: Dict[int, Dict]) -> None:
        """Add a single object to the scene."""
        oid = obj.get("id")
        is_interactive = obj.get("type") == "interactive"

        # Merge layout data
        merged = dict(obj)
        layout_obj = layout_objects.get(oid)
        if layout_obj:
            for key in ("obb", "center3d"):
                if key in layout_obj:
                    merged[key] = layout_obj[key]

        # Determine asset path
        asset_path = obj.get("asset_path")
        asset_type = obj.get("asset_type")

        if not asset_path:
            if is_interactive:
                base_dir = obj.get("interactive_output") or f"{self.assets_prefix}/interactive/obj_{oid}"
                asset_path = f"{base_dir}/obj_{oid}.urdf"
                asset_type = asset_type or "urdf"
            else:
                # Default to GLB, will be converted to USDZ
                base_dir = f"{self.assets_prefix}/obj_{oid}"
                glb_rel = f"{base_dir}/asset.glb"
                asset_path = glb_rel
                asset_type = asset_type or "glb"
        else:
            # Infer type from extension
            if not asset_type and isinstance(asset_path, str):
                lower = asset_path.lower()
                if lower.endswith(".urdf"):
                    asset_type = "urdf"
                elif lower.endswith((".usdz", ".usd", ".usda", ".usdc")):
                    asset_type = "usd"
                elif lower.endswith((".glb", ".gltf")):
                    asset_type = "gltf"
                else:
                    asset_type = "unknown"

        # Load metadata
        metadata = load_object_metadata(
            self.root,
            {"id": oid, "asset_path": asset_path, "metadata_path": obj.get("metadata_path")},
            self.assets_prefix,
        )
        translation, mesh_half_extents = alignment_from_metadata(metadata or {})

        # Create object prim
        obj_path = f"/World/Objects/obj_{oid}"
        obj_xform = UsdGeom.Xform.Define(self.stage, obj_path)
        prim = obj_xform.GetPrim()

        # Set attributes
        prim.CreateAttribute("interactive", Sdf.ValueTypeNames.Bool).Set(is_interactive)
        if asset_type:
            prim.CreateAttribute("assetType", Sdf.ValueTypeNames.String).Set(asset_type)
        if asset_path:
            prim.CreateAttribute("asset_path", Sdf.ValueTypeNames.String).Set(asset_path)

        if is_interactive:
            manifest = obj.get("interactive_output") or f"{self.assets_prefix}/interactive/obj_{oid}"
            prim.CreateAttribute("urdf_manifest", Sdf.ValueTypeNames.String).Set(
                f"{manifest}/interactive_manifest.json"
            )

        # Build transform
        xform = np.eye(4, dtype=np.float64)
        applied = False

        # 1) Recenter mesh
        if translation is not None:
            T_align = np.eye(4, dtype=np.float64)
            T_align[:3, 3] = translation
            xform = T_align @ xform
            applied = True

        # 2) Scale to match OBB
        obb = merged.get("obb")
        obb_extents = None
        if obb:
            obb_extents = obb.get("extents")
        if obb_extents is not None and mesh_half_extents is not None:
            mesh_half_extents = np.where(mesh_half_extents == 0, 1.0, mesh_half_extents)
            scale_vec = np.array(obb_extents, dtype=np.float64) / mesh_half_extents
            S = np.eye(4, dtype=np.float64)
            S[0, 0], S[1, 1], S[2, 2] = scale_vec
            xform = S @ xform
            applied = True

        # 3) Place in world space via OBB pose
        if obb:
            T = matrix_from_obb(obb)
            if T is not None:
                xform = T @ xform
                applied = True

            extents = obb.get("extents")
            if extents:
                prim.CreateAttribute("halfExtents", Sdf.ValueTypeNames.Double3).Set(
                    Gf.Vec3d(*extents)
                )

        # Apply transform
        if applied:
            obj_xform.MakeMatrixXform().Set(numpy_to_gf_matrix(xform))

        # Mesh metadata
        if metadata:
            mesh_bounds = metadata.get("mesh_bounds") or {}
            export_bounds = mesh_bounds.get("export") or mesh_bounds.get("bounds") or mesh_bounds
            center = export_bounds.get("center")
            size = export_bounds.get("size")
            if center:
                prim.CreateAttribute("meshCenter", Sdf.ValueTypeNames.Double3).Set(
                    Gf.Vec3d(*center)
                )
            if size:
                prim.CreateAttribute("meshSize", Sdf.ValueTypeNames.Double3).Set(
                    Gf.Vec3d(*size)
                )

        if obj.get("class_name"):
            prim.CreateAttribute("className", Sdf.ValueTypeNames.String).Set(
                obj["class_name"]
            )
        if obj.get("pipeline"):
            prim.CreateAttribute("pipeline", Sdf.ValueTypeNames.String).Set(
                obj["pipeline"]
            )

        # Add geometry reference
        usdz_rel = resolve_usdz_asset_path(
            self.root, self.assets_prefix, self.usd_prefix, oid, asset_path
        )

        if usdz_rel:
            geom_path = f"{obj_path}/Geom"
            geom_xform = UsdGeom.Xform.Define(self.stage, geom_path)
            geom_prim = geom_xform.GetPrim()
            geom_prim.GetReferences().AddReference(usdz_rel)
            print(f"[USD] obj_{oid}: referenced {usdz_rel}")
        else:
            # No USDZ found - record pending conversion info
            if asset_path and (asset_path.endswith(".glb") or asset_path.endswith(".gltf")):
                prim.CreateAttribute("pendingConversion", Sdf.ValueTypeNames.Bool).Set(True)
                print(f"[USD] obj_{oid}: marked for GLB->USDZ conversion ({asset_path})")

    def add_objects(self, objects: List[Dict], layout_objects: Dict[int, Dict]) -> None:
        """Add all objects to the scene."""
        objects_scope = UsdGeom.Scope.Define(self.stage, "/World/Objects")

        for obj in objects:
            try:
                self.add_object(obj, layout_objects)
            except Exception as e:
                print(f"[WARN] Failed to add object {obj.get('id')}: {e}")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def build_scene(
    layout_path: Path,
    assets_path: Path,
    output_path: Path,
    root: Path,
    assets_prefix: str,
    usd_prefix: str,
) -> Tuple[Usd.Stage, List[Dict]]:
    """
    Build a USD scene from layout and assets JSON.

    Returns:
        Tuple of (stage, objects_list) for further processing
    """
    # Load input files
    layout = load_json(layout_path)
    scene_assets = load_json(assets_path)

    # Extract data
    cameras = layout.get("camera_trajectory") or []
    room_planes = layout.get("room_planes", {})
    layout_objects = {o.get("id"): o for o in layout.get("objects", [])}

    # Merge objects from assets with layout data
    objects: List[Dict] = []
    for obj in scene_assets.get("objects", []):
        merged = dict(obj)
        layout_obj = layout_objects.get(obj.get("id"))
        if layout_obj:
            for key in ("obb", "center3d"):
                if key in layout_obj:
                    merged[key] = layout_obj[key]
        objects.append(merged)

    # Create stage
    ensure_dir(output_path.parent)
    stage = Usd.Stage.CreateNew(str(output_path))

    # Build scene
    builder = SceneBuilder(stage, root, assets_prefix, usd_prefix)
    builder.add_room_planes(room_planes)
    builder.add_cameras(cameras)
    builder.add_objects(scene_assets.get("objects", []), layout_objects)

    # Save stage
    stage.GetRootLayer().Save()
    print(f"[USD] Wrote stage to {output_path}")
    print(f"[USD] Cameras: {len(cameras)} | Objects: {len(objects)}")

    return stage, objects


def main() -> None:
    """CLI entry point."""
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    layout_prefix = os.getenv("LAYOUT_PREFIX")
    assets_prefix = os.getenv("ASSETS_PREFIX")
    usd_prefix = os.getenv("USD_PREFIX") or assets_prefix

    if not layout_prefix or not assets_prefix:
        print("[USD] LAYOUT_PREFIX and ASSETS_PREFIX are required", file=sys.stderr)
        sys.exit(1)

    root = Path("/mnt/gcs")

    layout_path = root / layout_prefix / "scene_layout_scaled.json"
    assets_path = root / assets_prefix / "scene_assets.json"
    stage_path = root / usd_prefix / "scene.usda"

    build_scene(
        layout_path=layout_path,
        assets_path=assets_path,
        output_path=stage_path,
        root=root,
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
    )


if __name__ == "__main__":
    main()
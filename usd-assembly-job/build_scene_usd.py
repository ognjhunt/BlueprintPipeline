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
from typing import Any, Dict, List, Optional, Tuple, Union

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
        "mesh.usdz",
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
# Scene Shell Geometry Creation
# -----------------------------------------------------------------------------


def create_plane_mesh_points(
    normal: np.ndarray,
    d: float,
    room_box: Dict,
    margin: float = 0.1,
) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    """
    Create a quad mesh for a plane bounded by room_box.

    Args:
        normal: Plane normal [nx, ny, nz]
        d: Plane equation constant (n·x + d = 0)
        room_box: {"min": [x,y,z], "max": [x,y,z]}
        margin: Extra margin around the plane

    Returns:
        (points, face_indices) for the quad
    """
    normal = np.array(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    room_min = np.array(room_box.get("min", [-5, -5, -5]), dtype=np.float64)
    room_max = np.array(room_box.get("max", [5, 5, 5]), dtype=np.float64)
    room_center = (room_min + room_max) / 2
    room_size = room_max - room_min + 2 * margin

    # Find two vectors perpendicular to normal
    up = np.array([0, 1, 0], dtype=np.float64)
    if abs(np.dot(normal, up)) > 0.99:
        up = np.array([1, 0, 0], dtype=np.float64)

    axis1 = np.cross(normal, up)
    axis1 = axis1 / (np.linalg.norm(axis1) + 1e-8)
    axis2 = np.cross(normal, axis1)

    # Find point on plane closest to room center
    # Plane: n·x + d = 0 => distance from origin along normal is -d
    plane_point = -d * normal

    # Project room bounds onto the plane axes to determine quad size
    half_size = max(room_size) / 2 + margin

    # Create quad vertices
    corners = [
        plane_point - half_size * axis1 - half_size * axis2,
        plane_point + half_size * axis1 - half_size * axis2,
        plane_point + half_size * axis1 + half_size * axis2,
        plane_point - half_size * axis1 + half_size * axis2,
    ]

    points = [(float(p[0]), float(p[1]), float(p[2])) for p in corners]
    face_indices = [4, 0, 1, 2, 3]  # Single quad with 4 vertices

    return points, face_indices


def create_box_shell_geometry(
    room_box: Dict,
    include_floor: bool = True,
    include_ceiling: bool = True,
    include_walls: bool = True,
) -> Tuple[List[Tuple[float, float, float]], List[int], List[Tuple[float, float, float]]]:
    """
    Create a box shell (inverted box) from room_box dimensions.

    This creates the interior surfaces of a room box that faces can be seen from inside.

    Args:
        room_box: {"min": [x,y,z], "max": [x,y,z]}
        include_floor: Include bottom face
        include_ceiling: Include top face
        include_walls: Include side walls

    Returns:
        (points, face_vertex_counts, face_vertex_indices, normals)
    """
    room_min = np.array(room_box.get("min", [-5, 0, -5]), dtype=np.float64)
    room_max = np.array(room_box.get("max", [5, 3, 5]), dtype=np.float64)

    # Define the 8 corners of the box
    # Using Y-up convention
    x0, y0, z0 = room_min
    x1, y1, z1 = room_max

    corners = [
        (x0, y0, z0),  # 0: min corner
        (x1, y0, z0),  # 1: +x
        (x1, y0, z1),  # 2: +x +z
        (x0, y0, z1),  # 3: +z
        (x0, y1, z0),  # 4: +y
        (x1, y1, z0),  # 5: +x +y
        (x1, y1, z1),  # 6: +x +y +z
        (x0, y1, z1),  # 7: +y +z
    ]

    points = corners
    face_vertex_counts = []
    face_vertex_indices = []
    normals = []

    # Floor (y = y0, normal pointing UP into room)
    if include_floor:
        face_vertex_counts.append(4)
        face_vertex_indices.extend([0, 1, 2, 3])  # CCW from above
        normals.append((0.0, 1.0, 0.0))

    # Ceiling (y = y1, normal pointing DOWN into room)
    if include_ceiling:
        face_vertex_counts.append(4)
        face_vertex_indices.extend([7, 6, 5, 4])  # CCW from below
        normals.append((0.0, -1.0, 0.0))

    if include_walls:
        # Wall 1: x = x0 (normal pointing +X into room)
        face_vertex_counts.append(4)
        face_vertex_indices.extend([0, 3, 7, 4])  # CCW from +X
        normals.append((1.0, 0.0, 0.0))

        # Wall 2: x = x1 (normal pointing -X into room)
        face_vertex_counts.append(4)
        face_vertex_indices.extend([1, 5, 6, 2])  # CCW from -X
        normals.append((-1.0, 0.0, 0.0))

        # Wall 3: z = z0 (normal pointing +Z into room)
        face_vertex_counts.append(4)
        face_vertex_indices.extend([0, 4, 5, 1])  # CCW from +Z
        normals.append((0.0, 0.0, 1.0))

        # Wall 4: z = z1 (normal pointing -Z into room)
        face_vertex_counts.append(4)
        face_vertex_indices.extend([3, 2, 6, 7])  # CCW from -Z
        normals.append((0.0, 0.0, -1.0))

    return points, face_vertex_counts, face_vertex_indices, normals


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

    def add_room_planes(self, room_planes: Dict, room_box: Optional[Dict] = None) -> None:
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

        # Store room_box if available
        if room_box:
            room_min = room_box.get("min", [-5, 0, -5])
            room_max = room_box.get("max", [5, 3, 5])
            prim.CreateAttribute("roomMin", Sdf.ValueTypeNames.Double3).Set(
                Gf.Vec3d(*room_min)
            )
            prim.CreateAttribute("roomMax", Sdf.ValueTypeNames.Double3).Set(
                Gf.Vec3d(*room_max)
            )

    def add_scene_shell_geometry(self, room_box: Dict) -> None:
        """
        Create actual mesh geometry for the scene shell (walls, floor, ceiling).

        This creates collision/physics-ready geometry based on room_box dimensions.
        The shell is rendered as an inverted box (normals facing inward).
        """
        if not room_box:
            print("[USD] WARNING: No room_box data available, skipping scene shell geometry")
            return

        # Create shell geometry
        points, face_counts, face_indices, normals = create_box_shell_geometry(
            room_box,
            include_floor=True,
            include_ceiling=True,
            include_walls=True,
        )

        # Create the scene shell prim
        shell_path = "/World/Room/SceneShell"
        shell_xform = UsdGeom.Xform.Define(self.stage, shell_path)

        # Create mesh under the shell
        mesh_path = f"{shell_path}/ShellMesh"
        mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)

        # Set mesh points
        mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])

        # Set face vertex counts and indices
        mesh.GetFaceVertexCountsAttr().Set(face_counts)
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)

        # Set mesh purpose to physics for simulation
        mesh.GetPurposeAttr().Set(UsdGeom.Tokens.default_)

        # Set subdivision scheme to none (we want faceted geometry)
        mesh.GetSubdivisionSchemeAttr().Set("none")

        # Add custom attributes for robotics simulation
        prim = mesh.GetPrim()
        prim.CreateAttribute("isSceneShell", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("sim_role", Sdf.ValueTypeNames.String).Set("scene_shell")

        # Store room dimensions
        room_min = room_box.get("min", [-5, 0, -5])
        room_max = room_box.get("max", [5, 3, 5])
        prim.CreateAttribute("shellMin", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*room_min))
        prim.CreateAttribute("shellMax", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*room_max))

        print(f"[USD] Created scene shell geometry at {shell_path}")
        print(f"[USD]   Room bounds: min={room_min}, max={room_max}")

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

    def add_object(
        self,
        obj: Dict,
        layout_objects: Dict[Any, Dict],
        room_box: Optional[Dict] = None,
    ) -> None:
        """Add a single object to the scene."""
        oid = obj.get("id")
        is_interactive = obj.get("type") == "interactive"
        class_name = obj.get("class_name", "")

        # Merge layout data - try both the original ID and string version
        merged = dict(obj)
        layout_obj = layout_objects.get(oid) or layout_objects.get(str(oid))
        if layout_obj:
            for key in ("obb", "center3d"):
                if key in layout_obj:
                    merged[key] = layout_obj[key]

        # Special handling for scene_background: use room_box for positioning
        is_scene_background = (
            class_name == "scene_background" or
            oid == "scene_background" or
            obj.get("sim_role") == "scene_shell"
        )

        if is_scene_background and room_box and not merged.get("obb"):
            # Create synthetic OBB from room_box for scene_background
            room_min = np.array(room_box.get("min", [-5, 0, -5]), dtype=np.float64)
            room_max = np.array(room_box.get("max", [5, 3, 5]), dtype=np.float64)
            room_center = (room_min + room_max) / 2
            room_extents = (room_max - room_min) / 2

            merged["obb"] = {
                "center": room_center.tolist(),
                "extents": room_extents.tolist(),
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity rotation
            }
            merged["center3d"] = room_center.tolist()
            print(f"[USD] obj_{oid}: Created synthetic OBB from room_box for scene_background")

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

        # Get OBB data (from layout merge)
        obb = merged.get("obb")
        obb_extents = None
        obb_center = None
        if obb:
            obb_extents = obb.get("extents")
            obb_center = obb.get("center")

        # 1) Recenter mesh (if we have metadata about mesh center)
        if translation is not None:
            T_align = np.eye(4, dtype=np.float64)
            T_align[:3, 3] = translation
            xform = T_align @ xform
            applied = True

        # 2) Scale to match OBB (if we have both mesh bounds and OBB extents)
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

            if obb_extents:
                prim.CreateAttribute("halfExtents", Sdf.ValueTypeNames.Double3).Set(
                    Gf.Vec3d(*obb_extents)
                )

        # 4) Fallback: If no OBB but we have center3d from layout, use that
        if not applied:
            center3d = merged.get("center3d")
            if center3d is not None:
                T_pos = np.eye(4, dtype=np.float64)
                T_pos[:3, 3] = np.array(center3d, dtype=np.float64)
                xform = T_pos @ xform
                applied = True
                print(f"[USD] obj_{oid}: using center3d fallback position {center3d}")

        # Apply transform (always apply even if identity, to ensure consistent behavior)
        obj_xform.MakeMatrixXform().Set(numpy_to_gf_matrix(xform))
        if applied:
            print(f"[USD] obj_{oid}: applied transform")
        else:
            print(
                f"[USD] obj_{oid}: no spatial data found; left at origin (check layout merge)"
            )

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
            # Add reference with explicit prim path "/Root" as fallback.
            # This ensures the reference works even if the USDZ file doesn't have
            # a default prim set (which is required for prim-less references to work).
            # The glb_to_usd converter creates all geometry under /Root.
            geom_prim.GetReferences().AddReference(usdz_rel, primPath=Sdf.Path("/Root"))
            print(f"[USD] obj_{oid}: referenced {usdz_rel}")
        else:
            # No USDZ found - try to discover a GLB candidate so we can flag it
            # for conversion in the wiring phase. This catches assets where the
            # GLB exists (model.glb/mesh.glb/etc.) but the USDZ was never
            # produced, which previously left the prim with no geometry.
            obj_dir = self.root / self.assets_prefix / f"obj_{oid}"
            glb_candidates = [
                obj_dir / "asset.glb",
                obj_dir / "model.glb",
                obj_dir / "mesh.glb",
            ]
            existing_glb = next((p for p in glb_candidates if p.exists()), None)

            if existing_glb:
                # Update asset_path so downstream conversion can pick it up even
                # if the original asset_path was missing or pointed elsewhere.
                rel_glb = str(existing_glb.relative_to(self.root)).replace("\\", "/")
                prim.CreateAttribute("asset_path", Sdf.ValueTypeNames.String).Set(rel_glb)
                prim.CreateAttribute("pendingConversion", Sdf.ValueTypeNames.Bool).Set(True)
                print(
                    f"[USD] obj_{oid}: missing USDZ, queued GLB for conversion ({rel_glb})"
                )
            elif asset_path and (asset_path.endswith(".glb") or asset_path.endswith(".gltf")):
                prim.CreateAttribute("pendingConversion", Sdf.ValueTypeNames.Bool).Set(True)
                print(f"[USD] obj_{oid}: marked for GLB->USDZ conversion ({asset_path})")

    def add_objects(
        self,
        objects: List[Dict],
        layout_objects: Dict[Any, Dict],
        room_box: Optional[Dict] = None,
    ) -> None:
        """Add all objects to the scene."""
        objects_scope = UsdGeom.Scope.Define(self.stage, "/World/Objects")

        for obj in objects:
            try:
                self.add_object(obj, layout_objects, room_box)
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
    room_box = layout.get("room_box", {})

    # Build layout_objects dict with both integer and string keys for compatibility
    layout_objects: Dict[Any, Dict] = {}
    for o in layout.get("objects", []):
        oid = o.get("id")
        layout_objects[oid] = o
        # Also store with string key if it's not already a string
        if not isinstance(oid, str):
            layout_objects[str(oid)] = o

    # Merge objects from assets with layout data so downstream logic always
    # sees the spatial information produced by DA3/scale jobs. This merged
    # list is passed to the builder below instead of the raw assets list to
    # avoid dropping transforms when the IDs match but the assets.json entry
    # doesn't carry spatial fields.
    objects: List[Dict] = []
    for obj in scene_assets.get("objects", []):
        merged = dict(obj)
        oid = obj.get("id")
        # Try both the original ID and string version
        layout_obj = layout_objects.get(oid) or layout_objects.get(str(oid))
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
    builder.add_room_planes(room_planes, room_box)
    builder.add_cameras(cameras)

    # Create scene shell geometry from room_box if available
    if room_box:
        builder.add_scene_shell_geometry(room_box)
    else:
        print("[USD] WARNING: No room_box data in layout, skipping scene shell geometry")

    builder.add_objects(objects, layout_objects, room_box)

    # Save stage
    stage.GetRootLayer().Save()
    print(f"[USD] Wrote stage to {output_path}")
    print(f"[USD] Cameras: {len(cameras)} | Objects: {len(objects)}")
    if room_box:
        print(f"[USD] Room box: min={room_box.get('min')}, max={room_box.get('max')}")

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
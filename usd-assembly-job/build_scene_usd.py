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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.scene_manifest.loader import load_manifest_or_scene_assets

import numpy as np
from tools.asset_catalog import AssetCatalogClient

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
    """Build a 4x4 transform matrix from an OBB (Oriented Bounding Box) record.

    (P2-13) OBB to Transform Conversion
    ===================================

    Input OBB format (from regen3d_adapter/adapter.py):
        obb["R"]      -> 3x3 rotation matrix [row0, row1, row2]
        obb["center"] -> 3D center position [x, y, z]
        obb["extents"] -> Half-sizes [hx, hy, hz] (used separately for mesh scaling)

    The OBB represents the world-space placement of an object with:
    - center: Position in world coordinates
    - R: Rotation/orientation in world space
    - extents: Used to scale the local mesh to fit the bounding box

    This function creates a 4x4 transformation matrix:
        T_world = [R  center]
                  [0    1   ]

    Where R is the 3x3 rotation matrix and center is the 3D position.

    The full transform pipeline in build_scene_usd.py (lines 750-784):
        1. Extract OBB data (center, extents, R)
        2. Create T_world from R and center using this function
        3. Apply OBB.extents to scale the mesh to match the bounding box
        4. Combine all transforms to place the mesh in world space

    Returns:
        4x4 row-major numpy array, or None if OBB is incomplete
    """
    center = obb.get("center")
    R = obb.get("R")
    if center is None or R is None:
        return None

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(R, dtype=np.float64)
    T[:3, 3] = np.array(center, dtype=np.float64)
    return T


def approx_location_to_coords(
    approx_loc: str, index: int, room_box: Optional[Dict], total_objects: int = 1
) -> Tuple[float, float, float]:
    """
    Convert approx_location strings (e.g., "top left", "middle center") to 3D coordinates.

    Uses the room_box if available, otherwise defaults to a 10m x 3m x 10m room.
    Returns (x, y, z) world coordinates.
    """
    # Determine room bounds
    if room_box:
        min_pt = room_box.get("min", [-5, 0, -5])
        max_pt = room_box.get("max", [5, 3, 5])
    else:
        min_pt = [-5, 0, -5]
        max_pt = [5, 3, 5]

    x_min, y_min, z_min = min_pt
    x_max, y_max, z_max = max_pt
    x_range = x_max - x_min
    z_range = z_max - z_min

    # Default: center of room at floor level
    x = (x_min + x_max) / 2
    y = y_min  # Place objects on floor
    z = (z_min + z_max) / 2

    if not approx_loc:
        # Grid fallback: distribute objects evenly
        grid_cols = max(1, int(np.ceil(np.sqrt(total_objects))))
        row = index // grid_cols
        col = index % grid_cols
        margin = 0.15
        x = x_min + margin * x_range + (col + 0.5) * (x_range * (1 - 2 * margin)) / grid_cols
        z = z_min + margin * z_range + (row + 0.5) * (z_range * (1 - 2 * margin)) / grid_cols
        return (x, y, z)

    loc_lower = approx_loc.lower().strip()

    # Horizontal mapping (X axis)
    if "left" in loc_lower:
        x = x_min + 0.25 * x_range
    elif "right" in loc_lower:
        x = x_min + 0.75 * x_range
    else:  # center or default
        x = (x_min + x_max) / 2

    # Depth mapping (Z axis) - "top" = far, "bottom" = near
    if "top" in loc_lower:
        z = z_min + 0.75 * z_range
    elif "bottom" in loc_lower:
        z = z_min + 0.25 * z_range
    else:  # middle or default
        z = (z_min + z_max) / 2

    # Add small offset based on index to prevent overlapping
    offset = 0.3 * (index % 5)
    x += offset
    z += offset * 0.5

    return (x, y, z)


def generate_synthetic_spatial_data(
    objects: List[Dict], room_box: Optional[Dict]
) -> Dict[str, Dict]:
    """
    Generate synthetic OBB/center3d data for objects without spatial information.

    Uses approx_location if available, otherwise distributes objects in a grid.
    Returns a dict mapping object ID to synthetic spatial data.
    """
    synthetic = {}
    total = len(objects)

    for idx, obj in enumerate(objects):
        oid = obj.get("id")
        if oid is None:
            continue

        # Skip if object already has spatial data
        if obj.get("obb") or obj.get("center3d"):
            continue

        approx_loc = obj.get("approx_location", "")
        x, y, z = approx_location_to_coords(approx_loc, idx, room_box, total)

        # Default extents (will be refined by mesh metadata if available)
        default_extents = [0.25, 0.25, 0.25]

        synthetic[oid] = {
            "center3d": [x, y + default_extents[1], z],  # Raise to half-height
            "obb": {
                "center": [x, y + default_extents[1], z],
                "extents": default_extents,
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Identity rotation
            },
            "synthetic": True,  # Flag for downstream processing
            "approx_location": approx_loc,
        }
        print(f"[USD] obj_{oid}: generated synthetic position from approx_location='{approx_loc}' -> ({x:.2f}, {y:.2f}, {z:.2f})")

    return synthetic


def numpy_to_gf_matrix(m: np.ndarray) -> Gf.Matrix4d:
    """Convert a numpy 4x4 matrix to Gf.Matrix4d."""
    m = np.array(m, dtype=np.float64).reshape(4, 4)
    return Gf.Matrix4d(*m.flatten().tolist())


# -----------------------------------------------------------------------------
# Metadata Loading
# -----------------------------------------------------------------------------


def load_object_metadata(
    root: Path,
    obj: Dict,
    assets_prefix: str,
    catalog_client: Optional[AssetCatalogClient] = None,
) -> Optional[dict]:
    """
    Load per-object metadata if present.

    Resolution order:
      1. central catalog lookup (asset_id or asset_path)
      2. obj["metadata_path"] if present (bucket-relative)
      3. metadata.json next to the asset_path
      4. assets_prefix/obj_{id}/metadata.json (standard layout)
    """
    if catalog_client is not None:
        logical_id = obj.get("logical_asset_id") or obj.get("logical_id")
        try:
            catalog_meta = catalog_client.lookup_metadata(
                asset_id=obj.get("id"),
                asset_path=obj.get("asset_path"),
                logical_id=logical_id,
            )
            if catalog_meta:
                return catalog_meta
        except Exception as exc:  # pragma: no cover - network errors
            print(f"[USD] WARNING: catalog lookup failed: {exc}", file=sys.stderr)

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
        "simready.usda",
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
        self.catalog_client = AssetCatalogClient()

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
            self.catalog_client,
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
# USD Stage Validation
# -----------------------------------------------------------------------------


def _validate_usd_stage(stage: Usd.Stage, output_path: Path, objects: List[Dict]) -> None:
    """
    Validate USD stage after modifications to catch common errors.

    GAP-USD-002 FIX: Validate stage after save to ensure it's properly formed.
    """
    if not output_path.exists():
        raise ValueError(f"USD file not found after save: {output_path}")

    # Check file size
    file_size = output_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Empty USD file written: {output_path}")

    # Validate stage has root prim
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        print("[USD] WARNING: Stage has no valid default prim")

    # Validate World prim exists
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim or not world_prim.IsValid():
        raise ValueError("Required /World prim not found in stage")

    # Validate Objects scope exists
    objects_scope = stage.GetPrimAtPath("/World/Objects")
    if not objects_scope or not objects_scope.IsValid():
        print("[USD] WARNING: /World/Objects scope not found")

    # Validate at least some objects have references
    objects_with_refs = 0
    for obj in objects:
        oid = obj.get("id")
        if oid is not None:
            obj_prim = stage.GetPrimAtPath(f"/World/Objects/obj_{oid}")
            if obj_prim and obj_prim.IsValid():
                geom_prim = stage.GetPrimAtPath(f"/World/Objects/obj_{oid}/Geom")
                if geom_prim and geom_prim.IsValid():
                    refs = geom_prim.GetReferences()
                    if refs.GetAddedOrExplicitItems():
                        objects_with_refs += 1

    if len(objects) > 0 and objects_with_refs == 0:
        print(f"[USD] WARNING: None of {len(objects)} objects have geometry references")
    else:
        print(f"[USD] Validation: {objects_with_refs}/{len(objects)} objects have geometry references")

    print(f"[USD] Stage validation passed: {output_path}")


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
    assets_root = root / assets_prefix
    print(f"[USD] Loading asset manifest from {assets_path}")

    def _has_spatial_data(layout_json: Dict[str, Any]) -> bool:
        return any(
            obj.get("obb") is not None or obj.get("center3d") is not None
            for obj in layout_json.get("objects", [])
        )

    # Load input files
    primary_layout = load_json(layout_path)
    layout = primary_layout

    # Discover candidate layouts for fallback if the scaled layout lacks spatial
    # data. Preference order:
    #   1. scene_layout_scaled.json (primary)
    #   2. scene_layout.json in the same prefix
    #   3. DA3 layout path (../layout/scene_layout.json)
    fallback_layouts: List[Tuple[Path, Dict[str, Any]]] = []

    same_prefix_fallback = layout_path.with_name("scene_layout.json")
    if same_prefix_fallback != layout_path and same_prefix_fallback.is_file():
        try:
            fallback_layouts.append((same_prefix_fallback, load_json(same_prefix_fallback)))
        except Exception:
            pass

    da3_layout_path = layout_path.parent.parent / "layout" / "scene_layout.json"
    if da3_layout_path not in {layout_path, same_prefix_fallback} and da3_layout_path.is_file():
        try:
            fallback_layouts.append((da3_layout_path, load_json(da3_layout_path)))
        except Exception:
            pass

    layout_used_path = layout_path
    if not _has_spatial_data(layout):
        print(
            f"[USD] WARNING: {layout_path} has no objects with 'obb' or 'center3d'; "
            "attempting to fall back to DA3 layout outputs"
        )

        for candidate_path, candidate_layout in fallback_layouts:
            if _has_spatial_data(candidate_layout):
                layout = candidate_layout
                layout_used_path = candidate_path
                print(
                    f"[USD] Using layout from {candidate_path} because the scaled layout "
                    "was missing spatial data"
                )
                break

    # Flag to track if we need to generate synthetic positions
    using_synthetic_positions = False

    if not _has_spatial_data(layout):
        tried_paths = [str(layout_path)] + [str(p) for p, _ in fallback_layouts]
        print(
            f"[USD] WARNING: No spatial data (obb/center3d) found in any layout. "
            f"Checked: {', '.join(tried_paths)}. "
            "Will generate synthetic positions from approx_location or grid layout."
        )
        using_synthetic_positions = True

    if layout_used_path != layout_path:
        print(f"[USD] Layout override: using {layout_used_path}")
    else:
        print(f"[USD] Using scaled layout at {layout_used_path}")

    if layout_used_path != layout_path:
        fallback_layout: Dict[str, Any] = primary_layout
    else:
        fallback_layout = next(
            (data for path, data in fallback_layouts if path != layout_used_path),
            {},
        )
    scene_assets = load_manifest_or_scene_assets(assets_root)
    if scene_assets is None:
        raise FileNotFoundError(
            f"scene manifest not found at {assets_root / 'scene_manifest.json'} "
            f"or legacy plan at {assets_root / 'scene_assets.json'}"
        )

    # Extract data
    cameras = layout.get("camera_trajectory") or []
    room_planes = layout.get("room_planes", {})
    room_box = layout.get("room_box", {})

    # Build layout_objects dict with both integer and string keys for compatibility.
    # If the scaled layout is missing spatial fields, fall back to the original
    # (unscaled) layout to recover obb/center information.
    def _merge_spatial(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key in ("obb", "center3d", "center", "bounds", "scale"):
            if key not in target and key in source:
                target[key] = source[key]

    layout_objects: Dict[Any, Dict] = {}
    layout_objects_by_class: Dict[str, List[Dict]] = {}

    # First, populate from the primary (scaled) layout
    for o in layout.get("objects", []):
        oid = o.get("id")
        if oid is None:
            continue

        fallback_obj = None
        if fallback_layout:
            fallback_obj = next(
                (
                    obj
                    for obj in fallback_layout.get("objects", [])
                    if obj.get("id") == oid or str(obj.get("id")) == str(oid)
                ),
                None,
            )
            if fallback_obj:
                _merge_spatial(o, fallback_obj)

        layout_objects[oid] = o
        if not isinstance(oid, str):
            layout_objects[str(oid)] = o

        cls = (o.get("class_name") or "").lower()
        if cls:
            layout_objects_by_class.setdefault(cls, []).append(o)

    # Next, add any fallback-only objects so we don't drop transforms entirely
    if fallback_layout:
        for o in fallback_layout.get("objects", []):
            oid = o.get("id")
            if oid is None:
                continue
            if oid in layout_objects or str(oid) in layout_objects:
                continue
            layout_objects[oid] = o
            if not isinstance(oid, str):
                layout_objects[str(oid)] = o

            cls = (o.get("class_name") or "").lower()
            if cls:
                layout_objects_by_class.setdefault(cls, []).append(o)

    # Merge objects from assets with layout data so downstream logic always
    # sees the spatial information produced by DA3/scale jobs. This merged
    # list is passed to the builder below instead of the raw assets list to
    # avoid dropping transforms when the IDs match but the assets.json entry
    # doesn't carry spatial fields.
    objects: List[Dict] = []
    matched_layout_ids: Set[Any] = set()
    for obj in scene_assets.get("objects", []):
        merged = dict(obj)
        oid = obj.get("id")
        cls = (obj.get("class_name") or "").lower()

        # Try both the original ID and string version
        layout_obj = layout_objects.get(oid) or layout_objects.get(str(oid))

        # Fallback: if IDs don't line up (common after inventory remapping),
        # try to match by class name so we still apply DA3/scale spatial data.
        if not layout_obj and cls:
            candidates = layout_objects_by_class.get(cls, [])
            layout_obj = next((o for o in candidates if o.get("id") not in matched_layout_ids), None)
            if layout_obj:
                matched_layout_ids.add(layout_obj.get("id"))
                merged["layout_match_source"] = layout_obj.get("id")

        if layout_obj:
            for key in ("obb", "center3d", "center", "bounds", "scale", "approx_location"):
                if key in layout_obj:
                    merged[key] = layout_obj[key]
        objects.append(merged)

    # Generate synthetic spatial data for objects without obb/center3d
    if using_synthetic_positions:
        synthetic_data = generate_synthetic_spatial_data(objects, room_box or None)
        for obj in objects:
            oid = obj.get("id")
            if oid and oid in synthetic_data:
                synth = synthetic_data[oid]
                for key in ("obb", "center3d", "synthetic", "approx_location"):
                    if key in synth:
                        obj[key] = synth[key]

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

    # GAP-USD-002 FIX: Validate USD stage after save
    _validate_usd_stage(stage, output_path, objects)

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
    manifest_path = root / assets_prefix / "scene_manifest.json"
    assets_path = manifest_path if manifest_path.is_file() else root / assets_prefix / "scene_assets.json"
    stage_path = root / usd_prefix / "scene.usda"

    if not assets_path.is_file():
        print(
            f"[USD] scene manifest not found at {manifest_path} or legacy scene_assets.json",
            file=sys.stderr,
        )
        sys.exit(1)

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

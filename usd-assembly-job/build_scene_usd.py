import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def matrix_from_obb(obb: Dict) -> Optional[np.ndarray]:
    """
    Build a 4x4 transform matrix from an OBB record:

      obb["R"]      -> 3x3 rotation matrix
      obb["center"] -> 3D center

    The result is a row-major matrix suitable for a USD matrix4d.
    """
    center = obb.get("center")
    R = obb.get("R")
    if center is None or R is None:
        return None

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(R, dtype=np.float64)
    T[:3, 3] = np.array(center, dtype=np.float64)
    return T


def fmt_matrix(m: np.ndarray) -> str:
    flat = m.reshape(-1)
    return "(" + ", ".join(f"{v:.6f}" for v in flat) + ")"


def fmt_vec(v: List[float]) -> str:
    return "(" + ", ".join(f"{float(x):.6f}" for x in v) + ")"


def usd_for_camera(cam: Dict) -> List[str]:
    """
    Encode per-camera metadata under World.Cameras:

      - cameraExtrinsics: 4x4 matrix (3x4 extrinsics + [0,0,0,1])
      - intrinsics: fx, fy, cx
      - imagePath: source RGB path (if provided)
    """
    lines = [f'    def Xform "cam_{cam.get("id")}" {{']

    extr = cam.get("extrinsics")
    if extr:
        mat = np.eye(4, dtype=np.float64)
        ext = np.array(extr, dtype=np.float64)
        if ext.shape == (3, 4):
            mat[:3, :4] = ext
        lines.append(f"      matrix4d cameraExtrinsics = {fmt_matrix(mat)}")

    intr = cam.get("intrinsics")
    if intr:
        # intrinsics[0][0] = fx, intrinsics[1][1] = fy, intrinsics[0][2] = cx
        lines.append(
            f"      double3 intrinsics = "
            f"{fmt_vec([intr[0][0], intr[1][1], intr[0][2]])}"
        )

    if cam.get("image_path"):
        lines.append(f"      string imagePath = \"{cam['image_path']}\"")

    lines.append("    }")
    return lines


def safe_path_join(root: Path, rel: str) -> Path:
    """
    Join a relative path against the GCS fuse root, stripping any leading "/".
    """
    rel_path = rel.lstrip("/")
    return root / rel_path


def load_object_metadata(root: Path, obj: Dict, assets_prefix: str) -> Optional[dict]:
    """
    Load per-object metadata if present.

    Resolution order:

      1. obj["metadata_path"] if present (bucket-relative).
      2. metadata.json next to the asset_path.
      3. assets_prefix/static/obj_{id}/metadata.json (legacy layout).
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

    # Fall back to static/obj_{id}/metadata.json under the assets prefix.
    oid = obj.get("id")
    if oid is not None:
        static_dir = safe_path_join(root, f"{assets_prefix}/static/obj_{oid}")
        candidate = static_dir / "metadata.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())

    return None


def alignment_from_metadata(metadata: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract alignment information from metadata.

    Returns:
      - translation (3,) that recenters mesh to the origin, if available.
      - half_extents (3,) in mesh-local space, if available.
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
        # We want to recenter the mesh so the origin is at the bbox center.
        translation = -np.array(center, dtype=np.float64)

    if size is not None:
        half_extents = 0.5 * np.array(size, dtype=np.float64)

    return translation, half_extents


def usd_for_object(obj: Dict, assets_prefix: str, root: Path) -> List[str]:
    """
    Emit a single Object prim under World.Objects.

    Handles both:
      - interactive objects (URDF-based) and
      - static meshes (GLB / USDZ from SAM3D or Hunyuan).
    """
    oid = obj.get("id")
    is_interactive = obj.get("type") == "interactive"

    # Decide which asset the stage should point at and what kind it is.
    asset_path = obj.get("asset_path")
    asset_type = obj.get("asset_type")

    if not asset_path:
        if is_interactive:
            # Interactive objects default to URDFs produced by the interactive job.
            base_dir = obj.get("interactive_output") or f"{assets_prefix}/interactive/obj_{oid}"
            asset_path = f"{base_dir}/obj_{oid}.urdf"
            asset_type = asset_type or "urdf"
        else:
            # Static objects default to GLB / USDZ produced by SAM3D or Hunyuan jobs.
            base_dir = f"{assets_prefix}/obj_{oid}"
            usdz_rel = f"{base_dir}/model.usdz"
            glb_rel = f"{base_dir}/asset.glb"
            if safe_path_join(root, usdz_rel).is_file():
                asset_path = usdz_rel
                asset_type = asset_type or "usdz"
            else:
                asset_path = glb_rel
                asset_type = asset_type or "glb"
    else:
        # If we were given an explicit asset_path, try to infer the type when not present.
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

    # Resolve metadata using the resolved asset_path.
    metadata = load_object_metadata(
        root,
        {"id": oid, "asset_path": asset_path, "metadata_path": obj.get("metadata_path")},
        assets_prefix,
    )
    translation, mesh_half_extents = alignment_from_metadata(metadata or {})

    lines = [f'    def Xform "obj_{oid}" {{']

    # High-level object descriptors.
    lines.append(f"      bool interactive = {'true' if is_interactive else 'false'}")
    if asset_type:
        lines.append(f"      string assetType = \"{asset_type}\"")
    if asset_path:
        lines.append(f"      string asset_path = \"{asset_path}\"")

    if is_interactive:
        # Manifest from the interactive job (joint frames, articulation info, etc.)
        manifest = obj.get("interactive_output") or f"{assets_prefix}/interactive/obj_{oid}"
        lines.append(f"      string urdf_manifest = \"{manifest}/interactive_manifest.json\"")

    xform = np.eye(4, dtype=np.float64)
    applied = False

    # 1) Recenter the mesh in its local space if we know its bounds.
    if translation is not None:
        T_align = np.eye(4, dtype=np.float64)
        T_align[:3, 3] = translation
        xform = T_align @ xform
        applied = True

    # 2) Scale the mesh so that its extents match the layout OBB extents.
    obb = obj.get("obb")
    obb_extents = None
    if obb:
        obb_extents = obb.get("extents")
    if obb_extents is not None and mesh_half_extents is not None:
        # Avoid divide-by-zero if any component is zero.
        mesh_half_extents = np.where(mesh_half_extents == 0, 1.0, mesh_half_extents)
        scale_vec = np.array(obb_extents, dtype=np.float64) / mesh_half_extents
        S = np.eye(4, dtype=np.float64)
        S[0, 0], S[1, 1], S[2, 2] = scale_vec
        xform = S @ xform
        applied = True

    # 3) Finally, place into world space via OBB pose.
    if obb:
        T = matrix_from_obb(obb)
        if T is not None:
            xform = T @ xform
            applied = True

        extents = obb.get("extents")
        if extents:
            # These are layout-space half-extents on the OBB axes.
            lines.append(f"      double3 halfExtents = {fmt_vec(extents)}")

    # If we applied any local -> world transform, emit it as a single xformOp.
    if applied:
        lines.append(f"      matrix4d xformOp:transform = {fmt_matrix(xform)}")
        lines.append("      uniform token[] xformOpOrder = [\"xformOp:transform\"]")

    # Optional mesh-space metadata (useful in Isaac tools).
    if metadata:
        mesh_bounds = metadata.get("mesh_bounds") or {}
        export_bounds = mesh_bounds.get("export") or mesh_bounds.get("bounds") or mesh_bounds
        center = export_bounds.get("center")
        size = export_bounds.get("size")
        if center:
            lines.append(f"      double3 meshCenter = {fmt_vec(center)}")
        if size:
            lines.append(f"      double3 meshSize = {fmt_vec(size)}")

    if obj.get("class_name"):
        lines.append(f"      string className = \"{obj['class_name']}\"")
    if obj.get("pipeline"):
        lines.append(f"      string pipeline = \"{obj['pipeline']}\"")

    lines.append("    }")
    return lines


def usd_for_planes(room_planes: Dict) -> List[str]:
    """
    Encode room planes as equations ax + by + cz + d = 0, which Isaac
    can use as hints for floor/ceiling/walls when constructing the scene.
    """
    lines = ["  def Scope \"Room\" {"]
    floor = room_planes.get("floor") or {}
    lines.append(f"    double4 floorEquation = {fmt_vec(floor.get('equation', [0, 1, 0, 0]))}")
    ceiling = room_planes.get("ceiling") or {}
    lines.append(f"    double4 ceilingEquation = {fmt_vec(ceiling.get('equation', [0, 1, 0, 0]))}")
    for idx, wall in enumerate(room_planes.get("walls", [])):
        lines.append(f"    double4 wall{idx}Equation = {fmt_vec(wall.get('equation', [1, 0, 0, 0]))}")
    lines.append("  }")
    return lines


def main() -> None:
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

    out_dir = root / usd_prefix
    ensure_dir(out_dir)
    stage_path = out_dir / "scene.usda"

    layout = load_json(layout_path)
    scene_assets = load_json(assets_path)

    cameras = layout.get("camera_trajectory") or []
    room_planes = layout.get("room_planes", {})

    # Map layout objects by id so we can graft their OBB / center back
    # into the scene_assets objects.
    layout_objects = {o.get("id"): o for o in layout.get("objects", [])}

    objects = []
    for obj in scene_assets.get("objects", []):
        merged = dict(obj)
        layout_obj = layout_objects.get(obj.get("id"))
        if layout_obj:
            for key in ("obb", "center3d"):
                if key in layout_obj:
                    merged[key] = layout_obj[key]
        objects.append(merged)

    # ------------------------------------------------------------------
    # Emit a simple, readable USDA stage.
    # ------------------------------------------------------------------
    lines: List[str] = []
    lines.append("#usda 1.0")
    lines.append("def Xform \"World\" {")

    # Room planes (optional).
    lines.extend(usd_for_planes(room_planes))

    # Cameras.
    lines.append("  def Scope \"Cameras\" {")
    for cam in cameras:
        lines.extend(usd_for_camera(cam))
    lines.append("  }")

    # Objects.
    lines.append("  def Scope \"Objects\" {")
    for obj in objects:
        lines.extend(usd_for_object(obj, assets_prefix=assets_prefix, root=root))
    lines.append("  }")

    lines.append("}")

    stage_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[USD] Wrote stage to {stage_path}")
    print(f"[USD] Cameras: {len(cameras)} | Objects: {len(objects)}")


if __name__ == "__main__":
    main()

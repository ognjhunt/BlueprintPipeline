import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def matrix_from_obb(obb: Dict) -> Optional[np.ndarray]:
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
        lines.append(f"      double3 intrinsics = {fmt_vec([intr[0][0], intr[1][1], intr[0][2]])}")
    if cam.get("image_path"):
        lines.append(f"      string imagePath = \"{cam['image_path']}\"")
    lines.append("    }")
    return lines


def usd_for_object(obj: Dict, assets_prefix: str) -> List[str]:
    oid = obj.get("id")
    is_interactive = obj.get("type") == "interactive"
    asset_path = obj.get("asset_path")
    if not asset_path:
        if is_interactive:
            base_dir = obj.get("interactive_output") or f"{assets_prefix}/interactive/obj_{oid}"
            asset_path = f"{base_dir}/obj_{oid}.urdf"
        else:
            asset_path = f"{assets_prefix}/obj_{oid}/asset.glb"
    obb = obj.get("obb")
    lines = [f'    def Xform "obj_{oid}" {{']
    lines.append(f"      bool interactive = {'true' if is_interactive else 'false'}")
    lines.append(f"      string asset_path = \"{asset_path}\"")
    if is_interactive:
        manifest = obj.get("interactive_output") or f"{assets_prefix}/interactive/obj_{oid}"
        lines.append(f"      string urdf_manifest = \"{manifest}/interactive_manifest.json\"")
    if obb:
        T = matrix_from_obb(obb)
        if T is not None:
            lines.append(f"      matrix4d xformOp:transform = {fmt_matrix(T)}")
            lines.append("      uniform token[] xformOpOrder = [\"xformOp:transform\"]")
        extents = obb.get("extents")
        if extents:
            lines.append(f"      double3 halfExtents = {fmt_vec(extents)}")
    if obj.get("class_name"):
        lines.append(f"      string className = \"{obj['class_name']}\"")
    if obj.get("pipeline"):
        lines.append(f"      string pipeline = \"{obj['pipeline']}\"")
    lines.append("    }")
    return lines


def usd_for_planes(room_planes: Dict) -> List[str]:
    lines = ["  def Scope \"Room\" {"]
    floor = room_planes.get("floor") or {}
    lines.append(f"    double4 floorEquation = {fmt_vec(floor.get('equation', [0,1,0,0]))}")
    ceiling = room_planes.get("ceiling") or {}
    lines.append(f"    double4 ceilingEquation = {fmt_vec(ceiling.get('equation', [0,1,0,0]))}")
    for idx, wall in enumerate(room_planes.get("walls", [])):
        lines.append(f"    double4 wall{idx}Equation = {fmt_vec(wall.get('equation', [1,0,0,0]))}")
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

    lines: List[str] = []
    lines.append("#usda 1.0")
    lines.append("def Xform \"World\" {")
    lines.extend(usd_for_planes(room_planes))
    lines.append("  def Scope \"Cameras\" {")
    for cam in cameras:
        lines.extend(usd_for_camera(cam))
    lines.append("  }")

    lines.append("  def Scope \"Objects\" {")
    for obj in objects:
        lines.extend(usd_for_object(obj, assets_prefix=assets_prefix))
    lines.append("  }")
    lines.append("}")

    stage_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[USD] Wrote stage to {stage_path}")
    print(f"[USD] Cameras: {len(cameras)} | Objects: {len(objects)}")


if __name__ == "__main__":
    main()

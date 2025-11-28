#!/usr/bin/env python3
"""
convert_glb_to_usd_and_rewire.py

Utility script to:
  1. Convert per-object GLB meshes to USDZ.
  2. Rewrite a scene.usda so that each obj_N prim:
       - has a child Xform "Geom" that references the USDZ
         via a path *relative* to scene.usda
       - (optionally) updates its asset_path/assetType metadata
         to point at the USDZ.

Intended usage in your pipeline:
  - Run after build_scene_usd.py in the usd-assembly job.
  - Point it at the scene.usda under /mnt/gcs/${USD_PREFIX}/scene.usda.
  - Let GLB_TO_USD_CONVERTER point to a CLI that accepts:
        <input.glb> <output.usdz>

Examples:

  # Dry run, just to see what it would do
  python convert_glb_to_usd_and_rewire.py \
      --scene /mnt/gcs/scenes/SCENE_ID/usd/scene.usda \
      --converter /usr/local/bin/glb_to_usdz \
      --dry-run

  # Real run
  python convert_glb_to_usd_and_rewire.py \
      --scene /mnt/gcs/scenes/SCENE_ID/usd/scene.usda \
      --converter /usr/local/bin/glb_to_usdz \
      --root /mnt/gcs
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, Set

from pxr import Usd, UsdGeom


def run_converter(converter: str, glb_path: Path, usdz_path: Path) -> None:
    """
    Call an external GLB->USDZ converter.

    The converter must accept two arguments:
        converter <input.glb> <output.usdz>
    """
    usdz_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [converter, str(glb_path), str(usdz_path)]
    print(f"[convert] {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Converter failed for {glb_path} -> {usdz_path}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def infer_usdz_path_from_glb(glb_rel: str) -> str:
    """
    Given a bucket-relative GLB path like:
        'scenes/ID/assets/obj_10/asset.glb'

    return the bucket-relative USDZ path:
        'scenes/ID/assets/obj_10/asset.usdz'
    """
    if glb_rel.lower().endswith(".glb"):
        return glb_rel[:-4] + ".usdz"
    return glb_rel + ".usdz"


def collect_obj_prims(stage: Usd.Stage) -> Dict[Usd.Prim, str]:
    """
    Find all obj_* prims that carry an 'asset_path' attribute (string).

    Returns:
      dict {prim -> asset_path_string}
    """
    prim_to_asset_path: Dict[Usd.Prim, str] = {}

    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.startswith("obj_"):
            continue

        # Try attribute first.
        attr = prim.GetAttribute("asset_path")
        asset_path_val = None
        if attr and attr.HasAuthoredValueOpinion():
            asset_path_val = attr.Get()

        # Fallback: customData, if you ever start storing it there.
        if not asset_path_val:
            custom = prim.GetCustomDataByKey("asset_path")
            if custom:
                asset_path_val = custom

        if not asset_path_val or not isinstance(asset_path_val, str):
            continue

        prim_to_asset_path[prim] = asset_path_val

    return prim_to_asset_path


def rewire_scene_to_usdz(
    scene_path: Path,
    converter: str,
    root: Path,
    dry_run: bool = False,
) -> None:
    """
    Main logic:
      - open scene.usda
      - for each obj_* prim whose asset_path ends in *.glb:
          - convert that GLB (bucket-relative) into a USDZ next to it
          - add/override child 'Geom' Xform with a reference to that USDZ,
            using a path *relative* to scene.usda
          - optionally update the prim's asset_path / assetType metadata
            to describe the USDZ instead of the GLB.
    """
    print(f"[info] Opening scene: {scene_path}")
    stage = Usd.Stage.Open(str(scene_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {scene_path}")

    stage_dir = scene_path.parent
    prim_to_asset_path = collect_obj_prims(stage)

    if not prim_to_asset_path:
        print("[warn] No obj_* prims with asset_path found; nothing to do.")
        return

    print(f"[info] Found {len(prim_to_asset_path)} obj_* prims with asset_path")

    converted: Set[Path] = set()

    for prim, asset_rel in prim_to_asset_path.items():
        # asset_rel is bucket-relative, e.g. 'scenes/ID/assets/obj_10/asset.glb'
        asset_rel_norm = asset_rel.lstrip("/").replace("\\", "/")

        # Only handle GLBs here; skip URDF, USD(Z), etc.
        if not asset_rel_norm.lower().endswith(".glb"):
            print(f"[info] Skipping non-GLB asset for {prim.GetPath()}: {asset_rel_norm}")
            continue

        glb_path = (root / asset_rel_norm).resolve()

        if not glb_path.exists():
            print(f"[warn] GLB not found on disk for prim {prim.GetPath()}: {glb_path}")
            continue

        usdz_rel_bucket = infer_usdz_path_from_glb(asset_rel_norm)
        usdz_path = (root / usdz_rel_bucket).resolve()

        # Convert GLB -> USDZ if we haven't already (or if file not present).
        if usdz_path not in converted and not usdz_path.exists():
            if dry_run:
                print(f"[dry-run] Would convert {glb_path} -> {usdz_path}")
            else:
                run_converter(converter, glb_path, usdz_path)
            converted.add(usdz_path)
        else:
            print(f"[info] Reusing existing USDZ for {glb_path}: {usdz_path}")

        # Compute a path for the reference that is relative to scene.usda
        asset_path_for_reference = os.path.relpath(usdz_path, stage_dir)
        asset_path_for_reference = asset_path_for_reference.replace("\\", "/")

        # Override or create the 'Geom' child prim under this object.
        geom_path = prim.GetPath().AppendChild("Geom")
        geom_prim = stage.OverridePrim(geom_path)

        # Ensure it's an Xform (nice for Hydra/Blender, not strictly required)
        UsdGeom.Xform.Define(stage, geom_path)

        rel_refs = geom_prim.GetReferences()
        rel_refs.ClearReferences()

        print(
            f"[info] Prim {prim.GetPath()} -> Geom references {asset_path_for_reference}"
        )

        if not dry_run:
            rel_refs.AddReference(asset_path_for_reference)

            # Optionally update metadata so downstream tools know the "real" asset.
            asset_attr = prim.GetAttribute("asset_path")
            if asset_attr:
                asset_attr.Set(usdz_rel_bucket)
            type_attr = prim.GetAttribute("assetType")
            if type_attr:
                type_attr.Set("usdz")

    if dry_run:
        print("[dry-run] Skipping save of root layer")
    else:
        print(f"[info] Saving modified scene to {scene_path}")
        stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GLB assets to USDZ and rewire scene.usda to reference them."
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Path to scene.usda (or any USD stage with obj_* prims).",
    )
    parser.add_argument(
        "--converter",
        required=True,
        help="Executable to convert GLB->USDZ (e.g. '/usr/local/bin/glb_to_usdz').",
    )
    parser.add_argument(
        "--root",
        default="/mnt/gcs",
        help="Filesystem root for bucket contents (default: /mnt/gcs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any files; just log what would happen.",
    )

    args = parser.parse_args()

    scene_path = Path(args.scene).resolve()
    if not scene_path.exists():
        raise SystemExit(f"Scene file does not exist: {scene_path}")

    root = Path(args.root).resolve()

    rewire_scene_to_usdz(
        scene_path=scene_path,
        converter=args.converter,
        root=root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

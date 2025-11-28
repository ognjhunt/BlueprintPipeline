#!/usr/bin/env python3
"""
convert_glb_to_usd_and_rewire.py

Utility script to:
  1. Convert per-object GLB meshes to USD.
  2. Rewrite a scene.usda so that each obj_N prim references asset.usd
     instead of only storing a string asset_path to asset.glb.

Intended usage in your pipeline:
  - Run after your current usd-assembly job.
  - Point it at the scene.usda under /mnt/gcs/${USD_PREFIX}/scene.usda.

Examples:

  # Dry run, just to see what it would do
  python convert_glb_to_usd_and_rewire.py \
      --scene /mnt/gcs/scenes/SCENE_ID/usd/scene.usda \
      --converter usd_from_gltf \
      --dry-run

  # Real run
  python convert_glb_to_usd_and_rewire.py \
      --scene /mnt/gcs/scenes/SCENE_ID/usd/scene.usda \
      --converter usd_from_gltf \
      --root /mnt/gcs
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, Set

from pxr import Usd, UsdGeom


def run_converter(converter: str, glb_path: Path, usd_path: Path) -> None:
    """
    Call an external glTF->USD converter.

    This is deliberately simple; adapt it to your actual converter.
    Many converters take 'input.glb output.usd' as arguments.
    """
    usd_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [converter, str(glb_path), str(usd_path)]
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
            f"Converter failed for {glb_path} -> {usd_path}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def infer_usd_path_from_glb(glb_rel: str) -> str:
    """
    Given a bucket-relative GLB path like 'scenes/ID/assets/obj_10/asset.glb',
    return 'scenes/ID/assets/obj_10/asset.usd'.
    """
    if glb_rel.lower().endswith(".glb"):
        return glb_rel[:-4] + ".usd"
    return glb_rel + ".usd"


def collect_obj_prims(stage: Usd.Stage) -> Dict[Usd.Prim, str]:
    """
    Find all obj_* prims that carry an 'asset_path' attribute.

    Returns:
      dict {prim -> asset_path_string}
    """
    prim_to_asset_path: Dict[Usd.Prim, str] = {}

    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.startswith("obj_"):
            continue

        # Try both attribute and customData, depending on how you authored it.
        attr = prim.GetAttribute("asset_path")
        asset_path_val = None
        if attr and attr.HasAuthoredValueOpinion():
            asset_path_val = attr.Get()

        if not asset_path_val:
            # customData fallback
            custom = prim.GetCustomDataByKey("asset_path")
            if custom:
                asset_path_val = custom

        if not asset_path_val:
            continue

        if not isinstance(asset_path_val, str):
            continue

        prim_to_asset_path[prim] = asset_path_val

    return prim_to_asset_path


def rewire_scene_to_usd(
    scene_path: Path,
    converter: str,
    root: Path,
    dry_run: bool = False,
) -> None:
    """
    Main logic:
      - open scene.usda
      - for each obj_* prim with asset_path = scenes/.../assets/.../*.glb
        - run converter once per GLB to produce a *.usd next to it
        - add a 'Visual' child prim with a reference to the USD
          using a path relative to the scene.usda location.
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

        # We only handle GLBs here; skip URDF, USD(Z), etc.
        if not asset_rel_norm.lower().endswith(".glb"):
            print(f"[info] Skipping non-GLB asset for {prim.GetPath()}: {asset_rel_norm}")
            continue

        glb_path = (root / asset_rel_norm).resolve()

        if not glb_path.exists():
            print(f"[warn] GLB not found on disk for prim {prim.GetPath()}: {glb_path}")
            continue

        usd_rel_bucket = infer_usd_path_from_glb(asset_rel_norm)
        usd_path = (root / usd_rel_bucket).resolve()

        if usd_path not in converted and not usd_path.exists():
            if dry_run:
                print(f"[dry-run] Would convert {glb_path} -> {usd_path}")
            else:
                run_converter(converter, glb_path, usd_path)
            converted.add(usd_path)
        else:
            print(f"[info] Reusing existing USD for {glb_path}: {usd_path}")

        # Compute a path for the reference that is relative to scene.usda
        asset_path_for_reference = os.path.relpath(usd_path, stage_dir)
        asset_path_for_reference = asset_path_for_reference.replace("\\", "/")

        visual_path = prim.GetPath().AppendChild("Visual")
        visual_prim = stage.OverridePrim(visual_path)

        # Ensure it's an Xform (nice to have; not strictly required)
        UsdGeom.Xform.Define(stage, visual_path)

        rel_refs = visual_prim.GetReferences()
        rel_refs.ClearReferences()

        print(
            f"[info] Prim {prim.GetPath()} -> Visual references {asset_path_for_reference}"
        )

        if not dry_run:
            rel_refs.AddReference(asset_path_for_reference)

    if dry_run:
        print("[dry-run] Skipping save of root layer")
    else:
        print(f"[info] Saving modified scene to {scene_path}")
        stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GLB assets to USD and rewire scene.usda to reference them."
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Path to scene.usda (or any USD stage with obj_* prims).",
    )
    parser.add_argument(
        "--converter",
        required=True,
        help="Executable to convert GLB->USD (e.g. 'usd_from_gltf', 'my_gltf_to_usd').",
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

    rewire_scene_to_usd(
        scene_path=scene_path,
        converter=args.converter,
        root=root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

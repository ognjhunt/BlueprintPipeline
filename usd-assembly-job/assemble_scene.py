#!/usr/bin/env python3
"""
assemble_scene.py - Orchestrate the full USD scene assembly pipeline.

This script:
  1. Reads scene_assets.json to find all objects
  2. Converts GLB assets to USDZ format
  3. Builds the scene.usda with proper references
  4. Validates the final scene

Usage:
  python assemble_scene.py

Environment variables:
  BUCKET         - GCS bucket name
  SCENE_ID       - Scene identifier
  LAYOUT_PREFIX  - Path prefix for layout files
  ASSETS_PREFIX  - Path prefix for asset files
  USD_PREFIX     - Path prefix for USD output (defaults to ASSETS_PREFIX)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from pxr import Sdf, Usd, UsdGeom
except ImportError:
    print("ERROR: usd-core is required. Install with: pip install usd-core")
    sys.exit(1)

# Import our modules
from glb_to_usd import convert_glb_to_usd
from build_scene_usd import (
    build_scene,
    load_json,
    safe_path_join,
    ensure_dir,
)


# -----------------------------------------------------------------------------
# Asset Discovery
# -----------------------------------------------------------------------------


def find_glb_assets(
    scene_assets: Dict,
    root: Path,
    assets_prefix: str,
) -> List[Tuple[int, Path, Path]]:
    """
    Find all GLB assets that need conversion.

    Returns:
        List of (object_id, glb_path, usdz_path) tuples
    """
    conversions = []

    for obj in scene_assets.get("objects", []):
        oid = obj.get("id")
        is_interactive = obj.get("type") == "interactive"

        # Skip interactive objects (they use URDF)
        if is_interactive:
            continue

        # Skip if object has simready.usda (physics-enabled asset)
        obj_dir = root / assets_prefix / f"obj_{oid}"
        simready_path = obj_dir / "simready.usda"
        legacy_simready_path = root / assets_prefix / "static" / f"obj_{oid}" / "simready.usda"

        if simready_path.exists():
            print(f"[INFO] Using simready.usda for obj_{oid} (skipping GLB conversion)")
            continue
        if legacy_simready_path.exists():
            print(f"[INFO] Using simready.usda for obj_{oid} (skipping GLB conversion)")
            continue

        # Determine asset path
        asset_path = obj.get("asset_path")
        if not asset_path:
            # Default GLB location
            asset_path = f"{assets_prefix}/obj_{oid}/asset.glb"

        # Check if it's a GLB
        if not asset_path.lower().endswith((".glb", ".gltf")):
            continue

        glb_path = safe_path_join(root, asset_path)

        # Skip if GLB doesn't exist
        if not glb_path.exists():
            print(f"[WARN] GLB not found for obj_{oid}: {glb_path}")
            continue

        # Target USDZ path (same directory, .usdz extension)
        usdz_path = glb_path.with_suffix(".usdz")

        # Skip if USDZ already exists and is newer than GLB
        if usdz_path.exists():
            if usdz_path.stat().st_mtime >= glb_path.stat().st_mtime:
                print(f"[INFO] USDZ already up-to-date for obj_{oid}: {usdz_path}")
                continue

        conversions.append((oid, glb_path, usdz_path))

    return conversions


# -----------------------------------------------------------------------------
# Conversion Pipeline
# -----------------------------------------------------------------------------


def convert_single_asset(
    oid: int,
    glb_path: Path,
    usdz_path: Path,
) -> Tuple[int, bool, str]:
    """
    Convert a single GLB to USDZ.

    Returns:
        (object_id, success, message)
    """
    try:
        success = convert_glb_to_usd(glb_path, usdz_path, create_usdz=True)
        if success:
            return (oid, True, f"Converted {glb_path.name} -> {usdz_path.name}")
        else:
            return (oid, False, f"Conversion returned False for {glb_path.name}")
    except Exception as e:
        return (oid, False, f"Error converting {glb_path.name}: {e}")


def convert_all_assets(
    conversions: List[Tuple[int, Path, Path]],
    max_workers: int = 4,
) -> Tuple[int, int]:
    """
    Convert all GLB assets to USDZ in parallel.

    Returns:
        (success_count, failure_count)
    """
    if not conversions:
        print("[INFO] No GLB conversions needed")
        return (0, 0)

    print(f"[INFO] Converting {len(conversions)} GLB assets to USDZ...")

    success_count = 0
    failure_count = 0

    # Use thread pool for I/O-bound conversions
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_single_asset, oid, glb, usdz): oid
            for oid, glb, usdz in conversions
        }

        for future in as_completed(futures):
            oid = futures[future]
            try:
                obj_id, success, message = future.result()
                if success:
                    print(f"[OK] obj_{obj_id}: {message}")
                    success_count += 1
                else:
                    print(f"[FAIL] obj_{obj_id}: {message}")
                    failure_count += 1
            except Exception as e:
                print(f"[FAIL] obj_{oid}: Exception - {e}")
                failure_count += 1

    return (success_count, failure_count)


# -----------------------------------------------------------------------------
# Scene Wiring
# -----------------------------------------------------------------------------


def wire_usdz_references(
    stage: Usd.Stage,
    root: Path,
    assets_prefix: str,
    usd_prefix: str,
) -> int:
    """
    Update scene.usda to reference converted USDZ files.

    Returns:
        Number of objects wired
    """
    stage_dir = root / usd_prefix
    wired_count = 0

    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.startswith("obj_"):
            continue

        # Check if this object has pending conversion
        pending_attr = prim.GetAttribute("pendingConversion")
        if not pending_attr or not pending_attr.Get():
            continue

        # Get object ID
        try:
            oid = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Find USDZ file
        asset_path_attr = prim.GetAttribute("asset_path")
        asset_path = asset_path_attr.Get() if asset_path_attr else None

        usdz_path = None
        if asset_path:
            glb_path = safe_path_join(root, asset_path)
            candidate = glb_path.with_suffix(".usdz")
            if candidate.exists():
                usdz_path = candidate
        
        if not usdz_path:
            # Check standard location
            candidate = root / assets_prefix / f"obj_{oid}" / "asset.usdz"
            if candidate.exists():
                usdz_path = candidate

        if not usdz_path:
            print(f"[WARN] No USDZ found for obj_{oid}")
            continue

        # Calculate relative path
        rel_path = os.path.relpath(usdz_path, stage_dir)
        rel_path = rel_path.replace("\\", "/")

        # Create or update Geom child with reference
        geom_path = prim.GetPath().AppendChild("Geom")
        geom_prim = stage.OverridePrim(geom_path)
        UsdGeom.Xform.Define(stage, geom_path)

        refs = geom_prim.GetReferences()
        refs.ClearReferences()
        refs.AddReference(rel_path)

        # Clear pending flag
        prim.RemoveProperty("pendingConversion")

        # Update asset metadata
        if asset_path_attr:
            asset_path_attr.Set(str(usdz_path.relative_to(root)).replace("\\", "/"))
        
        type_attr = prim.GetAttribute("assetType")
        if type_attr:
            type_attr.Set("usdz")

        print(f"[WIRE] obj_{oid}: -> {rel_path}")
        wired_count += 1

    return wired_count


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def validate_scene(stage: Usd.Stage) -> List[str]:
    """
    Validate the scene for common issues.

    Returns:
        List of warning/error messages
    """
    issues = []

    for prim in stage.Traverse():
        # Check for unresolved references
        if prim.HasAuthoredReferences():
            refs = prim.GetPrimStack()
            # This is a simplified check - full validation would use UsdUtils.ComputeAllDependencies

        # Check for objects without geometry
        name = prim.GetName()
        if name.startswith("obj_"):
            geom_path = prim.GetPath().AppendChild("Geom")
            geom_prim = stage.GetPrimAtPath(geom_path)
            if not geom_prim or not geom_prim.IsValid():
                issues.append(f"{prim.GetPath()}: No Geom child found")

    return issues


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def assemble_scene() -> int:
    """
    Run the complete scene assembly pipeline.

    Returns:
        Exit code (0 for success)
    """
    # Get environment configuration
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    layout_prefix = os.getenv("LAYOUT_PREFIX")
    assets_prefix = os.getenv("ASSETS_PREFIX")
    usd_prefix = os.getenv("USD_PREFIX") or assets_prefix

    if not layout_prefix or not assets_prefix:
        print("[ERROR] LAYOUT_PREFIX and ASSETS_PREFIX are required", file=sys.stderr)
        return 1

    root = Path("/mnt/gcs")
    layout_path = root / layout_prefix / "scene_layout_scaled.json"
    assets_path = root / assets_prefix / "scene_assets.json"
    stage_path = root / usd_prefix / "scene.usda"

    print("=" * 60)
    print("USD Scene Assembly Pipeline")
    print("=" * 60)
    print(f"  Bucket:        {bucket}")
    print(f"  Scene ID:      {scene_id}")
    print(f"  Layout:        {layout_path}")
    print(f"  Assets:        {assets_path}")
    print(f"  Output:        {stage_path}")
    print("=" * 60)

    # Phase 1: Discover GLB assets
    print("\n[PHASE 1] Discovering GLB assets...")
    try:
        scene_assets = load_json(assets_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    conversions = find_glb_assets(scene_assets, root, assets_prefix)
    print(f"  Found {len(conversions)} GLB files to convert")

    # Phase 2: Convert GLB to USDZ
    print("\n[PHASE 2] Converting GLB assets to USDZ...")
    success, failures = convert_all_assets(conversions, max_workers=4)
    print(f"  Converted: {success} | Failed: {failures}")

    if failures > 0:
        print(f"[WARN] {failures} conversions failed - continuing with available assets")

    # Phase 3: Build scene
    print("\n[PHASE 3] Building USD scene...")
    try:
        stage, objects = build_scene(
            layout_path=layout_path,
            assets_path=assets_path,
            output_path=stage_path,
            root=root,
            assets_prefix=assets_prefix,
            usd_prefix=usd_prefix,
        )
    except Exception as e:
        print(f"[ERROR] Failed to build scene: {e}")
        traceback.print_exc()
        return 1

    # Phase 4: Wire references
    print("\n[PHASE 4] Wiring USDZ references...")
    wired = wire_usdz_references(stage, root, assets_prefix, usd_prefix)
    print(f"  Wired {wired} object references")

    # Save updated stage
    stage.GetRootLayer().Save()
    print(f"  Saved updated scene: {stage_path}")

    # Phase 5: Validate
    print("\n[PHASE 5] Validating scene...")
    issues = validate_scene(stage)
    if issues:
        print(f"  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print("  Scene validation passed!")

    # Summary
    print("\n" + "=" * 60)
    print("Assembly Complete!")
    print("=" * 60)
    print(f"  Output: {stage_path}")
    print(f"  Objects: {len(objects)}")
    print(f"  GLB->USDZ: {success} converted, {failures} failed")
    print(f"  References: {wired} wired")
    print("=" * 60)

    return 0 if failures == 0 else 1


def main() -> None:
    """CLI entry point."""
    sys.exit(assemble_scene())


if __name__ == "__main__":
    main()
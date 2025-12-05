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
    resolve_usdz_asset_path,
)


# -----------------------------------------------------------------------------
# Asset Discovery
# -----------------------------------------------------------------------------


def _discover_glb_in_obj_dir(obj_dir: Path) -> Optional[Path]:
    """Return the first GLB found in an object directory."""
    GLB_CANDIDATES = ["asset.glb", "model.glb", "mesh.glb"]

    for candidate_name in GLB_CANDIDATES:
        candidate = obj_dir / candidate_name
        if candidate.exists():
            return candidate

    for candidate in obj_dir.glob("*.glb"):
        return candidate

    return None


def find_glb_assets(
    scene_assets: Dict,
    root: Path,
    assets_prefix: str,
) -> List[Tuple[str, Path, Path]]:
    """
    Find all GLB assets that need conversion.

    Returns:
        List of (object_id, glb_path, usdz_path) tuples
    """
    conversions: List[Tuple[str, Path, Path]] = []
    seen_ids: Set[str] = set()

    # Candidate GLB filenames to check (in priority order)
    GLB_CANDIDATES = ["asset.glb", "model.glb", "mesh.glb"]

    for obj in scene_assets.get("objects", []):
        oid = obj.get("id")
        oid_str = str(oid)
        seen_ids.add(oid_str)

        # Determine asset path - check explicit path first
        asset_path = obj.get("asset_path")
        glb_path = None

        if asset_path:
            # Check if it's a GLB path
            if asset_path.lower().endswith((".glb", ".gltf")):
                candidate = safe_path_join(root, asset_path)
                if candidate.exists():
                    glb_path = candidate

        # If no explicit path or it doesn't exist, try candidate filenames
        if glb_path is None:
            obj_dir = root / assets_prefix / f"obj_{oid}"
            glb_path = _discover_glb_in_obj_dir(obj_dir)
            if glb_path:
                print(f"[INFO] obj_{oid}: found GLB at {glb_path.name}")

        # Skip if no GLB found
        if glb_path is None:
            print(f"[WARN] GLB not found for obj_{oid} (tried: {GLB_CANDIDATES})")
            continue

        # Target USDZ path (same directory, .usdz extension)
        usdz_path = glb_path.with_suffix(".usdz")

        # Skip if USDZ already exists and is newer than GLB
        if usdz_path.exists():
            if usdz_path.stat().st_mtime >= glb_path.stat().st_mtime:
                print(f"[INFO] USDZ already up-to-date for obj_{oid}: {usdz_path}")
                continue

        conversions.append((oid_str, glb_path, usdz_path))

    # Secondary sweep: look for GLBs on disk that weren't present in scene_assets
    assets_root = root / assets_prefix
    for obj_dir in assets_root.glob("obj_*"):
        oid = obj_dir.name.replace("obj_", "")
        if oid in seen_ids:
            continue

        glb_path = _discover_glb_in_obj_dir(obj_dir)
        if not glb_path:
            continue

        usdz_path = glb_path.with_suffix(".usdz")
        if usdz_path.exists() and usdz_path.stat().st_mtime >= glb_path.stat().st_mtime:
            continue

        conversions.append((oid, glb_path, usdz_path))
        print(f"[INFO] obj_{oid}: discovered GLB without scene_assets entry ({glb_path.name})")

    return conversions


# -----------------------------------------------------------------------------
# Conversion Pipeline
# -----------------------------------------------------------------------------


def convert_single_asset(
    oid: str,
    glb_path: Path,
    usdz_path: Path,
) -> Tuple[str, bool, str]:
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
    conversions: List[Tuple[str, Path, Path]],
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

        oid = name[len("obj_") :]

        # Check if Geom already has a valid reference
        geom_path = prim.GetPath().AppendChild("Geom")
        geom_prim = stage.GetPrimAtPath(geom_path)
        has_ref = False
        if geom_prim and geom_prim.IsValid():
            refs = geom_prim.GetReferences().GetAddedOrExplicitItems()
            has_ref = len(refs) > 0

        pending_attr = prim.GetAttribute("pendingConversion")
        pending = pending_attr.Get() if pending_attr else False

        # Skip objects that already have a reference and are not pending conversion
        if has_ref and not pending:
            continue

        asset_path_attr = prim.GetAttribute("asset_path")
        asset_path = asset_path_attr.Get() if asset_path_attr else None

        usdz_rel = resolve_usdz_asset_path(root, assets_prefix, usd_prefix, oid, asset_path)

        if usdz_rel:
            geom_prim = stage.OverridePrim(geom_path)
            UsdGeom.Xform.Define(stage, geom_path)

            refs = geom_prim.GetReferences()
            refs.ClearReferences()
            refs.AddReference(usdz_rel, primPath=Sdf.Path("/Root"))

            if asset_path_attr and asset_path:
                asset_path_attr.Set(usdz_rel)

            type_attr = prim.GetAttribute("assetType")
            if type_attr:
                type_attr.Set("usdz")

            if pending_attr:
                prim.RemoveProperty("pendingConversion")

            print(f"[WIRE] obj_{oid}: -> {usdz_rel}")
            wired_count += 1
            continue

        # If USDZ is missing but a GLB exists, mark for conversion
        obj_dir = root / assets_prefix / f"obj_{oid}"
        glb_path = _discover_glb_in_obj_dir(obj_dir)
        if glb_path:
            rel_glb = str(glb_path.relative_to(root)).replace("\\", "/")
            prim.CreateAttribute("asset_path", Sdf.ValueTypeNames.String).Set(rel_glb)
            prim.CreateAttribute("pendingConversion", Sdf.ValueTypeNames.Bool).Set(True)
            print(f"[WIRE] obj_{oid}: USDZ missing, queued GLB for conversion ({rel_glb})")

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
    convert_only = os.getenv("CONVERT_ONLY", "false").lower() in {"1", "true", "yes"}

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

    if convert_only:
        print("\n[INFO] CONVERT_ONLY enabled - skipping scene assembly and exiting after conversions")
        return 0 if failures == 0 else 1

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
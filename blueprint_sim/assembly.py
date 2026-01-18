from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .manifest import DEFAULT_ROOT, load_canonical_manifest

# Import articulation wiring (optional - graceful degradation if not available)
try:
    from tools.articulation_wiring import ArticulationWiring
    HAVE_ARTICULATION_WIRING = True
except ImportError:
    HAVE_ARTICULATION_WIRING = False

# Import scale authority (optional)
try:
    from tools.scale_authority import ScaleAuthority, apply_scale_to_manifest
    HAVE_SCALE_AUTHORITY = True
except ImportError:
    HAVE_SCALE_AUTHORITY = False

REPO_ROOT = Path(__file__).resolve().parents[1]
USD_ASSEMBLY_DIR = REPO_ROOT / "usd-assembly-job"
if str(USD_ASSEMBLY_DIR) not in sys.path:
    sys.path.append(str(USD_ASSEMBLY_DIR))

try:
    from pxr import Sdf, Usd, UsdGeom
except ImportError as exc:  # pragma: no cover - runtime guard
    print(
        "[USD-ASSEMBLY] ❌ OpenUSD (pxr) Python bindings are missing. "
        "Install OpenUSD (usd-core) in the job image before running USD assembly.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from build_scene_usd import (  # noqa: E402
    build_scene,
    load_json,
    ObjectAddFailures,
    safe_path_join,
    resolve_usdz_asset_path,
)
from glb_to_usd import convert_glb_to_usd  # noqa: E402


# -----------------------------------------------------------------------------
# Asset Discovery
# -----------------------------------------------------------------------------


def _discover_glb_in_obj_dir(obj_dir: Path) -> Optional[Path]:
    """Return the first GLB found in an object directory."""
    glb_candidates = ["asset.glb", "model.glb", "mesh.glb"]

    for candidate_name in glb_candidates:
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

    glb_candidates = ["asset.glb", "model.glb", "mesh.glb"]

    for obj in scene_assets.get("objects", []):
        oid = obj.get("id")
        oid_str = str(oid)
        seen_ids.add(oid_str)

        asset_path = obj.get("asset_path")
        glb_path = None

        if asset_path:
            if asset_path.lower().endswith((".glb", ".gltf")):
                candidate = safe_path_join(root, asset_path)
                if candidate.exists():
                    glb_path = candidate

        if glb_path is None:
            obj_dir = root / assets_prefix / f"obj_{oid}"
            glb_path = _discover_glb_in_obj_dir(obj_dir)
            if glb_path:
                print(f"[INFO] obj_{oid}: found GLB at {glb_path.name}")

        if glb_path is None:
            print(
                f"[WARN] GLB not found for obj_{oid} (tried: {glb_candidates})"
            )
            continue

        usdz_path = glb_path.with_suffix(".usdz")

        if usdz_path.exists():
            if usdz_path.stat().st_mtime >= glb_path.stat().st_mtime:
                print(f"[INFO] USDZ already up-to-date for obj_{oid}: {usdz_path}")
                continue

        conversions.append((oid_str, glb_path, usdz_path))

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
        print(
            f"[INFO] obj_{oid}: discovered GLB without scene_assets entry ({glb_path.name})"
        )

    return conversions


# -----------------------------------------------------------------------------
# Conversion Pipeline
# -----------------------------------------------------------------------------


def convert_single_asset(
    oid: str,
    glb_path: Path,
    usdz_path: Path,
) -> Tuple[str, bool, str]:
    try:
        success = convert_glb_to_usd(glb_path, usdz_path, create_usdz=True)
        if success:
            return (oid, True, f"Converted {glb_path.name} -> {usdz_path.name}")
        else:
            return (oid, False, f"Conversion returned False for {glb_path.name}")
    except Exception as e:  # pragma: no cover - conversion failures are runtime
        return (oid, False, f"Error converting {glb_path.name}: {e}")


def convert_all_assets(
    conversions: List[Tuple[str, Path, Path]],
    max_workers: int = 4,
) -> Tuple[int, int]:
    if not conversions:
        print("[INFO] No GLB conversions needed")
        return (0, 0)

    print(f"[INFO] Converting {len(conversions)} GLB assets to USDZ...")

    success_count = 0
    failure_count = 0

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
            except Exception as e:  # pragma: no cover - concurrency guard
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
    stage_dir = root / usd_prefix
    wired_count = 0

    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.startswith("obj_"):
            continue

        oid = name[len("obj_") :]

        geom_path = prim.GetPath().AppendChild("Geom")
        geom_prim = stage.GetPrimAtPath(geom_path)
        has_ref = False
        if geom_prim and geom_prim.IsValid():
            has_ref = geom_prim.HasAuthoredReferences()

        pending_attr = prim.GetAttribute("pendingConversion")
        pending = pending_attr.Get() if pending_attr else False

        if has_ref and not pending:
            continue

        asset_path_attr = prim.GetAttribute("asset_path")
        asset_path = asset_path_attr.Get() if asset_path_attr else None

        usdz_rel = resolve_usdz_asset_path(root, assets_prefix, usd_prefix, oid, asset_path)

        if usdz_rel:
            geom_prim = stage.OverridePrim(geom_path)
            UsdGeom.Xform.Define(stage, geom_path)

            get_references = getattr(geom_prim, "GetReferences", None)
            if not callable(get_references):
                print(
                    "[ERROR] Usd.Prim is missing GetReferences; incompatible USD build detected",
                    file=sys.stderr,
                )
                raise RuntimeError("USD API missing GetReferences")

            refs = get_references()
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
    issues = []

    for prim in stage.Traverse():
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


def assemble_scene(
    layout_prefix: str,
    assets_prefix: str,
    usd_prefix: Optional[str] = None,
    bucket: str = "",
    scene_id: str = "",
    convert_only: bool = False,
    root: Path = DEFAULT_ROOT,
) -> int:
    usd_prefix = usd_prefix or assets_prefix
    layout_path = root / layout_prefix / "scene_layout_scaled.json"
    stage_path = root / usd_prefix / "scene.usda"

    print("=" * 60)
    print("USD Scene Assembly Pipeline")
    print("=" * 60)
    print(f"  Bucket:        {bucket}")
    print(f"  Scene ID:      {scene_id}")
    print(f"  Layout:        {layout_path}")
    print(f"  Assets Prefix: {assets_prefix}")
    print(f"  Output:        {stage_path}")
    print("=" * 60)

    try:
        scene_assets = load_canonical_manifest(assets_prefix, root)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("\n[PHASE 1] Discovering GLB assets...")
    conversions = find_glb_assets(scene_assets, root, assets_prefix)
    print(f"  Found {len(conversions)} GLB files to convert")

    print("\n[PHASE 2] Converting GLB assets to USDZ...")
    success, failures = convert_all_assets(conversions, max_workers=4)
    print(f"  Converted: {success} | Failed: {failures}")

    if failures > 0:
        print(f"[WARN] {failures} conversions failed - continuing with available assets")

    if convert_only:
        print("\n[INFO] CONVERT_ONLY enabled - skipping scene assembly and exiting after conversions")
        return 0 if failures == 0 else 1

    print("\n[PHASE 3] Building USD scene...")
    try:
        stage, objects = build_scene(
            layout_path=layout_path,
            assets_path=root / assets_prefix / "scene_manifest.json",
            output_path=stage_path,
            root=root,
            assets_prefix=assets_prefix,
            usd_prefix=usd_prefix,
        )
    except ObjectAddFailures as exc:
        print("[ERROR] Failed to add one or more objects to the USD scene:")
        for oid, obj_exc in exc.failures:
            obj_label = f"obj_{oid}" if oid is not None else "obj_unknown"
            print(f"  - {obj_label}: {obj_exc}")
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"[ERROR] Failed to build scene: {e}")
        traceback.print_exc()
        return 1

    print("\n[PHASE 4] Wiring USDZ references...")
    try:
        wired = wire_usdz_references(stage, root, assets_prefix, usd_prefix)
    except Exception as e:
        print(f"[ERROR] Failed to wire USDZ references: {e}")
        traceback.print_exc()
        return 1
    print(f"  Wired {wired} object references")

    # [PHASE 4.5] Wire articulated assets from interactive-job
    articulated_count = 0
    if HAVE_ARTICULATION_WIRING:
        print("\n[PHASE 4.5] Wiring articulated assets...")
        try:
            articulation_wiring = ArticulationWiring(root, assets_prefix, usd_prefix)
            articulated_count, articulated_assets = articulation_wiring.wire_scene(
                stage, scene_assets
            )
            print(f"  Wired {articulated_count} articulated objects")

            # Update manifest with articulation data
            if articulated_count > 0:
                manifest_path = root / assets_prefix / "scene_manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text())
                    updated_manifest = articulation_wiring.update_manifest(
                        manifest, articulated_assets
                    )
                    manifest_path.write_text(json.dumps(updated_manifest, indent=2))
                    print(f"  Updated manifest with articulation data")
        except Exception as e:
            print(f"[WARN] Articulation wiring failed (non-fatal): {e}")
    else:
        print("\n[PHASE 4.5] Articulation wiring not available (skipping)")

    stage.GetRootLayer().Save()
    print(f"  Saved updated scene: {stage_path}")

    print("\n[PHASE 5] Validating scene...")
    issues = validate_scene(stage)
    if issues:
        print(f"  Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print("  Scene validation passed!")

    print("\n" + "=" * 60)
    print("Assembly Complete!")
    print("=" * 60)
    print(f"  Output: {stage_path}")
    print(f"  Objects: {len(objects)}")
    print(f"  GLB->USDZ: {success} converted, {failures} failed")
    print(f"  References: {wired} wired")
    print(f"  Articulated: {articulated_count} objects")
    print("=" * 60)

    return 0 if failures == 0 else 1


def assemble_from_env() -> int:
    import os

    from tools.validation.entrypoint_checks import validate_required_env_vars

    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
            "LAYOUT_PREFIX": "Path prefix for layout files (scenes/<sceneId>/layout)",
            "ASSETS_PREFIX": "Path prefix for assets (scenes/<sceneId>/assets)",
        },
        label="[ASSEMBLY]",
    )

    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]
    layout_prefix = os.environ["LAYOUT_PREFIX"]
    assets_prefix = os.environ["ASSETS_PREFIX"]
    usd_prefix = os.getenv("USD_PREFIX") or assets_prefix
    convert_only = os.getenv("CONVERT_ONLY", "false").lower() in {"1", "true", "yes"}

    return assemble_scene(
        layout_prefix=layout_prefix,
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
        bucket=bucket,
        scene_id=scene_id,
        convert_only=convert_only,
        root=DEFAULT_ROOT,
    )


__all__ = [
    "assemble_scene",
    "assemble_from_env",
    "find_glb_assets",
    "convert_all_assets",
    "wire_usdz_references",
]

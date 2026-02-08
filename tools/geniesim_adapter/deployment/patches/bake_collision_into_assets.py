#!/usr/bin/env python3
"""
Bake PhysicsCollisionAPI + convexHull approximation into USD asset files,
and fix internal RigidBodyAPI settings that conflict with scene-level physics.

This modifies the USD files in-place so that:
1. All mesh prims have collision enabled (PhysX sees collision geometry on load)
2. Internal RigidBodyAPI prim kinematic overrides are only adjusted for explicit
   --kinematic-names targets (default behavior leaves kinematic flags untouched)

Run inside Isaac Sim container with pxr available:
  PYTHONPATH=/isaac-sim/extscache/omni.usd.libs-*/ LD_LIBRARY_PATH=.../bin
  python3 bake_collision_into_assets.py <assets_dir> [--kinematic-names NAME1,NAME2,...]
"""
import sys
import os
import argparse


def bake_collision(asset_dir, kinematic_names=None):
    """Add CollisionAPI to all mesh prims and fix kinematic flags."""
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf

    usd_files = []
    for root, dirs, files in os.walk(asset_dir):
        for f in files:
            if f.endswith((".usd", ".usda", ".usdc")):
                usd_files.append(os.path.join(root, f))

    print("[BAKE] Found %d USD files in %s" % (len(usd_files), asset_dir))

    # Build set of asset names that should be kinematic
    kin_set = set()
    if kinematic_names:
        for name in kinematic_names:
            kin_set.add(name.strip())

    total_meshes = 0
    total_collision_added = 0
    total_kinematic_fixed_true_count = 0
    total_kinematic_untouched_count = 0
    failed_files = []

    for usd_path in usd_files:
        try:
            stage = Usd.Stage.Open(usd_path)
            if stage is None:
                print("[BAKE] SKIP %s: could not open" % usd_path)
                failed_files.append(usd_path)
                continue

            basename = os.path.basename(usd_path)
            asset_name = os.path.splitext(basename)[0]  # e.g. "Table049"
            changes = []
            file_meshes = 0

            for prim in Usd.PrimRange(stage.GetPrimAtPath("/")):
                # --- Fix 1: Add collision to meshes ---
                if prim.IsA(UsdGeom.Mesh):
                    file_meshes += 1
                    total_meshes += 1
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        col_api = UsdPhysics.CollisionAPI.Apply(prim)
                        col_api.CreateCollisionEnabledAttr().Set(True)
                        prim.CreateAttribute(
                            "physics:approximation",
                            Sdf.ValueTypeNames.Token,
                        ).Set("convexHull")
                        changes.append("collision:%s" % prim.GetName())
                        total_collision_added += 1

                # --- Fix 2: Fix internal RigidBodyAPI kinematic flag ---
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    kin_attr = prim.GetAttribute("physics:kinematicEnabled")
                    if kin_attr and kin_attr.Get() is False:
                        # Check if this asset should be kinematic
                        should_be_kinematic = False
                        if kin_set:
                            should_be_kinematic = asset_name in kin_set
                        else:
                            # Default: if --kinematic-names is not provided,
                            # do NOT mutate kinematic flags.
                            should_be_kinematic = False

                        if should_be_kinematic:
                            kin_attr.Set(True)
                            changes.append("kinematic:%s" % prim.GetName())
                            total_kinematic_fixed_true_count += 1
                        else:
                            total_kinematic_untouched_count += 1

            if changes:
                stage.GetRootLayer().Save()
                print("[BAKE] %s: %s" % (basename, ", ".join(changes)))

                # ── Verification: reopen and check collision persisted ──
                verify_stage = Usd.Stage.Open(usd_path)
                if verify_stage is None:
                    print("[BAKE] VERIFY FAILED %s: could not reopen after save" % basename)
                    failed_files.append(usd_path)
                    continue

                verify_coll = 0
                verify_meshes = 0
                for p in Usd.PrimRange(verify_stage.GetPrimAtPath("/")):
                    if p.IsA(UsdGeom.Mesh):
                        verify_meshes += 1
                        if p.HasAPI(UsdPhysics.CollisionAPI):
                            verify_coll += 1

                if verify_coll < verify_meshes:
                    print(
                        "[BAKE] VERIFY FAILED %s: only %d/%d meshes have collision after save"
                        % (basename, verify_coll, verify_meshes)
                    )
                    failed_files.append(usd_path)
                else:
                    print("[BAKE] VERIFIED %s: %d/%d meshes OK" % (basename, verify_coll, verify_meshes))
            else:
                # No changes needed — but still verify existing collision
                verify_coll = 0
                for p in Usd.PrimRange(stage.GetPrimAtPath("/")):
                    if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
                        verify_coll += 1
                if verify_coll < file_meshes:
                    print(
                        "[BAKE] WARNING %s: no changes made but only %d/%d meshes have collision"
                        % (basename, verify_coll, file_meshes)
                    )
                    failed_files.append(usd_path)
                else:
                    print("[BAKE] %s: already OK (%d meshes)" % (basename, file_meshes))

        except Exception as e:
            print("[BAKE] ERROR %s: %s" % (usd_path, e))
            failed_files.append(usd_path)

    # ── Final summary ──
    print("[BAKE] === SUMMARY ===")
    print("[BAKE] Files processed: %d" % len(usd_files))
    print("[BAKE] Total meshes: %d" % total_meshes)
    print("[BAKE] collision_added_count: %d" % total_collision_added)
    print("[BAKE] kinematic_fixed_true_count: %d" % total_kinematic_fixed_true_count)
    print("[BAKE] kinematic_untouched_count: %d" % total_kinematic_untouched_count)
    print("[BAKE] Failed files: %d" % len(failed_files))
    for fp in failed_files:
        print("[BAKE]   FAILED: %s" % fp)

    if failed_files:
        print("[BAKE] ERROR: %d files failed verification — collision may not persist" % len(failed_files))
        return False
    else:
        print("[BAKE] VERIFIED: all %d files have complete collision coverage" % len(usd_files))
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake collision into USD assets")
    parser.add_argument("asset_dir", help="Directory containing USD asset files")
    parser.add_argument(
        "--kinematic-names",
        help="Comma-separated list of asset names that should be kinematic",
        default=None,
    )
    args = parser.parse_args()

    kin_names = None
    if args.kinematic_names:
        kin_names = args.kinematic_names.split(",")

    success = bake_collision(args.asset_dir, kin_names)
    sys.exit(0 if success else 1)

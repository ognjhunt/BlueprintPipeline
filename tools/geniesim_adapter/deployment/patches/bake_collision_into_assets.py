#!/usr/bin/env python3
"""
Bake PhysicsCollisionAPI + convexHull approximation into USD asset files,
and fix internal RigidBodyAPI settings that conflict with scene-level physics.

This modifies the USD files in-place so that:
1. All mesh prims have collision enabled (PhysX sees collision geometry on load)
2. Internal RigidBodyAPI prims marked kinematic=False are fixed to kinematic=True
   for objects that should be kinematic in the scene (controlled by scene_role_map)

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

    for usd_path in usd_files:
        try:
            stage = Usd.Stage.Open(usd_path)
            if stage is None:
                print("[BAKE] SKIP %s: could not open" % usd_path)
                continue

            basename = os.path.basename(usd_path)
            asset_name = os.path.splitext(basename)[0]  # e.g. "Table049"
            changes = []

            for prim in Usd.PrimRange(stage.GetPrimAtPath("/")):
                # --- Fix 1: Add collision to meshes ---
                if prim.IsA(UsdGeom.Mesh):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        col_api = UsdPhysics.CollisionAPI.Apply(prim)
                        col_api.CreateCollisionEnabledAttr().Set(True)
                        prim.CreateAttribute(
                            "physics:approximation",
                            Sdf.ValueTypeNames.Token,
                        ).Set("convexHull")
                        changes.append("collision:%s" % prim.GetName())

                # --- Fix 2: Fix internal RigidBodyAPI kinematic flag ---
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    kin_attr = prim.GetAttribute("physics:kinematicEnabled")
                    if kin_attr and kin_attr.Get() is False:
                        # Check if this asset should be kinematic
                        should_be_kinematic = False
                        if kin_set:
                            should_be_kinematic = asset_name in kin_set
                        else:
                            # Default: if --kinematic-names not provided,
                            # remove internal RigidBodyAPI to let scene.usda control it
                            should_be_kinematic = True

                        if should_be_kinematic:
                            kin_attr.Set(True)
                            changes.append("kinematic:%s" % prim.GetName())

            if changes:
                stage.GetRootLayer().Save()
                print("[BAKE] %s: %s" % (basename, ", ".join(changes)))
            else:
                print("[BAKE] %s: no changes needed" % basename)

        except Exception as e:
            print("[BAKE] ERROR %s: %s" % (usd_path, e))


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

    bake_collision(args.asset_dir, kin_names)
